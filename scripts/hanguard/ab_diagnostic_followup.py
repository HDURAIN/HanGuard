"""Continue registered diagnostics after A and the base readouts finish.

This coordinator never trains A/base or signals their existing processes. It
waits for the identity-bound A child to end and for complete validated artifacts,
then uses the original adapted-extraction/readout/error-analysis commands. The
minimal version runs sequentially on GPU4 with strict ownership and60s idle
confirmation; no scientific settings are changed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import ab_diagnostic_queue as queue
from scripts.hanguard import ab_readout_parallel_queue as ownership

SCHEMA = 'hanguard_ab_strict_followup_v1'
FOLLOWUP_NAMES = ('extract_adapted', 'adapted_last_readouts', 'adapted_mean_readouts', 'error_analysis')


class PrerequisiteFailure(RuntimeError):
    pass


def followup_stages(study, python):
    stages = [stage for stage in queue.build_plan(study, python) if stage.name in FOLLOWUP_NAMES]
    if tuple(stage.name for stage in stages) != FOLLOWUP_NAMES:
        raise ValueError('Original registered continuation stage plan differs')
    return stages


def process_snapshot(pid):
    directory = Path('/proc') / str(int(pid))
    try:
        stat = (directory / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        raw = (directory / 'cmdline').read_bytes()
        return dict(pid=int(pid), uid=directory.stat().st_uid, state=fields[0],
            start_time_ticks=int(fields[19]), command_sha256=hashlib.sha256(raw).hexdigest(),
            exit_code=int(fields[49]) if len(fields) > 49 else None)
    except FileNotFoundError:
        return None


def original_process_finished(expected, reader=process_snapshot):
    current = reader(expected['pid'])
    if current is None:
        return dict(finished=True, reason='Original PID no longer exists')
    if current['start_time_ticks'] != expected['start_time_ticks']:
        return dict(finished=True, reason='PID reused by another process; no signal is permitted', current=current)
    if current['uid'] != expected['uid']:
        raise PrerequisiteFailure('A process UID changed while its start identity remained live')
    if current['state'] in ('Z', 'X'):
        if current.get('exit_code') not in (None, 0):
            raise PrerequisiteFailure(f'A child ended unsuccessfully: encoded exit status {current["exit_code"]}')
        return dict(finished=True, reason='Original child exited; paused parent may still hold its zombie', current=current)
    if current['command_sha256'] != expected['command_sha256']:
        raise PrerequisiteFailure('Live A process command differs from the registered identity')
    return dict(finished=False, reason='Registered A child is still running', current=current)


def verify_A_completion(study, expected_process, process_reader=process_snapshot):
    study = Path(study)
    completion = original_process_finished(expected_process, process_reader)
    if not completion['finished']:
        return dict(ready=False, process=completion)
    training = queue.read_json(study / 'lora_study/protocol.json')
    stage = next(stage for stage in queue.build_plan(study) if stage.name == 'train_A_E03')
    queue.validate_outputs(stage)
    lineage = queue.read_json(stage.outputs[1])
    partition = queue.read_json(training['partition_manifest_path'])
    expected = dict(protocol_sha256=queue.sha(study / 'lora_study/protocol.json'),
        partition_manifest_sha256=training['partition_manifest_sha256'],
        A_train_sha256=training['data_sha256']['train'],
        A_selection_sha256=training['data_sha256']['validation'],
        B_sha256=partition['artifact_sha256']['B.parquet'], formal_test_opened=False)
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise PrerequisiteFailure('Completed A adapter lineage differs from its exact registered A/B protocol')
    state = queue.read_json(study / 'lora_study/state.json')
    if state.get('state') != 'A_LoRA_frozen' or state.get('adapter_sha256') != lineage['adapter_sha256']:
        raise PrerequisiteFailure('A-only training did not publish its final frozen state')
    return dict(ready=True, process=completion,
        artifact_sha256={name: queue.sha(name) for name in stage.outputs},
        adapter_sha256=lineage['adapter_sha256'], no_A_retraining=True)


def verify_base_completion(study, attempt, process_reader=process_snapshot):
    study, attempt = Path(study), Path(attempt)
    state_path = attempt / 'state.json'
    if not state_path.exists():
        return dict(ready=False, reason='Registered base dispatcher has not published state')
    state = queue.read_json(state_path)
    adoption = None
    if state.get('state') == 'stopped':
        trigger_path = attempt / 'ownership_stop_trigger.json'
        if not trigger_path.exists():
            raise PrerequisiteFailure('Base attempt ended without completion: stopped')
        trigger = queue.read_json(trigger_path)
        unknown = trigger.get('trigger_unknown_ownership_pids', [])
        active = trigger.get('trigger_active_workers', {})
        launch_path = attempt / 'root_launch.json'
        launch = queue.read_json(launch_path)
        registered = queue.read_json(attempt / 'protocol.json')
        if (not unknown or trigger.get('trigger_confirmed_foreign_pids') or
                trigger.get('trigger_dispatcher_protocol_sha256') != queue.sha(attempt / 'protocol.json') or
                launch.get('pid') != state.get('supervisor_pid') or
                Path(launch['command'][launch['command'].index('--output') + 1]).resolve() != attempt.resolve()):
            raise PrerequisiteFailure('Stopped base lacks a verified unknown-own-exit adoption chain')
        launcher_identity = launch.get('process_identity', {})
        if (launcher_identity.get('pid') != launch['pid'] or
                launcher_identity.get('argv') != launch['command'] or
                not ownership._study_process(launcher_identity, study) or
                state.get('trigger_active_workers') != active):
            raise PrerequisiteFailure('Stopped base launch/runtime identity chain differs')
        launcher = process_reader(launch['pid'])
        original_launcher_live = (launcher is not None and
            launcher['start_time_ticks'] == launcher_identity['start_time_ticks'] and
            launcher['state'] not in ('Z', 'X'))
        if original_launcher_live:
            raise PrerequisiteFailure('Original base dispatcher is still alive; no aggregation handoff permitted')
        for worker in active.values():
            current = process_reader(worker['pid'])
            if current is not None and current['state'] not in ('Z', 'X'):
                raise PrerequisiteFailure('Original base attempt still has a live writer')
        for pid in unknown:
            match = next(((fold, value) for fold, value in active.items() if value['pid'] == pid), None)
            if match is None or str(match[0]) not in registered['commands']:
                raise PrerequisiteFailure('Unknown blocker was not a registered dispatcher child')
            command = registered['commands'][str(match[0])]
            if '--fold' not in command or command[command.index('--fold') + 1] != str(match[0]):
                raise PrerequisiteFailure('Registered child fold command differs')
            current = process_reader(pid)
            if current is not None and (current['state'] not in ('Z', 'X') or current.get('exit_code') not in (None, 0)):
                raise PrerequisiteFailure('Unknown-own child has not successfully ended')
        adoption = dict(ownership_trigger_sha256=queue.sha(trigger_path),
            original_launch_receipt_sha256=queue.sha(launch_path),
            original_stopped_state_sha256=queue.sha(state_path),
            registered_dispatcher_sha256=queue.sha(attempt / 'protocol.json'),
            registered_child_runtime_record=active, unknown_owned_child_pids=unknown,
            original_stopped_state_preserved=True)
    elif state.get('state') in ('failed', 'waiting_no_eligible_gpu'):
        raise PrerequisiteFailure(f'Base attempt ended without completion: {state.get("state")}')
    if state.get('state') != 'completed' and adoption is None:
        return dict(ready=False, reason='Registered base readouts/report are still running', state=state.get('state'))
    stage = next(stage for stage in queue.build_plan(study) if stage.name == 'base_last_readouts')
    readout = study / 'readouts/base_last'
    protocol_sha = queue.sha(readout / 'protocol.json')
    bindings = {}
    for fold in range(5):
        bindings.update(ownership.validate_fold(readout, fold, protocol_sha))
    if adoption is not None and not (readout / 'analysis/results.json').exists():
        return dict(ready=False, needs_CPU_report=True, adoption=adoption,
                    completed_fold_artifact_sha256=bindings)
    queue.validate_outputs(stage)
    if adoption is None and state.get('analysis_results_sha256') != queue.sha(readout / 'analysis/results.json'):
        raise PrerequisiteFailure('Base dispatcher completion SHA differs from its actual results')
    return dict(ready=True, artifact_sha256={name: queue.sha(name) for name in stage.outputs},
        completed_fold_artifact_sha256=bindings, no_base_retraining=True, adoption=adoption)


def finalize_verified_base_CPU(protocol, output):
    original = next(stage for stage in queue.build_plan(protocol['study'], protocol['python'])
                    if stage.name == 'base_last_readouts')
    command = list(original.command)
    command[2] = 'report'
    command[command.index('--device') + 1] = 'cpu'
    stage = queue.Stage('aggregate_completed_base_CPU', tuple(command), original.inputs,
                        original.outputs, original.validator)
    supervisor = queue.SequentialSupervisor(protocol['study'], [stage], protocol['bindings'],
        queue_state_dir=Path(output) / 'base_CPU_report',
        environment=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                         OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1'))
    supervisor.stop_directories = tuple(dict.fromkeys(supervisor.stop_directories +
        (Path(output), Path(protocol['study']) / 'lora_study')))
    supervisor.run()


def initialize(study, output, base_attempt, a_pid=302664, python=queue.DEFAULT_PYTHON):
    study, output, base_attempt = map(lambda p: Path(p).resolve(), (study, output, base_attempt))
    if output.exists():
        raise FileExistsError('Continuation registration is immutable; use a fresh attempt directory')
    bindings, training = queue.preflight_bindings(study)
    base_protocol_path = base_attempt / 'protocol.json'
    base = queue.read_json(base_protocol_path)
    identity = next((value for value in base['owned_processes'] if value['pid'] == a_pid), None)
    if identity is None or not ownership._study_process(identity, study):
        raise ValueError('Base dispatcher must contain the verified original A process identity')
    if 'train' not in identity['argv'] or not any(Path(value).name == 'ab_diagnostic_experiment.py' for value in identity['argv']):
        raise ValueError('A identity must name the original A-only training command')
    bindings[str(base_protocol_path)] = queue.sha(base_protocol_path)
    for name, digest in base['bindings'].items():
        if queue.sha(name) != digest:
            raise ValueError(f'Original base dispatcher binding changed: {name}')
        bindings[name] = digest
    base_readout = study / 'readouts/base_last/protocol.json'
    bindings[str(base_readout)] = queue.sha(base_readout)
    from scripts.hanguard import ab_frozen_readout_diagnostics as readouts
    original_base = next(stage for stage in queue.build_plan(study, python) if stage.name == 'base_last_readouts')
    expected_args = readouts.build_parser().parse_args(original_base.command[2:])
    actual_base = queue.read_json(base_readout)
    for key in ownership.SETTINGS + ('joint_features',):
        if actual_base['settings'][key] != getattr(expected_args, key):
            raise ValueError(f'Base scientific setting differs from original plan: {key}')
    if Path(actual_base['cache_manifest_path']).resolve() != expected_args.cache_manifest.resolve():
        raise ValueError('Base readouts use a different registered feature cache')
    bindings[str(Path(__file__).resolve())] = queue.sha(__file__)
    stages = followup_stages(study, python)
    protocol = dict(schema=SCHEMA, study=str(study), base_attempt=str(base_attempt),
        python=str(Path(python).absolute()), original_A_process_identity=identity,
        bindings=bindings, stages=[stage.registration() for stage in stages],
        resource_policy=dict(physical_gpu=4, no_foreign_or_unknown_processes=True,
            continuous_empty_gpu_confirmation_seconds=60,
            adapted_extraction_minimum_free_gib=26, readout_minimum_free_gib=4,
            existing_paused_controller_signals_permitted=False),
        scientific_settings_changed=False, A_retraining=False, base_retraining=False,
        formal_test_enabled=False, automatic_retries=False,
        training_protocol_sha256=queue.sha(study / 'lora_study/protocol.json'),
        partition_manifest_sha256=training['partition_manifest_sha256'])
    output.mkdir(parents=True)
    queue.dump(output / 'continuation_protocol.json', protocol)
    queue.dump(output / 'registration.json', dict(protocol_sha256=queue.sha(output / 'continuation_protocol.json')))
    return protocol


def check_registration(output):
    output = Path(output)
    protocol = queue.read_json(output / 'continuation_protocol.json')
    registration = queue.read_json(output / 'registration.json')
    if registration.get('protocol_sha256') != queue.sha(output / 'continuation_protocol.json'):
        raise ValueError('Continuation protocol differs from its independent SHA registration')
    if protocol.get('schema') != SCHEMA:
        raise ValueError('Unknown continuation protocol')
    for name, digest in protocol['bindings'].items():
        if queue.sha(name) != digest:
            raise ValueError(f'Frozen continuation binding changed: {name}')
    for folder in (output, Path(protocol['study']), Path(protocol['study']) / 'lora_study'):
        stop = folder / 'STOP'
        if stop.exists():
            raise queue.StopRequested(stop.read_text().strip() or 'STOP requested')
    return protocol


def wait_prerequisites(output, maximum_wait_seconds=86400, poll_seconds=5):
    output = Path(output)
    began = time.monotonic()
    while True:
        protocol = check_registration(output)
        a = verify_A_completion(protocol['study'], protocol['original_A_process_identity'])
        evidence_path = output / 'base_verified_evidence.json'
        if evidence_path.exists():
            base = queue.read_json(evidence_path)
            for name, digest in base['small_sentinel_sha256'].items():
                if queue.sha(name) != digest:
                    raise PrerequisiteFailure('Verified base sentinel changed while waiting for A')
        else:
            base = verify_base_completion(protocol['study'], protocol['base_attempt'])
            if base.get('needs_CPU_report'):
                finalize_verified_base_CPU(protocol, output)
                base = verify_base_completion(protocol['study'], protocol['base_attempt'])
            if base['ready']:
                small = {name: digest for name, digest in base['completed_fold_artifact_sha256'].items()
                         if Path(name).name == 'complete.json'}
                small.update({name: digest for name, digest in base['artifact_sha256'].items()
                              if Path(name).suffix == '.json'})
                if base.get('adoption'):
                    attempt = Path(protocol['base_attempt'])
                    small.update({str(attempt / name): queue.sha(attempt / name)
                                  for name in ('state.json', 'ownership_stop_trigger.json', 'root_launch.json')})
                base['small_sentinel_sha256'] = small
                queue.dump(evidence_path, base)
        status = dict(schema=SCHEMA, state='prerequisites_ready' if a['ready'] and base['ready'] else 'waiting_prerequisites',
            A=a, base=base, supervisor_pid=os.getpid(), updated_unix=time.time(),
            no_existing_processes_signalled=True)
        queue.dump(output / 'continuation_state.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if a['ready'] and base['ready']:
            status['base'] = verify_base_completion(protocol['study'], protocol['base_attempt'])
            queue.dump(output / 'prerequisites.json', status)
            return status
        if time.monotonic() - began >= maximum_wait_seconds:
            raise PrerequisiteFailure('Prerequisite wait expired without restarting any A/base work')
        time.sleep(min(poll_seconds, 5))


class StrictFollowup(queue.SequentialSupervisor):
    def __init__(self, *args, inventory=ownership.gpu_inventory, admission_seconds=75, **kwargs):
        super().__init__(*args, **kwargs)
        self.inventory, self.admission_seconds = inventory, admission_seconds
        self.idle = ownership.IdleConfirmation(60)
        self.gpu_stages = {stage.name for stage in self.stages if stage.validator in ('cache', 'probes', 'heads')}
        self.trigger = None
        self.stop_directories = tuple(dict.fromkeys(self.stop_directories +
            (self.output.parent, self.study_output / 'lora_study')))

    def observation(self):
        gpu = next((value for value in self.inventory() if value['index'] == 4), None)
        if gpu is None:
            raise queue.StopRequested('Registered physical GPU4 is unavailable')
        minimum = 26 if self.active_stage == 'extract_adapted' else 4
        decision = ownership.assess_gpu(gpu, self.owned_processes, self.study_output, minimum_free_gib=minimum)
        return self.idle.apply(decision, time.monotonic())

    def _gpu_observation(self):
        if not self.gpu_stages:
            return dict(gpu_required=False, policy='CPU-only stages do not query GPU resources')
        return self.observation()

    def stop_for_ownership(self, observation):
        evidence = dict(trigger_observed_unix=time.time(), trigger_gpu_observation=observation,
            owned_child_pid=self.child.pid if self.child else None, stage=self.active_stage,
            queue_protocol_sha256=self.protocol_sha, termination_started=False)
        self.trigger = evidence
        queue.dump(self.output / 'ownership_stop_trigger.json', evidence)
        self._state('stopping_for_ownership_conflict', trigger_evidence=evidence)
        raise queue.StopRequested('GPU4 has foreign/unknown ownership; only this continuation child may be terminated')

    def _state(self, state, **details):
        if self.trigger is not None and state in ('stopped', 'failed'):
            details['trigger_evidence'] = dict(self.trigger, termination_started=True,
                termination_completed=self.child is None or self.child.poll() is not None)
        return super()._state(state, **details)

    def _guard_gpu_idle(self):
        if self.active_stage not in self.gpu_stages:
            return
        deadline = time.monotonic() + self.admission_seconds
        while True:
            super()._check()
            observation = self.observation()
            self._state('waiting_strict_GPU4_admission', resource_observation=observation)
            if observation['eligible']:
                return
            if time.monotonic() >= deadline:
                self.stop_for_ownership(observation)
            time.sleep(.5)

    def _check(self):
        super()._check()
        if self.active_stage not in self.gpu_stages or self.child is None or self.child.poll() is not None:
            return
        if self.child.pid not in self.owned_processes:
            deadline = time.monotonic() + 1
            while True:
                try:
                    self.owned_processes[self.child.pid] = ownership.capture_owned(self.child.pid, self.study_output)
                    break
                except (OSError, ValueError):
                    if self.child.poll() is not None:
                        return
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        observation = self.observation()
        if observation['blocking_processes']:
            code = self.child.poll()
            own_exit = (code == 0 and all(value['classification'] == 'unknown' and
                        value['pid'] == self.child.pid for value in observation['blocking_processes']))
            if own_exit:
                stage = next(value for value in self.stages if value.name == self.active_stage)
                queue.validate_outputs(stage, self.stop_directories)
                queue.dump(self.output / 'verified_own_exit_cleanup.json', dict(pid=self.child.pid,
                    exit_code=code, observation=observation, completed_artifacts_verified=True))
                return
            self.stop_for_ownership(observation)


def _run_registered(output):
    output = Path(output).resolve()
    protocol = check_registration(output)
    state_path = output / 'continuation_state.json'
    if state_path.exists() and queue.read_json(state_path).get('state') in ('failed', 'stopped'):
        raise RuntimeError('Previous continuation failed/stopped; automatic retry is forbidden')
    try:
        wait_prerequisites(output)
        protocol = check_registration(output)
        stages = followup_stages(protocol['study'], protocol['python'])
        if [stage.registration() for stage in stages] != protocol['stages']:
            raise ValueError('Continuation commands differ from frozen registration')
        bindings = dict(protocol['bindings'])
        bindings[str(output / 'continuation_protocol.json')] = queue.sha(output / 'continuation_protocol.json')
        bindings[str(output / 'prerequisites.json')] = queue.sha(output / 'prerequisites.json')
        environment = dict(CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        supervisor = StrictFollowup(protocol['study'], stages, bindings,
            queue_state_dir=output / 'execution', environment=environment)
        supervisor.run()
        queue.summarize_results(protocol['study'], output / 'execution')
        queue.dump(state_path, dict(schema=SCHEMA, state='completed', updated_unix=time.time(),
            summary_path=str(output / 'execution/diagnostic_summary.json'), scientific_settings_changed=False))
    except BaseException as exc:
        queue.dump(state_path, dict(schema=SCHEMA,
            state='stopped' if isinstance(exc, queue.StopRequested) else 'failed',
            error_type=type(exc).__name__, reason=str(exc), updated_unix=time.time(),
            automatic_retry=False, scientific_conclusion=None, no_A_or_base_retraining=True))
        raise


def run(output):
    output = Path(output).resolve()
    with (output / 'coordinator.pid').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('An existing continuation owns this attempt lock') from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        return _run_registered(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'plan', 'run', 'status'))
    parser.add_argument('--study', type=Path, default=queue.DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-attempt', type=Path)
    parser.add_argument('--a-pid', type=int, default=302664)
    parser.add_argument('--python', type=Path, default=queue.DEFAULT_PYTHON)
    args = parser.parse_args(argv)
    if args.action == 'init':
        base = args.base_attempt or args.study / 'readout_parallel_base_last_attempt2'
        print(json.dumps(initialize(args.study, args.output, base, args.a_pid, args.python), ensure_ascii=False, indent=2))
    elif args.action == 'plan':
        print(json.dumps([stage.registration() for stage in followup_stages(args.study, args.python)], indent=2))
    elif args.action == 'status':
        print((args.output / 'continuation_state.json').read_text())
    else:
        run(args.output)


if __name__ == '__main__':
    main()
