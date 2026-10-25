"""Single-card, identity-bound queue for the exploratory sparse-token study.

Registration never launches CUDA. An occupied candidate card is waited on using
CPU only; admission is immediate once it is empty, as explicitly requested by
the user. A conflict during a child stage stops only this attempt's own child.
Stopped attempts require a separately reviewed, freshly registered continuation.
The historical A/B source and results are imported read-only and never edited.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import ab_diagnostic_queue as legacy
from scripts.hanguard import ab_readout_parallel_queue as inventory_module

SCHEMA = 'hanguard_sparse_token_single_gpu_queue_v1'
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')


@dataclass(frozen=True)
class RegisteredStage(legacy.Stage):
    gpu_required: bool = True
    minimum_free_gib: float = 4.0

    def registration(self):
        return dict(super().registration(), gpu_required=self.gpu_required,
                    minimum_free_gib=self.minimum_free_gib)

    @classmethod
    def from_registration(cls, value):
        return cls(name=value['name'], command=tuple(value['command']),
            inputs=tuple(value.get('inputs', ())), outputs=tuple(value.get('outputs', ())),
            validator=value.get('validator', 'exists'),
            gpu_required=value.get('gpu_required', True),
            minimum_free_gib=value.get('minimum_free_gib', 4.0))


def validate_stage(stage, stop_directories=()):
    """Validate outputs before publishing a successful immutable receipt.

``complete_json`` requires all declared outputs to be JSON completion records
with ``complete: true``. An optional ``artifacts`` map is relative to that
record's directory, following the scientific entrypoint's own publication.
More detailed cohort/model assertions remain the entrypoint's responsibility.
"""
    if stage.validator != 'complete_json':
        legacy.validate_outputs(stage, stop_directories)
        return
    if not stage.outputs:
        raise ValueError('Completion validation requires declared outputs')
    for filename in stage.outputs:
        path = Path(filename)
        value = legacy.read_json(path)
        if value.get('complete') is not True:
            raise ValueError(f'{stage.name}: JSON completion record is incomplete: {path}')
        for name, digest in value.get('artifacts', {}).items():
            artifact = path.parent / name
            if legacy.sha(artifact, stop_directories) != digest:
                raise ValueError(f'{stage.name}: completion artifact changed: {artifact}')


def _validate_registration(study, stages, gpu, reserved_gpu):
    if gpu not in (2, 4) or reserved_gpu not in (2, 4) or gpu == reserved_gpu:
        raise ValueError('Use exactly one of GPU2/GPU4 and leave the other card unused')
    if not stages or len({stage.name for stage in stages}) != len(stages):
        raise ValueError('A nonempty plan with unique stage names is required')
    study = Path(study).resolve()
    for stage in stages:
        if (not stage.name or '/' in stage.name or stage.name in ('.', '..') or not stage.command or
                not all(isinstance(part, str) for part in stage.command)):
            raise ValueError('Stage names and argv must be explicit and valid')
        if (not isinstance(stage.gpu_required, bool) or
                not math.isfinite(stage.minimum_free_gib) or stage.minimum_free_gib <= 0):
            raise ValueError('Stage GPU requirement and minimum memory are invalid')
        if stage.validator not in ('exists', 'complete_json'):
            raise ValueError('Sparse study uses exists or complete_json validators only')
        for output in stage.outputs:
            if not Path(output).is_absolute() or not Path(output).resolve().is_relative_to(study):
                raise ValueError('New outputs must be absolute paths within the new study')


def initialize(study, output, stages, bindings, *, gpu=2, reserved_gpu=4, cwd=ROOT):
    study, output, cwd = map(lambda path: Path(path).resolve(), (study, output, cwd))
    if output.exists():
        raise FileExistsError('Resource attempt is immutable; use a fresh attempt directory')
    stages = list(stages)
    _validate_registration(study, stages, gpu, reserved_gpu)
    bound = {str(Path(name).resolve()): digest for name, digest in bindings.items()}
    for source in (Path(__file__), Path(legacy.__file__), Path(inventory_module.__file__)):
        bound[str(source.resolve())] = legacy.sha(source)
    for filename, digest in bound.items():
        if legacy.sha(filename) != digest:
            raise ValueError(f'Input binding does not match at registration: {filename}')
    protocol = dict(schema=SCHEMA, study=str(study), cwd=str(cwd),
        stages=[stage.registration() for stage in stages], bindings=bound,
        resource_policy=dict(physical_gpu=gpu, reserved_gpu=reserved_gpu, max_cuda_workers=1,
            admission_seconds=0, no_foreign_or_unknown_processes=True,
            busy_before_launch='CPU-only indefinite wait',
            conflict_during_child='Stop only identity-bound child; no automatic retry',
            memory_or_utilization_never_authorizes_foreign_sharing=True,
            authorization='User: immediately use one empty card and leave the other card'),
        evaluation_scope='exploratory train-internal B group OOF already inspected',
        formal_test_enabled=False, formal_validation_enabled=False,
        automatic_retries=False, created_unix=time.time())
    output.mkdir(parents=True)
    legacy.dump(output / 'resource_protocol.json', protocol)
    legacy.dump(output / 'resource_registration.json',
                dict(protocol_sha256=legacy.sha(output / 'resource_protocol.json')))
    return protocol


def check_registration(output):
    output = Path(output)
    protocol = legacy.read_json(output / 'resource_protocol.json')
    registration = legacy.read_json(output / 'resource_registration.json')
    if (protocol.get('schema') != SCHEMA or
            registration.get('protocol_sha256') != legacy.sha(output / 'resource_protocol.json')):
        raise ValueError('Resource protocol differs from its independent SHA registration')
    stages = [RegisteredStage.from_registration(value) for value in protocol['stages']]
    policy = protocol['resource_policy']
    _validate_registration(protocol['study'], stages, policy['physical_gpu'], policy['reserved_gpu'])
    if (policy['admission_seconds'] != 0 or policy['max_cuda_workers'] != 1 or
            protocol.get('automatic_retries') is not False or
            protocol.get('formal_test_enabled') is not False or
            protocol.get('formal_validation_enabled') is not False):
        raise ValueError('Registered resource or formal-holdout policy changed')
    for filename, digest in protocol['bindings'].items():
        if legacy.sha(filename) != digest:
            raise ValueError(f'Frozen sparse study binding changed: {filename}')
    return protocol


def process_snapshot(pid):
    """Read birth identity even when a zombie no longer has a command line."""
    directory = Path('/proc') / str(int(pid))
    try:
        text = (directory / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return dict(pid=int(pid), uid=directory.stat().st_uid, state=fields[0],
            start_time_ticks=int(fields[19]), exit_code=int(fields[49]) if len(fields) > 49 else None)
    except FileNotFoundError:
        return None


class SingleCardSupervisor(legacy.SequentialSupervisor):
    def __init__(self, *args, gpu=2, reserved_gpu=4, inventory=inventory_module.gpu_inventory,
                 identity_reader=inventory_module.process_identity, snapshot_reader=process_snapshot, **kwargs):
        super().__init__(*args, environment=CPU_ENV, **kwargs)
        _validate_registration(self.study_output, self.stages, gpu, reserved_gpu)
        self.gpu, self.reserved_gpu = gpu, reserved_gpu
        self.inventory, self.identity_reader = inventory, identity_reader
        self.snapshot_reader = snapshot_reader
        self.stage_by_name = {stage.name: stage for stage in self.stages}
        self.trigger = None
        self.stop_directories = tuple(dict.fromkeys(self.stop_directories + (self.output.parent,)))
        self.registration['schema'] = SCHEMA
        self.registration['resource_policy'] = dict(physical_gpu=gpu, reserved_gpu=reserved_gpu,
            max_cuda_workers=1, admission_seconds=0, strict_identity=True,
            never_terminate_foreign_processes=True, automatic_retry=False)
        self.registration['evaluation_scope'] = 'exploratory train-internal B OOF'

    def _state(self, state, **details):
        value = dict(schema=SCHEMA, state=state, supervisor_pid=os.getpid(), updated_unix=time.time(),
            queue_protocol_sha256=self.protocol_sha, stage=self.active_stage,
            physical_gpu=self.gpu, reserved_gpu=self.reserved_gpu,
            formal_test_opened=False, formal_validation_opened=False, **details)
        if self.trigger is not None and state in ('failed', 'stopped', 'interrupted'):
            value['trigger_evidence'] = dict(self.trigger, termination_started=True,
                termination_completed=self.child is None or self.child.poll() is not None)
        legacy.dump(self.output / 'queue_state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return value

    def _capture_child(self):
        expected = self.identity_reader(self.child.pid)
        stage = self.stage_by_name[self.active_stage]
        if (expected['uid'] != os.getuid() or expected['argv'] != list(stage.command) or
                Path(expected['cwd']).resolve() != self.cwd):
            raise ValueError('Child identity does not match the exact registered command, UID and cwd')
        if os.getpgid(self.child.pid) != self.child.pid:
            raise ValueError('Registered child does not own its exclusive process group')
        self.owned_processes[self.child.pid] = expected
        legacy.dump(self.output / f'child_identity_{stage.name}.json', expected)

    def observation(self):
        stage = self.stage_by_name[self.active_stage]
        gpu = next((item for item in self.inventory() if item['index'] == self.gpu), None)
        if gpu is None:
            raise legacy.StopRequested(f'Registered physical GPU{self.gpu} is unavailable')
        accepted, blocked = [], []
        for pid in gpu['pids']:
            try:
                current = self.identity_reader(pid)
                expected = self.owned_processes.get(pid)
                same = expected is not None and all(current.get(key) == expected.get(key)
                    for key in ('pid', 'uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd'))
                if same and current['uid'] == os.getuid() and self.child is not None and pid == self.child.pid:
                    accepted.append(pid)
                    continue
                try:
                    owner = pwd.getpwuid(current['uid']).pw_name
                except KeyError:
                    owner = f'uid:{current["uid"]}'
                blocked.append(dict(pid=pid, classification='foreign', uid=current['uid'], owner=owner,
                    start_time_ticks=current['start_time_ticks'], command_sha256=current['command_sha256'],
                    reason='Not the identity-bound live child of this exact attempt'))
            except (OSError, ValueError, KeyError) as exc:
                blocked.append(dict(pid=pid, classification='unknown', error_type=type(exc).__name__,
                    reason='Unverified ownership blocks admission'))
        return dict(**gpu, observed_unix=time.time(), owned_pids=accepted, blocking_processes=blocked,
            completely_idle=not gpu['pids'], admission_seconds=0,
            eligible=not blocked and not accepted and not gpu['pids'] and
                     gpu['free_gib'] >= stage.minimum_free_gib,
            reserved_gpu=self.reserved_gpu, reserved_gpu_used_by_this_attempt=False)

    def _guard_gpu_idle(self):
        if not self.stage_by_name[self.active_stage].gpu_required:
            return
        last_status = -math.inf
        while True:
            super()._check()
            observation = self.observation()
            if observation['eligible']:
                self._state('empty_gpu_admitted', resource_observation=observation)
                return
            if time.monotonic() - last_status >= 5:
                self._state('CPU_wait_for_empty_gpu', resource_observation=observation,
                            GPU_child_started=False, automatic_retry=False)
                last_status = time.monotonic()
            time.sleep(self.poll_seconds)

    def _stop_for_conflict(self, observation):
        self.trigger = dict(observed_unix=time.time(), stage=self.active_stage,
            gpu_observation=observation, owned_child_pid=self.child.pid if self.child else None,
            queue_protocol_sha256=self.protocol_sha, foreign_processes_signalled=False,
            termination_started=False)
        legacy.dump(self.output / 'ownership_stop_trigger.json', self.trigger)
        self._state('stopping_for_ownership_conflict', trigger_evidence=self.trigger)
        raise legacy.StopRequested(f'GPU{self.gpu} has foreign/unknown ownership; stop only this owned child')

    def _check(self):
        super()._check()
        stage = self.stage_by_name.get(self.active_stage)
        if stage is None or not stage.gpu_required or self.child is None or self.child.poll() is not None:
            return
        if self.child.pid not in self.owned_processes:
            deadline = time.monotonic() + 1
            while True:
                try:
                    self._capture_child()
                    break
                except (OSError, ValueError):
                    if self.child.poll() is not None:
                        return
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        observation = self.observation()
        if observation['blocking_processes']:
            # A just-exited own PID may linger in nvidia-smi. Do not treat any
            # foreign process, reused live PID, or incomplete publication as it.
            code = self.child.poll()
            own_exit = code == 0 and all(item['classification'] == 'unknown' and
                item['pid'] == self.child.pid for item in observation['blocking_processes'])
            if own_exit:
                current = self.snapshot_reader(self.child.pid)
                expected = self.owned_processes[self.child.pid]
                ended = current is None or (current['state'] in ('Z', 'X') and
                    current['start_time_ticks'] == expected['start_time_ticks'] and
                    current['uid'] == expected['uid'] and current.get('exit_code') in (None, 0))
                if ended:
                    validate_stage(stage, self.stop_directories)
                    legacy.dump(self.output / f'verified_own_exit_{stage.name}.json',
                        dict(pid=self.child.pid, exit_code=code, observation=observation,
                             registered_process_identity=expected, ended_process_snapshot=current,
                             completed_artifacts_verified=True))
                    return
            self._stop_for_conflict(observation)

    def _execute(self, stage):
        self.active_stage = stage.name
        self._check()
        if self._receipt(stage).exists():
            raise RuntimeError('An execution attempt cannot relaunch or silently adopt old receipts')
        self._guard_gpu_idle()
        inputs = {name: legacy.sha(name, self.stop_directories) for name in stage.inputs}
        # Input hashing can be lengthy; close the admission-to-launch interval
        # with another ownership check immediately before the child starts.
        self._guard_gpu_idle()
        log_path = self.output / 'queue_logs' / f'{stage.name}.log'
        log_path.parent.mkdir(exist_ok=True)
        environment = dict(os.environ, **CPU_ENV)
        environment['CUDA_VISIBLE_DEVICES'] = str(self.gpu) if stage.gpu_required else ''
        began = time.time()
        with log_path.open('a') as log:
            log.write(json.dumps(dict(stage=stage.name, command=list(stage.command),
                queue_protocol_sha256=self.protocol_sha, started_unix=began)) + '\n')
            log.flush()
            self.child = subprocess.Popen(stage.command, cwd=self.cwd, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self._state('running', child_pid=self.child.pid, log_path=str(log_path), started_unix=began)
            last_heartbeat = time.monotonic()
            while self.child.poll() is None:
                self._check()
                if time.monotonic() - last_heartbeat >= 5:
                    self._state('running', child_pid=self.child.pid, log_path=str(log_path),
                                started_unix=began, elapsed_stage_seconds=time.time() - began)
                    last_heartbeat = time.monotonic()
                time.sleep(self.poll_seconds)
            self._check()
            returncode = self.child.returncode
        if returncode != 0:
            raise RuntimeError(f'{stage.name} exited with code {returncode}; inspect {log_path}')
        validate_stage(stage, self.stop_directories)
        if inputs != {name: legacy.sha(name, self.stop_directories) for name in stage.inputs}:
            raise ValueError(f'{stage.name}: registered stage inputs changed during execution')
        receipt = dict(state='completed', stage=stage.name,
            queue_protocol_sha256=self.protocol_sha, command_sha256=legacy.canonical_sha(stage.registration()),
            command=list(stage.command), input_sha256=inputs,
            output_sha256={name: legacy.sha(name, self.stop_directories) for name in stage.outputs},
            owned_process_identity=self.owned_processes.get(self.child.pid),
            started_unix=began, finished_unix=time.time(), log_path=str(log_path))
        legacy.dump(self._receipt(stage), receipt)
        self._state('stage_completed', receipt_path=str(self._receipt(stage)), seconds=time.time() - began)
        self.owned_processes.clear()
        self.child = None


def run(output):
    output = Path(output).resolve()
    with (output / 'resource_supervisor.pid').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another guardian owns this resource attempt') from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        if (output / 'resource_state.json').exists():
            raise RuntimeError('Resource attempt was already executed; no automatic retry')
        protocol = check_registration(output)
        stages = [RegisteredStage.from_registration(value) for value in protocol['stages']]
        policy = protocol['resource_policy']
        bindings = dict(protocol['bindings'])
        for path in (output / 'resource_protocol.json', output / 'resource_registration.json'):
            bindings[str(path)] = legacy.sha(path)
        legacy.dump(output / 'resource_state.json', dict(schema=SCHEMA, state='running',
            supervisor_pid=os.getpid(), physical_gpu=policy['physical_gpu'],
            reserved_gpu=policy['reserved_gpu'], automatic_retry=False))
        try:
            supervisor = SingleCardSupervisor(protocol['study'], stages, bindings,
                queue_state_dir=output / 'execution', cwd=protocol['cwd'],
                gpu=policy['physical_gpu'], reserved_gpu=policy['reserved_gpu'])
            result = supervisor.run()
            legacy.dump(output / 'resource_state.json', dict(schema=SCHEMA, state='completed',
                stages_completed=len(stages), automatic_retry=False,
                formal_test_opened=False, formal_validation_opened=False,
                physical_gpu=policy['physical_gpu'], reserved_gpu=policy['reserved_gpu']))
            return result
        except BaseException as exc:
            kind = 'stopped' if isinstance(exc, legacy.StopRequested) else 'failed'
            legacy.dump(output / 'resource_state.json', dict(schema=SCHEMA, state=kind,
                error_type=type(exc).__name__, reason=str(exc), automatic_retry=False,
                scientific_conclusion=None, foreign_processes_signalled=False))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, help='JSON with study, stages and frozen bindings')
    parser.add_argument('--gpu', type=int, choices=(2, 4), default=2)
    parser.add_argument('--reserved-gpu', type=int, choices=(2, 4), default=4)
    args = parser.parse_args(argv)
    if args.action == 'init':
        if args.plan is None:
            parser.error('init requires --plan')
        plan = legacy.read_json(args.plan)
        bindings = dict(plan['bindings'], **{str(args.plan.resolve()): legacy.sha(args.plan)})
        result = initialize(plan['study'], args.output,
            [RegisteredStage.from_registration(value) for value in plan['stages']], bindings,
            gpu=args.gpu, reserved_gpu=args.reserved_gpu, cwd=plan.get('cwd', ROOT))
    elif args.action == 'run':
        result = run(args.output)
    else:
        state = args.output / 'execution/queue_state.json'
        result = legacy.read_json(state) if state.exists() else check_registration(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
