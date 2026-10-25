"""Resume registered diagnostics, waiting for genuinely idle public GPU2/4.

The backbone extraction stays on GPU4. Only the existing five independent B
folds use GPU2/4, one worker per card, with last and mean views sequential.
No original scientific or resource source is edited by this wrapper.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import ab_diagnostic_followup as original
from scripts.hanguard import ab_diagnostic_queue as queue
from scripts.hanguard import ab_readout_parallel_queue as ownership

SCHEMA = 'hanguard_ab_dual_gpu_resource_followup_v1'
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')


def cpu_stage(original_stage, phase, name):
    command = list(original_stage.command)
    command[2] = phase
    command[command.index('--device') + 1] = 'cpu'
    outputs = original_stage.outputs[:1] if phase == 'init' else original_stage.outputs
    return queue.Stage(name, tuple(command), original_stage.inputs, outputs,
                       'exists' if phase == 'init' else original_stage.validator)


def partial_snapshot(study):
    import numpy as np
    study = Path(study) / 'lora_study'
    folder = study / 'features/adapted'
    progress_path = folder / 'progress.json'
    if not progress_path.exists():
        if any(folder.glob('*.partial.npy')):
            raise ValueError('Orphan partial cache has no committed extraction progress')
        return dict(resume_progress_present=False)
    progress = queue.read_json(progress_path)
    protocol = queue.read_json(study / 'protocol.json')
    partition = queue.read_json(protocol['partition_manifest_path'])
    adapter = study / 'runs' / f'E03_s{protocol["seeds"][0]}' / 'adapter.pt'
    expected = dict(B_sha256=partition['artifact_sha256']['B.parquet'], representation='adapted',
        adapter_sha256=queue.sha(adapter), training_protocol_sha256=queue.sha(study / 'protocol.json'),
        shape=progress['identity']['shape'])
    model = queue.read_json(Path(protocol['model_path']) / 'config.json')
    expected['shape'] = [partition['partitions']['B']['rows'], 32, model['text_config']['hidden_size']]
    if (progress['identity'] != expected or not 0 <= progress['next_group'] <= progress['groups'] or
            not 0 <= progress['rows'] <= progress['total_rows'] or
            progress['total_rows'] != expected['shape'][0]):
        raise ValueError('Partial extraction differs from the original A/B/adapter identity')
    artifacts = {}
    for pool in ('last', 'mean'):
        path = folder / f'{pool}.partial.npy'
        array = np.load(path, mmap_mode='r')
        if list(array.shape) != expected['shape'] or array.dtype != np.float16:
            raise ValueError('Partial cache dimensions or dtype changed')
        artifacts[str(path)] = queue.sha(path)
    return dict(resume_progress_present=True, identity=expected, next_group=progress['next_group'],
        committed_rows=progress['rows'], total_rows=progress['total_rows'],
        progress_path=str(progress_path), progress_sha256=queue.sha(progress_path), artifact_sha256=artifacts)


def initialize(study, output, base_attempt, python=queue.DEFAULT_PYTHON):
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol = original.initialize(study, output, base_attempt, python=python)
    A = original.verify_A_completion(study, protocol['original_A_process_identity'])
    base = original.verify_base_completion(study, base_attempt)
    if not A['ready'] or not base['ready']:
        raise ValueError('This resource continuation requires frozen A and complete BASE')
    prerequisites = dict(A=A, base=base, existing_processes_signalled=False)
    queue.dump(output / 'prerequisites.json', prerequisites)
    snapshot = partial_snapshot(study)
    queue.dump(output / 'initial_partial_snapshot.json', snapshot)
    resource = dict(schema=SCHEMA, study=str(study), python=str(Path(python).absolute()),
        scientific_protocol_path=str(output / 'continuation_protocol.json'),
        scientific_protocol_sha256=queue.sha(output / 'continuation_protocol.json'),
        bindings={str(Path(__file__).resolve()): queue.sha(__file__),
                  str(output / 'prerequisites.json'): queue.sha(output / 'prerequisites.json'),
                  str(output / 'initial_partial_snapshot.json'): queue.sha(output / 'initial_partial_snapshot.json')},
        resource_policy=dict(extraction_gpu=4, readout_candidates=[2, 4], max_workers=2,
            one_fold_per_gpu=True, views_sequential=['last', 'mean'],
            no_foreign_or_unknown_processes=True, continuous_empty_gpu_confirmation_seconds=60,
            wait_without_gpu_children='indefinite', minimum_extraction_free_gib=26,
            minimum_readout_free_gib=4, own_successful_exit_race_requires_complete_fold_SHA=True,
            supersedes_scientific_snapshot_resource_policy_for_readouts_only=True),
        scientific_settings_changed=False, retrain_A=False, retrain_BASE=False, automatic_retries=False)
    queue.dump(output / 'resource_protocol.json', resource)
    queue.dump(output / 'resource_registration.json',
               dict(protocol_sha256=queue.sha(output / 'resource_protocol.json')))
    return resource


def check_registration(output):
    output = Path(output)
    original_protocol = original.check_registration(output)
    path = output / 'resource_protocol.json'
    resource = queue.read_json(path)
    if (resource.get('schema') != SCHEMA or queue.sha(path) !=
            queue.read_json(output / 'resource_registration.json')['protocol_sha256'] or
            resource['scientific_protocol_sha256'] != queue.sha(resource['scientific_protocol_path'])):
        raise ValueError('Resource or scientific protocol differs from independent registration')
    for name, digest in resource['bindings'].items():
        if queue.sha(name) != digest:
            raise ValueError(f'Frozen resource continuation binding changed: {name}')
    return resource, original_protocol


class WaitingExtraction(original.StrictFollowup):
    """No GPU child exists while this guard waits, however long the card is busy."""
    def _guard_gpu_idle(self):
        if self.active_stage not in self.gpu_stages:
            return
        last_status = -math.inf
        while True:
            queue.SequentialSupervisor._check(self)
            observation = self.observation()
            if observation['eligible']:
                self._state('strict_GPU4_admitted', resource_observation=observation)
                return
            if time.monotonic() - last_status >= 5:
                self._state('CPU_wait_for_genuinely_idle_GPU4', resource_observation=observation,
                            GPU_child_started=False, automatic_retry=False)
                last_status = time.monotonic()
            time.sleep(.5)


class RaceSafeParallelQueue(ownership.ParallelQueue):
    """Reconcile one exited own worker; live foreign/unknown ownership still blocks."""
    def __init__(self, *args, raw_identity_reader=ownership.process_identity,
                 snapshot_reader=original.process_snapshot, extra_stop_dirs=(), **kwargs):
        self.raw_identity_reader, self.snapshot_reader = raw_identity_reader, snapshot_reader
        super().__init__(*args, identity_reader=self.resolve_identity, **kwargs)
        self.stop_dirs = tuple(dict.fromkeys(self.stop_dirs + tuple(map(Path, extra_stop_dirs))))
        self.protocol['own_exit_reconciliation'] = 'poll0+registered_birth_UID+ended_process+complete_fold_SHA'
        self.protocol_file_sha = None

    def check(self):
        super().check()
        path = self.output / 'protocol.json'
        if queue.read_json(path) != self.protocol:
            raise ValueError('Immutable parallel resource protocol changed')
        current = queue.sha(path)
        if self.protocol_file_sha is None:
            self.protocol_file_sha = current
        elif current != self.protocol_file_sha:
            raise ValueError('Immutable parallel resource protocol SHA changed')

    def resolve_identity(self, pid):
        try:
            return self.raw_identity_reader(pid)
        except (OSError, ValueError) as exc:
            match = next(((fold, worker) for fold, worker in self.children.items()
                          if worker['process'].pid == pid), None)
            expected = self.owned.get(pid)
            if (match is None or expected is None or not ownership._study_process(expected, self.study) or
                    match[1]['process'].poll() != 0 or self.validator is None):
                raise
            current = self.snapshot_reader(pid)
            if current is not None and (current['state'] not in ('Z', 'X') or
                    current.get('start_time_ticks') != expected['start_time_ticks'] or
                    current.get('uid') != expected['uid'] or current.get('exit_code') not in (None, 0)):
                raise
            artifacts = self.validator(match[0])
            queue.dump(self.output / f'verified_own_exit_fold_{match[0]}.json',
                dict(pid=pid, outer_fold=match[0], exit_code=0, original_error_type=type(exc).__name__,
                    registered_process_identity=expected, ended_process_snapshot=current,
                    complete_fold_artifact_sha256=artifacts, no_foreign_exception=True,
                    process_exit_before_ownership_read_reconciled=True))
            return expected


def run_cpu(study, output, stages, bindings):
    supervisor = queue.SequentialSupervisor(study, stages, bindings, queue_state_dir=output,
                                            environment=CPU_ENV)
    supervisor.stop_directories = tuple(dict.fromkeys(supervisor.stop_directories +
        (Path(output).parent, Path(study) / 'lora_study')))
    return supervisor.run()


def _run_steps(output):
    output = Path(output)
    state_path = output / 'resource_state.json'
    if state_path.exists() and queue.read_json(state_path)['state'] in ('failed', 'stopped'):
        raise RuntimeError('Failed/stopped attempt cannot retry; explicit new attempt required')
    resource, scientific = check_registration(output)
    bindings = dict(scientific['bindings'], **resource['bindings'])
    bindings[str(output / 'resource_protocol.json')] = queue.sha(output / 'resource_protocol.json')
    bindings[str(output / 'continuation_protocol.json')] = queue.sha(output / 'continuation_protocol.json')
    study, python = Path(resource['study']), resource['python']
    stages = original.followup_stages(study, python)
    if [stage.registration() for stage in stages] != scientific['stages']:
        raise ValueError('Original scientific commands changed')
    try:
        A = original.verify_A_completion(study, scientific['original_A_process_identity'])
        base = original.verify_base_completion(study, scientific['base_attempt'])
        if not A['ready'] or not base['ready']:
            raise ValueError('Frozen A or complete BASE prerequisite is no longer ready')
        snapshot = queue.read_json(output / 'initial_partial_snapshot.json')
        if snapshot != partial_snapshot(study):
            raise ValueError('Committed partial extraction snapshot changed before resumption')
        extraction = WaitingExtraction(study, stages[:1], bindings,
            queue_state_dir=output / 'extraction', environment=dict(CPU_ENV, CUDA_VISIBLE_DEVICES='4'))
        queue.dump(state_path, dict(state='waiting_then_extracting_adapted', automatic_retry=False))
        extraction.run()
        queue.validate_outputs(stages[0], (study, output, study / 'lora_study'))
        for stage, view in zip(stages[1:3], ('last', 'mean')):
            check_registration(output)
            run_cpu(study, output / f'{view}_init',
                    [cpu_stage(stage, 'init', f'initialize_{view}_original_protocol')], bindings)
            readout = study / 'readouts' / f'adapted_{view}'
            protocol, plan_bindings, commands, report, _ = ownership.registered_plan(study, readout, python)
            ownership.ensure_no_existing_readout_writer(readout)
            merged = dict(bindings, **plan_bindings)
            dispatcher = RaceSafeParallelQueue(study, output / f'{view}_folds', readout, commands, merged,
                candidates=(2, 4), max_workers=2, watch_seconds=math.inf,
                validator=lambda fold, readout=readout: ownership.validate_fold(readout, fold, queue.sha(readout / 'protocol.json')),
                extra_stop_dirs=(output,))
            queue.dump(state_path, dict(state=f'parallel_{view}_folds', candidates=[2, 4], max_workers=2))
            result = dispatcher.run()
            if result['state'] != 'folds_completed':
                raise RuntimeError('Registered folds did not finish; no automatic retry')
            run_cpu(study, output / f'{view}_report',
                    [cpu_stage(stage, 'report', f'aggregate_{view}_original_report')], merged)
            queue.validate_outputs(stage, (study, output, study / 'lora_study'))
        check_registration(output)
        run_cpu(study, output / 'error_analysis', [stages[-1]], bindings)
        queue.summarize_results(study, output)
        queue.dump(state_path, dict(state='completed', scientific_settings_changed=False,
                                   summary_path=str(output / 'diagnostic_summary.json')))
    except BaseException as exc:
        queue.dump(state_path, dict(state='stopped' if isinstance(exc, queue.StopRequested) else 'failed',
            reason=str(exc), error_type=type(exc).__name__, automatic_retry=False,
            scientific_conclusion=None, no_A_or_BASE_retraining=True))
        raise


def run(output):
    output = Path(output).resolve()
    with (output / 'resource_supervisor.pid').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another resource coordinator owns this attempt') from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        try:
            return _run_steps(output)
        except BaseException as exc:
            state_path = output / 'resource_state.json'
            prior = queue.read_json(state_path) if state_path.exists() else {}
            if prior.get('state') not in ('failed', 'stopped'):
                queue.dump(state_path, dict(state='stopped' if isinstance(exc, queue.StopRequested) else 'failed',
                    reason=str(exc), error_type=type(exc).__name__, automatic_retry=False,
                    scientific_conclusion=None, no_A_or_BASE_retraining=True))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'status'))
    parser.add_argument('--study', type=Path, default=queue.DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-attempt', type=Path)
    parser.add_argument('--python', type=Path, default=queue.DEFAULT_PYTHON)
    args = parser.parse_args(argv)
    if args.action == 'init':
        print(json.dumps(initialize(args.study, args.output,
            args.base_attempt or args.study / 'readout_parallel_base_last_attempt2', args.python), indent=2))
    elif args.action == 'status':
        print((args.output / 'resource_state.json').read_text())
    else:
        run(args.output)


if __name__ == '__main__':
    main()
