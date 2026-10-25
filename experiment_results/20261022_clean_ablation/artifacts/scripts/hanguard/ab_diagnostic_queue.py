"""Sequential, fail-closed supervisor for the registered A/B diagnosis.

``plan`` only prints commands. ``run`` requires the already completed preflight
gate, uses physical GPU 4 exclusively, and never retries a failed stage. A STOP
file terminates only the supervisor's own subprocess group. Completed stages
can be skipped only after command, input, output, and protocol hashes match.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/hanguard_ab_diagnosis_20261012'
DEFAULT_PYTHON = ROOT / '.venv-hanguard/bin/python'
SCHEMA = 'hanguard_ab_sequential_supervisor_v1'


def sha(path, stop_directory=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            stop_directories = ((stop_directory,) if isinstance(stop_directory, (str, Path))
                                else (stop_directory or ()))
            for folder in stop_directories:
                stop = Path(folder) / 'STOP'
                if stop.exists():
                    raise StopRequested(stop.read_text().strip() or 'STOP requested')
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    validator: str = 'exists'

    def registration(self):
        return dict(name=self.name, command=list(self.command), inputs=list(self.inputs),
                    outputs=list(self.outputs), validator=self.validator)


class StopRequested(RuntimeError):
    pass


def preflight_bindings(output):
    """Validate the existing gate; deliberately never launches preflight."""
    study = Path(output) / 'lora_study'
    protocol_path = study / 'protocol.json'
    protocol = read_json(protocol_path)
    registration = read_json(study / 'registration.json')
    if registration.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('LoRA study protocol registration differs')
    if (protocol.get('training_entrypoint') != 'hanguard_ab_diagnostic_v1' or
            protocol.get('adapter_training_cohort') != 'A' or
            protocol.get('formal_test_enabled') is not False or protocol.get('gpu_id') != 4):
        raise ValueError('Queue requires the registered A-only diagnostic on physical GPU 4')
    gate_path = study / 'preflight_gate.json'
    gate = read_json(gate_path)
    if gate.get('passed') is not True or gate.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('An already completed, matching GPU preflight gate is required')
    files = [protocol_path, study / 'registration.json', study / 'arms.json', gate_path,
             Path(protocol['partition_manifest_path'])]
    if sha(files[-1]) != protocol['partition_manifest_sha256']:
        raise ValueError('Partition manifest differs from A-only protocol')
    for arm in ('E03', 'E04'):
        path = study / 'preflight' / f'{arm}_s{protocol["seeds"][0]}.json'
        if sha(path) != gate['result_sha256'][arm] or read_json(path).get('passed') is not True:
            raise ValueError(f'{arm} preflight failed or changed')
        files.append(path)
    for name, digest in protocol['code_sha256'].items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError(f'Registered training code changed: {name}')
        files.append(path)
    for name in ('scripts/hanguard/ab_diagnostic_queue.py',
                 'scripts/hanguard/ab_frozen_readout_diagnostics.py',
                 'scripts/hanguard/ab_diagnostic_error_analysis.py',
                 'scripts/hanguard/ab_readout_parallel_queue.py'):
        files.append(ROOT / name)
    return {str(path.resolve()): sha(path) for path in files}, protocol


def build_plan(output=DEFAULT_OUTPUT, python=DEFAULT_PYTHON):
    # Keep the venv executable symlink path: resolving it bypasses venv setup.
    output, python = Path(output).resolve(), str(Path(python).absolute())
    study = output / 'lora_study'
    seed = read_json(study / 'protocol.json')['seeds'][0]
    experiment = str(ROOT / 'scripts/hanguard/ab_diagnostic_experiment.py')
    readouts = str(ROOT / 'scripts/hanguard/ab_frozen_readout_diagnostics.py')
    features = study / 'features'
    adapter = study / 'runs' / f'E03_s{seed}'
    common = ('--output', str(study))
    protocol = str(study / 'protocol.json')
    base_manifests = tuple(str(features / 'base' / f'{pool}.manifest.json') for pool in ('last', 'mean'))
    adapted_manifests = tuple(str(features / 'adapted' / f'{pool}.manifest.json') for pool in ('last', 'mean'))
    stages = [
        Stage('extract_base', (python, experiment, 'extract', *common, '--representation', 'base'),
              (protocol,), base_manifests, 'cache'),
        Stage('train_A_E03', (python, experiment, 'train', *common),
              (protocol, str(study / 'preflight_gate.json')),
              tuple(str(adapter / name) for name in ('selection.json', 'diagnostic_lineage.json', 'adapter.pt', 'head.pt')),
              'adapter'),
        Stage('extract_adapted', (python, experiment, 'extract', *common, '--representation', 'adapted'),
              (protocol, str(adapter / 'selection.json'), str(adapter / 'diagnostic_lineage.json')),
              adapted_manifests, 'cache'),
    ]
    # The readout entrypoint registers immutable settings before its first fit.
    for name, representation, pooling, phase, joint in (
        ('base_last_readouts', 'base', 'last', 'run', True),
        ('adapted_last_readouts', 'adapted', 'last', 'run', True),
        ('adapted_mean_readouts', 'adapted', 'mean', 'run', True),
    ):
        manifest = str(features / representation / f'{pooling}.manifest.json')
        destination = output / 'readouts' / f'{representation}_{pooling}'
        command = (python, readouts, phase, '--cache-manifest', manifest,
                   '--output', str(destination), '--device', 'cuda', '--seed', str(seed),
                   '--probe-epochs', '12', '--head-epochs', '12', '--max-epochs', '40',
                   '--bootstrap-draws', '2000') + (('--joint-features',) if joint else ())
        stages.append(Stage(name, command, (protocol, manifest),
            tuple(str(destination / name) for name in
                  ('protocol.json', 'analysis/results.json', 'analysis/probe_outer_predictions.parquet') +
                  (('analysis/head_outer_predictions.parquet',) if joint else ())),
            'heads' if joint else 'probes'))
    stages.append(Stage('error_analysis',
        (python, str(ROOT / 'scripts/hanguard/ab_diagnostic_error_analysis.py'),
         '--study', str(output), '--output', str(output / 'error_analysis')),
        tuple(str(output / 'readouts' / name / 'analysis/results.json')
              for name in ('base_last', 'adapted_last', 'adapted_mean')),
        (str(output / 'error_analysis/results.json'),), 'error_analysis'))
    return stages


def validate_outputs(stage, stop_directory=None):
    for path in stage.outputs:
        if not Path(path).is_file():
            raise ValueError(f'{stage.name}: expected completed artifact missing: {path}')
    if stage.validator == 'cache':
        protocol_path = Path(stage.inputs[0]).resolve()
        protocol = read_json(protocol_path)
        partition_path = Path(protocol['partition_manifest_path']).resolve()
        partition = read_json(partition_path)
        expected_representation = {'extract_base': 'base', 'extract_adapted': 'adapted'}.get(stage.name)
        if expected_representation is None:
            raise ValueError('Cache stage has no registered representation')
        for path in stage.outputs:
            manifest = read_json(path)
            if manifest.get('complete') is not True or manifest.get('cohort') != 'B':
                raise ValueError('Extraction did not publish a complete B cache')
            if (Path(manifest['training_protocol_path']).resolve() != protocol_path or
                    manifest['training_protocol_sha256'] != sha(protocol_path) or
                    Path(manifest['ab_protocol_path']).resolve() != partition_path or
                    manifest['ab_protocol_sha256'] != protocol['partition_manifest_sha256'] or
                    manifest['B_sha256'] != partition['artifact_sha256']['B.parquet'] or
                    manifest.get('representation') != expected_representation or
                    manifest.get('adapter_training_cohort') != ('none' if expected_representation == 'base' else 'A')):
                raise ValueError('Completed cache differs from the current A/B protocol or representation')
            if expected_representation == 'base' and manifest.get('adapter_sha256') is not None:
                raise ValueError('Base cache cannot contain a trained adapter')
            if expected_representation == 'adapted':
                lineage_path = Path(stage.inputs[-1])
                lineage = read_json(lineage_path)
                if manifest.get('adapter_sha256') != lineage['adapter_sha256']:
                    raise ValueError('Adapted cache differs from selected A-only adapter')
            for key, digest_key in (('path', 'sha256'), ('metadata_path', 'metadata_sha256'),
                                    ('training_protocol_path', 'training_protocol_sha256'),
                                    ('ab_protocol_path', 'ab_protocol_sha256')):
                if sha(manifest[key], stop_directory) != manifest[digest_key]:
                    raise ValueError(f'Completed B cache artifact changed: {key}')
            replay = Path(manifest['metadata_path']).parent / 'replay_audit.json'
            if sha(replay) != manifest['replay_audit_sha256'] or read_json(replay).get('passed') is not True:
                raise ValueError('Fresh cache replay did not pass')
    elif stage.validator == 'adapter':
        selection_path, lineage_path = map(Path, stage.outputs[:2])
        selection, lineage = read_json(selection_path), read_json(lineage_path)
        for name, digest in selection['checkpoint_hashes'].items():
            if sha(selection_path.parent / name) != digest:
                raise ValueError('Selected checkpoint hash changed')
        if (lineage.get('adapter_training_cohort') != 'A' or
                lineage.get('formal_test_opened') is not False or
                lineage.get('adapter_sha256') != selection['checkpoint_hashes']['adapter.pt']):
            raise ValueError('Selected adapter lacks valid A-only lineage')
    elif stage.validator in ('probes', 'heads'):
        protocol_path, results_path = map(Path, stage.outputs[:2])
        result = read_json(results_path)
        if result.get('protocol_sha256') != sha(protocol_path) or not result.get('layer_metrics'):
            raise ValueError('Readout results do not match their registered protocol')
        if stage.validator == 'heads' and not result.get('head_metrics'):
            raise ValueError('Full readout stage has no completed head results')
    elif stage.validator == 'error_analysis':
        result = read_json(stage.outputs[0])
        if result.get('complete') is not True or result.get('schema') != 'hanguard_ab_error_analysis_v1':
            raise ValueError('Strict paired error analysis is incomplete')
    elif stage.validator != 'exists':
        raise ValueError(f'Unknown stage validator: {stage.validator}')


class SequentialSupervisor:
    """A generic CPU-testable execution engine; production bindings are external."""
    def __init__(self, output, stages, bindings, *, poll_seconds=.5, terminate_seconds=10,
                 environment=None, cwd=ROOT, queue_state_dir=None, minimum_free_gib=None,
                 resource_evidence=None, owned_processes=None):
        self.study_output = Path(output).resolve()
        self.output = Path(queue_state_dir).resolve() if queue_state_dir else self.study_output
        self.stop_directories = tuple(dict.fromkeys((self.study_output, self.output)))
        self.stages = list(stages)
        self.bindings = dict(bindings)
        self.poll_seconds = float(poll_seconds)
        self.terminate_seconds = float(terminate_seconds)
        if not 0 < self.poll_seconds <= 5 or not 0 < self.terminate_seconds <= 30:
            raise ValueError('STOP polling must be <=5 seconds and termination <=30 seconds')
        self.environment = dict(environment or {})
        if minimum_free_gib is not None and (not math.isfinite(minimum_free_gib) or minimum_free_gib <= 0):
            raise ValueError('Minimum free GPU memory must be positive and finite')
        self.minimum_free_gib = minimum_free_gib
        self.owned_processes = dict(owned_processes or {})
        self.cwd = Path(cwd).resolve()
        self.child = None
        self.active_stage = None
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError('Stage names must be unique')
        self.registration = dict(schema=SCHEMA, stages=[stage.registration() for stage in self.stages],
            bindings=self.bindings, environment=self.environment, cwd=str(self.cwd),
            study_output=str(self.study_output), queue_state_dir=str(self.output),
            resource_policy=dict(mode='strict_ownership',
                physical_gpu=4, minimum_free_gib=minimum_free_gib,
                owned_processes=list(self.owned_processes.values()),
                memory_or_utilization_never_authorizes_foreign_sharing=True,
                evidence=resource_evidence or {}, never_terminate_foreign_processes=True),
            stop_poll_seconds=self.poll_seconds, terminate_seconds=self.terminate_seconds,
            automatic_retries=False, formal_test_enabled=False)

    def _check(self):
        for folder in self.stop_directories:
            stop = folder / 'STOP'
            if stop.exists():
                raise StopRequested(stop.read_text().strip() or 'STOP requested')
        for filename, digest in self.bindings.items():
            if sha(filename) != digest:
                raise ValueError(f'Frozen queue binding changed: {filename}')
        if sha(self.output / 'queue_protocol.json') != self.protocol_sha:
            raise ValueError('Frozen queue protocol changed')

    def _state(self, state, **details):
        value = dict(schema=SCHEMA, state=state, supervisor_pid=os.getpid(),
            updated_unix=time.time(), queue_protocol_sha256=self.protocol_sha,
            stage=self.active_stage, formal_test_opened=False, **details)
        dump(self.output / 'queue_state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return value

    def _terminate_owned_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        # start_new_session below makes this group exclusively ours.
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.terminate_seconds
        while self.child.poll() is None and time.monotonic() < deadline:
            time.sleep(min(self.poll_seconds, .2))
        if self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.child.wait()

    def _receipt(self, stage):
        return self.output / 'queue_receipts' / f'{stage.name}.json'

    def _progress(self, stage, began):
        details = dict(elapsed_stage_seconds=time.time() - began,
                       estimated_stage_remaining_seconds=None,
                       remaining_time_scope='Current stage only; later stages are not timed yet.')
        paths = []
        if stage.name.startswith('extract_') and stage.outputs:
            paths = [Path(stage.outputs[0]).parent / 'progress.json']
        elif stage.name == 'train_A_E03' and stage.outputs:
            paths = [Path(stage.outputs[0]).parent / 'status.json']
        for path in paths:
            if not path.exists():
                continue
            try:
                value = read_json(path)
                rows, total = value.get('rows', 0), value.get('total_rows', 0)
                elapsed = value.get('training_seconds', value.get('seconds', 0))
                details['child_progress'] = value
                if rows > 0 and total >= rows and elapsed > 0:
                    details['estimated_stage_remaining_seconds'] = (total - rows) * elapsed / rows
                    details['remaining_time_scope'] = 'Throughput estimate; loading/validation and later stages are excluded.'
            except (ValueError, OSError):
                pass  # Advisory only; completion identities remain strict.
        return details

    def _gpu_observation(self):
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True, timeout=10)
        row = next(([value.strip() for value in line.split(',')] for line in gpu.stdout.splitlines()
                    if line.split(',')[0].strip() == '4'), None)
        if row is None or len(row) != 4:
            raise ValueError('Physical GPU 4 is unavailable')
        uuid, free_gib, total_gib = row[1], float(row[2]) / 1024, float(row[3]) / 1024
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                              check=True, capture_output=True, text=True, timeout=10)
        occupied = [int(line.split(',')[1].strip()) for line in apps.stdout.splitlines()
                    if line.split(',')[0].strip() == uuid]
        from scripts.hanguard.ab_readout_parallel_queue import assess_gpu
        decision = assess_gpu(dict(index=4, uuid=uuid, free_gib=free_gib, pids=occupied),
            self.owned_processes, self.study_output, minimum_free_gib=self.minimum_free_gib or 0)
        return dict(physical_gpu=4, uuid=uuid, free_gib=free_gib, total_gib=total_gib,
            compute_pids=occupied, owned_compute_pids=decision['owned_pids'],
            foreign_compute_pids=[process['pid'] for process in decision['foreign_processes']],
            foreign_processes=decision['foreign_processes'], observed_unix=time.time(),
            policy='Zero foreign processes; memory is checked only after ownership')

    def _guard_gpu_idle(self):
        if self.environment.get('CUDA_VISIBLE_DEVICES') != '4':
            return
        observation = self._gpu_observation()
        occupied = observation['foreign_compute_pids']
        dump(self.output / 'resource_observation.json', dict(**observation,
            stage=self.active_stage, resource_policy=self.registration['resource_policy']))
        if occupied:
            raise RuntimeError(f'Physical GPU 4 is occupied by foreign/unknown compute PIDs {occupied}; '
                               'queue stopped without terminating them; free memory does not permit sharing')
        if self.minimum_free_gib is not None:
            if observation['free_gib'] < self.minimum_free_gib:
                raise RuntimeError(f'Physical GPU 4 free memory {observation["free_gib"]:.3f} GiB is below required '
                                   f'{self.minimum_free_gib:.3f} GiB; foreign PIDs {occupied} are untouched')
        self._state('resource_check_passed', resource_observation=observation,
                    resource_policy=self.registration['resource_policy'])

    def _skip_completed(self, stage):
        path = self._receipt(stage)
        if not path.exists():
            return False
        receipt = read_json(path)
        expected = dict(command_sha256=canonical_sha(stage.registration()),
                        queue_protocol_sha256=self.protocol_sha,
                        input_sha256={name: sha(name) for name in stage.inputs},
                        output_sha256={name: sha(name) for name in stage.outputs})
        if receipt.get('state') != 'completed' or any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Completed stage provenance changed: {stage.name}')
        validate_outputs(stage, self.stop_directories)
        self._state('skipped_completed', receipt_path=str(path))
        return True

    def _execute(self, stage):
        self.active_stage = stage.name
        self._check()
        if self._skip_completed(stage):
            return
        # The parent may have explicitly finished base extraction before the
        # queue starts. Adopt only a fully hash-validated base cache, never an
        # adapter fit or readout stage with unknown completion provenance.
        if stage.name == 'extract_base' and all(Path(path).is_file() for path in stage.outputs):
            validate_outputs(stage, self.stop_directories)
            receipt = dict(state='completed', stage=stage.name,
                queue_protocol_sha256=self.protocol_sha, command_sha256=canonical_sha(stage.registration()),
                command=list(stage.command), input_sha256={name: sha(name) for name in stage.inputs},
                output_sha256={name: sha(name) for name in stage.outputs},
                adopted_existing_validated_base_cache=True, executed_command_not_recorded_by_queue=True,
                finished_unix=time.time())
            dump(self._receipt(stage), receipt)
            self._state('adopted_completed_base_cache', receipt_path=str(self._receipt(stage)))
            return
        self._guard_gpu_idle()
        inputs = {name: sha(name) for name in stage.inputs}
        log_path = self.output / 'queue_logs' / f'{stage.name}.log'
        log_path.parent.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment.update(self.environment)
        began = time.time()
        with log_path.open('a') as log:
            log.write(json.dumps(dict(queue_stage=stage.name, command=list(stage.command),
                queue_protocol_sha256=self.protocol_sha, started_unix=began)) + '\n')
            log.flush()
            self.child = subprocess.Popen(stage.command, cwd=self.cwd, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self._state('running', child_pid=self.child.pid, log_path=str(log_path),
                        command=list(stage.command), started_unix=began)
            last_heartbeat = time.monotonic()
            while self.child.poll() is None:
                self._check()
                if time.monotonic() - last_heartbeat >= 5:
                    self._state('running', child_pid=self.child.pid, log_path=str(log_path),
                                command=list(stage.command), started_unix=began,
                                **self._progress(stage, began))
                    last_heartbeat = time.monotonic()
                time.sleep(self.poll_seconds)
            self._check()
            returncode = self.child.returncode
        if returncode != 0:
            raise RuntimeError(f'{stage.name} exited with code {returncode}; inspect {log_path}')
        validate_outputs(stage, self.stop_directories)
        if inputs != {name: sha(name) for name in stage.inputs}:
            raise ValueError(f'{stage.name}: frozen stage inputs changed during execution')
        receipt = dict(state='completed', stage=stage.name,
            queue_protocol_sha256=self.protocol_sha, command_sha256=canonical_sha(stage.registration()),
            command=list(stage.command), input_sha256=inputs,
            output_sha256={name: sha(name) for name in stage.outputs},
            started_unix=began, finished_unix=time.time(), log_path=str(log_path))
        dump(self._receipt(stage), receipt)
        self._state('stage_completed', receipt_path=str(self._receipt(stage)),
                    seconds=time.time() - began)
        self.child = None

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        lock = (self.output / 'supervisor.pid').open('a+')
        previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            def terminate_signal(signum, frame):
                raise StopRequested('Supervisor received SIGTERM')
            signal.signal(signal.SIGTERM, terminate_signal)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another supervisor holds the study lock') from exc
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid()) + '\n')
            lock.flush()
            path = self.output / 'queue_protocol.json'
            if path.exists():
                existing = read_json(path)
                if 'initial_resource_observation' in existing:
                    self.registration['initial_resource_observation'] = existing['initial_resource_observation']
                if existing != self.registration:
                    raise ValueError('Existing queue registration differs; create a new version')
            else:
                if self.environment.get('CUDA_VISIBLE_DEVICES') == '4':
                    self.registration['initial_resource_observation'] = self._gpu_observation()
                dump(path, self.registration)
            self.protocol_sha = sha(path)
            previous_path = self.output / 'queue_state.json'
            if previous_path.exists():
                previous = read_json(previous_path)
                if previous.get('state') in ('failed', 'stopped', 'interrupted'):
                    raise RuntimeError('Previous queue stopped or failed; no automatic retry is permitted')
                if previous.get('state') == 'running':
                    stage = next((stage for stage in self.stages if stage.name == previous.get('stage')), None)
                    if stage is None or not self._receipt(stage).exists():
                        raise RuntimeError('Previous stage has no completed receipt; inspect orphan/incomplete work before a new version')
            try:
                for stage in self.stages:
                    self._execute(stage)
                self.active_stage = None
                return self._state('completed', stages_completed=len(self.stages))
            except BaseException as exc:
                self._terminate_owned_child()
                kind = 'stopped' if isinstance(exc, StopRequested) else (
                    'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed')
                detail = self._state(kind, error_type=type(exc).__name__, reason=str(exc),
                    child_pid=self.child.pid if self.child else None,
                    child_returncode=self.child.returncode if self.child else None,
                    automatic_retry=False, scientific_conclusion=None,
                    interpretation='Execution stop gate; this is not evidence about fusion quality.')
                dump(self.output / 'diagnostic_summary.json', dict(
                    status=kind, failure=detail, scientific_conclusions=[],
                    outstanding_stage=self.active_stage, formal_test_opened=False))
                raise
        finally:
            lock.close()
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)


def summarize_results(output, queue_state_dir=None):
    """Bound interpretation to this fixed adapter and pre-registered readouts."""
    output = Path(output).resolve()
    results = {name: read_json(output / 'readouts' / name / 'analysis/results.json')
               for name in ('base_last', 'adapted_last', 'adapted_mean')}
    comparisons = results['adapted_last']['comparisons']

    def comparison(candidate, baseline):
        return next((entry for entry in comparisons
                     if entry['candidate'] == candidate and entry['baseline'] == baseline), None)

    def positive(entry):
        return entry is not None and entry['group_bootstrap']['accuracy']['ci95'][0] > 0

    best_layer = comparison('best_single', 'last')
    capacity = comparison('repeat_best', 'best_single')
    fusion = [comparison(arm, 'repeat_best') for arm in
              ('fusion_fixed', 'fusion_selected', 'fusion_fixed_best_anchor', 'fusion_selected_best_anchor')]
    joint = [comparison(arm, 'joint_repeat_best') for arm in ('joint_fixed', 'joint_selected')]
    anchor = [comparison('fusion_fixed_best_anchor', 'fusion_fixed'),
              comparison('fusion_selected_best_anchor', 'fusion_selected')]
    fusion_signal = any(map(positive, fusion))
    readout_signal = any(map(positive, joint))
    matrix = [
        dict(hypothesis='Last-layer choice hides a stronger single layer',
             status='supported_in_this_screening' if positive(best_layer) else 'not_established',
             evidence=[best_layer], required_followup='Replicate with fresh LoRA seeds before a general claim.'),
        dict(hypothesis='Additional head capacity explains apparent fusion gain',
             status='capacity_signal_present' if positive(capacity) else 'not_established',
             evidence=[capacity] + fusion,
             required_followup='Judge fusion relative to both best_single and matched repeat_best.'),
        dict(hypothesis='Existing fusion readout loses accessible complementary information',
             status='candidate_hypothesis' if readout_signal and not fusion_signal else 'not_established',
             evidence=joint + fusion,
             required_followup='Direct paired joint_selected versus fusion_selected_best_anchor comparison is required; different significance against a baseline is not evidence of a difference. Readout model capacity also differs.'),
        dict(hypothesis='Last-layer anchor choice limits existing fusion',
             status='anchor_signal_present' if any(map(positive, anchor)) else 'not_established',
             evidence=anchor,
             required_followup='Confirm against repeat_best capacity controls and independent LoRA seeds.'),
        dict(hypothesis='Last-token readout hides useful sequence information', status='diagnostic_curve_only',
             evidence=dict(last_layer_last=results['adapted_last']['layer_metrics']['32'],
                           last_layer_mean=results['adapted_mean']['layer_metrics']['32']),
             required_followup='Use the strict paired error analysis of matched mean/last heads; do not select new layers from outer-held curves.'),
        dict(hypothesis='Last-layer LoRA adaptation reduces accessible layer complementarity',
             status='descriptive_comparison_only',
             evidence=dict(base=results['base_last']['layer_error_complementarity'],
                           adapted=results['adapted_last']['layer_error_complementarity']),
             required_followup='Account for the changed single-layer baseline/headroom; matched LoRA-objective intervention is required for causality.'),
        dict(hypothesis='Data or translation defects explain small relative fusion gains',
             status='not_tested_by_this_stage', evidence=[],
             required_followup='Blind disagreement review, bilingual pairs and formatting controls; common errors alone do not prove relative-gain cause.'),
    ]
    status = 'screening_fusion_signal_present' if fusion_signal else (
        'screening_joint_readout_signal_present' if readout_signal else 'no_positive_signal_established')
    summary = dict(schema=SCHEMA, status=status, formal_test_opened=False,
        scope='One A-only LoRA, one readout seed, group-held-out B; exploratory and conditional.',
        paired_error_analysis_path=str(output / 'error_analysis/results.json'),
        paired_error_analysis_sha256=sha(output / 'error_analysis/results.json'),
        hypothesis_matrix=matrix, results_sha256={name: sha(output / 'readouts' / name / 'analysis/results.json') for name in results},
        directional_stop=dict(required=False,
            reason='This screening cannot certify a global cause; any proposed experiment expansion requires reviewing these bounded findings.'),
        limitations=['Failure to show a gain is not proof of no possible multilayer gain.',
                     'Probe oracle disagreement is not a deployable gain or universal feature-combination bound.',
                     'No formal holdout evaluation or automatic seed expansion occurred.',
                     'Shared split and backbone limit conclusions to this dataset/model/budget.'])
    destination = Path(queue_state_dir).resolve() if queue_state_dir else output
    dump(destination / 'diagnostic_summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'status', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--queue-state-dir', type=Path,
                        help='Separate attempt directory for immutable queue state, logs, receipts and summary')
    parser.add_argument('--minimum-free-gib', type=float,
                        help='Additional free-memory budget AFTER zero foreign processes; never permits foreign sharing')
    parser.add_argument('--owned-processes-json', type=Path,
                        help='Explicit PID/start-time/UID/command identities captured for this exact study')
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(((args.queue_state_dir or args.output) / 'queue_state.json').read_text())
    elif args.action == 'plan':
        print(json.dumps([stage.registration() for stage in build_plan(args.output, args.python)], indent=2))
    elif args.action == 'summarize':
        print(json.dumps(summarize_results(args.output, args.queue_state_dir), ensure_ascii=False, indent=2))
    else:
        try:
            bindings, protocol = preflight_bindings(args.output)
            owned = read_json(args.owned_processes_json)['processes'] if args.owned_processes_json else []
            if args.owned_processes_json:
                bindings[str(args.owned_processes_json.resolve())] = sha(args.owned_processes_json)
            stages = build_plan(args.output, args.python)
            environment = dict(CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            preflight = {arm: read_json(args.output / 'lora_study/preflight' / f'{arm}_s{protocol["seeds"][0]}.json')['peak_gpu_gb']
                         for arm in ('E03', 'E04')}
            evidence = dict(preflight_peak_gb_decimal=preflight,
                preflight_peak_gib=max(preflight.values()) * 1e9 / 1024**3,
                reserved_free_gib_above_preflight=(args.minimum_free_gib - max(preflight.values()) * 1e9 / 1024**3)
                    if args.minimum_free_gib is not None else None,
                scientific_protocol_unchanged=True,
                decision='Resource admission policy only; A/B rows, model, LoRA settings and selection are unchanged.')
            queue = SequentialSupervisor(args.output, stages, bindings, environment=environment,
                queue_state_dir=args.queue_state_dir, minimum_free_gib=args.minimum_free_gib,
                resource_evidence=evidence, owned_processes={int(value['pid']): value for value in owned})
            queue.run()
            summarize_results(args.output, args.queue_state_dir)
        except BaseException as exc:
            dump((args.queue_state_dir or args.output) / 'queue_launch_failure.json', dict(error_type=type(exc).__name__,
                reason=str(exc), automatic_retry=False, scientific_conclusion=None,
                interpretation='Execution/registration failure, not a fusion-quality conclusion.'))
            raise


if __name__ == '__main__':
    main()
