"""Parallelize registered B folds only on GPUs with no foreign processes.

An idle-looking occupied GPU is forbidden. Same-UID processes are foreign unless
explicitly whitelisted by PID, process start identity, command hash, and this
study's command. This scheduler never trains LoRA, changes settings, or kills
other processes. ``capture-owned`` and ``plan`` do not launch GPU work.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.ab_diagnostic_queue import dump, read_json, sha, StopRequested

SCHEMA = 'hanguard_ab_parallel_readout_folds_v1'
SETTINGS = ('seed', 'probe_epochs', 'head_epochs', 'max_epochs', 'probe_lr', 'head_lr',
            'weight_decay', 'width', 'dropout', 'batch_size', 'bootstrap_draws')
ALLOWED_SCRIPTS = {'ab_diagnostic_experiment.py', 'ab_frozen_readout_diagnostics.py',
                   'ab_readout_parallel_queue.py'}


def process_identity(pid):
    directory = Path('/proc') / str(int(pid))
    stat = (directory / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    raw = (directory / 'cmdline').read_bytes()
    argv = [value.decode('utf-8', errors='replace') for value in raw.split(b'\0') if value]
    if not argv:
        raise ValueError('Process command is unavailable')
    return dict(pid=int(pid), uid=directory.stat().st_uid, start_time_ticks=int(fields[19]),
        command_sha256=hashlib.sha256(raw).hexdigest(), argv=argv, cwd=str((directory / 'cwd').resolve()))


def _study_process(identity, study):
    argv = identity.get('argv', [])
    cwd = Path(identity.get('cwd', ROOT))
    scripts = [((Path(value) if Path(value).is_absolute() else cwd / value).resolve())
               for value in argv if value.endswith('.py')]
    allowed = {ROOT / 'scripts/hanguard' / name for name in ALLOWED_SCRIPTS}
    study = Path(study).resolve()
    return (identity.get('uid') == os.getuid() and any(value in allowed for value in scripts)
            and any(value.startswith('/') and (Path(value).resolve() == study or
                    Path(value).resolve().is_relative_to(study)) for value in argv))


def capture_owned(pid, study):
    identity = process_identity(pid)
    if not _study_process(identity, study):
        raise ValueError('Owned process must belong to this UID and this exact study command')
    return identity


def ensure_no_existing_readout_writer(readout):
    """Raw legacy workers do not take our fold locks; require a clean handoff."""
    readout = Path(readout).resolve()
    conflicts = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            identity = process_identity(int(directory.name))
            argv = identity['argv']
            if not any(Path(value).name == 'ab_frozen_readout_diagnostics.py' for value in argv):
                continue
            if '--output' in argv and Path(argv[argv.index('--output') + 1]).resolve() == readout:
                conflicts.append(identity['pid'])
        except (OSError, ValueError, IndexError):
            continue
    if conflicts:
        raise RuntimeError(f'Existing readout workers {conflicts} still write this output; no dispatcher launched and no processes terminated')


def gpu_inventory():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                          '--format=csv,noheader,nounits'], check=True, text=True,
                         capture_output=True, timeout=10)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                           '--format=csv,noheader,nounits'], check=True, text=True,
                          capture_output=True, timeout=10)
    result = []
    for line in gpu.stdout.splitlines():
        index, uuid, free = [value.strip() for value in line.split(',')]
        result.append(dict(index=int(index), uuid=uuid, free_gib=float(free) / 1024, pids=[]))
    by_uuid = {gpu['uuid']: gpu for gpu in result}
    for line in apps.stdout.splitlines():
        uuid, pid = [value.strip() for value in line.split(',')]
        if uuid not in by_uuid:
            raise ValueError('Compute process belongs to an unrecognized GPU UUID')
        by_uuid[uuid]['pids'].append(int(pid))
    return result


def assess_gpu(gpu, owned, study, minimum_free_gib=4, identity_reader=process_identity):
    blocked, foreign, unknown, accepted = [], [], [], []
    for pid in gpu['pids']:
        try:
            current = identity_reader(pid)
            expected = owned.get(int(pid))
            same = expected is not None and all(current.get(key) == expected.get(key)
                for key in ('pid', 'uid', 'start_time_ticks', 'command_sha256'))
            if same and _study_process(current, study):
                accepted.append(pid)
            else:
                try:
                    owner = pwd.getpwuid(current['uid']).pw_name
                except KeyError:
                    owner = f'uid:{current["uid"]}'
                record = dict(pid=pid, uid=current.get('uid'), owner=owner,
                    classification='foreign', start_time_ticks=current['start_time_ticks'],
                    executable=Path(current['argv'][0]).name, command_sha256=current['command_sha256'],
                    command_script=next((value for value in current['argv'] if value.endswith('.py')), None),
                    reason='Not an identity-verified process belonging to this study')
                foreign.append(record)
                blocked.append(record)
        except (OSError, ValueError, KeyError) as exc:
            record = dict(pid=pid, classification='unknown',
                          reason='Unknown process ownership blocks admission', error_type=type(exc).__name__)
            unknown.append(record)
            blocked.append(record)
    # The legacy sequential guard reads this field to block both conditions.
    # Each record is explicitly classified; new decisions use blocking_processes.
    return dict(**gpu, owned_pids=accepted, foreign_processes=blocked,
        blocking_processes=blocked, confirmed_foreign_processes=foreign, unknown_processes=unknown,
        legacy_foreign_field_includes_unknown=True,
        eligible=not blocked and gpu['free_gib'] >= minimum_free_gib,
        completely_idle=not gpu['pids'], policy='No foreign processes; free VRAM is an additional check only')


class IdleConfirmation:
    """Confirm empty public cards continuously; existing verified own jobs are exempt."""
    def __init__(self, seconds=60):
        if seconds < 60:
            raise ValueError('Public GPU idle confirmation must be at least60seconds')
        self.seconds, self.since = float(seconds), {}

    def apply(self, decision, now):
        uuid = decision['uuid']
        if decision['completely_idle']:
            began = self.since.setdefault(uuid, now)
            elapsed = max(0, now - began)
            confirmed = elapsed >= self.seconds
        else:
            self.since.pop(uuid, None)
            elapsed = 0
            confirmed = bool(decision['owned_pids']) and not decision['blocking_processes']
        return dict(decision, idle_observed_seconds=elapsed,
                    idle_confirmation_required_seconds=self.seconds, idle_confirmed=confirmed,
                    eligible=decision['eligible'] and confirmed)


def _arguments(protocol, output, python, phase, fold=None):
    command = [str(Path(python).absolute()), str(ROOT / 'scripts/hanguard/ab_frozen_readout_diagnostics.py'),
               phase, '--cache-manifest', protocol['cache_manifest_path'], '--output', str(output),
               '--device', 'cpu' if phase == 'report' else 'cuda']
    for key in SETTINGS:
        command.extend(['--' + key.replace('_', '-'), str(protocol['settings'][key])])
    if protocol['settings']['joint_features']:
        command.append('--joint-features')
    if fold is not None:
        command.extend(['--fold', str(fold)])
    return tuple(command)


def registered_plan(study, readout, python):
    import pandas as pd
    study, readout = Path(study).resolve(), Path(readout).resolve()
    protocol_path = readout / 'protocol.json'
    protocol = read_json(protocol_path)
    if protocol.get('schema') != 'hanguard_ab_frozen_readouts_v1' or protocol['outer_folds'] != list(range(5)):
        raise ValueError('Exactly five registered frozen-readout folds are required')
    cache_path = Path(protocol['cache_manifest_path'])
    cache = read_json(cache_path)
    if sha(cache_path) != protocol['cache_manifest_sha256'] or cache.get('complete') is not True or cache.get('cohort') != 'B':
        raise ValueError('Frozen B cache identity differs')
    if (cache['representation'] != protocol['representation'] or cache['pooling'] != protocol['pooling'] or
            cache['ab_protocol_sha256'] != protocol['ab_protocol_sha256']):
        raise ValueError('Representation/pooling/partition identity differs')
    stop_dirs = tuple(dict.fromkeys([study, readout.parent, readout] + [Path(p) for p in protocol['stop_directories']]))
    for key, digest_key in (('path', 'sha256'), ('metadata_path', 'metadata_sha256'),
                            ('ab_protocol_path', 'ab_protocol_sha256'),
                            ('training_protocol_path', 'training_protocol_sha256')):
        if sha(cache[key], stop_dirs) != cache[digest_key]:
            raise ValueError(f'Frozen cache artifact changed: {key}')
    metadata = pd.read_parquet(cache['metadata_path'])
    ordered = hashlib.sha256(('\n'.join(metadata.base_id.astype(str)) + '\n').encode()).hexdigest()
    if ordered != protocol['ordered_base_id_sha256'] or len(metadata) != protocol['rows']:
        raise ValueError('Frozen row order or row count differs')
    if tuple(cache['shape'][:2]) != (len(metadata), 32):
        raise ValueError('Frozen feature dimensions differ')
    bindings = {str(protocol_path): sha(protocol_path), str(cache_path): sha(cache_path),
        str(Path(cache['ab_protocol_path'])): cache['ab_protocol_sha256'],
        str(Path(cache['training_protocol_path'])): cache['training_protocol_sha256'],
        str(Path(__file__).resolve()): sha(__file__),
        str(ROOT / 'scripts/hanguard/ab_diagnostic_queue.py'): sha(ROOT / 'scripts/hanguard/ab_diagnostic_queue.py')}
    for name, digest in protocol['code_sha256'].items():
        path = ROOT / 'scripts/hanguard' / name
        if sha(path) != digest:
            raise ValueError(f'Registered scientific source changed: {name}')
        bindings[str(path)] = digest
    commands = {fold: _arguments(protocol, readout, python, 'run', fold) for fold in range(5)}
    return protocol, bindings, commands, _arguments(protocol, readout, python, 'report'), stop_dirs


def validate_fold(readout, fold, protocol_sha):
    artifacts = {}
    for stage in ('probes', 'heads'):
        directory = Path(readout) / f'fold_{fold}' / stage
        complete = directory / 'complete.json'
        value = read_json(complete)
        provenance = value['provenance']
        if (value.get('complete') is not True or provenance['outer_fold'] != fold or
                provenance['stage'] != stage or provenance['protocol_sha256'] != protocol_sha):
            raise ValueError('Fold completion does not match the immutable readout protocol')
        for name, digest in value['artifacts'].items():
            path = directory / name
            if sha(path) != digest:
                raise ValueError('Completed fold artifact changed')
            artifacts[str(path)] = digest
        artifacts[str(complete)] = sha(complete)
    return artifacts


class ParallelQueue:
    def __init__(self, study, output, readout, commands, bindings, *, owned=None,
                 candidates=tuple(range(8)), max_workers=3, watch_seconds=50,
                 minimum_free_gib=4, inventory=gpu_inventory, validator=None, poll=.5,
                 idle_confirmation_seconds=60, clock=time.monotonic, identity_reader=None):
        self.study, self.output, self.readout = map(lambda p: Path(p).resolve(), (study, output, readout))
        self.commands, self.bindings = dict(commands), dict(bindings)
        self.owned = dict(owned or {})
        self.candidates = tuple(candidates)
        if not 1 <= max_workers <= 3 or len(set(candidates)) != len(candidates):
            raise ValueError('At most three workers and distinct candidate GPUs are required')
        self.max_workers = max_workers
        self.idle_confirmation = IdleConfirmation(idle_confirmation_seconds)
        self.watch_seconds = max(watch_seconds, idle_confirmation_seconds + 5)
        self.clock = clock
        self.identity_reader = identity_reader or process_identity
        self.minimum_free_gib, self.inventory, self.validator = minimum_free_gib, inventory, validator
        self.poll, self.children, self.completed = poll, {}, {}
        self.stop_evidence = {}
        self.stop_dirs = (self.study, self.readout, self.readout.parent, self.output, self.study / 'lora_study')
        self.protocol = dict(schema=SCHEMA, commands={str(k): list(v) for k, v in self.commands.items()},
            bindings=self.bindings, candidates=list(candidates), max_workers=max_workers,
            minimum_free_gib=minimum_free_gib, owned_processes=list(self.owned.values()),
            empty_public_gpu_confirmation_seconds=idle_confirmation_seconds,
            policy='Never use GPUs with foreign processes, regardless of memory or utilization',
            scientific_settings_changed=False, automatic_retries=False)

    def check(self):
        for folder in self.stop_dirs:
            stop = folder / 'STOP'
            if stop.exists():
                raise StopRequested(stop.read_text().strip() or 'STOP requested')
        for name, digest in self.bindings.items():
            if sha(name) != digest:
                raise ValueError(f'Frozen dispatcher binding changed: {name}')

    def state(self, state, **details):
        value = dict(schema=SCHEMA, state=state, supervisor_pid=os.getpid(), updated_unix=time.time(),
            active={str(fold): dict(pid=worker['process'].pid, gpu=worker['gpu']) for fold, worker in self.children.items()},
            completed_folds=sorted(self.completed), scientific_settings_changed=False, **details)
        dump(self.output / 'state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return value

    def terminate_ours(self):
        for worker in self.children.values():
            process = worker['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while any(worker['process'].poll() is None for worker in self.children.values()) and time.monotonic() < deadline:
            time.sleep(.1)
        for worker in self.children.values():
            process = worker['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()

    def launch(self, fold, gpu):
        folder = self.readout / f'fold_{fold}'
        folder.mkdir(parents=True, exist_ok=True)
        lock = (folder / '.parallel_worker.lock').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log = (self.output / f'fold_{fold}.log').open('a')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            process = subprocess.Popen(self.commands[fold], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.children[fold] = dict(process=process, gpu=gpu, log=log, lock=lock)
            deadline = time.monotonic() + 1
            while True:
                try:
                    self.owned[process.pid] = process_identity(process.pid)
                    break
                except (OSError, ValueError):
                    if process.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        except BaseException:
            lock.close()
            raise

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'supervisor.pid').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol_path = self.output / 'protocol.json'
            if protocol_path.exists() and read_json(protocol_path) != self.protocol:
                raise ValueError('Dispatcher registration changed')
            if not protocol_path.exists():
                dump(protocol_path, self.protocol)
            previous = self.output / 'state.json'
            if previous.exists() and read_json(previous).get('state') in ('failed', 'stopped'):
                raise RuntimeError('Failed/stopped dispatcher cannot automatically retry')
            pending = list(sorted(self.commands))
            for fold in pending[:]:
                receipt = self.output / f'fold_{fold}.receipt.json'
                if (not receipt.exists() and self.validator is not None and
                        all((self.readout / f'fold_{fold}' / phase / 'complete.json').is_file()
                            for phase in ('probes', 'heads'))):
                    artifacts = self.validator(fold)
                    value = dict(complete=True, fold=fold, gpu=None, command=list(self.commands[fold]),
                        protocol_sha256=sha(protocol_path), artifacts=artifacts,
                        adopted_verified_completed_fold=True, command_not_executed_by_this_attempt=True)
                    dump(receipt, value)
                if receipt.exists():
                    value = read_json(receipt)
                    artifacts = self.validator(fold) if self.validator else value['artifacts']
                    if value['command'] != list(self.commands[fold]) or value['artifacts'] != artifacts:
                        raise ValueError('Completed fold receipt changed')
                    self.completed[fold] = value
                    pending.remove(fold)
            last_progress, last_state = time.monotonic(), 0
            old_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(StopRequested('SIGTERM received')))
            try:
                while pending or self.children:
                    self.check()
                    for fold, worker in list(self.children.items()):
                        code = worker['process'].poll()
                        if code is None:
                            continue
                        if code != 0:
                            raise RuntimeError(f'Fold {fold} exited {code}; no retries')
                        artifacts = self.validator(fold) if self.validator else {}
                        value = dict(complete=True, fold=fold, gpu=worker['gpu'], command=list(self.commands[fold]),
                                     protocol_sha256=sha(protocol_path), artifacts=artifacts)
                        dump(self.output / f'fold_{fold}.receipt.json', value)
                        self.completed[fold] = value
                        worker['log'].close()
                        worker['lock'].close()
                        self.owned.pop(worker['process'].pid, None)
                        del self.children[fold]
                        last_progress = time.monotonic()
                    checked_at = self.clock()
                    observations = [self.idle_confirmation.apply(
                        assess_gpu(gpu, self.owned, self.study, self.minimum_free_gib, self.identity_reader), checked_at)
                        for gpu in self.inventory() if gpu['index'] in self.candidates]
                    used = {worker['gpu'] for worker in self.children.values()}
                    triggering = [gpu for gpu in observations if gpu['index'] in used and gpu['blocking_processes']]
                    if triggering:
                        foreign_pids = [process['pid'] for gpu in triggering for process in gpu['confirmed_foreign_processes']]
                        unknown_pids = [process['pid'] for gpu in triggering for process in gpu['unknown_processes']]
                        self.stop_evidence = dict(trigger_observed_unix=time.time(),
                            trigger_dispatcher_protocol_sha256=sha(protocol_path),
                            trigger_gpu_observations=observations, triggering_gpu_observations=triggering,
                            trigger_confirmed_foreign_pids=foreign_pids, trigger_unknown_ownership_pids=unknown_pids,
                            trigger_active_workers={str(fold): dict(pid=value['process'].pid, gpu=value['gpu'])
                                                    for fold, value in self.children.items()},
                            termination_started=False)
                        # Persist the exact triggering sample before any signal:
                        # a later sample after cleanup cannot reconstruct it.
                        dump(self.output / 'ownership_stop_trigger.json', self.stop_evidence)
                        self.state('stopping_for_ownership_conflict', **self.stop_evidence)
                        kind = 'confirmed foreign' if foreign_pids else 'unknown ownership'
                        raise StopRequested(f'{kind} process observed on an active GPU; stopping only our fold workers')
                    available = sorted((gpu for gpu in observations if gpu['eligible'] and gpu['index'] not in used),
                                       key=lambda gpu: (not gpu['completely_idle'], gpu['index']))
                    while pending and available and len(self.children) < self.max_workers:
                        self.launch(pending.pop(0), available.pop(0)['index'])
                        last_progress = time.monotonic()
                    if time.monotonic() - last_state >= 5:
                        self.state('running' if self.children else 'waiting_for_no_foreign_process_gpu',
                                   gpu_observations=observations, pending_folds=pending)
                        last_state = time.monotonic()
                    if pending and not self.children and time.monotonic() - last_progress >= self.watch_seconds:
                        return self.state('waiting_no_eligible_gpu', pending_folds=pending,
                                          gpu_observations=observations)
                    time.sleep(self.poll)
                return self.state('folds_completed')
            except BaseException as exc:
                self.terminate_ours()
                evidence = dict(self.stop_evidence)
                if evidence:
                    evidence.update(termination_started=True, termination_completed=True)
                self.state('stopped' if isinstance(exc, StopRequested) else 'failed',
                           error_type=type(exc).__name__, reason=str(exc), no_foreign_processes_terminated=True,
                           **evidence)
                raise
            finally:
                signal.signal(signal.SIGTERM, old_sigterm)
                for worker in self.children.values():
                    worker['log'].close()
                    worker['lock'].close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'capture-owned'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--readout-output', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=ROOT / '.venv-hanguard/bin/python')
    parser.add_argument('--owned-processes-json', type=Path)
    parser.add_argument('--pid', type=int, action='append', default=[])
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--watch-seconds', type=float, default=50)
    parser.add_argument('--idle-confirmation-seconds', type=float, default=60,
                        help='Continuous idle confirmation for empty public GPUs; minimum60seconds')
    args = parser.parse_args(argv)
    if args.action == 'capture-owned':
        dump(args.output, dict(processes=[capture_owned(pid, args.study) for pid in args.pid]))
        return
    readout = args.readout_output or args.study / 'readouts/base_last'
    protocol, bindings, commands, report, _ = registered_plan(args.study, readout, args.python)
    if args.action == 'plan':
        print(json.dumps(dict(commands=commands, report_command=report, ownership_policy='No foreign processes'), indent=2))
        return
    ensure_no_existing_readout_writer(readout)
    owned = read_json(args.owned_processes_json)['processes'] if args.owned_processes_json else []
    if args.owned_processes_json:
        bindings[str(args.owned_processes_json.resolve())] = sha(args.owned_processes_json)
    queue = ParallelQueue(args.study, args.output, readout, commands, bindings,
        owned={int(value['pid']): value for value in owned}, candidates=args.gpus,
        watch_seconds=args.watch_seconds, idle_confirmation_seconds=args.idle_confirmation_seconds,
        validator=lambda fold: validate_fold(readout, fold, sha(readout / 'protocol.json')))
    state = queue.run()
    if state['state'] == 'folds_completed':
        queue.check()
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (args.output / 'report.log').open('a') as log:
            process = subprocess.Popen(report, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    queue.check()
                    time.sleep(.5)
                if process.returncode:
                    raise RuntimeError(f'CPU report exited {process.returncode}')
            except BaseException as exc:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                queue.state('stopped' if isinstance(exc, StopRequested) else 'failed', reason=str(exc))
                raise
        queue.state('completed', analysis_results_sha256=sha(readout / 'analysis/results.json'))


if __name__ == '__main__':
    main()
