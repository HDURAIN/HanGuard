"""Identity-bound parallel frozen features without changing original batch groups.

CPU preparation fixes tokenization and all original global microbatch shapes.
Workers steal disjoint groups in chunks; only complete, hash-bound chunk receipts
can enter the canonical cache. Incomplete attempts are retained and never read.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard import ab_readout_parallel_queue as inventory

SCHEMA = 'hanguard_parallel_frozen_features_v1'


def dump(path, value):
    """Unique temporary paths avoid colliding writers' atomic JSON updates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def worker_directory(study):
    return Path(study) / 'parallel_execution' / 'workers'


def check_stop(study, worker_id=None):
    base.check_stop(study)
    if worker_id is not None:
        path = worker_directory(study) / f'{worker_id}.STOP'
        if path.exists():
            raise RuntimeError(f'Parallel worker stopped: {path.read_text().strip()}')


def safe_worker_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', str(value)):
        raise ValueError('Invalid exact worker identity')
    return str(value)


def bind_worker(study, worker_id, split=None):
    """Bind an exact scheduler child before any CUDA/model/data operation.

    The same helper supports the unchanged single GPU head-fit command. GPU
    workers never spawn CUDA subprocesses, so only their own PID is accepted.
    """
    study = Path(study).resolve()
    worker_id = safe_worker_id(worker_id)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in tuple(str(i) for i in range(8)):
        raise ValueError('Exactly one scheduler-registered physical GPU is required')
    protocol = base.load_protocol(study, full_hash=False)
    path = worker_directory(study) / f'{worker_id}.json'
    registration_path = path.with_suffix('.registration.json')
    deadline = time.monotonic() + 10
    while not path.exists() or not registration_path.exists():
        check_stop(study, worker_id)
        if time.monotonic() >= deadline:
            raise ValueError('No exact scheduler worker ledger; direct GPU launch forbidden')
        time.sleep(.05)
    ledger = read(path)
    if read(registration_path).get('ledger_sha256') != base.sha(path):
        raise ValueError('Exact worker ledger registration changed')
    resource_path = study / 'parallel_resource_protocol.json'
    resource_sha = base.sha(resource_path)
    if (read(study / 'parallel_resource_registration.json').get('protocol_sha256') != resource_sha
            or ledger.get('resource_protocol_sha256') != resource_sha):
        raise ValueError('Exact worker resource protocol differs')
    expected = ledger.get('process_identity', {})
    actual = inventory.process_identity(os.getpid())
    birth = expected.get('start_ticks', expected.get('start_time_ticks'))
    if (ledger.get('worker_id') != worker_id or ledger.get('physical_gpu') != int(visible)
            or ledger.get('CUDA_VISIBLE_DEVICES') != visible
            or ledger.get('root_protocol_sha256') != base.sha(study / 'protocol.json')
            or expected.get('uid') != os.getuid() or birth != actual['start_time_ticks']
            or any(expected.get(k) != actual.get(k) for k in ('pid', 'uid', 'command_sha256', 'argv', 'cwd'))
            or (split is not None and ledger.get('split') != split)):
        raise ValueError('Worker does not match exact registered identity/card/split')
    cwd = Path(actual['cwd'])
    scripts = [(Path(value) if Path(value).is_absolute() else cwd / value).resolve()
               for value in actual['argv'] if value.endswith('.py')]
    if not scripts or any(protocol['code_sha256'].get(str(p)) != base.sha(p) for p in scripts):
        raise ValueError('Worker entrypoint is not an immutable registered implementation')
    # command and script_sha256 are supplementary ledger bindings. Identity argv
    # above is authoritative, including all arguments and the absolute study.
    if ledger.get('command') != actual['argv']:
        raise ValueError('Worker command ledger differs')
    if ledger.get('script_sha256') != base.sha(scripts[0]):
        raise ValueError('Worker entrypoint SHA differs')
    if str(study) not in actual['argv']:
        raise ValueError('Worker argv must name this exact absolute study')
    card = next((item for item in inventory.gpu_inventory() if item['index'] == int(visible)), None)
    if card is None or any(pid != os.getpid() for pid in card['pids']):
        raise ValueError('Registered card contains foreign/unknown compute processes')
    return ledger


def cuda_startup(study, worker_id, split=None):
    ledger = bind_worker(study, worker_id, split)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one bound CUDA device must be visible')
    readiness = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    if not torch.isfinite(readiness).all():
        raise RuntimeError('Actual CUDA allocation readiness failed')
    result = subprocess.run(['nvidia-smi', '-i', str(ledger['physical_gpu']),
        '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
        check=True, text=True, capture_output=True, timeout=10)
    record = dict(passed=True, actual_cuda_allocation_verified=True,
        pid=os.getpid(), worker_id=worker_id, physical_gpu=ledger['physical_gpu'],
        inventory=result.stdout.strip(), device_name=torch.cuda.get_device_name(0),
        runtime=base.runtime_identity(), model_or_data_used_before_full_hash=False,
        protocol_sha256=base.sha(Path(study) / 'protocol.json'))
    dump(worker_directory(study) / f'{worker_id}.cuda.json', record)
    return record


def split_directory(study, split):
    if split not in base.SPLITS:
        raise ValueError(split)
    return Path(study) / 'parallel_features' / split


def barrier(study, split):
    check_stop(study)
    if split == 'test':
        base.validate_selection_lock(study)


@contextmanager
def claim_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'claims.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_split(study, split, *, tokenizer=None):
    """Called by the CPU coordinator once, including before test prompt access."""
    study = Path(study).resolve()
    barrier(study, split)
    protocol = base.load_protocol(study, full_hash=True)
    folder = split_directory(study, split)
    manifest = folder / 'prepared_manifest.json'
    if manifest.exists():
        return load_prepared(study, split, protocol=protocol)[0]
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Incomplete preparation is retained; fresh study required')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    # load_frame is called only by this coordinator; it owns sticky root state.
    frame = base.load_frame(study, protocol, split)
    data = base.make_data(frame, tokenizer, protocol)
    lengths = data['lengths']
    order = np.argsort(lengths, kind='stable')
    extraction = protocol['extraction']
    groups = base.legacy.microbatches(order, lengths, extraction['token_budget'],
                                     extraction['max_micro'], extraction['pad_multiple'])
    chunk_size = int(protocol.get('parallel_features', {}).get('group_chunk_size', 32))
    if chunk_size < 1:
        raise ValueError('Positive original-group chunk size required')
    joined = np.concatenate(groups)
    if not np.array_equal(joined, order) or sorted(joined.tolist()) != list(range(len(lengths))):
        raise ValueError('Original batching must cover every sorted sample exactly once')
    chunks = [dict(chunk_id=n // chunk_size, group_start=n,
                   group_stop=min(n + chunk_size, len(groups)),
                   rows=sum(len(g) for g in groups[n:n + chunk_size]))
              for n in range(0, len(groups), chunk_size)]
    plan = dict(schema=SCHEMA, split=split, protocol_sha256=base.sha(study / 'protocol.json'),
        input_sha256=protocol['inputs'][str(Path(protocol['data_dir']) / f'{split}.parquet')],
        ordered_base_id_sha256=base.ordered_base_id_sha256(data['meta'].base_id.tolist()),
        rows=len(lengths), shape=[len(lengths), protocol['num_layers'], protocol['hidden_size']],
        grouping='original_global_stable_length_order_without_rebatching',
        groups=[g.tolist() for g in groups], group_chunk_size=chunk_size, chunks=chunks)
    folder.mkdir(parents=True)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    flat = np.asarray([token for sequence in data['encoded'] for token in sequence], dtype=np.int64)
    if flat.size != int(offsets[-1]) or flat.min() < 0 or flat.max() > np.iinfo(np.int32).max:
        raise ValueError('Token offsets/IDs are invalid')
    np.save(folder / 'token_ids.npy', flat.astype(np.int32), allow_pickle=False)
    np.save(folder / 'offsets.npy', offsets, allow_pickle=False)
    np.save(folder / 'lengths.npy', lengths, allow_pickle=False)
    data['meta'].to_parquet(folder / 'metadata.parquet', index=False)
    dump(folder / 'batch_plan.json', plan)
    artifacts = {name: base.sha(folder / name, study) for name in
        ('token_ids.npy', 'offsets.npy', 'lengths.npy', 'metadata.parquet', 'batch_plan.json')}
    prepared = dict(schema=SCHEMA, complete=True, split=split,
        protocol_sha256=plan['protocol_sha256'], input_sha256=plan['input_sha256'],
        artifacts=artifacts, plan_sha256=artifacts['batch_plan.json'],
        rows=plan['rows'], original_groups=len(groups), chunks=len(chunks),
        test_selection_lock_sha256=base.sha(study / 'selection_lock.json') if split == 'test' else None)
    dump(manifest, prepared)
    return prepared


class EncodedRows:
    """No pickle and no duplicate token lists in every GPU worker."""
    def __init__(self, tokens, offsets):
        self.tokens, self.offsets = tokens, offsets
    def __getitem__(self, row):
        return self.tokens[self.offsets[row]:self.offsets[row + 1]].tolist()
    def __len__(self):
        return len(self.offsets) - 1


def load_prepared(study, split, *, protocol=None):
    barrier(study, split)
    protocol = protocol or base.load_protocol(study)
    folder = split_directory(study, split)
    prepared = read(folder / 'prepared_manifest.json')
    expected_input = protocol['inputs'][str(Path(protocol['data_dir']) / f'{split}.parquet')]
    if (prepared.get('schema') != SCHEMA or prepared.get('complete') is not True
            or prepared.get('split') != split or prepared.get('protocol_sha256') != base.sha(Path(study) / 'protocol.json')
            or prepared.get('input_sha256') != expected_input):
        raise ValueError('Prepared token/cache identity differs')
    if split == 'test' and prepared.get('test_selection_lock_sha256') != base.sha(Path(study) / 'selection_lock.json'):
        raise ValueError('Test selections changed after token preparation')
    expected_files = {'token_ids.npy', 'offsets.npy', 'lengths.npy', 'metadata.parquet', 'batch_plan.json'}
    if set(prepared.get('artifacts', {})) != expected_files:
        raise ValueError('Prepared artifact list differs')
    for name, digest in prepared['artifacts'].items():
        if base.sha(folder / name, study) != digest:
            raise ValueError(f'Prepared {name} bytes changed')
    plan = read(folder / 'batch_plan.json')
    tokens = np.load(folder / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    offsets = np.load(folder / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    lengths = np.load(folder / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    meta = pd.read_parquet(folder / 'metadata.parquet')
    shape = [protocol['rows'][split], protocol['num_layers'], protocol['hidden_size']]
    if (plan.get('shape') != shape or len(meta) != shape[0] or lengths.shape != (shape[0],)
            or offsets.shape != (shape[0] + 1,) or offsets[0] != 0 or offsets[-1] != len(tokens)
            or not np.array_equal(np.diff(offsets), lengths)
            or (lengths < 1).any() or (lengths > protocol['extraction']['max_tokens']).any()
            or plan.get('ordered_base_id_sha256') != base.ordered_base_id_sha256(meta.base_id.tolist())
            or not meta.base_id.is_unique):
        raise ValueError('Prepared row identity/token dimensions differ')
    groups = [np.asarray(g, dtype=np.int64) for g in plan['groups']]
    expected_groups = base.legacy.microbatches(np.argsort(lengths, kind='stable'), lengths,
        protocol['extraction']['token_budget'], protocol['extraction']['max_micro'],
        protocol['extraction']['pad_multiple'])
    if len(groups) != len(expected_groups) or any(not np.array_equal(a, b) for a, b in zip(groups, expected_groups)):
        raise ValueError('Prepared groups differ from original global microbatch shapes')
    chunk_size = int(protocol.get('parallel_features', {}).get('group_chunk_size', 32))
    expected_chunks = [dict(chunk_id=n // chunk_size, group_start=n,
        group_stop=min(n + chunk_size, len(groups)), rows=sum(len(g) for g in groups[n:n + chunk_size]))
        for n in range(0, len(groups), chunk_size)]
    if plan.get('chunks') != expected_chunks or plan.get('group_chunk_size') != chunk_size:
        raise ValueError('Chunk plan changes or omits original groups')
    return prepared, plan, dict(encoded=EncodedRows(tokens, offsets), lengths=lengths, meta=meta)


def chunk_directory(study, split, chunk_id):
    return split_directory(study, split) / 'chunks' / f'{int(chunk_id):06d}'


def _quick_receipt(study, split, prepared, chunk):
    path = chunk_directory(study, split, chunk['chunk_id']) / 'receipt.json'
    if not path.exists():
        return None
    receipt = read(path)
    if (receipt.get('complete') is not True or receipt.get('schema') != SCHEMA
            or receipt.get('plan_sha256') != prepared['plan_sha256']
            or receipt.get('protocol_sha256') != prepared['protocol_sha256']
            or receipt.get('split') != split or receipt.get('chunk') != chunk):
        raise ValueError('Published complete chunk receipt differs')
    return receipt


def _same_live_process(identity):
    """PID birth/UID only, avoiding inaccessible other users' cwd/cmdline."""
    path = Path('/proc') / str(identity['pid'])
    try:
        stat = (path / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return (fields[0] != 'Z' and path.stat().st_uid == identity['uid']
                and int(fields[19]) == identity['start_time_ticks'])
    except FileNotFoundError:
        return False
    except (OSError, ValueError, IndexError):
        return True  # Unknown process identity is never treated as absent.


def split_progress(study, split):
    barrier(study, split)
    folder = split_directory(study, split)
    prepared, plan = read(folder / 'prepared_manifest.json'), read(folder / 'batch_plan.json')
    complete = claimed = pending = rows = 0
    for chunk in plan['chunks']:
        receipt = _quick_receipt(study, split, prepared, chunk)
        if receipt is not None:
            complete += 1
            rows += chunk['rows']
            continue
        claim_path = chunk_directory(study, split, chunk['chunk_id']) / 'claim.json'
        claim = read(claim_path) if claim_path.exists() else None
        if claim and _same_live_process(claim['identity']):
            claimed += 1
        else:
            pending += 1
    return dict(split=split, total_chunks=len(plan['chunks']), complete_chunks=complete,
        claimed_chunks=claimed, pending_chunks=pending, completed_rows=rows,
        total_rows=plan['rows'], complete=complete == len(plan['chunks']))


def has_work(study, split):
    return split_progress(study, split)['pending_chunks'] > 0


def split_complete(study, split):
    return split_progress(study, split)['complete']


def claim_chunk(study, split, worker_id, prepared, plan):
    check_stop(study, worker_id)
    identity = inventory.process_identity(os.getpid())
    with claim_lock(split_directory(study, split)):
        for chunk in plan['chunks']:
            if _quick_receipt(study, split, prepared, chunk) is not None:
                continue
            directory = chunk_directory(study, split, chunk['chunk_id'])
            path = directory / 'claim.json'
            previous = read(path) if path.exists() else None
            if previous and _same_live_process(previous['identity']):
                continue
            attempt = 1 if previous is None else int(previous['attempt']) + 1
            if previous is not None:
                dump(directory / f'abandoned_attempt_{previous["attempt"]:03d}.json',
                     dict(previous, abandoned=True, partial_rows_reused=False,
                          reason='Previous exact process birth is absent', updated_unix=time.time()))
            target = directory / f'attempt_{attempt:03d}'
            target.mkdir(parents=True)
            claim = dict(schema=SCHEMA, worker_id=worker_id, identity=identity,
                split=split, chunk=chunk, attempt=attempt, directory=str(target.resolve()),
                plan_sha256=prepared['plan_sha256'], created_unix=time.time())
            dump(path, claim)
            return claim
    return None


def abandon_worker(study, split, worker_id, reason='GPU ownership conflict'):
    """Retire only uncommitted extraction attempts after the exact child exits.

    This never sends signals, discards complete chunks, or adopts partial rows.
    A reused PID is a different process birth; no operation touches that process.
    """
    study = Path(study).resolve()
    worker_id = safe_worker_id(worker_id)
    ledger_path = worker_directory(study) / f'{worker_id}.json'
    ledger = read(ledger_path)
    if read(ledger_path.with_suffix('.registration.json')).get('ledger_sha256') != base.sha(ledger_path):
        raise ValueError('Worker ledger changed during retirement')
    identity = ledger['process_identity']
    identity = dict(identity, start_time_ticks=identity.get('start_ticks', identity.get('start_time_ticks')))
    if (ledger.get('worker_id') != worker_id or ledger.get('split') != split
            or identity.get('uid') != os.getuid()
            or ledger.get('root_protocol_sha256') != base.sha(study / 'protocol.json')):
        raise ValueError('Cannot retire an unrelated worker identity')
    if _same_live_process(identity):
        raise ValueError('Exact worker is still live or its identity is unknown')
    folder = split_directory(study, split)
    prepared, plan = read(folder / 'prepared_manifest.json'), read(folder / 'batch_plan.json')
    retired = []
    with claim_lock(folder):
        for chunk in plan['chunks']:
            directory = chunk_directory(study, split, chunk['chunk_id'])
            path = directory / 'claim.json'
            receipt = _quick_receipt(study, split, prepared, chunk)
            if receipt is not None:
                if receipt.get('worker_id') == worker_id:
                    expected_root = directory / f'attempt_{receipt["attempt"]:03d}'
                    for field, hash_field in (('path', 'sha256'), ('row_indices_path', 'row_indices_sha256'),
                            ('replay_audit_path', 'replay_audit_sha256')):
                        if base.sha(_within(receipt[field], expected_root), study) != receipt[hash_field]:
                            raise ValueError('Committed chunk changed before worker retirement')
                    calibration = _within(receipt['calibration_path'], worker_directory(study))
                    if base.sha(calibration) != receipt['calibration_sha256']:
                        raise ValueError('Committed calibration changed before worker retirement')
                    rows = np.load(receipt['row_indices_path'], allow_pickle=False)
                    expected_rows = np.concatenate([np.asarray(g, dtype=np.int64) for g in
                        plan['groups'][chunk['group_start']:chunk['group_stop']]])
                    values = np.load(receipt['path'], mmap_mode='r', allow_pickle=False)
                    if (not np.array_equal(rows, expected_rows) or values.dtype != np.float16
                            or list(values.shape) != [len(rows), *plan['shape'][1:]]
                            or not np.isfinite(values).all()):
                        raise ValueError('Committed rows/features invalid before worker retirement')
                continue
            if not path.exists():
                continue
            claim = read(path)
            if claim.get('worker_id') != worker_id or claim.get('abandoned') is True:
                continue
            if any(claim['identity'].get(k) != identity.get(k) for k in ('pid', 'uid', 'start_time_ticks')):
                raise ValueError('Uncommitted claim does not belong to exact stopped worker')
            original = _within(claim['directory'], directory)
            retired_path = directory / 'failed_attempts' / f'attempt_{claim["attempt"]:03d}_{worker_id}'
            retired_path.parent.mkdir(exist_ok=True)
            if original.exists():
                original.rename(retired_path)
            retirement = dict(claim, abandoned=True, directory=str(retired_path.resolve()),
                original_directory=str(original), partial_rows_reused=False,
                reason=reason, updated_unix=time.time())
            dump(directory / f'abandoned_attempt_{claim["attempt"]:03d}.json', retirement)
            dump(path, retirement)
            retired.append(chunk['chunk_id'])
    return dict(worker_id=worker_id, split=split, retired_chunk_ids=retired,
                completed_chunks_retained=True, partial_rows_reused=False)


def global_calibration(capture, data, tokenizer, extraction, groups, *, stop=lambda: None):
    """Identical global groups on every GPU, with exact FP16 byte signatures."""
    records = []
    for number in sorted(set((0, len(groups) // 2, len(groups) - 1))):
        stop()
        indices = groups[number]
        batch = base.legacy.make_batch(data, indices, tokenizer, extraction)
        stop()
        first = capture(batch)
        stop()
        second = capture(batch)
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise RuntimeError('Identical global-batch replay is not exact and finite')
        records.append(dict(group=number, rows=len(indices),
            sample_ids=data['meta'].base_id.iloc[indices].tolist(),
            shape=list(first.shape), dtype='float16',
            feature_sha256=hashlib.sha256(first.numpy().tobytes(order='C')).hexdigest(),
            forward_repeated_equal=True))
    return dict(passed=True, records=records,
        rule='same original global groups; exact FP16 byte signature across every contributing GPU',
        batch_shape_equivalence_claimed=False)


class LocalRows:
    def __init__(self, values, indices):
        self.values = values
        self.local = {int(row): n for n, row in enumerate(indices)}
    def __getitem__(self, indices):
        return self.values[[self.local[int(row)] for row in indices]]


def extract_chunk(study, split, worker_id, claim, prepared, plan, data, tokenizer,
                  capture, calibration_path, protocol):
    target = Path(claim['directory'])
    groups = [np.asarray(g, dtype=np.int64) for g in plan['groups']
              [claim['chunk']['group_start']:claim['chunk']['group_stop']]]
    indices = np.concatenate(groups)
    shape = (len(indices), protocol['num_layers'], protocol['hidden_size'])
    temporary = target / 'features.partial.npy'
    if any(target.iterdir()):
        raise ValueError('Incomplete chunk attempts cannot be reused')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=shape)
    began, offset = time.monotonic(), 0
    for number, group in enumerate(groups):
        check_stop(study, worker_id)
        actual = capture(base.legacy.make_batch(data, group, tokenizer, protocol['extraction'])).numpy()
        if actual.shape != (len(group), shape[1], shape[2]) or actual.dtype != np.float16 or not np.isfinite(actual).all():
            raise FloatingPointError('Chunk captured feature dimensions/values invalid')
        values[offset:offset + len(group)] = actual
        offset += len(group)
        dump(worker_directory(study) / f'{worker_id}.progress.json', dict(
            state='extracting', split=split, chunk_id=claim['chunk']['chunk_id'],
            completed_groups=number + 1, total_groups=len(groups), completed_chunk_rows=offset,
            pid=os.getpid(), worker_id=worker_id, updated_unix=time.time()))
    values.flush()
    def stopped_capture(batch):
        check_stop(study, worker_id)
        return capture(batch)
    check_stop(study, worker_id)
    audit = base.exact_replay(stopped_capture, data, tokenizer, protocol['extraction'],
                             groups, LocalRows(values, indices))
    audit_path = target / 'replay_audit.json'
    dump(audit_path, audit)
    final = target / 'features.npy'
    del values
    temporary.replace(final)
    np.save(target / 'row_indices.npy', indices, allow_pickle=False)
    receipt = dict(schema=SCHEMA, complete=True, split=split, chunk=claim['chunk'],
        worker_id=worker_id, attempt=claim['attempt'], identity=claim['identity'],
        protocol_sha256=prepared['protocol_sha256'], plan_sha256=prepared['plan_sha256'],
        shape=list(shape), path=str(final.resolve()), sha256=base.sha(final, study),
        row_indices_path=str((target / 'row_indices.npy').resolve()),
        row_indices_sha256=base.sha(target / 'row_indices.npy'),
        replay_audit_path=str(audit_path.resolve()), replay_audit_sha256=base.sha(audit_path),
        calibration_path=str(Path(calibration_path).resolve()), calibration_sha256=base.sha(calibration_path),
        elapsed_seconds=time.monotonic() - began, completed_unix=time.time())
    with claim_lock(split_directory(study, split)):
        current = read(chunk_directory(study, split, claim['chunk']['chunk_id']) / 'claim.json')
        if current != claim:
            raise ValueError('Chunk claim identity changed before complete publication')
        check_stop(study, worker_id)
        dump(chunk_directory(study, split, claim['chunk']['chunk_id']) / 'receipt.json', receipt)
    return receipt


def run_worker(study, worker_id, split):
    study = Path(study).resolve()
    worker_id = safe_worker_id(worker_id)
    runtime = cuda_startup(study, worker_id, split)
    # All large original inputs are fully hashed after CUDA admission and before
    # either token payloads or model weights are loaded.
    protocol = base.load_protocol(study, full_hash=True)
    check_stop(study, worker_id)
    prepared, plan, data = load_prepared(study, split, protocol=protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, backbone, tokenizer = base.load_frozen_backbone(protocol)
    capture = base.LastBlockCapture(backbone)
    try:
        groups = [np.asarray(group, dtype=np.int64) for group in plan['groups']]
        calibration = global_calibration(capture, data, tokenizer, protocol['extraction'], groups,
            stop=lambda: check_stop(study, worker_id))
        calibration.update(worker_id=worker_id, runtime_gpu=runtime,
                           protocol_sha256=prepared['protocol_sha256'], plan_sha256=prepared['plan_sha256'])
        calibration_path = worker_directory(study) / f'{worker_id}.calibration.json'
        dump(calibration_path, calibration)
        while True:
            check_stop(study, worker_id)
            claim = claim_chunk(study, split, worker_id, prepared, plan)
            if claim is not None:
                receipt = extract_chunk(study, split, worker_id, claim, prepared, plan, data,
                    tokenizer, capture, calibration_path, protocol)
                print(json.dumps(dict(state='chunk_completed', split=split,
                    worker_id=worker_id, chunk_id=claim['chunk']['chunk_id'], rows=receipt['shape'][0])), flush=True)
                continue
            progress = split_progress(study, split)
            if progress['complete']:
                break
            time.sleep(.2)
        dump(worker_directory(study) / f'{worker_id}.progress.json', dict(
            state='completed', split=split, worker_id=worker_id, pid=os.getpid(), updated_unix=time.time()))
    finally:
        capture.close()
        del capture, model, backbone
        gc.collect()
        torch.cuda.empty_cache()


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Chunk artifact is outside the exact study')
    return path


def merge_split(study, split):
    """CPU merge verifies complete chunks, exact row coverage and GPU agreement."""
    study = Path(study).resolve()
    barrier(study, split)
    protocol = base.load_protocol(study)
    if base.cache_complete(study, protocol, split):
        return read(study / 'caches' / f'{split}.manifest.json')
    prepared, plan, data = load_prepared(study, split, protocol=protocol)
    if not split_complete(study, split):
        raise ValueError('All original chunks must be complete before merge')
    folder = study / 'caches'
    folder.mkdir(exist_ok=True)
    temporary = folder / f'{split}.parallel_merge.partial.npy'
    if temporary.exists():
        raise ValueError('Incomplete canonical merge retained; fresh study required')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=tuple(plan['shape']))
    seen = np.zeros(plan['rows'], dtype=np.bool_)
    chunks, calibrations, signature = [], {}, None
    for chunk in plan['chunks']:
        check_stop(study)
        receipt = _quick_receipt(study, split, prepared, chunk)
        if receipt is None:
            raise ValueError('Complete receipt vanished during merge')
        expected_root = chunk_directory(study, split, chunk['chunk_id']) / f'attempt_{receipt["attempt"]:03d}'
        for field, hash_field in (('path', 'sha256'), ('row_indices_path', 'row_indices_sha256'),
                                  ('replay_audit_path', 'replay_audit_sha256')):
            artifact = _within(receipt[field], expected_root)
            if base.sha(artifact, study) != receipt[hash_field]:
                raise ValueError(f'Complete chunk {field} hash changed')
        calibration_path = _within(receipt['calibration_path'], worker_directory(study))
        if calibration_path != worker_directory(study) / f'{safe_worker_id(receipt["worker_id"])}.calibration.json':
            raise ValueError('Complete chunk calibration worker differs')
        if base.sha(calibration_path) != receipt['calibration_sha256']:
            raise ValueError('GPU calibration hash changed')
        calibration = read(calibration_path)
        if (calibration.get('passed') is not True or calibration.get('worker_id') != receipt['worker_id']
                or calibration.get('protocol_sha256') != prepared['protocol_sha256']
                or calibration.get('plan_sha256') != prepared['plan_sha256']):
            raise ValueError('GPU calibration identity differs')
        records = calibration.get('records', [])
        expected_numbers = sorted(set((0, len(plan['groups']) // 2, len(plan['groups']) - 1)))
        if [r.get('group') for r in records] != expected_numbers:
            raise ValueError('Calibration omits common global groups')
        for record, number in zip(records, expected_numbers):
            group = plan['groups'][number]
            if (record.get('rows') != len(group) or record.get('shape') != [len(group), *plan['shape'][1:]]
                    or record.get('dtype') != 'float16' or record.get('forward_repeated_equal') is not True
                    or record.get('sample_ids') != data['meta'].base_id.iloc[group].tolist()
                    or not re.fullmatch('[0-9a-f]{64}', record.get('feature_sha256', ''))):
                raise ValueError('Calibration group/sample/shape binding differs')
        current_signature = [(r['group'], r['feature_sha256']) for r in records]
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise RuntimeError('Same original global-batch FP16 calibration differs across contributing GPUs')
        calibrations[receipt['worker_id']] = dict(path=str(calibration_path), sha256=receipt['calibration_sha256'],
            physical_gpu=calibration['runtime_gpu']['physical_gpu'], runtime_gpu=calibration['runtime_gpu'])
        rows = np.load(receipt['row_indices_path'], allow_pickle=False)
        expected = np.concatenate([np.asarray(g, dtype=np.int64) for g in
            plan['groups'][chunk['group_start']:chunk['group_stop']]])
        feature = np.load(receipt['path'], mmap_mode='r', allow_pickle=False)
        if (not np.array_equal(rows, expected) or seen[rows].any() or feature.dtype != np.float16
                or list(feature.shape) != [len(rows), *plan['shape'][1:]]
                or receipt.get('shape') != list(feature.shape) or not np.isfinite(feature).all()):
            raise ValueError('Complete chunk row coverage/shape/finite gate failed')
        audit = read(receipt['replay_audit_path'])
        if audit.get('passed') is not True or not audit.get('records') or any(
                r.get('saved_equal') is not True or r.get('forward_repeated_equal') is not True for r in audit['records']):
            raise ValueError('Complete chunk original-batch replay gate failed')
        values[rows] = feature
        seen[rows] = True
        chunks.append(dict(chunk_id=chunk['chunk_id'], receipt_path=str(chunk_directory(study, split, chunk['chunk_id']) / 'receipt.json'),
                           receipt_sha256=base.sha(chunk_directory(study, split, chunk['chunk_id']) / 'receipt.json')))
    if not seen.all():
        raise ValueError('Parallel merge omitted samples')
    values.flush()
    metadata_path = folder / f'{split}.metadata.parquet'
    data['meta'].to_parquet(metadata_path, index=False)
    audit_path = folder / f'{split}.replay_audit.json'
    dump(audit_path, dict(passed=True, rule='Exact original-group chunk replays and common global calibration across contributing GPUs',
        complete_chunk_receipts=chunks, contributing_workers=calibrations,
        cross_gpu_same_global_group_fp16_sha256=signature,
        every_row_exactly_once=True, incomplete_chunk_rows_reused=False,
        batch_shape_equivalence_claimed=False))
    checked = base.validate_feature_cache(temporary, plan['shape'], 'float16',
        plan['ordered_base_id_sha256'], data['meta'].base_id.tolist())
    del values
    final = folder / f'{split}.npy'
    temporary.replace(final)
    manifest = dict(complete=True, protocol_sha256=prepared['protocol_sha256'], split=split,
        cohort=split, shape=plan['shape'], ordered_base_id_sha256=plan['ordered_base_id_sha256'],
        rows=plan['rows'], num_layers=protocol['num_layers'], layers=list(range(1, protocol['num_layers'] + 1)),
        hidden_size=protocol['hidden_size'], path=str(final), sha256=base.sha(final, study),
        metadata_path=str(metadata_path), metadata_sha256=base.sha(metadata_path),
        replay_audit_path=str(audit_path), replay_audit_sha256=base.sha(audit_path),
        pooling='last_valid_token', raw_block_outputs=True, truncation=False, dtype='float16',
        adapter_path=None, adapter_sha256=None, backbone_trainable_parameters=0,
        input_sha256=prepared['input_sha256'], validation=checked,
        parallel_original_group_plan_sha256=prepared['plan_sha256'], contributing_gpu_workers=len(calibrations))
    dump(folder / f'{split}.manifest.json', manifest)
    if not base.cache_complete(study, protocol, split):
        raise ValueError('Published parallel cache did not pass the original cache contract')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['feature-worker'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--split', choices=base.SPLITS, required=True)
    args = parser.parse_args(argv)
    run_worker(args.study, args.worker_id, args.split)


if __name__ == '__main__':
    main()
