"""Ragged, untruncated, raw final-block token features for the four-arm study.

This cache reuses the immutable parent tokenization and original global batches.
It never merges large feature chunks or loads an adapter.  Test payload access
is guarded by the new study's selected-model lock, even though the old parent
already contains test features.
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
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard import frozen_base_parallel_features as original
from scripts.hanguard import ab_readout_parallel_queue as inventory

SCHEMA = 'hanguard_four_arm_raw_l32_tokens_v1'
PREPARED_FILES = ('token_ids.npy', 'offsets.npy', 'lengths.npy', 'metadata.parquet', 'batch_plan.json')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    original.dump(path, value)


def sha(path):
    return base.sha(path)


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Token artifact escaped its registered directory')
    return path


def _protocol(study):
    study = Path(study).resolve()
    base.check_stop(study)
    protocol = read(study / 'protocol.json')
    registration = read(study / 'protocol_registration.json')
    if registration.get('protocol_sha256') != sha(study / 'protocol.json'):
        raise ValueError('Four-arm protocol registration changed')
    if protocol.get('use_lora') is not False or protocol.get('adapter_path') is not None:
        raise ValueError('Token features require an unadapted frozen backbone')
    for name, digest in protocol.get('code_sha256', {}).items():
        if sha(name) != digest:
            raise ValueError(f'Registered source changed: {name}')
    return protocol


def _barrier(study, split):
    if split not in ('train', 'validation', 'test'):
        raise ValueError(split)
    base.check_stop(study)
    if split == 'test':
        base.validate_selection_lock(study)


def _parent(protocol):
    return Path(protocol['parent_study']).resolve()


def _folder(study, split):
    study = Path(study).resolve()
    protocol = read(study / 'protocol.json')
    root = Path(protocol.get('token_cache_dir', study / 'token_cache')).resolve()
    if not root.is_relative_to(study):
        raise ValueError('Token cache must be inside the new study')
    return root / split


def _parent_identity(protocol, split, *, full_hash=True):
    parent = _parent(protocol)
    parent_protocol = read(parent / 'protocol.json')
    if read(parent / 'protocol_registration.json').get('protocol_sha256') != sha(parent / 'protocol.json'):
        raise ValueError('Parent protocol registration changed')
    if (parent_protocol.get('use_lora') is not False or parent_protocol.get('adapter_path') is not None
            or Path(parent_protocol['data_dir']).resolve() != Path(protocol['data_dir']).resolve()):
        raise ValueError('Parent is not the same unadapted data release')
    for key in ('model_path', 'model_type', 'num_layers', 'hidden_size'):
        if key in protocol and protocol[key] != parent_protocol[key]:
            raise ValueError(f'Parent backbone {key} differs')
    folder = parent / 'parallel_features' / split
    prepared = read(folder / 'prepared_manifest.json')
    input_path = str(Path(protocol['data_dir']).resolve() / f'{split}.parquet')
    if (prepared.get('complete') is not True or prepared.get('protocol_sha256') != sha(parent / 'protocol.json')
            or prepared.get('input_sha256') != parent_protocol['inputs'][input_path]):
        raise ValueError('Parent prepared split identity differs')
    if protocol.get('inputs', {}).get(input_path, parent_protocol['inputs'][input_path]) != prepared['input_sha256']:
        raise ValueError('New and parent data bytes differ')
    if set(prepared.get('artifacts', {})) != set(PREPARED_FILES):
        raise ValueError('Parent prepared artifact list differs')
    if full_hash:
        for name in PREPARED_FILES:
            if sha(folder / name) != prepared['artifacts'][name]:
                raise ValueError(f'Parent token artifact changed: {name}')
    return parent_protocol, prepared, folder


def prepare(study, split):
    """Bind parent token IDs and exact batches, without copying their payloads."""
    study = Path(study).resolve()
    _barrier(study, split)
    protocol = _protocol(study)
    folder = _folder(study, split)
    if (folder / 'plan.json').exists():
        return _prepared(study, split)[0]
    pp, prepared, source = _parent_identity(protocol, split)
    source_plan = read(source / 'batch_plan.json')
    lengths = np.load(source / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    offsets = np.load(source / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    if (offsets.shape != (len(lengths) + 1,) or offsets[0] != 0
            or not np.array_equal(np.diff(offsets), lengths) or np.any(lengths < 1)):
        raise ValueError('Parent token offsets are invalid')
    if source_plan['rows'] != len(lengths) or source_plan['shape'][-1] != pp['hidden_size']:
        raise ValueError('Parent row/hidden dimensions differ')
    groups = source_plan['groups']
    coverage = np.concatenate([np.asarray(g, np.int64) for g in groups])
    if sorted(coverage.tolist()) != list(range(len(lengths))):
        raise ValueError('Parent original batches do not cover every row exactly once')
    expected_groups = base.legacy.microbatches(np.argsort(lengths, kind='stable'), lengths,
        pp['extraction']['token_budget'], pp['extraction']['max_micro'], pp['extraction']['pad_multiple'])
    if len(groups) != len(expected_groups) or any(not np.array_equal(a, b) for a, b in zip(groups, expected_groups)):
        raise ValueError('Parent global batch shapes changed')
    chunks = []
    for c in source_plan['chunks']:
        ids = np.concatenate([np.asarray(g, np.int64) for g in groups[c['group_start']:c['group_stop']]])
        chunks.append(dict(c, tokens=int(lengths[ids].sum())))
    meta = pd.read_parquet(source / 'metadata.parquet')
    if len(meta) != len(lengths) or not meta.base_id.is_unique:
        raise ValueError('Parent row identity is not unique')
    plan = dict(schema=SCHEMA, complete=True, split=split, protocol_sha256=sha(study / 'protocol.json'),
        parent_protocol_sha256=sha(_parent(protocol) / 'protocol.json'),
        parent_prepared_path=str((source / 'prepared_manifest.json').resolve()),
        parent_prepared_sha256=sha(source / 'prepared_manifest.json'),
        parent_artifacts={name: dict(path=str((source / name).resolve()), sha256=prepared['artifacts'][name])
                          for name in PREPARED_FILES},
        input_sha256=prepared['input_sha256'], ordered_base_id_sha256=base.ordered_base_id_sha256(meta.base_id.tolist()),
        rows=len(lengths), total_tokens=int(offsets[-1]), hidden_size=pp['hidden_size'], layer=pp['num_layers'],
        dtype='float16', adapter_sha256=None, raw_block_outputs=True, truncation=False,
        grouping='original_global_stable_length_order_without_rebatching', groups=groups, chunks=chunks,
        extraction=pp['extraction'], projected_cache_bytes=int(offsets[-1]) * pp['hidden_size'] * 2,
        test_selection_lock_sha256=sha(study / 'selection_lock.json') if split == 'test' else None)
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / 'plan.json', plan)
    return plan


def _prepared(study, split, *, full_hash=True):
    _barrier(study, split)
    protocol = _protocol(study)
    plan = read(_folder(study, split) / 'plan.json')
    if (plan.get('schema') != SCHEMA or plan.get('complete') is not True or plan.get('split') != split
            or plan.get('protocol_sha256') != sha(Path(study) / 'protocol.json')
            or plan.get('adapter_sha256') is not None or plan.get('raw_block_outputs') is not True
            or plan.get('truncation') is not False):
        raise ValueError('Token preparation identity differs')
    if split == 'test' and plan.get('test_selection_lock_sha256') != sha(Path(study) / 'selection_lock.json'):
        raise ValueError('Selected models changed after token preparation')
    pp, prepared, source = _parent_identity(protocol, split, full_hash=full_hash)
    if (plan['parent_protocol_sha256'] != sha(_parent(protocol) / 'protocol.json')
            or plan['parent_prepared_sha256'] != sha(source / 'prepared_manifest.json')):
        raise ValueError('Parent preparation registration changed')
    for name in PREPARED_FILES:
        record = plan['parent_artifacts'][name]
        if Path(record['path']).resolve() != (source / name).resolve() or record['sha256'] != prepared['artifacts'][name]:
            raise ValueError('Token preparation points to a different parent artifact')
    tokens = np.load(source / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    offsets = np.load(source / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    lengths = np.load(source / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    meta = pd.read_parquet(source / 'metadata.parquet')
    if (len(meta) != plan['rows'] or plan['total_tokens'] != len(tokens) or offsets[-1] != len(tokens)
            or not np.array_equal(np.diff(offsets), lengths) or np.any(lengths < 1)
            or not meta.base_id.is_unique or base.ordered_base_id_sha256(meta.base_id.tolist()) != plan['ordered_base_id_sha256']):
        raise ValueError('Token preparation row/token dimensions changed')
    parent_plan = read(source / 'batch_plan.json')
    expected_chunks = []
    for c in parent_plan['chunks']:
        ids = np.concatenate([np.asarray(g, np.int64) for g in parent_plan['groups'][c['group_start']:c['group_stop']]])
        expected_chunks.append(dict(c, tokens=int(lengths[ids].sum())))
    if plan['groups'] != parent_plan['groups'] or plan['chunks'] != expected_chunks or plan['extraction'] != pp['extraction']:
        raise ValueError('Token plan changed original global batches')
    return plan, dict(encoded=original.EncodedRows(tokens, offsets), lengths=lengths, meta=meta)


@contextmanager
def _lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'claims.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _chunk(study, split, chunk_id):
    return _folder(study, split) / 'chunks' / f'{int(chunk_id):06d}'


def _receipt(study, split, plan, chunk):
    path = _chunk(study, split, chunk['chunk_id']) / 'receipt.json'
    if not path.exists():
        return None
    receipt = read(path)
    if (receipt.get('schema') != SCHEMA or receipt.get('complete') is not True
            or receipt.get('split') != split or receipt.get('chunk') != chunk
            or receipt.get('plan_sha256') != sha(_folder(study, split) / 'plan.json')
            or receipt.get('protocol_sha256') != plan['protocol_sha256']):
        raise ValueError('Token chunk receipt identity differs')
    return receipt


def progress(study, split):
    _barrier(study, split)
    plan = read(_folder(study, split) / 'plan.json')
    complete = claimed = rows = tokens = 0
    for chunk in plan['chunks']:
        if _receipt(study, split, plan, chunk):
            complete += 1
            rows += chunk['rows']
            tokens += chunk['tokens']
        else:
            path = _chunk(study, split, chunk['chunk_id']) / 'claim.json'
            if path.exists():
                claim = read(path)
                if not claim.get('released') and original._same_live_process(claim['identity']):
                    claimed += 1
    return dict(split=split, complete=complete == len(plan['chunks']), total_chunks=len(plan['chunks']),
        complete_chunks=complete, claimed_chunks=claimed, pending_chunks=len(plan['chunks']) - complete - claimed,
        completed_rows=rows, total_rows=plan['rows'], completed_tokens=tokens, total_tokens=plan['total_tokens'])


def all_complete(study, split):
    return progress(study, split)['complete']


def release_dead_claims(study, split):
    """Explicitly retire only absent process births; never signal any process."""
    _barrier(study, split)
    plan = read(_folder(study, split) / 'plan.json')
    retired = []
    with _lock(_folder(study, split)):
        for chunk in plan['chunks']:
            directory = _chunk(study, split, chunk['chunk_id'])
            path = directory / 'claim.json'
            if _receipt(study, split, plan, chunk) or not path.exists():
                continue
            claim = read(path)
            if claim.get('released') or original._same_live_process(claim['identity']):
                continue
            claim = dict(claim, released=True, release_reason='Exact process birth no longer exists', released_unix=time.time())
            dump(directory / f'abandoned_attempt_{claim["attempt"]:03d}.json', claim)
            dump(path, claim)
            retired.append(chunk['chunk_id'])
    return retired


def claim_chunk(study, split, worker_id, plan):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', worker_id):
        raise ValueError('Invalid worker ID')
    _barrier(study, split)
    with _lock(_folder(study, split)):
        for chunk in plan['chunks']:
            if _receipt(study, split, plan, chunk):
                continue
            directory = _chunk(study, split, chunk['chunk_id'])
            path = directory / 'claim.json'
            previous = read(path) if path.exists() else None
            if previous and not previous.get('released'):
                continue  # dead claims require explicit coordinator release
            attempt = 1 if previous is None else previous['attempt'] + 1
            target = directory / f'attempt_{attempt:03d}'
            target.mkdir(parents=True)
            claim = dict(worker_id=worker_id, identity=inventory.process_identity(os.getpid()),
                chunk=chunk, attempt=attempt, directory=str(target.resolve()), created_unix=time.time(),
                plan_sha256=sha(_folder(study, split) / 'plan.json'))
            dump(path, claim)
            return claim
    return None


class RawLastTokenCapture:
    """Retain valid token outputs from the final raw transformer block only."""
    def __init__(self, backbone):
        self.backbone, self.values, self.mask = backbone, None, None
        self.handle = backbone.layers[-1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.mask is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        self.values = hidden[self.mask].detach().to(torch.float16).cpu()

    @torch.inference_mode()
    def __call__(self, batch):
        self.mask = batch['attention_mask'].bool()
        self.values = None
        try:
            kind = batch['input_ids'].device.type
            with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == 'cuda'):
                self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
            if self.values is None or self.values.shape[0] != int(self.mask.sum()):
                raise ValueError('Final-block capture omitted valid tokens')
            return self.values
        finally:
            self.mask = None

    def close(self):
        self.handle.remove()


def _signature(value):
    return hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()


def _legacy_last(protocol, split):
    parent = _parent(protocol)
    manifest = read(parent / 'caches' / f'{split}.manifest.json')
    if (manifest.get('complete') is not True or manifest.get('adapter_sha256') is not None
            or manifest.get('raw_block_outputs') is not True
            or manifest.get('protocol_sha256') != sha(parent / 'protocol.json')):
        raise ValueError('Comparison cache is not the original unadapted raw-block cache')
    path = _within(manifest['path'], parent)
    # Hash once per worker; parent raw features are never silently trusted.
    if sha(path) != manifest['sha256']:
        raise ValueError('Original last-token feature bytes changed')
    return np.load(path, mmap_mode='r', allow_pickle=False)


def _compare_last(actual, group, lengths, previous, protocol):
    ends = np.cumsum(lengths[group], dtype=np.int64) - 1
    last = actual[ends]
    expected = np.asarray(previous[group, -1, :])
    atol = float(protocol.get('token_cache', {}).get('atol', 0))
    rtol = float(protocol.get('token_cache', {}).get('rtol', 0))
    if not np.allclose(last, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f'Raw L32 last tokens differ from parent (maxabs={float(np.max(np.abs(last.astype(np.float32)-expected.astype(np.float32))))})')
    return dict(passed=True, rows=len(group), atol=atol, rtol=rtol,
        exact_equal=bool(np.array_equal(last, expected)), old_last_sha256=_signature(expected),
        new_last_sha256=_signature(last))


def _capture_group(capture, data, group, tokenizer, plan, device):
    batch = base.legacy.make_batch(data, group, tokenizer, plan['extraction'], device=device)
    actual = capture(batch).numpy()
    expected_shape = (int(data['lengths'][group].sum()), plan['hidden_size'])
    if actual.dtype != np.float16 or actual.shape != expected_shape or not np.isfinite(actual).all():
        raise ValueError('Captured raw token shape/dtype/values are invalid')
    return actual


def _calibration(study, split, worker_id, capture, data, tokenizer, plan, protocol, previous, device):
    records = []
    groups = plan['groups']
    for number in sorted(set((0, len(groups) // 2, len(groups) - 1))):
        _barrier(study, split)
        group = np.asarray(groups[number], np.int64)
        actual = _capture_group(capture, data, group, tokenizer, plan, device)
        repeated = _capture_group(capture, data, group, tokenizer, plan, device)
        if not np.array_equal(actual, repeated):
            raise RuntimeError('Repeated full token calibration batch differs')
        check = _compare_last(actual, group, data['lengths'], previous, protocol)
        records.append(dict(group=number, shape=list(actual.shape), base_ids=data['meta'].base_id.iloc[group].tolist(),
            tokens_sha256=_signature(actual), repeated_exact=True, parent_last_check=check))
    path = _folder(study, split) / 'workers' / f'{worker_id}.calibration.json'
    dump(path, dict(schema=SCHEMA, passed=True, worker_id=worker_id, records=records,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), protocol_sha256=plan['protocol_sha256'],
        plan_sha256=sha(_folder(study, split) / 'plan.json')))
    return path


def extract_chunk(study, split, worker_id, claim, plan, data, tokenizer, capture,
                  calibration_path, protocol, previous, device='cuda'):
    """One immutable ragged feature array; exposed for synthetic CPU validation."""
    target = Path(claim['directory'])
    if any(target.iterdir()):
        raise ValueError('Partial token attempts cannot be reused')
    chunk = claim['chunk']
    groups = [np.asarray(g, np.int64) for g in plan['groups'][chunk['group_start']:chunk['group_stop']]]
    rows = np.concatenate(groups)
    lengths = np.asarray(data['lengths'][rows], np.int32)
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    shape = (int(offsets[-1]), plan['hidden_size'])
    temporary = target / 'tokens.partial.npy'
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=shape)
    begin = time.monotonic()
    position = 0
    first_check = None
    for number, group in enumerate(groups):
        _barrier(study, split)
        actual = _capture_group(capture, data, group, tokenizer, plan, device)
        if number == 0:
            first_check = _compare_last(actual, group, data['lengths'], previous, protocol)
        values[position:position + len(actual)] = actual
        position += len(actual)
        dump(_folder(study, split) / 'workers' / f'{worker_id}.progress.json', dict(
            state='extracting', split=split, worker_id=worker_id, chunk_id=chunk['chunk_id'],
            completed_groups=number + 1, total_groups=len(groups), completed_chunk_tokens=position,
            updated_unix=time.time()))
    if position != shape[0] or shape[0] != chunk['tokens']:
        raise ValueError('Chunk token coverage differs from the fixed plan')
    values.flush()
    del values
    final = target / 'tokens.npy'
    temporary.replace(final)
    np.save(target / 'rows.npy', rows, allow_pickle=False)
    np.save(target / 'offsets.npy', offsets, allow_pickle=False)
    np.save(target / 'lengths.npy', lengths, allow_pickle=False)
    dump(target / 'row_identity.json', dict(base_ids=data['meta'].base_id.iloc[rows].tolist(), first_batch_last_token_check=first_check))
    artifacts = {name: dict(path=str((target / name).resolve()), sha256=sha(target / name)) for name in
                 ('tokens.npy', 'rows.npy', 'offsets.npy', 'lengths.npy', 'row_identity.json')}
    receipt = dict(schema=SCHEMA, complete=True, split=split, chunk=chunk, worker_id=worker_id,
        attempt=claim['attempt'], identity=claim['identity'], protocol_sha256=plan['protocol_sha256'],
        plan_sha256=sha(_folder(study, split) / 'plan.json'), shape=list(shape), artifacts=artifacts,
        calibration_path=str(Path(calibration_path).resolve()), calibration_sha256=sha(calibration_path),
        elapsed_seconds=time.monotonic() - begin, completed_unix=time.time())
    with _lock(_folder(study, split)):
        if read(_chunk(study, split, chunk['chunk_id']) / 'claim.json') != claim:
            raise ValueError('Token claim changed before completion publication')
        _barrier(study, split)
        dump(_chunk(study, split, chunk['chunk_id']) / 'receipt.json', receipt)
    return receipt


def extract_worker(study, split, worker_id, device='cuda', *, admission_already_checked=False):
    """Called only after the root scheduler's exact worker admission."""
    study = Path(study).resolve()
    _barrier(study, split)
    protocol = _protocol(study)
    if not str(device).startswith('cuda'):
        raise ValueError('Real frozen-backbone extraction requires an admitted GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit():
        raise ValueError('Exactly one physical GPU must be visible')
    if admission_already_checked:
        record = read(study / 'workers' / f'{worker_id}.admitted.json')
        actual = inventory.process_identity(os.getpid())
        expected = record.get('identity', record.get('process_identity', {}))
        if (expected.get('pid') != os.getpid() or expected.get('uid') != actual['uid']
                or expected.get('start_time_ticks', expected.get('start_ticks')) != actual['start_time_ticks']
                or record.get('protocol_sha256') != sha(study / 'protocol.json')
                or int(record.get('physical_gpu', -1)) != int(visible)):
            raise ValueError('Token worker was not admitted with this exact process identity')
    else:
        card = next((c for c in inventory.gpu_inventory() if c['index'] == int(visible)), None)
        if card is None or any(pid != os.getpid() for pid in card['pids']):
            raise RuntimeError('GPU is no longer empty at worker admission')
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')
    plan, data = _prepared(study, split)
    pp, _, _ = _parent_identity(protocol, split, full_hash=False)
    # Fully hash pinned model weights before loading the unadapted backbone.
    for filename, digest in pp['inputs'].items():
        if Path(filename).is_relative_to(Path(pp['model_path'])) and sha(filename) != digest:
            raise ValueError(f'Frozen backbone artifact changed: {filename}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, backbone, tokenizer = base.load_frozen_backbone(pp)
    capture = RawLastTokenCapture(backbone)
    try:
        previous = _legacy_last(protocol, split)
        calibration = _calibration(study, split, worker_id, capture, data, tokenizer, plan, protocol, previous, device)
        while True:
            _barrier(study, split)
            claim = claim_chunk(study, split, worker_id, plan)
            if claim is None:
                if all_complete(study, split) or progress(study, split)['pending_chunks'] == 0:
                    break
                time.sleep(.2)
                continue
            receipt = extract_chunk(study, split, worker_id, claim, plan, data, tokenizer, capture,
                                    calibration, protocol, previous, device)
            print(json.dumps(dict(state='token_chunk_completed', split=split, worker_id=worker_id,
                chunk_id=claim['chunk']['chunk_id'], shape=receipt['shape'])), flush=True)
        dump(_folder(study, split) / 'workers' / f'{worker_id}.complete.json',
             dict(complete=True, split=split, worker_id=worker_id, completed_unix=time.time()))
    finally:
        capture.close()
        del capture, model, backbone
        gc.collect()
        torch.cuda.empty_cache()


def _validate_chunks(study, split, plan, data, *, full_hash=True):
    seen = np.zeros(plan['rows'], dtype=bool)
    signature = None
    records = []
    for chunk in plan['chunks']:
        receipt = _receipt(study, split, plan, chunk)
        if receipt is None:
            raise ValueError('All token chunks must complete before use')
        expected_root = _chunk(study, split, chunk['chunk_id']) / f'attempt_{receipt["attempt"]:03d}'
        if set(receipt['artifacts']) != {'tokens.npy', 'rows.npy', 'offsets.npy', 'lengths.npy', 'row_identity.json'}:
            raise ValueError('Token chunk artifact list differs')
        for name, artifact in receipt['artifacts'].items():
            path = _within(artifact['path'], expected_root)
            if path.name != name or full_hash and sha(path) != artifact['sha256']:
                raise ValueError('Committed token artifact bytes changed')
        arrays = {name: np.load(receipt['artifacts'][name]['path'], mmap_mode='r', allow_pickle=False)
                  for name in ('tokens.npy', 'rows.npy', 'offsets.npy', 'lengths.npy')}
        rows, lengths, offsets, values = [arrays[n] for n in ('rows.npy', 'lengths.npy', 'offsets.npy', 'tokens.npy')]
        expected_rows = np.concatenate([np.asarray(g, np.int64) for g in plan['groups'][chunk['group_start']:chunk['group_stop']]])
        if (not np.array_equal(rows, expected_rows) or seen[rows].any()
                or not np.array_equal(lengths, data['lengths'][rows]) or offsets.shape != (len(rows) + 1,)
                or offsets[0] != 0 or not np.array_equal(np.diff(offsets), lengths)
                or values.shape != (int(offsets[-1]), plan['hidden_size']) or values.dtype != np.float16
                or values.shape[0] != chunk['tokens']):
            raise ValueError('Token chunk row/length/shape coverage differs')
        for start in range(0, len(values), 65536):
            if not np.isfinite(values[start:start + 65536]).all():
                raise ValueError('Token feature values are not finite')
        row_identity = read(receipt['artifacts']['row_identity.json']['path'])
        if (row_identity['base_ids'] != data['meta'].base_id.iloc[rows].tolist()
                or row_identity['first_batch_last_token_check'].get('passed') is not True):
            raise ValueError('Token chunk row IDs or parent-last-token gate differ')
        cp = _within(receipt['calibration_path'], _folder(study, split) / 'workers')
        if sha(cp) != receipt['calibration_sha256']:
            raise ValueError('Token calibration bytes changed')
        calibration = read(cp)
        expected_numbers = sorted(set((0, len(plan['groups']) // 2, len(plan['groups']) - 1)))
        if (calibration.get('passed') is not True or calibration.get('worker_id') != receipt['worker_id']
                or calibration.get('plan_sha256') != receipt['plan_sha256']
                or calibration.get('protocol_sha256') != plan['protocol_sha256']
                or [r['group'] for r in calibration['records']] != expected_numbers):
            raise ValueError('Token calibration identity/global groups differ')
        for rec in calibration['records']:
            group = np.asarray(plan['groups'][rec['group']], np.int64)
            if (rec.get('repeated_exact') is not True or rec['base_ids'] != data['meta'].base_id.iloc[group].tolist()
                    or rec['shape'] != [int(data['lengths'][group].sum()), plan['hidden_size']]
                    or not rec['parent_last_check'].get('passed')
                    or not re.fullmatch('[0-9a-f]{64}', rec.get('tokens_sha256', ''))):
                raise ValueError('Token calibration row/token binding differs')
        current = [(r['group'], r['tokens_sha256']) for r in calibration['records']]
        if signature is None:
            signature = current
        elif signature != current:
            raise RuntimeError('Same full-batch token FP16 bytes differ across GPUs')
        seen[rows] = True
        records.append(dict(chunk_id=chunk['chunk_id'], receipt_path=str((_chunk(study, split, chunk['chunk_id']) / 'receipt.json').resolve()),
                            receipt_sha256=sha(_chunk(study, split, chunk['chunk_id']) / 'receipt.json'),
                            rows=len(rows), tokens=len(values), artifacts=receipt['artifacts']))
    if not seen.all():
        raise ValueError('Token cache omitted rows')
    return records, signature


def completion_manifest_finalize(study, split):
    """Validate chunks in place; publish only a small manifest, never a copy."""
    plan, data = _prepared(study, split)
    records, signature = _validate_chunks(study, split, plan, data)
    manifest = dict(schema=SCHEMA, complete=True, split=split, protocol_sha256=plan['protocol_sha256'],
        plan_sha256=sha(_folder(study, split) / 'plan.json'), shape=[plan['total_tokens'], plan['hidden_size']],
        rows=plan['rows'], total_tokens=plan['total_tokens'], dtype='float16', layer=plan['layer'],
        adapter_sha256=None, raw_block_outputs=True, truncation=False, chunks=records,
        calibration_signature=[list(x) for x in signature], ordered_base_id_sha256=plan['ordered_base_id_sha256'],
        test_selection_lock_sha256=plan['test_selection_lock_sha256'], completed_unix=time.time())
    path = _folder(study, split) / 'manifest.json'
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != 'completed_unix'} != {k: v for k, v in manifest.items() if k != 'completed_unix'}:
            raise ValueError('Published token manifest changed')
        return previous
    dump(path, manifest)
    return manifest


class TokenCache:
    """Row-addressed mmap view over completed ragged chunks."""
    def __init__(self, plan, data, manifest):
        self.metadata = data['meta']
        self.meta = self.metadata
        self.lengths = np.asarray(data['lengths'])
        self.hidden_size = plan['hidden_size']
        self.shape = (plan['rows'], None, self.hidden_size)
        self.manifest = manifest
        self._chunks = []
        self._where = np.empty((plan['rows'], 2), dtype=np.int64)
        for number, record in enumerate(manifest['chunks']):
            art = record['artifacts']
            values = np.load(art['tokens.npy']['path'], mmap_mode='r', allow_pickle=False)
            rows = np.load(art['rows.npy']['path'], mmap_mode='r', allow_pickle=False)
            offsets = np.load(art['offsets.npy']['path'], mmap_mode='r', allow_pickle=False)
            self._chunks.append((values, offsets))
            self._where[rows, 0] = number
            self._where[rows, 1] = np.arange(len(rows))

    def __len__(self):
        return len(self.lengths)

    def get_row(self, row):
        row = int(row)
        if row < 0 or row >= len(self):
            raise IndexError(row)
        chunk, local = self._where[row]
        values, offsets = self._chunks[chunk]
        return values[offsets[local]:offsets[local + 1]]

    def __getitem__(self, rows):
        if np.isscalar(rows):
            return self.get_row(rows)
        return self.padded_batch(rows)

    def padded_batch(self, rows, *, device='cpu', dtype=torch.float32):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not len(rows):
            raise ValueError('A nonempty flat row-index batch is required')
        if (rows < 0).any() or (rows >= len(self)).any():
            raise IndexError('Row index out of range')
        lengths = self.lengths[rows]
        values = torch.zeros((len(rows), int(lengths.max()), self.hidden_size), dtype=dtype)
        mask = torch.zeros((len(rows), int(lengths.max())), dtype=torch.bool)
        for i, row in enumerate(rows):
            tokens = self.get_row(row)
            values[i, :len(tokens)] = torch.as_tensor(np.array(tokens, copy=True), dtype=dtype)
            mask[i, :len(tokens)] = True
        return values.to(device), mask.to(device)


def load(study, split):
    """Verify complete manifest, chunks and row identities before exposing data."""
    plan, data = _prepared(study, split)
    path = _folder(study, split) / 'manifest.json'
    if not path.exists():
        raise ValueError('Token completion manifest has not been published')
    manifest = read(path)
    records, signature = _validate_chunks(study, split, plan, data)
    if (manifest.get('schema') != SCHEMA or manifest.get('complete') is not True
            or manifest.get('protocol_sha256') != plan['protocol_sha256']
            or manifest.get('plan_sha256') != sha(_folder(study, split) / 'plan.json')
            or manifest.get('chunks') != records
            or manifest.get('shape') != [plan['total_tokens'], plan['hidden_size']]
            or manifest.get('calibration_signature') != [list(x) for x in signature]
            or manifest.get('ordered_base_id_sha256') != plan['ordered_base_id_sha256']
            or manifest.get('test_selection_lock_sha256') != plan['test_selection_lock_sha256']):
        raise ValueError('Token completion manifest identity differs')
    return TokenCache(plan, data, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'extract', 'progress', 'finalize', 'release-dead'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--worker-id')
    args = parser.parse_args()
    if args.command == 'extract':
        if not args.worker_id:
            parser.error('--worker-id is required for extract')
        extract_worker(args.study, args.split, args.worker_id)
        return
    fn = {'prepare': prepare, 'progress': progress, 'finalize': completion_manifest_finalize,
          'release-dead': release_dead_claims}[args.command]
    print(json.dumps(fn(args.study, args.split), ensure_ascii=False))


if __name__ == '__main__':
    main()
