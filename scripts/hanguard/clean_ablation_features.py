"""Fresh joint feature extraction with no historical experiment dependencies.

Only the raw registered parquet release, the pristine registered foundation
model, and files below this study are read.  All transformer-block last tokens
and final-block full tokens come from the same forward call.  Chunk payloads
are immutable and remain in place; interrupted attempts are never reused.
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
from scripts.hanguard import ab_readout_parallel_queue as inventory
from scripts.hanguard import clean_ablation_runtime as runtime

SCHEMA = 'hanguard_clean_joint_all_blocks_full_tokens_v1'
PREPARED_FILES = ('token_ids.npy', 'offsets.npy', 'lengths.npy',
                  'metadata.parquet', 'batch_plan.json')
PAYLOAD_FILES = ('last.npy', 'tokens.npy', 'rows.npy', 'lengths.npy', 'offsets.npy', 'row_identity.json')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return base.sha(path)


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Clean feature artifact escaped its registered directory')
    return path


def _stop(study):
    base.check_stop(study)


def _protocol(study):
    study = Path(study).resolve()
    _stop(study)
    protocol = read(study / 'protocol.json')
    if read(study / 'protocol_registration.json').get('protocol_sha256') != sha(study / 'protocol.json'):
        raise ValueError('Clean protocol registration changed')
    if protocol.get('use_lora') is not False or protocol.get('adapter_path') is not None:
        raise ValueError('Clean joint features require a pristine frozen foundation model')
    forbidden = ('parent_study', 'parent_cache', 'last_token_cache_dir', 'token_cache_dir',
                 'exclusion_ids_path', 'legacy_results', 'adapter_checkpoint')
    if any(protocol.get(name) is not None for name in forbidden):
        raise ValueError('Historical experiment input is forbidden')
    allowed = [Path(protocol['model_path']).resolve(), Path(protocol['data_dir']).resolve()]
    for path in protocol.get('inputs', {}):
        if not any(Path(path).resolve().is_relative_to(root) for root in allowed):
            raise ValueError('Registered input is outside the raw release and pristine model')
    for path, digest in protocol.get('code_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError(f'Registered source changed: {path}')
    return protocol


def _barrier(study, split):
    if split not in ('train', 'validation', 'test'):
        raise ValueError(split)
    _stop(study)
    if split == 'test':
        from scripts.hanguard import clean_ablation_study
        clean_ablation_study.verify_selection_lock(study)


def _folder(study, split):
    study = Path(study).resolve()
    protocol = read(study / 'protocol.json')
    return _within(protocol.get('features_dir', study / 'features'), study) / split


@contextmanager
def _lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'claims.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _save_array(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def _input_hash(protocol, split):
    path = (Path(protocol['data_dir']) / f'{split}.parquet').resolve()
    hashes = protocol.get('data_sha256', {})
    expected = protocol.get('inputs', {}).get(str(path), hashes.get(split, hashes.get(f'{split}.parquet')))
    if expected is None or sha(path) != expected:
        raise ValueError(f'Raw registered data changed: {split}')
    return path, expected


def _tokenizer(protocol):
    from transformers import AutoTokenizer
    result = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
    if result.pad_token_id is None:
        result.pad_token = result.eos_token
    if result.pad_token_id is None:
        raise ValueError('Tokenizer has no usable padding token')
    result.padding_side = 'right'
    return result


class EncodedRows:
    def __init__(self, values, offsets):
        self.values, self.offsets = values, offsets

    def __getitem__(self, row):
        row = int(row)
        return self.values[self.offsets[row]:self.offsets[row + 1]].tolist()


def _calibration_groups(groups, lengths):
    # Every number is selected only from fresh length-sorted global groups.
    medians = np.asarray([np.median(lengths[g]) for g in groups])
    chosen = {0, len(groups) - 1}
    for target in (1, 42, 128, 129, 130):
        chosen.add(int(np.argmin(np.abs(medians - target))))
    return sorted(chosen)


def prepare(study, split):
    """Tokenize raw text afresh and freeze this study's own batch/chunk plan."""
    study = Path(study).resolve()
    _barrier(study, split)
    protocol = _protocol(study)
    folder = _folder(study, split)
    with _lock(folder):
        if (folder / 'plan.json').exists():
            return _prepared(study, split)[0]
        partial_files = [folder / name for name in PREPARED_FILES if (folder / name).exists()]
        if partial_files:
            raise ValueError('Uncommitted preparation exists; do not reuse partial tokenization')
        path, digest = _input_hash(protocol, split)
        columns = ['prompt', 'base_id', 'group_id', 'source', 'prompt_harm_label',
                   'split', 'category_id', 'type_label_mask']
        frame = pd.read_parquet(path, columns=columns).sort_values('base_id', kind='stable').reset_index(drop=True)
        base.validate_metadata(frame, split)
        if protocol.get('rows', {}).get(split, len(frame)) != len(frame):
            raise ValueError('Raw split row count changed')
        tokenizer = _tokenizer(protocol)
        encoded = tokenizer(frame.prompt.tolist(), add_special_tokens=False, truncation=False)['input_ids']
        lengths = np.asarray([len(row) for row in encoded], dtype=np.int32)
        extraction = protocol['extraction']
        if len(lengths) != len(frame) or np.any(lengths < 1) or np.any(lengths > extraction['max_tokens']):
            raise ValueError('Empty/over-budget text; truncation is forbidden')
        offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
        tokens = np.concatenate([np.asarray(row, dtype=np.int32) for row in encoded])
        meta = frame.drop(columns=['prompt', 'prompt_harm_label']).copy()
        meta['label'] = frame.prompt_harm_label.eq('harmful').astype(np.int8)
        meta['prompt_tokens'] = lengths
        order = np.argsort(lengths, kind='stable')
        groups = [g.tolist() for g in base.legacy.microbatches(order, lengths,
            extraction['token_budget'], extraction['max_micro'], extraction['pad_multiple'])]
        chunks = []
        groups_per_chunk = int(extraction.get('groups_per_chunk', extraction.get('group_chunk_size', 32)))
        if groups_per_chunk < 1:
            raise ValueError('groups_per_chunk must be positive')
        for start in range(0, len(groups), groups_per_chunk):
            stop = min(start + groups_per_chunk, len(groups))
            rows = np.concatenate([np.asarray(group, np.int64) for group in groups[start:stop]])
            chunks.append(dict(chunk_id=len(chunks), group_start=start, group_stop=stop,
                               rows=len(rows), tokens=int(lengths[rows].sum())))
        batch_plan = dict(groups=groups, chunks=chunks, grouping='fresh_stable_length_global_microbatches')
        _save_array(folder / 'token_ids.npy', tokens)
        _save_array(folder / 'offsets.npy', offsets)
        _save_array(folder / 'lengths.npy', lengths)
        temporary = folder / 'metadata.partial.parquet'
        meta.to_parquet(temporary, index=False)
        temporary.replace(folder / 'metadata.parquet')
        dump(folder / 'batch_plan.json', batch_plan)
        plan = dict(schema=SCHEMA, complete=True, split=split, protocol_sha256=sha(study / 'protocol.json'),
            input_path=str(path), input_sha256=digest, rows=len(frame), total_tokens=int(offsets[-1]),
            num_layers=protocol['num_layers'], hidden_size=protocol['hidden_size'], dtype='float16',
            adapter_sha256=None, raw_block_outputs=True, truncation=False,
            ordered_base_id_sha256=base.ordered_base_id_sha256(meta.base_id.tolist()),
            artifacts={name: sha(folder / name) for name in PREPARED_FILES},
            tokenizer=dict(add_special_tokens=False, padding_side='right', pad_token_id=int(tokenizer.pad_token_id)),
            extraction=extraction, groups=groups, chunks=chunks,
            calibration_groups=_calibration_groups(groups, lengths),
            projected_cache_bytes=int((offsets[-1] + len(frame) * protocol['num_layers']) * protocol['hidden_size'] * 2),
            test_selection_lock_sha256=sha(study / 'selection_lock.json') if split == 'test' else None)
        dump(folder / 'plan.json', plan)
        return plan


def _prepared(study, split):
    _barrier(study, split)
    protocol = _protocol(study)
    folder = _folder(study, split)
    plan = read(folder / 'plan.json')
    if (plan.get('schema') != SCHEMA or plan.get('complete') is not True or plan.get('split') != split
            or plan.get('protocol_sha256') != sha(Path(study) / 'protocol.json')
            or plan.get('adapter_sha256') is not None or plan.get('raw_block_outputs') is not True
            or plan.get('truncation') is not False or plan.get('extraction') != protocol['extraction']
            or plan.get('hidden_size') != protocol['hidden_size'] or plan.get('num_layers') != protocol['num_layers']):
        raise ValueError('Clean preparation identity changed')
    path, digest = _input_hash(protocol, split)
    if plan.get('input_path') != str(path) or plan.get('input_sha256') != digest:
        raise ValueError('Clean preparation raw source changed')
    if split == 'test' and plan.get('test_selection_lock_sha256') != sha(Path(study) / 'selection_lock.json'):
        raise ValueError('Selections changed after test preparation')
    if set(plan.get('artifacts', {})) != set(PREPARED_FILES):
        raise ValueError('Clean prepared artifact list changed')
    for name, digest in plan['artifacts'].items():
        if sha(folder / name) != digest:
            raise ValueError(f'Clean prepared artifact changed: {name}')
    tokens = np.load(folder / 'token_ids.npy', mmap_mode='r', allow_pickle=False)
    offsets = np.load(folder / 'offsets.npy', mmap_mode='r', allow_pickle=False)
    lengths = np.load(folder / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    meta = pd.read_parquet(folder / 'metadata.parquet')
    groups = plan['groups']
    coverage = np.concatenate([np.asarray(g, np.int64) for g in groups])
    expected = [g.tolist() for g in base.legacy.microbatches(np.argsort(lengths, kind='stable'), lengths,
        protocol['extraction']['token_budget'], protocol['extraction']['max_micro'], protocol['extraction']['pad_multiple'])]
    if (tokens.dtype != np.int32 or offsets.dtype != np.int64 or lengths.dtype != np.int32
            or offsets.shape != (len(lengths) + 1,) or offsets[0] != 0
            or not np.array_equal(np.diff(offsets), lengths) or np.any(lengths < 1)
            or np.any(lengths > protocol['extraction']['max_tokens']) or offsets[-1] != len(tokens)
            or plan['total_tokens'] != len(tokens) or plan['rows'] != len(meta) or len(meta) != len(lengths)
            or not meta.base_id.is_unique or not meta.base_id.is_monotonic_increasing
            or not np.array_equal(meta.prompt_tokens, lengths)
            or sorted(coverage.tolist()) != list(range(len(meta))) or groups != expected
            or read(folder / 'batch_plan.json') != dict(groups=groups, chunks=plan['chunks'],
                grouping='fresh_stable_length_global_microbatches')
            or base.ordered_base_id_sha256(meta.base_id.tolist()) != plan['ordered_base_id_sha256']):
        raise ValueError('Clean preparation row/token/batch coverage changed')
    expected_chunks = []
    for c in plan['chunks']:
        rows = np.concatenate([np.asarray(g, np.int64) for g in groups[c['group_start']:c['group_stop']]])
        expected_chunks.append(dict(chunk_id=len(expected_chunks), group_start=c['group_start'],
            group_stop=c['group_stop'], rows=len(rows), tokens=int(lengths[rows].sum())))
    chunk_groups = [i for c in plan['chunks'] for i in range(c['group_start'], c['group_stop'])]
    if expected_chunks != plan['chunks'] or chunk_groups != list(range(len(groups))):
        raise ValueError('Clean chunk plan omits/duplicates global groups')
    if plan['calibration_groups'] != _calibration_groups(groups, lengths):
        raise ValueError('Fresh length-selected calibration groups changed')
    return plan, dict(encoded=EncodedRows(tokens, offsets), lengths=lengths, meta=meta)


class JointCapture:
    """Capture every raw block's last valid token and final-block full tokens."""
    def __init__(self, backbone, kernel_path=None):
        self.backbone = backbone
        self.count = len(backbone.layers)
        self.kernel_path = kernel_path
        self.values, self.tokens, self.mask, self.indices = {}, None, None, None
        self.handles = [layer.register_forward_hook(self._hook(index))
                        for index, layer in enumerate(backbone.layers)]

    def _hook(self, index):
        def hook(module, inputs, output):
            if self.mask is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(len(hidden), device=hidden.device)
            self.values[index] = hidden[rows, self.indices].detach().to(torch.float16)
            if index == self.count - 1:
                self.tokens = hidden[self.mask].detach().to(torch.float16)
        return hook

    @torch.inference_mode()
    def __call__(self, batch):
        mask = batch['attention_mask'].bool()
        if mask.ndim != 2 or not mask.any(dim=1).all():
            raise ValueError('Each row requires at least one valid token')
        if not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(1)[:, None]):
            raise ValueError('Clean extraction requires right padding without internal mask holes')
        self.mask = mask
        self.indices = mask.sum(dim=1) - 1
        self.values.clear()
        self.tokens = None
        try:
            kind = batch['input_ids'].device.type
            with runtime.math_sdpa(), torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == 'cuda'):
                self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
            if set(self.values) != set(range(self.count)) or self.tokens is None:
                raise ValueError('Joint capture omitted a transformer block')
            last = torch.stack([self.values[i] for i in range(self.count)], dim=1)
            ends = mask.sum(1).cumsum(0) - 1
            if not torch.equal(last[:, -1], self.tokens[ends]):
                raise RuntimeError('Same-forward last tokens disagree with full tokens')
            return last.cpu().numpy(), self.tokens.cpu().numpy()
        finally:
            self.mask, self.indices, self.tokens = None, None, None
            self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def _signature(values):
    return hashlib.sha256(np.asarray(values).tobytes(order='C')).hexdigest()


def _capture_group(capture, data, group, tokenizer, plan, device):
    group = np.asarray(group, np.int64)
    batch = base.legacy.make_batch(data, group, tokenizer, plan['extraction'], device=device)
    last, tokens = capture(batch)
    last, tokens = np.asarray(last), np.asarray(tokens)
    if (last.shape != (len(group), plan['num_layers'], plan['hidden_size'])
            or tokens.shape != (int(data['lengths'][group].sum()), plan['hidden_size'])
            or last.dtype != np.float16 or tokens.dtype != np.float16
            or not np.isfinite(last).all() or not np.isfinite(tokens).all()
            or not np.array_equal(last[:, -1], tokens[np.cumsum(data['lengths'][group], dtype=np.int64) - 1])):
        raise ValueError('Joint capture shape/finite/same-forward consistency failed')
    return last, tokens


def _calibration(study, worker_id, capture, tokenizer, device, *, compare_baseline=True):
    plan, data = _prepared(study, 'train')
    records = []
    for number in plan['calibration_groups']:
        _stop(study)
        group = np.asarray(plan['groups'][number], np.int64)
        last, tokens = _capture_group(capture, data, group, tokenizer, plan, device)
        repeated_last, repeated_tokens = _capture_group(capture, data, group, tokenizer, plan, device)
        if not np.array_equal(last, repeated_last) or not np.array_equal(tokens, repeated_tokens):
            raise RuntimeError('Repeated fresh joint calibration is not exact')
        records.append(dict(group=number, base_ids=data['meta'].base_id.iloc[group].tolist(),
            lengths=data['lengths'][group].tolist(), last_shape=list(last.shape), tokens_shape=list(tokens.shape),
            last_sha256=_signature(last), tokens_sha256=_signature(tokens), repeated_exact=True,
            same_forward_last_exact=True))
    result = dict(schema=SCHEMA, passed=True, worker_id=worker_id,
        identity=inventory.process_identity(os.getpid()), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        protocol_sha256=sha(Path(study) / 'protocol.json'), train_plan_sha256=sha(_folder(study, 'train') / 'plan.json'),
        records=records, completed_unix=time.time())
    kernel_path = getattr(capture, 'kernel_path', None)
    if kernel_path is not None:
        kernel_record = Path(study) / 'calibration' / f'{worker_id}.kernels.json'
        dump(kernel_record, kernel_path.snapshot())
        result['kernel_path_record'] = str(kernel_record.resolve())
        result['kernel_path_sha256_at_calibration'] = sha(kernel_record)
    directory = Path(study) / 'calibration'
    with _lock(directory):
        baseline_path = directory / 'baseline.json'
        if baseline_path.exists():
            baseline = read(baseline_path)
            if (baseline.get('protocol_sha256') != result['protocol_sha256']
                    or baseline.get('train_plan_sha256') != result['train_plan_sha256']
                    or baseline.get('records') != result['records']):
                failure_path = directory / f'{worker_id}.failed.json'
                dump(failure_path, dict(result, passed=False, reason='Fresh cross-process/GPU calibration mismatch'))
                raise RuntimeError('Fresh joint features differ from this study calibration baseline')
        elif compare_baseline:
            dump(baseline_path, result)
        else:
            raise ValueError('A calibration baseline must exist')
        path = directory / f'{worker_id}.json'
        if path.exists():
            raise ValueError('Calibration worker ID has already been used')
        dump(path, result)
    return path


def _admit(study, worker_id, device, admission_already_checked):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', worker_id):
        raise ValueError('Invalid worker ID')
    if not str(device).startswith('cuda'):
        raise ValueError('Real foundation feature extraction requires an admitted GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit():
        raise ValueError('Exactly one physical GPU must be visible')
    actual = inventory.process_identity(os.getpid())
    if admission_already_checked:
        record = read(Path(study) / 'workers' / f'{worker_id}.admitted.json')
        expected = record.get('identity', record.get('process_identity', {}))
        keys = ('pid', 'uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd')
        if (any(expected.get(key) != actual.get(key) for key in keys)
                or record.get('protocol_sha256') != sha(Path(study) / 'protocol.json')
                or int(record.get('physical_gpu', -1)) != int(visible)):
            raise ValueError('GPU admission does not bind this exact worker birth/command')
    else:
        card = next((c for c in inventory.gpu_inventory() if c['index'] == int(visible)), None)
        if card is None or any(pid != os.getpid() for pid in card['pids']):
            raise RuntimeError('GPU is not empty at admission')
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU required')


def _setup_model(study):
    protocol = _protocol(study)
    model_path = Path(protocol['model_path']).resolve()
    for path, digest in protocol.get('inputs', {}).items():
        if Path(path).resolve().is_relative_to(model_path) and sha(path) != digest:
            raise ValueError(f'Pristine foundation artifact changed: {path}')
    torch.set_num_threads(min(8, int(protocol.get('cpu_threads', 4))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, backbone, tokenizer = base.load_frozen_backbone(protocol)
    if tokenizer.padding_side != 'right':
        raise ValueError('Frozen backbone tokenizer must right-pad')
    return model, backbone, tokenizer


def run_calibration(study, worker_id, device='cuda', *, admission_already_checked=True):
    study = Path(study).resolve()
    _protocol(study)
    prepare(study, 'train')
    _admit(study, worker_id, device, admission_already_checked)
    model, backbone, tokenizer = _setup_model(study)
    kernel_path = runtime.DeterministicKernelPath(require_fla=read(study / 'protocol.json')['model_type'] == 'qwen3_5')
    capture = JointCapture(backbone, kernel_path)
    try:
        path = _calibration(study, worker_id, capture, tokenizer, device)
        return read(path)
    finally:
        capture.close()
        del capture, model, backbone
        gc.collect()
        torch.cuda.empty_cache()


def _chunk(study, split, chunk_id):
    return _folder(study, split) / 'chunks' / f'{int(chunk_id):06d}'


def _receipt(study, split, plan, chunk):
    path = _chunk(study, split, chunk['chunk_id']) / 'receipt.json'
    if not path.exists():
        return None
    result = read(path)
    if (result.get('schema') != SCHEMA or result.get('complete') is not True
            or result.get('split') != split or result.get('chunk') != chunk
            or result.get('protocol_sha256') != plan['protocol_sha256']
            or result.get('plan_sha256') != sha(_folder(study, split) / 'plan.json')):
        raise ValueError('Committed joint feature receipt identity changed')
    return result


def _same_live_process(identity):
    try:
        actual = inventory.process_identity(identity['pid'])
        return all(actual.get(k) == identity.get(k) for k in
                   ('pid', 'uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd'))
    except (OSError, ValueError, KeyError):
        return False


def progress(study, split):
    _barrier(study, split)
    plan = read(_folder(study, split) / 'plan.json')
    completed = claimed = rows = tokens = 0
    for chunk in plan['chunks']:
        if _receipt(study, split, plan, chunk):
            completed += 1
            rows += chunk['rows']
            tokens += chunk['tokens']
        else:
            path = _chunk(study, split, chunk['chunk_id']) / 'claim.json'
            if path.exists():
                record = read(path)
                claimed += int(not record.get('released') and _same_live_process(record['identity']))
    return dict(split=split, complete=completed == len(plan['chunks']), total_chunks=len(plan['chunks']),
        complete_chunks=completed, claimed_chunks=claimed, pending_chunks=len(plan['chunks']) - completed - claimed,
        completed_rows=rows, total_rows=plan['rows'], completed_tokens=tokens, total_tokens=plan['total_tokens'])


def all_complete(study, split):
    return progress(study, split)['complete']


def release_dead_claims(study, split):
    _barrier(study, split)
    plan = read(_folder(study, split) / 'plan.json')
    released = []
    with _lock(_folder(study, split)):
        for chunk in plan['chunks']:
            directory = _chunk(study, split, chunk['chunk_id'])
            path = directory / 'claim.json'
            if _receipt(study, split, plan, chunk) or not path.exists():
                continue
            claim = read(path)
            if claim.get('released') or _same_live_process(claim['identity']):
                continue
            claim = dict(claim, released=True, release_reason='Exact process birth and command no longer exist',
                         released_unix=time.time())
            dump(directory / f'abandoned_attempt_{claim["attempt"]:03d}.json', claim)
            dump(path, claim)
            released.append(chunk['chunk_id'])
    return released


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
                continue
            attempt = 1 if previous is None else int(previous['attempt']) + 1
            target = directory / f'attempt_{attempt:03d}'
            target.mkdir(parents=True)
            result = dict(worker_id=worker_id, identity=inventory.process_identity(os.getpid()),
                chunk=chunk, attempt=attempt, directory=str(target.resolve()), created_unix=time.time(),
                plan_sha256=sha(_folder(study, split) / 'plan.json'))
            dump(path, result)
            return result
    return None


def extract_chunk(study, split, worker_id, claim, plan, data, tokenizer, capture, calibration_path, device='cuda'):
    """Write exactly one fresh attempt, publishing a receipt only after success."""
    target = Path(claim['directory'])
    if any(target.iterdir()):
        raise ValueError('Partial joint feature attempts cannot be reused')
    chunk = claim['chunk']
    groups = [np.asarray(g, np.int64) for g in plan['groups'][chunk['group_start']:chunk['group_stop']]]
    rows = np.concatenate(groups)
    lengths = np.asarray(data['lengths'][rows], np.int32)
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    last = np.lib.format.open_memmap(target / 'last.partial.npy', mode='w+', dtype=np.float16,
        shape=(len(rows), plan['num_layers'], plan['hidden_size']))
    tokens = np.lib.format.open_memmap(target / 'tokens.partial.npy', mode='w+', dtype=np.float16,
        shape=(int(offsets[-1]), plan['hidden_size']))
    row_start = token_start = 0
    begin = time.monotonic()
    for number, group in enumerate(groups):
        _barrier(study, split)
        actual_last, actual_tokens = _capture_group(capture, data, group, tokenizer, plan, device)
        last[row_start:row_start + len(group)] = actual_last
        tokens[token_start:token_start + len(actual_tokens)] = actual_tokens
        row_start += len(group)
        token_start += len(actual_tokens)
        dump(_folder(study, split) / 'workers' / f'{worker_id}.progress.json', dict(state='extracting',
            split=split, worker_id=worker_id, chunk_id=chunk['chunk_id'], completed_groups=number + 1,
            total_groups=len(groups), completed_chunk_rows=row_start, completed_chunk_tokens=token_start,
            updated_unix=time.time()))
    if row_start != len(rows) or token_start != chunk['tokens'] or token_start != len(tokens):
        raise ValueError('Joint chunk coverage differs from the fixed fresh plan')
    last.flush()
    tokens.flush()
    del last, tokens
    (target / 'last.partial.npy').replace(target / 'last.npy')
    (target / 'tokens.partial.npy').replace(target / 'tokens.npy')
    _save_array(target / 'rows.npy', rows)
    _save_array(target / 'lengths.npy', lengths)
    _save_array(target / 'offsets.npy', offsets)
    dump(target / 'row_identity.json', dict(base_ids=data['meta'].base_id.iloc[rows].tolist(),
        every_batch_same_forward_last_exact=True, every_batch_finite=True))
    result = dict(schema=SCHEMA, complete=True, split=split, chunk=chunk, worker_id=worker_id,
        attempt=claim['attempt'], identity=claim['identity'], protocol_sha256=plan['protocol_sha256'],
        plan_sha256=sha(_folder(study, split) / 'plan.json'),
        last_shape=[len(rows), plan['num_layers'], plan['hidden_size']],
        tokens_shape=[int(offsets[-1]), plan['hidden_size']],
        artifacts={name: dict(path=str((target / name).resolve()), sha256=sha(target / name)) for name in PAYLOAD_FILES},
        calibration_path=str(Path(calibration_path).resolve()), calibration_sha256=sha(calibration_path),
        elapsed_seconds=time.monotonic() - begin, completed_unix=time.time())
    kernel_path = getattr(capture, 'kernel_path', None)
    if kernel_path is not None:
        kernel_record = target / 'kernel_path.json'
        dump(kernel_record, kernel_path.snapshot())
        result['kernel_path_record'] = str(kernel_record.resolve())
        result['kernel_path_sha256'] = sha(kernel_record)
        dump(Path(study) / 'workers' / f'{worker_id}.kernels.json', kernel_path.snapshot())
    with _lock(_folder(study, split)):
        if read(_chunk(study, split, chunk['chunk_id']) / 'claim.json') != claim:
            raise ValueError('Joint extraction claim changed before receipt publication')
        _barrier(study, split)
        dump(_chunk(study, split, chunk['chunk_id']) / 'receipt.json', result)
    return result


def extract_worker(study, split, worker_id, device='cuda', *, admission_already_checked=True):
    study = Path(study).resolve()
    _barrier(study, split)
    _protocol(study)
    _admit(study, worker_id, device, admission_already_checked)
    plan, data = _prepared(study, split)
    model, backbone, tokenizer = _setup_model(study)
    kernel_path = runtime.DeterministicKernelPath(require_fla=read(study / 'protocol.json')['model_type'] == 'qwen3_5')
    capture = JointCapture(backbone, kernel_path)
    try:
        if not (study / 'calibration' / 'baseline.json').exists():
            raise ValueError('Independent fresh calibration must precede formal extraction')
        calibration_path = _calibration(study, worker_id, capture, tokenizer, device, compare_baseline=False)
        while True:
            _barrier(study, split)
            claim = claim_chunk(study, split, worker_id, plan)
            if claim is None:
                break
            result = extract_chunk(study, split, worker_id, claim, plan, data, tokenizer, capture,
                                   calibration_path, device=device)
            print(json.dumps(dict(state='joint_chunk_completed', split=split, worker_id=worker_id,
                chunk_id=claim['chunk']['chunk_id'], last_shape=result['last_shape'],
                tokens_shape=result['tokens_shape'])), flush=True)
        dump(_folder(study, split) / 'workers' / f'{worker_id}.complete.json',
             dict(complete=True, split=split, worker_id=worker_id, completed_unix=time.time()))
    finally:
        capture.close()
        del capture, model, backbone
        gc.collect()
        torch.cuda.empty_cache()


def _validate_chunks(study, split, plan, data, *, full_hash=True, check_values=False):
    seen = np.zeros(plan['rows'], dtype=bool)
    records = []
    baseline_path = Path(study) / 'calibration' / 'baseline.json'
    baseline = read(baseline_path)
    for chunk in plan['chunks']:
        receipt = _receipt(study, split, plan, chunk)
        if receipt is None:
            raise ValueError('All fresh joint feature chunks must complete before use')
        target = _chunk(study, split, chunk['chunk_id']) / f'attempt_{receipt["attempt"]:03d}'
        if set(receipt.get('artifacts', {})) != set(PAYLOAD_FILES):
            raise ValueError('Joint chunk artifact list differs')
        for name, artifact in receipt['artifacts'].items():
            path = _within(artifact['path'], target)
            if path.name != name or (full_hash and sha(path) != artifact['sha256']):
                raise ValueError(f'Committed joint artifact bytes changed: {name}')
        if receipt.get('kernel_path_record'):
            path = _within(receipt['kernel_path_record'], target)
            if sha(path) != receipt.get('kernel_path_sha256'):
                raise ValueError('Committed process-local kernel-path record changed')
        arrays = {name: np.load(receipt['artifacts'][name]['path'], mmap_mode='r', allow_pickle=False)
                  for name in ('rows.npy', 'lengths.npy', 'offsets.npy', 'last.npy', 'tokens.npy')}
        rows, lengths, offsets, last, tokens = [arrays[n] for n in
            ('rows.npy', 'lengths.npy', 'offsets.npy', 'last.npy', 'tokens.npy')]
        expected_rows = np.concatenate([np.asarray(g, np.int64) for g in plan['groups'][chunk['group_start']:chunk['group_stop']]])
        if (not np.array_equal(rows, expected_rows) or seen[rows].any()
                or not np.array_equal(lengths, data['lengths'][rows]) or offsets.shape != (len(rows) + 1,)
                or offsets[0] != 0 or not np.array_equal(np.diff(offsets), lengths)
                or last.shape != (len(rows), plan['num_layers'], plan['hidden_size'])
                or tokens.shape != (int(offsets[-1]), plan['hidden_size'])
                or last.dtype != np.float16 or tokens.dtype != np.float16
                or len(tokens) != chunk['tokens'] or list(last.shape) != receipt.get('last_shape')
                or list(tokens.shape) != receipt.get('tokens_shape')):
            raise ValueError('Joint chunk row/shape/length coverage differs')
        if not np.array_equal(last[:, -1], tokens[offsets[1:] - 1]):
            raise ValueError('Stored same-forward final tokens disagree')
        if check_values:
            for value in (last, tokens):
                for start in range(0, len(value), 8192):
                    if not np.isfinite(value[start:start + 8192]).all():
                        raise ValueError('Stored joint features are not finite')
        identity = read(receipt['artifacts']['row_identity.json']['path'])
        if (identity.get('base_ids') != data['meta'].base_id.iloc[rows].tolist()
                or identity.get('every_batch_same_forward_last_exact') is not True
                or identity.get('every_batch_finite') is not True):
            raise ValueError('Joint chunk identity/forward consistency receipt differs')
        calibration_path = _within(receipt['calibration_path'], Path(study) / 'calibration')
        if sha(calibration_path) != receipt['calibration_sha256']:
            raise ValueError('Fresh calibration record bytes changed')
        calibration = read(calibration_path)
        if (calibration.get('passed') is not True or calibration.get('worker_id') != receipt['worker_id']
                or calibration.get('protocol_sha256') != plan['protocol_sha256']
                or calibration.get('records') != baseline.get('records')
                or calibration.get('train_plan_sha256') != baseline.get('train_plan_sha256')):
            raise ValueError('Fresh cross-worker calibration signature differs')
        if calibration.get('kernel_path_record'):
            path = _within(calibration['kernel_path_record'], Path(study) / 'calibration')
            if sha(path) != calibration.get('kernel_path_sha256_at_calibration'):
                raise ValueError('Immutable calibration kernel-path record changed')
        seen[rows] = True
        path = _chunk(study, split, chunk['chunk_id']) / 'receipt.json'
        records.append(dict(chunk_id=chunk['chunk_id'], receipt_path=str(path.resolve()), receipt_sha256=sha(path),
                            rows=len(rows), tokens=len(tokens), artifacts=receipt['artifacts']))
    if not seen.all():
        raise ValueError('Joint feature cache omitted rows')
    return records


def completion_manifest_finalize(study, split):
    plan, data = _prepared(study, split)
    records = _validate_chunks(study, split, plan, data, check_values=True)
    result = dict(schema=SCHEMA, complete=True, split=split, protocol_sha256=plan['protocol_sha256'],
        plan_sha256=sha(_folder(study, split) / 'plan.json'),
        last_shape=[plan['rows'], plan['num_layers'], plan['hidden_size']],
        tokens_shape=[plan['total_tokens'], plan['hidden_size']], rows=plan['rows'],
        total_tokens=plan['total_tokens'], dtype='float16', adapter_sha256=None,
        raw_block_outputs=True, truncation=False, joint_same_forward=True, chunks=records,
        ordered_base_id_sha256=plan['ordered_base_id_sha256'],
        calibration_baseline_sha256=sha(Path(study) / 'calibration' / 'baseline.json'),
        test_selection_lock_sha256=plan['test_selection_lock_sha256'], completed_unix=time.time())
    path = _folder(study, split) / 'manifest.json'
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != 'completed_unix'} != {k: v for k, v in result.items() if k != 'completed_unix'}:
            raise ValueError('Published joint feature manifest changed')
        return previous
    dump(path, result)
    return result


finalize = completion_manifest_finalize


class LastTokenRows:
    def __init__(self, cache, layers):
        self.cache = cache
        self.shape = (len(cache), layers, cache.hidden_size)
        self.dtype = np.dtype('float16')

    def __getitem__(self, key):
        row_key, suffix = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        indices = np.arange(len(self.cache), dtype=np.int64)[row_key]
        if np.ndim(indices) == 0:
            chunk, local = self.cache._where[int(indices)]
            selected = self.cache._chunks[chunk][2][local]
            return selected[suffix] if suffix else selected
        indices = np.asarray(indices)
        # The trainer's common [rows, single_layer] access reads only that
        # layer from each mmap instead of copying all 32 blocks each step.
        if suffix and all(isinstance(value, (int, np.integer, slice)) for value in suffix):
            sample = np.empty(self.shape[1:], dtype=np.float16)[suffix]
            selected = np.empty(indices.shape + sample.shape, dtype=np.float16)
            for position in np.ndindex(indices.shape):
                chunk, local = self.cache._where[int(indices[position])]
                selected[position] = self.cache._chunks[chunk][2][local][suffix]
            return selected
        selected = np.empty(indices.shape + self.shape[1:], dtype=np.float16)
        for position in np.ndindex(indices.shape):
            chunk, local = self.cache._where[int(indices[position])]
            selected[position] = self.cache._chunks[chunk][2][local]
        return selected[(slice(None),) * indices.ndim + suffix] if suffix else selected


class JointCache:
    def __init__(self, plan, data, manifest, manifest_path):
        self.meta = self.metadata = data['meta']
        self.lengths = np.asarray(data['lengths'])
        self.hidden_size = plan['hidden_size']
        self.manifest, self.manifest_path = manifest, Path(manifest_path)
        self._chunks, self._where = [], np.empty((plan['rows'], 2), np.int64)
        for number, record in enumerate(manifest['chunks']):
            artifacts = record['artifacts']
            tokens = np.load(artifacts['tokens.npy']['path'], mmap_mode='r', allow_pickle=False)
            offsets = np.load(artifacts['offsets.npy']['path'], mmap_mode='r', allow_pickle=False)
            last = np.load(artifacts['last.npy']['path'], mmap_mode='r', allow_pickle=False)
            rows = np.load(artifacts['rows.npy']['path'], mmap_mode='r', allow_pickle=False)
            self._chunks.append((tokens, offsets, last))
            self._where[rows, 0], self._where[rows, 1] = number, np.arange(len(rows))
        self.last = LastTokenRows(self, plan['num_layers'])
        self.tokens = self

    def __len__(self):
        return len(self.lengths)

    def get_row(self, row):
        row = int(row)
        if row < 0 or row >= len(self):
            raise IndexError(row)
        chunk, local = self._where[row]
        tokens, offsets, _ = self._chunks[chunk]
        return tokens[offsets[local]:offsets[local + 1]]

    def padded_batch(self, rows, *, device='cpu', dtype=torch.float32):
        rows = np.asarray(rows, np.int64)
        if rows.ndim != 1 or not len(rows):
            raise ValueError('A nonempty flat row-index batch is required')
        if (rows < 0).any() or (rows >= len(self)).any():
            raise IndexError('Row index out of range')
        lengths = self.lengths[rows]
        values = torch.zeros((len(rows), int(lengths.max()), self.hidden_size), dtype=dtype)
        mask = torch.zeros((len(rows), int(lengths.max())), dtype=torch.bool)
        for index, row in enumerate(rows):
            value = self.get_row(row)
            values[index, :len(value)] = torch.as_tensor(np.array(value, copy=True), dtype=dtype)
            mask[index, :len(value)] = True
        return values.to(device), mask.to(device)


def load(study, split):
    plan, data = _prepared(study, split)
    path = _folder(study, split) / 'manifest.json'
    if not path.exists():
        raise ValueError('Fresh joint feature completion manifest is absent')
    manifest = read(path)
    records = _validate_chunks(study, split, plan, data)
    if (manifest.get('schema') != SCHEMA or manifest.get('complete') is not True
            or manifest.get('protocol_sha256') != plan['protocol_sha256']
            or manifest.get('plan_sha256') != sha(_folder(study, split) / 'plan.json')
            or manifest.get('chunks') != records or manifest.get('joint_same_forward') is not True
            or manifest.get('last_shape') != [plan['rows'], plan['num_layers'], plan['hidden_size']]
            or manifest.get('tokens_shape') != [plan['total_tokens'], plan['hidden_size']]
            or manifest.get('ordered_base_id_sha256') != plan['ordered_base_id_sha256']
            or manifest.get('calibration_baseline_sha256') != sha(Path(study) / 'calibration' / 'baseline.json')
            or manifest.get('test_selection_lock_sha256') != plan['test_selection_lock_sha256']):
        raise ValueError('Fresh joint feature manifest identity changed')
    return JointCache(plan, data, manifest, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'calibrate', 'extract', 'progress', 'finalize', 'release-dead'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--worker-id')
    args = parser.parse_args()
    if args.command in ('calibrate', 'extract'):
        if not args.worker_id:
            parser.error('--worker-id is required')
        result = run_calibration(args.study, args.worker_id) if args.command == 'calibrate' else extract_worker(args.study, args.split, args.worker_id)
    else:
        fn = {'prepare': prepare, 'progress': progress, 'finalize': completion_manifest_finalize,
              'release-dead': release_dead_claims}[args.command]
        result = fn(args.study, args.split)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
