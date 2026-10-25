"""Fresh, canonical singleton Qwen3Guard features for supervised controls.

The backbone sees the complete official guard template.  Token pooling reads
only tokens overlapping the original user text; last-token controls read the
last valid token of the complete template.  Both are raw final-block outputs,
before the backbone's final RMSNorm, captured during the same forward call.
This module deliberately imports torch/transformers only inside GPU routines.
Every source row is forwarded alone, with no padding or batch-dependent GEMM
shape.  Failed batched v1 features are never inputs to this revision.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

SCHEMA = 'qwen3guard_supervised_content_features_v2_singleton'
SPLITS = ('train', 'validation', 'test')
META_COLUMNS = ('base_id', 'group_id', 'source', 'prompt_harm_label',
                'category_id', 'type_label_mask', 'split')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def _protocol(output):
    output = Path(output).resolve()
    if (output / 'STOP').exists():
        raise RuntimeError('Study STOP file exists')
    protocol = read(output / 'protocol.json')
    extraction = protocol.get('extraction', {})
    if extraction.get('canonical_singleton') is not True or extraction.get('max_micro') != 1:
        raise ValueError('V2 requires registered canonical singleton extraction')
    return protocol, sha(output / 'protocol.json')


def _barrier(output, split):
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split}')
    protocol, digest = _protocol(output)
    if split == 'test':
        lock = read(Path(output) / 'selection_lock.json')
        if lock.get('protocol_sha256') != digest:
            raise ValueError('Test features require a selection lock for this protocol')
    return protocol, digest


def verify_test_lock(output):
    """Verify all selected heads without importing training or torch."""
    output = Path(output).resolve()
    protocol, protocol_sha = _barrier(output, 'test')
    lock = read(output / 'selection_lock.json')
    expected_jobs = {f'{task}/s{seed}': (task, seed)
                     for task in protocol['tasks'] for seed in protocol['seeds']}
    if not lock.get('complete') or set(lock.get('selections', {})) != set(expected_jobs):
        raise ValueError('Global test selections are incomplete')
    expected_artifacts = {}
    for name, (task, seed) in expected_jobs.items():
        folder = output / 'jobs' / task / f's{seed}'
        path = folder / 'selection.json'
        digest = sha(path)
        if lock['selections'][name] != digest:
            raise ValueError('Global test selection digest changed')
        selection = read(path)
        if (selection.get('protocol_sha256') != protocol_sha or selection.get('task') != task
                or selection.get('seed') != seed
                or set(selection.get('arms', {})) != set(protocol['arms'])):
            raise ValueError('Global test selection identity changed')
        expected_artifacts[str(path)] = digest
        for filename in ('selected_heads.pt', 'validation_predictions.parquet'):
            artifact = folder / filename
            artifact_sha = selection['artifacts'].get(str(artifact))
            if not artifact_sha or sha(artifact) != artifact_sha:
                raise ValueError('Global selected validation artifact changed')
        expected_artifacts.update(selection['artifacts'])
    if lock.get('artifacts') != expected_artifacts:
        raise ValueError('Global test artifact registration is incomplete')
    for path, digest in expected_artifacts.items():
        if not Path(path).resolve().is_relative_to(output / 'jobs') or sha(path) != digest:
            raise ValueError('Global test artifact changed or escaped study jobs')
    return lock


@contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def template_parts(tokenizer):
    """Find the exact insertion span without searching for the real text."""
    marker = '__HANGUARD_USER_CONTENT_7f4d1d4587d04ce390102ca076d81370__'
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': marker}], tokenize=False)
    if rendered.count(marker) != 1:
        raise ValueError('Official template must contain the sentinel exactly once')
    return rendered.split(marker)


def encode_text(tokenizer, text, prefix, suffix, max_tokens=32768):
    if not isinstance(text, str) or not text:
        raise ValueError('User text must be a non-empty string')
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False)
    if rendered != prefix + text + suffix:
        raise ValueError('Official template is not a fixed single-user insertion')
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if not ids or len(ids) > max_tokens or len(ids) != len(offsets):
        raise ValueError('Empty or over-budget input; truncation is forbidden')
    char_start, char_end = len(prefix), len(prefix) + len(text)
    positions = [i for i, (start, end) in enumerate(offsets)
                 if end > char_start and start < char_end and end > start]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError('Original user text must map to a nonempty contiguous token span')
    # Byte-level tokens may share character offsets.  Their union must cover all
    # source characters, including whitespace, without relying on decode cleanup.
    cursor = char_start
    for position in positions:
        start, end = offsets[position]
        if start > cursor:
            raise ValueError('Tokenizer offsets leave a gap in original user text')
        cursor = max(cursor, end)
    if offsets[positions[0]][0] > char_start or cursor < char_end:
        raise ValueError('Tokenizer offsets do not cover the full original user text')
    return ids, positions[0], positions[-1] + 1


def _input_path(protocol, split):
    folder = Path(protocol['parent_data_dir']).resolve()
    path = folder / f'{split}.parquet'
    expected = protocol.get('data_sha256', {}).get(split)
    if not expected or sha(path) != expected:
        raise ValueError(f'Registered raw data changed: {split}')
    return path


def prepare(output, split):
    """Fresh CPU tokenization, immutable row-aligned token/shard registration."""
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    output = Path(output).resolve()
    protocol, protocol_sha = _barrier(output, split)
    if split == 'test':
        verify_test_lock(output)
    folder = output / 'features' / split
    folder.mkdir(parents=True, exist_ok=True)
    with _lock(folder / '.prepare.lock'):
        if (folder / 'manifest.json').exists():
            return manifest(output, split, verify=True)
        if (folder / 'prepared').exists():
            raise ValueError('Uncommitted prepared files exist; refusing partial reuse')
        path = _input_path(protocol, split)
        frame = pd.read_parquet(path, columns=['prompt', *META_COLUMNS]).reset_index(drop=True)
        if len(frame) == 0 or not frame['base_id'].is_unique or not frame['split'].eq(split).all():
            raise ValueError('Invalid raw identity or split')
        tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
        if not tokenizer.is_fast:
            raise ValueError('A fast tokenizer with original-character offsets is required')
        prefix, suffix = template_parts(tokenizer)
        extraction = protocol.get('extraction', {})
        maximum = int(extraction.get('max_tokens', 32768))
        ids, row_offsets, starts, ends, lengths = [], [0], [], [], []
        for row, text in enumerate(frame['prompt']):
            values, start, end = encode_text(tokenizer, text, prefix, suffix, maximum)
            ids.extend(values)
            row_offsets.append(len(ids))
            starts.append(start)
            ends.append(end)
            lengths.append(len(values))
            if row % 1000 == 0 or row + 1 == len(frame):
                write(folder / 'preparation_progress.json', dict(rows_done=row + 1,
                    rows=len(frame), split=split, input_tokens=len(ids), updated_unix=time.time()))
        content_lengths = np.asarray(ends, dtype=np.int32) - np.asarray(starts, dtype=np.int32)
        shard_rows = int(protocol.get('shard_rows', 512))
        if shard_rows < 1:
            raise ValueError('shard_rows must be positive')
        shards = []
        for start in range(0, len(frame), shard_rows):
            end = min(start + shard_rows, len(frame))
            shards.append(dict(shard_index=len(shards), row_start=start, row_end=end,
                               content_tokens=int(content_lengths[start:end].sum())))
        tmp = folder / f'.prepared.{os.getpid()}.{uuid.uuid4().hex}'
        tmp.mkdir()
        try:
            for name, values in [('token_ids', np.asarray(ids, dtype=np.int32)),
                                 ('row_offsets', np.asarray(row_offsets, dtype=np.int64)),
                                 ('user_starts', np.asarray(starts, dtype=np.int32)),
                                 ('user_ends', np.asarray(ends, dtype=np.int32)),
                                 ('lengths', np.asarray(lengths, dtype=np.int32)),
                                 ('content_lengths', content_lengths)]:
                np.save(tmp / f'{name}.npy', values, allow_pickle=False)
            frame[list(META_COLUMNS)].to_parquet(tmp / 'metadata.parquet', index=False)
            files = {f'prepared/{p.name}': sha(p) for p in sorted(tmp.iterdir())}
            result = dict(schema=SCHEMA, split=split, rows=len(frame),
                          hidden_size=int(protocol.get('hidden_size', 2560)),
                          model_path=str(Path(protocol['model_path']).resolve()),
                          protocol_sha256=protocol_sha, raw_data_sha256=sha(path),
                          metadata_sha256=files['prepared/metadata.parquet'], files=files,
                          shards=shards, input_tokens=int(row_offsets[-1]),
                          content_tokens=int(content_lengths.sum()), max_input_tokens=max(lengths),
                          template_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                          template_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                          content_selection='offset overlap with original user characters',
                          representation='raw last decoder block before final RMSNorm',
                          padding_side='right', truncation=False,
                          canonical_singleton=True,
                          boundary_token_policy='include every token overlapping user text',
                          created_unix=time.time())
            os.replace(tmp, folder / 'prepared')
            write(folder / 'manifest.json', result)
            return result
        except BaseException:
            if tmp.exists():
                shutil.rmtree(tmp)
            raise


def manifest(output, split, verify=False):
    _, protocol_sha = _barrier(output, split)
    folder = Path(output) / 'features' / split
    result = read(folder / 'manifest.json')
    if result.get('schema') != SCHEMA or result.get('protocol_sha256') != protocol_sha:
        raise ValueError('Feature manifest schema/protocol mismatch')
    if verify:
        for name, digest in result['files'].items():
            if sha(folder / name) != digest:
                raise ValueError(f'Prepared feature input changed: {name}')
    return result


def length_batches(rows, lengths, max_micro=8, token_budget=4096):
    if max_micro < 1 or token_budget < 1:
        raise ValueError('Invalid batch budget')
    for row in map(int, rows):
        if int(lengths[row]) < 1:
            raise ValueError('Invalid input token length')
        yield [row]


def load_model(output, device='cuda:0'):
    import torch
    from transformers import AutoModel
    protocol, _ = _protocol(output)
    torch.set_num_threads(int(protocol.get('threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = AutoModel.from_pretrained(protocol['model_path'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').eval().requires_grad_(False).to(device)
    if model.config.hidden_size != int(protocol.get('hidden_size', 2560)):
        raise ValueError('Model hidden width does not match registered cache width')
    if len(model.layers) != int(protocol.get('num_hidden_layers', 36)):
        raise ValueError('Model block count differs from registration')
    return model


def forward_rows(model, sequences, starts, ends, device='cuda:0', pad_token_id=151643):
    """Return the raw block states, never normalized model.last_hidden_state."""
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not sequences or len(sequences) != len(starts) or len(starts) != len(ends):
        raise ValueError('Invalid batch dimensions')
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), int(pad_token_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, values in enumerate(sequences):
        if not 0 <= int(starts[i]) < int(ends[i]) <= len(values):
            raise ValueError('Invalid content token span')
        ids[i, :len(values)] = torch.as_tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
    captured = []
    def hook(_module, _args, result):
        captured.append(result[0] if isinstance(result, (tuple, list)) else result)
    handle = model.layers[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        if len(captured) != 1 or tuple(captured[0].shape[:2]) != tuple(ids.shape):
            raise ValueError('Expected one final-block capture with exact padded batch shape')
        raw = captured.pop()
        content, last = [], []
        for i, values in enumerate(sequences):
            content.append(raw[i, int(starts[i]):int(ends[i])].cpu().float().numpy().astype(np.float16))
            last.append(raw[i, len(values) - 1].cpu().float().numpy().astype(np.float16))
        return content, np.asarray(last, dtype=np.float16)
    finally:
        handle.remove()


def _calibrate(model, sequences, starts, ends, device, pad_token_id):
    import numpy as np
    if len(sequences) != 1:
        raise ValueError('V2 calibration requires the canonical singleton input')
    batch_tokens, batch_last = forward_rows(model, sequences, starts, ends, device, pad_token_id)
    repeated_tokens, repeated_last = forward_rows(model, sequences, starts, ends, device, pad_token_id)
    if (not np.array_equal(batch_last, repeated_last)
            or any(not np.array_equal(a, b) for a, b in zip(batch_tokens, repeated_tokens))):
        raise ValueError('Identical canonical singleton extraction is not deterministic')
    if (not np.isfinite(batch_last).all()
            or any(not np.isfinite(values).all() for values in batch_tokens)):
        raise ValueError('Nonfinite canonical singleton calibration features')
    return [dict(batch_index=0, relative_rmse=0., cosine=1., max_abs=0.,
                 canonical_singleton=True, repeated_padded_batch_bitwise_equal=True,
                 repeated_canonical_input_bitwise_equal=True)]


def _valid_receipt(folder, expected_manifest_sha, descriptor, verify=False):
    receipt_path = folder / 'receipt.json'
    if not receipt_path.exists():
        return None
    result = read(receipt_path)
    if (result.get('manifest_sha256') != expected_manifest_sha
            or result.get('row_start') != descriptor['row_start']
            or result.get('row_end') != descriptor['row_end']):
        raise ValueError('Feature shard receipt identity mismatch')
    for name, record in result['files'].items():
        path = folder / name
        stat = path.stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'Feature shard file stat changed: {path}')
        if verify and sha(path) != record['sha256']:
            raise ValueError(f'Feature shard file hash changed: {path}')
    return result


def complete_jobs(output, split, verify=False):
    folder = Path(output) / 'features' / split
    if not (folder / 'manifest.json').exists():
        return []
    registration = manifest(output, split)
    digest = sha(folder / 'manifest.json')
    return [item['shard_index'] for item in registration['shards']
            if _valid_receipt(folder / f"shard_{item['shard_index']:05d}", digest, item, verify)]


def pending_jobs(output, split):
    completed = set(complete_jobs(output, split))
    return [item['shard_index'] for item in manifest(output, split)['shards']
            if item['shard_index'] not in completed]


def extract(output, split, shard, device='cuda:0', model=None, _test_lock_verified=False):
    import numpy as np
    import pandas as pd
    import torch
    output = Path(output).resolve()
    protocol, protocol_sha = _barrier(output, split)
    if split == 'test' and not _test_lock_verified:
        verify_test_lock(output)
    registration = manifest(output, split, verify=True)
    folder = output / 'features' / split
    registration_sha = sha(folder / 'manifest.json')
    descriptor = registration['shards'][int(shard)]
    if descriptor['shard_index'] != int(shard):
        raise ValueError('Invalid shard descriptor index')
    target = folder / f'shard_{int(shard):05d}'
    with _lock(folder / f'.shard_{int(shard):05d}.lock'):
        previous = _valid_receipt(target, registration_sha, descriptor, verify=True)
        if previous:
            return previous
        if target.exists():
            raise ValueError('Uncommitted feature shard exists; refusing partial reuse')
        prepared = folder / 'prepared'
        arrays = {name: np.load(prepared / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('token_ids', 'row_offsets', 'user_starts', 'user_ends', 'lengths', 'content_lengths')}
        first, stop = descriptor['row_start'], descriptor['row_end']
        lengths = arrays['lengths']
        selected_lengths = arrays['content_lengths'][first:stop]
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(selected_lengths, dtype=np.int64)])
        tmp = folder / f'.shard_{int(shard):05d}.{os.getpid()}.{uuid.uuid4().hex}'
        tmp.mkdir()
        try:
            tokens = np.lib.format.open_memmap(tmp / 'token_features.npy', mode='w+', dtype=np.float16,
                shape=(int(offsets[-1]), registration['hidden_size']))
            last = np.lib.format.open_memmap(tmp / 'last.npy', mode='w+', dtype=np.float16,
                shape=(stop - first, registration['hidden_size']))
            np.save(tmp / 'offsets.npy', offsets, allow_pickle=False)
            if model is None:
                model = load_model(output, device)
            extraction = protocol.get('extraction', {})
            batches = list(length_batches(range(first, stop), lengths,
                int(extraction.get('max_micro', 8)), int(extraction.get('token_budget', 4096))))
            calibrations, oom_splits = [], []
            def run(rows):
                nonlocal calibrations
                sequences = [arrays['token_ids'][arrays['row_offsets'][i]:arrays['row_offsets'][i + 1]].tolist() for i in rows]
                starts = [int(arrays['user_starts'][i]) for i in rows]
                ends = [int(arrays['user_ends'][i]) for i in rows]
                retry_after_oom = False
                try:
                    result, final = forward_rows(model, sequences, starts, ends, device,
                                                 int(protocol.get('pad_token_id', 151643)))
                    if not calibrations:
                        calibrations = _calibrate(model, sequences, starts, ends, device,
                                                 int(protocol.get('pad_token_id', 151643)))
                    for index, row in enumerate(rows):
                        local = row - first
                        if result[index].shape != (int(selected_lengths[local]), registration['hidden_size']):
                            raise ValueError('Captured content feature shape mismatch')
                        if not np.isfinite(result[index]).all() or not np.isfinite(final[index]).all():
                            raise ValueError('Nonfinite feature tensor')
                        tokens[offsets[local]:offsets[local + 1]] = result[index]
                        last[local] = final[index]
                except torch.cuda.OutOfMemoryError:
                    if len(rows) == 1:
                        raise
                    retry_after_oom = True
                # Leave the exception scope before retrying: its traceback can
                # otherwise keep failed-forward CUDA tensors alive recursively.
                if retry_after_oom:
                    torch.cuda.empty_cache()
                    midpoint = len(rows) // 2
                    oom_splits.append(rows)
                    run(rows[:midpoint])
                    run(rows[midpoint:])
            for batch_index, rows in enumerate(batches):
                _barrier(output, split)
                run(rows)
                write(folder / f'shard_{int(shard):05d}.progress.json', dict(
                    shard_index=int(shard), batches_done=batch_index + 1, batches=len(batches),
                    rows_done=sum(map(len, batches[:batch_index + 1])), rows=stop - first, updated_unix=time.time()))
            tokens.flush()
            last.flush()
            del tokens, last
            metadata_frame = pd.read_parquet(prepared / 'metadata.parquet').iloc[first:stop]
            metadata = dict(schema=SCHEMA, **descriptor, split=split, hidden_size=registration['hidden_size'],
                            base_ids=metadata_frame['base_id'].tolist(),
                            protocol_sha256=protocol_sha, manifest_sha256=registration_sha,
                            calibration=calibrations, oom_splits=oom_splits,
                            input_batches=batches, representation=registration['representation'],
                            created_unix=time.time())
            write(tmp / 'metadata.json', metadata)
            files = {}
            for path in sorted(tmp.iterdir()):
                stat = path.stat()
                files[path.name] = dict(sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            receipt = dict(schema=SCHEMA, split=split, **descriptor, files=files,
                           protocol_sha256=protocol_sha, manifest_sha256=registration_sha,
                           completed_unix=time.time())
            write(tmp / 'receipt.json', receipt)
            os.replace(tmp, target)
            return receipt
        except BaseException:
            # Preserve failed partials for diagnosis, but never count or reuse them.
            if tmp.exists():
                write(tmp / 'failed.json', dict(failed_unix=time.time(), pid=os.getpid()))
            raise


def extract_many(output, split, shards, device='cuda:0'):
    _barrier(output, split)
    if split == 'test':
        verify_test_lock(output)
    model = load_model(output, device)
    return [extract(output, split, int(shard), device, model=model,
                    _test_lock_verified=split == 'test') for shard in shards]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'extract', 'pending', 'verify'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', required=True, choices=SPLITS)
    parser.add_argument('--shards', type=int, nargs='+')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.output, args.split)
    elif args.command == 'extract':
        if not args.shards:
            parser.error('extract requires --shards')
        result = extract_many(args.output, args.split, args.shards, args.device)
    elif args.command == 'pending':
        result = pending_jobs(args.output, args.split)
    else:
        result = complete_jobs(args.output, args.split, verify=True)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
