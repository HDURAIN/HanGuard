"""Fresh full-data frozen-base readouts, with no adapter and a test barrier.

Only ``init`` uses CPU. ``run`` must be launched by the identity-bound GPU
guardian. Train/validation features are regenerated; test payloads remain
unread until every readout and its validation threshold have been locked.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import remediated_dataset_contract as contract
from scripts.hanguard import repaired_study as legacy
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    dump, last_true_indices, ordered_base_id_sha256, validate_feature_cache,
)

SCHEMA = 'hanguard_fresh_full_data_frozen_base_v1'
SPLITS = ('train', 'validation', 'test')
SOURCE_NAMES = (
    'scripts/hanguard/frozen_base_readout_study.py',
    'scripts/hanguard/frozen_base_readout_heads.py',
    'scripts/hanguard/frozen_base_readout_queue.py',
    'scripts/hanguard/repaired_heads.py',
    'scripts/hanguard/repaired_study.py',
    'scripts/hanguard/remediated_dataset_contract.py',
    'scripts/hanguard/frozen_lora_layer_diagnostics.py',
    'scripts/hanguard/ab_frozen_readout_diagnostics.py',
    'scripts/hanguard/sparse_token_readout_queue.py',
    'scripts/hanguard/ab_diagnostic_queue.py',
    'scripts/hanguard/ab_readout_parallel_queue.py',
    'hanguard_model.py',
)


def check_stop(study):
    for directory in (Path(study), Path(study).parent):
        p = directory / 'STOP'
        if p.exists():
            raise RuntimeError(f'Fresh frozen-base run stopped: {p.read_text().strip()}')


def sha(path, study=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            if study is not None:
                check_stop(study)
            digest.update(block)
    return digest.hexdigest()


def runtime_identity():
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'peft', 'triton',
                 'causal-conv1d', 'flash-linear-attention', 'numpy', 'pandas'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=str(Path(sys.executable).absolute()), prefix=sys.prefix,
                versions=versions, torch_cuda=torch.version.cuda)


def source_identity():
    return {str(ROOT / name): sha(ROOT / name) for name in SOURCE_NAMES}


def validate_metadata(frame, split):
    required = {'base_id', 'group_id', 'source', 'prompt_harm_label', 'split',
                'category_id', 'type_label_mask'}
    if not required <= set(frame):
        raise ValueError(f'{split}: missing fields {required - set(frame)}')
    if not len(frame) or not frame.base_id.is_unique or frame.base_id.isna().any():
        raise ValueError(f'{split}: empty or repeated sample identity')
    if frame.group_id.isna().any() or frame.group_id.astype(str).eq('').any():
        raise ValueError(f'{split}: missing group identity')
    if not frame.split.eq(split).all():
        raise ValueError(f'{split}: mixed split membership')
    contract.label_arrays(frame, require_type_mask=True)


def initialize(args):
    study = Path(args.output).resolve()
    if study.exists() and any(study.iterdir()):
        raise FileExistsError('Use a fresh study directory; prior runs stay immutable')
    model, data = Path(args.model).resolve(), Path(args.data).resolve()
    release = contract.release_contract(data)
    if not release:
        raise ValueError('This fresh run requires a completed remediated release')
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    depth, hidden = int(text['num_hidden_layers']), int(text['hidden_size'])
    if depth != 32:
        raise ValueError('This registered comparison requires 32 blocks; revise explicitly for another architecture')
    index = model / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    model_files = [model / name for name in shards]
    model_files += [p for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')]
    model_files = sorted(set(model_files))
    inputs = {str(p): sha(p) for p in model_files}
    inputs.update({str(data / f'{split}.parquet'): sha(data / f'{split}.parquet') for split in SPLITS})
    for name in ('manifest.json', 'release_qa.json', 'split_policy.json'):
        inputs[str(data / name)] = sha(data / name)
    metadata_columns = ['base_id', 'group_id', 'source', 'prompt_harm_label',
                        'split', 'category_id', 'type_label_mask']
    frames = {s: pd.read_parquet(data / f'{s}.parquet', columns=metadata_columns)
              for s in ('train', 'validation')}
    for split, frame in frames.items():
        validate_metadata(frame, split)
    # Test identities and file bytes may be bound before the barrier; no prompt
    # or label payloads from that split are read here.
    test_ids = pd.read_parquet(data / 'test.parquet', columns=['base_id', 'group_id'])
    if not test_ids.base_id.is_unique or test_ids.isna().any().any():
        raise ValueError('Test identity metadata is invalid')
    identity_frames = dict(frames, test=test_ids)
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        for field in ('base_id', 'group_id'):
            if set(identity_frames[a][field]) & set(identity_frames[b][field]):
                raise ValueError(f'{a}/{b}: {field} overlap')
    rows = {s: pq.ParquetFile(data / f'{s}.parquet').metadata.num_rows for s in SPLITS}
    bytes_required = sum(rows.values()) * depth * hidden * 2
    if shutil.disk_usage(study.parent).free < bytes_required + 8 * 1024**3:
        raise ValueError('Insufficient local disk for fresh feature caches and checkpoints')
    value = dict(
        schema=SCHEMA, model_path=str(model), data_dir=str(data),
        dataset_release=release, model_type=config['model_type'],
        num_layers=depth, hidden_size=hidden, rows=rows,
        inputs=inputs,
        input_stats={p: dict(size=Path(p).stat().st_size,
                            mtime_ns=Path(p).stat().st_mtime_ns) for p in inputs},
        code_sha256=source_identity(), runtime=runtime_identity(),
        settings=dict(seed=42, width=128, dropout=.1, batch_size=256, lr=.001,
                      weight_decay=.001, grad_clip=5., epochs=12, max_epochs=40,
                      layers=[8, 16, 24, 32], bootstrap_draws=2000),
        extraction=dict(token_budget=16384, max_micro=64, pad_multiple=32,
                        max_tokens=4096, pooling='last_valid_token',
                        raw_block_outputs=True, dtype='float16', truncation=False),
        adapter_path=None, adapter_sha256=None, use_lora=False,
        backbone_trainable_parameters=0, fresh_features_required=True,
        methods=['last', 'best_single', 'fusion_fixed', 'repeat_best', 'fusion_fixed_best_anchor'],
        best_single_definition='Train 32 independent same-capacity MLPs; select layer/checkpoint by validation BCE only',
        checkpoint_selection='minimum validation BCE independently per layer/arm',
        threshold_selection='validation maximum accuracy; fixed 0.5 also reported',
        evaluation_scope='Fresh full train/validation readout fitting, one locked current-remediated test evaluation',
        formal_test_enabled=True, formal_test_barrier_required=True,
        no_test_layer_or_checkpoint_selection=True,
        execution=dict(single_identity_bound_gpu_worker=True,
            cuda_startup='Verify guardian child/card and establish CUDA readiness before slow hashing/imports',
            full_content_hash_required_before_model_or_data_use=True,
            incomplete_attempt_reuse=False),
        expected_cache_bytes_without_headers=bytes_required,
        limitations=[
            'Prior dataset identities were explored and regrouped; this is not pristine external blind confirmation.',
            'One fixed head seed measures this run, not cross-seed reproducibility.',
            'Selected-layer and multi-layer differences must be reported separately from capacity effects.',
            'Full-train readouts cannot be directly substituted for previous B-subset OOF scores.',
        ], created_unix=time.time(),
    )
    study.mkdir(parents=True, exist_ok=True)
    dump(study / 'protocol.json', value)
    snapshots = study / 'source_snapshot'
    snapshots.mkdir()
    for filename, digest in value['code_sha256'].items():
        target = snapshots / str(Path(filename).relative_to(ROOT)).replace('/', '__')
        shutil.copy2(filename, target)
        if sha(target) != digest:
            raise ValueError('Source changed while snapshotting')
    dump(study / 'protocol_registration.json', dict(
        protocol_sha256=sha(study / 'protocol.json'),
        source_snapshot_sha256={str(p.relative_to(study)): sha(p) for p in snapshots.iterdir()},
    ))
    dump(study / 'state.json', dict(state='registered', training_started=False,
        backbone_training_started=False, formal_validation_opened=False,
        formal_test_opened=False, protocol_sha256=sha(study / 'protocol.json')))
    return value


def load_protocol(study, *, full_hash=False):
    study = Path(study).resolve()
    check_stop(study)
    p = json.loads((study / 'protocol.json').read_text())
    registration = json.loads((study / 'protocol_registration.json').read_text())
    if p.get('schema') != SCHEMA or registration['protocol_sha256'] != sha(study / 'protocol.json'):
        raise ValueError('Registered frozen-base protocol changed')
    if p.get('use_lora') is not False or p.get('adapter_path') is not None:
        raise ValueError('This study permits no adapter')
    if runtime_identity() != p['runtime']:
        raise ValueError('Pinned Python/library runtime changed')
    for filename, digest in p['code_sha256'].items():
        if sha(filename) != digest:
            raise ValueError(f'Registered implementation changed: {filename}')
    for filename, digest in p['inputs'].items():
        path = Path(filename)
        actual_stat = dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        if actual_stat != p['input_stats'][filename]:
            raise ValueError(f'Registered input metadata changed: {filename}')
        if full_hash and sha(path, study) != digest:
            raise ValueError(f'Registered input bytes changed: {filename}')
    return p


def validate_selection_lock(study):
    study = Path(study)
    path = study / 'selection_lock.json'
    if not path.exists():
        raise ValueError('All validation selections must be locked before reading test payloads')
    lock = json.loads(path.read_text())
    if (lock.get('complete') is not True or lock.get('test_opened') is not False
            or lock.get('protocol_sha256') != sha(study / 'protocol.json')):
        raise ValueError('Incomplete or mismatched test-selection lock')
    if not lock.get('artifacts'):
        raise ValueError('Selection lock must bind actual selected model artifacts')
    protocol = json.loads((study / 'protocol.json').read_text())
    allowed_external = dict(protocol.get('code_sha256', {}), **protocol.get('inputs', {}))
    for filename, digest in lock['artifacts'].items():
        artifact = Path(filename)
        if not artifact.is_absolute():
            artifact = study / artifact
        if (not artifact.resolve().is_relative_to(study.resolve())
                and allowed_external.get(str(artifact.resolve())) != digest):
            raise ValueError(f'Unregistered external selection artifact: {filename}')
        if sha(artifact) != digest:
            raise ValueError(f'Selection artifact changed: {filename}')
    return lock


def mark_state(study, state, **updates):
    path = Path(study) / 'state.json'
    previous = json.loads(path.read_text())
    dump(path, dict(previous, state=state, updated_unix=time.time(), **updates))


def load_frame(study, protocol, split):
    if split not in SPLITS:
        raise ValueError(split)
    if split == 'test':
        validate_selection_lock(study)
        mark_state(study, 'extracting_test', formal_test_opened=True)
    elif split == 'validation':
        mark_state(study, f'extracting_{split}', formal_validation_opened=True)
    else:
        mark_state(study, f'extracting_{split}')
    path = Path(protocol['data_dir']) / f'{split}.parquet'
    if sha(path, study) != protocol['inputs'][str(path)]:
        raise ValueError(f'{split}: registered data bytes changed')
    cols = ['prompt', 'base_id', 'group_id', 'source', 'prompt_harm_label',
            'category_id', 'type_label_mask', 'split']
    frame = pd.read_parquet(path, columns=cols).sort_values('base_id', kind='stable').reset_index(drop=True)
    validate_metadata(frame, split)
    if len(frame) != protocol['rows'][split]:
        raise ValueError(f'{split}: row count changed')
    return frame


def load_frozen_backbone(protocol):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    if protocol['model_type'] == 'qwen3_5':
        from hanguard_model import load_base_model
        model = load_base_model(protocol['model_path'], local_files_only=True,
                                device_map={'': 0}, attn_implementation='sdpa')
        backbone = model.model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(protocol['model_path'],
            local_files_only=True, device_map={'': 0}, attn_implementation='sdpa',
            dtype=torch.bfloat16)
        backbone = model.model
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError('The foundation backbone must be completely frozen')
    if any('lora_' in name for name, _ in model.named_parameters()):
        raise AssertionError('An adapter was unexpectedly loaded')
    if len(backbone.layers) != protocol['num_layers']:
        raise ValueError('Backbone block count differs from registered model')
    return model, backbone, tokenizer


class LastBlockCapture:
    """Retain only one valid-position vector per example and block."""
    def __init__(self, backbone):
        self.backbone = backbone
        self.count = len(backbone.layers)
        self.indices = None
        self.values = {}
        self.handles = [layer.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(backbone.layers)]

    def _hook(self, index):
        def hook(module, inputs, output):
            if self.indices is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(len(hidden), device=hidden.device)
            self.values[index] = hidden[rows, self.indices].detach().to(torch.float16)
        return hook

    @torch.inference_mode()
    def __call__(self, batch):
        self.values.clear()
        self.indices = last_true_indices(batch['attention_mask'])
        try:
            device_type = batch['input_ids'].device.type
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
                self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
            if set(self.values) != set(range(self.count)):
                raise ValueError('Missing captured transformer blocks')
            return torch.stack([self.values[i] for i in range(self.count)], dim=1).cpu()
        finally:
            self.indices = None
            self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_data(frame, tokenizer, protocol):
    encoded = tokenizer(frame.prompt.tolist(), add_special_tokens=False, truncation=False)['input_ids']
    lengths = np.asarray([len(ids) for ids in encoded], dtype=np.int32)
    if not len(lengths) or lengths.min() < 1 or lengths.max() > protocol['extraction']['max_tokens']:
        raise ValueError('Empty/over-budget text; truncation is forbidden')
    meta = frame.drop(columns=['prompt', 'prompt_harm_label']).copy()
    meta['label'] = frame.prompt_harm_label.eq('harmful').astype(np.int8)
    meta['prompt_tokens'] = lengths
    return dict(encoded=encoded, lengths=lengths, meta=meta)


def cache_complete(study, protocol, split):
    manifest_path = Path(study) / 'caches' / f'{split}.manifest.json'
    if not manifest_path.exists():
        return False
    m = json.loads(manifest_path.read_text())
    expected_shape = [protocol['rows'][split], protocol['num_layers'], protocol['hidden_size']]
    split_path = str(Path(protocol['data_dir']) / f'{split}.parquet')
    if (m.get('complete') is not True or m.get('protocol_sha256') != sha(Path(study) / 'protocol.json')
            or m.get('adapter_sha256') is not None or m.get('cohort') != split
            or m.get('split') != split or m.get('shape') != expected_shape
            or m.get('input_sha256') != protocol['inputs'][split_path]
            or m.get('layers') != list(range(1, protocol['num_layers'] + 1))
            or m.get('raw_block_outputs') is not True or m.get('truncation') is not False
            or m.get('backbone_trainable_parameters') != 0):
        raise ValueError('Published feature-cache identity differs')
    for key, digest_key in [('path', 'sha256'), ('metadata_path', 'metadata_sha256'),
                            ('replay_audit_path', 'replay_audit_sha256')]:
        if sha(m[key], study) != m[digest_key]:
            raise ValueError(f'{split}: published {key} changed')
    meta = pd.read_parquet(m['metadata_path'])
    validate_feature_cache(Path(m['path']), m['shape'], 'float16',
                           m['ordered_base_id_sha256'], meta.base_id.tolist())
    if not json.loads(Path(m['replay_audit_path']).read_text()).get('passed'):
        raise ValueError('Published cache replay gate did not pass')
    return True


def exact_replay(capture, data, tokenizer, extraction, groups, cache=None):
    # Replay original batch shapes, not singleton-versus-padded comparisons.
    # This isolates cache/capture validity from legitimate BF16 shape effects.
    selected = sorted(set([0, len(groups) // 2, len(groups) - 1]))
    records = []
    for number in selected:
        indices = groups[number]
        batch = legacy.make_batch(data, indices, tokenizer, extraction)
        first, second = capture(batch), capture(batch)
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise RuntimeError('Identical-batch frozen forward replay is not exact and finite')
        equals_cache = cache is None or np.array_equal(np.asarray(cache[indices]), first.numpy())
        if not equals_cache:
            raise RuntimeError('Fresh original-batch replay differs from saved features')
        records.append(dict(group=number, rows=len(indices),
            min_tokens=int(data['lengths'][indices].min()), max_tokens=int(data['lengths'][indices].max()),
            forward_repeated_equal=True, saved_equal=equals_cache,
            sample_ids=data['meta'].base_id.iloc[indices].tolist()))
    return dict(passed=True, rule='exact repeated original-group FP16 vectors',
                records=records, batch_shape_equivalence_claimed=False)


def extract_split(study, protocol, split):
    study = Path(study)
    if load_protocol(study) != protocol:
        raise ValueError('The registered scientific protocol changed between extraction phases')
    if split == 'test':
        validate_selection_lock(study)
    if cache_complete(study, protocol, split):
        return
    frame = load_frame(study, protocol, split)
    model, backbone, tokenizer = load_frozen_backbone(protocol)
    capture = LastBlockCapture(backbone)
    try:
        data = make_data(frame, tokenizer, protocol)
        del frame
        extraction = dict(protocol['extraction'])
        order = np.argsort(data['lengths'], kind='stable')
        groups = legacy.microbatches(order, data['lengths'], extraction['token_budget'],
                                     extraction['max_micro'], extraction['pad_multiple'])
        shape = (len(order), protocol['num_layers'], protocol['hidden_size'])
        folder = study / 'caches'
        folder.mkdir(exist_ok=True)
        temporary = folder / f'{split}.partial.npy'
        progress_path = folder / f'{split}.progress.json'
        identity = dict(protocol_sha256=sha(study / 'protocol.json'), split=split,
                        shape=list(shape), ordered_base_id_sha256=ordered_base_id_sha256(data['meta'].base_id.tolist()),
                        runtime_gpu_sha256=sha(study / 'runtime_gpu.json'))
        if progress_path.exists():
            raise ValueError('Incomplete cache is retained for audit; use a fresh study rather than adopting unverified partial rows')
        else:
            if temporary.exists():
                raise ValueError('Orphan partial cache has no trusted progress journal')
            cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=shape)
            start = 0
        preflight = exact_replay(capture, data, tokenizer, extraction, groups)
        dump(folder / f'{split}.preflight.json', preflight)
        began = time.monotonic()
        completed_rows = sum(len(g) for g in groups[:start])
        for number in range(start, len(groups)):
            check_stop(study)
            indices = groups[number]
            actual = capture(legacy.make_batch(data, indices, tokenizer, extraction)).numpy()
            if actual.shape != (len(indices), shape[1], shape[2]) or not np.isfinite(actual).all():
                raise FloatingPointError('Captured feature dimensions/values are invalid')
            cache[indices] = actual
            completed_rows += len(indices)
            if (number + 1) % 25 == 0 or number + 1 == len(groups):
                cache.flush()
                progress = dict(identity=identity, next_group=number + 1, groups=len(groups),
                    completed_rows=completed_rows, total_rows=len(order),
                    elapsed_seconds=time.monotonic() - began, updated_unix=time.time(), pid=os.getpid())
                dump(progress_path, progress)
                print(json.dumps(dict(state='extracting', split=split, **progress), ensure_ascii=False), flush=True)
        audit = exact_replay(capture, data, tokenizer, extraction, groups, cache)
        audit_path = folder / f'{split}.replay_audit.json'
        dump(audit_path, audit)
        cache.flush()
        metadata_path = folder / f'{split}.metadata.parquet'
        data['meta'].to_parquet(metadata_path, index=False)
        final = folder / f'{split}.npy'
        checked = validate_feature_cache(temporary, shape, 'float16',
            identity['ordered_base_id_sha256'], data['meta'].base_id.tolist())
        temporary.replace(final)
        dump(folder / f'{split}.manifest.json', dict(identity, complete=True,
            cohort=split, rows=len(order), num_layers=shape[1], layers=list(range(1, shape[1] + 1)),
            hidden_size=shape[2], path=str(final), metadata_path=str(metadata_path),
            sha256=sha(final, study), metadata_sha256=sha(metadata_path),
            replay_audit_path=str(audit_path), replay_audit_sha256=sha(audit_path),
            pooling='last_valid_token', raw_block_outputs=True, truncation=False,
            dtype='float16', adapter_path=None, adapter_sha256=None,
            backbone_trainable_parameters=0, validation=checked,
            input_sha256=protocol['inputs'][str(Path(protocol['data_dir']) / f'{split}.parquet')]))
        progress_path.unlink()
        del cache, data
    finally:
        capture.close()
        del capture, model, backbone
        gc.collect()
        torch.cuda.empty_cache()


def verify_guardian_child(study, visible):
    """Reject direct CUDA launches or any process outside the live attempt."""
    from scripts.hanguard.ab_readout_parallel_queue import process_identity, gpu_inventory
    study = Path(study).resolve()
    ledger = study / 'resource_attempt_001/execution/child_identity_frozen_base_pipeline.json'
    if not ledger.exists():
        raise ValueError('No identity-bound guardian child ledger; direct GPU run is forbidden')
    expected = json.loads(ledger.read_text())
    actual = process_identity(os.getpid())
    keys = ('pid', 'uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd')
    if expected.get('uid') != os.getuid() or any(expected.get(k) != actual.get(k) for k in keys):
        raise ValueError('This process is not the exact registered guardian child')
    card = next((item for item in gpu_inventory() if item['index'] == int(visible)), None)
    if card is None or any(pid != os.getpid() for pid in card['pids']):
        raise ValueError('The registered GPU has foreign or unknown compute processes')
    return actual


def bind_gpu(study):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in tuple(str(i) for i in range(8)):
        raise ValueError('Launch only through the single-card guardian on one registered physical GPU')
    plan_path = Path(study) / 'resource_plan.json'
    if (not plan_path.exists() or
            json.loads(plan_path.read_text())['resource_policy']['physical_gpu'] != int(visible)):
        raise ValueError('Visible GPU differs from the exact registered resource plan')
    verify_guardian_child(study, visible)
    result = subprocess.run(['nvidia-smi', '-i', visible,
        '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
        check=True, text=True, capture_output=True)
    value = dict(physical_gpu=int(visible), inventory=result.stdout.strip(),
                 runtime=runtime_identity(), backbone_dtype='bfloat16', cache_dtype='float16')
    path = Path(study) / 'runtime_gpu.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('The registered GPU/runtime identity changed on continuation')
    if not path.exists():
        dump(path, value)


def run(study):
    study = Path(study).resolve()
    protocol = load_protocol(study, full_hash=False)
    bind_gpu(study)
    # Complete this experiment's real CUDA runtime initialization immediately
    # after empty-card admission. Long content hashing and transformer imports
    # otherwise leave the admitted card apparently empty to other users.
    # This is the same identity-bound worker, not a separate reservation job.
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one registered CUDA device must be visible')
    cuda_readiness = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    if not torch.isfinite(cuda_readiness).all():
        raise RuntimeError('CUDA allocation/readiness preflight failed')
    dump(study / 'cuda_startup_preflight.json', dict(passed=True, pid=os.getpid(),
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        visible_device_count=1, device_name=torch.cuda.get_device_name(0),
        actual_cuda_allocation_verified=True,
        protocol_sha256=sha(study / 'protocol.json'),
        model_or_data_used_before_full_hash=False))
    protocol = load_protocol(study, full_hash=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for split in ('train', 'validation'):
        extract_split(study, protocol, split)
    from scripts.hanguard import frozen_base_readout_heads as heads
    mark_state(study, 'fitting_readouts', training_started=True)
    heads.fit(study, device='cuda')
    validate_selection_lock(study)
    mark_state(study, 'validation_selections_locked')
    # Freeze all selections before the first test prompt/label is loaded.
    extract_split(study, protocol, 'test')
    mark_state(study, 'evaluating_locked_test')
    heads.evaluate(study, device='cuda')
    path = study / 'analysis/results.json'
    if not path.exists() or json.loads(path.read_text()).get('complete') is not True:
        raise ValueError('Readout evaluation did not publish a completed result')
    mark_state(study, 'completed', backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'run', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3.5-4B')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/three_source_remediated_20261012')
    args = parser.parse_args(argv)
    if args.action == 'init':
        protocol = initialize(args)
        print(json.dumps(dict(state='registered', output=str(args.output),
            rows=protocol['rows'], model_path=protocol['model_path'], use_lora=False), ensure_ascii=False))
    elif args.action == 'run':
        try:
            run(args.output)
        except BaseException as exc:
            if (args.output / 'state.json').exists():
                mark_state(args.output, 'stopped' if isinstance(exc, KeyboardInterrupt) or
                    (args.output / 'STOP').exists() else 'failed',
                    error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise
    else:
        print((args.output / 'state.json').read_text())


if __name__ == '__main__':
    main()
