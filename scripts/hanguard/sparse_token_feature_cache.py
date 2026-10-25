"""Full-token, frozen-A-LoRA cache for the registered B-only readout experiment.

``init`` is CPU-only. CUDA actions must be launched by an exclusive resource
guardian. The flat binary contains raw block outputs, never padding or a chat
template. A SHA journal commits each deterministic extraction group; interrupted
uncommitted groups are replayed without changing any committed feature bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import ab_diagnostic_experiment as source
from scripts.hanguard import repaired_study as legacy
from scripts.hanguard import remediated_dataset_contract as contract
from scripts.hanguard.frozen_lora_layer_diagnostics import dump, ordered_base_id_sha256

DEFAULT_SOURCE = ROOT / 'outputs/hanguard_ab_diagnosis_20261012/lora_study'
DEFAULT_LAYERS = (21, 22, 25, 29)
CACHE_POLICY = 'frozen_A_B_common_gpu_ragged_raw_token_v3'


def sha(path: Path) -> str:
    return contract.sha(Path(path))


def row_offsets(lengths: Sequence[int]) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or not len(lengths) or (lengths < 1).any():
        raise ValueError('Every input must have at least one valid token')
    return np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]


def validate_offsets(offsets: np.ndarray, lengths: np.ndarray, tokens: int) -> None:
    if offsets.dtype != np.int64 or not np.array_equal(offsets, row_offsets(lengths)):
        raise ValueError('Ragged offsets changed or are not exact full-input lengths')
    if int(offsets[-1]) != int(tokens):
        raise ValueError('Token extent differs from offsets')


def raw_memmap(path: Path, shape: Sequence[int], mode: str) -> np.memmap:
    """Reject NPY headers, trailing data and truncated raw feature files."""
    shape = tuple(int(x) for x in shape)
    if not shape or any(x < 1 for x in shape):
        raise ValueError('Positive raw feature shape required')
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype('float16').itemsize
    path = Path(path)
    if mode != 'w+':
        if path.stat().st_size != expected:
            raise ValueError('Raw feature byte extent changed (NPY headers are prohibited)')
        with path.open('rb') as stream:
            if stream.read(6) == b'\x93NUMPY':
                raise ValueError('Raw feature file must not contain an NPY header')
    return np.memmap(path, mode=mode, shape=shape, dtype=np.float16)


def group_digest(cache: np.ndarray, indices: Sequence[int], offsets: np.ndarray) -> str:
    result = hashlib.sha256()
    for row in map(int, indices):
        begin, end = map(int, offsets[row:row + 2])
        result.update(np.asarray([row, begin, end], dtype='<i8').tobytes())
        result.update(np.ascontiguousarray(cache[begin:end]).tobytes())
    return result.hexdigest()


def verify_journal(progress: dict, identity: dict, cache: np.ndarray,
                   groups: Sequence[Sequence[int]], offsets: np.ndarray) -> int:
    if progress.get('identity') != identity:
        raise ValueError('Resume identity changed')
    commits = progress.get('commits', [])
    if progress.get('next_group') != len(commits) or len(commits) > len(groups):
        raise ValueError('Resume group journal is not a contiguous prefix')
    for number, commit in enumerate(commits):
        indices = groups[number]
        tokens = sum(int(offsets[int(row) + 1] - offsets[int(row)]) for row in indices)
        if commit.get('group') != number or commit.get('rows') != len(indices) or commit.get('tokens') != tokens:
            raise ValueError('Committed group ordering/extent changed')
        if commit.get('sha256') != group_digest(cache, indices, offsets):
            raise ValueError(f'Committed feature bytes changed in group {number}')
    if progress.get('rows') != sum(map(len, groups[:len(commits)])):
        raise ValueError('Committed row count differs from the exact group prefix')
    return len(commits)


def pack_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError('Hidden-state and mask shapes must agree')
    mask = mask.bool()
    if not mask.any(1).all():
        raise ValueError('Every input must have valid tokens')
    # Boolean selection preserves row order and token order for either padding
    # side, including noncontiguous valid positions; no padded value is retained.
    return hidden[mask].detach().to(device='cpu', dtype=torch.float16)


class TokenBlockCapture:
    def __init__(self, backbone, layers: Sequence[int]):
        self.backbone, self.layers = backbone, tuple(layers)
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError('Captured layers must be unique and increasing')
        if self.layers[-1] > len(backbone.layers) or self.layers[0] < 1:
            raise ValueError('Captured block index outside the backbone')
        self.mask, self.values = None, {}
        self.handles = [backbone.layers[layer - 1].register_forward_hook(self._hook(layer))
                        for layer in self.layers]

    def _hook(self, layer):
        def capture(module, inputs, output):
            if self.mask is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            self.values[layer] = pack_valid(hidden, self.mask)
        return capture

    def _packed_values(self):
        if set(self.values) != set(self.layers):
            raise RuntimeError('Not all registered raw blocks were captured')
        packed = torch.stack([self.values[layer] for layer in self.layers], dim=1).numpy()
        if not np.isfinite(packed).all():
            raise FloatingPointError('Captured valid-token feature is nonfinite')
        return packed

    @torch.inference_mode()
    def __call__(self, batch) -> np.ndarray:
        self.values.clear()
        self.mask = batch['attention_mask'].bool()
        try:
            device = batch['input_ids'].device.type
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
            return self._packed_values()
        finally:
            self.mask = None
            self.values.clear()

    @torch.inference_mode()
    def capture_with_original(self, batch, original_capture):
        """Both independent hook conventions see the very same forward outputs."""
        if original_capture.backbone is not self.backbone:
            raise ValueError('Same-forward comparison requires the same frozen backbone')
        self.values.clear()
        self.mask = batch['attention_mask'].bool()
        try:
            original_last, _ = original_capture(batch)
            return self._packed_values(), original_last.numpy()
        finally:
            self.mask = None
            self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
        self.mask = None


def write_group(cache, packed, rows, offsets):
    expected = sum(int(offsets[int(row) + 1] - offsets[int(row)]) for row in rows)
    if packed.shape != (expected, *cache.shape[1:]) or packed.dtype != np.float16:
        raise ValueError('Packed group layout differs from the registered raw token layout')
    if not np.isfinite(packed).all():
        raise FloatingPointError('Cannot commit nonfinite token features')
    cursor = 0
    for row in map(int, rows):
        begin, end = map(int, offsets[row:row + 2])
        cache[begin:end] = packed[cursor:cursor + end - begin]
        cursor += end - begin


def selected_probe_identity(folder: Path, fold: int, readout_protocol_sha256: str):
    """A completed probe selection must be tied to the same frozen readout."""
    folder = Path(folder)
    selection_path, complete_path = folder / 'selection.json', folder / 'complete.json'
    selection = json.loads(selection_path.read_text())
    complete = json.loads(complete_path.read_text())
    provenance = selection.get('provenance', {})
    if (provenance.get('outer_fold') != fold or provenance.get('stage') != 'probes'
            or provenance.get('protocol_sha256') != readout_protocol_sha256):
        raise ValueError('Selected layer is not bound to this registered inner-dev probe protocol')
    if complete.get('complete') is not True or complete.get('provenance') != provenance:
        raise ValueError('Probe completion provenance differs from its selection')
    artifacts = complete.get('artifacts', {})
    if artifacts.get('selection.json') != sha(selection_path):
        raise ValueError('Probe selection SHA differs from the completed stage')
    files = {str(complete_path): sha(complete_path)}
    for name, digest in artifacts.items():
        path = folder / name
        if sha(path) != digest:
            raise ValueError(f'Completed probe artifact changed: {name}')
        files[str(path)] = digest
    return int(selection['best_single_layer']), files


def _source_identity(source_study: Path, readout_study: Path, layers: Sequence[int]):
    source_study, readout_study = Path(source_study).resolve(), Path(readout_study).resolve()
    protocol = source.load_protocol(source_study)
    protocol['study_output'] = str(source_study)
    partition = source.partition_manifest(protocol['partition_manifest_path'])
    run = source_study / 'runs' / f'E03_s{protocol["seeds"][0]}'
    lineage = json.loads((run / 'diagnostic_lineage.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    expected = dict(adapter_training_cohort='A', protocol_sha256=sha(source_study / 'protocol.json'),
        A_train_sha256=protocol['data_sha256']['train'], A_selection_sha256=protocol['data_sha256']['validation'],
        B_sha256=partition['artifact_sha256']['B.parquet'], formal_test_opened=False,
        adapter_sha256=sha(run / 'adapter.pt'), head_sha256=sha(run / 'head.pt'))
    if any(lineage.get(k) != v for k, v in expected.items()):
        raise ValueError('A-only selected checkpoint lineage changed')
    if selection.get('checkpoint_hashes') != {'adapter.pt': expected['adapter_sha256'], 'head.pt': expected['head_sha256']}:
        raise ValueError('A-selected checkpoint identities changed')
    old_path = source_study / 'features/adapted/last.manifest.json'
    old = json.loads(old_path.read_text())
    if old.get('complete') is not True or old.get('cohort') != 'B' or old.get('raw_block_outputs') is not True:
        raise ValueError('Completed B-only raw adapted last cache required')
    for k in ('adapter_sha256', 'B_sha256'):
        if old.get(k) != expected[k]:
            raise ValueError(f'Legacy adapted feature identity differs: {k}')
    if old.get('training_protocol_sha256') != expected['protocol_sha256']:
        raise ValueError('Legacy cache training protocol differs')
    if sha(Path(old['path'])) != old['sha256'] or sha(Path(old['metadata_path'])) != old['metadata_sha256']:
        raise ValueError('Legacy adapted feature/metadata bytes changed')
    if sha(source_study / 'features/adapted/replay_audit.json') != old['replay_audit_sha256']:
        raise ValueError('Legacy adapted replay audit changed')
    readout_protocol_path = readout_study / 'protocol.json'
    readout_protocol = json.loads(readout_protocol_path.read_text())
    readout_digest = sha(readout_protocol_path)
    for key, value in dict(cache_manifest_sha256=sha(old_path), adapter_sha256=expected['adapter_sha256'],
            ab_protocol_sha256=protocol['partition_manifest_sha256'], pooling='last', representation='adapted',
            ordered_base_id_sha256=old['ordered_base_id_sha256'], rows=old['rows']).items():
        if readout_protocol.get(key) != value:
            raise ValueError(f'Probe readout is not bound to the same frozen adapted B cache: {key}')
    fold_layers, fold_sha = {}, {}
    probe_files = {}
    for fold in range(5):
        path = readout_study / f'fold_{fold}/probes/selection.json'
        layer, files = selected_probe_identity(path.parent, fold, readout_digest)
        fold_layers[str(fold)] = layer
        fold_sha[str(fold)] = sha(path)
        probe_files.update(files)
        head_folder = readout_study / f'fold_{fold}/heads'
        head_complete_path = head_folder / 'complete.json'
        head_complete = json.loads(head_complete_path.read_text())
        head_provenance = head_complete.get('provenance', {})
        if (head_complete.get('complete') is not True or head_provenance.get('protocol_sha256') != readout_digest
                or head_provenance.get('outer_fold') != fold or head_provenance.get('stage') != 'heads'):
            raise ValueError('Reference decision replay requires a completed same-protocol frozen head')
        for name in ('heads.pt', 'selection.json'):
            head_path = head_folder / name
            digest = sha(head_path)
            if head_complete.get('artifacts', {}).get(name) != digest:
                raise ValueError('Frozen reference classifier differs from its completed stage')
            probe_files[str(head_path)] = digest
        probe_files[str(head_complete_path)] = sha(head_complete_path)
    if tuple(sorted(set(fold_layers.values()))) != tuple(layers):
        raise ValueError('Cache layers must equal the union of the five inner-dev selected best layers')
    files = {str(p): sha(p) for p in [source_study / 'protocol.json', Path(protocol['partition_manifest_path']),
        run / 'diagnostic_lineage.json', run / 'selection.json', old_path, readout_study / 'protocol.json']}
    files.update(probe_files)
    identity = dict(source_study=str(source_study), readout_study=str(readout_study),
        adapter_sha256=expected['adapter_sha256'], training_protocol_sha256=expected['protocol_sha256'],
        checkpoint_lineage_sha256=sha(run / 'diagnostic_lineage.json'),
        ab_protocol_sha256=protocol['partition_manifest_sha256'], B_sha256=expected['B_sha256'],
        last_cache_manifest_path=str(old_path), last_cache_manifest_sha256=sha(old_path),
        ordered_base_id_sha256=old['ordered_base_id_sha256'], source_artifact_sha256=files,
        fold_best_layers=fold_layers, layer_selection_sha256=fold_sha)
    return identity, protocol, partition, old


def _load_cpu_data(protocol, partition):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
    tokenizer.pad_token, tokenizer.padding_side = '<|im_end|>', 'right'
    frame = pd.read_parquet(partition['b_path']).sort_values('base_id', kind='stable').reset_index(drop=True)
    if not frame.diagnostic_partition.eq('B').all() or not frame.split.eq('train').all() or not frame.base_id.is_unique:
        raise ValueError('Only unique B original-training rows can be tokenized')
    data = source.encode_frame(frame, tokenizer, protocol, partition['artifact_sha256']['B.parquet'])
    return data, tokenizer


def initialize(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output; registered token-cache inputs are immutable')
    layers = tuple(args.layers)
    readout = Path(args.readout_study or Path(args.source_study).parent / 'readouts/adapted_last')
    identity, protocol, partition, old = _source_identity(args.source_study, readout, layers)
    data, tokenizer = _load_cpu_data(protocol, partition)
    meta = data['meta']
    if ordered_base_id_sha256(meta.base_id.tolist()) != identity['ordered_base_id_sha256']:
        raise ValueError('Ordered B identities differ from the existing adapted cache')
    old_meta = pd.read_parquet(old['metadata_path'])
    if not meta.equals(old_meta):
        raise ValueError('Metadata/token lengths differ from the completed adapted cache')
    if args.token_budget < int(data['lengths'].max()) or args.max_micro < 1:
        raise ValueError('Full-input extraction budget must fit the longest B row')
    batching = dict(token_budget=args.token_budget, max_micro=args.max_micro, pad_multiple=protocol['pad_multiple'])
    if batching != {key: protocol[key] for key in batching}:
        raise ValueError('Token extraction must preserve the original adapted-cache batch plan')
    groups = legacy.microbatches(np.argsort(data['lengths'], kind='stable'), data['lengths'], **batching)
    offsets = row_offsets(data['lengths'])
    shape = [int(offsets[-1]), len(layers), int(old['hidden_size'])]
    required = int(np.prod(shape, dtype=np.int64)) * 2
    disk = shutil.disk_usage(output.parent if output.parent.exists() else ROOT)
    if disk.free < required + max(2 * 1024 ** 3, required // 10):
        raise OSError(f'Insufficient disk: need {required} bytes plus publication reserve, free {disk.free}')
    output.mkdir(parents=True)
    np.save(output / 'offsets.npy', offsets, allow_pickle=False)
    np.save(output / 'input_ids.npy', np.concatenate([np.asarray(ids, dtype=np.int32) for ids in data['encoded']]), allow_pickle=False)
    meta.to_parquet(output / 'B.metadata.parquet', index=False)
    dump(output / 'groups.json', [ids.tolist() for ids in groups])
    artifacts = {name: sha(output / name) for name in ('offsets.npy', 'input_ids.npy', 'B.metadata.parquet', 'groups.json')}
    registration = dict(policy_id=CACHE_POLICY, **identity, layers=list(layers), shape=shape, dtype='float16',
        rows=len(meta), tokens=int(offsets[-1]), hidden_size=shape[-1], raw_bytes=required,
        batching=batching, groups=len(groups), ordered_metadata_sha256=artifacts['B.metadata.parquet'],
        local_artifact_sha256=artifacts, cache_source_path=str(Path(__file__).resolve()), cache_source_sha256=sha(Path(__file__)),
        raw_block_outputs=True, special_tokens=False, chat_template=False, truncation=False,
        global_convention='last valid raw token at offsets[row + 1] - 1', padding='omitted entirely',
        cache_estimand='Fresh common-cache readout mechanism comparison at five previously selected single layers; not old-cache equivalence or newly optimal single-layer selection.',
        numerical_reference_policy='Old-cache RMSE/cosine/head-decision comparisons are descriptive; legacy .025/.9995 gate result is retained unchanged.',
        internal_reproducibility_gate='On the fixed three original groups: all valid-token FP16 exact across two same-GPU forwards, same-forward old/new end-token hooks exact, and published-cache replay exact.',
        predetermined_replay_rows=_strata(data['lengths']).tolist(),
        formal_validation_used=False, formal_test_opened=False, created_unix=time.time(),
        disk_free_bytes_at_init=disk.free, length_summary=dict(min=int(data['lengths'].min()),
            median=float(np.median(data['lengths'])), maximum=int(data['lengths'].max())))
    dump(output / 'registration.json', registration)
    return registration


def load_registration(output: Path, verify_source: bool = True):
    output = Path(output).resolve()
    r = json.loads((output / 'registration.json').read_text())
    if r.get('policy_id') != CACHE_POLICY or r.get('formal_test_opened') is not False:
        raise ValueError('A registered B-only token cache is required')
    if r['cache_source_sha256'] != sha(Path(r['cache_source_path'])):
        raise ValueError('Registered token cache implementation changed')
    for name, digest in r['local_artifact_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Registered token-cache input changed: {name}')
    if verify_source:
        for name, digest in r['source_artifact_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError(f'Registered scientific source changed: {name}')
        identity, _, _, _ = _source_identity(Path(r['source_study']), Path(r['readout_study']), r['layers'])
        if any(r.get(k) != value for k, value in identity.items()):
            raise ValueError('Registered source checkpoint/layer identity changed')
    offsets = np.load(output / 'offsets.npy', allow_pickle=False)
    metadata = pd.read_parquet(output / 'B.metadata.parquet')
    if len(metadata) != r['rows'] or ordered_base_id_sha256(metadata.base_id.tolist()) != r['ordered_base_id_sha256']:
        raise ValueError('Registered ordered row identities changed')
    validate_offsets(offsets, metadata.prompt_tokens.to_numpy(), r['tokens'])
    groups = json.loads((output / 'groups.json').read_text())
    if len(groups) != r['groups'] or sorted(row for ids in groups for row in ids) != list(range(r['rows'])):
        raise ValueError('Extraction groups must cover every B row exactly once')
    ids = np.load(output / 'input_ids.npy', mmap_mode='r', allow_pickle=False)
    if ids.shape != (r['tokens'],) or ids.dtype != np.int32:
        raise ValueError('Registered no-truncation token IDs changed shape')
    return r, offsets, metadata, groups


def check_stop(output, registration):
    for folder in (Path(output), Path(output).parent, Path(registration['source_study'])):
        path = folder / 'STOP'
        if path.exists():
            raise RuntimeError(f'Token-cache work stopped: {path.read_text().strip()}')


def require_exclusive_gpu(gpu: int):
    query = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True)
    pids = [int(value.strip()) for value in query.stdout.splitlines() if value.strip()]
    other = [pid for pid in pids if pid != os.getpid()]
    if other:
        raise RuntimeError(f'Exclusive GPU {gpu} required; other compute PIDs present: {other}')


def _data_from_registered(output, registration, offsets, metadata):
    ids = np.load(Path(output) / 'input_ids.npy', mmap_mode='r', allow_pickle=False)
    encoded = [ids[int(offsets[row]):int(offsets[row + 1])].tolist() for row in range(registration['rows'])]
    return dict(encoded=encoded, lengths=metadata.prompt_tokens.to_numpy(dtype=np.int32), meta=metadata)


def _model(registration, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    require_exclusive_gpu(gpu)
    protocol = source.load_protocol(Path(registration['source_study']))
    protocol['study_output'] = registration['source_study']
    model, backbone, tokenizer, digest = source._frozen_base(protocol, 'adapted')
    if digest != registration['adapter_sha256'] or len(backbone.layers) != 32:
        raise ValueError('Extracted backbone differs from registered frozen A adapter')
    protocol.update(registration['batching'])
    return model, backbone, tokenizer, protocol


def runtime_fingerprint(gpu: int):
    """Record the actual implementations and physical card used by CUDA work."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'tokenizers', 'triton', 'fla-core',
                 'flash-linear-attention', 'causal-conv1d', 'numpy', 'safetensors', 'accelerate'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in ('torch', 'transformers.models.qwen3_5.modeling_qwen3_5', 'triton', 'causal_conv1d'):
        spec = importlib.util.find_spec(name)
        path = Path(spec.origin) if spec is not None and spec.origin else None
        sources[name] = dict(path=str(path), sha256=sha(path)) if path is not None and path.is_file() else None
    query = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,name,driver_version',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [line.strip().split(',') for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError('Cannot obtain an unambiguous physical GPU/driver fingerprint')
    uuid, name, driver = map(str.strip, rows[0])
    return dict(python_executable=sys.executable, python_version=sys.version.split()[0],
        packages=packages, module_source_sha256=sources, torch_runtime_version=torch.__version__,
        cuda_build_version=torch.version.cuda, qwen_fast_path_available=bool(modeling_qwen3_5.is_fast_path_available),
        qwen_causal_conv1d_available=modeling_qwen3_5.causal_conv1d_fn is not None,
        qwen_chunk_gated_delta_rule_available=modeling_qwen3_5.chunk_gated_delta_rule is not None,
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32), float32_matmul_precision=torch.get_float32_matmul_precision(),
        physical_gpu_uuid=uuid, physical_gpu_name=name, nvidia_driver_version=driver)


def require_same_runtime(actual: dict, expected: dict):
    if actual != expected:
        changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise RuntimeError(f'Common-cache runtime/GPU fingerprint changed: {changed}')


def vector_similarity(left, right):
    left, right = torch.as_tensor(np.asarray(left).copy()).float(), torch.as_tensor(np.asarray(right).copy()).float()
    scale = right.square().mean(-1).sqrt().clamp_min(1e-6)
    return dict(relative_rmse=((left - right).square().mean(-1).sqrt() / scale).numpy(),
                cosine=F.cosine_similarity(left, right, dim=-1).numpy())


def _strata(lengths):
    order = np.argsort(lengths, kind='stable')
    return order[[0, len(order) // 2, -1]].astype(int)


def original_group_plan(lengths, protocol):
    """Reproduce the exact batch shapes that produced the old adapted cache."""
    return legacy.microbatches(np.argsort(lengths, kind='stable'), lengths,
        protocol['token_budget'], protocol['max_micro'], protocol['pad_multiple'])


def summary_similarity(left, right):
    values = vector_similarity(left, right)
    return dict(relative_rmse_max=float(values['relative_rmse'].max()), cosine_min=float(values['cosine'].min()))


def scalar_relative_rmse(saved, current):
    """Original PoolCapture replay normalization over all 32 layers and D."""
    saved = torch.as_tensor(np.asarray(saved).copy()).float()
    current = torch.as_tensor(np.asarray(current).copy()).float()
    scale = current.square().mean().sqrt().clamp_min(1e-6)
    return float((saved - current).square().mean().sqrt() / scale)


def legacy_replay_gate(records, decisions):
    """Retain the original strict numerical gate as a descriptive result."""
    return (all(record['new_vs_original_capture_exact'] for record in records)
            and all(value['relative_rmse_max'] <= .025 and value['cosine_min'] > .9995
                    for record in records for value in record['comparisons'].values())
            and all(value['decisions_equal'] for value in decisions))


def replay_gate(records, require_cached=False):
    """V3 hard gate is internal exact reproducibility, never legacy tolerance."""
    return bool(records) and all(record.get('new_vs_original_capture_exact') is True
        and record.get('same_gpu_repeated_group_all_tokens_exact') is True
        and (record.get('cached_group_all_tokens_exact') is True if require_cached
             else record.get('cached_group_all_tokens_exact') in (True, None))
        for record in records)


def exact_group_checks(packed, repeated, original_last, repeated_last, local_offsets, layer_indices):
    """Check full token reproducibility separately from the endpoint hooks."""
    group_exact = (np.array_equal(packed[local_offsets[1:] - 1], original_last[:, layer_indices])
        and np.array_equal(repeated[local_offsets[1:] - 1], repeated_last[:, layer_indices]))
    return bool(group_exact), bool(np.array_equal(packed, repeated))


def cached_group_exact(cache, packed, indices, offsets):
    cursor, exact = 0, True
    for group_row in indices:
        a, b = map(int, offsets[group_row:group_row + 2])
        exact = exact and np.array_equal(cache[a:b], packed[cursor:cursor + b - a])
        cursor += b - a
    if cursor != len(packed):
        raise ValueError('Sampled original-group replay lost valid tokens')
    return bool(exact)


@torch.inference_mode()
def replay_audit(capture, data, tokenizer, protocol, registration, cache=None, offsets=None, groups=None):
    old = json.loads(Path(registration['last_cache_manifest_path']).read_text())
    old_cache = np.load(old['path'], mmap_mode='r', allow_pickle=False)
    rows = _strata(data['lengths'])
    if registration.get('predetermined_replay_rows', rows.tolist()) != rows.tolist():
        raise ValueError('The fixed three predetermined replay rows changed')
    planned = original_group_plan(data['lengths'], protocol)
    if groups is not None and [list(map(int, value)) for value in groups] != [value.tolist() for value in planned]:
        raise ValueError('Registered extraction groups differ from original adapted-cache batch shapes')
    layer_indices = np.asarray(registration['layers']) - 1
    records, fresh_last, group_results = [], [], {}
    original_capture = source.PoolCapture(capture.backbone)
    try:
        # The causal model can produce slightly different BF16 activations at
        # different batch shapes. First replay each fixed stratum in the exact
        # stable group used for the original cache, with unchanged tolerances.
        for row in rows:
            matches = [(number, indices, int(np.flatnonzero(indices == row)[0]))
                       for number, indices in enumerate(planned) if row in indices]
            if len(matches) != 1:
                raise ValueError('Stratum row must belong to exactly one original group')
            number, indices, position = matches[0]
            if number not in group_results:
                batch = legacy.make_batch(data, indices, tokenizer, protocol)
                packed, raw_last = capture.capture_with_original(batch, original_capture)
                repeated, repeated_last = capture.capture_with_original(batch, original_capture)
                local_offsets = row_offsets(data['lengths'][indices])
                group_exact, repeated_exact = exact_group_checks(packed, repeated, raw_last, repeated_last, local_offsets, layer_indices)
                cached_exact = None
                if cache is not None:
                    # Compare every token in every row of the sampled original
                    # group, including rows other than the three strata anchors.
                    cached_exact = cached_group_exact(cache, packed, indices, offsets)
                repeat_details = dict(all_valid_tokens=len(packed), selected_layers=registration['layers'],
                    packed_shape=list(packed.shape), first_forward_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                    second_forward_sha256=hashlib.sha256(repeated.tobytes()).hexdigest())
                group_results[number] = (packed, raw_last, local_offsets, group_exact, repeated_exact, cached_exact, repeat_details)
            packed, raw_last, local_offsets, group_exact, repeated_exact, cached_exact, repeat_details = group_results[number]
            begin, end = map(int, local_offsets[position:position + 2])
            current = packed[end - 1]
            original_selected = raw_last[position, layer_indices]
            fresh_last.append(current)
            comparisons = {'legacy_last': summary_similarity(old_cache[row, layer_indices], current),
                           'new_vs_original_capture': summary_similarity(current, original_selected)}
            if cache is not None:
                a, b = map(int, offsets[row:row + 2])
                comparisons['cache_last'] = summary_similarity(cache[b - 1], current)
                comparisons['cache_all_tokens'] = summary_similarity(cache[a:b], packed[begin:end])
            records.append(dict(base_id=str(data['meta'].base_id.iloc[row]), row=int(row), tokens=int(data['lengths'][row]),
                original_group_index=number, original_group_rows=indices.tolist(),
                original_group_ordered_id_sha256=ordered_base_id_sha256(data['meta'].base_id.iloc[indices].tolist()),
                original_group_lengths=data['lengths'][indices].tolist(),
                new_vs_original_capture_exact=bool(group_exact),
                same_gpu_repeated_group_all_tokens_exact=bool(repeated_exact),
                cached_group_all_tokens_exact=bool(cached_exact) if cached_exact is not None else None,
                repeated_forward_scope=repeat_details,
                capture_comparison_scope='One shared forward; all rows in this original group, all selected layers, exact final-token FP16 equality',
                legacy_all_32_layers_scalar_relative_rmse=scalar_relative_rmse(old_cache[row], raw_last[position]),
                comparisons=comparisons))

        # Preserve the previous mixed and singleton audit as descriptive data.
        # It does not replace the mandatory same-batch or exact hook checks.
        mixed_batch = legacy.make_batch(data, rows, tokenizer, protocol)
        mixed, mixed_all_last = capture.capture_with_original(mixed_batch, original_capture)
        mixed_offsets = row_offsets(data['lengths'][rows])
        for position, row in enumerate(rows):
            mixed_last = mixed[int(mixed_offsets[position + 1]) - 1]
            singleton = capture(legacy.make_batch(data, np.asarray([row]), tokenizer, protocol))
            records[position]['cross_batch_descriptive'] = dict(
                mixed_vs_legacy_last=summary_similarity(old_cache[row, layer_indices], mixed_last),
                mixed_vs_matched_last=summary_similarity(mixed_last, fresh_last[position]),
                singleton_vs_matched_last=summary_similarity(singleton[-1], fresh_last[position]),
                mixed_all_32_layers_scalar_relative_rmse=scalar_relative_rmse(old_cache[row], mixed_all_last[position]))
    finally:
        original_capture.close()
    # The old selected classifiers are a descriptive cross-cache comparison.
    # Their agreement does not control publication of this fresh common cache.
    from scripts.hanguard.repaired_heads import ExperimentHead
    decisions = []
    for fold in range(5):
        folder = Path(registration['readout_study']) / f'fold_{fold}/heads'
        checkpoint = torch.load(folder / 'heads.pt', map_location='cpu', weights_only=True)
        layer = registration['fold_best_layers'][str(fold)]
        slot = registration['layers'].index(layer)
        width = int(checkpoint['states']['best_single']['project.weight'].shape[0])
        head = ExperimentHead(registration['hidden_size'], mode='last', width=width, layers=(layer,), dropout=0).eval()
        head.load_state_dict(checkpoint['states']['best_single'])
        threshold = checkpoint['selection']['thresholds']['best_single']
        probabilities = []
        for values in (np.stack(fresh_last)[:, slot], np.asarray(old_cache[rows, layer - 1])):
            values = torch.from_numpy(np.array(values, copy=True)).float()
            logits = head({'layers': (values[:, None],), 'mask': torch.ones((len(rows), 1), dtype=torch.bool)})['logits']
            probabilities.append(logits.sigmoid().numpy())
        decisions.append(dict(fold=fold, layer=layer, threshold=threshold,
            decisions_equal=bool(np.array_equal(probabilities[0] >= threshold, probabilities[1] >= threshold)),
            max_absolute_probability_delta=float(abs(probabilities[0] - probabilities[1]).max()),
            checkpoint_sha256=sha(folder / 'heads.pt')))
    passed = replay_gate(records, require_cached=cache is not None)
    legacy_passed = legacy_replay_gate(records, decisions)
    return dict(policy_id=CACHE_POLICY, passed=passed, predetermined_length_strata=records, frozen_classifier_replay=decisions,
        legacy_gate_passed=legacy_passed, legacy_gate_role='Descriptive, not a hard gate or claim of old-cache equivalence.',
        tolerances=dict(relative_rmse_max=.025, cosine_min=.9995),
        gate='Internal exact FP16 equality on all valid tokens of three fixed original groups across two same-GPU forwards; same-forward end-token hooks exact; cache replay exact if present.',
        cross_batch_policy='Mixed/singleton and original all32 scalar RMSE are descriptive; no tolerance was increased.',
        limitation='Sampled reproducibility covers all valid tokens of three fixed original groups, not repeated forwards for all B groups. Old-cache equivalence is not established.')


def preflight(args):
    output = Path(args.output).resolve()
    r, offsets, metadata, groups = load_registration(output)
    check_stop(output, r)
    torch.set_num_threads(4)
    model, backbone, tokenizer, protocol = _model(r, args.gpu)
    capture = TokenBlockCapture(backbone, r['layers'])
    try:
        data = _data_from_registered(output, r, offsets, metadata)
        began = time.monotonic()
        audit = replay_audit(capture, data, tokenizer, protocol, r, groups=groups)
        audit.update(registration_sha256=sha(output / 'registration.json'), seconds=time.monotonic() - began,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated(), resource_gpu=args.gpu,
                     runtime_fingerprint=runtime_fingerprint(args.gpu))
        dump(output / 'preflight.json', audit)
        if not audit['passed']:
            raise RuntimeError('Sparse-token cache GPU preflight failed')
        return audit
    finally:
        capture.close()
        del model, backbone
        torch.cuda.empty_cache()


def _progress_identity(output, r):
    return dict(registration_sha256=sha(Path(output) / 'registration.json'), shape=r['shape'],
                adapter_sha256=r['adapter_sha256'], B_sha256=r['B_sha256'],
                input_ids_sha256=r['local_artifact_sha256']['input_ids.npy'], groups_sha256=r['local_artifact_sha256']['groups.json'],
                preflight_sha256=sha(Path(output) / 'preflight.json'))


@torch.inference_mode()
def extract(args):
    output = Path(args.output).resolve()
    if (output / 'manifest.json').exists():
        return validate(output)
    r, offsets, metadata, groups = load_registration(output)
    check_stop(output, r)
    gate = json.loads((output / 'preflight.json').read_text())
    if (gate.get('passed') is not True or gate.get('policy_id') != CACHE_POLICY
            or not replay_gate(gate.get('predetermined_length_strata', []))
            or gate.get('registration_sha256') != sha(output / 'registration.json')):
        raise ValueError('A passed GPU preflight bound to this registration is required')
    identity = _progress_identity(output, r)
    progress_path, partial, final = output / 'progress.json', output / 'tokens.partial.bin', output / 'tokens.bin'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if partial.exists() and final.exists():
            raise ValueError('Ambiguous partial and final feature files')
        # Recover the narrowly defined rename/publication interruption only
        # after all exact groups had already been committed and audited.
        path = partial if partial.exists() else final
        if path == final and progress.get('next_group') != len(groups):
            raise ValueError('Unpublished final feature bytes without a completed journal')
        cache = raw_memmap(path, r['shape'], 'r+')
        start = verify_journal(progress, identity, cache, groups, offsets)
    else:
        if partial.exists() or final.exists():
            raise ValueError('Orphan raw feature cache without trustworthy commit journal')
        cache = raw_memmap(partial, r['shape'], 'w+')
        progress = dict(identity=identity, next_group=0, rows=0, commits=[], groups=len(groups), total_rows=r['rows'])
        dump(progress_path, progress)
        start = 0
    torch.set_num_threads(4)
    model, backbone, tokenizer, protocol = _model(r, args.gpu)
    actual_runtime = runtime_fingerprint(args.gpu)
    require_same_runtime(actual_runtime, gate['runtime_fingerprint'])
    capture = TokenBlockCapture(backbone, r['layers'])
    began = time.monotonic()
    try:
        data = _data_from_registered(output, r, offsets, metadata)
        for number in range(start, len(groups)):
            check_stop(output, r)
            require_exclusive_gpu(args.gpu)
            rows = np.asarray(groups[number], dtype=np.int64)
            packed = capture(legacy.make_batch(data, rows, tokenizer, protocol))
            write_group(cache, packed, rows, offsets)
            cache.flush()
            with partial.open('rb') as stream:
                os.fsync(stream.fileno())
            progress['commits'].append(dict(group=number, rows=len(rows), tokens=len(packed),
                sha256=group_digest(cache, rows, offsets)))
            progress.update(next_group=number + 1, rows=progress['rows'] + len(rows),
                            seconds=time.monotonic() - began, updated_unix=time.time(), pid=os.getpid())
            dump(progress_path, progress)
            if (number + 1) % 25 == 0 or number + 1 == len(groups):
                print(json.dumps({key: progress[key] for key in ('next_group', 'groups', 'rows', 'total_rows', 'seconds', 'pid')}), flush=True)
        audit = replay_audit(capture, data, tokenizer, protocol, r, cache=cache, offsets=offsets, groups=groups)
        if not audit['passed']:
            dump(output / 'replay_audit.failed.json', audit)
            raise RuntimeError('Full-token fresh replay failed; no feature manifest published')
        finite_tokens = 0
        for begin in range(0, r['tokens'], 8192):
            chunk = np.asarray(cache[begin:begin + 8192])
            if not np.isfinite(chunk).all():
                raise FloatingPointError('Full token cache contains nonfinite values')
            finite_tokens += len(chunk)
        # Complete last-position checks and descriptive similarity across every
        # B row. All legacy-cache numerical comparisons are descriptive.
        old = json.loads(Path(r['last_cache_manifest_path']).read_text())
        old_cache = np.load(old['path'], mmap_mode='r', allow_pickle=False)
        last = np.asarray(cache[offsets[1:] - 1])
        comparison = vector_similarity(last, old_cache[:, np.asarray(r['layers']) - 1])
        audit['complete_end_token_check'] = dict(rows=r['rows'], layers=r['layers'],
            tokens_preserved=int(offsets[-1]), offsets_sha256=r['local_artifact_sha256']['offsets.npy'],
            input_ids_sha256=r['local_artifact_sha256']['input_ids.npy'],
            relative_rmse_max=float(comparison['relative_rmse'].max()),
            relative_rmse_p99=float(np.quantile(comparison['relative_rmse'], .99)),
            cosine_min=float(comparison['cosine'].min()),
            numeric_gate='Descriptive old-cache comparison; v3 requires internal bitwise reproducibility.')
        audit['full_finite_tokens_checked'] = finite_tokens
        audit['runtime_fingerprint'] = actual_runtime
        dump(output / 'replay_audit.json', audit)
        cache.flush()
        if partial.exists():
            partial.replace(final)
        digest = sha(final)
        manifest = dict(**r, complete=True, path=str(final), sha256=digest,
            metadata_path=str(output / 'B.metadata.parquet'), metadata_sha256=r['local_artifact_sha256']['B.metadata.parquet'],
            offsets_path=str(output / 'offsets.npy'), input_ids_path=str(output / 'input_ids.npy'),
            registration_path=str(output / 'registration.json'), registration_sha256=identity['registration_sha256'],
            replay_audit_path=str(output / 'replay_audit.json'), replay_audit_sha256=sha(output / 'replay_audit.json'),
            progress_sha256=sha(progress_path), full_finite_tokens_checked=finite_tokens,
            preflight_sha256=identity['preflight_sha256'], runtime_fingerprint=actual_runtime,
            completed_unix=time.time(), extraction_seconds=time.monotonic() - began, resource_gpu=args.gpu)
        dump(output / 'manifest.json', manifest)
        return manifest
    finally:
        capture.close()
        del cache, model, backbone
        torch.cuda.empty_cache()


def validate(output: Path):
    output = Path(output).resolve()
    r, offsets, metadata, groups = load_registration(output)
    m = json.loads((output / 'manifest.json').read_text())
    if m.get('complete') is not True or any(m.get(k) != value for k, value in r.items()):
        raise ValueError('Published manifest differs from the immutable registration')
    if m['registration_sha256'] != sha(output / 'registration.json'):
        raise ValueError('Published registration changed')
    if m['replay_audit_sha256'] != sha(output / 'replay_audit.json') or m['progress_sha256'] != sha(output / 'progress.json'):
        raise ValueError('Published replay/commit journal changed')
    audit = json.loads((output / 'replay_audit.json').read_text())
    gate = json.loads((output / 'preflight.json').read_text())
    if (m['preflight_sha256'] != sha(output / 'preflight.json') or gate.get('policy_id') != CACHE_POLICY
            or not gate.get('passed') or not replay_gate(gate.get('predetermined_length_strata', []))):
        raise ValueError('Published preflight changed or did not pass')
    require_same_runtime(m['runtime_fingerprint'], gate['runtime_fingerprint'])
    require_same_runtime(audit['runtime_fingerprint'], gate['runtime_fingerprint'])
    if (audit.get('policy_id') != CACHE_POLICY or not audit.get('passed')
            or not replay_gate(audit.get('predetermined_length_strata', []), require_cached=True)):
        raise ValueError('Published replay audit did not pass')
    cache = raw_memmap(Path(m['path']), r['shape'], 'r')
    if sha(Path(m['path'])) != m['sha256']:
        raise ValueError('Published feature bytes changed')
    start = verify_journal(json.loads((output / 'progress.json').read_text()), _progress_identity(output, r), cache, groups, offsets)
    if start != len(groups) or m['full_finite_tokens_checked'] != r['tokens']:
        raise ValueError('Published cache does not have complete exact-group and finite coverage')
    return dict(passed=True, rows=r['rows'], tokens=r['tokens'], layers=r['layers'], shape=r['shape'],
                manifest_sha256=sha(output / 'manifest.json'), adapter_sha256=r['adapter_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'preflight', 'extract', 'validate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-study', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--readout-study', type=Path)
    parser.add_argument('--layers', type=int, nargs='+', default=list(DEFAULT_LAYERS))
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--token-budget', type=int, default=24576)
    parser.add_argument('--max-micro', type=int, default=64)
    args = parser.parse_args()
    result = {'init': initialize, 'preflight': preflight, 'extract': extract,
              'validate': lambda value: validate(value.output)}[args.action](args)
    if args.action in ('init', 'extract'):
        result = {key: result[key] for key in ('rows', 'tokens', 'layers', 'shape', 'raw_bytes', 'adapter_sha256') if key in result}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
