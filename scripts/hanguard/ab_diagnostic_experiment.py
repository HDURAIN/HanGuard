"""A-only fresh LoRA and B-only feature extraction; no formal-test command.

This entrypoint reuses the unchanged historical optimizer and head implementation
with an independently validated train-internal loader. The partition and model
identities are frozen before GPU work, and B is never loadable by the trainer.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import repaired_study as legacy
from scripts.hanguard import remediated_dataset_contract as contract
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    LastTokenBlockCapture, ordered_base_id_sha256, validate_lora_load,
)

ARM = 'E03'
SPECS = {'E03': dict(lora=True, mode='last', readout='mlp', objective='bce'),
         'E04': dict(lora=True, mode='fusion', readout='mlp', objective='bce')}
CODE = ['scripts/hanguard/ab_diagnostic_experiment.py',
        'scripts/hanguard/diagnostic_ab_split.py', 'scripts/hanguard/repaired_study.py',
        'scripts/hanguard/repaired_heads.py', 'hanguard_model.py',
        'scripts/hanguard/remediated_dataset_contract.py',
        'scripts/hanguard/frozen_lora_layer_diagnostics.py']


def check_stop(output):
    for folder in (Path(output), Path(output).parent):
        path = folder / 'STOP'
        if path.exists():
            raise RuntimeError(f'Diagnostic stopped: {path.read_text().strip()}')


def partition_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get('policy_id') != 'hanguard_train_internal_ab_grouped_v1':
        raise ValueError('A registered train-internal grouped A/B partition is required')
    for name, digest in value['artifact_sha256'].items():
        if contract.sha(path.parent / name) != digest:
            raise ValueError(f'Partition artifact changed: {name}')
    if value['invariants'].get('final_test_enabled') is not False:
        raise ValueError('Formal test must remain disabled during diagnosis')
    return value


def initialize(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing diagnostic study is immutable; use a fresh output')
    manifest_path = Path(args.partition_manifest).resolve()
    partition = partition_manifest(manifest_path)
    model = Path(args.model).resolve()
    protocol = dict(legacy.DEFAULTS, model_path=str(model), data_dir=partition['lora_data_dir'],
        training_entrypoint='hanguard_ab_diagnostic_v1',
        partition_manifest_path=str(manifest_path), partition_manifest_sha256=contract.sha(manifest_path),
        rows={s: partition['partitions'][p]['rows'] for s, p in
              [('train', 'A_train'), ('validation', 'A_validation')]},
        data_sha256={s: partition['artifact_sha256'][f'lora_data/{s}.parquet']
                     for s in ('train', 'validation')},
        model_config_sha256=contract.sha(model / 'config.json'),
        tokenizer_sha256=contract.sha(model / 'tokenizer.json'),
        model_weight_index_sha256=contract.sha(model / 'model.safetensors.index.json'),
        model_weight_files={p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                            for p in sorted(model.glob('model*.safetensors'))},
        epochs=args.epochs, warm_head_epochs=1, token_budget=args.token_budget,
        max_micro=64, effective_batch=128, seeds=[args.seed], gpu_id=args.gpu,
        max_tokens=4096, head_width=128, adapter_lr=2e-5, head_lr=1e-4,
        warm_head_lr=1e-3, warmup_fraction=.05, layers=[8, 16, 24, 32],
        code_sha256={name: contract.sha(ROOT / name) for name in CODE},
        adapter_training_cohort='A', fresh_base=True, reuse_old_adapter=False,
        formal_validation_used=False, formal_test_enabled=False,
        checkpoint_selection='minimum_A_inner_validation_BCE',
        diagnostic_scope='conditional frozen-representation diagnosis, not final model evaluation',
        extraction_poolings=['last', 'mean'], created_unix=time.time(),
        limitations=['Last-layer supervision may bias the adapted representation toward the last layer.',
                     'One fresh LoRA seed for diagnosis; this cannot establish cross-LoRA reproducibility.',
                     'B outer-CV findings are exploratory; formal evaluation follows a later selection barrier.'])
    if protocol['epochs'] < 1 or protocol['token_budget'] < protocol['max_tokens']:
        raise ValueError('Positive epoch budget and full-input token budget are required')
    output.mkdir(parents=True)
    legacy.dump(output / 'protocol.json', protocol)
    legacy.dump(output / 'arms.json', SPECS)
    legacy.dump(output / 'registration.json', dict(protocol_sha256=contract.sha(output / 'protocol.json'),
        arms_sha256=contract.sha(output / 'arms.json'), registered_unix=time.time()))
    legacy.dump(output / 'state.json', dict(state='registered', fresh_classifier_training_started=False,
                                          formal_test_opened=False, protocol_sha256=contract.sha(output / 'protocol.json')))
    return protocol


def load_protocol(output):
    output = Path(output)
    check_stop(output)
    value = json.loads((output / 'protocol.json').read_text())
    registration = json.loads((output / 'registration.json').read_text())
    if registration['protocol_sha256'] != contract.sha(output / 'protocol.json') or registration['arms_sha256'] != contract.sha(output / 'arms.json'):
        raise ValueError('Registered protocol or arms changed')
    if value.get('training_entrypoint') != 'hanguard_ab_diagnostic_v1' or value.get('formal_test_enabled') is not False:
        raise ValueError('Not an A-only diagnostic protocol')
    if value.get('adapter_training_cohort') != 'A' or value.get('reuse_old_adapter') is not False:
        raise ValueError('Only a fresh A-trained adapter is permitted')
    for name, digest in value['code_sha256'].items():
        if contract.sha(ROOT / name) != digest:
            raise ValueError(f'Registered source changed: {name}; create a new version')
    path = Path(value['partition_manifest_path'])
    if contract.sha(path) != value['partition_manifest_sha256']:
        raise ValueError('Partition manifest changed')
    partition = partition_manifest(path)
    if Path(value['data_dir']).resolve() != Path(partition['lora_data_dir']).resolve():
        raise ValueError('LoRA data directory differs from registered A partition')
    for split, role in [('train', 'A_train'), ('validation', 'A_validation')]:
        if value['rows'][split] != partition['partitions'][role]['rows'] or value['data_sha256'][split] != partition['artifact_sha256'][f'lora_data/{split}.parquet']:
            raise ValueError('LoRA row/hash identity differs from registered A partition')
    for filename, key in [('config.json', 'model_config_sha256'), ('tokenizer.json', 'tokenizer_sha256'),
                          ('model.safetensors.index.json', 'model_weight_index_sha256')]:
        if contract.sha(Path(value['model_path']) / filename) != value[key]:
            raise ValueError(f'Base model artifact changed: {filename}')
    for name, identity in value['model_weight_files'].items():
        stat = (Path(value['model_path']) / name).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity:
            raise ValueError(f'Base weight shard changed: {name}')
    a = [pd.read_parquet(partition[key], columns=['base_id', 'group_id'])
         for key in ('a_train_path', 'a_validation_path')]
    b = pd.read_parquet(partition['b_path'], columns=['base_id', 'group_id'])
    for key in ('base_id', 'group_id'):
        if set(a[0][key]) & set(a[1][key]) or (set(a[0][key]) | set(a[1][key])) & set(b[key]):
            raise ValueError(f'A/B or A inner-validation leakage: {key}')
    if json.loads((output / 'arms.json').read_text()) != SPECS:
        raise ValueError('Registered diagnostic arm definitions changed')
    return value


def load_a_split(split, tokenizer, protocol):
    if split not in ('train', 'validation'):
        raise ValueError('The A-only trainer cannot access B or formal test')
    path = Path(protocol['data_dir']) / f'{split}.parquet'
    if contract.sha(path) != protocol['data_sha256'][split]:
        raise ValueError('A-only input hash changed')
    frame = pd.read_parquet(path).sort_values('base_id', kind='stable').reset_index(drop=True)
    role = 'A_train' if split == 'train' else 'A_validation'
    if len(frame) != protocol['rows'][split] or not frame.diagnostic_partition.eq(role).all():
        raise ValueError('The LoRA loader received data outside its registered A partition')
    if not frame.split.eq('train').all() or not frame.base_id.is_unique:
        raise ValueError('LoRA inputs must be unique original-training identities')
    contract.label_arrays(frame, require_type_mask=True)
    return encode_frame(frame, tokenizer, protocol, contract.sha(path))


def encode_frame(frame, tokenizer, protocol, digest):
    encoded = tokenizer(frame.prompt.tolist(), add_special_tokens=False, truncation=False)['input_ids']
    lengths = np.asarray([len(ids) for ids in encoded], dtype=np.int32)
    if lengths.min() < 1 or lengths.max() > protocol['max_tokens']:
        raise ValueError('Input budget exceeded; truncation is prohibited')
    columns = [c for c in ['base_id', 'group_id', 'source', 'category_id', 'type_label_mask',
                           'diagnostic_partition', 'readout_outer_fold', 'outer_fold'] if c in frame]
    meta = frame[columns].copy()
    meta['label'] = frame.prompt_harm_label.eq('harmful').astype(int)
    meta['prompt_tokens'] = lengths
    if 'readout_outer_fold' in meta:
        meta['outer_fold'] = meta.readout_outer_fold
    return dict(encoded=encoded, lengths=lengths, meta=meta,
                y=torch.tensor(meta.label.to_numpy(), dtype=torch.float32), sha256=digest)


@contextlib.contextmanager
def a_loader(output):
    previous_load, previous_protocol = legacy.load_split, legacy.json_protocol
    legacy.load_split = load_a_split
    legacy.json_protocol = lambda study: (load_protocol(study), SPECS)
    try:
        yield
    finally:
        legacy.load_split, legacy.json_protocol = previous_load, previous_protocol


def preflight(args):
    protocol = load_protocol(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(protocol['gpu_id'])
    for arm in (ARM, 'E04'):
        local = argparse.Namespace(output=args.output, arm=arm, seed=protocol['seeds'][0], steps=2,
                                   include_longest=True, include_warmup=True, preflight_cycles=1)
        with a_loader(args.output):
            legacy.preflight(local)
        gc.collect()
        torch.cuda.empty_cache()
    legacy.dump(Path(args.output) / 'preflight_gate.json', dict(passed=True,
        protocol_sha256=contract.sha(Path(args.output) / 'protocol.json'),
        result_sha256={arm: contract.sha(Path(args.output) / 'preflight' / f'{arm}_s{protocol["seeds"][0]}.json')
                       for arm in (ARM, 'E04')}))
    legacy.dump(Path(args.output) / 'state.json', dict(state='preflight_passed', formal_test_opened=False))


def train(args):
    protocol = load_protocol(args.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(protocol['gpu_id'])
    gate_path = Path(args.output) / 'preflight_gate.json'
    if not gate_path.exists():
        raise ValueError('Both GPU preflights must pass before LoRA training')
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True or gate.get('protocol_sha256') != contract.sha(Path(args.output) / 'protocol.json'):
        raise ValueError('GPU preflight gate belongs to a different protocol')
    for arm in (ARM, 'E04'):
        path = Path(args.output) / 'preflight' / f'{arm}_s{protocol["seeds"][0]}.json'
        if not path.exists() or contract.sha(path) != gate['result_sha256'][arm] or json.loads(path.read_text()).get('passed') is not True:
            raise ValueError('Registered GPU preflight failed or changed')
    original_step = legacy.macro_step
    def checked_step(*step_args, **step_kwargs):
        check_stop(args.output)
        return original_step(*step_args, **step_kwargs)
    legacy.macro_step = checked_step
    legacy.dump(Path(args.output) / 'state.json', dict(state='A_LoRA_training', fresh_classifier_training_started=True,
                                                     pid=os.getpid(), formal_test_opened=False))
    try:
        with a_loader(args.output):
            legacy.train(argparse.Namespace(output=args.output, arm=ARM, seed=protocol['seeds'][0]))
    finally:
        legacy.macro_step = original_step
    run = Path(args.output) / 'runs' / f'{ARM}_s{protocol["seeds"][0]}'
    selection = json.loads((run / 'selection.json').read_text())
    for name, digest in selection['checkpoint_hashes'].items():
        if contract.sha(run / name) != digest:
            raise ValueError('Selected A checkpoint hash mismatch')
    legacy.dump(run / 'diagnostic_lineage.json', dict(adapter_training_cohort='A',
        partition_manifest_sha256=protocol['partition_manifest_sha256'],
        A_train_sha256=protocol['data_sha256']['train'], A_selection_sha256=protocol['data_sha256']['validation'],
        B_sha256=partition_manifest(protocol['partition_manifest_path'])['artifact_sha256']['B.parquet'],
        adapter_sha256=selection['checkpoint_hashes']['adapter.pt'],
        head_sha256=selection['checkpoint_hashes']['head.pt'],
        checkpoint_selection='minimum_A_inner_validation_BCE', formal_test_opened=False,
        protocol_sha256=contract.sha(Path(args.output) / 'protocol.json')))
    legacy.dump(Path(args.output) / 'state.json', dict(state='A_LoRA_frozen', formal_test_opened=False,
                                                     adapter_sha256=selection['checkpoint_hashes']['adapter.pt']))


class PoolCapture(LastTokenBlockCapture):
    """Retain 32 [B,D] last vectors and masked FP32 mean vectors inside hooks."""
    def __init__(self, backbone):
        self.mean_values = {}
        self.mask = None
        super().__init__(backbone)

    def _hook(self, layer):
        parent = super()._hook(layer)
        def capture(module, inputs, output):
            parent(module, inputs, output)
            if not self.active:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            total = torch.zeros((len(hidden), hidden.shape[-1]), device=hidden.device, dtype=torch.float32)
            for start in range(0, hidden.shape[1], 128):
                stop = start + 128
                total += (hidden[:, start:stop].float() * self.mask[:, start:stop, None]).sum(1)
            self.mean_values[layer] = (total / self.mask.sum(1)[:, None]).to(torch.float16)
        return capture

    def __call__(self, batch):
        self.mean_values.clear()
        self.mask = batch['attention_mask'].bool()
        last = super().__call__(batch)
        mean = torch.stack([self.mean_values[i] for i in range(1, 33)], dim=1).cpu()
        self.mean_values.clear()
        self.mask = None
        return last, mean


def _frozen_base(protocol, representation):
    from transformers import AutoTokenizer
    from hanguard_model import load_base_model
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
    tokenizer.pad_token = '<|im_end|>'
    tokenizer.padding_side = 'right'
    model = load_base_model(protocol['model_path'], local_files_only=True,
                            device_map={'': 0}, attn_implementation='sdpa')
    adapter_digest = None
    if representation == 'adapted':
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        run = Path(protocol['study_output']) / 'runs' / f'E03_s{protocol["seeds"][0]}'
        lineage = json.loads((run / 'diagnostic_lineage.json').read_text())
        if lineage.get('protocol_sha256') != contract.sha(Path(protocol['study_output']) / 'protocol.json'):
            raise ValueError('Selected adapter belongs to another A-only protocol')
        adapter_digest = contract.sha(run / 'adapter.pt')
        if adapter_digest != lineage['adapter_sha256'] or lineage.get('adapter_training_cohort') != 'A':
            raise ValueError('Only the selected fresh A-only adapter may be extracted')
        model = get_peft_model(model, LoraConfig(r=protocol['lora_rank'], lora_alpha=protocol['lora_alpha'],
            lora_dropout=protocol['lora_dropout'], target_modules=legacy.TARGET, bias='none'))
        state = torch.load(run / 'adapter.pt', map_location='cpu', weights_only=True)
        validate_lora_load(state, set_peft_model_state_dict(model, state))
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    root = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return model, root.model.language_model, tokenizer, adapter_digest


@torch.inference_mode()
def extract(args):
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    protocol['study_output'] = str(output)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(protocol['gpu_id'])
    torch.set_num_threads(4)
    partition = partition_manifest(protocol['partition_manifest_path'])
    folder = output / 'features' / args.representation
    folder.mkdir(parents=True, exist_ok=True)
    existing = [folder / f'{pool}.manifest.json' for pool in ('last', 'mean')]
    if any(p.exists() for p in existing):
        if not all(p.exists() for p in existing):
            raise ValueError('Partial published cache; repair before resuming')
        for path in existing:
            manifest = json.loads(path.read_text())
            if manifest['training_protocol_sha256'] != contract.sha(output / 'protocol.json'):
                raise ValueError('Existing cache is bound to another protocol')
            if contract.sha(manifest['path']) != manifest['sha256']:
                raise ValueError('Existing feature cache changed')
            if manifest['B_sha256'] != partition['artifact_sha256']['B.parquet'] or manifest['ab_protocol_sha256'] != protocol['partition_manifest_sha256']:
                raise ValueError('Existing cache belongs to another B partition')
            if contract.sha(manifest['metadata_path']) != manifest['metadata_sha256'] or contract.sha(folder / 'replay_audit.json') != manifest['replay_audit_sha256']:
                raise ValueError('Existing metadata or replay audit changed')
            if args.representation == 'adapted':
                run = output / 'runs' / f'E03_s{protocol["seeds"][0]}'
                lineage = json.loads((run / 'diagnostic_lineage.json').read_text())
                if manifest['adapter_sha256'] != contract.sha(run / 'adapter.pt') or lineage['adapter_sha256'] != manifest['adapter_sha256']:
                    raise ValueError('Existing cache belongs to another selected adapter')
        return
    model, backbone, tokenizer, adapter_digest = _frozen_base(protocol, args.representation)
    frame = pd.read_parquet(partition['b_path']).sort_values('base_id', kind='stable').reset_index(drop=True)
    if not frame.diagnostic_partition.eq('B').all() or not frame.split.eq('train').all():
        raise ValueError('Extraction is B-only')
    data = encode_frame(frame, tokenizer, protocol, partition['artifact_sha256']['B.parquet'])
    shape = (len(frame), 32, model.config.text_config.hidden_size)
    if len(backbone.layers) != 32:
        raise ValueError('The layer diagnostic requires exactly 32 blocks')
    capture = PoolCapture(backbone)
    order = np.argsort(data['lengths'], kind='stable')
    groups = legacy.microbatches(order, data['lengths'], protocol['token_budget'], protocol['max_micro'], protocol['pad_multiple'])
    identity = dict(B_sha256=data['sha256'], representation=args.representation, adapter_sha256=adapter_digest,
                    training_protocol_sha256=contract.sha(output / 'protocol.json'), shape=list(shape))
    progress_path = folder / 'progress.json'
    temporary = {pool: folder / f'{pool}.partial.npy' for pool in ('last', 'mean')}
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['identity'] != identity or not all(p.exists() for p in temporary.values()):
            raise ValueError('Extraction resume identity/files changed')
        start = progress['next_group']
        caches = {pool: np.lib.format.open_memmap(p, mode='r+') for pool, p in temporary.items()}
        if any(c.shape != shape or c.dtype != np.float16 for c in caches.values()):
            raise ValueError('Resume cache dimensions changed')
    else:
        if any(p.exists() for p in temporary.values()):
            raise ValueError('Orphan partial cache; no trustworthy resume progress')
        caches = {pool: np.lib.format.open_memmap(p, mode='w+', shape=shape, dtype=np.float16)
                  for pool, p in temporary.items()}
    began = time.monotonic()
    try:
        for number in range(start, len(groups)):
            check_stop(output)
            indices = groups[number]
            values = capture(legacy.make_batch(data, indices, tokenizer, protocol))
            for pool, value in zip(('last', 'mean'), values):
                array = value.numpy()
                if array.shape != (len(indices), 32, shape[-1]) or not np.isfinite(array).all():
                    raise FloatingPointError('Invalid captured full-input features')
                caches[pool][indices] = array
            if (number + 1) % 25 == 0 or number + 1 == len(groups):
                for cache in caches.values():
                    cache.flush()
                record = dict(identity=identity, next_group=number + 1, groups=len(groups),
                    rows=sum(map(len, groups[:number + 1])), total_rows=len(frame),
                    seconds=time.monotonic() - began, pid=os.getpid(), updated=time.time())
                legacy.dump(progress_path, record)
                print(json.dumps(dict(state='extract', **record), ensure_ascii=False), flush=True)
        # Check cache reads, isolated-vs-padded batching, online head replay,
        # and masked mean vectors on predetermined short/median/long samples.
        audit = extraction_replay(capture, data, tokenizer, protocol, caches, args.representation)
        legacy.dump(folder / 'replay_audit.json', audit)
        if not audit['passed']:
            raise RuntimeError('Fresh feature replay failed; stop before readout diagnosis')
        metadata_path = folder / 'B.metadata.parquet'
        data['meta'].to_parquet(metadata_path, index=False)
        ordered = ordered_base_id_sha256(data['meta'].base_id.tolist())
        for pool in ('last', 'mean'):
            caches[pool].flush()
            final = folder / f'{pool}.npy'
            temporary[pool].replace(final)
            for offset in range(0, len(frame), 1024):
                if not np.isfinite(caches[pool][offset:offset + 1024]).all():
                    raise FloatingPointError('Published cache has nonfinite rows')
            legacy.dump(folder / f'{pool}.manifest.json', dict(**identity, complete=True, cohort='B',
                pooling=pool, rows=len(frame), layers=32, hidden_size=shape[-1], dtype='float16',
                path=str(final), sha256=contract.sha(final), metadata_path=str(metadata_path),
                metadata_sha256=contract.sha(metadata_path), ordered_base_id_sha256=ordered,
                adapter_training_cohort='A' if args.representation == 'adapted' else 'none',
                ab_protocol_path=protocol['partition_manifest_path'],
                ab_protocol_sha256=protocol['partition_manifest_sha256'],
                training_protocol_path=str(output / 'protocol.json'),
                checkpoint_lineage_path=str(output / 'runs' / f'E03_s{protocol["seeds"][0]}' / 'diagnostic_lineage.json')
                    if args.representation == 'adapted' else None,
                replay_audit_sha256=contract.sha(folder / 'replay_audit.json'),
                raw_block_outputs=True, truncation=False, full_finite_rows_checked=len(frame)))
        progress_path.unlink()
    finally:
        capture.close()
        del caches, model, backbone
        torch.cuda.empty_cache()


@torch.inference_mode()
def extraction_replay(capture, data, tokenizer, protocol, caches, representation):
    from scripts.hanguard.repaired_heads import ExperimentHead
    positions = np.argsort(data['lengths'], kind='stable')
    ids = positions[[0, len(positions) // 2, -1]]
    # Singleton + batched numerical tolerance is checked on all32 blocks,
    # without imposing bitwise equality across different GEMM shapes.
    batch = legacy.make_batch(data, ids, tokenizer, protocol)
    actual_last, actual_mean = capture(batch)
    records = []
    for column, row in enumerate(ids):
        solo_last, solo_mean = capture(legacy.make_batch(data, np.asarray([row]), tokenizer, protocol))
        per_pool = {}
        for pool, batched, solo in [('last', actual_last, solo_last), ('mean', actual_mean, solo_mean)]:
            saved = torch.from_numpy(np.array(caches[pool][row], copy=True)).float()
            current = batched[column].float()
            singleton = solo[0].float()
            # BF16 backbone + FP16 extraction can differ near zero. Compare
            # scale-adjusted RMS and cosine, and separately replay decisions.
            scale = current.square().mean().sqrt().clamp_min(1e-6)
            cache_rel = float((saved - current).square().mean().sqrt() / scale)
            singleton_rel = float((singleton - current).square().mean().sqrt() / scale)
            per_pool[pool] = dict(cache_relative_rmse=cache_rel, singleton_relative_rmse=singleton_rel,
                cache_cosine_min=float(F.cosine_similarity(saved, current, dim=-1).min()),
                singleton_cosine_min=float(F.cosine_similarity(singleton, current, dim=-1).min()))
        records.append(dict(base_id=str(data['meta'].base_id.iloc[row]), tokens=int(data['lengths'][row]), poolings=per_pool))
    passed = all(x['cache_relative_rmse'] < .025 and x['singleton_relative_rmse'] < .025 and
                 x['cache_cosine_min'] > .995 and x['singleton_cosine_min'] > .995
                 for record in records for x in record['poolings'].values())
    head_replay = None
    if representation == 'adapted':
        run = Path(protocol['study_output']) / 'runs' / f'E03_s{protocol["seeds"][0]}'
        head = ExperimentHead(actual_last.shape[-1], mode='last', width=protocol['head_width'],
                              layers=tuple(protocol['layers'])).cuda().eval()
        head.load_state_dict(torch.load(run / 'head.pt', map_location='cpu', weights_only=True))
        # TapOnline expects the full model. Capture's backbone has all raw hooks;
        # a direct final-block hook replays the actual online head convention.
        captured = {}
        handle = capture.backbone.layers[-1].register_forward_hook(
            lambda module, inputs, value: captured.update(raw=value[0] if isinstance(value, tuple) else value))
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                capture.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
                logits_online = head({'layers': (captured['raw'],), 'mask': batch['attention_mask']})['logits'].float().cpu()
                raw_cached = torch.from_numpy(np.array(caches['last'][ids, -1:], copy=True)).cuda()
                logits_cache = head({'layers': (raw_cached[:, 0, None],),
                                     'mask': torch.ones((len(ids), 1), dtype=torch.bool, device='cuda')})['logits'].float().cpu()
            threshold = json.loads((run / 'selection.json').read_text())['threshold']
            decisions_equal = torch.equal(logits_online.sigmoid() >= threshold, logits_cache.sigmoid() >= threshold)
            head_replay = dict(max_abs_logit_delta=float((logits_online - logits_cache).abs().max()),
                               decisions_equal=decisions_equal, rows=len(ids))
            passed = passed and head_replay['max_abs_logit_delta'] < .15
        finally:
            handle.remove()
    return dict(passed=passed, records=records, head_replay=head_replay,
                note='Predetermined3lengthstrata; toleranceaccountsforBF16batchshape/FP16cache, not a full-model equivalence proof')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'preflight', 'train', 'extract', 'status'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition-manifest', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3.5-4B')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--token-budget', type=int, default=24576)
    parser.add_argument('--representation', choices=['base', 'adapted'], default='adapted')
    args = parser.parse_args()
    if args.action == 'init':
        if args.partition_manifest is None:
            parser.error('init requires --partition-manifest')
        print(json.dumps(initialize(args), ensure_ascii=False, indent=2))
    elif args.action == 'preflight':
        preflight(args)
    elif args.action == 'train':
        train(args)
    elif args.action == 'extract':
        extract(args)
    else:
        print((args.output / 'state.json').read_text())


if __name__ == '__main__':
    main()
