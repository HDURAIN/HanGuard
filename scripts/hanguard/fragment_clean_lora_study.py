"""Fresh E03 on a fragment-filtered release, then five frozen-LoRA readouts.

Only this new entrypoint supplies the adapter-aware contracts. Historical
scientific sources are imported unchanged. ``init`` is CPU-only; ``run`` waits
for genuinely empty compute cards and admits work as cards become available.
Foreign arrivals after CUDA admission are recorded and never stop our worker.
This is an exploratory re-evaluation of previously inspected test identities.
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
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCHEMA = 'hanguard_fragment_clean_fresh_lora_v1'
READOUT_SCHEMA = 'hanguard_fragment_clean_frozen_lora_readouts_v1'
SPLITS = ('train', 'validation', 'test')
POLICY = dict(prelaunch_foreign='block', active_foreign='record_only',
              max_cuda_workers=8, reserved_gpus=[], never_terminate_foreign=True)


class NeedsReview(RuntimeError):
    """The selected checkpoint does not establish effective joint adaptation."""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def check_stop(study):
    study = Path(study)
    for directory in (study, study.parent, study / 'readouts'):
        if (directory / 'STOP').exists():
            raise RuntimeError(f'Stopped by {directory / "STOP"}')


def identity(pid):
    from scripts.hanguard.ab_readout_parallel_queue import process_identity
    return process_identity(pid)


def same_process(expected, actual):
    return all(expected.get(k) == actual.get(k) for k in
               ('pid', 'uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd'))


def inventory():
    from scripts.hanguard.ab_readout_parallel_queue import gpu_inventory
    return gpu_inventory()


def empty_cards(cards, allowed, minimum_free_gib):
    """Compute process presence blocks admission even at zero utilization."""
    return [row['index'] for row in cards if row['index'] in allowed and
            not row['pids'] and row['free_gib'] >= minimum_free_gib]


def source_paths():
    from scripts.hanguard import current_training as training
    names = set(training.code_files('binary')) | {
        'scripts/hanguard/fragment_clean_lora_study.py',
        'scripts/hanguard/frozen_base_readout_study.py',
        'scripts/hanguard/frozen_base_readout_heads.py',
        'scripts/hanguard/frozen_base_resumable_heads.py',
        'scripts/hanguard/frozen_lora_layer_diagnostics.py',
        'scripts/hanguard/ab_frozen_readout_diagnostics.py',
        'scripts/hanguard/ab_readout_parallel_queue.py',
    }
    return sorted(ROOT / name for name in names)


def initialize(args):
    from scripts.hanguard import current_training as training
    from scripts.hanguard import frozen_base_readout_study as base
    study, data, model = args.output.resolve(), args.data.resolve(), args.model.resolve()
    if study.exists() and any(study.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or set(args.gpus) - set(range(8)):
        raise ValueError('Unique GPU indices in 0..7 required')
    if not 0 < args.poll_seconds <= 5 or args.minimum_free_gib <= 0 or args.chunk_groups < 1:
        raise ValueError('Invalid polling, admission memory, or chunk size')
    train_args = training.parser().parse_args([
        'binary', 'init', '--output', str(study / 'lora'), '--data', str(data),
        '--model', str(model), '--binary-arms', 'E03', '--seeds', str(args.seed),
        '--gpus', *[str(g) for g in args.gpus], '--epochs', str(args.lora_epochs),
        '--warm-head-epochs', '1', '--effective-batch', '128', '--max-micro', '64',
        '--token-budget', '24576', '--max-tokens', '4096', '--head-width', '128',
        '--head-lr', '0.0001', '--adapter-lr', '0.00002'])
    # Audit everything before creating a registration, including full base bytes.
    training.build_protocol(train_args)
    config = read(model / 'config.json')
    text = config.get('text_config', config)
    if int(text['num_hidden_layers']) != 32:
        raise ValueError('The five-arm comparison requires exactly 32 blocks')
    import pyarrow.parquet as pq
    total_rows = sum(pq.ParquetFile(data / f'{s}.parquet').metadata.num_rows for s in SPLITS)
    feature_bytes = total_rows * 32 * int(text['hidden_size']) * 2
    existing_parent = study.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 2 * feature_bytes + 8 * 1024**3:
        raise ValueError('Insufficient disk for complete chunk files plus canonical feature caches')
    model_files = {model / name for name in read(model / 'model.safetensors.index.json')['weight_map'].values()}
    model_files.update(p for p in model.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.jinja'))
    files = model_files | {data / f'{s}.parquet' for s in SPLITS}
    files.update(data / name for name in ('manifest.json', 'release_qa.json', 'split_policy.json'))
    inputs = {str(p): sha(p) for p in sorted(files)}
    sources = {str(p): sha(p) for p in source_paths()}
    protocol = dict(schema=SCHEMA, data_dir=str(data), model_path=str(model),
        num_layers=32, hidden_size=int(text['hidden_size']), seed=args.seed,
        gpu_ids=args.gpus, policy=POLICY, minimum_free_gib=args.minimum_free_gib,
        poll_seconds=args.poll_seconds, chunk_groups=args.chunk_groups,
        code_sha256=sources, inputs=inputs,
        input_stats={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in inputs},
        runtime=base.runtime_identity(), created_unix=time.time(),
        methods=['last', 'best_single', 'repeat_best', 'fusion_fixed', 'fusion_fixed_best_anchor'],
        settings=dict(seed=args.seed, width=128, dropout=.1, batch_size=256,
            lr=.001, weight_decay=.001, grad_clip=5., epochs=12, max_epochs=40,
            layers=[8, 16, 24, 32], bootstrap_draws=2000),
        extraction=dict(token_budget=16384, max_micro=64, pad_multiple=32,
            max_tokens=4096, pooling='last_valid_token', raw_block_outputs=True,
            dtype='float16', truncation=False),
        limitations=['Filtering reviewed previously inspected text; this is not a new blind test.',
            'One fresh E03 LoRA seed; all five readouts share exactly that selected frozen adapter.',
            'The data release filtering rule must be independent of which model predicts correctly.',
            'Complete original microbatch groups are distributed; different batch shapes are not claimed equivalent.'])
    training.init(train_args)
    training.register(study / 'lora')
    protocol['lora_registration_sha256'] = sha(study / 'lora/registration.json')
    protocol['lora_protocol_sha256'] = sha(study / 'lora/protocol.json')
    write(study / 'protocol.json', protocol)
    snapshots = study / 'source_snapshot'
    snapshots.mkdir()
    for filename in sources:
        target = snapshots / str(Path(filename).relative_to(ROOT)).replace('/', '__')
        shutil.copyfile(filename, target)
        if sha(target) != sources[filename]:
            raise ValueError('Source changed during registration')
    write(study / 'registration.json', dict(protocol_sha256=sha(study / 'protocol.json'),
        snapshots={str(p): sha(p) for p in snapshots.iterdir()}))
    state(study, 'registered', training_started=False, formal_test_opened=False)
    return protocol


def verify(study, full_hash=False):
    from scripts.hanguard import current_training as training
    from scripts.hanguard import frozen_base_readout_study as base
    study = Path(study).resolve()
    check_stop(study)
    p, registration = read(study / 'protocol.json'), read(study / 'registration.json')
    if (p.get('schema') != SCHEMA or p.get('policy') != POLICY or
            registration['protocol_sha256'] != sha(study / 'protocol.json')):
        raise ValueError('Registered fragment study protocol changed')
    if base.runtime_identity() != p['runtime']:
        raise ValueError('Registered Python/library runtime changed')
    for name, digest in {**p['code_sha256'], **registration['snapshots']}.items():
        if sha(name) != digest:
            raise ValueError(f'Registered source changed: {name}')
    for name, digest in p['inputs'].items():
        path, expected = Path(name), p['input_stats'][name]
        if dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) != expected:
            raise ValueError(f'Registered input metadata changed: {name}')
        if full_hash and sha(path) != digest:
            raise ValueError(f'Registered input bytes changed: {name}')
    if sha(study / 'lora/registration.json') != p['lora_registration_sha256'] or sha(study / 'lora/protocol.json') != p['lora_protocol_sha256']:
        raise ValueError('LoRA child registration changed')
    training.verify(study / 'lora')
    return p


def state(study, phase, **details):
    path = Path(study) / 'state.json'
    prior = read(path) if path.exists() else {}
    prior.update(state=phase, updated_unix=time.time(), coordinator_pid=os.getpid(), **details)
    write(path, prior)


def selected_adapter(study):
    study = Path(study)
    p = read(study / 'protocol.json')
    run = study / 'lora/runs' / f'E03_s{p["seed"]}'
    selected = read(run / 'selection.json')
    child = read(study / 'lora/protocol.json')
    if selected['protocol_sha256'] != sha(study / 'lora/protocol.json') or selected['arms_sha256'] != sha(study / 'lora/arms.json'):
        raise ValueError('E03 selection belongs to a different protocol')
    if selected.get('arm') != 'E03' or selected.get('seed') != p['seed'] or set(selected['checkpoint_hashes']) != {'head.pt', 'adapter.pt'}:
        raise ValueError('Fresh complete E03 selection required')
    for name, digest in selected['checkpoint_hashes'].items():
        if sha(run / name) != digest:
            raise ValueError('Selected E03 checkpoint changed')
    digest = sha(run / 'adapter.pt')
    lineage_path = study / 'adapter_lineage.json'
    lineage = read(lineage_path) if lineage_path.exists() else None
    if lineage is not None:
        if (read(study / 'adapter_lineage.registration.json')['lineage_sha256'] != sha(lineage_path) or
                lineage['adapter_sha256'] != digest or lineage['selection_sha256'] != sha(run / 'selection.json') or
                lineage['actual_selected_phase'] != selected['best_epoch']):
            raise ValueError('Previously recorded adapter lineage changed')
    else:
        import torch
        weights = torch.load(run / 'adapter.pt', map_location='cpu', weights_only=True)
        b_weights = [value.float() for name, value in weights.items() if 'lora_B' in name]
        delta = dict(basis='Standard PEFT initialization has zero LoRA B matrices; selected checkpoint compared with zero B.',
            lora_b_tensors=len(b_weights),
            nonzero_lora_b_tensors=sum(int(torch.count_nonzero(value)) > 0 for value in b_weights),
            lora_b_squared_l2=sum(float(value.square().sum()) for value in b_weights),
            lora_b_max_abs=max((float(value.abs().max()) for value in b_weights), default=0.))
        lineage = dict(adapter_sha256=digest, selection_sha256=sha(run / 'selection.json'),
            actual_selected_phase=selected['best_epoch'], adapter_delta=delta)
        write(lineage_path, lineage)
        write(study / 'adapter_lineage.registration.json', dict(lineage_sha256=sha(lineage_path)))
    return dict(adapter_path=str(run / 'adapter.pt'), adapter_sha256=digest,
        selection_path=str(run / 'selection.json'), selection_sha256=sha(run / 'selection.json'),
        lora_rank=child['lora_rank'], lora_alpha=child['lora_alpha'], lora_dropout=child['lora_dropout'],
        selected_lora_phase=selected['best_epoch'], actual_selected_phase=lineage['actual_selected_phase'],
        adapter_delta=lineage['adapter_delta'], rows=child['rows'])


def require_adapted_adapter(study, adapter):
    phase, delta = adapter['actual_selected_phase'], adapter['adapter_delta']
    if phase.get('phase_name') != 'joint' or delta['nonzero_lora_b_tensors'] < 1:
        state(study, 'needs_review', actual_selected_phase=phase, adapter_delta=delta,
            reason='Validation selected a warm-up or unchanged LoRA checkpoint; retained all training artifacts.')
        raise NeedsReview('Selected E03 checkpoint does not establish joint LoRA adaptation; selection rule unchanged')
    state(study, 'adapted_lora_selected', actual_selected_phase=phase, adapter_delta=delta)


def initialize_readouts(study):
    study = Path(study).resolve()
    p, adapter = verify(study), selected_adapter(study)
    require_adapted_adapter(study, adapter)
    folder = study / 'readouts'
    target = folder / 'protocol.json'
    if target.exists():
        return verify_readouts(folder)
    from scripts.hanguard import remediated_dataset_contract as contract
    rp = dict(p, schema=READOUT_SCHEMA, use_lora=True, **adapter,
        parent_study=str(study), parent_protocol_sha256=sha(study / 'protocol.json'),
        dataset_release=contract.release_contract(p['data_dir']),
        backbone_trainable_parameters=0, formal_test_barrier_required=True,
        fresh_features_required=True)
    write(target, rp)
    write(folder / 'registration.json', dict(protocol_sha256=sha(target)))
    shutil.copyfile(__file__, folder / 'adapter_readout_runtime.py.snapshot')
    return rp


def verify_readouts(folder, full_hash=False):
    folder = Path(folder).resolve()
    p = read(folder / 'protocol.json')
    parent = Path(p['parent_study'])
    verify(parent, full_hash=full_hash)
    if (p.get('schema') != READOUT_SCHEMA or p.get('use_lora') is not True or
            p['parent_protocol_sha256'] != sha(parent / 'protocol.json') or
            read(folder / 'registration.json')['protocol_sha256'] != sha(folder / 'protocol.json')):
        raise ValueError('Frozen LoRA readout registration changed')
    adapter = selected_adapter(parent)
    if any(p.get(k) != v for k, v in adapter.items()):
        raise ValueError('Frozen LoRA identity differs from E03 selection')
    snapshot = folder / 'adapter_readout_runtime.py.snapshot'
    if sha(snapshot) != p['code_sha256'][str(Path(__file__).resolve())]:
        raise ValueError('Readout runtime snapshot changed')
    return p


def require_barrier(folder, split):
    if split not in SPLITS:
        raise ValueError('Unknown split')
    if split == 'test':
        from scripts.hanguard.frozen_base_readout_heads import verify_selection_lock
        verify_selection_lock(folder)


def load_feature_data(folder, split, protocol=None):
    import pandas as pd
    from transformers import AutoTokenizer
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import remediated_dataset_contract as contract
    require_barrier(folder, split)
    p = protocol or verify_readouts(folder)
    source = Path(p['data_dir']) / f'{split}.parquet'
    if sha(source) != p['inputs'][str(source)]:
        raise ValueError('Feature split input changed')
    frame = pd.read_parquet(source).sort_values('base_id', kind='stable').reset_index(drop=True)
    base.validate_metadata(frame, split)
    contract.label_arrays(frame, require_type_mask=True)
    if len(frame) != p['rows'][split]:
        raise ValueError('Feature split rows differ from registration')
    tokenizer = AutoTokenizer.from_pretrained(p['model_path'], local_files_only=True)
    tokenizer.pad_token = '<|im_end|>'
    tokenizer.padding_side = 'right'
    return base.make_data(frame, tokenizer, p), tokenizer


def prepare_features(folder, split):
    import numpy as np
    from scripts.hanguard import repaired_study as legacy
    from scripts.hanguard.frozen_lora_layer_diagnostics import ordered_base_id_sha256
    folder = Path(folder)
    require_barrier(folder, split)
    p = verify_readouts(folder)
    area = folder / 'features' / split
    path = area / 'plan.json'
    if path.exists():
        return verify_plan(folder, split)
    data, _ = load_feature_data(folder, split, p)
    order = np.argsort(data['lengths'], kind='stable')
    ex = p['extraction']
    groups = legacy.microbatches(order, data['lengths'], ex['token_budget'], ex['max_micro'], ex['pad_multiple'])
    groups = [list(map(int, group)) for group in groups]
    area.mkdir(parents=True, exist_ok=True)
    meta = area / 'metadata.parquet'
    data['meta'].to_parquet(meta, index=False)
    count = p['chunk_groups']
    plan = dict(split=split, protocol_sha256=sha(folder / 'protocol.json'),
        input_sha256=p['inputs'][str(Path(p['data_dir']) / f'{split}.parquet')],
        rows=len(order), shape=[len(order), p['num_layers'], p['hidden_size']],
        metadata_path=str(meta), metadata_sha256=sha(meta),
        ordered_base_id_sha256=ordered_base_id_sha256(data['meta'].base_id.astype(str)),
        groups=groups, chunks=[list(range(k, min(k + count, len(groups)))) for k in range(0, len(groups), count)])
    write(path, plan)
    write(area / 'plan.registration.json', dict(plan_sha256=sha(path)))
    return plan


def verify_plan(folder, split):
    import numpy as np
    require_barrier(folder, split)
    folder = Path(folder)
    p = verify_readouts(folder)
    area = folder / 'features' / split
    plan = read(area / 'plan.json')
    if (read(area / 'plan.registration.json')['plan_sha256'] != sha(area / 'plan.json') or
            plan['protocol_sha256'] != sha(folder / 'protocol.json') or plan['split'] != split or
            plan['shape'] != [p['rows'][split], p['num_layers'], p['hidden_size']] or
            sha(plan['metadata_path']) != plan['metadata_sha256']):
        raise ValueError('Registered extraction plan changed')
    rows = [i for group in plan['groups'] for i in group]
    chunks = [i for chunk in plan['chunks'] for i in chunk]
    if not np.array_equal(np.sort(rows), np.arange(plan['rows'])) or sorted(chunks) != list(range(len(plan['groups']))):
        raise ValueError('Extraction plan must cover every row/group exactly once')
    return plan


def chunk_rows(plan, number):
    return [row for group in plan['chunks'][number] for row in plan['groups'][group]]


def chunk_receipt(folder, split, number):
    return Path(folder) / 'features' / split / 'chunks' / f'{number:05d}.complete.json'


def chunks_complete(folder, split):
    plan = verify_plan(folder, split)
    return all(chunk_receipt(folder, split, i).exists() for i in range(len(plan['chunks'])))


def extract_features(study, worker_id, split):
    import numpy as np
    import torch
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import repaired_study as legacy
    from scripts.hanguard.frozen_lora_layer_diagnostics import _load_frozen_model
    study, folder = Path(study), Path(study) / 'readouts'
    p = verify_readouts(folder, full_hash=True)
    plan = verify_plan(folder, split)
    data, _ = load_feature_data(folder, split, p)
    if not data['meta'].equals(__import__('pandas').read_parquet(plan['metadata_path'])):
        raise ValueError('Worker tokenization/row metadata differs from CPU extraction plan')
    model, backbone, tokenizer = _load_frozen_model(p)
    capture = base.LastBlockCapture(backbone)
    directory = folder / 'features' / split / 'chunks'
    directory.mkdir(exist_ok=True)
    ex = p['extraction']
    try:
        # Every card replays the same first original group. Merge checks its digest.
        first = plan['groups'][0]
        batch = legacy.make_batch(data, first, tokenizer, ex)
        calibration = capture(batch)
        if not torch.equal(calibration, capture(batch)) or not torch.isfinite(calibration).all():
            raise RuntimeError('Frozen feature repeated-original-batch replay failed')
        calibration_sha = hashlib.sha256(calibration.numpy().tobytes()).hexdigest()
        for number in range(len(plan['chunks'])):
            check_stop(study)
            receipt = chunk_receipt(folder, split, number)
            if receipt.exists():
                continue
            claim = directory / f'{number:05d}.claim.json'
            try:
                with claim.open('x') as stream:
                    json.dump(dict(worker_id=worker_id, identity=identity(os.getpid())), stream)
            except FileExistsError:
                continue
            ids = chunk_rows(plan, number)
            result = np.empty((len(ids), p['num_layers'], p['hidden_size']), dtype=np.float16)
            offset = 0
            for group_number in plan['chunks'][number]:
                check_stop(study)
                group = plan['groups'][group_number]
                batch = legacy.make_batch(data, group, tokenizer, ex)
                values = capture(batch).numpy()
                if not np.isfinite(values).all():
                    raise RuntimeError('Nonfinite extracted feature')
                result[offset:offset + len(group)] = values
                offset += len(group)
            # Re-read an original group before publishing each completed chunk.
            group = plan['groups'][plan['chunks'][number][0]]
            repeated = capture(legacy.make_batch(data, group, tokenizer, ex)).numpy()
            if not np.array_equal(repeated, result[:len(group)]):
                raise RuntimeError('Saved chunk differs from original-batch replay')
            path = directory / f'{number:05d}.{worker_id}.npy'
            np.save(path, result, allow_pickle=False)
            write(receipt, dict(complete=True, split=split, chunk=number, worker_id=worker_id,
                plan_sha256=sha(folder / 'features' / split / 'plan.json'),
                protocol_sha256=sha(folder / 'protocol.json'), adapter_sha256=p['adapter_sha256'],
                row_indices=ids, shape=list(result.shape), path=str(path), sha256=sha(path),
                calibration_sha256=calibration_sha, original_batch_replay_passed=True))
            print(json.dumps(dict(stage='feature_chunk', split=split, chunk=number, rows=len(ids))), flush=True)
    finally:
        capture.close()
        del model


def merge_features(folder, split):
    import numpy as np
    import pandas as pd
    from scripts.hanguard.frozen_lora_layer_diagnostics import validate_feature_cache
    folder = Path(folder)
    require_barrier(folder, split)
    p, plan = verify_readouts(folder), verify_plan(folder, split)
    manifest_path = folder / 'caches' / f'{split}.manifest.json'
    if manifest_path.exists():
        return load_cache(folder, split)[2]
    receipts = [read(chunk_receipt(folder, split, i)) for i in range(len(plan['chunks']))]
    if len({r['calibration_sha256'] for r in receipts}) != 1:
        raise ValueError('Cross-GPU identical original batch calibration differs')
    area = folder / 'caches'
    area.mkdir(exist_ok=True)
    path = area / f'{split}.npy'
    temporary = area / f'{split}.partial.npy'
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=tuple(plan['shape']))
    seen = np.zeros(plan['rows'], dtype=np.int8)
    for number, receipt in enumerate(receipts):
        ids = chunk_rows(plan, number)
        if (receipt.get('complete') is not True or receipt['row_indices'] != ids or
                receipt['chunk'] != number or receipt['split'] != split or
                receipt['plan_sha256'] != sha(folder / 'features' / split / 'plan.json') or
                receipt['protocol_sha256'] != sha(folder / 'protocol.json') or
                receipt['adapter_sha256'] != p['adapter_sha256'] or
                receipt.get('original_batch_replay_passed') is not True or
                receipt['sha256'] != sha(receipt['path'])):
            raise ValueError('Completed feature chunk identity changed')
        part = np.load(receipt['path'], mmap_mode='r', allow_pickle=False)
        if part.shape != (len(ids), *plan['shape'][1:]) or part.dtype != np.float16 or not np.isfinite(part).all():
            raise ValueError('Completed feature chunk shape/dtype/values invalid')
        values[ids] = part
        seen[ids] += 1
    if not np.all(seen == 1):
        raise ValueError('Feature merge must cover every row exactly once')
    values.flush()
    del values
    meta = pd.read_parquet(plan['metadata_path'])
    validation = validate_feature_cache(temporary, plan['shape'], 'float16', plan['ordered_base_id_sha256'], meta.base_id.astype(str))
    temporary.replace(path)
    audit = area / f'{split}.replay.json'
    write(audit, dict(passed=True, every_row_exactly_once=True,
        calibration_sha256=receipts[0]['calibration_sha256'], same_original_batch_across_cards=True,
        original_batch_chunk_replay_passed=True, batch_shape_equivalence_claimed=False,
        receipts={str(chunk_receipt(folder, split, i)): sha(chunk_receipt(folder, split, i)) for i in range(len(receipts))}))
    manifest = dict(complete=True, split=split, cohort=split, representation='frozen_lora',
        adapter_path=p['adapter_path'], adapter_sha256=p['adapter_sha256'],
        protocol_sha256=sha(folder / 'protocol.json'), shape=plan['shape'],
        path=str(path), sha256=sha(path), metadata_path=plan['metadata_path'],
        metadata_sha256=plan['metadata_sha256'], ordered_base_id_sha256=plan['ordered_base_id_sha256'],
        input_sha256=plan['input_sha256'], replay_audit_path=str(audit), replay_audit_sha256=sha(audit),
        pooling='last_valid_token', raw_block_outputs=True, truncation=False,
        backbone_trainable_parameters=0, validation=validation)
    write(manifest_path, manifest)
    return load_cache(folder, split)[2]


def load_cache(folder, split):
    """Adapter-aware strict cache contract, injected only in this worker process."""
    import numpy as np
    import pandas as pd
    from scripts.hanguard.frozen_lora_layer_diagnostics import ordered_base_id_sha256, validate_feature_cache
    folder = Path(folder).resolve()
    require_barrier(folder, split)
    p = verify_readouts(folder)
    manifest = read(folder / 'caches' / f'{split}.manifest.json')
    expected = [p['rows'][split], p['num_layers'], p['hidden_size']]
    if (manifest.get('complete') is not True or manifest.get('split') != split or
            manifest.get('protocol_sha256') != sha(folder / 'protocol.json') or
            manifest.get('representation') != 'frozen_lora' or
            manifest.get('adapter_sha256') != p['adapter_sha256'] or
            manifest.get('adapter_path') != p['adapter_path'] or manifest.get('shape') != expected or
            manifest.get('input_sha256') != p['inputs'][str(Path(p['data_dir']) / f'{split}.parquet')] or
            manifest.get('raw_block_outputs') is not True or manifest.get('truncation') is not False or
            manifest.get('pooling') != 'last_valid_token' or manifest.get('backbone_trainable_parameters') != 0):
        raise ValueError('Frozen LoRA cache identity mismatch')
    for key, digest in (('path', 'sha256'), ('metadata_path', 'metadata_sha256'), ('replay_audit_path', 'replay_audit_sha256')):
        if not Path(manifest[key]).is_absolute() or sha(manifest[key]) != manifest[digest]:
            raise ValueError(f'Frozen LoRA cache artifact changed: {key}')
    if read(manifest['replay_audit_path']).get('passed') is not True:
        raise ValueError('Feature replay audit did not pass')
    meta = pd.read_parquet(manifest['metadata_path'])
    required = ['base_id', 'group_id', 'source', 'label', 'prompt_tokens']
    if (not set(required) <= set(meta) or len(meta) != expected[0] or
            meta[required].isna().any().any() or not meta.base_id.is_unique or
            not meta.label.isin([0, 1]).all() or not meta.prompt_tokens.gt(0).all() or
            ordered_base_id_sha256(meta.base_id.astype(str)) != manifest['ordered_base_id_sha256']):
        raise ValueError('Feature row metadata invalid')
    validate_feature_cache(Path(manifest['path']), expected, 'float16', manifest['ordered_base_id_sha256'], meta.base_id.astype(str))
    return np.load(manifest['path'], mmap_mode='r', allow_pickle=False), meta, manifest


@contextmanager
def adapted_heads(folder):
    """Preserve old training math; correct adapter provenance before hash locking."""
    from scripts.hanguard import frozen_base_readout_heads as heads
    from scripts.hanguard import frozen_base_resumable_heads as resumable
    folder = Path(folder)
    p = verify_readouts(folder)
    patches = []
    def patch(module, key, value):
        patches.append((module, key, getattr(module, key)))
        setattr(module, key, value)
    def adapter_dump(path, value):
        value = dict(value) if isinstance(value, dict) else value
        if isinstance(value, dict) and 'no_lora' in value:
            value.update(no_lora=False, use_lora=True, adapter_sha256=p['adapter_sha256'],
                adapter_path=p['adapter_path'])
            if 'scope' in value:
                value['scope'] = 'One freshly trained E03 frozen adapter; inspected release identities; validation-only readout selection'
        write(path, value)
    try:
        for module in (heads, resumable):
            patch(module, 'load_cache', load_cache)
            patch(module, 'SCHEMA', READOUT_SCHEMA)
            patch(module, 'dump', adapter_dump)
        yield heads, resumable
    finally:
        for module, key, prior in reversed(patches):
            setattr(module, key, prior)


def worker_ledger(study, worker_id):
    return Path(study) / 'workers' / f'{worker_id}.json'


def admit_worker(study, worker_id):
    """Same exact child must own its card before slow hashing and model setup."""
    path = worker_ledger(study, worker_id)
    registration_path = path.with_suffix('.registration.json')
    deadline = time.monotonic() + 10
    while not path.exists() or not registration_path.exists():
        check_stop(study)
        if time.monotonic() > deadline:
            raise ValueError('Exact worker ledger was not published')
        time.sleep(.05)
    ledger = read(path)
    p = verify(study)
    if (read(registration_path)['ledger_sha256'] != sha(path) or
            ledger['protocol_sha256'] != sha(Path(study) / 'protocol.json') or
            not same_process(ledger['identity'], identity(os.getpid())) or
            ledger['gpu'] not in p['gpu_ids'] or os.environ.get('CUDA_VISIBLE_DEVICES') != str(ledger['gpu'])):
        raise ValueError('Exact worker/card/protocol identity mismatch')
    card = next(row for row in inventory() if row['index'] == ledger['gpu'])
    if any(pid != os.getpid() for pid in card['pids']):
        return False
    import torch
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError('Worker must see exactly one card')
    # This allocation establishes the real work context, never a placeholder job.
    readiness = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    if not torch.isfinite(readiness).all():
        raise RuntimeError('CUDA allocation readiness failed')
    write(path.with_suffix('.admitted.json'), dict(admitted=True, pid=os.getpid(), gpu=ledger['gpu'],
        identity=ledger['identity'], protocol_sha256=ledger['protocol_sha256'], admitted_unix=time.time()))
    return True


def worker(study, worker_id, phase, split=None):
    study = Path(study).resolve()
    if not admit_worker(study, worker_id):
        return 75  # CPU admission race only; safe to wait for a different empty card.
    verify(study, full_hash=True)
    if phase == 'lora':
        from scripts.hanguard import current_training as training
        from scripts.hanguard import remediated_binary_study as binary
        from scripts.hanguard import repaired_study as legacy
        import gc
        import torch
        p = verify(study)
        args = SimpleNamespace(output=str(study / 'lora'), arm='E03', seed=p['seed'],
            steps=2, include_longest=True, include_warmup=True, preflight_cycles=1)
        training.verify(study / 'lora')
        previous = legacy.load_split
        legacy.load_split = binary.load_split
        try:
            evidence = study / 'lora/preflight' / f'E03_s{p["seed"]}.json'
            if not evidence.exists():
                legacy.preflight(args)
                gc.collect()
                torch.cuda.empty_cache()
            if read(evidence).get('passed') is not True:
                raise ValueError('E03 preflight did not pass')
            marker = study / 'lora/training_started.json'
            if not marker.exists():
                write(marker, dict(training_started=True, pid=os.getpid(), started_unix=time.time(),
                    protocol_sha256=sha(study / 'lora/protocol.json')))
            legacy.train(args)
        finally:
            legacy.load_split = previous
        training.verify(study / 'lora')
        selected_adapter(study)
    elif phase == 'extract':
        if split not in SPLITS:
            raise ValueError('Extraction worker requires a split')
        extract_features(study, worker_id, split)
    elif phase in ('fit', 'evaluate'):
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        folder = study / 'readouts'
        with adapted_heads(folder) as (heads, resumable):
            if phase == 'fit':
                resumable.fit(folder, device='cuda', extra_source_paths=[folder / 'adapter_readout_runtime.py.snapshot'])
            else:
                heads.evaluate(folder, device='cuda')
    else:
        raise ValueError('Unknown worker phase')
    write(worker_ledger(study, worker_id).with_suffix('.complete.json'), dict(complete=True,
        phase=phase, split=split, protocol_sha256=sha(study / 'protocol.json')))
    return 0


class Scheduler:
    """Empty admission, exact process ownership, and record-only foreign arrivals."""
    def __init__(self, study, *, inventory_reader=inventory, identity_reader=identity,
                 popen=subprocess.Popen, sleeper=time.sleep):
        self.study = Path(study).resolve()
        self.protocol = verify(self.study)
        self.inventory_reader, self.identity_reader = inventory_reader, identity_reader
        self.popen, self.sleep = popen, sleeper
        self.active, self.arrivals = {}, set()
        self.sequence = len(list((self.study / 'workers').glob('*.registration.json')))

    def cards(self):
        try:
            cards = self.inventory_reader()
        except Exception as error:
            state(self.study, 'gpu_inventory_unavailable', gpu_inventory_error=repr(error), active_workers=len(self.active))
            return []  # No admission without telemetry; keep already admitted work running.
        for record in self.active.values():
            row = next((row for row in cards if row['index'] == record['gpu']), None)
            if row is None:
                continue
            for pid in row['pids']:
                if pid == record['identity']['pid']:
                    continue
                key = (record['worker_id'], pid)
                if key not in self.arrivals:
                    self.arrivals.add(key)
                    try:
                        owner = self.identity_reader(pid)
                    except Exception as error:
                        owner = dict(pid=pid, identity_error=repr(error))
                    write(self.study / 'foreign_arrivals' / f'{time.time_ns()}_{pid}.json',
                        dict(action='record_only_continue_owned_worker', gpu=record['gpu'],
                            worker_id=record['worker_id'], process=owner))
        return cards

    def launch(self, gpu, phase, split=None):
        cards = self.cards()
        if gpu not in empty_cards(cards, self.protocol['gpu_ids'], self.protocol['minimum_free_gib']):
            return False
        self.sequence += 1
        worker_id = f'{phase}_{split or "none"}_{self.sequence:05d}_gpu{gpu}'
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output', str(self.study),
            '--worker-id', worker_id, '--phase', phase]
        if split is not None:
            command += ['--split', split]
        directory = self.study / 'workers'
        directory.mkdir(exist_ok=True)
        with (directory / f'{worker_id}.log').open('x') as stream:
            child = self.popen(command, cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        exact = None
        for _ in range(40):
            if child.poll() is not None:
                break
            try:
                current = self.identity_reader(child.pid)
                if current['uid'] == os.getuid() and current['argv'] == command and current['cwd'] == str(ROOT):
                    exact = current
                    break
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
            self.sleep(.05)
        if exact is None:
            if child.poll() is None:
                child.terminate()  # Exact Popen child, never a card-wide kill.
                child.wait(timeout=20)
            raise RuntimeError('New CPU worker identity could not be registered')
        ledger = dict(worker_id=worker_id, identity=exact, command=command, gpu=gpu,
            phase=phase, split=split, protocol_sha256=sha(self.study / 'protocol.json'))
        write(worker_ledger(self.study, worker_id), ledger)
        write(worker_ledger(self.study, worker_id).with_suffix('.registration.json'),
            dict(ledger_sha256=sha(worker_ledger(self.study, worker_id))))
        self.active[worker_id] = dict(ledger, child=child)
        return True

    def poll(self):
        for worker_id, record in list(self.active.items()):
            child = record['child']
            code = child.poll()
            if code is None:
                try:
                    actual = self.identity_reader(record['identity']['pid'])
                    exact = same_process(record['identity'], actual)
                except (FileNotFoundError, ProcessLookupError, ValueError):
                    exact = False
                if not exact:
                    # /proc can disappear immediately before Popen observes exit.
                    for _ in range(4):
                        code = child.poll()
                        if code is not None:
                            break
                        self.sleep(.05)
                    if code is None:
                        raise RuntimeError('Live worker process identity became unverifiable')
                else:
                    continue
            write(worker_ledger(self.study, worker_id).with_suffix('.exit.json'), dict(exit_code=code, exited_unix=time.time()))
            del self.active[worker_id]
            if code == 75 and not worker_ledger(self.study, worker_id).with_suffix('.admitted.json').exists():
                continue
            if code != 0 or not worker_ledger(self.study, worker_id).with_suffix('.complete.json').exists():
                raise RuntimeError(f'{worker_id} failed with exit {code}; inspect its log')

    def stop_owned(self):
        for record in self.active.values():
            child = record['child']
            if child.poll() is None:
                # Popen still owns this unreaped child; its PID cannot be reused.
                child.terminate()
        for record in self.active.values():
            try:
                record['child'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                record['child'].kill()
                record['child'].wait(timeout=10)
        self.active.clear()

    def phase(self, phase, completed, *, split=None, parallel=False, work_remaining=None):
        while True:
            check_stop(self.study)
            self.poll()
            done = completed()
            cards = self.cards()
            if done and not self.active:
                return
            if not done and (work_remaining is None or work_remaining()):
                available = empty_cards(cards, self.protocol['gpu_ids'], self.protocol['minimum_free_gib'])
                occupied = {record['gpu'] for record in self.active.values()}
                for gpu in available:
                    if gpu in occupied or (not parallel and self.active):
                        continue
                    self.launch(gpu, phase, split)
                    if not parallel:
                        break
            state(self.study, f'{phase}_{split or "work"}',
                training_started=(self.study / 'lora/training_started.json').exists(),
                active_workers=len(self.active), active_gpus=[r['gpu'] for r in self.active.values()],
                gpu_candidates=cards, automatic_wait_for_empty_gpu=True)
            self.sleep(self.protocol['poll_seconds'])


def verify_retired_workers(study):
    for path in (Path(study) / 'workers').glob('*.json'):
        if any(word in path.name for word in ('.registration.', '.exit.', '.complete.', '.admitted.')):
            continue
        ledger = read(path)
        expected = ledger['identity']
        try:
            actual = identity(expected['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if same_process(expected, actual):
            raise RuntimeError(f'Prior exact worker is still alive: {expected["pid"]}')
        if actual.get('start_time_ticks') == expected['start_time_ticks']:
            raise RuntimeError('Prior worker changed command within the same process birth')


def release_dead_claims(study):
    """Only explicit resume can retire abandoned, never completed chunk claims."""
    for claim in (Path(study) / 'readouts/features').glob('*/chunks/*.claim.json'):
        value = read(claim)
        expected = value['identity']
        try:
            actual = identity(expected['pid'])
        except (FileNotFoundError, ProcessLookupError):
            actual = None
        if actual and actual.get('start_time_ticks') == expected['start_time_ticks']:
            raise RuntimeError('Feature claim belongs to a still-live process')
        complete = claim.with_name(claim.name.replace('.claim.json', '.complete.json'))
        if not complete.exists():
            claim.rename(claim.with_name(f'{claim.name}.{time.time_ns()}.abandoned'))


def run(study, resume=False):
    study = Path(study).resolve()
    verify(study, full_hash=True)
    with (study / 'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(study / 'state.json')
        if prior['state'] == 'completed':
            return read(study / 'readouts/analysis/results.json')
        if prior['state'] != 'registered' and not resume:
            raise RuntimeError('Interrupted study requires explicit --resume after reviewing state/logs')
        verify_retired_workers(study)
        if resume:
            release_dead_claims(study)
        scheduler = Scheduler(study)
        try:
            p = verify(study)
            selected = study / 'lora/runs' / f'E03_s{p["seed"]}/selection.json'
            scheduler.phase('lora', selected.exists)
            folder = study / 'readouts'
            initialize_readouts(study)
            for split in ('train', 'validation'):
                prepare_features(folder, split)
                scheduler.phase('extract', lambda s=split: chunks_complete(folder, s), split=split,
                    parallel=True, work_remaining=lambda s=split: any(
                        not chunk_receipt(folder, s, i).exists() and not
                        (folder / 'features' / s / 'chunks' / f'{i:05d}.claim.json').exists()
                        for i in range(len(verify_plan(folder, s)['chunks']))))
                merge_features(folder, split)
            scheduler.phase('fit', (folder / 'selection_lock.json').exists)
            require_barrier(folder, 'test')
            prepare_features(folder, 'test')
            state(study, 'test_extraction_prepared', formal_test_opened=True)
            scheduler.phase('extract', lambda: chunks_complete(folder, 'test'), split='test', parallel=True,
                work_remaining=lambda: any(not chunk_receipt(folder, 'test', i).exists() and not
                    (folder / 'features/test/chunks' / f'{i:05d}.claim.json').exists()
                    for i in range(len(verify_plan(folder, 'test')['chunks']))))
            merge_features(folder, 'test')
            scheduler.phase('evaluate', (folder / 'analysis/complete.json').exists)
            result = read(folder / 'analysis/results.json')
            if result.get('complete') is not True or result.get('no_lora') is not False:
                raise ValueError('Adapter-aware evaluation did not complete')
            state(study, 'completed', active_workers=0, active_gpus=[], automatic_wait_for_empty_gpu=False)
            return result
        except BaseException as error:
            scheduler.stop_owned()
            state(study, 'needs_review' if isinstance(error, NeedsReview) else 'requires_review', error_type=type(error).__name__, reason=str(error),
                active_workers=0, automatic_wait_for_empty_gpu=False)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('init', 'run', 'status', 'worker'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data/three_source_fragment_clean_20261018')
    p.add_argument('--model', type=Path, default=ROOT / 'models/Qwen3.5-4B')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lora-epochs', type=int, default=3)
    p.add_argument('--minimum-free-gib', type=float, default=36.)
    p.add_argument('--poll-seconds', type=float, default=2.)
    p.add_argument('--chunk-groups', type=int, default=24)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-id')
    p.add_argument('--phase', choices=('lora', 'extract', 'fit', 'evaluate'))
    p.add_argument('--split', choices=SPLITS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    if args.action == 'init':
        p = initialize(args)
        print(json.dumps(dict(state='registered', output=str(args.output), seed=p['seed'], gpu_policy=POLICY), ensure_ascii=False))
    elif args.action == 'run':
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        result = run(args.output, resume=args.resume)
        print(json.dumps(dict(complete=result.get('complete'), rows=result.get('rows')), ensure_ascii=False))
    elif args.action == 'status':
        print(json.dumps(read(args.output / 'state.json'), ensure_ascii=False, indent=2))
    else:
        if not args.worker_id or not args.phase:
            raise ValueError('worker requires --worker-id and --phase')
        raise SystemExit(worker(args.output, args.worker_id, args.phase, args.split))


if __name__ == '__main__':
    main()
