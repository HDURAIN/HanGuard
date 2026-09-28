"""Frozen E04 features and three heads for registered single primary categories.

The legacy task retains its original-label contract. The explicit intent task
accepts audited WildGuard-only relabels. Only known IDs 1..5 contribute type CE;
the frozen binary gate supplies category 0 at inference.
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.hanguard import multilabel_study as shared
from scripts.hanguard import primary_category_contract as contract
from scripts.hanguard.repaired_study import dump, sha, atomic_torch_save, cpu_state, batch_order, microbatches

ARMS = shared.ARMS
LABEL_IDS = shared.LABEL_IDS


def read_protocol(study):
    protocol = shared.read_protocol(study)
    if protocol.get('task') not in ('existing_primary_category_classification', contract.TASK):
        raise ValueError('This trainer requires the registered primary-category task')
    if protocol.get('new_multilabel_annotations_used') is not False:
        raise ValueError('Supplementary annotations must be explicitly excluded')
    if set(protocol['all_arms']) != set(ARMS) or len(protocol['all_arms']) != 3:
        raise ValueError('All three prespecified arms are required')
    seeds = protocol['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Unique nonnegative integer training seeds are required')
    widths = protocol.get('head_width_by_arm', {})
    if not isinstance(widths, dict) or set(widths) - set(ARMS) or any(type(width) is not int or width < 1 for width in widths.values()):
        raise ValueError('Per-arm head widths must be positive integers for known arms')
    early_stopping_progress([], protocol)
    if contract.is_intent(protocol):
        contract.validate_protocol(protocol)
    elif Path(protocol['data_dir']).resolve() != Path(protocol['input_data_dir']).resolve():
        raise ValueError('Use the original input labels, not an annotation export')
    if set(protocol.get('data_sha256', {})) != {'train', 'validation', 'test'}:
        raise ValueError('Register the three original labeled dataset hashes')
    cache = Path(protocol.get('feature_cache_dir', study / 'feature_cache'))
    if cache.resolve() != (study / 'feature_cache').resolve():
        raise ValueError('Feature cache path and study symlink disagree')
    _, description_hash = shared.read_descriptions(protocol)
    if protocol.get('description_sha256') != description_hash:
        raise ValueError('Registered category descriptions changed')
    return protocol


def cache_manifest(study, protocol):
    manifest = shared.cache_manifest(study, protocol)
    cache = study / 'feature_cache'
    common_keys = ('parent', 'text_identities', 'description_sha256', 'label_order',
                   'layer', 'dtype', 'input', 'pad_multiple')
    if shared.json_sha({key: manifest[key] for key in common_keys}) != manifest['fingerprint']:
        raise ValueError('Feature cache fingerprint is invalid')
    if manifest['label_order'] != list(LABEL_IDS) or manifest['layer'] != 32:
        raise ValueError('Expected five descriptions and last-layer token features')
    if manifest['dtype'] != 'bfloat16_uint16_bits' or manifest['pad_multiple'] != protocol['pad_multiple']:
        raise ValueError('Feature cache input settings differ from the protocol')
    descriptions, description_hash = shared.read_descriptions(protocol)
    sidecar = json.loads((cache / 'descriptions.json').read_text())
    if sidecar.get('descriptions') != descriptions or sidecar.get('description_sha256') != description_hash:
        raise ValueError('Cached description text identity differs')
    if sidecar.get('embedding_sha256') != manifest['description_embedding_sha256']:
        raise ValueError('Description embedding metadata differs')
    return manifest


def category_arrays(frame, protocol=None):
    """Preserve one existing class; safe rows have no type-loss target."""
    if protocol is not None and contract.is_intent(protocol):
        categories, indices, _, _ = contract.label_arrays(frame)
        return categories, indices
    if 'label_origin' in frame and frame.label_origin.eq('wildguard_intent_reannotation').any():
        raise ValueError('Intent reannotations require the separate registered intent task')
    if 'category_id' not in frame or frame.category_id.isna().any():
        raise ValueError('Existing category_id is required for every row')
    numeric = pd.to_numeric(frame.category_id, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, np.arange(6)).all():
        raise ValueError('Expected original integral category_id in 0..5')
    categories = numeric.astype(np.int64)
    if 'prompt_harm_label' in frame:
        labels = frame.prompt_harm_label
        if not labels.isin(['harmful', 'unharmful']).all():
            raise ValueError('Invalid original binary label')
        if not np.array_equal(labels.eq('harmful').to_numpy(), categories > 0):
            raise ValueError('Existing category and binary labels contradict each other')
    return categories, np.flatnonzero(categories > 0)


def load_cached_split(study, protocol, manifest, split, device='cuda'):
    path = Path(protocol['data_dir']) / f'{split}.parquet'
    dataset_hash = sha(path)
    if dataset_hash != protocol['data_sha256'][split]:
        raise ValueError(f'Original {split} labels changed after registration')
    frame = shared.text_frame(path)
    if protocol.get('experiment_scope') == 'two_source_full_primary_category' and not frame.source.isin(['chinese_curated', 'jailbench']).all():
        raise ValueError('Two-source study cannot include WildGuard or another source')
    if protocol.get('rows', {}).get(split, len(frame)) != len(frame):
        raise ValueError('Dataset row count differs from registration')
    identity = shared.text_identity(frame)
    if identity != manifest['text_identities'][split]:
        raise ValueError(f'{split} source IDs or full prompts differ from cached inputs')
    if contract.is_intent(protocol):
        contract.validate_frame(frame, protocol, original=contract.original_frame(protocol, split))
    categories, harmful = category_arrays(frame, protocol)
    if len(harmful) == 0:
        raise ValueError('A split must contain harmful type supervision')
    cache = study / 'feature_cache'
    info = manifest['splits'][split]
    if info['rows'] != len(frame) or info['text_identity'] != identity:
        raise ValueError('Cached split identity mismatch')
    for name, key in [(f'{split}_index.parquet', 'index_sha256'),
                      (f'{split}_offsets.npy', 'offsets_sha256'),
                      (f'{split}_tokens.npy', 'tokens_sha256')]:
        if sha(cache / name) != info[key]:
            raise ValueError(f'Feature cache changed: {name}')
    meta = pd.read_parquet(cache / f'{split}_index.parquet')
    if meta.base_id.tolist() != frame.base_id.tolist() or meta.source.tolist() != frame.source.tolist():
        raise ValueError('Feature cache row ordering or source provenance mismatch')
    if meta.prompt_sha256.tolist() != frame.prompt.map(shared.text_sha).tolist():
        raise ValueError('Feature cache full-text hashes mismatch')
    binary_p = meta.binary_probability.to_numpy(dtype=float)
    if not np.isfinite(binary_p).all() or not ((binary_p >= 0) & (binary_p <= 1)).all():
        raise ValueError('Invalid cached binary probabilities')
    original_binary = (frame.original_prompt_harm_label.eq('harmful').to_numpy()
                       if contract.is_intent(protocol) else categories > 0)
    if 'original_binary_label' in meta and not np.array_equal(meta.original_binary_label.to_numpy(), original_binary):
        raise ValueError('Cache binary labels disagree with original categories')
    offsets = np.load(cache / f'{split}_offsets.npy', allow_pickle=False)
    bits = np.load(cache / f'{split}_tokens.npy', mmap_mode='r', allow_pickle=False)
    if offsets.shape != (len(frame) + 1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0:
        raise ValueError('Invalid cached offsets')
    lengths = np.diff(offsets)
    if (lengths <= 0).any() or lengths.max() > protocol['max_tokens']:
        raise ValueError('Invalid full-input lengths; truncation is forbidden')
    if not np.array_equal(lengths, meta.prompt_tokens.to_numpy()):
        raise ValueError('Cached token lengths mismatch')
    if bits.dtype != np.uint16 or bits.shape != (int(offsets[-1]), manifest['hidden_size']):
        raise ValueError('Invalid BF16 cache shape')
    if int(offsets[-1]) != info['tokens'] or int(lengths.max()) != info['maximum_tokens']:
        raise ValueError('Cache token count mismatch')
    if protocol['cache_on_gpu']:
        if str(device).startswith('cuda') and bits.size * 2 > torch.cuda.mem_get_info()[0] * .8:
            raise ValueError('Insufficient GPU space for the registered feature cache')
        tokens = torch.empty(bits.shape, dtype=torch.bfloat16, device=device)
        for start in range(0, len(bits), 8192):
            block = torch.from_numpy(np.array(bits[start:start + 8192], copy=True)).view(torch.bfloat16)
            tokens[start:start + len(block)].copy_(block)
    else:
        tokens = bits
    meta = meta.copy()
    meta['category_id'] = categories
    if contract.is_intent(protocol):
        for column in contract.PROVENANCE_COLUMNS:
            meta[column] = frame[column].to_numpy()
    return dict(tokens=tokens, offsets=offsets, device_offsets=torch.tensor(offsets, device=device),
                lengths=lengths, meta=meta, categories=categories, harmful_indices=harmful,
                y=torch.tensor(categories - 1, device=device), dataset_sha256=dataset_hash,
                text_identity=identity)


def multiclass_metrics(labels, predictions, classes):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    classes = list(classes)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError('Aligned one-dimensional class labels are required')
    if not np.isin(labels, classes).all() or not np.isin(predictions, classes).all():
        raise ValueError('Prediction or label outside the declared class set')
    mapping = {value: index for index, value in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, output in zip(labels, predictions):
        confusion[mapping[int(target)], mapping[int(output)]] += 1
    per_class = []
    for index, category in enumerate(classes):
        tp = int(confusion[index, index]); support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum()); fp = predicted - tp; fn = support - tp
        per_class.append(dict(category_id=category, label=str(category), support=support,
                              predicted=predicted, tp=tp, fp=fp, fn=fn,
                              precision=tp / max(1, predicted), recall=tp / max(1, support),
                              f1=2 * tp / max(1, 2 * tp + fp + fn),
                              support_flags=['no_true_support'] if support == 0 else []))
    return dict(rows=len(labels), accuracy=float((labels == predictions).mean()) if len(labels) else None,
                macro_f1=float(np.mean([item['f1'] for item in per_class])),
                per_class=per_class, confusion_matrix=confusion.tolist(), classes=classes)


def decisions(logits, binary_probability, binary_threshold):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    binary_probability = np.asarray(binary_probability, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 5 or binary_probability.shape != (len(logits),):
        raise ValueError('Expected five type logits and one binary probability per row')
    if not torch.isfinite(logits).all() or not np.isfinite(binary_probability).all():
        raise ValueError('Nonfinite predictions')
    if not ((binary_probability >= 0) & (binary_probability <= 1)).all() or not 0 <= binary_threshold <= 1:
        raise ValueError('Invalid parent binary probabilities or threshold')
    probabilities = logits.softmax(dim=-1).numpy()
    predicted = probabilities.argmax(axis=1) + 1
    gated = np.where(binary_probability >= binary_threshold, predicted, 0)
    return probabilities, predicted, gated


def metric_pair(categories, predicted, gated):
    categories = np.asarray(categories)
    harmful = categories > 0
    known = categories >= 0
    return dict(type_metrics=multiclass_metrics(categories[harmful], np.asarray(predicted)[harmful], range(1, 6)),
                gated_metrics=multiclass_metrics(categories[known], np.asarray(gated)[known], range(6)),
                coverage=contract.coverage(categories))


@torch.no_grad()
def prediction(head, data, protocol):
    head.eval()
    order = np.argsort(data['lengths'], kind='stable')
    logits = np.empty((len(order), 5), dtype=np.float32)
    for indices in microbatches(order, data['lengths'], protocol['token_budget'], protocol['max_micro'], protocol['pad_multiple']):
        hidden, mask = shared.feature_batch(data, indices, protocol)
        values = head(hidden, mask)['logits'].float()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite type logits')
        logits[indices] = values.cpu().numpy()
    harmful = data['harmful_indices']
    ce = float(F.cross_entropy(torch.from_numpy(logits[harmful]), torch.tensor(data['categories'][harmful] - 1)))
    return logits, ce


def save_predictions(data, logits, path, binary_threshold):
    columns = ['base_id', 'source', 'prompt_sha256', 'prompt_tokens', 'category_id', 'binary_probability']
    columns += [column for column in contract.PROVENANCE_COLUMNS if column in data['meta']]
    frame = data['meta'][columns].copy()
    probabilities, predicted, gated = decisions(logits, frame.binary_probability.to_numpy(), binary_threshold)
    for column, key in enumerate(LABEL_IDS):
        frame[f'p_{key}'] = probabilities[:, column]
    frame['predicted_category'] = predicted
    frame['gated_category'] = gated
    frame.to_csv(path, index=False)
    return metric_pair(data['categories'], predicted, gated)


def macro_step(head, optimizer, data, indices, protocol, rng_seed):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or not (data['categories'][indices] > 0).all():
        raise ValueError('Every type-training row must have an existing harmful class')
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for number, small in enumerate(microbatches(indices, data['lengths'], protocol['token_budget'], protocol['max_micro'], protocol['pad_multiple'])):
        torch.manual_seed(rng_seed + number)
        hidden, mask = shared.feature_batch(data, small, protocol)
        if hidden.requires_grad:
            raise AssertionError('Backbone features must remain frozen')
        output = head(hidden, mask)
        loss = F.cross_entropy(output['logits'], data['y'][small], reduction='sum')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite type cross entropy')
        (loss / len(indices)).backward()
        loss_sum += float(loss.detach())
    norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    return dict(ce=loss_sum / len(indices), grad_norm=norm, rows=len(indices))


def run_identity(study, protocol, manifest, arm, seed):
    if arm not in protocol['all_arms'] or seed not in protocol['seeds']:
        raise ValueError('Arm and seed must be registered')
    value = dict(arm=arm, seed=seed, protocol_sha256=sha(study / 'protocol.json'),
                cache_fingerprint=manifest['fingerprint'],
                train_dataset_sha256=protocol['data_sha256']['train'],
                validation_dataset_sha256=protocol['data_sha256']['validation'],
                code_sha256=sha(Path(__file__)), heads_sha256=sha(ROOT / 'scripts/hanguard/multilabel_heads.py'))
    if contract.is_intent(protocol):
        value.update(contract_sha256=sha(Path(contract.__file__)),
                     annotation_manifest_sha256=protocol['annotation_manifest_sha256'],
                     annotation_protocol_sha256=protocol['annotation_protocol_sha256'])
    return value


def setup_head(study, protocol, manifest, arm, seed):
    configured = dict(protocol)
    configured['head_width'] = protocol.get('head_width_by_arm', {}).get(arm, protocol['head_width'])
    return shared.setup_head(study, configured, manifest, arm, seed)


def early_stopping_progress(history, protocol):
    """Reconstruct validation-only patience from history for exact resumption.

    Checkpoint selection still uses the actual minimum CE; min_delta controls
    only when patience resets, so a small improvement is never lost.
    """
    patience = protocol.get('early_stopping_patience', 0)
    delta = protocol.get('early_stopping_min_delta', 0.)
    if type(patience) is not int or patience < 0 or not isinstance(delta, (int, float)) or not np.isfinite(delta) or delta < 0:
        raise ValueError('Invalid validation early-stopping patience or min_delta')
    reference = float('inf'); stale = 0
    for record in history:
        ce = record['validation_ce']
        if not np.isfinite(ce):
            raise ValueError('Early stopping requires finite validation CE')
        if ce < reference and reference - ce >= delta:
            reference = ce; stale = 0
        else:
            stale += 1
    return dict(enabled=patience > 0, patience=patience, min_delta=delta,
                reference_ce=reference if history else None, stale_epochs=stale,
                should_stop=bool(patience and stale >= patience))


def train(args):
    study = Path(args.output); protocol = read_protocol(study); manifest = cache_manifest(study, protocol)
    identity = run_identity(study, protocol, manifest, args.arm, args.seed)
    directory = study / 'runs' / f'{args.arm}_s{args.seed}'; directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'selection.json').exists():
        previous = json.loads((directory / 'selection.json').read_text())
        if any(previous.get(key) != value for key, value in identity.items()) or previous['checkpoint_sha256'] != sha(directory / 'head.pt'):
            raise ValueError('Completed run identity or checkpoint changed')
        return
    began = time.monotonic()
    dump(directory / 'status.json', dict(state='loading_features', **identity, pid=os.getpid(), updated=time.time()))
    data = load_cached_split(study, protocol, manifest, 'train')
    validation = load_cached_split(study, protocol, manifest, 'validation')
    head = setup_head(study, protocol, manifest, args.arm, args.seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol['head_lr'], weight_decay=protocol['weight_decay'])
    harmful = data['harmful_indices']; steps_epoch = math.ceil(len(harmful) / protocol['effective_batch'])
    threshold = manifest['parent']['binary_threshold']
    best = float('inf'); best_epoch = None; best_logits = None; history = []; start_epoch = 0
    resume = directory / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume protocol, data, cache, or code changed')
        head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']; best = state['best']; best_epoch = state['best_epoch']
        best_logits = state['best_logits']; history = state['history']
    for epoch in range(start_epoch, protocol['epochs']):
        if early_stopping_progress(history, protocol)['should_stop']:
            break
        head.train(); epoch_began = time.monotonic(); loss_sum = 0.; rows = 0
        order = harmful[batch_order(data['lengths'][harmful], args.seed * 1009 + epoch)]
        for cursor, start in enumerate(range(0, len(order), protocol['effective_batch'])):
            ids = order[start:start + protocol['effective_batch']]
            result = macro_step(head, optimizer, data, ids, protocol, args.seed * 1000003 + epoch * 10009 + cursor * 131)
            loss_sum += result['ce'] * len(ids); rows += len(ids)
            if cursor % 10 == 0 or cursor + 1 == steps_epoch:
                dump(directory / 'status.json', dict(state='training', **identity, epoch=epoch + 1, epochs=protocol['epochs'],
                     step=epoch * steps_epoch + cursor + 1, total_steps=protocol['epochs'] * steps_epoch,
                     rows_seen=epoch * len(harmful) + rows, harmful_train_rows=len(harmful), train_ce=loss_sum / rows,
                     seconds=time.monotonic() - began, pid=os.getpid(), updated=time.time()))
        logits, validation_ce = prediction(head, validation, protocol)
        _, predicted, gated = decisions(logits, validation['meta'].binary_probability.to_numpy(), threshold)
        metrics = metric_pair(validation['categories'], predicted, gated)
        record = dict(epoch=epoch + 1, train_ce=loss_sum / rows, validation_ce=validation_ce,
                      validation_type_metrics=metrics['type_metrics'], validation_gated_metrics=metrics['gated_metrics'],
                      seconds=time.monotonic() - epoch_began)
        history.append(record)
        record['early_stopping'] = early_stopping_progress(history, protocol)
        if validation_ce < best:
            best = validation_ce; best_epoch = epoch + 1; best_logits = logits.copy()
            atomic_torch_save(cpu_state(head), directory / 'head.pt')
            save_predictions(validation, best_logits, directory / 'validation.csv', threshold)
        dump(directory / 'history.json', history)
        atomic_torch_save(dict(identity=identity, head=cpu_state(head), optimizer=optimizer.state_dict(), epoch=epoch + 1,
                              best=best, best_epoch=best_epoch, best_logits=best_logits, history=history), resume)
    if best_logits is None or not np.isfinite(best):
        raise ValueError('No finite validation-selected checkpoint')
    metrics = save_predictions(validation, best_logits, directory / 'validation.csv', threshold)
    result = dict(**identity, best_epoch=best_epoch, validation_ce=best,
                  validation_type_metrics=metrics['type_metrics'], validation_gated_metrics=metrics['gated_metrics'],
                  checkpoint_sha256=sha(directory / 'head.pt'), head_parameters=sum(p.numel() for p in head.parameters()),
                  parent_binary_threshold=threshold, binary_parent=manifest['parent'],
                  harmful_train_rows=len(harmful), safe_train_rows_excluded=int((data['categories'] == 0).sum()),
                  harmful_validation_rows=len(validation['harmful_indices']), total_steps=len(history) * steps_epoch,
                  maximum_steps=protocol['epochs'] * steps_epoch, completed_epochs=len(history),
                  stopped_early=len(history) < protocol['epochs'],
                  early_stopping=early_stopping_progress(history, protocol),
                  head_width=protocol.get('head_width_by_arm', {}).get(args.arm, protocol['head_width']),
                  seconds=time.monotonic() - began, history=history,
                  unknown_train_rows_excluded=int((data['categories'] < 0).sum()),
                  unknown_validation_rows_excluded=int((validation['categories'] < 0).sum()),
                  limitation=('WildGuard intent labels are machine reannotations; other sources retained; not human gold'
                              if contract.is_intent(protocol) else
                              'Existing single primary categories include machine labels; not human gold or multilabel evaluation'))
    dump(directory / 'selection.json', result); resume.unlink(missing_ok=True)
    dump(directory / 'status.json', dict(state='selected', **identity, seconds=result['seconds'], updated=time.time()))


def validate_test_barrier(study, protocol):
    return shared.validate_test_barrier(study, protocol)


def evaluate(args):
    study = Path(args.output); protocol = read_protocol(study); manifest = cache_manifest(study, protocol)
    validate_test_barrier(study, protocol)
    identity = run_identity(study, protocol, manifest, args.arm, args.seed)
    directory = study / 'runs' / f'{args.arm}_s{args.seed}'
    selection = json.loads((directory / 'selection.json').read_text())
    if any(selection.get(key) != value for key, value in identity.items()):
        raise ValueError('Selected study identity changed')
    if selection['checkpoint_sha256'] != sha(directory / 'head.pt'):
        raise ValueError('Selected head checkpoint changed')
    if selection['parent_binary_threshold'] != manifest['parent']['binary_threshold']:
        raise ValueError('Selected parent binary gate changed')
    if (directory / 'test_results.json').exists():
        existing = json.loads((directory / 'test_results.json').read_text())
        if existing['test_dataset_sha256'] != protocol['data_sha256']['test'] or existing['checkpoint_sha256'] != selection['checkpoint_sha256']:
            raise ValueError('Existing test result belongs to different data/checkpoint')
        return
    data = load_cached_split(study, protocol, manifest, 'test')
    head = setup_head(study, protocol, manifest, args.arm, args.seed)
    head.load_state_dict(torch.load(directory / 'head.pt', weights_only=True, map_location='cpu'))
    logits, ce = prediction(head, data, protocol)
    threshold = manifest['parent']['binary_threshold']
    metrics = save_predictions(data, logits, directory / 'test.csv', threshold)
    _, predicted, gated = decisions(logits, data['meta'].binary_probability.to_numpy(), threshold)
    sources = data['meta'].source.to_numpy()
    by_source = {str(source): metric_pair(data['categories'][sources == source], predicted[sources == source], gated[sources == source])
                 for source in sorted(set(sources))}
    result = dict(arm=args.arm, seed=args.seed, **metrics, by_source=by_source, test_ce=ce,
                  test_dataset_sha256=data['dataset_sha256'], checkpoint_sha256=selection['checkpoint_sha256'],
                  parent_binary_threshold=threshold, binary_parent=manifest['parent'],
                  protocol_sha256=identity['protocol_sha256'], cache_fingerprint=manifest['fingerprint'])
    dump(directory / 'test_results.json', result)
    dump(directory / 'status.json', dict(state='complete', arm=args.arm, seed=args.seed, updated=time.time()))


def preflight(args):
    study = Path(args.output); protocol = read_protocol(study); manifest = cache_manifest(study, protocol)
    identity = run_identity(study, protocol, manifest, args.arm, args.seed)
    data = load_cached_split(study, protocol, manifest, 'train')
    head = setup_head(study, protocol, manifest, args.arm, args.seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=protocol['head_lr'], weight_decay=protocol['weight_decay'])
    harmful = data['harmful_indices']
    ordinary = harmful[:min(protocol['effective_batch'], len(harmful))]
    longest_harmful = harmful[int(data['lengths'][harmful].argmax())]
    longest_all = int(data['lengths'].argmax())
    records = []; torch.cuda.reset_peak_memory_stats(); head.train()
    for label, ids in [('ordinary_macro', ordinary), ('longest_harmful', np.asarray([longest_harmful]))]:
        for step in range(args.steps):
            torch.cuda.synchronize(); began = time.monotonic()
            result = macro_step(head, optimizer, data, ids, protocol, args.seed * 1000003 + step)
            torch.cuda.synchronize(); elapsed = time.monotonic() - began
            if result['grad_norm'] <= 0:
                raise AssertionError('No type-head gradient')
            records.append(dict(case=label, repeat=step + 1, **result, maximum_tokens=int(data['lengths'][ids].max()),
                                seconds=elapsed, samples_per_second=len(ids) / elapsed))
    with torch.no_grad():
        head.eval(); hidden, mask = shared.feature_batch(data, [longest_all], protocol)
        output = head(hidden, mask)
        if hidden.requires_grad or output['logits'].requires_grad or int(mask.sum()) != int(data['lengths'][longest_all]):
            raise AssertionError('Frozen full-length inference invariant failed')
        probs, _, _ = decisions(output['logits'].cpu(), data['meta'].binary_probability.to_numpy()[[longest_all]], manifest['parent']['binary_threshold'])
        if not np.allclose(probs.sum(1), 1.):
            raise AssertionError('Type decision must be five-way softmax')
    result = dict(passed=True, **identity, records=records, maximum_train_tokens=int(data['lengths'].max()),
                  harmful_train_rows=len(harmful), safe_train_rows_excluded=int((data['categories'] == 0).sum()),
                  peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, binary_backbone_loaded=False,
                  head_parameters=sum(p.numel() for p in head.parameters()), loss='five_way_cross_entropy',
                  source_category_counts={str(i): int((data['categories'] == i).sum()) for i in range(6)})
    dump(study / 'preflight' / f'{args.arm}_s{args.seed}.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'train', 'evaluate'])
    parser.add_argument('--output', required=True); parser.add_argument('--arm', choices=ARMS, default='last_mlp')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    try:
        if args.steps < 1:
            raise ValueError('Preflight steps must be positive')
        if not torch.cuda.is_available():
            raise RuntimeError('A root-assigned CUDA GPU is required')
        globals()[args.action](args)
    except Exception as exc:
        path = Path(args.output) / 'runs' / f'{args.arm}_s{args.seed}' / 'status.json'
        dump(path, dict(state='failed', action=args.action, arm=args.arm, error=repr(exc), pid=os.getpid(), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
