"""Prediction-only error audit and exclusive-GPU, uncached end-to-end timing.

Only the completed 20261022 clean study is a parent. The CPU audit never loads
model weights. The efficiency worker tokenizes fresh raw validation texts and
calls a real backbone for every timed invocation; no cached feature tensor is
read. Frozen length/identity metadata is used only for untimed sample selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
DEFAULT_OUTPUT = ROOT / 'outputs/hanguard_paper_followups_20261025/analysis'
GUARD = ROOT / 'outputs/qwen3guard_clean_test_20261025_v2'
SCHEMA = 'hanguard_paper_error_efficiency_v1'
SEEDS = (42, 43, 44)
MAIN = ('baseline', 'fusion_only', 'category_only', 'fusion_category')
ARMS = MAIN + ('fusion_repeat_last', 'category_fusion_repeat_last', 'category_shared',
               'fusion_category_shared', 'probe_selected_single')
CONTRASTS = (('fusion_category', 'baseline'), ('category_only', 'baseline'),
             ('fusion_only', 'baseline'), ('fusion_category', 'category_only'),
             ('fusion_category', 'fusion_only'), ('category_only', 'category_shared'),
             ('fusion_category', 'fusion_category_shared'),
             ('fusion_only', 'fusion_repeat_last'),
             ('fusion_category', 'category_fusion_repeat_last'),
             ('fusion_only', 'probe_selected_single'))
CLASS_NAMES = {0: '安全', 1: '违反社会主义核心价值观的内容', 2: '歧视性内容',
               3: '商业违法违规', 4: '侵犯他人合法权益', 5: '无法满足特定服务类型的安全需求'}
BENCHMARK = dict(samples=32, targets=[64, 256, 1024, 'longest'], per_target=8,
                 warmup_passes=3, measured_passes=10, batch_sizes=[1, 8],
                 seed=42, cpu_threads=4, guard_max_new_tokens=128,
                 isolation_poll_seconds=1., backbone='Qwen3.5-4B frozen BF16',
                 raw_block_storage_dtype='float16', sdpa='math',
                 system='one shared backbone forward; both binary and type head always computed')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=_json_default) + '\n')
    temporary.replace(path)


def _stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def _runtime():
    return dict(python=sys.version, packages={name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'triton', 'numpy', 'pandas')})


def metrics(y, p, classes):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    classes = list(classes)
    if y.shape != p.shape or not len(y) or not np.isin(y, classes).all() or not np.isin(p, classes).all():
        raise ValueError('Nonempty aligned decisions within declared classes required')
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(classes)}
    for target, decision in zip(y, p):
        confusion[positions[int(target)], positions[int(decision)]] += 1
    per = []
    for i, label in enumerate(classes):
        tp, support, predicted = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        per.append(dict(category_id=int(label), support=support,
                        precision=tp / max(1, predicted), recall=tp / max(1, support),
                        f1=2 * tp / max(1, support + predicted)))
    result = dict(n=len(y), accuracy=float((y == p).mean()),
                  macro_f1=float(np.mean([x['f1'] for x in per])),
                  classes=classes, per_class=per, confusion_matrix=confusion.tolist())
    if classes == [0, 1]:
        tn, fp, fn, tp = map(int, confusion.ravel())
        result.update(tp=tp, tn=tn, fp=fp, fn=fn,
                      harmful_f1=2 * tp / max(1, 2 * tp + fp + fn),
                      fpr=fp / max(1, tn + fp), fnr=fn / max(1, tp + fn))
    return result


def paired(y, before, after, groups, draws=2000, seed=42):
    y, before, after = map(np.asarray, (y, before, after))
    groups = np.asarray(groups).astype(str)
    if not len(y) or not (y.shape == before.shape == after.shape == groups.shape):
        raise ValueError('Aligned nonempty paired decisions required')
    first, last = before == y, after == y
    rescued, harmed = int((~first & last).sum()), int((first & ~last).sum())
    _, index = np.unique(groups, return_inverse=True)
    size = int(index.max()) + 1
    counts = np.bincount(index, minlength=size)
    changes = np.bincount(index, weights=last.astype(float) - first, minlength=size)
    rng, boot = np.random.default_rng(seed), np.empty(draws)
    for begin in range(0, draws, 16):
        pick = rng.integers(0, size, size=(min(16, draws - begin), size))
        boot[begin:begin + len(pick)] = changes[pick].sum(1) / counts[pick].sum(1)
    return dict(n=len(y), rescued=rescued, harmed=harmed, net_correct=rescued - harmed,
                delta_accuracy=(rescued - harmed) / len(y),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                bootstrap=dict(unit='group_id cluster', groups=size, draws=draws, seed=seed))


def select_benchmark_rows(frame, samples_per_target=8):
    if not {'base_id', 'prompt_tokens'} <= set(frame) or not frame.base_id.is_unique:
        raise ValueError('Unique validation identities and raw token lengths required')
    selected, records = set(), []
    lengths = frame.prompt_tokens.to_numpy(dtype=int)
    if (lengths < 1).any() or len(frame) < 4 * samples_per_target:
        raise ValueError('Not enough valid validation lengths')
    for target in (64, 256, 1024, 'longest'):
        order = (np.argsort(-lengths, kind='stable') if target == 'longest'
                 else np.argsort(np.abs(lengths - target), kind='stable'))
        chosen = [int(index) for index in order if int(index) not in selected][:samples_per_target]
        for index in chosen:
            selected.add(index)
            records.append(dict(base_id=str(frame.iloc[index].base_id), raw_tokens=int(lengths[index]),
                                length_target=target))
    return records


def _parent_paths(parent):
    paths = [parent / name for name in ('protocol.json', 'protocol_registration.json',
                                       'selection_lock.json', 'results.json', 'data/test.parquet',
                                       'data/train.parquet', 'data/validation.parquet')]
    paths += [parent / 'features' / split / 'metadata.parquet' for split in ('train', 'validation', 'test')]
    for task in ('binary', 'category'):
        for seed in SEEDS:
            folder = parent / 'heads' / task / f's{seed}'
            paths += [folder / name for name in ('selection.json', 'test_complete.json',
                                                'test_predictions.parquet', 'test_results.json')]
            if seed == 42:
                paths.append(folder / 'selected_heads.pt')
    return paths


def initialize(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        return verify(output)
    result, protocol = read(parent / 'results.json'), read(parent / 'protocol.json')
    lock = read(parent / 'selection_lock.json')
    parent_sha = sha(parent / 'protocol.json')
    if (not result.get('complete') or not lock.get('complete')
            or result.get('protocol_sha256') != parent_sha
            or read(parent / 'protocol_registration.json')['protocol_sha256'] != parent_sha
            or result.get('rootselection_lock_sha256') != sha(parent / 'selection_lock.json')):
        raise ValueError('A complete locked clean parent is required')
    for split in ('train', 'validation', 'test'):
        if sha(parent / f'data/{split}.parquet') != protocol['data_sha256'][split]:
            raise ValueError('Parent raw dataset changed')
    validation = pd.read_parquet(parent / 'features/validation/metadata.parquet', columns=['base_id', 'prompt_tokens'])
    validation = validation.sort_values('base_id').reset_index(drop=True)
    samples = select_benchmark_rows(validation)
    inputs = {str(path): dict(sha256=sha(path), stat=_stat(path))
              for path in [*_parent_paths(parent), GUARD / 'protocol.json']}
    files = [Path(__file__).resolve(), ROOT / 'scripts/hanguard/clean_ablation_heads.py',
             ROOT / 'scripts/hanguard/multilabel_heads.py', ROOT / 'scripts/hanguard/clean_ablation_runtime.py',
             ROOT / 'scripts/hanguard/frozen_base_readout_study.py', ROOT / 'hanguard_model.py',
             ROOT / 'scripts/hanguard/qwen3guard_clean_queue.py']
    model = Path(protocol['model_path'])
    model_inputs = {name: value for name, value in protocol['inputs'].items()
                    if Path(name).is_relative_to(model)}
    guard_model = ROOT / 'models/Qwen3Guard-Gen-4B'
    guard_protocol = read(GUARD / 'protocol.json')
    guard_inputs = {name: record['sha256'] for name, record in guard_protocol['inputs'].items()
                    if Path(name).is_relative_to(guard_model)}
    registered = dict(schema=SCHEMA, parent=str(parent), created_unix=time.time(), inputs=inputs,
        source_sha256={str(path): sha(path) for path in files},
        model_path=str(model), model_inputs=model_inputs,
        guard_model=str(guard_model), guard_model_inputs=guard_inputs,
        guard_parent_protocol_sha256=sha(GUARD / 'protocol.json'),
        benchmark=BENCHMARK, benchmark_sample=samples, runtime=_runtime(),
        seeds=list(SEEDS), arms=list(ARMS), draws=2000,
        scopes=dict(binary=8350, category=2180, corresponding_system=3704),
        limitations=['Same test split has already been inspected; no new blind holdout claim.',
                     'System timing always computes both heads, using one shared backbone forward.',
                     'Guard is externally pretrained zero-shot; hanguard heads used local supervised training.',
                     'Timing requires an exclusive compute GPU and is invalidated if foreign compute is observed.'])
    write(output / 'protocol.json', registered)
    write(output / 'protocol_registration.json', dict(protocol_sha256=sha(output / 'protocol.json')))
    snapshots = output / 'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for source in files:
        (snapshots / source.name).write_bytes(source.read_bytes())
    return registered


def verify(output):
    output = Path(output)
    protocol = read(output / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('benchmark') != BENCHMARK
            or protocol.get('runtime') != _runtime()
            or read(output / 'protocol_registration.json')['protocol_sha256'] != sha(output / 'protocol.json')):
        raise ValueError('Error/efficiency followup protocol changed')
    for name, digest in protocol['source_sha256'].items():
        if sha(name) != digest:
            raise ValueError(f'Registered source changed: {name}')
    for name, record in protocol['inputs'].items():
        if _stat(name) != record['stat'] or sha(name) != record['sha256']:
            raise ValueError(f'Registered analysis input changed: {name}')
    return protocol


def _aligned_predictions(parent, task, seed, truth):
    folder = parent / 'heads' / task / f's{seed}'
    receipt = read(folder / 'test_complete.json')
    prediction = folder / 'test_predictions.parquet'
    if not receipt.get('complete') or receipt['artifacts'].get(str(prediction)) != sha(prediction):
        raise ValueError('Parent prediction receipt mismatch')
    frame = pd.read_parquet(prediction).sort_values('base_id').reset_index(drop=True)
    if (len(frame) != len(truth) or not frame.base_id.is_unique
            or frame.base_id.tolist() != truth.base_id.tolist()
            or frame.source.tolist() != truth.source.tolist()
            or not np.array_equal(frame.category_id, truth.category_id)
            or not np.array_equal(frame.binary_label, truth.prompt_harm_label.eq('harmful').astype(int))):
        raise ValueError('Parent predictions and raw labels do not align')
    for arm in ARMS:
        width = 2 if task == 'binary' else 5
        probs = frame[[f'{arm}_p{n}' for n in range(width)]].to_numpy()
        if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                or not np.allclose(probs.sum(1), 1, atol=1e-8)):
            raise ValueError('Invalid parent probability columns')
    return frame, read(folder / 'selection.json')


def _aggregate(rows, keys, fields):
    frame = pd.DataFrame(rows)
    results = []
    for group, part in frame.groupby(keys, sort=True, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        item = dict(zip(keys, group))
        item['seeds'] = len(part)
        for field in fields:
            values = part[field].to_numpy(dtype=float)
            item[field + '_mean'] = float(values.mean())
            item[field + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        results.append(item)
    return results


def analyze(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT):
    output = Path(output).resolve()
    protocol = initialize(output, parent)
    parent = Path(protocol['parent'])
    truth = pd.read_parquet(parent / 'data/test.parquet', columns=[
        'base_id', 'group_id', 'source', 'category_id', 'prompt_harm_label', 'type_label_mask', 'prompt_tokens'])
    truth = truth.sort_values('base_id').reset_index(drop=True)
    fresh = pd.read_parquet(parent / 'features/test/metadata.parquet', columns=['base_id', 'prompt_tokens'])
    fresh = fresh.sort_values('base_id').reset_index(drop=True)
    if fresh.base_id.tolist() != truth.base_id.tolist():
        raise ValueError('Frozen fresh token lengths do not align with the test text')
    truth['prompt_tokens'] = fresh.prompt_tokens.to_numpy()
    train = pd.read_parquet(parent / 'features/train/metadata.parquet', columns=['prompt_tokens'])
    edges = np.unique(np.quantile(train.prompt_tokens, [.25, .5, .75])).astype(float)
    length_bins = np.searchsorted(edges, truth.prompt_tokens.to_numpy(), side='left')
    native = truth.source.isin(['chinese_curated', 'jailbench']).to_numpy()
    harmful = truth.prompt_harm_label.eq('harmful').to_numpy()
    typed = native & harmful & truth.type_label_mask.to_numpy(dtype=bool) & truth.category_id.isin(range(1, 6)).to_numpy()
    known = native & (~harmful | typed)
    if (len(truth), int(typed.sum()), int(known.sum())) != (8350, 2180, 3704):
        raise ValueError('Scope denominators changed')
    main = read(parent / 'results.json')
    rows, classes, confusion_rows, strata, changes, change_strata, parameters = [], [], [], [], [], [], []
    for seed in SEEDS:
        binary, bsel = _aligned_predictions(parent, 'binary', seed, truth)
        category, csel = _aligned_predictions(parent, 'category', seed, truth)
        parameters += [dict(task=task, seed=seed, arm=arm, trainable_parameters=selection['arms'][arm]['parameters'])
                       for task, selection in (('binary', bsel), ('category', csel)) for arm in ARMS]
        decisions = {
            'binary': {a: (binary[f'{a}_p1'].to_numpy() >= bsel['arms'][a]['threshold']).astype(int) for a in ARMS},
            'category': {a: category[f'{a}_argmax'].to_numpy(dtype=int) + 1 for a in ARMS}}
        decisions['corresponding_system'] = {
            a: np.where(decisions['binary'][a] == 1, decisions['category'][a], 0) for a in ARMS}
        for task, mask, y, labels in (
                ('binary', np.ones(len(truth), bool), harmful.astype(int), [0, 1]),
                ('category', typed, truth.category_id.to_numpy(), list(range(1, 6))),
                ('corresponding_system', known, np.where(harmful, truth.category_id, 0), list(range(6)))):
            scope = dict(task=task, seed=seed)
            for arm in ARMS:
                predicted = decisions[task][arm]
                value = metrics(y[mask], predicted[mask], labels)
                item = dict(scope, arm=arm, **{k: value[k] for k in ('n', 'accuracy', 'macro_f1')})
                if task == 'binary':
                    reference = main['per_seed'][str(seed)]['binary']['arms'][arm]['full_original_test']['validation_threshold']
                    if not np.isclose(value['accuracy'], reference['accuracy'], atol=1e-12):
                        raise ValueError('Recomputed binary accuracy differs from parent')
                    if not np.isclose(value['harmful_f1'], reference['f1'], atol=1e-12):
                        raise ValueError('Recomputed binary F1 differs from parent')
                    item.update({k: value[k] for k in ('harmful_f1', 'fpr', 'fnr')})
                elif task == 'category':
                    reference = main['per_seed'][str(seed)]['category']['arms'][arm]['native_known_harmful']
                    if not all(np.isclose(value[k], reference[k], atol=1e-12) for k in ('accuracy', 'macro_f1')):
                        raise ValueError('Recomputed category metrics differ from parent')
                elif arm in MAIN:
                    reference = main['per_seed'][str(seed)]['systems']['corresponding_both_stages']['arms'][arm]['metrics']
                    if not all(np.isclose(value[k], reference[k], atol=1e-12) for k in ('accuracy', 'macro_f1')):
                        raise ValueError('Recomputed system metrics differ from parent')
                rows.append(item)
                classes += [dict(scope, arm=arm,
                    category_name=('无害' if entry['category_id'] == 0 else '有害') if task == 'binary'
                                  else CLASS_NAMES[entry['category_id']], **entry)
                            for entry in value['per_class']]
                confusion_rows += [dict(scope, arm=arm, truth=target, predicted=decision,
                                        count=value['confusion_matrix'][i][j])
                                   for i, target in enumerate(labels) for j, decision in enumerate(labels)]
                for stratifier, values in (('source', truth.source.to_numpy()), ('train_length_quartile', length_bins)):
                    for name in sorted(set(values[mask])):
                        sub = mask & (values == name)
                        v = metrics(y[sub], predicted[sub], labels)
                        strata.append(dict(scope, arm=arm, stratifier=stratifier, stratum=str(name),
                                           n=v['n'], accuracy=v['accuracy'], macro_f1=v['macro_f1']))
            for candidate, baseline in CONTRASTS:
                before, after = decisions[task][baseline], decisions[task][candidate]
                value = paired(y[mask], before[mask], after[mask], truth.group_id.to_numpy()[mask],
                               draws=protocol['draws'], seed=seed)
                changes.append(dict(scope, candidate=candidate, baseline=baseline, **value))
                for stratifier, values in (('source', truth.source.to_numpy()),
                                           ('train_length_quartile', length_bins), ('true_class', y)):
                    for name in sorted(set(values[mask])):
                        sub = mask & (values == name)
                        rescued = int(((before != y) & (after == y) & sub).sum())
                        harmed_count = int(((before == y) & (after != y) & sub).sum())
                        change_strata.append(dict(scope, candidate=candidate, baseline=baseline,
                            stratifier=stratifier, stratum=str(name), n=int(sub.sum()),
                            rescued=rescued, harmed=harmed_count, net_correct=rescued - harmed_count,
                            delta_accuracy=(rescued - harmed_count) / int(sub.sum())))
    summary = _aggregate(rows, ['task', 'arm'], ['accuracy', 'macro_f1'])
    class_summary = _aggregate(classes, ['task', 'arm', 'category_id'], ['precision', 'recall', 'f1'])
    paired_summary = _aggregate(changes, ['task', 'candidate', 'baseline'], ['delta_accuracy', 'rescued', 'harmed'])
    strata_summary = _aggregate(strata, ['task', 'arm', 'stratifier', 'stratum'], ['accuracy', 'macro_f1'])
    change_strata_summary = _aggregate(change_strata,
        ['task', 'candidate', 'baseline', 'stratifier', 'stratum'],
        ['delta_accuracy', 'rescued', 'harmed', 'net_correct'])
    for name, values in (('metrics_by_seed.csv', rows), ('class_metrics_by_seed.csv', classes),
                         ('confusion_by_seed.csv', confusion_rows), ('strata_by_seed.csv', strata),
                         ('paired_changes_by_seed.csv', changes), ('paired_strata_by_seed.csv', change_strata),
                         ('seed_summary.csv', summary), ('class_summary.csv', class_summary),
                         ('paired_summary.csv', paired_summary), ('parameters.csv', parameters),
                         ('strata_summary.csv', strata_summary), ('paired_strata_summary.csv', change_strata_summary)):
        pd.DataFrame(values).to_csv(output / name, index=False)
    result = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(output / 'protocol.json'),
        scopes=protocol['scopes'], length_quartile_edges=edges.tolist(),
        aggregate=summary, class_aggregate=class_summary, paired_aggregate=paired_summary,
        paired_strata_aggregate=change_strata_summary,
        per_seed=rows, paired_per_seed=changes, parent_metrics_recomputed_and_matched=True,
        system_controls='All nine corresponding-stage systems are derived here; parent main four are exactly verified.',
        completed_unix=time.time())
    write(output / 'analysis_results.json', result)
    _analysis_readme(output, result)
    if (GUARD / 'results.json').exists() and read(GUARD / 'results.json').get('complete'):
        append_guard_reference(output)
    return result


def _analysis_readme(output, result):
    lines = ['# 本轮分类别错误分析与效率评测', '',
        'CPU 错误分析已完成；端到端效率状态见 efficiency_complete.json，未测时不填性能数字。', '',
        '二分类 8,350 条；原生已知有害五分类 2,180 条；对应双阶段系统 3,704 条。',
        '二分类使用各模型验证集选出的阈值；五分类独立于二分类门控。三 seed 全部保留。',
        '从逐样本预测重算并核对父结果；长度区间仅由训练集四分位数定义。', '',
        '| 任务 | 方法 | Accuracy / %（均值±SD） | Macro-F1 / %（均值±SD） |',
        '|---|---|---:|---:|']
    for row in result['aggregate']:
        if row['arm'] in MAIN:
            lines.append(f"| {row['task']} | {row['arm']} | {row['accuracy_mean']*100:.2f}±{row['accuracy_sd']*100:.2f} | {row['macro_f1_mean']*100:.2f}±{row['macro_f1_sd']*100:.2f} |")
    lines += ['', '## 五分类融合相对类别读出的增益分布', '',
              '| 真实类别 | 召回变化 / pp（均值） | 救回 / 条（均值） | 新增错误 / 条（均值） |',
              '|---|---:|---:|---:|']
    for row in result['paired_strata_aggregate']:
        if (row['task'] == 'category' and row['candidate'] == 'fusion_category'
                and row['baseline'] == 'category_only' and row['stratifier'] == 'true_class'):
            lines.append(f"| {CLASS_NAMES[int(row['stratum'])]} | {row['delta_accuracy_mean']*100:+.2f} | {row['rescued_mean']:.2f} | {row['harmed_mean']:.2f} |")
    lines += ['', '## 机制结论的边界', '',
        '- 类别共享查询仍保留类别向量参与最终打分；不能用它证明所有类别向量无效。',
        '- 五分类专属注意力较共享注意力的平均提升应报告，同时保留不确定性。',
        '- 组合相对末 token MLP 的提升不能全部归于融合；应看组合减类别读出以及同容量重复末层。',
        '- 九臂对应系统对照由相同预测派生；它不等于仅在二分类 gate 加融合的经典组合。',
        '- 共享卡运行速度仅用于进度估算；本模块正式效率数字必须来自空卡并通过进程隔离监测。', '',
        '逐类别/混淆矩阵、来源/长度、配对救回和新增错误、2,000 次 group bootstrap 均见同目录 CSV/JSON。',
        '不输出原始文本，不依据错误结果删除样本。', '']
    (output / 'README.md').write_text('\n'.join(lines))


def append_guard_reference(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    protocol = verify(output)
    result = read(GUARD / 'results.json')
    raw = Path(protocol['parent']) / 'data/test.parquet'
    if (not result.get('complete') or result.get('invalid_safety_rows') != 0
            or result.get('rows') != 8350 or result.get('data_sha256') != sha(raw)
            or result.get('protocol_sha256') != sha(GUARD / 'protocol.json')
            or result.get('predictions_sha256') != sha(GUARD / 'predictions.parquet')):
        raise ValueError('Guard reference is incomplete or not the same registered test data')
    artifact = dict(schema=SCHEMA, complete=True, parent_analysis_protocol_sha256=sha(output / 'protocol.json'),
        guard_protocol_sha256=sha(GUARD / 'protocol.json'), guard_results_sha256=sha(GUARD / 'results.json'),
        guard_predictions_sha256=result['predictions_sha256'], data_sha256=result['data_sha256'],
        rows=8350, mappings=result['mappings'], by_source=result['by_source'],
        five_class='N/A: incompatible native label taxonomy',
        comparison_note='Guard zero-shot versus locally supervised readouts; not a capacity-matched internal ablation')
    write(output / 'guard_reference.json', artifact)
    return artifact


class BenchmarkContended(RuntimeError):
    pass


def _exclusive_gpu(gpu):
    from scripts.hanguard.qwen3guard_clean_queue import gpu_inventory
    card = next((row for row in gpu_inventory() if row['index'] == gpu), None)
    if card is None or os.environ.get('CUDA_VISIBLE_DEVICES') != card['uuid']:
        raise BenchmarkContended('Unverified benchmark physical GPU / visible UUID')
    foreign = sorted(set(card['pids']) - {os.getpid()})
    if foreign:
        raise BenchmarkContended(f'Foreign compute/graphics present on benchmark GPU: {foreign}')
    return card


def _admitted(output, gpu):
    from scripts.hanguard.qwen3guard_clean_queue import identity
    expected = sha(Path(output) / 'protocol.json')
    if os.environ.get('HANGUARD_FOLLOWUP_ADMITTED') != expected:
        raise RuntimeError('Efficiency requires a registered root-scheduler admission')
    path = os.environ.get('HANGUARD_FOLLOWUP_ADMISSION_RECORD')
    if not path:
        raise RuntimeError('Missing efficiency admission record')
    record = read(path)
    owner = identity(os.getpid())
    if (not owner or any(record.get(key) != value for key, value in owner.items())
            or record.get('followup_protocol_sha256') != expected
            or record.get('physical_gpu') != gpu or record.get('exclusive') is not True
            or record.get('gpu_uuid') != os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise RuntimeError('Efficiency admission does not bind this exact worker')
    return _exclusive_gpu(gpu)


class IsolationMonitor:
    def __init__(self, gpu, interval=1.):
        self.gpu, self.interval = gpu, interval
        self.stop = threading.Event()
        self.invalid = threading.Event()
        self.observations = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                card = _exclusive_gpu(self.gpu)
                self.observations.append(dict(time=time.time(), valid=True, free_gib=card['free_gib']))
            except Exception as error:
                self.observations.append(dict(time=time.time(), valid=False, error=repr(error)))
                self.invalid.set()
                return
            self.stop.wait(self.interval)

    def __enter__(self):
        _exclusive_gpu(self.gpu)
        self.thread.start()
        return self

    def check(self):
        if self.invalid.is_set():
            raise BenchmarkContended('Foreign/unknown GPU activity observed; timings invalid')

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=20)


class OnlineCapture:
    """Minimal raw-block capture kept on device, never read from a feature cache."""
    def __init__(self, backbone, layers, full_tokens, torch, sdpa_context):
        self.backbone, self.layers, self.full_tokens = backbone, tuple(layers), full_tokens
        self.torch, self.sdpa_context = torch, sdpa_context
        self.mask, self.values, self.tokens = None, {}, None
        self.handles = [backbone.layers[layer - 1].register_forward_hook(self._hook(layer)) for layer in layers]
        self.forward_calls = 0

    def _hook(self, layer):
        def hook(module, inputs, output):
            if self.mask is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            rows = self.torch.arange(len(hidden), device=hidden.device)
            self.values[layer] = hidden[rows, self.mask.sum(1) - 1].to(self.torch.float16)
            if layer == len(self.backbone.layers) and self.full_tokens:
                self.tokens = hidden.to(self.torch.float16).masked_fill(~self.mask[..., None], 0)
        return hook

    def __call__(self, batch):
        torch = self.torch
        self.mask = batch['attention_mask'].bool()
        self.values, self.tokens = {}, None
        if self.mask.ndim != 2 or not self.mask.any(1).all():
            raise ValueError('Valid masks required')
        expected = torch.arange(self.mask.shape[1], device=self.mask.device)[None] < self.mask.sum(1)[:, None]
        if not torch.equal(self.mask, expected):
            raise ValueError('Online capture requires right padding')
        try:
            with torch.inference_mode(), self.sdpa_context(), torch.autocast(
                    device_type=batch['input_ids'].device.type, dtype=torch.bfloat16,
                    enabled=batch['input_ids'].device.type == 'cuda'):
                self.backbone(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    position_ids=(batch['attention_mask'].cumsum(-1) - 1).clamp_min(0), use_cache=False)
            self.forward_calls += 1
            if set(self.values) != set(self.layers):
                raise RuntimeError('Online backbone omitted a requested raw block')
            sample = self.values[self.layers[-1]]
            last = torch.zeros((len(sample), len(self.backbone.layers), sample.shape[-1]),
                               device=sample.device, dtype=sample.dtype)
            for layer, values in self.values.items():
                last[:, layer - 1] = values
            if self.full_tokens:
                if self.tokens is None:
                    raise RuntimeError('Online capture omitted full tokens')
                rows = torch.arange(len(last), device=last.device)
                if not torch.equal(last[:, -1], self.tokens[rows, self.mask.sum(1) - 1]):
                    raise RuntimeError('Online same-forward token identities differ')
            return last, self.tokens, self.mask
        finally:
            self.mask, self.values, self.tokens = None, {}, None

    def close(self):
        for handle in self.handles:
            handle.remove()


class OnlineReadout:
    def __init__(self, tokenizer, capture, heads, arm, task, threshold, torch):
        self.tokenizer, self.capture, self.heads, self.arm, self.task = tokenizer, capture, heads, arm, task
        self.threshold, self.torch = threshold, torch

    def __call__(self, texts):
        torch = self.torch
        start = time.perf_counter()
        batch = self.tokenizer(texts, padding=True, truncation=False,
                               add_special_tokens=False, return_tensors='pt')
        token_seconds = time.perf_counter() - start
        batch = batch.to('cuda')
        last, tokens, mask = self.capture(batch)
        with torch.inference_mode():
            if self.task in ('binary', 'system'):
                logits = self.heads['binary'][self.arm](last, tokens, mask)
                binary = logits.float().softmax(-1)[:, 1] >= self.threshold
            if self.task in ('category', 'system'):
                logits = self.heads['category'][self.arm](last, tokens, mask)
                category = logits.argmax(-1) + 1
            result = (binary if self.task == 'binary' else category if self.task == 'category'
                      else torch.where(binary, category, 0))
            result = result.cpu().tolist()
        return dict(tokenization_seconds=token_seconds, rows=len(result))


def measure_callable(function, batches, torch, monitor, warmup=3, repeats=10):
    for _ in range(warmup):
        for batch in batches:
            monitor.check()
            function(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed, tokenization, counts = [], [], []
    for _ in range(repeats):
        for batch in batches:
            monitor.check()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function(batch)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            tokenization.append(result['tokenization_seconds'])
            counts.append(result['rows'])
    monitor.check()
    return dict(invocations=len(elapsed), examples=sum(counts), elapsed_seconds=float(sum(elapsed)),
        latency_ms_p50=float(np.quantile(elapsed, .5) * 1000),
        latency_ms_p95=float(np.quantile(elapsed, .95) * 1000),
        mean_batch_ms=float(np.mean(elapsed) * 1000),
        examples_per_second=sum(counts) / sum(elapsed),
        tokenization_seconds=float(sum(tokenization)),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        warmup_passes=warmup, measured_passes=repeats,
        timing_scope='fresh tokenization + transfer + real backbone + head/generation + CPU output')


def _load_heads(parent, hidden, torch):
    from scripts.hanguard.clean_ablation_heads import CleanAblationHead
    heads, selections = {}, {}
    for task in ('binary', 'category'):
        folder = parent / 'heads' / task / 's42'
        selections[task] = read(folder / 'selection.json')
        selected = torch.load(folder / 'selected_heads.pt', map_location='cpu', weights_only=True)
        heads[task] = {}
        for arm in MAIN:
            model = CleanAblationHead(hidden, width=128, num_classes=2 if task == 'binary' else 5,
                                      arm=arm, dropout=.1, seed=42)
            model.load_state_dict(selected['states'][arm], strict=True)
            model.to('cuda').eval().requires_grad_(False)
            if sum(p.numel() for p in model.parameters()) != selections[task]['arms'][arm]['parameters']:
                raise ValueError('Online head parameter count differs from selected parent')
            heads[task][arm] = model
    return heads, selections


def benchmark(output=DEFAULT_OUTPUT, parent=None, gpu=0):
    output = Path(output).resolve()
    protocol = verify(output)
    if parent is not None and Path(parent).resolve() != Path(protocol['parent']):
        raise ValueError('Efficiency parent differs from registration')
    if (output / 'efficiency_complete.json').exists():
        receipt = read(output / 'efficiency_complete.json')
        if (receipt.get('protocol_sha256') != sha(output / 'protocol.json')
                or receipt.get('results_sha256') != sha(output / 'efficiency_results.json')
                or receipt.get('csv_sha256') != sha(output / 'efficiency.csv')):
            raise ValueError('Completed efficiency results changed')
        return 0
    try:
        entry = _admitted(output, gpu)
    except BenchmarkContended:
        return 75
    parent = Path(protocol['parent'])
    attempt = output / 'efficiency_attempts' / f'{os.getpid()}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    # Verify actual registered weights before importing CUDA-loading helpers.
    for name, digest in {**protocol['model_inputs'], **protocol['guard_model_inputs']}.items():
        if sha(name) != digest:
            raise ValueError('Registered efficiency model bytes changed')
    try:
        _exclusive_gpu(gpu)
    except BenchmarkContended:
        write(attempt / 'status.json', dict(state='invalid_contended_before_load', completed=False))
        return 75
    import torch
    from scripts.hanguard import clean_ablation_runtime as runtime
    from scripts.hanguard.frozen_base_readout_study import load_frozen_backbone
    torch.set_num_threads(BENCHMARK['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for env, name in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                       ('CUDA_CACHE_PATH', 'cuda')):
        os.environ[env] = str(attempt / name)
    validation = pd.read_parquet(parent / 'data/validation.parquet', columns=['base_id', 'prompt'])
    by_id = dict(zip(validation.base_id, validation.prompt))
    texts = [str(by_id[row['base_id']]) for row in protocol['benchmark_sample']]
    results, capture, model, heads, kernels = [], None, None, None, None
    monitor = IsolationMonitor(gpu, BENCHMARK['isolation_poll_seconds'])
    try:
        with monitor:
            parent_protocol = read(parent / 'protocol.json')
            model, backbone, tokenizer = load_frozen_backbone(parent_protocol)
            kernels = runtime.DeterministicKernelPath(require_fla=parent_protocol['model_type'] == 'qwen3_5')
            heads, selections = _load_heads(parent, parent_protocol['hidden_size'], torch)
            for task in ('binary', 'category', 'system'):
                for arm in MAIN:
                    capture = OnlineCapture(backbone, (8, 16, 24, 32) if arm in ('fusion_only', 'fusion_category') else (32,),
                                            arm in ('category_only', 'fusion_category'), torch, runtime.math_sdpa)
                    function = OnlineReadout(tokenizer, capture, heads, arm, task,
                                             selections['binary']['arms'][arm]['threshold'], torch)
                    for batch_size in BENCHMARK['batch_sizes']:
                        batches = [texts[start:start + batch_size] for start in range(0, len(texts), batch_size)]
                        measured = measure_callable(function, batches, torch, monitor,
                                                     BENCHMARK['warmup_passes'], BENCHMARK['measured_passes'])
                        parameters = (sum(selections[t]['arms'][arm]['parameters'] for t in ('binary', 'category'))
                                      if task == 'system' else selections[task]['arms'][arm]['parameters'])
                        results.append(dict(model='hanguard_Qwen3.5-4B', task=task, arm=arm,
                            batch_size=batch_size, head_parameters=parameters,
                            backbone_parameters=sum(p.numel() for p in model.parameters()),
                            backbone_forward_calls=capture.forward_calls, **measured))
                        write(attempt / 'progress.json', dict(completed_cases=len(results),
                              total_cases=26, last_case=results[-1], updated_unix=time.time()))
                    capture.close()
                    capture = None
            write(attempt / 'kernel_path.json', kernels.snapshot())
            del function, heads, backbone, model
            heads, model = None, None
            gc.collect()
            torch.cuda.empty_cache()
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(protocol['guard_model'], local_files_only=True,
                dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval().requires_grad_(False)
            tokenizer = AutoTokenizer.from_pretrained(protocol['guard_model'], local_files_only=True)
            tokenizer.padding_side = 'left'
            tokenizer.pad_token_id = model.generation_config.pad_token_id
            def guard_forward(batch_texts):
                started = time.perf_counter()
                prompts = [tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=False)
                           for text in batch_texts]
                batch = tokenizer(prompts, add_special_tokens=False, truncation=False,
                                  padding=True, return_tensors='pt')
                token_seconds = time.perf_counter() - started
                if batch.input_ids.shape[1] + BENCHMARK['guard_max_new_tokens'] > model.config.max_position_embeddings:
                    raise ValueError('Complete Guard benchmark input exceeds context')
                batch = batch.to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**batch, do_sample=False,
                        max_new_tokens=BENCHMARK['guard_max_new_tokens'], pad_token_id=tokenizer.pad_token_id)
                    outputs = tokenizer.batch_decode(generated[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
                return dict(rows=len(outputs), tokenization_seconds=token_seconds)
            for batch_size in BENCHMARK['batch_sizes']:
                batches = [texts[start:start + batch_size] for start in range(0, len(texts), batch_size)]
                measured = measure_callable(guard_forward, batches, torch, monitor,
                                             BENCHMARK['warmup_passes'], BENCHMARK['measured_passes'])
                results.append(dict(model='Qwen3Guard-Gen-4B', task='binary_native_generation', arm='official_zero_shot',
                    batch_size=batch_size, head_parameters=0,
                    backbone_parameters=sum(p.numel() for p in model.parameters()), **measured))
            _exclusive_gpu(gpu)
            monitor.check()
        artifact = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(output / 'protocol.json'),
                        attempt=str(attempt), gpu=entry, cases=results,
                        gpu_name=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda,
                        isolation_observations=monitor.observations,
                        sample=protocol['benchmark_sample'], settings=BENCHMARK,
                        completed_unix=time.time())
        write(output / 'efficiency_results.json', artifact)
        pd.DataFrame(results).to_csv(output / 'efficiency.csv', index=False)
        write(output / 'efficiency_complete.json', dict(complete=True,
            protocol_sha256=sha(output / 'protocol.json'), results_sha256=sha(output / 'efficiency_results.json'),
            csv_sha256=sha(output / 'efficiency.csv')))
        write(attempt / 'status.json', dict(state='complete', complete=True))
        return 0
    except BenchmarkContended as error:
        write(attempt / 'status.json', dict(state='invalid_contended', complete=False, error=repr(error),
            observations=monitor.observations, completed_cases=len(results)))
        return 75
    finally:
        if capture is not None:
            capture.close()
        heads, model = None, None
        gc.collect()
        torch.cuda.empty_cache()


def progress(output=DEFAULT_OUTPUT):
    output = Path(output)
    cpu = output / 'analysis_results.json'
    efficiency = output / 'efficiency_complete.json'
    return dict(cpu_analysis_complete=cpu.exists() and read(cpu).get('complete') is True,
                benchmark_complete=efficiency.exists() and read(efficiency).get('complete') is True,
                benchmark_pending=not efficiency.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('analyze', 'benchmark', 'status', 'guard-reference'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'analyze':
        result = analyze(args.output, args.parent)
        print(json.dumps(dict(complete=result['complete'], scopes=result['scopes']), ensure_ascii=False))
    elif args.action == 'guard-reference':
        append_guard_reference(args.output)
        print(json.dumps(dict(complete=True, rows=8350)))
    elif args.action == 'benchmark':
        if args.gpu is None:
            parser.error('benchmark needs --gpu')
        return benchmark(args.output, args.parent, args.gpu)
    else:
        print(json.dumps(progress(args.output), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
