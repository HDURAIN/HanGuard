"""Audited reports for original or explicitly registered intent-primary labels.

The two label protocols remain separate. Newly generated multilabel annotations
are never loaded by this file; intent relabels require immutable provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import primary_category_contract as contract


ARMS = ('last_mlp', 'learned_queries', 'description_queries')
TYPE_CLASSES = (1, 2, 3, 4, 5)
ALL_CLASSES = (0, 1, 2, 3, 4, 5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _integers(values, permitted, name):
    try:
        values = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid {name}') from error
    if not np.isin(values, permitted).all():
        raise ValueError(f'{name} must be in {list(permitted)}')
    return values.astype(np.int64)


def classification_metrics(truth, prediction, classes):
    """Fixed-class macro F1; absent classes score zero and carry support flags."""
    classes = tuple(classes)
    truth = _integers(truth, classes, 'truth')
    prediction = _integers(prediction, classes, 'predictions')
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError('Aligned one-dimensional truth and predictions required')
    lookup = {value: index for index, value in enumerate(classes)}
    true_index = np.asarray([lookup[value] for value in truth], dtype=np.int64)
    pred_index = np.asarray([lookup[value] for value in prediction], dtype=np.int64)
    confusion = np.bincount(true_index * len(classes) + pred_index,
                            minlength=len(classes) ** 2).reshape(len(classes), len(classes))
    per_class = []
    for index, category in enumerate(classes):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        per_class.append(dict(category_id=category, support=support, predicted=predicted,
                              tp=tp, fp=predicted-tp, fn=support-tp,
                              precision=tp/max(1, predicted), recall=tp/max(1, support),
                              f1=2*tp/max(1, support+predicted),
                              support_flags=['no_true_examples'] if support == 0 else []))
    return dict(rows=len(truth), classes=list(classes),
                accuracy=float((truth == prediction).mean()) if len(truth) else None,
                macro_f1=float(np.mean([row['f1'] for row in per_class])),
                per_class=per_class, confusion=confusion.tolist(),
                confusion_convention='rows=true labels, columns=predicted labels',
                macro_rule='mean over all registered classes; undefined class F1 = 0')


def _batch_scores(truth, prediction, classes):
    """Vectorized accuracy/macro F1 for bootstrap arrays [replicates,rows]."""
    classes = tuple(classes)
    if classes != tuple(range(classes[0], classes[-1] + 1)):
        raise ValueError('Bootstrap classes must be consecutive')
    count = len(classes)
    codes = (truth - classes[0]) * count + prediction - classes[0]
    codes = codes + np.arange(len(truth))[:, None] * count * count
    confusion = np.bincount(codes.ravel(), minlength=len(truth)*count*count).reshape(len(truth), count, count)
    tp = np.diagonal(confusion, axis1=1, axis2=2)
    f1 = 2 * tp / np.maximum(1, confusion.sum(1) + confusion.sum(2))
    return (truth == prediction).mean(1), f1.mean(1)


def paired_bootstrap(truth, candidate, baseline, classes, repetitions=1000, seed=42):
    truth = _integers(truth, classes, 'bootstrap truth')
    candidate = _integers(candidate, classes, 'candidate predictions')
    baseline = _integers(baseline, classes, 'baseline predictions')
    if truth.ndim != 1 or candidate.shape != truth.shape or baseline.shape != truth.shape or len(truth) == 0:
        raise ValueError('A nonempty set of paired sample predictions is required')
    if repetitions < 1:
        raise ValueError('Positive bootstrap repetition count required')
    observed_a = classification_metrics(truth, candidate, classes)
    observed_b = classification_metrics(truth, baseline, classes)
    rng = np.random.default_rng(seed)
    differences = [[], []]
    for start in range(0, repetitions, 64):
        indices = rng.integers(0, len(truth), size=(min(64, repetitions-start), len(truth)))
        a = _batch_scores(truth[indices], candidate[indices], classes)
        b = _batch_scores(truth[indices], baseline[indices], classes)
        for metric in range(2):
            differences[metric].extend((a[metric] - b[metric]).tolist())
    result = dict(rows=len(truth), repetitions=repetitions, bootstrap_seed=seed,
                  classes=list(classes), resampling='identical sample indices for both methods',
                  uncertainty_scope='sample resampling only, excludes label noise and training-seed variation')
    for index, metric in enumerate(('accuracy', 'macro_f1')):
        interval = np.percentile(differences[index], [2.5, 97.5]).tolist()
        result[metric] = dict(delta=observed_a[metric]-observed_b[metric], ci95=interval,
                              excludes_zero=bool(interval[0] > 0 or interval[1] < 0))
    candidate_only = int(((candidate == truth) & (baseline != truth)).sum())
    baseline_only = int(((baseline == truth) & (candidate != truth)).sum())
    discordant = candidate_only + baseline_only
    tail = sum(math.comb(discordant, index) for index in range(min(candidate_only, baseline_only)+1))
    probability = min(1., 2 * tail / (2 ** discordant)) if discordant else 1.
    result['mcnemar_exact'] = dict(candidate_only_correct=candidate_only,
                                   baseline_only_correct=baseline_only, discordant=discordant,
                                   two_sided_p=probability,
                                   caveat='Exploratory paired accuracy test; no multiple-comparison correction')
    return result


def _source_frame(data_dir, split, protocol):
    path = data_dir / f'{split}.parquet'
    frozen = protocol.get('data_sha256', protocol.get('input_data_sha256', {}))
    expected = frozen.get(split, frozen.get(f'{split}.parquet'))
    if expected is not None and sha(path) != expected:
        raise ValueError(f'Protocol dataset hash changed: {split}')
    sampling_path = data_dir.parent / 'sampling_manifest.json'
    if sampling_path.exists() and not contract.is_intent(protocol):
        sampling = json.loads(sampling_path.read_text())
        if sha(path) != sampling['input_sha256'][f'{split}.parquet']:
            raise ValueError(f'Frozen sampled input changed: {split}')
    frame = pd.read_parquet(path).sort_values('base_id').reset_index(drop=True)
    required = {'base_id', 'source', 'prompt', 'category_id'}
    if not required <= set(frame) or frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError('Original input needs unique identities and original category/text columns')
    if contract.is_intent(protocol):
        contract.validate_frame(frame, protocol, original=contract.original_frame(protocol, split))
        frame['category_id'] = _integers(frame.category_id, (-1,) + ALL_CLASSES, 'intent category_id')
    else:
        if 'label_origin' in frame and frame.label_origin.eq('wildguard_intent_reannotation').any():
            raise ValueError('Intent reannotations require the separate registered intent task')
        frame['category_id'] = _integers(frame.category_id, ALL_CLASSES, 'original category_id')
    actual = frame.prompt.map(text_sha)
    if 'prompt_sha256' in frame and not actual.equals(frame.prompt_sha256):
        raise ValueError('Original input prompt hash mismatch')
    frame['prompt_sha256'] = actual
    return frame, sha(path)


def _predictions(path, original, binary_threshold, protocol=None):
    frame = pd.read_csv(path).sort_values('base_id').reset_index(drop=True)
    required = {'base_id', 'source', 'prompt_sha256', 'category_id', 'binary_probability',
                'predicted_category', 'gated_category'} | {f'p_{label}' for label in TYPE_CLASSES}
    if not required <= set(frame) or frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError(f'Invalid prediction schema or identities: {path}')
    if frame.base_id.tolist() != original.base_id.tolist():
        raise ValueError('Predictions do not exactly cover original test identities')
    if not frame.source.equals(original.source) or not frame.prompt_sha256.equals(original.prompt_sha256):
        raise ValueError('Prediction source or text identity mismatch')
    intent = protocol is not None and contract.is_intent(protocol)
    truth = _integers(frame.category_id, (-1,) + ALL_CLASSES if intent else ALL_CLASSES, 'CSV category_id')
    if not np.array_equal(truth, original.category_id.to_numpy()):
        raise ValueError('CSV category labels differ from original single-primary labels')
    if intent:
        if not set(contract.PROVENANCE_COLUMNS) <= set(frame):
            raise ValueError('Predictions must preserve intent label provenance and masks')
        for column in contract.PROVENANCE_COLUMNS:
            if not np.array_equal(frame[column].fillna('').astype(str), original[column].fillna('').astype(str)):
                raise ValueError(f'Prediction annotation provenance changed: {column}')
    predicted = _integers(frame.predicted_category, TYPE_CLASSES, 'predicted_category')
    gated = _integers(frame.gated_category, ALL_CLASSES, 'gated_category')
    probability = frame[[f'p_{label}' for label in TYPE_CLASSES]].to_numpy(dtype=np.float64)
    binary = frame.binary_probability.to_numpy(dtype=np.float64)
    if (not np.isfinite(probability).all() or not np.isfinite(binary).all() or
            ((probability < 0) | (probability > 1)).any() or ((binary < 0) | (binary > 1)).any()):
        raise ValueError('Nonfinite or out-of-range probabilities')
    if not np.allclose(probability.sum(1), 1., atol=1e-5, rtol=0):
        raise ValueError('Primary-category probabilities must be a five-way softmax')
    if not np.array_equal(predicted, probability.argmax(1)+1):
        raise ValueError('Type prediction differs from five-way argmax')
    expected_gate = np.where(binary >= binary_threshold, predicted, 0)
    if not np.array_equal(gated, expected_gate):
        raise ValueError('End-to-end predictions did not use the frozen parent binary gate')
    return frame, truth, predicted, gated


def _verify_metrics(reported, actual, name):
    for key in ('accuracy', 'macro_f1'):
        if reported[key] is None and actual[key] is None:
            continue
        if reported[key] is None or actual[key] is None or abs(reported[key]-actual[key]) > 1e-6:
            raise ValueError(f'Reported {name}.{key} disagrees with CSV predictions')
    for key in ('confusion', 'confusion_matrix'):
        if key in reported and not np.array_equal(reported[key], actual['confusion']):
            raise ValueError(f'Reported {name} confusion matrix disagrees with CSV')


def seed_statistics(values):
    """Sample standard deviation describes seed variation, not a confidence CI."""
    values = [float(value) for value in values if value is not None]
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None,
                std_ddof=1)


def aggregate_seed_results(entries):
    summaries = []
    for arm in ARMS:
        rows = [entry for entry in entries if entry['arm'] == arm]
        if not rows:
            continue
        def metrics(items):
            return {scope: {metric: seed_statistics([row[scope][metric] for row in items])
                    for metric in ('accuracy', 'macro_f1')} for scope in ('type_metrics', 'gated_metrics')}
        sources = sorted(set.intersection(*(set(row['by_source']) for row in rows)))
        summaries.append(dict(arm=arm, seeds=[row['seed'] for row in rows], **metrics(rows),
            head_parameters=sorted(set(row['head_parameters'] for row in rows)),
            best_epoch=seed_statistics([row['best_epoch'] for row in rows]),
            by_source={source: metrics([row['by_source'][source] for row in rows]) for source in sources}))
    return summaries


def report(study, repetitions=1000, seed=42):
    study = Path(study)
    protocol_path = study / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    intent = contract.is_intent(protocol)
    if intent:
        contract.validate_protocol(protocol)
    protocol_hash = sha(protocol_path)
    barrier = json.loads((study / 'test_barrier.json').read_text())
    if barrier.get('locked') is not True or barrier.get('protocol_sha256') != protocol_hash:
        raise ValueError('A matching explicitly locked test barrier is required')
    arms, seeds = protocol.get('all_arms', list(ARMS)), protocol.get('seeds', [42])
    if set(arms) != set(ARMS):
        raise ValueError('Expected all three registered primary-category heads')
    data_dir = Path(protocol['data_dir'])
    original, dataset_hash = _source_frame(data_dir, 'test', protocol)
    train, train_hash = _source_frame(data_dir, 'train', protocol)
    validation, validation_hash = _source_frame(data_dir, 'validation', protocol)
    two_source = protocol.get('experiment_scope') == 'two_source_full_primary_category'
    if two_source and any(not frame.source.isin(['chinese_curated', 'jailbench']).all() for frame in (train, validation, original)):
        raise ValueError('Two-source report cannot contain WildGuard or another source')
    entries, arrays, artifact_hashes = [], {}, {}
    thresholds = []
    common_binary_probability = None
    parent_threshold = None
    if protocol.get('parent_run'):
        parent = json.loads((Path(protocol['parent_run']) / 'selection.json').read_text())
        parent_threshold = float(parent['threshold'])
    for training_seed in seeds:
        for arm in arms:
            directory = study / 'runs' / f'{arm}_s{training_seed}'
            selection_path = directory / 'selection.json'
            relative = str(selection_path.relative_to(study))
            if barrier.get('selection_sha256', {}).get(relative) != sha(selection_path):
                raise ValueError(f'Changed or unlocked selection: {relative}')
            selection = json.loads(selection_path.read_text())
            result_path = directory / 'test_results.json'
            result = json.loads(result_path.read_text())
            if selection['protocol_sha256'] != protocol_hash:
                raise ValueError('Selection protocol mismatch')
            if selection['checkpoint_sha256'] != sha(directory / 'head.pt'):
                raise ValueError('Selected category head checkpoint changed')
            if result['test_dataset_sha256'] != dataset_hash:
                raise ValueError('Result belongs to another test label release')
            for split, value in (('train', train_hash), ('validation', validation_hash)):
                if selection.get(f'{split}_dataset_sha256', value) != value:
                    raise ValueError(f'Selection {split} dataset hash mismatch')
            if result.get('arm', arm) != arm or int(result.get('seed', training_seed)) != training_seed:
                raise ValueError('Result arm or seed mismatch')
            if result.get('checkpoint_sha256', selection['checkpoint_sha256']) != selection['checkpoint_sha256']:
                raise ValueError('Result checkpoint hash mismatch')
            threshold = float(selection['parent_binary_threshold'])
            if result.get('parent_binary_threshold', threshold) != threshold:
                raise ValueError('Test result binary threshold differs from its selection')
            if parent_threshold is not None and threshold != parent_threshold:
                raise ValueError('A type arm changed the parent binary threshold')
            thresholds.append(threshold)
            csv_path = directory / 'test.csv'
            frame, truth, predicted, gated = _predictions(csv_path, original, threshold, protocol)
            binary_probability = frame.binary_probability.to_numpy()
            if common_binary_probability is None:
                common_binary_probability = binary_probability.copy()
            elif not np.array_equal(binary_probability, common_binary_probability):
                raise ValueError('All type arms must reuse identical frozen binary predictions')
            harmful = truth > 0
            known = truth >= 0
            type_metrics = classification_metrics(truth[harmful], predicted[harmful], TYPE_CLASSES)
            gated_metrics = classification_metrics(truth[known], gated[known], ALL_CLASSES)
            _verify_metrics(result['type_metrics'], type_metrics, 'type_metrics')
            _verify_metrics(result['gated_metrics'], gated_metrics, 'gated_metrics')
            sources = frame.source.to_numpy()
            by_source = {}
            for source in sorted(set(sources)):
                mask = sources == source
                value = dict(type_metrics=classification_metrics(truth[mask & harmful], predicted[mask & harmful], TYPE_CLASSES),
                             gated_metrics=classification_metrics(truth[mask & known], gated[mask & known], ALL_CLASSES),
                             coverage=contract.coverage(truth[mask], original.prompt_harm_label.ne('unknown').to_numpy()[mask]
                                                        if intent else np.ones(int(mask.sum()), dtype=bool)))
                if intent:
                    binary_known_source = mask & original.prompt_harm_label.ne('unknown').to_numpy()
                    value['binary_metrics'] = classification_metrics(
                        original.prompt_harm_label.eq('harmful').to_numpy()[binary_known_source].astype(int),
                        (binary_probability[binary_known_source] >= threshold).astype(int), (0, 1))
                if source in result.get('by_source', {}):
                    for metric in ('type_metrics', 'gated_metrics'):
                        _verify_metrics(result['by_source'][source][metric], value[metric], f'{source}.{metric}')
                by_source[str(source)] = value
            arrays[(arm, training_seed)] = (truth, predicted, gated, frame.binary_probability.to_numpy())
            entries.append(dict(arm=arm, seed=training_seed, head_parameters=selection['head_parameters'],
                                best_epoch=selection['best_epoch'], type_metrics=type_metrics,
                                completed_epochs=selection.get('completed_epochs'),
                                stopped_early=selection.get('stopped_early'),
                                head_width=selection.get('head_width'),
                                gated_metrics=gated_metrics, by_source=by_source,
                                coverage=contract.coverage(truth),
                                parent_binary_threshold=threshold,
                                test_ce=result.get('test_ce'), checkpoint_sha256=selection['checkpoint_sha256']))
            for path in (selection_path, result_path, csv_path, directory / 'head.pt'):
                artifact_hashes[str(path.relative_to(study))] = sha(path)
    if len(set(thresholds)) != 1:
        raise ValueError('Every category head must reuse the identical binary threshold')
    binary_known = np.ones(len(original), dtype=bool)
    if intent:
        _, _, binary_truth, binary_known = contract.label_arrays(original)
        binary_truth_source = 'registered intent labels for WildGuard; retained original labels for other sources'
    elif 'prompt_harm_label' in original:
        if not original.prompt_harm_label.isin(('harmful', 'unharmful')).all():
            raise ValueError('Expected explicit original binary harmfulness labels')
        binary_truth = original.prompt_harm_label.eq('harmful').to_numpy()
        binary_truth_source = 'original prompt_harm_label'
    else:
        binary_truth = original.category_id.to_numpy() > 0
        binary_truth_source = 'original category_id > 0 (binary-label field unavailable)'
    binary_prediction = common_binary_probability >= thresholds[0]
    common_binary_metrics = classification_metrics(binary_truth[binary_known].astype(int), binary_prediction[binary_known].astype(int), (0, 1))
    common_binary_metrics.update(threshold=thresholds[0], truth_source=binary_truth_source,
        identical_for_all_type_heads=True, type_mechanism_changes_binary_model=False,
        original_binary_vs_primary_category_disagreements=int((binary_known & (original.category_id.to_numpy() >= 0) &
                (binary_truth != (original.category_id.to_numpy() > 0))).sum()),
        coverage=contract.coverage(original.category_id.to_numpy(), binary_known))
    comparisons = []
    comparison_pairs = [('description_queries', 'learned_queries'), ('description_queries', 'last_mlp')]
    if two_source:
        comparison_pairs.append(('learned_queries', 'last_mlp'))
    for training_seed in seeds:
        for candidate_arm, baseline in comparison_pairs:
            candidate = arrays[(candidate_arm, training_seed)]
            other = arrays[(baseline, training_seed)]
            if not np.array_equal(candidate[3], other[3]):
                raise ValueError('Compared arms used different cached binary predictions')
            harmful = candidate[0] > 0
            known = candidate[0] >= 0
            comparisons.append(dict(candidate=candidate_arm, baseline=baseline, training_seed=training_seed,
                type=paired_bootstrap(candidate[0][harmful], candidate[1][harmful], other[1][harmful], TYPE_CLASSES, repetitions, seed),
                gated=paired_bootstrap(candidate[0][known], candidate[2][known], other[2][known], ALL_CLASSES, repetitions, seed)))
    paired_seed_summary = []
    for candidate_arm, baseline in comparison_pairs:
        matched = [row for row in comparisons if row['candidate'] == candidate_arm and row['baseline'] == baseline]
        paired_seed_summary.append(dict(candidate=candidate_arm, baseline=baseline, seeds=seeds,
            **{scope: {metric: dict(**seed_statistics([row[scope][metric]['delta'] for row in matched]),
                positive_seeds=sum(row[scope][metric]['delta'] > 0 for row in matched))
                for metric in ('accuracy', 'macro_f1')} for scope in ('type', 'gated')}))
    equal_parameters = all(
        next(row['head_parameters'] for row in entries if row['arm'] == 'learned_queries' and row['seed'] == training_seed) ==
        next(row['head_parameters'] for row in entries if row['arm'] == 'description_queries' and row['seed'] == training_seed)
        for training_seed in seeds)
    unused_rows = protocol.get('unused_multilabel_annotation_rows', 1734)
    notes = ['本轮使用原始单主类标签，旧标签含机器标注噪声；结果不能解释为人工金标上的五类完整覆盖。',
             f'新试标的 {unused_rows:,} 条多标签记录完全未使用；本报告仅读取原始 inputs 及本轮模型产物。',
             '类型头只以原 category_id 1–5 训练，安全类 0 不参加类型训练；五类预测使用 softmax argmax。',
             '类型指标在原标签 1–5 子集计算；六类端到端指标保留全部样本，使用固定父模型有害判别作为门控。',
             '随机 query 与描述 query 同结构同参数，通常各 706,177 参数；末 token MLP 为 328,453 参数。',
             '描述对随机 query 是语义初始化的主要机制对照；描述对 MLP 同时改变聚合结构与参数量。',
             '类别宏平均固定包含五类或六类，缺支持类记零并单列支持标记。',
             '配对区间仅反映测试样本重采样，不包含训练随机性和旧标签噪声；McNemar 为未经多重比较校正的探索检验。',
             '这些既有测试身份曾被探索过，不称全新盲测；注意力权重不等于已验证的忠实解释。']
    parameter_counts = {arm: sorted(set(row['head_parameters'] for row in entries if row['arm'] == arm)) for arm in arms}
    notes[4] = '各方法实际训练参数量：' + '；'.join(
        f"{arm} = {', '.join(f'{count:,}' for count in counts)}" for arm, counts in parameter_counts.items()) + '。'
    if protocol.get('head_width_by_arm'):
        notes[5] = '按注册宽度调整 MLP 容量；参数近似匹配不能消除网络结构、聚合方式和优化难度差异。描述对随机 query 仍是语义初始化的机制对照。'
    if two_source:
        notes[0] = '使用中文整理语料与 JailBench 的全量既有修复三集，保留原主类别；未重标，WildGuard 未进入本轮类型训练和评估。'
        notes[1] = '多标签试标及待对齐国标的 WildGuard 意图重标均未使用；继承标签可用不等于已经独立人工复核。'
    if intent:
        notes[:4] = [
            '仅 WildGuard 中文来源按核心意图和直接目标重标主类，无固定类别优先级；中文整理语料和 JailBench 保留原标签。',
            '类别标签独立于待评模型输出，由注册的机器标注协议产生；未经完整人工复核，不能称为人工金标。',
            '仅明确类别 1–5 参加五类交叉熵及类型指标；安全类 0 与未定类 -1 不参加类型训练，预测仍为 softmax argmax。',
            '六类端到端指标仅覆盖 category_id>=0；二分类另按已知 harmful/unharmful 掩码统计，待审类型可能仍有已知二分类标签。',
            '按来源报告覆盖率与未知标签排除数量，不能把排除难例后的准确率与旧标签全量结果直接比较。',
            '标注协议、数据发布清单及三集内容哈希均锁定；旧多标签试标未用于本轮。']
    if len(seeds) == 1:
        notes.append('仅一个训练种子，不能据此宣称跨种子稳定提升。')
    else:
        notes.append('均值和样本标准差在同一测试集的训练种子之间计算；同一条测试文本不会因多种子被当作多个独立样本。')
    if protocol.get('early_stopping_patience', 0):
        notes.append(f"各方法最多 {protocol['epochs']} 轮，以验证 CE 最小值选检查点；patience={protocol['early_stopping_patience']}、min_delta={protocol.get('early_stopping_min_delta', 0.)} 仅控制提前停止。测试数据不参与停止或检查点选择。")
    if not equal_parameters:
        notes.append('警告：随机 query 与描述 query 参数量不一致，等参数核验失败。')
    summary = dict(protocol_sha256=protocol_hash, dataset_sha256=dict(train=train_hash, validation=validation_hash, test=dataset_hash),
                   training_rows_total=len(train), training_type_rows=int((train.category_id > 0).sum()),
                   training_safe_rows_excluded=int((train.category_id == 0).sum()),
                   training_unknown_rows_excluded=int((train.category_id < 0).sum()),
                   coverage_by_split={name: contract.coverage(frame.category_id.to_numpy(),
                       frame.prompt_harm_label.ne('unknown').to_numpy() if intent else np.ones(len(frame), dtype=bool))
                       for name, frame in [('train', train), ('validation', validation), ('test', original)]},
                   task=protocol.get('task', 'existing_primary_category_classification'),
                   new_multilabel_annotations_used=False, unused_multilabel_annotation_rows=unused_rows,
                   independent_human_gold=False, new_blind_test=False,
                   seeds=seeds, random_description_parameters_equal=equal_parameters,
                   common_binary_metrics=common_binary_metrics,
                   entries=entries, seed_aggregates=aggregate_seed_results(entries),
                   paired_comparisons=comparisons, paired_seed_summary=paired_seed_summary,
                   experiment_scope=protocol.get('experiment_scope'),
                   artifact_sha256=artifact_hashes, notes=notes)
    if intent:
        summary['annotation_provenance'] = {key: protocol[key] for key in (
            'annotation_protocol_sha256', 'annotation_manifest_sha256', 'annotation_source_scope',
            'retained_source_scope', 'label_quality', 'input_data_sha256')}
    dump(study / 'report.json', summary)
    names = dict(last_mlp='末 token MLP', learned_queries='随机 query', description_queries='描述 query')
    def percentage(value):
        return '—' if value is None else f'{100 * value:.2f}'
    lines = ['# hanguard 主类别机制实验', '',
             f"使用原始单主类标签；新生成的 {unused_rows:,} 条多标签试标完全未使用。百分比指标如下，类型指标限原标签 1–5 子集，六类指标使用固定二分类门控。", '',
             f"三种类型头共用同一组冻结二分类预测：{len(original):,} 条测试上的二分类准确率为 {percentage(common_binary_metrics['accuracy'])}%。类型机制没有改变有害／无害模型。", '',
             '| 方法 | 种子 | 参数量 | 类型准确率 | 类型 Macro F1 | 六类端到端准确率 | 六类 Macro F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    if intent:
        coverage = summary['coverage_by_split']['test']
        lines[2] = 'WildGuard 采用无优先级意图主类机器重标；其余两来源保留原标签。类型指标与二分类指标分别使用各自已知标签子集。'
        lines[4] = (f"测试共 {len(original):,} 条；明确有害类型 {coverage['type_rows']:,} 条，六类指标覆盖 "
                    f"{coverage['end_to_end_rows']:,} 条（{percentage(coverage['end_to_end_fraction'])}%），"
                    f"二分类覆盖 {coverage['binary_rows']:,} 条（{percentage(coverage['binary_fraction'])}%）。"
                    f"共用冻结二分类门控，已知二分类标签上的准确率为 {percentage(common_binary_metrics['accuracy'])}%。")
    elif two_source:
        lines[2] = '使用中文整理语料与 JailBench 的全量既有修复三集。各组共享冻结特征与原标签；类型指标只覆盖有害类 1–5，六类指标使用固定二分类门控。'
    for row in entries:
        t, g = row['type_metrics'], row['gated_metrics']
        lines.append(f"| {names[row['arm']]} | {row['seed']} | {row['head_parameters']:,} | {percentage(t['accuracy'])} | {percentage(t['macro_f1'])} | {percentage(g['accuracy'])} | {percentage(g['macro_f1'])} |")
    def mean_std(value):
        return f"{percentage(value['mean'])} ± {percentage(value['std'])}"
    if len(seeds) > 1:
        lines.extend(['', '按训练种子汇总：均值 ± 样本标准差，单位为百分点；不汇总重复样本计算置信区间。', '',
                      '| 方法 | 种子数 | 类型准确率 | 类型 Macro F1 | 六类准确率 | 六类 Macro F1 |',
                      '|---|---:|---:|---:|---:|---:|'])
        for row in summary['seed_aggregates']:
            t, g = row['type_metrics'], row['gated_metrics']
            lines.append(f"| {names[row['arm']]} | {len(row['seeds'])} | {mean_std(t['accuracy'])} | {mean_std(t['macro_f1'])} | {mean_std(g['accuracy'])} | {mean_std(g['macro_f1'])} |")
    lines.extend(['', f'配对 bootstrap：{repetitions} 次，种子 {seed}；差值单位为百分点。', '',
                  '| 比较 | 训练种子 | 指标范围 | 准确率差值 [95% 区间] | Macro F1 差值 [95% 区间] | McNemar p |',
                  '|---|---:|---|---:|---:|---:|'])
    def interval(value):
        return f"{percentage(value['delta'])} [{percentage(value['ci95'][0])}, {percentage(value['ci95'][1])}]"
    for comparison in comparisons:
        for key, label in (('type', '明确标签 1–5' if intent else '原标签 1–5'), ('gated', '门控六类')):
            value = comparison[key]
            lines.append(f"| {names[comparison['candidate']]} − {names[comparison['baseline']]} | {comparison['training_seed']} | {label} | {interval(value['accuracy'])} | {interval(value['macro_f1'])} | {value['mcnemar_exact']['two_sided_p']:.4g} |")
    if len(seeds) > 1:
        lines.extend(['', '| 同种子差值汇总 | 类型准确率差值均值 ± 标准差 | 类型 Macro F1 差值均值 ± 标准差 | 准确率提升种子数 |',
                      '|---|---:|---:|---:|'])
        for row in paired_seed_summary:
            values = row['type']
            lines.append(f"| {names[row['candidate']]} − {names[row['baseline']]} | {mean_std(values['accuracy'])} | {mean_std(values['macro_f1'])} | {values['accuracy']['positive_seeds']}/{len(seeds)} |")
    lines.extend(['', '区间包含 0 时，当前测试样本不足以支持明确的提升方向；不含 0 也仍受训练随机性和继承标签质量限制。', '',
                  '| 方法 | 种子 | 来源 | 类型样本数 | 类型准确率 | 类型 Macro F1 | 六类准确率 | 六类 Macro F1 |',
                  '|---|---:|---|---:|---:|---:|---:|---:|'])
    for row in entries:
        for source, metrics in row['by_source'].items():
            t, g = metrics['type_metrics'], metrics['gated_metrics']
            lines.append(f"| {names[row['arm']]} | {row['seed']} | {source} | {t['rows']} | {percentage(t['accuracy'])} | {percentage(t['macro_f1'])} | {percentage(g['accuracy'])} | {percentage(g['macro_f1'])} |")
    if len(seeds) > 1:
        lines.extend(['', '| 方法 | 来源 | 类型准确率均值 ± 标准差 | 类型 Macro F1 均值 ± 标准差 | 六类准确率均值 ± 标准差 | 六类 Macro F1 均值 ± 标准差 |',
                      '|---|---|---:|---:|---:|---:|'])
        for row in summary['seed_aggregates']:
            for source, values in row['by_source'].items():
                t, g = values['type_metrics'], values['gated_metrics']
                lines.append(f"| {names[row['arm']]} | {source} | {mean_std(t['accuracy'])} | {mean_std(t['macro_f1'])} | {mean_std(g['accuracy'])} | {mean_std(g['macro_f1'])} |")
    if intent:
        lines.extend(['', '| 来源 | 总数 | 明确有害类型 | 六类已知数 | 六类覆盖率 | 二分类已知数 | 二分类覆盖率 |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for source, value in entries[0]['by_source'].items():
            c = value['coverage']
            lines.append(f"| {source} | {c['total_rows']} | {c['type_rows']} | {c['end_to_end_rows']} | {percentage(c['end_to_end_fraction'])}% | {c['binary_rows']} | {percentage(c['binary_fraction'])}% |")
    lines.extend(['', '每类支持数、精确率、召回率、F1 与完整混淆矩阵见同目录 report.json；混淆矩阵行是真实原标签，列是预测。', ''])
    lines.extend(f'- {note}' for note in notes)
    (study / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap-repetitions', default=1000, type=int)
    parser.add_argument('--bootstrap-seed', default=42, type=int)
    args = parser.parse_args()
    result = report(args.output, args.bootstrap_repetitions, args.bootstrap_seed)
    print(json.dumps(dict(report=str(args.output / 'report.md'), runs=len(result['entries']),
                          random_description_parameters_equal=result['random_description_parameters_equal']), ensure_ascii=False))


if __name__ == '__main__':
    main()
