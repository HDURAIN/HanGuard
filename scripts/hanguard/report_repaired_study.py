"""Independent CPU report for the repaired-data study, including partial progress."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/three_source_translation_repaired'
METRICS = ('accuracy', 'f1', 'recall', 'fpr', 'auroc', 'auprc')
FOUR_CELLS = {
    'lora_depth': ('E01', 'E02', 'E03', 'E04'),
    'prototype': ('E03', 'E04', 'E07', 'E08'),
    'frozen_pseudolabel': ('E01', 'E02', 'E10', 'E12'),
    'lora_pseudolabel': ('E03', 'E04', 'E14', 'E16'),
    'attention': ('E03', 'E04', 'E17', 'E18'),
    'scl': ('E03', 'E04', 'E19', 'E20'),
}
CONTROLS = [('E04', 'E05'), ('E10', 'E09'), ('E12', 'E11'),
            ('E14', 'E13'), ('E16', 'E15'), ('E17', 'E06'),
            ('E09', 'E01'), ('E11', 'E02'), ('E13', 'E03'), ('E15', 'E04')]
NAMES=dict(E01='冻结＋末层 MLP',E02='冻结＋多层融合',E03='LoRA＋末层 MLP',E04='LoRA＋多层融合',
    E05='LoRA＋等参数末层对照',E06='LoRA＋全文均值聚合',E07='LoRA＋多原型',E08='LoRA＋融合＋多原型',
    E09='冻结＋固定位置监督',E10='冻结＋动态伪标签',E11='冻结＋融合＋固定位置监督',E12='冻结＋融合＋动态伪标签',
    E13='LoRA＋固定位置监督',E14='LoRA＋动态伪标签',E15='LoRA＋融合＋固定位置监督',E16='LoRA＋融合＋动态伪标签',
    E17='LoRA＋注意力聚合',E18='LoRA＋融合＋注意力聚合',E19='LoRA＋SCL',E20='LoRA＋融合＋SCL')


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics(y, probability, threshold=.5, ranking=True):
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    require(len(y) == len(p) and len(y) > 0, 'Empty or mismatched predictions')
    require(np.isin(y, [0, 1]).all(), 'Labels must be binary')
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), 'Invalid probability')
    require(np.isfinite(threshold), 'Nonfinite threshold')
    pred = p >= threshold
    positive, negative = y == 1, y == 0
    tp, fp = int((pred & positive).sum()), int((pred & negative).sum())
    tn, fn = int((~pred & negative).sum()), int((~pred & positive).sum())
    return dict(rows=len(y), threshold=float(threshold), accuracy=float((pred == y).mean()),
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else None,
                fpr=fp / (fp + tn) if fp + tn else None,
                auroc=float(roc_auc_score(y, p)) if ranking and positive.any() and negative.any() else None,
                auprc=float(average_precision_score(y, p)) if ranking and positive.any() else None,
                tp=tp, fp=fp, tn=tn, fn=fn)


def accuracy_threshold(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    metrics(y, p)
    choices = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    correct = len(pos) - np.searchsorted(pos, choices, side='left') + np.searchsorted(neg, choices, side='left')
    best = np.flatnonzero(correct == correct.max())
    return float(choices[best[np.argmin(np.abs(choices[best] - .5))]])


def metadata(data, split):
    frame = pd.read_parquet(Path(data) / f'{split}.parquet')
    require({'base_id', 'source', 'prompt_harm_label', 'prompt_tokens'} <= set(frame), 'Missing dataset metadata')
    require(frame.base_id.is_unique, f'{split}: duplicate IDs')
    require(set(frame.prompt_harm_label) <= {'harmful', 'unharmful'}, f'{split}: invalid labels')
    result = frame[['base_id', 'source', 'prompt_tokens']].copy()
    result['base_id'] = result.base_id.astype(str)
    result['label'] = frame.prompt_harm_label.eq('harmful').astype(int)
    return result.sort_values('base_id').reset_index(drop=True)


def verified_predictions(path, expected, probability=True):
    frame = pd.read_csv(path, dtype={'base_id': str, 'source': str}, float_precision='round_trip')
    required = {'base_id', 'label', 'source', 'prompt_tokens'} | ({'probability'} if probability else set())
    require(required <= set(frame), f'{path}: missing columns')
    require(frame.base_id.is_unique and len(frame) == len(expected), f'{path}: wrong IDs/count')
    frame = frame.sort_values('base_id').reset_index(drop=True)
    for key in ('base_id', 'label', 'source', 'prompt_tokens'):
        require(np.array_equal(frame[key], expected[key]), f'{path}: changed {key}')
    if probability:
        metrics(frame.label, frame.probability)
    return frame


def stratified(frame, threshold=.5, ranking=True):
    compute = lambda group: metrics(group.label, group.probability, threshold, ranking)
    source = {str(name): compute(group) for name, group in frame.groupby('source', sort=True)}
    lengths = frame.prompt_tokens
    masks = {'le_370': lengths <= 370, 'gt_370': lengths > 370,
             '371_512': (lengths > 370) & (lengths <= 512),
             '513_1024': (lengths > 512) & (lengths <= 1024),
             '1025_2048': (lengths > 1024) & (lengths <= 2048), 'gt_2048': lengths > 2048}
    groups = {name: compute(frame[mask]) if mask.any() else None for name, mask in masks.items()}
    return dict(metrics=compute(frame), by_source=source, length_groups=groups)


def aggregate(rows, key='metrics'):
    output = {'count': len(rows), 'seeds': sorted(row['seed'] for row in rows), 'metrics': {}}
    for metric in METRICS:
        values = [row[key][metric] for row in rows if row[key].get(metric) is not None]
        output['metrics'][metric] = dict(mean=float(np.mean(values)) if values else None,
            std=float(np.std(values, ddof=1)) if len(values) > 1 else None, count=len(values))
    return output


def paired_contrast(records, frames, left, right, seeds):
    lookup = {(row['arm'], row['seed']): row for row in records}
    common = [seed for seed in seeds if (left, seed) in lookup and (right, seed) in lookup]
    output = dict(left=left, right=right, seeds=common, complete=len(common) == len(seeds),
                  differences_pp={}, discordance=[])
    for metric in METRICS:
        values = [100 * (lookup[left, seed]['metrics'][metric] - lookup[right, seed]['metrics'][metric])
                  for seed in common if lookup[left, seed]['metrics'][metric] is not None
                  and lookup[right, seed]['metrics'][metric] is not None]
        output['differences_pp'][metric] = dict(values=values, mean=float(np.mean(values)) if values else None,
            std=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    for seed in common:
        a, b = frames[left, seed], frames[right, seed]
        require(np.array_equal(a.base_id, b.base_id), 'Paired samples differ')
        ac = (a.probability >= lookup[left, seed]['threshold']).to_numpy() == a.label.to_numpy()
        bc = (b.probability >= lookup[right, seed]['threshold']).to_numpy() == b.label.to_numpy()
        won, lost = int((ac & ~bc).sum()), int((~ac & bc).sum())
        output['discordance'].append(dict(seed=seed, left_only_correct=won, right_only_correct=lost,
            net_correct=won-lost, mcnemar_exact_p=float(binomtest(won, won+lost, .5).pvalue) if won+lost else 1.))
    return output


def verify_recorded(computed, recorded, context):
    for key in ('accuracy', 'f1', 'recall', 'fpr', 'auroc'):
        if key not in recorded:
            continue
        actual, saved = computed[key], recorded[key]
        require((actual is None and saved is None) or (actual is not None and saved is not None and
                np.isclose(actual, saved, atol=1e-10, rtol=0)), f'{context}: {key} disagrees')


def report(study, data=DATA):
    study, data = Path(study), Path(data)
    protocol = read_json(study / 'protocol.json')
    seeds = protocol.get('seeds', [42, 43, 44])
    specs = read_json(study / 'arms.json') if (study / 'arms.json').exists() else {f'E{i:02d}': {} for i in range(1,21)}
    available_arms = list(specs) if isinstance(specs, dict) else [x['arm'] if isinstance(x, dict) else x for x in specs]
    arms = protocol.get('all_arms', available_arms)
    reference_names = protocol.get('references', ['R01', 'R02'])
    require(arms and len(arms)==len(set(arms)) and set(arms)<=set(available_arms), 'Invalid registered arms')
    require(seeds and len(seeds)==len(set(seeds)), 'Invalid registered seeds')
    require(len(reference_names)==len(set(reference_names)) and set(reference_names)<={'R01','R02'},
            'Invalid registered references')
    single_seed = len(seeds)==1
    hashes = {split: sha256(data / f'{split}.parquet') for split in ('train', 'validation', 'test')}
    registered = protocol.get('data_sha256', protocol.get('data_hashes', protocol.get('dataset_hashes', {})))
    for split, digest in registered.items():
        key = split.removesuffix('.parquet')
        if key in hashes:
            require(digest == hashes[key], f'Dataset hash changed: {key}')
    barrier_path = study / 'test_barrier.json'
    barrier = read_json(barrier_path) if barrier_path.exists() else {}
    if barrier.get('locked'):
        require(barrier.get('protocol_sha256') == sha256(study/'protocol.json'), 'Frozen protocol changed')
    expected = {split: metadata(data, split) for split in ('validation', 'test')}
    records, progress, frames = [], [], {}
    for arm in arms:
        for seed in seeds:
            directory = study / 'runs' / f'{arm}_s{seed}'
            status = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {'state': 'pending'}
            item = dict(arm=arm, seed=seed, status=status, selection_complete=False, test_complete=False)
            selection_path = directory / 'selection.json'
            if selection_path.exists():
                selection = read_json(selection_path)
                require(selection.get('arm', arm) == arm and selection.get('seed', seed) == seed, 'Selection identity differs')
                val = verified_predictions(directory / 'validation.csv', expected['validation'])
                threshold = selection['threshold']
                require(threshold == accuracy_threshold(val.label, val.probability), f'{arm}/{seed}: non-validation threshold')
                for name, digest in selection.get('checkpoint_hashes', {}).items():
                    require(sha256(directory / name) == digest, f'{arm}/{seed}: checkpoint hash changed')
                verify_recorded(metrics(val.label, val.probability, threshold),
                    selection.get('metrics', selection.get('validation_metrics', {})), f'{arm}/{seed} validation')
                item.update(selection_complete=True, validation_accuracy=metrics(val.label, val.probability, threshold)['accuracy'])
                if (directory / 'test.csv').exists():
                    require(barrier.get('locked') is True, 'Test predictions exist before selection barrier')
                    relative=str(selection_path.relative_to(study))
                    require(barrier.get('selection_sha256',{}).get(relative)==sha256(selection_path),
                            f'Frozen selection changed or missing: {relative}')
                    test = verified_predictions(directory / 'test.csv', expected['test'])
                    result = dict(arm=arm, seed=seed, threshold=threshold, **stratified(test, threshold),
                                  fixed_05=stratified(test, .5), selection_sha256=sha256(selection_path))
                    saved_path = directory / 'test_results.json'
                    if saved_path.exists():
                        verify_recorded(result['metrics'], read_json(saved_path).get('metrics', {}), f'{arm}/{seed} test')
                    records.append(result); frames[arm, seed] = test; item['test_complete'] = True
            progress.append(item)
    for relative, digest in barrier.get('selection_sha256', barrier.get('selection_hashes', {})).items():
        require(sha256(study / relative) == digest, f'Frozen selection changed: {relative}')
    aggregates = {arm: aggregate([row for row in records if row['arm'] == arm]) for arm in arms}
    fixed_05 = {arm: aggregate([dict(row, fixed_metrics=row['fixed_05']['metrics']) for row in records if row['arm'] == arm], 'fixed_metrics') for arm in arms}
    pairs = set(CONTROLS)
    for baseline, a, b, ab in FOUR_CELLS.values():
        pairs.update([(a, baseline), (b, baseline), (ab, a), (ab, b)])
    pairs.add(('E03', 'E02'))
    pairs = {pair for pair in pairs if set(pair)<=set(arms)}
    contrasts = [paired_contrast(records, frames, left, right, seeds) for left,right in sorted(pairs)]
    contrast_map = {(item['left'],item['right']): item for item in contrasts}
    acceptance = {}
    for name, (baseline, a, b, ab) in FOUR_CELLS.items():
        if not {baseline,a,b,ab}<=set(arms):
            continue
        comparisons = [contrast_map[pair] for pair in [(a,baseline),(b,baseline),(ab,a),(ab,b)]]
        complete = all(c['complete'] for c in comparisons)
        acceptance[name] = dict(complete=complete, arms=[baseline,a,b,ab],
            positive_means=all(c['differences_pp']['accuracy']['mean'] is not None and c['differences_pp']['accuracy']['mean'] > 0 for c in comparisons) if complete else None,
            all_seed_directions_positive=all(all(v > 0 for v in c['differences_pp']['accuracy']['values']) for c in comparisons) if complete and not single_seed else None,
            minimum_gain_pp=min(c['differences_pp']['accuracy']['mean'] for c in comparisons) if complete else None,
            note='Direction and magnitude only; no post-test practical-gain threshold or success claim.')
    if {'E01','E02','E03','E04','E05'}<=set(arms):
        order_checks=[contrast_map[pair] for pair in [('E02','E01'),('E03','E02'),('E04','E03'),('E04','E05')]]
        order_complete=all(c['complete'] for c in order_checks)
        acceptance['lora_depth_order']=dict(complete=order_complete, arms=['E01','E02','E03','E04','E05'],
            positive_means=all(c['differences_pp']['accuracy']['mean']>0 for c in order_checks) if order_complete else None,
            all_seed_directions_positive=all(all(v>0 for v in c['differences_pp']['accuracy']['values']) for c in order_checks) if order_complete and not single_seed else None,
            minimum_gain_pp=min(c['differences_pp']['accuracy']['mean'] for c in order_checks) if order_complete else None,
            note='E04 > E03 > E02 > E01 and E04 > matched-capacity E05; no post-test magnitude cutoff.')
    strata = {}
    for arm in arms:
        rows = [r for r in records if r['arm'] == arm]
        strata[arm] = {}
        for family in ('by_source','length_groups'):
            names = sorted({name for row in rows for name in row[family]})
            strata[arm][family] = {name: aggregate([dict(seed=r['seed'], metrics=r[family][name]) for r in rows if r[family].get(name) is not None]) for name in names}
    references = reference_records(study, expected, barrier, reference_names)
    seed_note=('本轮只有一个训练种子，是单种子探索，不能作为跨种子稳定性或多种子提升的证明。'
               if single_seed else f'本轮登记 {len(seeds)} 个独立训练种子；这些种子共享相同测试样本，不能合并为 {len(seeds)} 倍独立样本。')
    summary = dict(generated=time.time(), data_hashes=hashes, arms=arms, seeds=seeds,
        single_seed_exploration=single_seed, expected_references=reference_names, expected_runs=len(arms)*len(seeds),
        selected_runs=sum(p['selection_complete'] for p in progress), completed_runs=len(records),
        training_complete=all(p['selection_complete'] for p in progress), test_barrier_locked=barrier.get('locked',False),
        complete=len(records)==len(arms)*len(seeds) and all(r.get('complete') for r in references.values()),
        progress=progress, records=records, aggregates=aggregates, fixed_05=fixed_05,
        strata=strata, contrasts=contrasts, acceptance=acceptance, references=references,
        caveat='既有划分上的修复后探索比较；测试历史版本已多轮查看，不是新的盲测。成对 p 值未校正多重比较。'+seed_note)
    dump(study / 'summary.json', summary)
    dump(study / 'verification.json', dict(passed=True, checked_runs=len(records), partial=not summary['complete'], data_hashes=hashes))
    lines = ['# hanguard 修复数据实验报告', '',
        f"已锁定训练选择 {summary['selected_runs']}/{summary['expected_runs']}；已完成测试 {len(records)}/{summary['expected_runs']}。" + ('全部完成。' if summary['complete'] else '当前是不完整进度报告。'), '',
        '主表采用验证集准确率阈值；数值为百分比。'+('本轮为单种子结果，不计算种子标准差。' if single_seed else '± 为独立训练种子的样本标准差。')+'固定 0.5、逐来源、长文本、成对错误和组合比较保存在 summary.json。', '',
        '| 配置 | 完成种子 | 准确率 | 有害 F1 | 召回率 | 误报率 |', '|---|---:|---:|---:|---:|---:|']
    def cell(value):
        if value['mean'] is None: return '—'
        suffix = f" ± {100*value['std']:.3f}" if value['std'] is not None else ''
        return f"{100*value['mean']:.3f}" + suffix
    for arm in arms:
        item=aggregates[arm]
        lines.append('| '+arm+' '+NAMES.get(arm,'')+' | '+str(item['count'])+' | '+' | '.join(cell(item['metrics'][k]) for k in ('accuracy','f1','recall','fpr'))+' |')
    lines += ['', '外部参考（不同基座／训练数据，属于系统效果对比）：', '']
    for name, ref in references.items():
        if not ref['complete']:
            lines.append(f'- {name}：尚未完成。'); continue
        for variant, item in ref['results'].items():
            label={'fixed_05_zero_shot':'固定 0.5 纯零样本','validation_calibrated':'验证集标签校准阈值',
                   'unsafe_only':'仅 Unsafe 有害','unsafe_or_controversial':'Unsafe＋Controversial 有害'}.get(variant,variant)
            m=item['metrics']; lines.append(f"- {name} / {label}：准确率 {100*m['accuracy']:.3f}%，F1 {100*m['f1']:.3f}%。")
    lines += ['', '组合比较（不依据测试筛掉失败配置）：', '']
    for name, result in acceptance.items():
        text = '未完成' if not result['complete'] else f"四项差值均为正：{result['positive_means']}；最小提升 {result['minimum_gain_pp']:.3f} 个百分点"+('；仅单种子探索' if single_seed else f"；各种子方向一致：{result['all_seed_directions_positive']}")
        lines.append(f'- {name}：{text}。')
    lines += ['', summary['caveat'], '']
    (study / 'report.md').write_text('\n'.join(lines))
    return summary


def reference_records(study, expected, barrier, reference_names=('R01','R02')):
    references = {}
    for name in reference_names:
        directory = study / 'references' / name
        status = read_json(directory/'status.json') if (directory/'status.json').exists() else {'state':'pending'}
        references[name] = dict(complete=False, status=status)
        if not (directory/'test.csv').exists(): continue
        require(barrier.get('locked') is True, 'Reference test before selection barrier')
        selection = read_json(directory/'selection.json')
        relative=str((directory/'selection.json').relative_to(study))
        require(barrier.get('selection_sha256',{}).get(relative)==sha256(directory/'selection.json'),
                f'Frozen reference selection changed or missing: {relative}')
        if name == 'R01':
            val=verified_predictions(directory/'validation.csv',expected['validation'])
            threshold=accuracy_threshold(val.label,val.probability)
            require(threshold == selection['threshold'], 'R01 threshold not validation-derived')
            frame=verified_predictions(directory/'test.csv',expected['test'])
            results=dict(fixed_05_zero_shot=stratified(frame,.5), validation_calibrated=stratified(frame,threshold))
        else:
            frame=verified_predictions(directory/'test.csv',expected['test'],probability=False)
            require('safety_label' in frame and frame.safety_label.isin(['safe','unsafe','controversial']).all(), 'Invalid Guard labels')
            results={}
            for variant, labels in [('unsafe_only',['unsafe']), ('unsafe_or_controversial',['unsafe','controversial'])]:
                results[variant]=stratified(frame.assign(probability=frame.safety_label.isin(labels).astype(float)),.5,ranking=False)
        references[name]=dict(complete=True, status=status, results=results)
    return references


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True); parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args()
    try:
        result=report(args.study,args.data)
        print(json.dumps({k:result[k] for k in ('selected_runs','completed_runs','expected_runs','complete')}))
    except Exception as error:
        dump(Path(args.study)/'verification.json',dict(passed=False,error=repr(error),time=time.time()))
        raise
