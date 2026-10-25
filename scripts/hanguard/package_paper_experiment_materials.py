"""Export paper tables from completed, immutable experiment evidence.

This command performs no training, inference, subset selection, or metric
tuning. Values are independently recomputed from the saved per-seed metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiment_results/20261025_paper_ready'
PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
FOLLOWUP = ROOT / 'outputs/hanguard_paper_followups_20261025'
NATIVE = ROOT / 'outputs/qwen3guard_clean_test_20261025_v2'
SEEDS = (42, 43, 44)
CORE = ('baseline', 'fusion_only', 'category_only', 'fusion_category')
GROUP = dict(zip(CORE, 'ABCD'))
SCHEMA = 'hanguard_paper_materials_20261025_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stats(values, scale=100.):
    values = [float(v) * scale for v in values]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Exactly three finite seed values are required')
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), values=values)


def binary_macro(m):
    tp, tn, fp, fn = (m[k] for k in ('tp', 'tn', 'fp', 'fn'))
    return .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


def source_paths():
    paths = [PARENT / 'protocol.json', PARENT / 'results.json', PARENT / 'selection_lock.json',
        NATIVE / 'protocol.json', NATIVE / 'results.json',
        FOLLOWUP / 'pooling/results.json', FOLLOWUP / 'external/results.json',
        ROOT / 'data/hanguard_two_source_remediated_20261012/descriptions.json',
        ROOT / 'scripts/hanguard/clean_ablation_heads.py',
        ROOT / 'scripts/hanguard/multilabel_heads.py', Path(__file__).resolve()]
    for task in ('binary', 'category'):
        paths += [PARENT / 'heads' / task / f's{s}' / 'selection.json' for s in SEEDS]
    paths += [FOLLOWUP / 'analysis' / name for name in (
        'paired_changes_by_seed.csv', 'paired_summary.csv', 'strata_summary.csv',
        'parameters.csv', 'efficiency.csv', 'efficiency_complete.json')]
    return paths


def load_evidence():
    result, protocol = read(PARENT / 'results.json'), read(PARENT / 'protocol.json')
    guard, guard_protocol = read(NATIVE / 'results.json'), read(NATIVE / 'protocol.json')
    pooling, external = read(FOLLOWUP / 'pooling/results.json'), read(FOLLOWUP / 'external/results.json')
    if not all(r.get('complete') for r in (result, guard, pooling, external)):
        raise ValueError('All included sources must be complete')
    if (result['protocol_sha256'] != sha(PARENT / 'protocol.json')
            or result['rootselection_lock_sha256'] != sha(PARENT / 'selection_lock.json')
            or guard['protocol_sha256'] != sha(NATIVE / 'protocol.json')
            or protocol['data_sha256']['test'] != guard['data_sha256']
            or guard['rows'] != 8350 or guard_protocol['training'] is not False
            or guard['invalid_safety_rows'] != 0
            or set(guard['mappings']) != {'unsafe_only', 'unsafe_or_controversial'}):
        raise ValueError('Native comparison identity, scope or protocol mismatch')
    if (protocol['seeds'] != list(SEEDS) or protocol['fusion_layers'] != [8, 16, 24, 32]
            or protocol['fusion_weights'] != [.25] * 4
            or protocol['posthoc_subset_enabled'] or protocol['selected_test_exclusions']):
        raise ValueError('This bundle must describe the full registered clean study')
    index = {(r['task'], r['arm']): r for r in result['aggregate']
             if r['task'] in ('binary', 'category')}
    raw, aggregates = {}, {}
    for task in ('binary', 'category'):
        for arm in protocol['arms']:
            entries = []
            for seed in SEEDS:
                value = result['per_seed'][str(seed)][task]['arms'][arm]
                m = dict(value['full_original_test']['validation_threshold'] if task == 'binary'
                         else value['native_known_harmful'])
                if m.get('n', m.get('rows')) != (8350 if task == 'binary' else 2180):
                    raise ValueError('Unexpected task denominator')
                if task == 'binary':
                    m['macro_f1'] = binary_macro(m)
                entries.append(m)
            raw[task, arm] = entries
            metrics = ('accuracy', 'f1', 'macro_f1', 'recall', 'precision', 'auroc', 'fpr') if task == 'binary' else ('accuracy', 'macro_f1')
            aggregates[task, arm] = {name: stats([v[name] for v in entries]) for name in metrics}
            for name, saved in index[task, arm]['metrics'].items():
                calc = aggregates[task, arm][name]
                if (not math.isclose(calc['mean'], 100 * saved['mean'], abs_tol=1e-10)
                        or not math.isclose(calc['sd'], 100 * saved['sample_std'], abs_tol=1e-10)):
                    raise ValueError(f'Saved aggregate differs from seeds: {task}/{arm}/{name}')
    for task in ('binary', 'category'):
        for seed in SEEDS:
            selected = read(PARENT / 'heads' / task / f's{seed}' / 'selection.json')
            if selected['final_budget'] != 12 or selected['extensions']:
                raise ValueError('Main study actual training budget changed')
    return dict(result=result, protocol=protocol, guard=guard, guard_protocol=guard_protocol,
                pooling=pooling, external=external, raw=raw, aggregates=aggregates)


def csv_text(rows):
    if not rows:
        raise ValueError('Empty table')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def tex_table(headers, rows, columns):
    return '\n'.join(['% Requires \\usepackage{booktabs}. Values are percentages unless marked pp.',
        '\\begin{tabular}{' + columns + '}', '\\toprule', ' & '.join(headers) + r' \\',
        '\\midrule', *[' & '.join(row) + r' \\' for row in rows], '\\bottomrule', '\\end{tabular}', ''])


def pm(metric):
    return f"${metric['mean']:.2f} \\pm {metric['sd']:.2f}$"


def build_tables(e):
    a, raw = e['aggregates'], e['raw']
    parameters = {}
    with (FOLLOWUP / 'analysis/parameters.csv').open() as f:
        for r in csv.DictReader(f):
            key, n = (r['task'], r['arm']), int(r['trainable_parameters'])
            if key in parameters and parameters[key] != n:
                raise ValueError('Architecture parameter count differs between seeds')
            parameters[key] = n
    main, main_tex = [], []
    for arm in CORE:
        row = dict(group=GROUP[arm], arm=arm, cqa=int(arm in CORE[2:]),
            mlf=int(arm in ('fusion_only', 'fusion_category')),
            binary_head_parameters=parameters['binary', arm], category_head_parameters=parameters['category', arm])
        for task in ('binary', 'category'):
            for name, value in a[task, arm].items():
                row[f'{task}_{name}_mean_percent'] = value['mean']
                row[f'{task}_{name}_sd_percent'] = value['sd']
        main.append(row)
        main_tex.append([GROUP[arm], str(row['cqa']), str(row['mlf']),
            pm(a['binary', arm]['accuracy']), pm(a['binary', arm]['f1']),
            pm(a['category', arm]['accuracy']), pm(a['category', arm]['macro_f1'])])
    seed_rows, class_rows = [], []
    for task in ('binary', 'category'):
        for arm in CORE:
            for seed, m in zip(SEEDS, raw[task, arm]):
                seed_rows.append(dict(task=task, group=GROUP[arm], arm=arm, seed=seed,
                    rows=m.get('n', m.get('rows')), accuracy_percent=100*m['accuracy'],
                    macro_f1_percent=100*m['macro_f1'], harmful_f1_percent=100*m['f1'] if task=='binary' else '',
                    auroc_percent=100*m['auroc'] if task=='binary' else '',
                    fpr_percent=100*m['fpr'] if task=='binary' else '',
                    threshold=m['threshold'] if task=='binary' else ''))
            if task == 'category':
                for category in range(1, 6):
                    values = [next(c for c in m['per_class'] if c['category_id']==category) for m in raw[task,arm]]
                    if len({v['support'] for v in values}) != 1:
                        raise ValueError('Class support differs between seeds')
                    row=dict(group=GROUP[arm], arm=arm, category_id=category, support=values[0]['support'])
                    for name in ('precision','recall','f1'):
                        v=stats([m[name] for m in values]);row[name+'_mean_percent']=v['mean'];row[name+'_sd_percent']=v['sd']
                    class_rows.append(row)
    delta, delta_tex = [], []
    for candidate, baseline in (('fusion_only','baseline'),('category_only','baseline'),
            ('fusion_category','baseline'),('fusion_category','category_only'),('fusion_category','fusion_only')):
        values={}
        for task in ('binary','category'):
            ds=[c['accuracy']-b['accuracy'] for c,b in zip(raw[task,candidate],raw[task,baseline])]
            v=stats(ds);values[task]=v
            delta.append(dict(task=task, candidate=candidate, baseline=baseline,
                comparison=f'{GROUP[candidate]}-{GROUP[baseline]}', delta_accuracy_mean_pp=v['mean'],
                delta_accuracy_sd_pp=v['sd'], **{f'seed_{seed}_delta_pp':x for seed,x in zip(SEEDS,v['values'])}))
        delta_tex.append([f'{GROUP[candidate]}$-${GROUP[baseline]}',pm(values['binary']),pm(values['category'])])
    native=[]
    for mapping, m in e['guard']['mappings'].items():
        native.append(dict(model='Qwen3Guard-Gen-4B', protocol='native_zero_shot', rule=mapping, rows=m['n'],
            accuracy_mean_percent=100*m['accuracy'],accuracy_sd_percent='',
            macro_f1_mean_percent=100*m['macro_f1'],macro_f1_sd_percent='',
            harmful_f1_mean_percent=100*m['harmful_f1'],harmful_f1_sd_percent='',
            harmful_recall_mean_percent=100*m['harmful_recall'],harmful_recall_sd_percent='',
            fpr_mean_percent=100*m['fpr'],fpr_sd_percent='',category_accuracy='N/A: incompatible native taxonomy'))
    d=a['binary','fusion_category']
    native.append(dict(model='hanguard CQA+MLF',protocol='local_supervised_components',rule='validation_selected_threshold',rows=8350,
        accuracy_mean_percent=d['accuracy']['mean'],accuracy_sd_percent=d['accuracy']['sd'],
        macro_f1_mean_percent=d['macro_f1']['mean'],macro_f1_sd_percent=d['macro_f1']['sd'],
        harmful_f1_mean_percent=d['f1']['mean'],harmful_f1_sd_percent=d['f1']['sd'],
        harmful_recall_mean_percent=d['recall']['mean'],harmful_recall_sd_percent=d['recall']['sd'],
        fpr_mean_percent=d['fpr']['mean'],fpr_sd_percent=d['fpr']['sd'],
        category_accuracy=f"{a['category','fusion_category']['accuracy']['mean']:.4f}"))
    native_tex=[]
    for r in native:
        def cell(name):
            mean,sd=r[name+'_mean_percent'],r[name+'_sd_percent']
            return f'${mean:.2f}$' if sd=='' else pm(dict(mean=mean,sd=sd))
        name={'unsafe_only':'Qwen3Guard (Unsafe)', 'unsafe_or_controversial':'Qwen3Guard (Unsafe + Controversial)',
              'validation_selected_threshold':'hanguard (CQA + MLF)'}[r['rule']]
        native_tex.append([name,*[cell(k) for k in ('accuracy','macro_f1','harmful_f1','harmful_recall','fpr')]])
    controls=[]
    for (task,arm), metrics in a.items():
        controls.append(dict(task=task,study='main_clean_study',arm=arm,
            accuracy_mean_percent=metrics['accuracy']['mean'],accuracy_sd_percent=metrics['accuracy']['sd'],
            f1_kind='harmful_F1' if task=='binary' else 'macro_F1',
            f1_mean_percent=metrics['f1' if task=='binary' else 'macro_f1']['mean'],
            f1_sd_percent=metrics['f1' if task=='binary' else 'macro_f1']['sd']))
    for r in e['pooling']['aggregate']:
        m=r['metrics'];f=m['f1' if r['task']=='binary' else 'macro_f1']
        controls.append(dict(task=r['task'],study='generic_pooling_followup',arm=r['arm'],
            accuracy_mean_percent=100*m['accuracy']['mean'],accuracy_sd_percent=100*m['accuracy']['sample_std'],
            f1_kind='harmful_F1' if r['task']=='binary' else 'macro_F1',
            f1_mean_percent=100*f['mean'],f1_sd_percent=100*f['sample_std']))
    tables={'main_ablation.csv':csv_text(main), 'main_ablation.tex':tex_table(
        ['Group','CQA','MLF','Binary Acc.','Harmful F1','5-class Acc.','Macro-F1'],main_tex,'lccrrrr'),
        'seed_results.csv':csv_text(seed_rows),'five_class_metrics.csv':csv_text(class_rows),
        'paired_deltas.csv':csv_text(delta),'paired_deltas.tex':tex_table(
            ['Contrast','Binary Acc. $\\Delta$ (pp)','5-class Acc. $\\Delta$ (pp)'],delta_tex,'lrr'),
        'native_guard_binary.csv':csv_text(native),'native_guard_binary.tex':tex_table(
            ['Model / native rule','Acc.','Macro-F1','Harmful F1','Harmful recall','FPR $\\downarrow$'],native_tex,'lrrrrr'),
        'supplementary_controls.csv':csv_text(controls)}
    return tables


def package(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();e=load_evidence();tables=build_tables(e)
    (output/'tables').mkdir(parents=True,exist_ok=True)
    for name,body in tables.items():(output/'tables'/name).write_text(body)
    copies={name:FOLLOWUP/'analysis'/name for name in ('paired_changes_by_seed.csv','paired_summary.csv','strata_summary.csv',
                                                      'efficiency.csv','parameters.csv')}
    (output/'supplementary').mkdir(exist_ok=True)
    for name,path in copies.items():shutil.copyfile(path,output/'supplementary'/name)
    source_map={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
    evidence=dict(schema=SCHEMA,scope='Paper tables and writing materials derived from completed runs; no new experiments',
        sources=source_map,method_names=dict(CQA='Category-Query Attention',MLF='Fixed Multi-Layer Residual Fusion'),
        raw_test_sha256=e['protocol']['data_sha256']['test'],binary_test_rows=8350,category_test_rows=2180,
        seeds=list(SEEDS),main_arms=list(CORE),native_baseline='Qwen3Guard-Gen-4B official generative interface',
        native_guard_local_five_class='N/A',encoder_guard_status='On hold at user request; no comparison result',
        metrics_units='Percent for metrics; percentage points for differences; standard deviations are sample SD across paired seeds')
    (output/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    readme='''# hanguard 论文实验材料（20261025）

主线为类别查询注意力聚合（CQA）＋固定多层残差融合（MLF），二分类与单标签五分类分别进行四组消融。

- [实验正文草稿](../../docs/paper_experiments_20261025.md)：方法命名、复现设置、四组主表与原生 Qwen3Guard 对照。
- [补充证据与写作边界](../../docs/paper_supporting_evidence_20261025.md)：参数匹配、选层、替代聚合、逐类分析与效率限制。
- [主消融 CSV](tables/main_ablation.csv)／[LaTeX](tables/main_ablation.tex)：二分类 F1 为有害类 F1，五分类采用 Macro-F1。
- [逐 seed 结果](tables/seed_results.csv)／[增益表](tables/paired_deltas.csv)／[增益 LaTeX](tables/paired_deltas.tex)。
- [原生 Qwen3Guard 二分类 CSV](tables/native_guard_binary.csv)／[LaTeX](tables/native_guard_binary.tex)：两种预先固定的原生安全标签合并规则同时保留；原生五分类 N/A。
- [五类 P/R/F1](tables/five_class_metrics.csv)、[全部机制对照](tables/supplementary_controls.csv)及 supplementary/ 中已有配对区间、来源/长度分层与在线效率记录。

表格均由固定结果中的三个 seed 重新计算；百分数与百分点分开。LaTeX 表使用 booktabs。
完整组合相对普通末位置 MLP 的准确率增益为二分类 +2.2675 pp、五分类 +5.8563 pp；这不是融合单模块的贡献。
组合相对仅 CQA 的平均增益为 +0.2555 pp、+0.9327 pp，存在种子差异。

此包只组织已完成证据，不重新训练、重新预测或修改数据。外部基线使用原始发布模型；额外监督适配结果留在原有归档，不混入原生比较。Encoder-Guard 对比保持搁置。
evidence.json 保存来源 SHA；manifest.json 校验本包与当前写作稿。指标是现有固定划分上的结果，不宣称整个项目从未查看测试集。

```bash
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py
.venv-hanguard/bin/python scripts/hanguard/package_paper_experiment_materials.py --verify
```
'''
    (output/'README.md').write_text(readme)
    documents=[ROOT/'docs/paper_experiments_20261025.md',ROOT/'docs/paper_supporting_evidence_20261025.md']
    if not all(p.exists() for p in documents):
        raise FileNotFoundError('Complete both writing documents before final packaging')
    manifest=dict(schema=SCHEMA,files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p!=output/'manifest.json'},documents={str(p.relative_to(ROOT)):sha(p) for p in documents})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return verify(output)


def verify(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();m=read(output/'manifest.json');evidence=read(output/'evidence.json')
    for relative,digest in m['files'].items():
        if sha(output/relative)!=digest:raise ValueError('Bundle file changed: '+relative)
    for relative,digest in {**m['documents'],**evidence['sources']}.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Source/document changed: '+relative)
    for name,expected in build_tables(load_evidence()).items():
        if (output/'tables'/name).read_text()!=expected:raise ValueError('Derived table mismatch: '+name)
    return dict(verified=True,files=len(m['files']),documents=len(m['documents']),source_files=len(evidence['sources']),
                groups=4,seeds=3,main_task_seed_rows=24,output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else package(args.output),ensure_ascii=False))


if __name__=='__main__':main()
