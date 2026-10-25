#!/usr/bin/env python3
"""Refresh/verify the small four-arm evidence bundle without importing training.

Corpus text, weights and feature arrays are excluded. Prediction tables are
converted to allowlisted, text-free CSV. Pending evaluations never become
scores, and old selected-subset scores are not inputs to this packager.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


DEFAULT_STUDY = Path('outputs/hanguard_four_arm_experiments_20261019')
DEFAULT_ARCHIVE = Path('experiment_results/20261019_four_arm_experiments')
TASKS = ('binary', 'category')
SEEDS = (42, 43, 44)
ARMS = ('baseline', 'fusion_only', 'category_only', 'fusion_category')
SCOPE_ROWS = {'binary_full_test_8350': 8350, 'binary_fixed_subset_8228': 8228,
              'category_known_harmful_2180': 2180, 'pipeline_known_labels_3704': 3704}
SAFE_METADATA = {'base_id', 'sample_id', 'group_id', 'source', 'label', 'category_id',
                 'type_label_mask', 'split', 'prompt_tokens', 'seed', 'arm', 'method',
                 'task', 'evaluation_scope', 'target', 'true_label', 'y_true',
                 'binary_label', 'known_native_category'}
SAFE_NUMBER = re.compile(r'(?:(?:probability|probabilities|prediction|predicted|logit|score|threshold|target|label)(?:_[A-Za-z0-9_]+)?|(?:baseline|fusion_only|category_only|fusion_category)_(?:p[0-9]+|logit[0-9]+|argmax))\Z')
RAW_KEYS = {'text', 'prompt', 'prompt_en', 'prompt_zh', 'original_text', 'original_prompt',
            'translated_text', 'translated_prompt', 'translation', 'response', 'messages',
            'instruction', 'input_text', 'english_text', 'chinese_text', 'raw_text'}
SUMMARY_FIELDS = ('task', 'evaluation_scope', 'seed', 'arm', 'status', 'rows',
                  'accuracy_percent', 'f1_percent', 'macro_f1_percent', 'precision_percent',
                  'recall_percent', 'gain_vs_baseline_pp', 'source')
MEAN_FIELDS = ('task', 'evaluation_scope', 'arm', 'status', 'completed_seeds', 'planned_seeds',
               'mean_accuracy_percent', 'sample_std_accuracy_pp', 'mean_macro_f1_percent',
               'sample_std_macro_f1_pp')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_columns(columns) -> list[str]:
    return [str(name) for name in columns
            if str(name) in SAFE_METADATA or SAFE_NUMBER.fullmatch(str(name))]


def sanitized_json(value: Any) -> tuple[Any, int]:
    """Defensively remove corpus-bearing fields, including unforeseen nesting."""
    removed = 0
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if str(key).lower() in RAW_KEYS:
                removed += 1
                continue
            result[key], count = sanitized_json(child)
            removed += count
        return result, removed
    if isinstance(value, list):
        result = []
        for child in value:
            item, count = sanitized_json(child)
            result.append(item)
            removed += count
        return result, removed
    return value, 0


def export_predictions(source: Path, target: Path) -> dict:
    """Only IDs/labels/scalar numeric predictions leave an input prediction file."""
    if source.suffix == '.parquet':
        import pandas as pd
        frame = pd.read_parquet(source)
        columns = safe_columns(frame.columns)
        records = frame[columns].to_dict(orient='records')
    elif source.suffix == '.csv':
        with source.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = safe_columns(reader.fieldnames or [])
            records = [{name: row[name] for name in columns} for row in reader]
    else:
        raise ValueError(f'Unsupported prediction table: {source}')
    if not {'base_id', 'sample_id'} & set(columns):
        raise ValueError(f'Prediction table has no allowed sample identity: {source}')
    seen = set()
    identity_key = 'base_id' if 'base_id' in columns else 'sample_id'
    for row in records:
        identity = str(row[identity_key])
        # An arm-column table may repeat IDs once per arm.
        key = (identity, str(row.get('arm', row.get('method', ''))))
        if key in seen:
            raise ValueError(f'Duplicate sample/method prediction in {source}: {key}')
        seen.add(key)
        for name in columns:
            if SAFE_NUMBER.fullmatch(name) and name not in SAFE_METADATA:
                value = row[name]
                if value is None or str(value) == '':
                    continue
                if not math.isfinite(float(value)):
                    raise ValueError(f'Nonfinite numeric prediction in {source}: {name}')
    write_csv(target, columns, records)
    return dict(transform='allowlisted_text_free_csv', rows=len(records), columns=columns)


def source_closure(root: Path, seeds) -> list[Path]:
    """Resolve local Python imports statically, never importing torch/training."""
    pending, seen = list(seeds), set()
    while pending:
        path = pending.pop().resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if (path in seen or not path.is_file() or path.suffix != '.py' or
                path.stat().st_size > 1024 * 1024 or relative.parts[0] not in ('scripts', 'tests', 'hanguard')):
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                base = node.module or ''
                modules = [base] + [f'{base}.{alias.name}' for alias in node.names]
            else:
                continue
            for module in modules:
                if module and all(part.isidentifier() for part in module.split('.')):
                    candidate = root.joinpath(*module.split('.'))
                    pending += [candidate.with_suffix('.py'), candidate / '__init__.py']
    return sorted(seen)


def _evaluation_complete(result: dict, task: str, seed: int) -> bool:
    # Per-task evaluator writes this atomic receipt only after prediction and
    # metric completion; the scientific schema deliberately has no complete flag.
    return (result.get('complete') is not False and result.get('task') == task
            and result.get('seed') == seed and set(result.get('arms', {})) == set(ARMS)
            and all(isinstance(result.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', result[key])
                    for key in ('predictions_sha256', 'rootselection_lock_sha256')))


def _metric_row(values: dict, task: str, scope: str, seed: int, arm: str, source: str,
                baseline=None) -> dict:
    count = int(values.get('n', values.get('rows', SCOPE_ROWS[scope])))
    if count != SCOPE_ROWS[scope]:
        raise ValueError(f'Test denominator changed for {scope}: {count}')
    row = dict(task=task, evaluation_scope=scope, seed=seed, arm=arm,
               status='complete', rows=count, source=source)
    for name in ('accuracy', 'f1', 'macro_f1', 'precision', 'recall'):
        if values.get(name) is not None:
            value = float(values[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite fraction in [0,1]')
            row[f'{name}_percent'] = value * 100
    if 'accuracy_percent' not in row:
        raise ValueError('Completed test metrics require accuracy')
    if baseline is not None:
        row['gain_vs_baseline_pp'] = row['accuracy_percent'] - float(baseline) * 100
    return row


def metric_rows(result: dict, task: str, seed: int, source: str) -> list[dict]:
    """Read the actual evaluator receipt; never read checkpoint/dev scores."""
    if not _evaluation_complete(result, task, seed):
        return []
    scope_specs = (('full_original_test', 'binary_full_test_8350'),
                   ('fixed_selected122_subset', 'binary_fixed_subset_8228')) if task == 'binary' else (
                   ('native_known_harmful', 'category_known_harmful_2180'),)
    output = []
    for original, scope in scope_specs:
        metrics = {}
        for arm in ARMS:
            values = result['arms'][arm].get(original, {})
            metrics[arm] = values.get('validation_threshold', {}) if task == 'binary' else values
        baseline = metrics['baseline'].get('accuracy')
        for arm, values in metrics.items():
            if 'accuracy' in values:
                output.append(_metric_row(values, task, scope, seed, arm, source, baseline))
    return output


def tables(study: Path, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows, coverage = [], []
    for task in TASKS:
        scopes = ('binary_full_test_8350', 'binary_fixed_subset_8228') if task == 'binary' else ('category_known_harmful_2180',)
        for seed in SEEDS:
            path = study / 'heads' / task / f's{seed}' / 'test_results.json'
            source = str(path.relative_to(root))
            result = read_json(path) if path.is_file() else {}
            completed = metric_rows(result, task, seed, source)
            lookup = {(row['evaluation_scope'], row['arm']): row for row in completed}
            for scope in scopes:
                for arm in ARMS:
                    rows.append(lookup.get((scope, arm), dict(task=task, evaluation_scope=scope, seed=seed,
                                arm=arm, status='pending', rows=SCOPE_ROWS[scope], source=source)))
            coverage.append(dict(task=task, seed=seed, status='complete' if len(completed) == len(scopes) * len(ARMS) else 'pending',
                                 source=source))
    # Full-system predictions are derived and bootstrapped by the scientific
    # summarizer after all head selections, rather than by this packager.
    analysis_path = study / 'analysis' / 'results.json'
    analysis = read_json(analysis_path) if analysis_path.is_file() else {}
    analysis_source = str(analysis_path.relative_to(root))
    for system in ('classic_gate_type_factorial', 'fusion_in_both_stages_factorial'):
        for seed in SEEDS:
            block = analysis.get('per_seed', {}).get(str(seed), {}).get('systems', {}).get(system, {}) if analysis.get('complete') is True else {}
            entries = block.get('arms', {})
            baseline = entries.get('baseline', {}).get('metrics', {}).get('accuracy')
            found = 0
            for arm in ARMS:
                values = entries.get(arm, {}).get('metrics', {})
                if 'accuracy' in values:
                    rows.append(_metric_row(values, system, 'pipeline_known_labels_3704', seed, arm, analysis_source, baseline))
                    found += 1
                else:
                    rows.append(dict(task=system, evaluation_scope='pipeline_known_labels_3704', seed=seed,
                                     arm=arm, status='pending', rows=3704, source=analysis_source))
            coverage.append(dict(task=system, seed=seed, status='complete' if found == len(ARMS) else 'pending', source=analysis_source))
    means = []
    keys = sorted({(row['task'], row['evaluation_scope'], row['arm']) for row in rows})
    for task, scope, arm in keys:
        complete = [row for row in rows if (row['task'], row['evaluation_scope'], row['arm']) == (task, scope, arm)
                    and row['status'] == 'complete']
        entry = dict(task=task, evaluation_scope=scope, arm=arm,
                     status='complete' if len(complete) == len(SEEDS) else 'pending',
                     completed_seeds='/'.join(str(row['seed']) for row in complete), planned_seeds='42/43/44')
        if len(complete) == len(SEEDS):
            for metric in ('accuracy', 'macro_f1'):
                values = [row[f'{metric}_percent'] for row in complete if f'{metric}_percent' in row]
                if len(values) == len(SEEDS):
                    entry[f'mean_{metric}_percent'] = statistics.mean(values)
                    entry[f'sample_std_{metric}_pp'] = statistics.stdev(values)
        means.append(entry)
    return rows, means, coverage


def readme(index: dict, means: list[dict]) -> str:
    text = '''# 固定权重融合与类别向量四组实验（20261019）

当前状态：**{status}**。这是独立登记的无 LoRA 四臂实验，未知结果保留 `pending`。

| 组别 | 固定四层等权融合 | 类别向量 |
|---|---|---|
| `baseline` | × | × |
| `fusion_only` | ✓ | × |
| `category_only` | × | ✓ |
| `fusion_category` | ✓ | ✓ |

骨干冻结为原始 Qwen3.5-4B；层为 L8/L16/L24/L32，每层权重固定 0.25。类别向量聚合 L32 的全部有效 token。二分类与五分类均独立运行上述四组，seeds 为 42/43/44，检查点、阈值只由验证集选择。

三个指标范围分开报告：

- **直接二分类**：三来源，训练/验证/测试为 66,802/8,350/8,350；同时列出沿用既有 122 个排除 ID 的固定 8,228 条子集。两组范围内所有方法样本一致。
- **五类主要风险识别**：两来源中既有标签的有害样本，17,440/2,180/2,180；不把未知类型当负例，不重新标注。
- **完整系统六类输出**：在有已知最终标签的 3,704 条测试样本上，比较二分类 gate 与五类类型头组合。经典 2×2 为 gate `baseline/fusion_only` × type `baseline/category_only`；其余类型融合组合单列补充表。这里的六类准确率不能与前两项混用分母。

旧 **+2.1026 个百分点**来自另一个有可学习层权重和最佳单层锚点的旧模型在事后选定子集上的固定预测；它不属于这次固定等权四组结果，不填入新主表。旧结果及已完成的细组件/seed 消融保留在[旧证据包](../20261019_fusion_ablation/README.md)。

## 三种子主表

只有全部三个种子完成，才显示均值 ± 样本标准差；单一种子完成不会被写成三种子结果。

'''.format(status=index['status'])
    for task, title in (('binary', '直接二分类'), ('category', '五类主要风险识别'),
                        ('classic_gate_type_factorial', '完整系统经典 2×2'),
                        ('fusion_in_both_stages_factorial', '完整系统：类型阶段也加入融合')):
        text += f"\n### {title}\n\n| 测试范围 | 方法 | 状态 | 准确率均值 ± SD / % | Macro-F1 均值 ± SD / % |\n|---|---|---|---:|---:|\n"
        for item in (row for row in means if row['task'] == task):
            accuracy = (f"{item['mean_accuracy_percent']:.4f} ± {item['sample_std_accuracy_pp']:.4f}"
                        if 'mean_accuracy_percent' in item else 'pending')
            macro = (f"{item['mean_macro_f1_percent']:.4f} ± {item['sample_std_macro_f1_pp']:.4f}"
                     if 'mean_macro_f1_percent' in item else ('—' if task == 'binary' else 'pending'))
            text += f"| {item['evaluation_scope']} | `{item['arm']}` | {item['status']} | {accuracy} | {macro} |\n"
    text += '''
逐种子数值见 [results_summary.csv](results_summary.csv)，均值见 [seed_summary.csv](seed_summary.csv)，任务完成状态见 [coverage.csv](coverage.csv)。[index.json](index.json)记录快照与范围，[manifest.json](manifest.json)记录文件及源文件 SHA256。

`artifacts/` 按原仓库相对路径保存小协议、选择记录、指标及 Python 源码。逐样本预测只输出 ID、标签、来源、长度和数值预测，不含中文/英文原文；权重、特征、日志与进程临时文件不打包。该包不能替代完整基座、数据和特征缓存。

刷新与独立校验（不启动训练）：

```bash
python scripts/hanguard/package_four_arm_experiments.py
python scripts/hanguard/package_four_arm_experiments.py --verify
```
'''
    return text


def refresh(root: Path, archive: Path | None = None, study: Path | None = None) -> dict:
    root = root.resolve()
    archive = (root / DEFAULT_ARCHIVE if archive is None else archive).resolve()
    study = (root / DEFAULT_STUDY if study is None else study).resolve()
    # Paths are intentionally repository-local and never point at an original
    # experiment directory: refresh only owns a separate evidence directory.
    archive.relative_to(root)
    study.relative_to(root)
    if archive == study or study in archive.parents or archive in study.parents:
        raise ValueError('Archive and study must be separate directory trees')
    archive.mkdir(parents=True, exist_ok=True)
    previous_manifest = read_json(archive / 'manifest.json') if (archive / 'manifest.json').is_file() else {}
    records = []

    def record_file(source: Path, transform: str = 'copy'):
        relative = source.relative_to(root)
        destination = archive / 'artifacts' / relative
        before = sha256(source)
        metadata = {}
        if source.suffix == '.py':
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        elif source.suffix == '.json':
            value, removed = sanitized_json(read_json(source))
            write_json(destination, value)
            metadata = dict(transform='sanitize_corpus_fields' if removed else 'canonical_json', removed_corpus_fields=removed)
        else:
            destination = destination.with_name(destination.name + '.csv') if source.suffix == '.parquet' else destination
            metadata = export_predictions(source, destination)
        if sha256(source) != before:
            raise RuntimeError(f'Source changed during packaging: {relative}')
        records.append(dict(path=str(destination.relative_to(archive)), sha256=sha256(destination),
                            source=str(relative), source_sha256=before, **metadata,
                            **({'transform': transform} if source.suffix == '.py' else {})))

    json_sources = list(study.glob('*.json')) if study.is_dir() else []
    for task in (*TASKS, 'pipeline'):
        for seed in SEEDS:
            head = study / 'heads' / task / f's{seed}'
            analysis = study / 'analysis' / task / f's{seed}'
            json_sources.extend(head.glob('selection*.json'))
            if (head / 'test_results.json').is_file():
                json_sources.append(head / 'test_results.json')
            json_sources.extend(analysis.glob('*.json'))
            for folder in (head, analysis):
                for pattern in ('*predictions*.csv', '*predictions*.parquet'):
                    for source in sorted(folder.glob(pattern)):
                        record_file(source)
    json_sources += list((study / 'analysis').glob('*.json'))
    for source in sorted(set(json_sources)):
        if source.stat().st_size <= 8 * 1024 * 1024:
            record_file(source)
    code_seeds = list((root / 'scripts' / 'hanguard').glob('four_arm_*.py'))
    code_seeds += list((root / 'tests').glob('test_four_arm*.py'))
    own = root / 'scripts' / 'hanguard' / 'package_four_arm_experiments.py'
    if own.is_file():
        code_seeds.append(own)
    own_test = root / 'tests' / 'test_package_four_arm_experiments.py'
    if own_test.is_file():
        code_seeds.append(own_test)
    for source in source_closure(root, code_seeds):
        record_file(source)
    rows, means, coverage = tables(study, root)
    complete = all(item['status'] == 'complete' for item in coverage)
    index = dict(schema='hanguard_four_arm_evidence_v1', generated_at=datetime.now(timezone.utc).isoformat(),
                 study=str(study.relative_to(root)), status='complete' if complete else 'pending',
                 registered=(study / 'protocol.json').is_file(), tasks=list(TASKS), seeds=list(SEEDS), arms=list(ARMS),
                 scopes=SCOPE_ROWS, complete=complete, coverage=coverage,
                 old_selected_subset_gain_is_not_a_new_result=True,
                 excluded_payloads=['corpus_text', 'weights', 'feature_arrays', 'logs', 'runtime_process_files'])
    write_csv(archive / 'results_summary.csv', SUMMARY_FIELDS, rows)
    write_csv(archive / 'seed_summary.csv', MEAN_FIELDS, means)
    write_csv(archive / 'coverage.csv', ('task', 'seed', 'status', 'source'), coverage)
    write_json(archive / 'index.json', index)
    (archive / 'README.md').write_text(readme(index, means), encoding='utf-8')
    for name in ('README.md', 'results_summary.csv', 'seed_summary.csv', 'coverage.csv', 'index.json'):
        records.append(dict(path=name, sha256=sha256(archive / name), source=None, source_sha256=None, transform='generated'))
    current_paths = {record['path'] for record in records}
    for old in previous_manifest.get('files', []):
        if old['path'] not in current_paths:
            stale = (archive / old['path']).resolve()
            if archive not in stale.parents:
                raise ValueError('Unsafe old archive path')
            if stale.is_file():
                if sha256(stale) != old['sha256']:
                    raise ValueError('Refuse to remove modified stale archive artifact')
                stale.unlink()
    write_json(archive / 'manifest.json', dict(schema='hanguard_four_arm_manifest_v1', files=records))
    verify(archive)
    return index


def verify(archive: Path) -> dict:
    archive = archive.resolve()
    manifest = read_json(archive / 'manifest.json')
    seen = set()
    for record in manifest['files']:
        target = (archive / record['path']).resolve()
        if archive not in target.parents or record['path'] in seen:
            raise ValueError('Invalid/duplicate manifest path')
        seen.add(record['path'])
        if not target.is_file() or sha256(target) != record['sha256']:
            raise ValueError(f'Archive hash differs: {record["path"]}')
        if target.suffix in ('.pt', '.npy', '.npz', '.parquet', '.safetensors'):
            raise ValueError('Forbidden binary artifact in evidence bundle')
        if record.get('transform') == 'allowlisted_text_free_csv':
            with target.open(encoding='utf-8', newline='') as stream:
                columns = next(csv.reader(stream), [])
            if columns != safe_columns(columns):
                raise ValueError('Unsafe prediction column in evidence bundle')
    actual = {str(path.relative_to(archive)) for path in archive.rglob('*') if path.is_file()}
    if actual != seen | {'manifest.json'}:
        raise ValueError('Archive contains unlisted or missing files')
    index = read_json(archive / 'index.json')
    return dict(valid=True, files=len(seen), complete=index['complete'], status=index['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    archive = root / DEFAULT_ARCHIVE if args.archive is None else (root / args.archive if not args.archive.is_absolute() else args.archive)
    study = None if args.study is None else (root / args.study if not args.study.is_absolute() else args.study)
    result = verify(archive) if args.verify else refresh(root, archive, study)
    print(json.dumps(result if args.verify else {key: result[key] for key in ('status', 'registered', 'complete')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
