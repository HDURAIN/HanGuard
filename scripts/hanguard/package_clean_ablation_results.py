#!/usr/bin/env python3
"""Package only the fresh 20261022 ablation study, without running training.

The bundle contains protocols, source hashes, selections, test metrics and
allowlisted numeric prediction tables. Corpus text, model weights, feature
arrays, and any historical experiment directory are never inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


DEFAULT_STUDY = Path('outputs/hanguard_clean_ablation_20261022')
DEFAULT_ARCHIVE = Path('experiment_results/20261022_clean_ablation')
TASKS = ('binary', 'category')
SEEDS = (42, 43, 44)
MAIN_ARMS = ('baseline', 'fusion_only', 'category_only', 'fusion_category')
ARMS = MAIN_ARMS + ('fusion_repeat_last', 'category_fusion_repeat_last',
                    'category_shared', 'fusion_category_shared', 'probe_selected_single')
SYSTEMS = ('corresponding_both_stages', 'classic_gate_type_factorial',
           'fusion_in_both_stages_factorial')
SCOPES = {'binary_full_test_8350': 8350, 'category_known_harmful_2180': 2180,
          'pipeline_known_labels_3704': 3704}
RAW_KEYS = {'text', 'prompt', 'prompt_en', 'prompt_zh', 'original_text', 'original_prompt',
            'translated_text', 'translated_prompt', 'translation', 'response', 'messages',
            'instruction', 'input_text', 'english_text', 'chinese_text', 'raw_text'}
SAFE_METADATA = {'base_id', 'sample_id', 'group_id', 'source', 'label', 'category_id',
                 'type_label_mask', 'split', 'prompt_tokens', 'seed', 'arm', 'method',
                 'task', 'evaluation_scope', 'target', 'true_label', 'y_true',
                 'binary_label', 'known_native_category', 'row_index', 'token_length'}
SAFE_NUMBER = re.compile(r'(?:(?:probability|probabilities|prediction|predicted|logit|score|threshold|target|label)(?:_[A-Za-z0-9_]+)?|L(?:[1-9]|[12][0-9]|3[0-2])_p[0-9]+|(?:' + '|'.join(ARMS) + r')_(?:p[0-9]+|logit[0-9]+|argmax|prediction))\Z')
METRIC_NAMES = ('accuracy', 'f1', 'macro_f1', 'precision', 'recall', 'auroc',
                'false_positive_rate', 'false_negative_rate')
SUMMARY_FIELDS = ('task', 'evaluation_scope', 'seed', 'arm', 'status', 'rows',
                  *(f'{name}_percent' for name in METRIC_NAMES), 'gain_vs_baseline_pp', 'source')
MEAN_FIELDS = ('task', 'evaluation_scope', 'arm', 'status', 'completed_seeds', 'planned_seeds',
               'mean_accuracy_percent', 'sample_std_accuracy_pp', 'mean_macro_f1_percent',
               'sample_std_macro_f1_pp')
SHA_PATTERN = re.compile(r'[0-9a-f]{64}\Z')


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


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sanitized_json(value: Any) -> tuple[Any, int]:
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


def safe_columns(columns) -> list[str]:
    return [str(name) for name in columns if str(name) in SAFE_METADATA or SAFE_NUMBER.fullmatch(str(name))]


def export_predictions(source: Path, target: Path) -> dict:
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
        raise ValueError('Only CSV and Parquet prediction tables can be exported')
    identity = 'base_id' if 'base_id' in columns else 'sample_id' if 'sample_id' in columns else None
    if identity is None:
        raise ValueError('Prediction table requires a sample identity')
    seen = set()
    for record in records:
        key = (str(record[identity]), str(record.get('arm', record.get('method', ''))))
        if key in seen:
            raise ValueError('Duplicate sample/method prediction')
        seen.add(key)
        for name in columns:
            if SAFE_NUMBER.fullmatch(name) and name not in SAFE_METADATA:
                value = record[name]
                if value is not None and str(value) != '' and not math.isfinite(float(value)):
                    raise ValueError('Nonfinite numeric prediction')
    write_csv(target, columns, records)
    return dict(transform='allowlisted_text_free_csv', rows=len(records), columns=columns)


def source_bindings(protocol: dict) -> list[tuple[str, str]]:
    """Recognize explicit registrations only; never traverse old imports/outputs."""
    sources = protocol.get('code_sha256', protocol.get('sources', protocol.get('source_files', protocol.get('source_sha256', []))))
    if isinstance(sources, dict):
        return [(str(path), str(value.get('sha256') if isinstance(value, dict) else value))
                for path, value in sources.items()]
    return [(str(item['path']), str(item['sha256'])) for item in sources if isinstance(item, dict)]


def frozen_source_report(root: Path, study: Path) -> dict:
    path = study / 'protocol.json'
    if not path.is_file():
        return dict(status='pending', passed=False, registered_sources=0, mismatches=[])
    bindings = source_bindings(read_json(path))
    mismatches = []
    registration = study / 'protocol_registration.json'
    if registration.is_file() and read_json(registration).get('protocol_sha256') != sha256(path):
        mismatches.append(dict(path=str(path), reason='protocol_registration_hash_differs'))
    for relative, digest in bindings:
        source = (root / relative).resolve()
        if root not in source.parents or source.suffix != '.py' or not SHA_PATTERN.fullmatch(digest):
            mismatches.append(dict(path=relative, reason='invalid_registration'))
        elif not source.is_file() or sha256(source) != digest:
            mismatches.append(dict(path=relative, reason='source_hash_differs'))
    return dict(status='passed' if bindings and not mismatches else 'failed' if mismatches else 'pending',
                passed=bool(bindings) and not mismatches, registered_sources=len(bindings), mismatches=mismatches)


def _metric_row(values: dict, task: str, scope: str, seed: int, arm: str, source: str, baseline=None) -> dict:
    rows = int(values.get('n', values.get('rows', SCOPES[scope])))
    if rows != SCOPES[scope]:
        raise ValueError(f'Unexpected denominator for {scope}: {rows}')
    values = dict(values)
    if 'fpr' in values:
        values.setdefault('false_positive_rate', values['fpr'])
    if 'recall' in values and task == 'binary':
        values.setdefault('false_negative_rate', 1. - float(values['recall']))
    result = dict(task=task, evaluation_scope=scope, seed=seed, arm=arm,
                  status='complete', rows=rows, source=source)
    for name in METRIC_NAMES:
        if values.get(name) is not None:
            number = float(values[name])
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f'{name} must be a finite fraction in [0,1]')
            result[f'{name}_percent'] = number * 100
    if 'accuracy_percent' not in result:
        raise ValueError('Completed metrics require accuracy')
    if baseline is not None:
        result['gain_vs_baseline_pp'] = result['accuracy_percent'] - float(baseline) * 100
    return result


def metric_rows(result: dict, task: str, seed: int, source: str) -> list[dict]:
    if (result.get('complete') is False or result.get('task') != task or result.get('seed') != seed
            or set(result.get('arms', {})) != set(ARMS)
            or any(not isinstance(result.get(key), str) or not SHA_PATTERN.fullmatch(result[key])
                   for key in ('predictions_sha256', 'rootselection_lock_sha256'))):
        return []
    scope = 'binary_full_test_8350' if task == 'binary' else 'category_known_harmful_2180'
    values = {}
    for arm in ARMS:
        entry = result['arms'][arm]
        block = entry.get('full_original_test', entry) if task == 'binary' else entry.get('native_known_harmful', entry)
        values[arm] = block.get('validation_threshold', block) if task == 'binary' else block
    baseline = values['baseline'].get('accuracy')
    return [_metric_row(values[arm], task, scope, seed, arm, source, baseline)
            for arm in ARMS if 'accuracy' in values[arm]]


def verified_test_result(study: Path, task: str, seed: int) -> dict:
    """Only the evaluator's final atomic receipt can promote pending scores."""
    folder = study / 'heads' / task / f's{seed}'
    receipt_path, result_path = folder / 'test_complete.json', folder / 'test_results.json'
    if not receipt_path.is_file() or not result_path.is_file():
        return {}
    receipt = read_json(receipt_path)
    if receipt.get('complete') is not True or receipt.get('task') != task or receipt.get('seed') != seed:
        return {}
    protocol_path, lock_path = study / 'protocol.json', study / 'selection_lock.json'
    if (not protocol_path.is_file() or not lock_path.is_file()
            or receipt.get('protocol_sha256') != sha256(protocol_path)
            or receipt.get('rootselection_lock_sha256') != sha256(lock_path)):
        raise ValueError('Test completion receipt does not match fresh protocol/selection lock')
    predictions = folder / 'test_predictions.parquet'
    expected = {str(path): sha256(path) for path in (result_path, predictions) if path.is_file()}
    if len(expected) != 2 or receipt.get('artifacts') != expected:
        raise ValueError('Completed test artifacts differ from atomic receipt')
    result = read_json(result_path)
    if (result.get('predictions_sha256') != expected[str(predictions)]
            or result.get('rootselection_lock_sha256') != sha256(lock_path)):
        raise ValueError('Completed test result identity differs')
    return result


def tables(study: Path, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows, coverage = [], []
    for task in TASKS:
        scope = 'binary_full_test_8350' if task == 'binary' else 'category_known_harmful_2180'
        for seed in SEEDS:
            path = study / 'heads' / task / f's{seed}' / 'test_results.json'
            source = str(path.relative_to(root))
            result = verified_test_result(study, task, seed)
            complete = {row['arm']: row for row in metric_rows(result, task, seed, source)}
            rows.extend(complete.get(arm, dict(task=task, evaluation_scope=scope, seed=seed,
                        arm=arm, status='pending', rows=SCOPES[scope], source=source)) for arm in ARMS)
            coverage.append(dict(task=task, seed=seed, status='complete' if len(complete) == len(ARMS) else 'pending', source=source))
    analysis_path = study / 'analysis' / 'results.json'
    if not analysis_path.is_file():
        analysis_path = study / 'results.json'
    analysis = read_json(analysis_path) if analysis_path.is_file() else {}
    for system in SYSTEMS:
        for seed in SEEDS:
            block = analysis.get('per_seed', {}).get(str(seed), {}).get('systems', {}).get(system, {}) if analysis.get('complete') is True else {}
            entries = block.get('arms', {})
            baseline = entries.get('baseline', {}).get('metrics', {}).get('accuracy')
            count = 0
            for arm in MAIN_ARMS:
                values = entries.get(arm, {}).get('metrics', {})
                common = dict(task=system, evaluation_scope='pipeline_known_labels_3704', seed=seed,
                              arm=arm, status='pending', rows=3704, source=str(analysis_path.relative_to(root)))
                if 'accuracy' in values:
                    rows.append(_metric_row(values, system, common['evaluation_scope'], seed, arm, common['source'], baseline))
                    count += 1
                else:
                    rows.append(common)
            coverage.append(dict(task=system, seed=seed, status='complete' if count == len(MAIN_ARMS) else 'pending', source=str(analysis_path.relative_to(root))))
    means = []
    for task, scope, arm in sorted({(row['task'], row['evaluation_scope'], row['arm']) for row in rows}):
        selected = [row for row in rows if (row['task'], row['evaluation_scope'], row['arm']) == (task, scope, arm) and row['status'] == 'complete']
        entry = dict(task=task, evaluation_scope=scope, arm=arm,
                     status='complete' if len(selected) == len(SEEDS) else 'pending',
                     completed_seeds='/'.join(str(row['seed']) for row in selected), planned_seeds='42/43/44')
        if len(selected) == len(SEEDS):
            for name in ('accuracy', 'macro_f1'):
                values = [row[f'{name}_percent'] for row in selected if f'{name}_percent' in row]
                if len(values) == len(SEEDS):
                    entry[f'mean_{name}_percent'] = statistics.mean(values)
                    entry[f'sample_std_{name}_pp'] = statistics.stdev(values)
        means.append(entry)
    return rows, means, coverage


def readme(index: dict, means: list[dict]) -> str:
    text = f'''# 全新缓存与训练的消融实验（20261022）

当前状态：**{index['status']}**。协议登记：**{index['registered']}**；注册源码核验：**{index['frozen_sources']['status']}**。

本轮只读取正式数据发布及原始 Qwen3.5-4B 基座，从原文重新分词，重新提取特征、训练、选层、选择检查点和阈值。历史实验的特征、预测、选择记录、权重、结果和测试排除名单均不作为输入；不沿用旧 122 条排除名单。

| 组别 | 方法 |
|---|---|
| `baseline` | L32 末 token MLP |
| `fusion_only` | 固定 L8/L16/L24/L32 各 0.25 的融合 |
| `category_only` | L32 全文 token 的类别查询读出 |
| `fusion_category` | 类别查询读出＋固定多层融合 |
| `fusion_repeat_last` | 融合结构保留，四路均输入 L32 |
| `category_fusion_repeat_last` | 组合结构保留，融合四路均输入 L32 |
| `category_shared` | 类别间共享 token 注意力分布 |
| `fusion_category_shared` | 共享读出＋固定多层融合 |
| `probe_selected_single` | 从新训练的 32 层线性 probe 通过验证集选层，再训练单层 MLP |

二分类、五分类各 9 组，seeds 42/43/44，共 **54 个正式分类头**；全部层 probe 共 **192 个**。骨干冻结，无 LoRA；全文 token 与各层句向量在同一次前向生成，所有组共用该轮新缓存。

测试范围分别为：二分类完整 **8,350** 条、原生已知有害类型 **2,180** 条、完整系统已知最终标签 **3,704** 条。两项监督任务与派生系统使用各自固定分母，未知类型不重新标注。测试在全体检查点及阈值由验证集锁定后开放。

完整系统报告三个组合：两阶段对应 A/A、B/B、C/C、D/D；经典组合 A/A、B/A、A/C、B/C；类型阶段也融合的原组合 A/A、B/B、A/C、B/D。每对字母分别表示二分类头／类型头。系统结果由同一批新预测派生，不增加训练。

三种子全部完成才显示均值及样本标准差；未完成项保留 `pending`。不将旧实验成绩填入本表。

'''
    for task in (*TASKS, *SYSTEMS):
        text += f'## {task}\n\n| 方法 | 状态 | Accuracy / %（均值 ± SD） | Macro-F1 / %（均值 ± SD） |\n|---|---|---:|---:|\n'
        for entry in (row for row in means if row['task'] == task):
            accuracy = f"{entry['mean_accuracy_percent']:.4f} ± {entry['sample_std_accuracy_pp']:.4f}" if 'mean_accuracy_percent' in entry else 'pending'
            macro = f"{entry['mean_macro_f1_percent']:.4f} ± {entry['sample_std_macro_f1_pp']:.4f}" if 'mean_macro_f1_percent' in entry else 'pending'
            text += f"| `{entry['arm']}` | {entry['status']} | {accuracy} | {macro} |\n"
        text += '\n'
    return text + '''逐种子结果见 [results_summary.csv](results_summary.csv)，三种子汇总见 [seed_summary.csv](seed_summary.csv)，完成状态见 [coverage.csv](coverage.csv)。[index.json](index.json) 保存运行状态快照和源码核验；[manifest.json](manifest.json) 记录每项源文件及归档 SHA256。

`artifacts/` 只保存新研究的协议、状态、小型选择和结果 JSON、注册 Python 源码以及无原文的数值预测 CSV。不包含原文、特征、模型权重和日志。

```bash
python scripts/hanguard/package_clean_ablation_results.py
python scripts/hanguard/package_clean_ablation_results.py --verify
```
'''


def refresh(root: Path, study: Path | None = None, archive: Path | None = None) -> dict:
    root = root.resolve()
    study = (root / DEFAULT_STUDY if study is None else study).resolve()
    archive = (root / DEFAULT_ARCHIVE if archive is None else archive).resolve()
    study.relative_to(root)
    archive.relative_to(root)
    if study == archive or study in archive.parents or archive in study.parents:
        raise ValueError('Study and evidence archive must be separate trees')
    # An archive may have a custom name, but the input study is deliberately
    # restricted to the newly registered clean rerun family.
    if not study.name.startswith('hanguard_clean_ablation_'):
        raise ValueError('Historical studies are not valid fresh-run inputs')
    source_report = frozen_source_report(root, study)
    if source_report['status'] == 'failed':
        raise ValueError(f'Registered scientific sources changed: {source_report["mismatches"]}')
    archive.mkdir(parents=True, exist_ok=True)
    previous = read_json(archive / 'manifest.json') if (archive / 'manifest.json').is_file() else {}
    records = []
    def record(source: Path):
        relative = source.relative_to(root)
        before = sha256(source)
        target = archive / 'artifacts' / relative
        if source.suffix == '.py':
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            metadata = dict(transform='copy')
        elif source.suffix == '.json':
            value, removed = sanitized_json(read_json(source))
            write_json(target, value)
            metadata = dict(transform='canonical_json' if not removed else 'sanitize_corpus_fields', removed_corpus_fields=removed)
        else:
            if source.suffix == '.parquet':
                target = target.with_name(target.name + '.csv')
            metadata = export_predictions(source, target)
        if sha256(source) != before:
            raise RuntimeError(f'Input changed during packaging: {relative}')
        records.append(dict(path=str(target.relative_to(archive)), source=str(relative),
                            source_sha256=before, sha256=sha256(target), **metadata))
    paths = set(study.glob('*.json'))
    paths.update((study / 'analysis').rglob('*.json'))
    paths.update((study / 'probes').glob('*.json'))
    for folder in ('heads', 'probes'):
        for task in TASKS:
            for seed in SEEDS:
                location = study / folder / task / f's{seed}'
                for pattern in ('selection*.json', 'test_results.json', 'test_complete.json', 'parameters.json'):
                    paths.update(location.glob(pattern))
                for pattern in ('*predictions*.parquet', '*predictions*.csv'):
                    for source in sorted(location.glob(pattern)):
                        record(source)
    for source in sorted(paths):
        if source.is_file() and source.stat().st_size <= 8 * 1024 * 1024:
            record(source)
    protocol = read_json(study / 'protocol.json') if (study / 'protocol.json').is_file() else {}
    # Copy precisely registered source paths and clean modules. Unlike static
    # import closure, this cannot pull in scripts describing old experiments.
    code = set(root.glob('scripts/hanguard/clean_ablation*.py'))
    code.update(root.glob('tests/test_clean_ablation*.py'))
    for name in ('scripts/hanguard/package_clean_ablation_results.py', 'tests/test_package_clean_ablation_results.py'):
        if (root / name).is_file():
            code.add(root / name)
    for relative, digest in source_bindings(protocol):
        source = (root / relative).resolve()
        if root in source.parents and source.is_file() and source.suffix == '.py':
            code.add(source)
    for source in sorted(code):
        record(source)
    rows, means, coverage = tables(study, root)
    complete = bool(coverage) and all(row['status'] == 'complete' for row in coverage)
    state = read_json(study / 'state.json') if (study / 'state.json').is_file() else {}
    index = dict(schema='hanguard_clean_ablation_evidence_v1', generated_at=datetime.now(timezone.utc).isoformat(),
                 study=str(study.relative_to(root)), registered=(study / 'protocol.json').is_file(),
                 status='complete' if complete else 'pending', complete=complete,
                 tasks=list(TASKS), seeds=list(SEEDS), arms=list(ARMS), systems=list(SYSTEMS),
                 formal_heads=54, linear_probes=192, scopes=SCOPES, coverage=coverage,
                 frozen_sources=source_report, queue_state=state,
                 historical_experiment_inputs_used=False, posthoc_test_exclusions=[],
                 excluded_payloads=['corpus_text', 'weights', 'feature_arrays', 'logs', 'runtime_process_files'])
    write_csv(archive / 'results_summary.csv', SUMMARY_FIELDS, rows)
    write_csv(archive / 'seed_summary.csv', MEAN_FIELDS, means)
    write_csv(archive / 'coverage.csv', ('task', 'seed', 'status', 'source'), coverage)
    write_json(archive / 'index.json', index)
    (archive / 'README.md').write_text(readme(index, means), encoding='utf-8')
    for name in ('README.md', 'results_summary.csv', 'seed_summary.csv', 'coverage.csv', 'index.json'):
        records.append(dict(path=name, source=None, source_sha256=None, sha256=sha256(archive / name), transform='generated'))
    current = {entry['path'] for entry in records}
    for entry in previous.get('files', []):
        if entry['path'] not in current:
            stale = (archive / entry['path']).resolve()
            if archive not in stale.parents:
                raise ValueError('Unsafe old archive manifest path')
            if stale.is_file():
                if sha256(stale) != entry['sha256']:
                    raise ValueError('Refuse to remove a modified stale archive artifact')
                stale.unlink()
    write_json(archive / 'manifest.json', dict(schema='hanguard_clean_ablation_manifest_v1', files=records))
    verify(archive)
    return index


def verify(archive: Path) -> dict:
    archive = archive.resolve()
    seen = set()
    source_records = {}
    for entry in read_json(archive / 'manifest.json')['files']:
        path = (archive / entry['path']).resolve()
        if archive not in path.parents or entry['path'] in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(entry['path'])
        if entry.get('source'):
            source_records[entry['source']] = entry
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'Archive hash differs: {entry["path"]}')
        if path.suffix in ('.pt', '.npy', '.npz', '.parquet', '.safetensors'):
            raise ValueError('Weights/features are forbidden in the evidence bundle')
        if entry.get('transform') == 'allowlisted_text_free_csv':
            with path.open(encoding='utf-8', newline='') as stream:
                columns = next(csv.reader(stream), [])
            if safe_columns(columns) != columns:
                raise ValueError('Unsafe prediction column')
    actual = {str(path.relative_to(archive)) for path in archive.rglob('*') if path.is_file()}
    if actual != seen | {'manifest.json'}:
        raise ValueError('Unlisted or missing archive file')
    index = read_json(archive / 'index.json')
    protocol_path = archive / 'artifacts' / index['study'] / 'protocol.json'
    if protocol_path.is_file():
        bindings = source_bindings(read_json(protocol_path))
        for registered_path, registered_digest in bindings:
            # Absolute registrations preserve the repository suffix in the
            # source manifest; portable bundles verify that suffix, not the
            # machine-specific repository location.
            matches = [entry for source, entry in source_records.items()
                       if source == registered_path or (Path(registered_path).is_absolute()
                          and Path(registered_path).as_posix().endswith('/' + source))]
            if len(matches) != 1 or matches[0]['sha256'] != registered_digest:
                raise ValueError(f'Archived registered source differs: {registered_path}')
        if bindings and (not index['frozen_sources']['passed'] or len(bindings) != index['frozen_sources']['registered_sources']):
            raise ValueError('Inconsistent frozen-source verification report')
    return dict(valid=True, files=len(seen), complete=index['complete'], status=index['status'],
                frozen_sources=index['frozen_sources']['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    study = None if args.study is None else (root / args.study).resolve()
    archive = (root / (args.archive or DEFAULT_ARCHIVE)).resolve()
    result = verify(archive) if args.verify else refresh(root, study, archive)
    print(json.dumps(result if args.verify else {key: result[key] for key in ('status', 'registered', 'complete', 'frozen_sources')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
