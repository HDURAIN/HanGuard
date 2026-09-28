"""评估 hanguard：二分类覆盖全部输入；正式类型指标限两来源已知类别。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from hanguard_model import CATEGORY_LABELS
from infer import add_model_arguments, load_input, load_model, records_with_predictions, save_output

CATEGORY_SOURCES = ('chinese_curated', 'jailbench')


def classification_metrics(truth, predictions, classes):
    if len(truth) != len(predictions):
        raise ValueError('Labels and predictions must have the same length')
    labels = list(classes)
    position = {label: index for index, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for target, predicted in zip(truth, predictions):
        if target not in position or predicted not in position:
            raise ValueError('Unknown label in evaluation')
        matrix[position[target]][position[predicted]] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        per_class.append(dict(category_id=label, support=support,
            precision=tp / predicted if predicted else 0., recall=tp / support if support else 0.,
            f1=2 * tp / (support + predicted) if support + predicted else 0.))
    return dict(rows=len(truth), accuracy=sum(matrix[i][i] for i in range(len(labels))) / len(truth) if truth else None,
                macro_f1=sum(row['f1'] for row in per_class) / len(labels) if truth else None,
                per_class=per_class, confusion_matrix=matrix, classes=labels)


def _category(value):
    if isinstance(value, bool):
        raise ValueError('Category must be an integer in 0..5')
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError('Category must be an integer in 0..5') from None
    if not math.isfinite(number) or number not in range(6):
        raise ValueError('Category must be an integer in 0..5')
    return int(number)


def metrics_for_records(rows):
    if not rows:
        raise ValueError('评估集不能为空')
    source_fields = ['source' in row for row in rows]
    if any(source_fields) and not all(source_fields):
        raise ValueError('source 字段须在全部记录中一致提供，或全部省略')
    has_sources = all(source_fields)
    if has_sources and any(not isinstance(row['source'], str) or not row['source'].strip() for row in rows):
        raise ValueError('source 必须是非空来源名称；无来源的通用数据请全部省略该字段')
    covered = [not has_sources or row['source'] in CATEGORY_SOURCES for row in rows]
    truth, gated, binary_truth, binary_pred, type_pred = [], [], [], [], []
    for index, row in enumerate(rows):
        # Untrusted/out-of-scope source type mappings cannot affect even binary evaluation.
        category = _category(row['category_id']) if covered[index] else -1
        prediction = _category(row['category_pred'])
        harm, output = row['prompt_harm_label'], row['harmful_pred']
        if harm not in ('harmful', 'unharmful') or output not in ('harmful', 'unharmful'):
            raise ValueError('评估需要明确 harmful/unharmful 标签')
        if ((covered[index] and (harm == 'harmful') != (category > 0)) or
                (output == 'harmful') != (prediction > 0)):
            raise ValueError('有害性与类别不一致')
        scores = [float(row[f'category_p_{index}']) for index in range(1, 6)]
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores) or not math.isclose(sum(scores), 1., abs_tol=1e-5):
            raise ValueError('类型评估需要完整的五类 softmax 分数')
        raw = max(range(5), key=scores.__getitem__) + 1
        if prediction > 0 and prediction != raw:
            raise ValueError('类别预测必须为五类分数的 argmax')
        truth.append(category)
        gated.append(prediction)
        binary_truth.append(int(harm == 'harmful'))
        binary_pred.append(int(output == 'harmful'))
        type_pred.append(raw)
    category_rows = [index for index, include in enumerate(covered) if include]
    harmful = [index for index in category_rows if truth[index] > 0]
    excluded_sources = {}
    for index, row in enumerate(rows):
        if not covered[index]:
            excluded_sources[row['source']] = excluded_sources.get(row['source'], 0) + 1
    scope = dict(mode='registered_two_source' if has_sources else 'user_provided_labels_without_source',
                 allowed_sources=list(CATEGORY_SOURCES) if has_sources else None,
                 total_rows=len(rows), category_covered_rows=len(category_rows), type_covered_rows=len(harmful),
                 excluded_category_rows=len(rows) - len(category_rows),
                 excluded_harmful_rows=sum(binary_truth[i] for i in range(len(rows)) if not covered[i]),
                 excluded_source_counts=excluded_sources,
                 description=('仅 chinese_curated/jailbench 纳入正式五类及门控六类指标；二分类保留全部来源。'
                              if has_sources else '输入无 source 字段，类型评估使用用户提供的类别标签；不视为注册两来源基准。'))
    binary = classification_metrics(binary_truth, binary_pred, [0, 1])
    positive = binary['per_class'][1]
    binary.update(precision=positive['precision'], recall=positive['recall'], f1=positive['f1'])
    return dict(binary=binary,
                primary_category=classification_metrics([truth[i] for i in harmful], [type_pred[i] for i in harmful], range(1, 6)),
                end_to_end=classification_metrics([truth[i] for i in category_rows], [gated[i] for i in category_rows], range(6)),
                category_scope=scope)


def evaluate_records(rows):
    result = dict(**metrics_for_records(rows), category_labels=CATEGORY_LABELS,
        metric_definitions={'binary': 'All rows, harmful is the positive class',
            'primary_category': 'True harmful rows within category_scope only; five-way argmax before binary gating',
            'end_to_end': 'All category-covered rows including safe inputs; binary negative maps to category0'},
        label_quality='Metrics reflect the provided labels; no automatic human-gold claim')
    if rows and all('source' in row for row in rows):
        result['by_source'] = {source: metrics_for_records([row for row in rows if row['source'] == source])
                               for source in sorted(set(row['source'] for row in rows))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--test', type=Path, help='含 prompt、prompt_harm_label、category_id 的评估文件')
    data.add_argument('--preds', type=Path, help='infer.py 导出的含真实标签和五类分数的预测文件')
    add_model_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation/report.json'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size、limit 必须为正')
    rows = load_input(args.preds or args.test)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error('没有可评估的记录')
    if args.test:
        with load_model(args.model, device=args.device) as model:
            predictions = model.predict([row['prompt'] for row in rows], batch_size=args.batch_size)
        rows = records_with_predictions(rows, predictions)
        prediction_path = args.output.with_name(args.output.stem + '_predictions.jsonl')
        if prediction_path.resolve() == args.test.resolve():
            raise ValueError('预测输出不能覆盖评估输入')
        save_output(prediction_path, rows)
    if args.output.resolve() == (args.preds or args.test).resolve():
        raise ValueError('报告输出不能覆盖评估输入')
    report = evaluate_records(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(report['category_scope']['description'])
    if report['category_scope']['excluded_category_rows']:
        print(f"类型指标排除 {report['category_scope']['excluded_category_rows']} 条范围外来源记录；其二分类结果仍保留。")
    for name, title in [('binary', '有害二分类'), ('primary_category', '有害样本五分类'), ('end_to_end', '门控后六分类')]:
        metric = report[name]
        if metric['rows']:
            print(f"{title}：{metric['rows']} 条，准确率 {metric['accuracy']:.2%}，Macro-F1 {metric['macro_f1']:.2%}")
        else:
            print(f'{title}：无可评估样本')
    print(f'报告：{args.output}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f'评估失败：{error}', file=sys.stderr)
        raise SystemExit(1) from None
