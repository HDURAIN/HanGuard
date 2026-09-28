"""hanguard 中文有害检测与主要风险分类：文本、交互及批量文件入口。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from hanguard_model import CATEGORY_LABELS, DEFAULT_MANIFEST, HanguardClassifier, format_completion


def add_model_arguments(parser):
    parser.add_argument('--model', type=Path, default=DEFAULT_MANIFEST,
                        help='hanguard model.json 清单或所在目录')
    parser.add_argument('--device', default='cuda:0', help='CUDA 设备，例如 cuda:0')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=8)


def load_model(model=DEFAULT_MANIFEST, *, device='cuda:0'):
    # Keep JSON stdout machine-readable even if the model loader prints status.
    with redirect_stdout(sys.stderr):
        return HanguardClassifier.from_manifest(model, device=device)


def infer_batch(prompts, model, batch_size=8):
    return model.predict(prompts, batch_size=batch_size)


def load_input(path):
    """Return records, preserving metadata; every row must contain prompt."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.parquet', '.csv'}:
        import pandas as pd
        frame = pd.read_parquet(path) if suffix == '.parquet' else pd.read_csv(path)
        rows = json.loads(frame.to_json(orient='records', force_ascii=False, date_format='iso'))
    elif suffix == '.jsonl':
        # JSONL uses physical newlines; U+2028/U+2029 are legal JSON string characters.
        with path.open('r', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    elif suffix == '.json':
        rows = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(rows, dict):
            rows = [rows]
    else:
        raise ValueError('输入格式须为 CSV、Parquet、JSON 或 JSONL')
    if not isinstance(rows, list):
        raise ValueError('JSON 输入须为对象或对象数组')
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'第 {index + 1} 条输入需要非空 prompt 字段')
    return rows


def records_with_predictions(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count differs from input rows')
    output = []
    for row, prediction in zip(rows, predictions):
        merged = dict(row)
        merged.update(harmful_pred=prediction['harmful_label'], category_pred=prediction['category_id'],
                      category_pred_label=prediction['category_label'], harmful_probability=prediction['harmful_probability'])
        merged.update({f'category_p_{category}': score for category, score in prediction['category_probabilities'].items()})
        output.append(merged)
    return output


def save_output(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {'.csv', '.parquet'}:
        import pandas as pd
        frame = pd.DataFrame(rows)
        if suffix == '.parquet':
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False, encoding='utf-8')
    elif suffix == '.json':
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    elif suffix == '.jsonl':
        path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    else:
        raise ValueError('输出格式须为 CSV、Parquet、JSON 或 JSONL')


def display(prediction, as_json=False):
    if as_json:
        print(json.dumps(prediction, ensure_ascii=False, allow_nan=False))
    else:
        print(format_completion(prediction['harmful_label'], prediction['category_id']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text', help='直接判断一条中文文本')
    inputs.add_argument('--interactive', action='store_true', help='每行输入一条文本，输入 /quit 退出')
    inputs.add_argument('--input', type=Path, help='含 prompt 字段的 CSV/Parquet/JSON/JSONL')
    add_model_arguments(parser)
    parser.add_argument('--json', action='store_true', help='以 JSON 输出单条或交互结果')
    parser.add_argument('--output', type=Path, help='批量结果路径；默认在输入旁生成 *_predictions.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.skip < 0 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size、limit 必须大于0；skip 不能小于0')
    if args.output is not None and args.input is None:
        parser.error('--output 仅用于批量文件')
    rows = None
    if args.input:
        rows = load_input(args.input)[args.skip:]
        if args.limit is not None:
            rows = rows[:args.limit]
        if not rows:
            parser.error('没有可推理的输入记录')
    with load_model(args.model, device=args.device) as model:
        if args.text is not None:
            display(model.classify(args.text), args.json)
        elif args.interactive:
            print('hanguard 已就绪；每行一条文本，/quit 退出。', file=sys.stderr)
            while True:
                try:
                    if sys.stdin.isatty():
                        print('文本> ', end='', flush=True, file=sys.stderr)
                    text = sys.stdin.readline()
                    if not text or text.strip() == '/quit':
                        break
                    if text.strip():
                        try:
                            display(model.classify(text.rstrip('\n')), args.json)
                        except ValueError as error:
                            print(str(error), file=sys.stderr)
                except KeyboardInterrupt:
                    break
        else:
            predictions = model.predict([row['prompt'] for row in rows], batch_size=args.batch_size)
            merged = records_with_predictions(rows, predictions)
            output = args.output or args.input.with_name(args.input.stem + '_predictions.jsonl')
            if output.resolve() == args.input.resolve():
                raise ValueError('输出路径不能覆盖输入文件')
            save_output(output, merged)
            print(f'已处理 {len(rows)} 条，结果：{output}', file=sys.stderr)
            if args.json:
                print(json.dumps(dict(rows=len(rows), output=str(output)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f'错误：{error}', file=sys.stderr)
        raise SystemExit(1) from None
