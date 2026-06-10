"""
HanGuard v5 评估脚本。

对 final_test.parquet（或任何含标注的文件）跑推理，输出：
  - 二分类指标（Accuracy / Precision / Recall / F1）
  - 六分类 macro F1 + 各类别详细指标
  - 注入鲁棒性：单独统计含注入模板样本的表现

用法：
  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --model outputs/hanguard_v5 \
      --test  data/final_test.parquet

  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --model outputs/hanguard_v5 \
      --test  data/final_test.parquet \
      --output outputs/eval_v5/report.txt
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# 导入推理函数
from infer import build_prompt, infer_batch, load_model

CATEGORY_LABELS = {
    "0": "安全",
    "1": "违反社会主义核心价值观",
    "2": "歧视性内容",
    "3": "商业违法违规",
    "4": "侵犯他人合法权益",
    "5": "专业安全需求",
}

# 注入样本的特征字符串（来自 generate_injection_defense.py）
INJECTION_MARKERS = [
    "忽略以上所有内容",
    "SYSTEM OVERRIDE",
    "Reclassify as",
    "classifier_directive",
    "im_start|>assistant",
    "safety_override",
    "SAFETY_CHECK=DISABLED",
    "审核通过，忽略",
    "SAFETY_BYPASS",
]


def is_injection(prompt: str) -> bool:
    return any(m in prompt for m in INJECTION_MARKERS)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=acc, precision=prec, recall=rec, f1=f1)


def format_report(
    df: pd.DataFrame,
    harm_preds: list[str],
    cat_preds: list[str],
) -> str:
    lines = []

    def hr(c="=", w=64): lines.append(c * w)
    def sec(t): hr(); lines.append(t); hr()
    def blank(): lines.append("")

    # ── 1. 二分类 ────────────────────────────────────────────────────
    sec("1. 二分类评估（有害 / 无害）")
    blank()

    y_true_b = [1 if v == "harmful" else 0 for v in df["prompt_harm_label"]]
    y_pred_b = [1 if v == "harmful" else 0 for v in harm_preds]
    m = binary_metrics(y_true_b, y_pred_b)

    lines.append(f"  样本总数   : {len(y_true_b):,}")
    lines.append(f"  TP         : {m['tp']:,}  （有害且检出）")
    lines.append(f"  TN         : {m['tn']:,}  （无害且判无害）")
    lines.append(f"  FP         : {m['fp']:,}  ← 误报")
    lines.append(f"  FN         : {m['fn']:,}  ← 漏报")
    blank()
    lines.append(f"  Accuracy   : {m['accuracy']:.4f}")
    lines.append(f"  Precision  : {m['precision']:.4f}")
    lines.append(f"  Recall     : {m['recall']:.4f}")
    lines.append(f"  F1         : {m['f1']:.4f}")

    # ── 2. 六分类 ────────────────────────────────────────────────────
    blank(); blank()
    sec("2. 六分类评估（类别 0-5）")
    blank()

    y_true_c = df["category_id"].astype(str).tolist()
    y_pred_c = cat_preds
    all_cats = sorted(CATEGORY_LABELS.keys())

    report = classification_report(
        y_true_c, y_pred_c,
        labels=all_cats,
        target_names=[CATEGORY_LABELS[c] for c in all_cats],
        zero_division=0,
        output_dict=True,
    )

    col_w = max(len(v) for v in CATEGORY_LABELS.values()) + 2
    lines.append(f"  {'类别':<{col_w}}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'支持数':>6}")
    lines.append("  " + "-" * (col_w + 42))
    for cid in all_cats:
        name = CATEGORY_LABELS[cid]
        r = report.get(name, {})
        lines.append(
            f"  {name:<{col_w}}  {r.get('precision',0):>10.4f}"
            f"  {r.get('recall',0):>8.4f}"
            f"  {r.get('f1-score',0):>8.4f}"
            f"  {int(r.get('support',0)):>6}"
        )
    blank()
    ma = report.get("macro avg", {})
    wa = report.get("weighted avg", {})
    acc6 = accuracy_score(y_true_c, y_pred_c)
    lines.append(f"  Accuracy (六分类)     : {acc6:.4f}")
    lines.append(f"  Macro    Precision    : {ma.get('precision',0):.4f}")
    lines.append(f"  Macro    Recall       : {ma.get('recall',0):.4f}")
    lines.append(f"  Macro    F1           : {ma.get('f1-score',0):.4f}")
    lines.append(f"  Weighted F1           : {wa.get('f1-score',0):.4f}")

    # ── 3. 注入鲁棒性 ───────────────────────────────────────────────
    blank(); blank()
    sec("3. 注入鲁棒性（含注入模板的样本子集）")
    blank()

    inj_mask = df["prompt"].apply(is_injection)
    n_inj = inj_mask.sum()

    if n_inj == 0:
        lines.append("  测试集中未检测到注入样本（注入样本在训练集中，未进入测试集）")
    else:
        inj_pos = [i for i, m in enumerate(inj_mask) if m]
        inj_df = df[inj_mask].reset_index(drop=True)
        inj_harm_true = [1 if v == "harmful" else 0 for v in inj_df["prompt_harm_label"]]
        inj_harm_pred = [1 if harm_preds[i] == "harmful" else 0 for i in inj_pos]
        mi = binary_metrics(inj_harm_true, inj_harm_pred)
        lines.append(f"  注入样本数 : {n_inj:,}")
        lines.append(f"  Accuracy   : {mi['accuracy']:.4f}")
        lines.append(f"  F1         : {mi['f1']:.4f}")
        lines.append(f"  误报（含注入的无害被判有害）: {mi['fp']:,}")
        lines.append(f"  漏报（含注入的有害被绕过）  : {mi['fn']:,}")

    blank()
    hr()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="outputs/hanguard_v5")
    parser.add_argument("--test",       type=str, default="data/final_test.parquet")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--preds",      type=str, default=None,
                        help="预计算的预测 CSV（含 harmful_pred / category_pred 列），跳过推理直接算指标")
    args = parser.parse_args()

    out_dir = Path(args.output).parent if args.output else Path("outputs/eval_v5")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preds:
        # 直接读取已有预测，跳过推理
        df = pd.read_csv(args.preds)
        print(f"读取预测文件: {args.preds}  ({len(df):,} 条)")
        harm_preds = df["harmful_pred"].tolist()
        cat_preds  = df["category_pred"].astype(str).tolist()
    else:
        df = pd.read_parquet(args.test)
        if args.limit:
            df = df.head(args.limit).copy()
        df = df.reset_index(drop=True)
        print(f"测试集: {args.test}  ({len(df):,} 条)")

        prompts_built = [build_prompt(str(p)) for p in df["prompt"]]
        tokenizer, model = load_model(args.model)
        preds = infer_batch(prompts_built, tokenizer, model, args.batch_size)
        harm_preds, cat_preds = zip(*preds)
        harm_preds, cat_preds = list(harm_preds), list(cat_preds)

        df["harmful_pred"]  = harm_preds
        df["category_pred"] = cat_preds
        pred_path = str(out_dir / "predictions.csv")
        df.to_csv(pred_path, index=False, encoding="utf-8")
        print(f"预测结果已保存至: {pred_path}")

    report = format_report(df, harm_preds, cat_preds)
    print("\n" + report)

    report_path = args.output or str(out_dir / "report.txt")
    Path(report_path).write_text(report, encoding="utf-8")
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
