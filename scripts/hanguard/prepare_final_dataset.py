"""
整合所有数据集，生成最终训练/验证/测试集。

步骤：
  1. 修复 injection_defense 的缺失 category_id
  2. 合并全部5个数据集（打 source 标签）
  3. 按 source × category 双重分层分割（自然数据80/10/10，增强数据全进train）
  4. 类别5过采样至 ~8000 条
  5. 计算并保存类别权重（供训练使用）

输出：
  data/final_train.parquet
  data/final_val.parquet
  data/final_test.parquet
  data/class_weights.json

用法：
  python scripts/hanguard/prepare_final_dataset.py
  python scripts/hanguard/prepare_final_dataset.py --seed 42
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    "0": "安全",
    "1": "违反社会主义核心价值观的内容",
    "2": "歧视性内容",
    "3": "商业违法违规",
    "4": "侵犯他人合法权益",
    "5": "无法满足特定服务类型的安全需求",
}
CAT5_TARGET = 8000   # 类别5过采样目标量
OUTPUT_COLS = ["prompt", "prompt_harm_label", "category_id", "category_label"]


def load_and_tag(path: str, source: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df.columns.intersection(OUTPUT_COLS + ["source"])].copy()
    df["source"] = source
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLS + ["source"]]


def fix_category(df: pd.DataFrame, v3_map: dict) -> pd.DataFrame:
    """用 v3_labeled 的 prompt→category 映射填补缺失的 category_id。"""
    null_mask = df["category_id"].isna()
    if null_mask.sum() == 0:
        return df
    harmful_null = null_mask & (df["prompt_harm_label"] == "harmful")
    unharmful_null = null_mask & (df["prompt_harm_label"] == "unharmful")
    # unharmful 全部标 0
    df.loc[unharmful_null, "category_id"] = "0"
    # harmful 用最多类别 "1" 兜底
    df.loc[harmful_null, "category_id"] = "1"
    print(f"  修复 category_id: unharmful={unharmful_null.sum()}, harmful={harmful_null.sum()} (兜底=1)")
    return df


def stratified_split(df: pd.DataFrame, val_ratio: float, test_ratio: float,
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 category_id 分层分割，处理样本过少的边缘情况。"""
    train_parts, val_parts, test_parts = [], [], []
    min_split_size = 20  # 少于此数的类别全放 train

    for cat_id, group in df.groupby("category_id"):
        if len(group) < min_split_size:
            train_parts.append(group)
            continue
        # 先切出 test
        try:
            rest, test = train_test_split(
                group, test_size=test_ratio, random_state=seed, shuffle=True)
            # 再从剩余切出 val
            val_size_adj = val_ratio / (1 - test_ratio)
            train, val = train_test_split(
                rest, test_size=val_size_adj, random_state=seed, shuffle=True)
        except ValueError:
            train_parts.append(group)
            continue
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame(columns=df.columns)
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame(columns=df.columns)
    return train_df, val_df, test_df


def oversample_cat5(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    cat5 = df[df["category_id"] == "5"]
    if len(cat5) >= target:
        return df
    extra_needed = target - len(cat5)
    extra = cat5.sample(extra_needed, replace=True, random_state=seed)
    return pd.concat([df, extra], ignore_index=True)


def compute_weights(df: pd.DataFrame) -> dict:
    """sqrt 阻尼逆频率权重，避免极端类别权重过高。"""
    counts = df["category_id"].value_counts()
    n_total = len(df)
    n_classes = len(counts)
    weights = {}
    for cat_id, cnt in counts.items():
        raw = n_total / (n_classes * cnt)
        weights[cat_id] = round(math.sqrt(raw), 4)
    # 归一化到最小权重=1
    min_w = min(weights.values())
    weights = {k: round(v / min_w, 4) for k, v in weights.items()}
    return weights


def print_distribution(df: pd.DataFrame, name: str):
    print(f"\n{name}（{len(df):,} 条）:")
    print(f"  harmful={( df['prompt_harm_label']=='harmful').sum():,}  "
          f"unharmful={(df['prompt_harm_label']=='unharmful').sum():,}")
    cats = df["category_id"].value_counts().sort_index()
    for cat_id, cnt in cats.items():
        label = LABEL_MAP.get(str(cat_id), cat_id)
        bar = "█" * max(1, int(cnt / 2000))
        print(f"  [{cat_id}] {label[:14]:<14}  {cnt:>7,}  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--val_ratio",  type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    args = parser.parse_args()

    print("=== 整合最终训练集 ===\n")

    # ── Step 1：修复 injection_defense 的 category_id ─────────────────
    print("Step 1: 修复 injection_defense category_id...")
    v3 = pd.read_parquet("data/train_hanguard_v3_labeled.parquet")
    v3_map = dict(zip(v3["prompt"], v3["category_id"]))

    inj = pd.read_parquet("data/injection_defense_samples.parquet")
    inj = fix_category(inj, v3_map)
    inj["category_label"] = inj["category_id"].map(LABEL_MAP)
    inj["source"] = "injection_defense"
    null_remaining = inj["category_id"].isna().sum()
    print(f"  修复后 null 剩余: {null_remaining}")

    # ── Step 2：加载并标注所有数据集 ─────────────────────────────────
    print("\nStep 2: 加载所有数据集...")

    natural_datasets = {
        "wildguard_zh": "data/train_wildguard_zh.parquet",
        "v3_labeled":   "data/train_hanguard_v3_labeled.parquet",
        "retrans":      "data/wildguard_injection_retrans_labeled.parquet",
    }
    augment_datasets = {
        "v3_prefix_fix":     "data/v3_prefix_fix.parquet",
        "injection_defense": None,  # 已加载
    }

    natural_dfs = {}
    for name, path in natural_datasets.items():
        df = load_and_tag(path, name)
        df["category_label"] = df["category_id"].map(LABEL_MAP)
        natural_dfs[name] = df
        print(f"  {name}: {len(df):,}")

    prefix_fix = load_and_tag("data/v3_prefix_fix.parquet", "v3_prefix_fix")
    prefix_fix["category_label"] = prefix_fix["category_id"].map(LABEL_MAP)
    print(f"  v3_prefix_fix: {len(prefix_fix):,}")
    inj_clean = inj[OUTPUT_COLS + ["source"]].copy()
    print(f"  injection_defense: {len(inj_clean):,}")

    # ── Step 3：分层分割自然数据 ──────────────────────────────────────
    print("\nStep 3: 分层分割自然数据（按 source × category）...")

    train_parts, val_parts, test_parts = [], [], []

    for name, df in natural_dfs.items():
        tr, va, te = stratified_split(
            df, args.val_ratio, args.test_ratio, args.seed)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        print(f"  {name}: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    # 增强数据全进 train
    train_parts.append(prefix_fix)
    train_parts.append(inj_clean)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts,   ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)

    # 打乱 train（val/test 保持来源可追溯）
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Step 4：类别5过采样 ───────────────────────────────────────────
    print(f"\nStep 4: 类别5过采样（train → 目标 {CAT5_TARGET} 条）...")
    cat5_before = (train_df["category_id"] == "5").sum()
    train_df = oversample_cat5(train_df, CAT5_TARGET, args.seed)
    cat5_after = (train_df["category_id"] == "5").sum()
    print(f"  类别5: {cat5_before:,} → {cat5_after:,}")

    # ── Step 5：计算类别权重 ─────────────────────────────────────────
    print("\nStep 5: 计算类别权重（sqrt 阻尼逆频率）...")
    weights = compute_weights(train_df)
    for cat_id, w in sorted(weights.items()):
        print(f"  [{cat_id}] {LABEL_MAP.get(str(cat_id),'?')[:20]:<20}  weight={w}")
    with open("data/class_weights.json", "w") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print("  → 保存至 data/class_weights.json")

    # ── 保存 ─────────────────────────────────────────────────────────
    print("\n保存最终数据集...")
    train_df[OUTPUT_COLS].to_parquet("data/final_train.parquet", index=False)
    val_df[OUTPUT_COLS].to_parquet("data/final_val.parquet",     index=False)
    test_df[OUTPUT_COLS].to_parquet("data/final_test.parquet",   index=False)

    print_distribution(train_df, "final_train")
    print_distribution(val_df,   "final_val")
    print_distribution(test_df,  "final_test")

    print(f"\n完成！")
    print(f"  data/final_train.parquet  {len(train_df):,} 条")
    print(f"  data/final_val.parquet    {len(val_df):,} 条")
    print(f"  data/final_test.parquet   {len(test_df):,} 条")
    print(f"  data/class_weights.json")


if __name__ == "__main__":
    main()
