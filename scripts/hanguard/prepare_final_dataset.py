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
# JailBench 一级领域 → category_id
JAILBENCH_DOMAIN_MAP = {
    "违反社会主义核心价值观":      "1",
    "歧视":                      "2",
    "商业违法违规":               "3",
    "侵犯他人合法权益":            "4",
    "无法满足特定服务类型的安全需求": "5",
}
CAT5_TARGET = 4000   # 类别5过采样目标量（2.7x，避免过度重复）
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

    # ── Step 1：加载 injection_defense（可选，文件不存在则跳过）────────
    inj_path = Path("data/sources/injection_defense.parquet")
    if inj_path.exists():
        print("Step 1: 加载 injection_defense...")
        inj = pd.read_parquet(str(inj_path))
        # unharmful → 0，harmful null → 兜底 1
        inj.loc[(inj["prompt_harm_label"] == "unharmful") & inj["category_id"].isna(), "category_id"] = "0"
        inj.loc[(inj["prompt_harm_label"] == "harmful")   & inj["category_id"].isna(), "category_id"] = "1"
        inj["category_label"] = inj["category_id"].map(LABEL_MAP)
        inj["source"] = "injection_defense"
        print(f"  injection_defense: {len(inj):,} 条")
        inj_clean = inj[OUTPUT_COLS + ["source"]].copy()
    else:
        inj_clean = None
        print("Step 1: injection_defense.parquet 不存在，跳过")

    # ── Step 2：加载并标注所有数据集 ─────────────────────────────────
    # 使用经过重标注的 _step2.parquet（harmful+category=0 矛盾已修复）
    print("\nStep 2: 加载所有数据集...")

    natural_datasets = {
        "wildguard_zh": "data/sources/wildguard_zh.parquet",
        "v3_labeled":   "data/sources/hanguard_v3.parquet",
        "retrans":      "data/sources/wildguard_retrans.parquet",
    }

    natural_dfs = {}
    for name, path in natural_datasets.items():
        df = load_and_tag(path, name)
        df["category_label"] = df["category_id"].map(LABEL_MAP)
        before = len(df)
        # 去除空 prompt
        df = df[df["prompt"].str.strip() != ""].copy()
        # 对同一 prompt 标签冲突的行，保留多数票标签；票数相等则全部丢弃
        conflict_mask = df.groupby("prompt")["prompt_harm_label"].transform("nunique") > 1
        if conflict_mask.sum() > 0:
            def majority_row(g):
                winner = g["prompt_harm_label"].mode()
                if len(winner) > 1:
                    return pd.DataFrame()   # 票数相等，全部丢弃
                return g[g["prompt_harm_label"] == winner.iloc[0]].head(1)
            conflict_df   = df[conflict_mask]
            clean_df      = df[~conflict_mask]
            resolved      = conflict_df.groupby("prompt", group_keys=False).apply(majority_row)
            df = pd.concat([clean_df, resolved], ignore_index=True)
        # 按 prompt 去重，保留第一条
        df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
        after = len(df)
        print(f"  {name}: {before:,} → {after:,}（去重 {before - after:,} 条）")
        natural_dfs[name] = df

    # ── JailBench：高质量越狱攻击数据集 ──────────────────────────────
    jb_query_path = Path("data/sources/jailbench/JailBench.csv")
    jb_seed_path  = Path("data/sources/jailbench/JailBench-seed.csv")
    if jb_query_path.exists() and jb_seed_path.exists():
        jb_queries = pd.read_csv(str(jb_query_path))[["query", "一级领域"]].rename(columns={"query": "prompt"})
        jb_seeds   = pd.read_csv(str(jb_seed_path))[["seed",  "一级领域"]].rename(columns={"seed":  "prompt"})
        jb_all = pd.concat([jb_queries, jb_seeds], ignore_index=True)
        jb_all["category_id"]     = jb_all["一级领域"].map(JAILBENCH_DOMAIN_MAP)
        jb_all["category_label"]  = jb_all["category_id"].map(LABEL_MAP)
        jb_all["prompt_harm_label"] = "harmful"
        jb_all["source"] = "jailbench"
        jb_all = jb_all.dropna(subset=["category_id"])
        jb_all = jb_all[jb_all["prompt"].str.strip() != ""]
        jb_all = jb_all.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
        # 跨数据源去重：排除已存在于其他源的 prompt
        existing_prompts = set()
        for df in natural_dfs.values():
            existing_prompts.update(df["prompt"].str.strip())
        before = len(jb_all)
        jb_all = jb_all[~jb_all["prompt"].str.strip().isin(existing_prompts)].reset_index(drop=True)
        after = len(jb_all)
        natural_dfs["jailbench"] = jb_all[OUTPUT_COLS + ["source"]]
        print(f"  jailbench: {before:,} → {after:,}（跨源去重 {before - after:,} 条）")
        for cat_id, cnt in jb_all["category_id"].value_counts().sort_index().items():
            print(f"    [{cat_id}] {LABEL_MAP.get(cat_id,'?')[:20]:<20}  {cnt:,}")
    else:
        print("  jailbench: CSV 文件不存在，跳过")

    # cat5_generated：类别5补充样本，存在则作为自然数据参与分层分割
    cat5_path = Path("data/sources/cat5_generated.parquet")
    if cat5_path.exists():
        cat5_df = load_and_tag(str(cat5_path), "cat5_generated")
        cat5_df["category_label"] = cat5_df["category_id"].map(LABEL_MAP)
        cat5_df = cat5_df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
        natural_dfs["cat5_generated"] = cat5_df
        print(f"  cat5_generated: {len(cat5_df):,}")
    else:
        print("  cat5_generated: 文件不存在，跳过")

    # v3_prefix_fix 是历史遗留增强文件，没有对应生成脚本；存在则加载，否则跳过
    prefix_fix_path = Path("data/v3_prefix_fix.parquet")
    if prefix_fix_path.exists():
        prefix_fix = load_and_tag(str(prefix_fix_path), "v3_prefix_fix")
        prefix_fix["category_label"] = prefix_fix["category_id"].map(LABEL_MAP)
        print(f"  v3_prefix_fix: {len(prefix_fix):,}")
    else:
        prefix_fix = None
        print("  v3_prefix_fix: 文件不存在，跳过（如需包含请手动放置 data/v3_prefix_fix.parquet）")

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
    if prefix_fix is not None:
        train_parts.append(prefix_fix)
    if inj_clean is not None:
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
