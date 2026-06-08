"""
对 WildGuard 数据集做六分类标注（并发版）。

策略：
  - subcategory != 'others'：规则映射，直接转换
  - subcategory == 'others'：并发调 Claude API 标注
  - benign：直接标为 0（安全）

输入：
  data/hanguard_training_data/train.parquet  （含 subcategory）
  data/train_wildguard_zh.parquet           （含中文翻译 prompt）

输出：
  data/train_wildguard_labeled.parquet      （含 category_label 字段）

用法：
  python scripts/hanguard/annotate_others.py
  python scripts/hanguard/annotate_others.py --dry-run      # 只跑前20条测试
  python scripts/hanguard/annotate_others.py --concurrency 15
"""

import argparse
import asyncio
import time
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from annotate_system_prompt import SYSTEM_PROMPT, LABEL_MAP, build_messages

# ── 路径 ──────────────────────────────────────────────────────────────
DATA_EN   = Path("data/hanguard_training_data/train.parquet")
DATA_ZH   = Path("data/train_wildguard_zh.parquet")
OUTPUT    = Path("data/train_wildguard_labeled.parquet")
CKPT_FILE = Path("data/train_wildguard_labeled_ckpt.parquet")

# ── API 配置 ──────────────────────────────────────────────────────────
API_KEY      = "sk-dtbqRykhzrJAkif8RDX77CTmdjFWaeNdjMIvB98s2gUQDWIX"
BASE_URL     = "https://vip.aipro.love"
MODEL        = "claude-sonnet-4-6"
RETRY        = 3
CONCURRENCY  = 10   # 默认并发数
CKPT_EVERY   = 200  # 每处理 N 条保存一次断点

# ── 规则映射（subcategory → category_id）────────────────────────────
RULE_MAP = {
    "benign":                                                                        "0",
    "social_stereotypes_and_unfair_discrimination":                                  "2",
    "toxic_language_hate_speech":                                                    "1",
    "violence_and_physical_harm":                                                    "1",
    "sexual_content":                                                                "1",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": "1",
    "causing_material_harm_by_disseminating_misinformation":                         "1",
    "fraud_assisting_illegal_activities":                                            "3",
    "cyberattack":                                                                   "3",
    "copyright_violations":                                                          "3",
    "private_information_individual":                                                "4",
    "sensitive_information_organization_government":                                 "4",
    "defamation_encouraging_unethical_or_unsafe_actions":                            "4",
    "mental_health_over-reliance_crisis":                                            "5",
}


def parse_label(text: str) -> str:
    text = text.strip()
    if text in LABEL_MAP:
        return text
    for ch in text:
        if ch in LABEL_MAP:
            return ch
    return "0"


async def call_api_async(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    idx: int,
    prompt: str,
) -> tuple[int, str]:
    async with sem:
        for attempt in range(RETRY):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=4,
                    system=SYSTEM_PROMPT,
                    messages=build_messages(prompt),
                )
                return idx, parse_label(msg.content[0].text)
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                await asyncio.sleep(wait)
            except Exception as e:
                if attempt == RETRY - 1:
                    return idx, "0"
                await asyncio.sleep(5)
    return idx, "0"


async def run_annotation(df: pd.DataFrame, api_indices: list, concurrency: int) -> pd.DataFrame:
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(proxy=None, timeout=60),
    )
    sem = asyncio.Semaphore(concurrency)
    results = {}
    done_count = 0

    tasks = [
        call_api_async(client, sem, idx, df.at[idx, "prompt_zh"])
        for idx in api_indices
    ]

    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="API标注"):
        idx, label = await coro
        results[idx] = label
        done_count += 1

        if done_count % CKPT_EVERY == 0:
            for i, l in results.items():
                df.at[i, "category_id"] = l
            df.to_parquet(CKPT_FILE)

    for i, l in results.items():
        df.at[i, "category_id"] = l

    await client.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只处理前20条")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    print("加载数据...")
    df_en = pd.read_parquet(DATA_EN)[["prompt", "subcategory"]]
    df_zh = pd.read_parquet(DATA_ZH)[["prompt", "prompt_harm_label"]]
    df_zh = df_zh.rename(columns={"prompt": "prompt_zh"})

    df = pd.concat([df_en, df_zh], axis=1)
    df["category_id"] = None

    # 断点续传
    if CKPT_FILE.exists():
        df_ckpt = pd.read_parquet(CKPT_FILE)
        df.update(df_ckpt[["category_id"]])
        done = df["category_id"].notna().sum()
        print(f"断点续传：已完成 {done:,} 条")

    # 规则映射
    rule_mask = df["subcategory"].isin(RULE_MAP) & df["category_id"].isna()
    df.loc[rule_mask, "category_id"] = df.loc[rule_mask, "subcategory"].map(RULE_MAP)
    print(f"规则映射完成：{rule_mask.sum():,} 条")

    # 需要 API 标注的样本
    api_mask = (df["subcategory"] == "others") & df["category_id"].isna()
    api_indices = df[api_mask].index.tolist()

    if args.dry_run:
        api_indices = api_indices[:20]
        print(f"dry-run 模式：只标注 {len(api_indices)} 条，并发数 {args.concurrency}")
    else:
        print(f"需要 API 标注：{len(api_indices):,} 条，并发数 {args.concurrency}")
        est = len(api_indices) * 13 / args.concurrency / 60
        print(f"预计耗时：{est:.0f} 分钟")

    if api_indices:
        t0 = time.time()
        df = asyncio.run(run_annotation(df, api_indices, args.concurrency))
        elapsed = time.time() - t0
        print(f"\nAPI 标注耗时：{elapsed/60:.1f} 分钟，"
              f"平均 {elapsed/len(api_indices):.1f}s/条（并发{args.concurrency}）")
        df.to_parquet(CKPT_FILE)

    # 最终输出
    df["category_label"] = df["category_id"].map(LABEL_MAP)
    out_cols = ["prompt_zh", "prompt_harm_label", "category_id", "category_label"]
    df_out = df[out_cols].rename(columns={"prompt_zh": "prompt"})
    df_out.to_parquet(OUTPUT)

    print(f"\n完成！保存至 {OUTPUT}，共 {len(df_out):,} 条")
    print("\n分布：")
    print(df_out["category_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
