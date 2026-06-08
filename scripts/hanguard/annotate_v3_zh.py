import os
"""
对 train_hanguard_v3.parquet 中的中文样本补标 category_label。

策略：
  - prompt_harm_label == 'unharmful' → category_id = 0（安全），无需 API
  - prompt_harm_label == 'harmful'   → 并发调 Claude API 标注六分类

输入：  data/train_hanguard_v3.parquet
输出：  data/train_hanguard_v3_labeled.parquet

用法：
  python scripts/hanguard/annotate_v3_zh.py
  python scripts/hanguard/annotate_v3_zh.py --dry-run
  python scripts/hanguard/annotate_v3_zh.py --concurrency 15
"""

import argparse
import asyncio
import re
import time
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from annotate_system_prompt import SYSTEM_PROMPT, LABEL_MAP, build_messages

# ── 路径 ──────────────────────────────────────────────────────────────
INPUT   = Path("data/train_hanguard_v3.parquet")
OUTPUT  = Path("data/train_hanguard_v3_labeled.parquet")
CKPT    = Path("data/train_hanguard_v3_labeled_ckpt.parquet")

# ── API 配置 ──────────────────────────────────────────────────────────
API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL    = "https://vip.aipro.love"
MODEL       = "claude-sonnet-4-6"
RETRY       = 3
CONCURRENCY = 10
CKPT_EVERY  = 200


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", str(text)))


def parse_label(text: str) -> str:
    text = text.strip()
    if text in LABEL_MAP:
        return text
    for ch in text:
        if ch in LABEL_MAP:
            return ch
    return "1"  # 对 harmful 样本，解析失败时默认最严格类别


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
                await asyncio.sleep(30 * (attempt + 1))
            except Exception:
                if attempt == RETRY - 1:
                    return idx, "1"
                await asyncio.sleep(5)
    return idx, "1"


async def run_annotation(df: pd.DataFrame, api_indices: list, concurrency: int) -> pd.DataFrame:
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(proxy=None, timeout=60),
    )
    sem = asyncio.Semaphore(concurrency)
    results: dict[int, str] = {}
    done_count = 0

    tasks = [
        call_api_async(client, sem, idx, df.at[idx, "prompt"])
        for idx in api_indices
    ]

    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="API标注"):
        idx, label = await coro
        results[idx] = label
        done_count += 1
        if done_count % CKPT_EVERY == 0:
            for i, l in results.items():
                df.at[i, "category_id"] = l
            df.to_parquet(CKPT)

    for i, l in results.items():
        df.at[i, "category_id"] = l

    await client.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只处理前20条 harmful 样本")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    print("加载数据...")
    df_raw = pd.read_parquet(INPUT)

    # 只取中文样本
    df = df_raw[df_raw["prompt"].apply(is_chinese)].copy().reset_index(drop=True)
    print(f"中文样本: {len(df):,} 条（harmful: {(df['prompt_harm_label']=='harmful').sum():,}，"
          f"unharmful: {(df['prompt_harm_label']=='unharmful').sum():,}）")

    df["category_id"] = None

    # 断点续传
    if CKPT.exists():
        df_ckpt = pd.read_parquet(CKPT)
        # 按 prompt 对齐（ckpt 可能来自不同 index）
        ckpt_map = dict(zip(df_ckpt["prompt"], df_ckpt["category_id"]))
        df["category_id"] = df["prompt"].map(ckpt_map)
        done = df["category_id"].notna().sum()
        print(f"断点续传：已完成 {done:,} 条")

    # unharmful → 直接标为 0（安全）
    unharmful_mask = (df["prompt_harm_label"] == "unharmful") & df["category_id"].isna()
    df.loc[unharmful_mask, "category_id"] = "0"
    print(f"规则标注（unharmful→安全）: {unharmful_mask.sum():,} 条")

    # harmful → 需要 API
    api_mask = (df["prompt_harm_label"] == "harmful") & df["category_id"].isna()
    api_indices = df[api_mask].index.tolist()

    if args.dry_run:
        api_indices = api_indices[:20]
        print(f"dry-run：只标注 {len(api_indices)} 条")
    else:
        print(f"需要 API 标注: {len(api_indices):,} 条，并发数 {args.concurrency}")
        est = len(api_indices) * 1.3 / args.concurrency / 60
        print(f"预计耗时: {est:.0f} 分钟")

    if api_indices:
        t0 = time.time()
        df = asyncio.run(run_annotation(df, api_indices, args.concurrency))
        elapsed = time.time() - t0
        print(f"\nAPI 标注耗时: {elapsed/60:.1f} 分钟，"
              f"平均 {elapsed/len(api_indices):.1f}s/条（并发{args.concurrency}）")

    # 生成最终输出
    df["category_label"] = df["category_id"].map(LABEL_MAP)
    out = df[["prompt", "prompt_harm_label", "category_id", "category_label"]].copy()
    out.to_parquet(OUTPUT, index=False)

    print(f"\n完成！保存至 {OUTPUT}，共 {len(out):,} 条")
    print("\ncategory_label 分布：")
    print(out["category_label"].value_counts().to_string())
    print("\nprompt_harm_label 分布：")
    print(out["prompt_harm_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
