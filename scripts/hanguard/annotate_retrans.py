"""
对 wildguard_injection_retrans.parquet 补标六分类 category_id。

输入：data/wildguard_injection_retrans.parquet
输出：data/wildguard_injection_retrans_labeled.parquet

用法：
  python scripts/hanguard/annotate_retrans.py
  python scripts/hanguard/annotate_retrans.py --concurrency 15
"""

import argparse
import asyncio
import time
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from tqdm.asyncio import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from annotate_system_prompt import SYSTEM_PROMPT, LABEL_MAP, build_messages

INPUT  = Path("data/wildguard_injection_retrans.parquet")
OUTPUT = Path("data/wildguard_injection_retrans_labeled.parquet")
CKPT   = Path("data/wildguard_injection_retrans_labeled_ckpt.parquet")

API_KEY     = "sk-dtbqRykhzrJAkif8RDX77CTmdjFWaeNdjMIvB98s2gUQDWIX"
BASE_URL    = "https://vip.aipro.love"
MODEL       = "claude-sonnet-4-6"
RETRY       = 3
CONCURRENCY = 15
CKPT_EVERY  = 200


def parse_label(text: str) -> str:
    text = text.strip()
    if text in LABEL_MAP:
        return text
    for ch in text:
        if ch in LABEL_MAP:
            return ch
    return "1"


async def call_api(client, sem, idx, prompt):
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


async def run(df, indices, concurrency):
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(proxy=None, timeout=60),
    )
    sem = asyncio.Semaphore(concurrency)
    results, done = {}, 0

    tasks = [call_api(client, sem, idx, df.at[idx, "prompt"]) for idx in indices]
    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="标注"):
        idx, label = await coro
        results[idx] = label
        done += 1
        if done % CKPT_EVERY == 0:
            for i, l in results.items():
                df.at[i, "category_id"] = l
            df.to_parquet(CKPT)

    for i, l in results.items():
        df.at[i, "category_id"] = l
    await client.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    df = pd.read_parquet(INPUT)
    df["category_id"] = None

    if CKPT.exists():
        ckpt = pd.read_parquet(CKPT)
        if "category_id" in ckpt.columns:
            df["category_id"] = ckpt["category_id"].values
            done = df["category_id"].notna().sum()
            print(f"断点续传：已完成 {done:,} 条")

    todo = df[df["category_id"].isna()].index.tolist()
    print(f"待标注: {len(todo):,} 条，并发数 {args.concurrency}")
    est = len(todo) * 1.3 / args.concurrency / 60
    print(f"预计耗时: {est:.0f} 分钟")

    if todo:
        t0 = time.time()
        df = asyncio.run(run(df, todo, args.concurrency))
        print(f"\n标注耗时: {(time.time()-t0)/60:.1f} 分钟")

    df["category_label"] = df["category_id"].map(LABEL_MAP)
    df.to_parquet(OUTPUT, index=False)

    print(f"\n完成！保存至 {OUTPUT}，共 {len(df):,} 条")
    print("\ncategory_label 分布：")
    print(df["category_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
