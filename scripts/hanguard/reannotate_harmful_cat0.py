import os
"""
对 harmful + category_id=0 的样本重新标注六分类（category 1-5）。

从三个 source 文件中筛出 harmful+category=0 的样本，
用"已知有害，请判断类别"的系统提示重新标注，结果写回原 source 文件。

用法：
  python scripts/hanguard/reannotate_harmful_cat0.py
  python scripts/hanguard/reannotate_harmful_cat0.py --dry-run
  python scripts/hanguard/reannotate_harmful_cat0.py --concurrency 20
"""

import argparse
import asyncio
import time
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from annotate_system_prompt import LABEL_MAP, build_messages

# ── 路径 ──────────────────────────────────────────────────────────────
SOURCE_FILES = [
    Path("data/sources/train_wildguard_zh.parquet"),
    Path("data/sources/train_hanguard_v3_labeled.parquet"),
    Path("data/sources/wildguard_injection_retrans_labeled.parquet"),
]
CKPT = Path("data/reannotate_harmful_cat0_ckpt.parquet")

# ── API 配置 ──────────────────────────────────────────────────────────
API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL    = "https://vip.aipro.love"
MODEL       = "claude-sonnet-4-6"
RETRY       = 3
CONCURRENCY = 15
CKPT_EVERY  = 200

# ── 系统提示：已知有害，强制输出 1-5 ─────────────────────────────────
SYSTEM_PROMPT_FORCED = """你是一个内容安全分类系统。以下请求已被判定为有害内容，你的任务是判断它属于哪种有害类别，输出对应编号（1-5）。

## 分类标准（依据 GB/T 45654-2025）

**1. 违反社会主义核心价值观的内容**
诱导生成以下任一内容：
- 煽动颠覆国家政权、分裂国家、破坏社会稳定
- 危害国家安全、损害国家形象
- 宣扬恐怖主义、极端主义、民族仇恨
- 宣扬暴力、淫秽色情
- 传播虚假有害信息、谣言

**2. 歧视性内容**
诱导生成针对特定群体的贬损、侮辱或排斥内容，涉及：
民族、宗教信仰、国籍、地域、性别、年龄、职业、健康状况等维度。
注意：煽动民族仇恨归类为1，单纯针对群体的偏见、标签化归类为2。

**3. 商业违法违规**
诱导生成以下任一内容：
- 侵犯知识产权（著作权、商标、专利）
- 虚假宣传、欺诈营销、恶意竞争
- 泄露企业商业秘密、内部数据
- 利用技术或平台优势实施垄断、不正当竞争

**4. 侵犯他人合法权益**
诱导生成以下任一内容：
- 危害他人身心健康（伤害指导、精神压迫）
- 侵害肖像权（深度伪造、恶意合成）
- 侵害名誉权、荣誉权（诽谤、造谣）
- 侵害隐私权、个人信息权益（泄露私密信息、身份信息、行踪轨迹等）
注意：侵犯个人隐私归类为4，侵犯企业商业秘密归类为3。

**5. 无法满足特定服务类型的安全需求**
诱导在医疗、法律、金融、心理咨询、关键基础设施等高风险专业场景中生成：
- 不准确的内容（与事实、专业知识严重不符）
- 不可靠的内容（缺乏可验证依据，可能误导专业判断）

## 优先级规则
当内容同时符合多个类别时，按以下顺序取最高优先级：
1 > 4 > 2 > 3 > 5

## 注意
- 该请求已确认有害，输出必须为 1~5 之一，不能输出 0
- 如果难以判断具体类别，输出 1
- 只输出一个数字（1~5），不要任何解释。"""


def parse_label_forced(text: str) -> str:
    for ch in text.strip():
        if ch in "12345":
            return ch
    return "1"


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
                    system=SYSTEM_PROMPT_FORCED,
                    messages=build_messages(prompt),
                )
                return idx, parse_label_forced(msg.content[0].text)
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

    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="重标注"):
        idx, label = await coro
        results[idx] = label
        done_count += 1
        if done_count % CKPT_EVERY == 0:
            tmp = df.copy()
            for i, l in results.items():
                tmp.at[i, "new_category_id"] = l
            tmp.to_parquet(CKPT)

    for i, l in results.items():
        df.at[i, "new_category_id"] = l

    await client.close()
    return df


def load_all_targets() -> pd.DataFrame:
    all_records = []
    for src in SOURCE_FILES:
        df = pd.read_parquet(src)
        mask = (df["prompt_harm_label"] == "harmful") & (df["category_id"] == "0")
        bad = df[mask].copy()
        bad["_source_file"] = str(src)
        bad["_source_idx"] = bad.index.tolist()
        all_records.append(bad)
        print(f"  {src.name}: {mask.sum():,} 条 harmful+category=0")
    df_all = pd.concat(all_records, ignore_index=True)
    df_all["new_category_id"] = None
    return df_all


def apply_checkpoint(df_all: pd.DataFrame) -> pd.DataFrame:
    if not CKPT.exists():
        return df_all
    df_ckpt = pd.read_parquet(CKPT)
    ckpt_map = dict(zip(df_ckpt["prompt"], df_ckpt["new_category_id"]))
    df_all["new_category_id"] = df_all["prompt"].map(ckpt_map)
    done = df_all["new_category_id"].notna().sum()
    print(f"断点续传：已完成 {done:,} 条，剩余 {len(df_all)-done:,} 条")
    return df_all


def write_back(df_all: pd.DataFrame):
    print("\n写回 source 文件...")
    for src in SOURCE_FILES:
        df_src = pd.read_parquet(src)
        subset = df_all[df_all["_source_file"] == str(src)]
        changed = 0
        for _, row in subset.iterrows():
            new_cat = row.get("new_category_id")
            if pd.notna(new_cat) and new_cat in "12345":
                orig_idx = row["_source_idx"]
                df_src.at[orig_idx, "category_id"]    = new_cat
                df_src.at[orig_idx, "category_label"] = LABEL_MAP[new_cat]
                changed += 1
        df_src.to_parquet(src, index=False)
        print(f"  {src.name}: 更新 {changed:,} 条")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="每个 source 文件各取 10 条标注，验证质量后再正式跑")
    parser.add_argument("--check", action="store_true",
                        help="打印已标注样本供人工 review（需先跑过 --dry-run 或正式标注）")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    # ── --check 模式：读 ckpt 打印样本，不调 API ──────────────────────────
    if args.check:
        if not CKPT.exists():
            print("尚无标注结果，请先运行 --dry-run 或正式标注。")
            return
        df_ckpt = pd.read_parquet(CKPT)
        done = df_ckpt[df_ckpt["new_category_id"].notna()]
        print(f"已标注样本：{len(done)} 条\n")
        print("new_category_id 分布：")
        print(done["new_category_id"].value_counts().sort_index().to_string())
        print()
        # 每个 source 文件各打印 5 条
        for src in SOURCE_FILES:
            subset = done[done["_source_file"] == str(src)].head(5)
            if subset.empty:
                continue
            print(f"── {src.name} ──")
            for _, row in subset.iterrows():
                old_label = LABEL_MAP.get("0", "安全")
                new_label = LABEL_MAP.get(row["new_category_id"], "?")
                print(f"  prompt : {str(row['prompt'])[:80]}")
                print(f"  旧类别 : 0 {old_label}  →  新类别 : {row['new_category_id']} {new_label}")
                print()
        return

    # ── 收集待标注样本 ────────────────────────────────────────────────────
    print("=== 收集 harmful+category=0 样本 ===")
    df_all = load_all_targets()
    print(f"合计需要重标: {len(df_all):,} 条")

    df_all = apply_checkpoint(df_all)

    # ── dry-run：每个 source 文件各取 10 条 ──────────────────────────────
    if args.dry_run:
        per_file = 10
        dry_indices = []
        for src in SOURCE_FILES:
            pending = df_all[
                (df_all["_source_file"] == str(src)) &
                df_all["new_category_id"].isna()
            ].index.tolist()
            dry_indices.extend(pending[:per_file])
        api_indices = dry_indices
        print(f"\ndry-run：每个文件各取 {per_file} 条，共 {len(api_indices)} 条")
    else:
        api_indices = df_all[df_all["new_category_id"].isna()].index.tolist()
        print(f"\n需要 API 标注: {len(api_indices):,} 条，并发数 {args.concurrency}")
        est = len(api_indices) * 1.3 / args.concurrency / 60
        print(f"预计耗时: {est:.0f} 分钟")

    # ── 调用 API ──────────────────────────────────────────────────────────
    if api_indices:
        t0 = time.time()
        df_all = asyncio.run(run_annotation(df_all, api_indices, args.concurrency))
        elapsed = time.time() - t0
        print(f"\n标注耗时: {elapsed/60:.1f} 分钟，"
              f"平均 {elapsed/len(api_indices):.2f}s/条（并发{args.concurrency}）")

    # ── 保存 ckpt ─────────────────────────────────────────────────────────
    df_all.to_parquet(CKPT)
    print(f"断点/结果保存至 {CKPT}")

    # ── 正式跑完才写回 source ─────────────────────────────────────────────
    if not args.dry_run:
        write_back(df_all)
        print("\n重标结果 new_category_id 分布：")
        print(df_all["new_category_id"].value_counts().sort_index().to_string())
        print("\n完成！运行 --check 可查看标注样本详情。")
    else:
        print("\ndry-run 完成，运行 --check 查看结果，确认质量后再正式跑。")


if __name__ == "__main__":
    main()
