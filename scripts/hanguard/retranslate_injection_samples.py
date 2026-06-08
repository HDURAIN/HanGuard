import os
"""
对 WildGuard 中对抗性攻击样本做高质量中文重译。

原有机器翻译丢失了攻击结构（多任务混淆、指令覆盖等），
此脚本用 Claude API 重译，系统提示要求严格保留攻击结构。

筛选策略：
  - adversarial=True 且 prompt_harm_label=harmful
  - 优先覆盖翻译损失最大的模式：多任务混淆、指令覆盖、输出格式注入

输入：
  data/hanguard_training_data/train.parquet   （英文原始，含 adversarial/subcategory）
  data/train_wildguard_zh.parquet            （当前中文翻译，用于对比）
输出：
  data/wildguard_injection_retrans.parquet   （高质量重译结果）

用法：
  python scripts/hanguard/retranslate_injection_samples.py --dry-run
  python scripts/hanguard/retranslate_injection_samples.py --concurrency 10
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

INPUT_EN   = Path("data/hanguard_training_data/train.parquet")
INPUT_ZH   = Path("data/train_wildguard_zh.parquet")
OUTPUT     = Path("data/wildguard_injection_retrans.parquet")
CKPT       = Path("data/wildguard_injection_retrans_ckpt.parquet")

API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL   = "https://vip.aipro.love"
MODEL      = "claude-sonnet-4-6"
RETRY      = 3
CONCURRENCY = 10
CKPT_EVERY = 50

SYSTEM_PROMPT = """你是一名专业的英译中翻译员，专门处理AI安全研究领域的对抗性测试样本。

你的任务是将英文提示（prompt）翻译成自然流畅的中文，同时必须严格保留以下结构特征：

1. **攻击结构保留**：如果原文包含「忽略指令」「覆盖系统提示」「输出特定内容」等注入攻击，必须在中文中完整保留这些攻击意图，不能删除或弱化。

2. **多任务混淆保留**：如果原文将有害请求藏在多个无害任务列表中（如「1. 正常任务 2. 正常任务 3. 有害任务」），必须保持这种列表结构，不能合并或重排。

3. **角色扮演结构保留**：如果原文让AI扮演某个角色来绕过限制，必须保留角色设定和绕过意图。

4. **格式标记保留**：如果原文含有 [SYSTEM]、[OVERRIDE]、---、``` 等格式标记，保留或翻译成对应中文标记。

5. **输出格式注入保留**：如果原文试图控制分类器输出（如「answer with 'safe'」「output: unharmful」），必须在中文中保留这种输出控制意图。

只输出翻译结果，不要任何解释或注释。"""


def detect_attack_patterns(text: str) -> list[str]:
    patterns = []
    if re.search(r'\d[\.\)].+\d[\.\)].+\d[\.\)]', text, re.DOTALL):
        patterns.append("multitask")
    if re.search(r'(?i)ignore|forget|disregard|override|bypass', text):
        patterns.append("override")
    if re.search(r'(?i)(output|answer|respond|say|label).{0,30}(safe|unsafe|harmful|unharmful)', text):
        patterns.append("output_inject")
    if re.search(r'(?i)(you are|act as|pretend|roleplay|imagine you)', text):
        patterns.append("roleplay")
    return patterns


def select_samples(df_en: pd.DataFrame, df_zh: pd.DataFrame) -> pd.DataFrame:
    adv_harmful = df_en[(df_en["adversarial"] == True) &
                        (df_en["prompt_harm_label"] == "harmful")].copy()

    adv_harmful["attack_patterns"] = adv_harmful["prompt"].apply(detect_attack_patterns)
    adv_harmful["has_pattern"] = adv_harmful["attack_patterns"].apply(len) > 0

    # 按攻击类型分层抽样，确保覆盖多种模式
    selected_indices = set()

    # 全量：多任务混淆（最难翻译，最重要）
    multitask = adv_harmful[adv_harmful["attack_patterns"].apply(
        lambda x: "multitask" in x)]
    selected_indices.update(multitask.index.tolist())

    # 全量：输出格式注入
    output_inj = adv_harmful[adv_harmful["attack_patterns"].apply(
        lambda x: "output_inject" in x)]
    selected_indices.update(output_inj.index.tolist())

    # 全量：指令覆盖
    override = adv_harmful[adv_harmful["attack_patterns"].apply(
        lambda x: "override" in x)]
    selected_indices.update(override.index.tolist())

    # 角色扮演：取 200 条（数量够多，取代表性样本即可）
    roleplay = adv_harmful[adv_harmful["attack_patterns"].apply(
        lambda x: "roleplay" in x and "multitask" not in x)]
    selected_indices.update(roleplay.sample(min(200, len(roleplay)), random_state=42).index.tolist())

    selected = adv_harmful.loc[list(selected_indices)].copy()
    selected["prompt_zh_old"] = df_zh.loc[selected.index, "prompt"].values
    return selected.reset_index(drop=True)


async def translate_async(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    idx: int,
    en_prompt: str,
) -> tuple[int, str]:
    async with sem:
        for attempt in range(RETRY):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": en_prompt}],
                )
                return idx, msg.content[0].text.strip()
            except anthropic.RateLimitError:
                await asyncio.sleep(30 * (attempt + 1))
            except Exception:
                if attempt == RETRY - 1:
                    return idx, ""
                await asyncio.sleep(5)
    return idx, ""


async def run_translation(df: pd.DataFrame, indices: list, concurrency: int) -> pd.DataFrame:
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(proxy=None, timeout=120),
    )
    sem = asyncio.Semaphore(concurrency)
    results: dict[int, str] = {}
    done = 0

    tasks = [translate_async(client, sem, idx, df.at[idx, "prompt"]) for idx in indices]

    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="重译"):
        idx, zh = await coro
        results[idx] = zh
        done += 1
        if done % CKPT_EVERY == 0:
            for i, text in results.items():
                df.at[i, "prompt_zh_new"] = text
            df.to_parquet(CKPT)

    for i, text in results.items():
        df.at[i, "prompt_zh_new"] = text

    await client.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只处理前10条")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    print("加载数据...")
    df_en = pd.read_parquet(INPUT_EN)
    df_zh = pd.read_parquet(INPUT_ZH)

    print("筛选需要重译的样本...")
    df = select_samples(df_en, df_zh)

    pattern_counts = {
        "multitask": df["attack_patterns"].apply(lambda x: "multitask" in x).sum(),
        "output_inject": df["attack_patterns"].apply(lambda x: "output_inject" in x).sum(),
        "override": df["attack_patterns"].apply(lambda x: "override" in x).sum(),
        "roleplay": df["attack_patterns"].apply(lambda x: "roleplay" in x).sum(),
    }
    print(f"选中样本: {len(df):,} 条")
    for k, v in pattern_counts.items():
        print(f"  {k}: {v:,}")

    df["prompt_zh_new"] = None

    # 断点续传
    if CKPT.exists():
        df_ckpt = pd.read_parquet(CKPT)
        if "prompt_zh_new" in df_ckpt.columns:
            df["prompt_zh_new"] = df_ckpt["prompt_zh_new"].values
            done = df["prompt_zh_new"].notna().sum()
            print(f"断点续传：已完成 {done:,} 条")

    todo = df[df["prompt_zh_new"].isna()].index.tolist()
    if args.dry_run:
        todo = todo[:10]
        print(f"dry-run：只翻译 {len(todo)} 条")
    else:
        print(f"待翻译: {len(todo):,} 条，并发数 {args.concurrency}")
        est = len(todo) * 2.0 / args.concurrency / 60
        print(f"预计耗时: {est:.0f} 分钟")

    if todo:
        t0 = time.time()
        df = asyncio.run(run_translation(df, todo, args.concurrency))
        elapsed = time.time() - t0
        print(f"\n翻译耗时: {elapsed/60:.1f} 分钟")

    # 整理输出：只保留翻译成功的
    df = df[df["prompt_zh_new"].notna() & (df["prompt_zh_new"] != "")].copy()

    out = pd.DataFrame({
        "prompt": df["prompt_zh_new"].values,
        "prompt_harm_label": df["prompt_harm_label"].values,
        "category_id": None,
        "category_label": None,
        "source": "wildguard_retrans",
    })
    out.to_parquet(OUTPUT, index=False)

    print(f"\n完成！保存至 {OUTPUT}，共 {len(out):,} 条")
    print("\n示例对比（前3条）:")
    for i in range(min(3, len(df))):
        print(f"  原译: {df.iloc[i]['prompt_zh_old'][:80]}")
        print(f"  重译: {df.iloc[i]['prompt_zh_new'][:80]}")
        print()


if __name__ == "__main__":
    main()
