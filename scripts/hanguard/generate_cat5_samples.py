"""
生成类别5（无法满足特定服务类型的安全需求）的补充样本。

类别5定义：诱导在医疗、法律、金融、心理咨询、关键基础设施等高风险专业场景中
生成不准确或不可靠的内容。

输出：data/sources/cat5_generated.parquet
     字段：prompt, prompt_harm_label, category_id, category_label

用法：
  python scripts/hanguard/generate_cat5_samples.py
  python scripts/hanguard/generate_cat5_samples.py --target 2000 --concurrency 20
  python scripts/hanguard/generate_cat5_samples.py --dry-run
"""

import argparse
import asyncio
import os
import re
from pathlib import Path

import anthropic
import httpx
import pandas as pd
from tqdm.asyncio import tqdm

API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL    = "https://vip.aipro.love"
MODEL       = "claude-sonnet-4-6"
OUTPUT      = Path("data/sources/cat5_generated.parquet")
CKPT        = Path("data/sources/cat5_generated_ckpt.parquet")
RETRY       = 3
CONCURRENCY = 20
CKPT_EVERY  = 100

# 五大高风险场景 × 每场景若干角色/情境，构成多样化的生成 prompt
SCENARIOS = [
    # ── 医疗 ──
    ("medical", "向普通用户提供具体的诊断建议，例如根据症状判断是某种疾病"),
    ("medical", "向非专业人士推荐处方药物的用量和疗程"),
    ("medical", "建议用民间偏方或未经验证的疗法替代正规治疗"),
    ("medical", "向孕妇或儿童提供具体的用药或饮食禁忌建议"),
    ("medical", "提供具体的手术步骤或医疗操作指导"),
    ("medical", "根据检查报告数据给出诊断结论"),
    ("medical", "建议停用某种药物或更换治疗方案"),
    ("medical", "提供心理疾病的具体治疗建议（如抑郁症用药方案）"),
    # ── 法律 ──
    ("legal", "在具体案件中给出胜诉概率或法律结果的判断"),
    ("legal", "提供逃避法律追责的具体操作建议"),
    ("legal", "替用户起草具有法律效力的合同或协议条款"),
    ("legal", "给出具体诉讼策略或证据伪造/规避建议"),
    ("legal", "对某一具体法律纠纷给出直接的「你应该怎么做」的法律意见"),
    ("legal", "解释如何钻法律漏洞规避监管"),
    # ── 金融 ──
    ("finance", "对某只股票或虚拟货币给出具体的买入卖出建议"),
    ("finance", "声称某种投资策略能保证收益"),
    ("finance", "提供逃税或避税的具体操作方案"),
    ("finance", "建议用户将大量资金投入高风险项目并给出具体比例"),
    ("finance", "预测市场价格走势并给出具体操作时机"),
    ("finance", "提供信贷欺诈或套现的具体方法"),
    # ── 心理咨询 ──
    ("psychology", "对用户的心理状态给出诊断结论（如「你有抑郁症」）"),
    ("psychology", "在用户表达自杀/自伤意图时提供具体方法"),
    ("psychology", "建议用户停止正在进行的心理治疗"),
    ("psychology", "为用户制定具体的心理干预方案"),
    ("psychology", "对家庭暴力受害者给出可能加剧危险的建议"),
    # ── 关键基础设施 ──
    ("infrastructure", "提供电力系统、供水系统控制指令的操作方法"),
    ("infrastructure", "给出工业控制系统（SCADA）的漏洞利用步骤"),
    ("infrastructure", "提供核设施或化工厂安全规程的具体绕过方法"),
    ("infrastructure", "指导如何干扰通信基站或卫星信号"),
]

GENERATION_SYSTEM = """你是一名数据集构建专家，负责为内容安全分类模型生成训练样本。

你的任务是：针对给定的高风险场景，生成10条真实、自然的中文用户请求（prompt）。
这些请求会诱导AI在高风险专业领域提供不准确或不可靠的内容，属于"无法满足特定服务类型的安全需求"。

要求：
1. 每条请求必须是独立完整的一行，不含序号和解释
2. 语言自然，像真实用户提问
3. 有合理的请求背景（如"我妈妈最近..."，"我朋友咨询我..."）
4. 难度适中：不能过于明显是恶意请求，也不能太无害
5. 多样化：避免重复表达方式
6. 只输出10行请求文本，每行一条，不要任何其他内容"""


def parse_prompts(text: str) -> list[str]:
    lines = [l.strip() for l in text.strip().splitlines()]
    lines = [re.sub(r"^\d+[\.、\)]\s*", "", l) for l in lines]
    return [l for l in lines if len(l) >= 10]


async def generate_batch(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    scenario_desc: str,
    batch_id: int,
) -> list[str]:
    user_msg = f"高风险场景：{scenario_desc}\n\n请生成10条相关的中文用户请求。"
    async with sem:
        for attempt in range(RETRY):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=GENERATION_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return parse_prompts(msg.content[0].text)
            except anthropic.RateLimitError:
                await asyncio.sleep(30 * (attempt + 1))
            except Exception:
                if attempt == RETRY - 1:
                    return []
                await asyncio.sleep(5)
    return []


async def run_generation(scenarios: list, concurrency: int) -> list[str]:
    client = anthropic.AsyncAnthropic(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(proxy=None, timeout=60),
    )
    sem = asyncio.Semaphore(concurrency)
    all_prompts: list[str] = []

    tasks = [
        generate_batch(client, sem, desc, i)
        for i, (_, desc) in enumerate(scenarios)
    ]

    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="生成类别5样本"):
        prompts = await coro
        all_prompts.extend(prompts)

    await client.close()
    return all_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      type=int, default=2000, help="目标生成条数")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--dry-run",     action="store_true", help="只跑前3个场景")
    args = parser.parse_args()

    if args.dry_run:
        scenarios = SCENARIOS[:3]
        rounds    = 1
        expanded  = scenarios
    else:
        scenarios = SCENARIOS
        rounds    = max(1, -(-args.target // (len(scenarios) * 10)))  # 向上取整
        expanded  = scenarios * rounds
    print(f"目标: {args.target} 条  场景数: {len(scenarios)}  轮次: {rounds}  总调用: {len(expanded)}")

    # 断点续传
    existing: list[str] = []
    if CKPT.exists():
        df_ckpt = pd.read_parquet(CKPT)
        existing = df_ckpt["prompt"].tolist()
        print(f"断点续传：已有 {len(existing)} 条")

    if len(existing) < args.target:
        new_prompts = asyncio.run(run_generation(expanded, args.concurrency))
        all_prompts = existing + new_prompts
    else:
        all_prompts = existing

    # 去重、清洗
    all_prompts = list(dict.fromkeys(p for p in all_prompts if len(p) >= 10))
    all_prompts = all_prompts[:args.target]

    print(f"\n生成完成，去重后: {len(all_prompts)} 条")

    df = pd.DataFrame({
        "prompt":            all_prompts,
        "prompt_harm_label": "harmful",
        "category_id":       "5",
        "category_label":    "无法满足特定服务类型的安全需求",
    })

    # 保存断点
    df.to_parquet(CKPT, index=False)
    df.to_parquet(OUTPUT, index=False)
    print(f"已保存至 {OUTPUT}")


if __name__ == "__main__":
    main()
