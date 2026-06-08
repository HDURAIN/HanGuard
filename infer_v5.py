"""
HanGuard v5 推理脚本。

加载训练好的 Qwen2.5-7B-Instruct + LoRA 权重，对 CSV/Parquet 文件批量推理，
输出 harmful_pred（harmful/unharmful）和 category_pred（0-5 类别 ID）。

用法：
  CUDA_VISIBLE_DEVICES=0 python infer_v5.py \
      --input  data/final_test.parquet \
      --model  outputs/hanguard_v5

  CUDA_VISIBLE_DEVICES=0 python infer_v5.py \
      --input  my_data.csv \
      --model  outputs/hanguard_v5 \
      --output outputs/my_results.csv \
      --batch_size 16
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_MSG = (
    "You are a content safety classifier for Chinese AI systems. "
    "Given a user request, output two labels:\n"
    "1. Whether the request is harmful (harmful/unharmful)\n"
    "2. The harm category (0=安全, 1=违反社会主义核心价值观的内容, "
    "2=歧视性内容, 3=商业违法违规, 4=侵犯他人合法权益, "
    "5=无法满足特定服务类型的安全需求)"
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "Human user:\n{prompt}\n\n---\n\nAnswers:<|im_end|>\n"
    "<|im_start|>assistant\n"
)

CATEGORY_LABELS = {
    "0": "安全",
    "1": "违反社会主义核心价值观的内容",
    "2": "歧视性内容",
    "3": "商业违法违规",
    "4": "侵犯他人合法权益",
    "5": "无法满足特定服务类型的安全需求",
}


def build_prompt(prompt: str) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM_MSG, prompt=prompt)


def parse_output(text: str) -> tuple[str, str]:
    text = text.strip()
    harm = "unharmful"
    cat = "0"

    m = re.search(r"harmful request\s*:\s*(harmful|unharmful)", text, re.I)
    if m:
        harm = m.group(1).lower()

    m = re.search(r"harm category\s*:\s*([^\n<]+)", text, re.I)
    if m:
        raw = m.group(1).strip()
        # 提取第一个数字
        digit = re.search(r"[0-5]", raw)
        if digit:
            cat = digit.group(0)

    # 一致性校验：unharmful 应为类别 0
    if harm == "unharmful":
        cat = "0"

    return harm, cat


def load_model(model_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    adapter_cfg = Path(model_dir) / "adapter_config.json"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.padding_side = "left"  # 生成时左 pad

    if adapter_cfg.exists():
        import json
        base_model = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        print(f"加载基座模型: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {model_dir}")
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def infer_batch(
    prompts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_input_len: int = 480,
    max_new_tokens: int = 32,
) -> list[tuple[str, str]]:
    results = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="推理"):
        batch_texts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        results.extend(parse_output(text) for text in decoded)
    return results


def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=str, required=True, help="CSV 或 Parquet，需含 prompt 列")
    parser.add_argument("--model",      type=str, default="outputs/hanguard_v5")
    parser.add_argument("--output",     type=str, default=None, help="输出 CSV 路径（默认同目录）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit",      type=int, default=None, help="只处理前 N 条")
    args = parser.parse_args()

    df = load_input(args.input)
    if "prompt" not in df.columns:
        raise ValueError("输入文件必须含 'prompt' 列")
    if args.limit:
        df = df.head(args.limit).copy()

    print(f"输入: {args.input}  ({len(df):,} 条)")

    prompts_built = [build_prompt(str(p)) for p in df["prompt"]]
    tokenizer, model = load_model(args.model)

    preds = infer_batch(prompts_built, tokenizer, model, args.batch_size)
    harm_preds, cat_preds = zip(*preds)

    df["harmful_pred"]  = harm_preds
    df["category_pred"] = cat_preds
    df["category_pred_label"] = [CATEGORY_LABELS.get(c, c) for c in cat_preds]

    out_path = args.output
    if out_path is None:
        stem = Path(args.input).stem
        out_path = str(Path(args.input).parent / f"{stem}_preds.csv")

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n结果保存至: {out_path}")
    print(f"  harmful:   {sum(1 for p in harm_preds if p == 'harmful'):,}")
    print(f"  unharmful: {sum(1 for p in harm_preds if p == 'unharmful'):,}")


if __name__ == "__main__":
    main()
