"""
HanGuard v5：二分类 + 六分类 + 防越狱，基于 Qwen2.5-7B-Instruct QLoRA。

与 v4 的区别：
  - 输出同时包含 Harmful request 和 Harm category 两个字段
  - 使用 final_train/val.parquet（含全部增强数据和六分类标签）
  - 支持 sqrt 阻尼逆频率类别权重（class_weights.json），真正作用于 loss
  - DataCollatorForCompletionOnlyLM 只对 assistant 回复部分计算 loss
  - eval_dataset 训练中评估验证集 loss
  - 支持 torchrun 多卡 DDP（通过 LOCAL_RANK 自动检测）

使用方法：
  # 单卡
  CUDA_VISIBLE_DEVICES=0 python scripts/hanguard/train_v5.py

  # 多卡 DDP（例：6 卡）
  CUDA_VISIBLE_DEVICES=0,1,3,4,5,7 torchrun --nproc_per_node=6 \\
      scripts/hanguard/train_v5.py --output_dir outputs/hanguard_v5
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM

LOCAL_MODEL   = "/mnt/data1/zhouhanyu/models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TRAIN = "data/final_train.parquet"
DEFAULT_VAL   = "data/final_val.parquet"
DEFAULT_OUT   = "outputs/hanguard_v5"
DEFAULT_WEIGHTS = "data/class_weights.json"
MAX_LENGTH    = 512

# ── 系统提示：明确要求输出双标签 ────────────────────────────────────
SYSTEM_MSG = (
    "You are a content safety classifier for Chinese AI systems. "
    "Given a user request, output two labels:\n"
    "1. Whether the request is harmful (harmful/unharmful)\n"
    "2. The harm category (0=安全, 1=违反社会主义核心价值观的内容, "
    "2=歧视性内容, 3=商业违法违规, 4=侵犯他人合法权益, "
    "5=无法满足特定服务类型的安全需求)"
)

PROMPT_ONLY_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "Human user:\n{prompt}\n\n---\n\nAnswers:<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Harmful request: {prompt_harm_label}\n"
    "Harm category: {category_label}<|im_end|>"
)

# DataCollatorForCompletionOnlyLM 用此字符串定位 assistant 回复起点
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def format_sample(sample: dict) -> str:
    return PROMPT_ONLY_TEMPLATE.format(
        system=SYSTEM_MSG,
        prompt=sample["prompt"],
        prompt_harm_label=sample["prompt_harm_label"],
        category_label=sample["category_label"],
    )


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["prompt", "prompt_harm_label", "category_label", "category_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据集缺少列: {missing}")
    df = df.dropna(subset=["prompt", "prompt_harm_label", "category_label"])
    df["category_id"] = df["category_id"].fillna("0").astype(str)
    return df.reset_index(drop=True)


def build_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    class_weights: dict | None,
) -> Dataset:
    """格式化 → tokenize → 打包成 HF Dataset（含 weight 列）。"""
    records = df.to_dict("records")
    texts = [format_sample(r) for r in records]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

    if class_weights is not None:
        data["weight"] = [
            float(class_weights.get(str(r.get("category_id", "0")), 1.0))
            for r in records
        ]

    return Dataset.from_dict(data)


class WeightedCollator:
    """从 batch 中提取 weight，交由 base_collator 处理其余字段。"""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        weights = torch.tensor(
            [float(f.pop("weight", 1.0)) for f in features],
            dtype=torch.float32,
        )
        batch = self.base_collator(features)
        batch["sample_weights"] = weights
        return batch


class WeightedTrainer(Trainer):
    """对每条样本按 category weight 加权后再平均 loss。"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        outputs = model(**inputs)

        if sample_weights is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits          # [B, T, V]
        labels = inputs["labels"]        # [B, T]

        # 标准 causal LM: shift by 1
        shift_logits = logits[..., :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()         # [B, T-1]

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(logits.size(0), -1)                          # [B, T-1]

        # 只统计非 padding（-100）位置
        mask = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)

        loss = (per_sample_loss * sample_weights.to(per_sample_loss.device)).mean()

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       type=str, default=DEFAULT_TRAIN)
    parser.add_argument("--val",         type=str, default=DEFAULT_VAL)
    parser.add_argument("--output_dir",  type=str, default=DEFAULT_OUT)
    parser.add_argument("--model",       type=str, default=LOCAL_MODEL)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--grad_accum",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--weights",     type=str, default=DEFAULT_WEIGHTS,
                        help="class_weights.json 路径，留空则不使用权重")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== HanGuard v5：二分类 + 六分类 + 防越狱 ===")
    print(f"基座模型:   {args.model}")
    print(f"训练数据:   {args.train}")
    print(f"验证数据:   {args.val}")
    print(f"输出目录:   {output_dir}")
    print()

    # ── 加载数据 ────────────────────────────────────────────────────
    print("加载训练数据...")
    train_df = load_data(args.train)
    val_df   = load_data(args.val)

    print(f"训练集: {len(train_df):,} 条")
    for label, cnt in train_df["category_label"].value_counts().items():
        print(f"  {label}: {cnt:,}")
    print(f"验证集: {len(val_df):,} 条")
    print()

    # ── 类别权重（可选）────────────────────────────────────────────
    class_weights = None
    if args.weights and Path(args.weights).exists():
        with open(args.weights) as f:
            class_weights = json.load(f)
        print("类别权重:", class_weights)
        print()

    # ── Tokenizer ───────────────────────────────────────────────────
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.padding_side = "right"

    # ── 模型 ────────────────────────────────────────────────────────
    # DDP：每个进程只使用自己的 GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("加载基座模型（4-bit QLoRA）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BNBCONFIG,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    print()

    # ── 数据集预 tokenize ───────────────────────────────────────────
    print("预处理训练集...")
    train_data = build_hf_dataset(train_df, tokenizer, MAX_LENGTH, class_weights)
    print("预处理验证集...")
    val_data   = build_hf_dataset(val_df,   tokenizer, MAX_LENGTH, None)  # val 不加权
    print()

    # ── Collator：只对 assistant 部分计算 loss ──────────────────────
    base_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )
    if class_weights is not None:
        data_collator = WeightedCollator(base_collator)
        trainer_cls = WeightedTrainer
    else:
        data_collator = base_collator
        trainer_cls = Trainer

    # ── 训练参数 ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,   # 保留 weight 列直到 collator 消费它
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("开始训练...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n训练完成，HanGuard v5 已保存到: {output_dir}")


if __name__ == "__main__":
    main()
