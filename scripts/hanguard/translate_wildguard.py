#!/usr/bin/env python3
"""
translate_wildguard.py

将 train_wildguard.parquet 中的英文 prompt 翻译为中文
使用 NLLB-200-3.3B，自动检测空闲显卡并行处理
输出: data/train_wildguard_zh.parquet（列：prompt, prompt_harm_label）

用法:
    python scripts/hanguard/translate_wildguard.py
    python scripts/hanguard/translate_wildguard.py --input data/train_wildguard.parquet --batch_size 64
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "/mnt/data1/zhouhanyu/models/nllb-200-3.3B"
INPUT_FILE  = "data/train_wildguard.parquet"
OUTPUT_FILE = "data/train_wildguard_zh.parquet"
CHECKPOINT_DIR = "data/translate_checkpoints"

BATCH_SIZE          = 32
MAX_SRC_LENGTH      = 512
MAX_NEW_TOKENS      = 512
FREE_MEM_THRESHOLD_GB = 30   # 空闲显存低于此值则跳过该卡
SRC_LANG = "eng_Latn"
TGT_LANG = "zho_Hans"


# ---------------------------------------------------------------------------
# GPU 检测
# ---------------------------------------------------------------------------

def get_free_gpus(threshold_gb: float = FREE_MEM_THRESHOLD_GB) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    free_gpus = []
    for line in result.stdout.strip().splitlines():
        idx_s, free_mb_s = line.strip().split(", ")
        if int(free_mb_s) / 1024 >= threshold_gb:
            free_gpus.append(int(idx_s))
    return free_gpus


# ---------------------------------------------------------------------------
# 单卡翻译 worker（在子进程中执行）
# ---------------------------------------------------------------------------

def translate_chunk(gpu_id: int, chunk_id: int, prompts: list,
                    labels: list, checkpoint_dir: str, batch_size: int,
                    model_path: str = MODEL_PATH) -> str:
    checkpoint_path = Path(checkpoint_dir) / f"chunk_{chunk_id:03d}.parquet"
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        if len(existing) == len(prompts):
            print(f"[GPU {gpu_id}] chunk {chunk_id} 已完成，跳过", flush=True)
            return str(checkpoint_path)

    # 必须在任何 CUDA 操作之前设置，spawn 子进程中 CUDA 尚未初始化
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] 加载模型...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.src_lang = SRC_LANG

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    model.eval()

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    translated = []
    total = len(prompts)
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SRC_LENGTH,
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
                num_beams=1,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)

        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        print(
            f"[GPU {gpu_id}] chunk {chunk_id}: {done}/{total}  "
            f"{speed:.0f} samples/min  ETA {eta/60:.1f} min",
            flush=True,
        )

    result_df = pd.DataFrame({"prompt": translated, "prompt_harm_label": labels})
    result_df.to_parquet(checkpoint_path, index=False)
    elapsed_total = time.time() - t0
    print(
        f"[GPU {gpu_id}] chunk {chunk_id} 完成：{total} 条，"
        f"耗时 {elapsed_total/60:.1f} min，保存至 {checkpoint_path}",
        flush=True,
    )
    return str(checkpoint_path)


def worker(args):
    return translate_chunk(*args)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=INPUT_FILE)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    parser.add_argument("--model",      default=MODEL_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold",  type=float, default=FREE_MEM_THRESHOLD_GB,
                        help="空闲显存阈值 (GB)，低于此值的卡不参与翻译")
    parser.add_argument("--gpus", type=str, default=None,
                        help="手动指定使用的 GPU，逗号分隔，如 0,1,2,4,5,7（覆盖自动检测）")
    args = parser.parse_args()

    model_path = args.model

    ckpt_dir = Path(args.input).stem + "_checkpoints"
    checkpoint_dir = str(Path(args.input).parent / ckpt_dir)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("读取数据...", flush=True)
    df = pd.read_parquet(args.input)[["prompt", "prompt_harm_label"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"总行数: {len(df)}（已随机打乱）", flush=True)

    if args.gpus is not None:
        free_gpus = [int(g) for g in args.gpus.split(",")]
        print(f"手动指定显卡: GPU {free_gpus}", flush=True)
    else:
        free_gpus = get_free_gpus(args.threshold)
        print(f"检测到空闲显卡（空闲显存 ≥ {args.threshold}GB）: GPU {free_gpus}", flush=True)

    if not free_gpus:
        raise RuntimeError("未找到空闲显卡，请降低 --threshold 或释放显存后重试。")

    n = len(free_gpus)
    chunks = np.array_split(df.reset_index(drop=True), n)

    worker_args = [
        (
            free_gpus[i],
            i,
            chunks[i]["prompt"].tolist(),
            chunks[i]["prompt_harm_label"].tolist(),
            checkpoint_dir,
            args.batch_size,
            model_path,
        )
        for i in range(n)
    ]

    print(f"\n启动 {n} 个子进程，每卡约 {len(chunks[0])} 条...\n", flush=True)
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n) as pool:
        checkpoint_paths = pool.map(worker, worker_args)

    print("\n合并结果...", flush=True)
    parts = [pd.read_parquet(p) for p in sorted(checkpoint_paths)]
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(args.output, index=False)

    elapsed = time.time() - t_start
    print(f"\n完成！总耗时 {elapsed/60:.1f} min")
    print(f"输出文件: {args.output}，共 {len(result)} 行")
    print("标签分布:")
    print(result["prompt_harm_label"].value_counts())


if __name__ == "__main__":
    main()
