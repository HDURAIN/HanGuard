#!/usr/bin/env bash
# 多卡并行推理 + 合并评估
# 用法: bash scripts/eval_parallel.sh [batch_size]
set -euo pipefail

BATCH=${1:-32}
MODEL="outputs/hanguard_v5"
TEST="data/final_test.parquet"
OUTDIR="outputs/eval_v5"
PYTHON="/mnt/data1/zhouhanyu/miniconda3/envs/hanguard/bin/python"
GPUS=(1 3 4 5)
TOTAL=9343
CHUNK=2336

mkdir -p "$OUTDIR"

echo "=== 并行推理：${#GPUS[@]} 卡 × ${CHUNK} 条，batch_size=${BATCH} ==="

pids=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SKIP=$((i * CHUNK))
    LIMIT=$CHUNK
    OUT="${OUTDIR}/preds_gpu${GPU}.csv"
    LOG="${OUTDIR}/infer_gpu${GPU}.log"

    echo "GPU ${GPU}: skip=${SKIP} limit=${LIMIT} -> ${OUT}"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer.py \
        --input      "$TEST" \
        --model      "$MODEL" \
        --output     "$OUT" \
        --batch_size "$BATCH" \
        --skip       "$SKIP" \
        --limit      "$LIMIT" \
        > "$LOG" 2>&1 &
    pids+=($!)
done

echo "等待所有推理进程完成（PID: ${pids[*]}）..."
for pid in "${pids[@]}"; do
    wait "$pid" && echo "PID $pid 完成" || { echo "PID $pid 失败"; exit 1; }
done

echo ""
echo "=== 合并预测结果 ==="
MERGED="${OUTDIR}/predictions.csv"
$PYTHON - <<'PYEOF'
import pandas as pd, glob, os

outdir = "outputs/eval_v5"
files  = sorted(glob.glob(f"{outdir}/preds_gpu*.csv"))
print(f"合并 {len(files)} 个分片: {files}")

parts = [pd.read_csv(f) for f in files]
df    = pd.concat(parts, ignore_index=True)
out   = f"{outdir}/predictions.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"合并完成: {len(df):,} 条 -> {out}")
PYEOF

echo ""
echo "=== 计算评估指标 ==="
$PYTHON evaluate.py \
    --preds  "$MERGED" \
    --output "${OUTDIR}/report.txt"
