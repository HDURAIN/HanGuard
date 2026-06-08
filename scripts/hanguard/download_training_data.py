"""
下载 HanGuard 训练数据集到本地（来源：allenai/wildguardmix，需申请访问权限）。
用法：
    export HF_TOKEN=your_token
    python scripts/hanguard/download_training_data.py
"""
import os
from pathlib import Path
from datasets import load_dataset

HF_DATASET_ID = "allenai/wildguardmix"   # HuggingFace 数据集 ID（外部来源，不可更改）
HF_CONFIGS = [("train", "wildguardtrain"), ("test", "wildguardtest")]

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    raise SystemExit("请先设置环境变量：export HF_TOKEN=your_token")

SAVE_DIR = Path(__file__).parent.parent / "data" / "hanguard_training_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for split_name, config in HF_CONFIGS:
    print(f"下载 {split_name} split...")
    ds = load_dataset(HF_DATASET_ID, config, token=TOKEN, split=split_name)
    out = SAVE_DIR / f"{split_name}.parquet"
    ds.to_parquet(str(out))
    print(f"  {len(ds)} 条 → {out}")

print(f"\n完成，数据保存在 {SAVE_DIR}")
