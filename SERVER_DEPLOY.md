# WildGuard Server Deployment

This project now runs two models in one pipeline:

- `allenai/wildguard`: keeps the original WildGuard outputs
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`: adds `prompt_category`

The final CSV keeps the original WildGuard fields and appends one extra column:

- `prompt_category`

## 1. Recommended server

- Ubuntu 20.04 or 22.04
- Python 3.11
- CUDA-capable GPU
- At least 24 GB GPU memory is safer for WildGuard

If GPU memory is tight, keep the prompt classifier on CPU and run WildGuard with `--batch-size 1`.

## 2. Clone and install

```bash
git clone <your-repo-url>
cd fxproject
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cloud.txt
```

## 3. Hugging Face authentication

If the server can access the Hugging Face website directly:

```bash
hf auth login
```

Or use an environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

`run_wildguard.py` will automatically use `HF_TOKEN` if `--token` is not passed.

## 4. Prepare input data

Input CSV must include:

- `prompt`

Optional but recommended:

- `response`
- `label`
- other metadata columns such as `id`

## 5. Smoke test

Run a small test first:

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_smoke.csv \
  --batch-size 1 \
  --category-batch-size 8 \
  --category-device cpu \
  --limit 5
```

What this does:

- runs the zero-shot prompt classifier first
- frees that classifier
- loads WildGuard and produces the original WildGuard outputs
- appends `prompt_category`

## 6. Full run

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_predictions.csv \
  --batch-size 4 \
  --category-batch-size 16 \
  --category-device cpu
```

If you have enough GPU memory, you can move the classifier to GPU:

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_predictions.csv \
  --batch-size 4 \
  --category-batch-size 16 \
  --category-device cuda
```

## 7. Output columns

The output CSV contains:

- original input columns
- `harmful_request`
- `refusal`
- `harmful_response`
- `raw_output`
- `model`
- `prompt_category`

## 8. Evaluation

WildGuard metrics:

```bash
python evaluate.py \
  --input outputs/wildguard_predictions.csv \
  --target harmful_request \
  --output outputs/wildguard_harmful_request_metrics.json
```

Optional response-level metrics:

```bash
python evaluate.py \
  --input outputs/wildguard_predictions.csv \
  --target refusal \
  --output outputs/wildguard_refusal_metrics.json

python evaluate.py \
  --input outputs/wildguard_predictions.csv \
  --target harmful_response \
  --output outputs/wildguard_harmful_response_metrics.json
```

## 9. Common issues

- `ModuleNotFoundError`: dependencies are not installed in the active virtual environment
- `Could not load ...`: Hugging Face auth, network, or model access issue
- CUDA OOM: reduce `--batch-size`, reduce `--max-length`, or set `--category-device cpu`
