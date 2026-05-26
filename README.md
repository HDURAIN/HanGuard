# Prompt Safety Evaluation

Evaluation scaffold for two safety classifiers:

- `meta-llama/Llama-Prompt-Guard-2-22M`: small request classifier, suitable for local or cloud CPU/GPU smoke tests.
- `allenai/wildguard`: larger generative safety classifier, intended for a cloud GPU server.

Both runners write the same output fields so the same `evaluate.py` script can be reused.

## Data Format

Put CSV files in `data/`. Required columns:

- `prompt`: user request text
- `label`: ground-truth harmful-request label, one of `yes`, `no`, `malicious`, `benign`, `1`, or `0`

Optional columns:

- `response`: assistant response text, used by WildGuard for `refusal` and `harmful_response`
- any metadata columns, such as `id`, `source`, or `category`

Prediction outputs include:

- `harmful_request`: `yes` or `no`
- `refusal`: `yes`, `no`, or `unknown`
- `harmful_response`: `yes`, `no`, or `unknown`
- `prompt_category`: zero-shot prompt category, one of the configured coarse safety categories

Prompt Guard only classifies the user request, so it writes `unknown` for `refusal` and `harmful_response`.
WildGuard additionally appends `prompt_category` using `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.

## Local Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
hf auth login
```

## Prompt Guard

Smoke test:

```powershell
python run_prompt_guard.py --input data/prompt_guard_demo.csv --output outputs/prompt_guard_demo_predictions.csv --batch-size 2
python evaluate.py --input outputs/prompt_guard_demo_predictions.csv --output outputs/prompt_guard_demo_metrics.json
```

Full dataset:

```powershell
python run_prompt_guard.py --input data/your_dataset.csv --output outputs/prompt_guard_predictions.csv --batch-size 16
python evaluate.py --input outputs/prompt_guard_predictions.csv --output outputs/prompt_guard_metrics.json
```

## Cloud Server Setup

Clone and install:

```bash
git clone git@github.com:HDURAIN/Prompt-Guard.git
cd Prompt-Guard
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cloud.txt
hf auth login
```

If SSH is not configured on the server, clone with HTTPS instead:

```bash
git clone https://github.com/HDURAIN/Prompt-Guard.git
```

## Cloud Smoke Tests

Prompt Guard:

```bash
python run_prompt_guard.py \
  --input data/prompt_guard_demo.csv \
  --output outputs/prompt_guard_cloud_smoke.csv \
  --batch-size 2 \
  --limit 5
```

WildGuard:

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_smoke.csv \
  --batch-size 1 \
  --limit 5
```

## WildGuard Full Run

```bash
python run_wildguard.py \
  --input data/your_dataset.csv \
  --output outputs/wildguard_predictions.csv \
  --batch-size 4
```

For smaller GPUs, reduce `--batch-size` to `1`. If memory is still insufficient, reduce `--max-length`.
The prompt category classifier runs before WildGuard and is released before WildGuard loads. You can force it to CPU with `--category-device cpu`.

## Evaluation

Prompt Guard and WildGuard harmful-request evaluation:

```bash
python evaluate.py --input outputs/prompt_guard_predictions.csv --target harmful_request --output outputs/prompt_guard_metrics.json
python evaluate.py --input outputs/wildguard_predictions.csv --target harmful_request --output outputs/wildguard_harmful_request_metrics.json
```

WildGuard can also evaluate response-level fields if your CSV has matching labels such as `refusal_label` or `harmful_response_label`:

```bash
python evaluate.py --input outputs/wildguard_predictions.csv --target refusal --output outputs/wildguard_refusal_metrics.json
python evaluate.py --input outputs/wildguard_predictions.csv --target harmful_response --output outputs/wildguard_harmful_response_metrics.json
```

## Notes

- Both model repositories require Hugging Face login and access/terms acceptance.
- Generated prediction and metric files are ignored by Git.
- Keep large datasets, model weights, and tokens out of the repository.
- Server deployment and smoke-test steps are in `SERVER_DEPLOY.md`.
