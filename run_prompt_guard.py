import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-22M"
TEXT_COLUMN = "prompt"


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    if value in {"1", "malicious", "unsafe", "attack", "jailbreak", "injection"}:
        return "yes"
    if value in {"0", "benign", "safe", "normal"}:
        return "no"
    return value


def malicious_probability(logits: torch.Tensor, id2label: dict[int, str]) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)[0]
    labels = {idx: label.lower() for idx, label in id2label.items()}
    malicious_idx = next((idx for idx, label in labels.items() if "malicious" in label or "unsafe" in label), None)
    if malicious_idx is None:
        malicious_idx = 1 if probs.numel() == 2 else int(torch.argmax(probs).item())
    predicted_idx = int(torch.argmax(probs).item())
    predicted_label = labels.get(predicted_idx, str(predicted_idx))
    if "malicious" in predicted_label or "unsafe" in predicted_label:
        predicted_label = "malicious"
    elif "benign" in predicted_label or "safe" in predicted_label:
        predicted_label = "benign"
    elif predicted_idx == malicious_idx:
        predicted_label = "malicious"
    else:
        predicted_label = "benign"
    return predicted_label, float(probs[malicious_idx].item())


def run(
    input_path: Path,
    output_path: Path,
    model_id: str,
    batch_size: int,
    max_length: int,
    token: str | None,
    limit: int | None,
) -> None:
    data = pd.read_csv(input_path)
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Input CSV must contain a '{TEXT_COLUMN}' column.")
    if limit is not None:
        data = data.head(limit).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token).to(device)
    except OSError as exc:
        message = (
            f"Could not load '{model_id}'. If this is a gated Hugging Face model, "
            "request access on the model page and run `hf auth login`, "
            "or pass a token with `--token`."
        )
        raise RuntimeError(message) from exc
    model.eval()

    harmful_requests: list[str] = []
    scores: list[float] = []
    prompts = data[TEXT_COLUMN].fillna("").astype(str).tolist()

    for start in tqdm(range(0, len(prompts), batch_size), desc="Prompt Guard"):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        for row_logits in logits:
            label, score = malicious_probability(row_logits.unsqueeze(0), model.config.id2label)
            harmful_requests.append("yes" if label == "malicious" else "no")
            scores.append(score)

    result = data.copy()
    if "label" in result.columns:
        result["label"] = result["label"].map(normalize_label)
    result["harmful_request"] = harmful_requests
    result["refusal"] = "unknown"
    result["harmful_response"] = "unknown"
    result["harmful_request_score"] = scores
    result["model"] = model_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama Prompt Guard 2 22M on a CSV dataset.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with a prompt column.")
    parser.add_argument("--output", type=Path, default=Path("outputs/prompt_guard_predictions.csv"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated models.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.model_id, args.batch_size, args.max_length, args.token, args.limit)
