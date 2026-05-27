import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_LABEL_COLUMN = "category_label"
DEFAULT_PREDICTION_COLUMN = "prompt_category"
HARMFUL_LABEL_COLUMN = "label"
HARMFUL_PREDICTION_COLUMN = "harmful_request"
SAFE_CATEGORY = "安全"


def normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def is_harmful(value: str | None) -> bool:
    return normalize_text(value).lower() in {"1", "yes", "malicious", "unsafe", "attack", "jailbreak", "injection", "harmful"}


def read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def require_columns(rows: list[dict[str, str]], columns: list[str]) -> None:
    if not rows:
        raise ValueError("Input CSV is empty.")
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def accuracy(rows: list[dict[str, str]], label_column: str, prediction_column: str) -> float | None:
    if not rows:
        return None
    correct = sum(
        normalize_text(row[label_column]) == normalize_text(row[prediction_column])
        for row in rows
    )
    return correct / len(rows)


def build_confusion_matrix(
    rows: list[dict[str, str]],
    label_column: str,
    prediction_column: str,
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        actual = normalize_text(row[label_column])
        predicted = normalize_text(row[prediction_column])
        matrix.setdefault(actual, {})
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def per_category_metrics(
    rows: list[dict[str, str]],
    label_column: str,
    prediction_column: str,
) -> dict[str, dict[str, float | int]]:
    labels = sorted(
        {
            normalize_text(row[label_column])
            for row in rows
        }
        | {
            normalize_text(row[prediction_column])
            for row in rows
        }
    )
    metrics: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(
            normalize_text(row[label_column]) == label
            and normalize_text(row[prediction_column]) == label
            for row in rows
        )
        fp = sum(
            normalize_text(row[label_column]) != label
            and normalize_text(row[prediction_column]) == label
            for row in rows
        )
        fn = sum(
            normalize_text(row[label_column]) == label
            and normalize_text(row[prediction_column]) != label
            for row in rows
        )
        support = sum(normalize_text(row[label_column]) == label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def evaluate_category(
    input_path: Path,
    output_path: Path,
    label_column: str,
    prediction_column: str,
) -> dict:
    rows = read_rows(input_path)
    require_columns(
        rows,
        [
            label_column,
            prediction_column,
            HARMFUL_LABEL_COLUMN,
            HARMFUL_PREDICTION_COLUMN,
        ],
    )

    true_harmful_rows = [
        row for row in rows if is_harmful(row[HARMFUL_LABEL_COLUMN])
    ]
    detected_harmful_rows = [
        row
        for row in rows
        if is_harmful(row[HARMFUL_LABEL_COLUMN])
        and is_harmful(row[HARMFUL_PREDICTION_COLUMN])
    ]
    safe_rows = [
        row for row in rows if not is_harmful(row[HARMFUL_LABEL_COLUMN])
    ]

    actual_counts = Counter(normalize_text(row[label_column]) for row in rows)
    predicted_counts = Counter(normalize_text(row[prediction_column]) for row in rows)
    correct_rows = [
        row
        for row in rows
        if normalize_text(row[label_column]) == normalize_text(row[prediction_column])
    ]

    metrics = {
        "target": prediction_column,
        "label_column": label_column,
        "samples": len(rows),
        "correct": len(correct_rows),
        "accuracy": accuracy(rows, label_column, prediction_column),
        "true_harmful_samples": len(true_harmful_rows),
        "true_harmful_category_accuracy": accuracy(true_harmful_rows, label_column, prediction_column),
        "detected_harmful_samples": len(detected_harmful_rows),
        "detected_harmful_category_accuracy": accuracy(detected_harmful_rows, label_column, prediction_column),
        "safe_samples": len(safe_rows),
        "safe_category_accuracy": accuracy(safe_rows, label_column, prediction_column),
        "actual_category_counts": dict(actual_counts),
        "predicted_category_counts": dict(predicted_counts),
        "confusion_matrix": build_confusion_matrix(rows, label_column, prediction_column),
        "per_category": per_category_metrics(rows, label_column, prediction_column),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coarse category predictions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/category_metrics.json"))
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--prediction-column", default=DEFAULT_PREDICTION_COLUMN)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate_category(
                args.input,
                args.output,
                args.label_column,
                args.prediction_column,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
