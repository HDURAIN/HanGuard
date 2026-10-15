"""Verify this result snapshot with Python's standard library only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or external archive file: {name}")
    return path


def pointer(value, name):
    for part in name.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def close(actual, expected, name):
    if abs(float(actual) - float(expected)) > 1e-10:
        raise ValueError(f"Metric differs from archived source: {name}")


def verify():
    manifest = read_json(ROOT / "manifest.json")
    listed = {entry["path"] for entry in manifest["files"]}
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
    if actual != listed | {"manifest.json"}:
        raise ValueError("Archive file inventory differs from manifest")
    for entry in manifest["files"]:
        raw = artifact(entry["path"]).read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive hash/size mismatch: {entry['path']}")
    sources = read_json(ROOT / "source_manifest.json")["files"]
    for entry in sources:
        raw = artifact(entry["archived_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["archived_sha256"]:
            raise ValueError("Source copy differs from recorded snapshot")
        if entry["transformation"] == "verbatim" and entry["source_sha256"] != entry["archived_sha256"]:
            raise ValueError("Verbatim copy has different source and archived hashes")
    metric_rows = list(csv.DictReader((ROOT / "metrics_summary.csv").open(encoding="utf-8", newline="")))
    for row in metric_rows:
        document = read_json(artifact(row["source_artifact"]))
        close(row["accuracy"], pointer(document, row["source_json_pointer"]), row["arm"])
        close(row["accuracy_percent"], 100 * float(row["accuracy"]), row["arm"])
        if not 0 <= float(row["accuracy"]) <= 1 or row["status"] not in ("complete", "partial"):
            raise ValueError("Invalid accuracy or completion status")
    comparisons = list(csv.DictReader((ROOT / "comparisons.csv").open(encoding="utf-8", newline="")))
    for row in comparisons:
        document = read_json(artifact(row["source_artifact"]))
        pointer(document, row["source_json_pointer"])
        close(row["candidate_accuracy"], pointer(document, row["candidate_source_json_pointer"]), row["candidate"])
        close(row["baseline_accuracy"], pointer(document, row["baseline_source_json_pointer"]), row["baseline"])
        close(row["accuracy_delta_pp"], 100 * (float(row["candidate_accuracy"]) - float(row["baseline_accuracy"])), row["candidate"])
    for entry in read_json(ROOT / "summary_sources.json")["source_artifacts"]:
        if hashlib.sha256(artifact(entry["source_artifact"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Summary source identity differs")
    index = read_json(ROOT / "index.json")
    for study in index["studies"]:
        for name in study["artifacts"]:
            artifact(name)
    for link in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text(encoding="utf-8")):
        if not link.startswith(("http://", "https://", "#")):
            artifact(link.split("#", 1)[0])
    print(json.dumps(dict(verified=True, files=len(listed), source_artifacts=len(sources),
                         study_groups=len(index["studies"]), metric_rows=len(metric_rows),
                         comparison_rows=len(comparisons)), ensure_ascii=False))


if __name__ == "__main__":
    verify()
