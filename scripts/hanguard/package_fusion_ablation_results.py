#!/usr/bin/env python3
"""Package small fusion study evidence without moving registered source files.

Only allowlisted numerical prediction columns are exported. Corpus text, model
weights, feature arrays and operational scratch files are never copied.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ARCHIVE = Path("experiment_results/20261019_fusion_ablation")
BASE = Path("outputs/hanguard_frozen_base_full_readouts_20261013_v5")
LORA = Path("outputs/hanguard_selected122_lora_20261018")
SUBSET = Path("data/three_source_wildguard122_subset_20261018")
ABLATION = Path("outputs/hanguard_nolora_ablation_20261019")
REFERENCE = Path("outputs/hanguard_nolora_reference_analysis_20261019")
METHODS = ("last", "best_single", "repeat_best", "fusion_fixed", "fusion_fixed_best_anchor")
COMPONENT_ARMS = ("anchor_replay", "uniform_weights", "no_anchor", "shared_projection", "top4", "two_layers", "mean_pool")
SCOPES = (("full_test", "full_test_8350"), ("selected_subset", "selected_test_8228"))
SAFE_METADATA = {
    "base_id", "sample_id", "group_id", "source", "category_id", "type_label_mask",
    "split", "label", "prompt_tokens", "seed", "method", "arm", "fold",
}
SAFE_NUMERIC = re.compile(r"(?:probability|threshold|prediction|logit)_[A-Za-z0-9_]+\Z")
JSON_NAMES = {
    "protocol.json", "scientific_protocol.json", "protocol_registration.json",
    "registration.json", "selection.json", "selection_lock.json", "results.json",
    "complete.json", "test_access.json", "adapter_lineage.json", "state.json",
    "manifest.json", "release_qa.json", "filter_policy.json", "split_policy.json",
    "old_model_subset_metrics.json", "quantitative_summary.json", "review_summary.json",
    "decision_record.json", "selection_arithmetic.json", "cpu_validation.json",
    "queue_protocol.json", "queue_registration.json", "queue_state.json",
}
EXCLUDED_DIRS = {
    "features", "feature_cache", "cache", "checkpoints", "runtime_scratch",
    "queue_control", "workers", "recovery_001", "__pycache__",
}
CSV_FIELDS = (
    "study", "evaluation_scope", "selection_policy", "no_lora", "method", "seed",
    "threshold_policy", "rows", "correct", "accuracy_percent", "f1_percent",
    "recall_percent", "precision_percent", "fpr_percent", "auroc_percent",
    "gain_vs_best_single_pp", "source",
)
CORE_SOURCES = (
    "package_fusion_ablation_results.py", "analyze_fusion_reference_results.py",
    "nolora_fusion_ablation.py", "nolora_ablation_queue.py", "selected_subset_lora_study.py",
    "frozen_base_readout_heads.py", "frozen_base_resumable_heads.py", "repaired_heads.py",
    "ab_frozen_readout_diagnostics.py", "frozen_lora_layer_diagnostics.py",
    "fragment_clean_lora_study.py", "ab_readout_parallel_queue.py", "ab_diagnostic_queue.py",
)
PRESENTATION_FIELDS = {
    "component_ablation.csv": ("evaluation_scope", "method", "status", "accuracy_percent", "gain_vs_anchor_replay_pp",
                               "parameters_total", "parameters_trainable", "layers", "anchor_layer", "training_device",
                               "common_epoch_budget", "source"),
    "seed_summary.csv": ("evaluation_scope", "method", "status", "completed_seeds", "planned_seeds",
                         "mean_accuracy_percent", "sample_std_accuracy_pp", "source"),
    "seed_gains.csv": ("evaluation_scope", "seed", "status", "best_single_accuracy_percent", "anchor_accuracy_percent",
                       "anchor_gain_vs_best_single_pp", "source"),
    "ablation_coverage.csv": ("ablation", "status", "evaluation_scope", "source", "note"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_repo_source(root: Path, candidate: Path) -> Path | None:
    """Allow only small Python sources in root or source/test directories."""
    path = candidate if candidate.is_absolute() else root / candidate
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if (path.suffix != ".py" or not path.is_file() or path.stat().st_size >= 1024 * 1024
            or (len(relative.parts) > 1 and relative.parts[0] not in {"scripts", "tests", "hanguard"})):
        return None
    return path


def bound_source_paths(root: Path, value: Any) -> dict[Path, set[str]]:
    sources: dict[Path, set[str]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"code_sha256", "source_sha256", "selection_source_sha256", "bindings"} and isinstance(child, dict):
                for filename, digest in child.items():
                    if isinstance(filename, str) and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                        path = safe_repo_source(root, Path(filename))
                        if path is not None:
                            sources.setdefault(path, set()).add(digest)
            for path, digests in bound_source_paths(root, child).items():
                sources.setdefault(path, set()).update(digests)
    elif isinstance(value, list):
        for child in value:
            for path, digests in bound_source_paths(root, child).items():
                sources.setdefault(path, set()).update(digests)
    return sources


def source_dependency_closure(root: Path, seeds: list[Path]) -> list[Path]:
    """Resolve project imports statically; never execute a training import."""
    pending = [path for item in seeds if (path := safe_repo_source(root, item)) is not None]
    seen: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent_parts = list(source.relative_to(root).with_suffix("").parts[:-1])
                    trim = node.level - 1
                    parent_parts = parent_parts[:-trim] if trim else parent_parts
                    base = ".".join(parent_parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                modules.append(base)
                modules.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        for module in modules:
            if not module or any(not part.isidentifier() for part in module.split(".")):
                continue
            relative = Path(*module.split("."))
            candidates = [root / relative.with_suffix(".py"), root / relative / "__init__.py"]
            if "." not in module:
                candidates.extend([source.parent / relative.with_suffix(".py"), source.parent / relative / "__init__.py"])
            for candidate in candidates:
                path = safe_repo_source(root, candidate)
                if path is not None and path not in seen:
                    pending.append(path)
    return sorted(seen)


def safe_prediction_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name in SAFE_METADATA or SAFE_NUMERIC.fullmatch(name)]


def export_predictions(source: Path, target: Path, excluded: set[str] | None = None) -> dict[str, Any]:
    """Stream CSV and strip all text-bearing fields, including unseen columns."""
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(newline="", encoding="utf-8") as src, target.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        columns = safe_prediction_columns(reader.fieldnames or [])
        if "base_id" not in columns or "label" not in columns:
            raise ValueError(f"Prediction file lacks base_id/label: {source}")
        writer = csv.DictWriter(dst, fieldnames=columns)
        writer.writeheader()
        seen: set[str] = set()
        for row in reader:
            identity = row["base_id"]
            if excluded and identity in excluded:
                continue
            if identity in seen:
                raise ValueError(f"Duplicate prediction base_id in {source}: {identity}")
            seen.add(identity)
            for key in columns:
                if SAFE_NUMERIC.fullmatch(key):
                    number = float(row[key])
                    if not math.isfinite(number):
                        raise ValueError(f"Non-finite prediction value in {source}: {key}")
            writer.writerow({key: row[key] for key in columns})
            count += 1
    return {"rows": count, "columns": columns, "transform": "allowlisted_metadata_and_predictions_only"}


def prediction_metrics(path: Path) -> dict[str, dict[str, float | int]]:
    counts = {method: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for method in METHODS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("Labels must be binary")
            for method in METHODS:
                predicted = int(row[f"prediction_{method}"])
                if predicted not in (0, 1):
                    raise ValueError("Predictions must be binary")
                key = ("tp" if label else "fp") if predicted else ("fn" if label else "tn")
                counts[method][key] += 1
    metrics: dict[str, dict[str, float | int]] = {}
    for method, values in counts.items():
        tp, tn, fp, fn = (values[key] for key in ("tp", "tn", "fp", "fn"))
        n = tp + tn + fp + fn
        if not n:
            raise ValueError(f"No predictions in {path}")
        metrics[method] = {
            **values, "n": n, "accuracy": (tp + tn) / n,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
        }
    return metrics


def result_rows(study: str, scope: str, policy: str, no_lora: bool,
                result: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for block, threshold in (("metrics", "validation_selected"), ("fixed_05_metrics", "fixed_0.5")):
        metrics = result.get(block, {})
        if not isinstance(metrics, dict):
            continue
        baseline = metrics.get("best_single", {}).get("accuracy")
        for method, item in metrics.items():
            if not isinstance(item, dict) or "accuracy" not in item:
                continue
            row: dict[str, Any] = {
                "study": study, "evaluation_scope": scope, "selection_policy": policy,
                "no_lora": str(no_lora).lower(), "method": method,
                "seed": result.get("seed", 42), "threshold_policy": threshold,
                "rows": item.get("n", result.get("rows", "")), "source": source,
            }
            if all(key in item for key in ("tp", "tn")):
                row["correct"] = item["tp"] + item["tn"]
            for key in ("accuracy", "f1", "recall", "precision", "fpr", "auroc"):
                if item.get(key) is not None:
                    row[f"{key}_percent"] = float(item[key]) * 100
            if baseline is not None:
                row["gain_vs_best_single_pp"] = (float(item["accuracy"]) - float(baseline)) * 100
            rows.append(row)
    return rows


def optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def completed_metrics(result: dict[str, Any], block: str) -> dict[str, Any]:
    """A plan, head selection, or unfinished result is never a test score."""
    view = result.get(block, {})
    return view.get("metrics", {}) if result.get("complete") is True and isinstance(view, dict) else {}


def presentation_tables(root: Path, subset_metrics: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in PRESENTATION_FIELDS}
    study = root / ABLATION
    aggregate = optional_json(study / "results.json")
    aggregate_tasks = aggregate.get("tasks", {})
    protocol = optional_json(study / "protocol.json")

    def task_result(name: str, folder: str) -> tuple[dict[str, Any], str]:
        path = ABLATION / folder / "analysis/results.json"
        direct = optional_json(root / path)
        if direct.get("complete") is True:
            return direct, str(path)
        archived = aggregate_tasks.get(name, {}) if isinstance(aggregate_tasks, dict) else {}
        if isinstance(archived, dict) and archived.get("complete") is True:
            return archived, f"{ABLATION}/results.json#tasks.{name}"
        return {}, str(path)

    seeds = {}
    for seed, task, folder in ((42, "seed42_imported", "seed_42_imported"), (43, "seed43", "seed_43"), (44, "seed44", "seed_44")):
        seeds[seed] = task_result(task, folder)
    # Before the queue writes its aggregate, the existing seed 42 fixed-model
    # predictions already provide exactly these two registered evaluation views.
    if not seeds[42][0]:
        full = optional_json(root / BASE / "analysis/results.json")
        if full.get("complete") is True:
            views: dict[str, Any] = {"complete": True, "full_test": full}
            if subset_metrics is not None:
                views["selected_subset"] = subset_metrics
            seeds[42] = views, f"{BASE}/analysis/results.json;{SUBSET}/old_model_subset_metrics.json"

    for block, scope in SCOPES:
        for seed in (42, 43, 44):
            result, source = seeds[seed]
            metrics = completed_metrics(result, block)
            best = metrics.get("best_single", {}).get("accuracy")
            anchor = metrics.get("fusion_fixed_best_anchor", {}).get("accuracy")
            done = best is not None and anchor is not None
            row = {"evaluation_scope": scope, "seed": seed, "status": "completed" if done else "pending", "source": source}
            if done:
                row.update(best_single_accuracy_percent=float(best) * 100, anchor_accuracy_percent=float(anchor) * 100,
                           anchor_gain_vs_best_single_pp=(float(anchor) - float(best)) * 100)
            tables["seed_gains.csv"].append(row)
        for method in METHODS:
            values, completed = [], []
            for seed in (42, 43, 44):
                item = completed_metrics(seeds[seed][0], block).get(method, {})
                if item.get("accuracy") is not None:
                    values.append(float(item["accuracy"]) * 100)
                    completed.append(seed)
            done = len(completed) == 3
            row = {"evaluation_scope": scope, "method": method, "status": "completed" if done else "pending",
                   "completed_seeds": "/".join(map(str, completed)), "planned_seeds": "42/43/44",
                   "source": ";".join(seeds[seed][1] for seed in completed)}
            if done:
                row.update(mean_accuracy_percent=statistics.mean(values), sample_std_accuracy_pp=statistics.stdev(values))
            tables["seed_summary.csv"].append(row)

    components, component_source = task_result("components", "components")
    selection = optional_json(study / "components/heads/selection.json")
    specs = components.get("specs", selection.get("specs", protocol.get("component_specs", {})))
    parameters = components.get("parameters", selection.get("parameters", {}))
    execution = components.get("execution", selection.get("execution", optional_json(study / "components/execution.json")))
    for block, scope in SCOPES:
        metrics = completed_metrics(components, block)
        reference = metrics.get("anchor_replay", {}).get("accuracy")
        for arm in COMPONENT_ARMS:
            accuracy = metrics.get(arm, {}).get("accuracy")
            done = accuracy is not None and reference is not None
            spec, count = specs.get(arm, {}), parameters.get(arm, {})
            if isinstance(count, int):
                count = {"total": count, "trainable": count}
            row = {"evaluation_scope": scope, "method": arm, "status": "completed" if done else "pending",
                   "layers": "/".join(map(str, spec.get("layers", []))),
                   "anchor_layer": "none" if arm in {"no_anchor", "mean_pool"} else spec.get("anchor", ""),
                   "parameters_total": count.get("total", ""), "parameters_trainable": count.get("trainable", ""),
                   "training_device": execution.get("training_device", ""),
                   "common_epoch_budget": components.get("final_common_epoch_budget", selection.get("final_common_epoch_budget", "")),
                   "source": component_source}
            if done:
                row.update(accuracy_percent=float(accuracy) * 100,
                           gain_vs_anchor_replay_pp=(float(accuracy) - float(reference)) * 100)
            tables["component_ablation.csv"].append(row)

    coverage = tables["ablation_coverage.csv"]

    def add_coverage(name: str, done: bool, source: str, note: str, scope: str = "full_test_8350;selected_test_8228") -> None:
        coverage.append({"ablation": name, "status": "completed" if done else "pending", "evaluation_scope": scope,
                         "source": source, "note": note})

    for method in METHODS:
        done = all(method in completed_metrics(seeds[42][0], block) for block, _ in SCOPES)
        add_coverage(f"core_{method}", done, seeds[42][1], "seed 42；五组同样本、验证集选择检查点与阈值")
    reference = optional_json(root / REFERENCE / "main_results.json")
    for name, filename, note in (
        ("threshold_0.5", "threshold_sensitivity.csv", "固定0.5和验证集阈值；不是在测试集重新选阈值"),
        ("capacity_match", "metrics.csv", "repeat_best与fusion头的参数量/拓扑对照"),
        ("by_source", "source_metrics.csv", "三个来源分别报告；子集范围沿用固定ID"),
        ("group_bootstrap", "error_complementarity.csv", "组重采样条件区间；不校正事后子集选择偏差"),
        ("all_layer_validation_probe", "layer_probe.csv", "32层独立MLP的验证BCE与最佳轮次"),
    ):
        add_coverage(name, reference.get("complete") is True and (root / REFERENCE / filename).is_file(),
                     str(REFERENCE / filename), note)
    lora = optional_json(root / LORA / "readouts/analysis/results.json")
    add_coverage("same_frozen_lora", lora.get("complete") is True,
                 str(LORA / "readouts/analysis/results.json"), "同一新LoRA适配器冻结后五臂对照，与无LoRA结果另列", "selected_test_8228")
    for seed in (43, 44):
        done = all(all(method in completed_metrics(seeds[seed][0], block) for method in METHODS) for block, _ in SCOPES)
        add_coverage(f"head_seed_{seed}", done, seeds[seed][1], "核心五臂重新训练；各自验证选层、检查点和阈值")
    for arm in COMPONENT_ARMS:
        rows = [row for row in tables["component_ablation.csv"] if row["method"] == arm]
        add_coverage(f"component_{arm}", all(row["status"] == "completed" for row in rows), component_source,
                     "同设备、同预算完整结构基线" if arm == "anchor_replay" else "以同设备同预算anchor_replay为组件基线")
    return tables


def presentation_readme(tables: dict[str, list[dict[str, Any]]]) -> str:
    def number(value: Any, signed: bool = False) -> str:
        return "pending" if value is None or value == "" else format(float(value), "+.4f" if signed else ".4f")

    lines = ["## 组件消融", "", "相对增益按“该消融准确率 − 同设备同预算完整结构 `anchor_replay` 准确率”计算；负值表示移除该组件后下降。旧 GPU seed 42 结果不作为 CPU 组件消融的直接基线。层组可在训练前登记，准确率只有完成测试评估后才显示。", ""]
    for _, scope in SCOPES:
        lines.extend([f"### {scope}", "", "| 方法 | 状态 | 准确率 % | 相对 replay 增益/百分点 | 参数量（总/可训练） | 层组 | 锚点层 | 设备 |", "|---|---|---:|---:|---:|---|---|---|"])
        for row in tables["component_ablation.csv"]:
            if row["evaluation_scope"] != scope:
                continue
            total, trainable = row.get("parameters_total"), row.get("parameters_trainable")
            params = f"{int(total):,}/{int(trainable):,}" if total != "" and trainable != "" else "pending"
            lines.append(f"| `{row['method']}` | {row['status']} | {number(row.get('accuracy_percent'))} | {number(row.get('gain_vs_anchor_replay_pp'), True)} | {params} | {row['layers'] or 'pending'} | {row['anchor_layer'] or 'pending'} | {row['training_device'] or 'pending'} |")
        lines.append("")
    lines.extend(["## 三个 head seed 的核心五臂", "", "仅 seeds 42、43、44 全部完成后显示三种子均值 ± 样本标准差；seed 42 复用既有模型。缺失种子不以计划结果填入均值。", "", "| 范围 | 方法 | 状态 | 已完成 seeds | 准确率均值 ± SD / % |", "|---|---|---|---|---:|"])
    for row in tables["seed_summary.csv"]:
        score = f"{number(row['mean_accuracy_percent'])} ± {number(row['sample_std_accuracy_pp'])}" if row['status'] == 'completed' else "pending"
        completed = len(row['completed_seeds'].split('/')) if row['completed_seeds'] else 0
        lines.append(f"| {row['evaluation_scope']} | `{row['method']}` | {row['status']} | {row['completed_seeds'] or 'none'}（{completed}/3） | {score} |")
    lines.extend(["", "### 逐 seed 锚点融合主增益", "", "| 范围 | seed | 状态 | 最佳单层准确率 % | 锚点融合准确率 % | 融合增益/百分点 |", "|---|---:|---|---:|---:|---:|"])
    for row in tables["seed_gains.csv"]:
        lines.append(f"| {row['evaluation_scope']} | {row['seed']} | {row['status']} | {number(row.get('best_single_accuracy_percent'))} | {number(row.get('anchor_accuracy_percent'))} | {number(row.get('anchor_gain_vs_best_single_pp'), True)} |")
    for _, scope in SCOPES:
        gains = [row["anchor_gain_vs_best_single_pp"] for row in tables["seed_gains.csv"] if row["evaluation_scope"] == scope and row["status"] == "completed"]
        text = f"{statistics.mean(gains):+.4f} ± {statistics.stdev(gains):.4f} 个百分点" if len(gains) == 3 else f"pending（{len(gains)}/3）"
        lines.extend(["", f"`{scope}` 三种子融合增益均值 ± 样本 SD：{text}。"])
    completed = sum(row["status"] == "completed" for row in tables["ablation_coverage.csv"])
    lines.extend(["", "## 消融覆盖", "", f"当前 {completed}/{len(tables['ablation_coverage.csv'])} 项已完成，逐项状态和证据路径见 [覆盖清单](ablation_coverage.csv)。组件数值见 [组件表](component_ablation.csv)，三种子汇总见 [种子汇总](seed_summary.csv)，逐种子差值见 [种子增益](seed_gains.csv)。", ""])
    return "\n".join(lines)


def verify_archive(archive: Path) -> dict[str, Any]:
    manifest = read_json(archive / "manifest.json")
    errors = []
    for entry in manifest["files"]:
        relative = Path(entry["archive_path"])
        path = archive / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            errors.append(f"Unsafe archive path: {relative}")
        elif not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            errors.append(f"Missing or changed: {relative}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"verified": True, "files": len(manifest["files"]), "bytes": sum(entry["bytes"] for entry in manifest["files"])}


def package(root: Path, archive: Path) -> dict[str, Any]:
    root, archive = root.resolve(), archive.resolve()
    if archive == root or root not in archive.parents:
        raise ValueError("Archive must be a subdirectory of repository root")
    archive.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}

    def register(target: Path, source: Path | None = None, **details: Any) -> None:
        relative = str(target.relative_to(archive))
        files[relative] = {
            "archive_path": relative, "bytes": target.stat().st_size,
            "sha256": sha256(target), **details,
        }
        if source is not None:
            files[relative].update(source_path=str(source.relative_to(root)), source_sha256=sha256(source))

    def copy(source: Path) -> None:
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 8 * 1024 * 1024:
            return
        target = archive / "artifacts" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        register(target, source, transform="byte_identical")

    def collect(directory: Path, recursive: bool = False) -> None:
        if not directory.exists():
            return
        paths = directory.rglob("*.json") if recursive else directory.glob("*.json")
        for source in sorted(paths):
            if source.name in JSON_NAMES and not (set(source.relative_to(directory).parts[:-1]) & EXCLUDED_DIRS):
                copy(source)

    for relative in (BASE, LORA):
        study = root / relative
        collect(study)
        for child in ("heads", "analysis", "readouts", "readouts/heads", "readouts/analysis"):
            collect(study / child)
    collect(root / SUBSET)
    collect(root / "outputs/hanguard_selected122_analysis_20261018")
    collect(root / BASE / "analysis/anchor_harm_review_20261018")
    collect(root / ABLATION, recursive=True)
    copy(root / ABLATION / "summary.csv")
    # These are aggregate numerical tables and a manually reviewed report; no
    # prompt-bearing review packets or full-text annotations are included.
    for name in ("main_results.json", "metrics.csv", "source_metrics.csv", "threshold_sensitivity.csv",
                 "error_complementarity.csv", "layer_probe.csv", "report.md", "artifact_manifest.json"):
        copy(root / REFERENCE / name)
    # Include registered Python sources and their project import closure. A new
    # protocol may be absent during the first pending snapshot; core seeds still
    # make this archive useful, and later refresh collects the new bindings.
    bindings: dict[Path, set[str]] = {}
    for entry in list(files.values()):
        if entry["archive_path"].endswith(".json"):
            for path, digests in bound_source_paths(root, read_json(archive / entry["archive_path"])).items():
                bindings.setdefault(path, set()).update(digests)
    seeds = [root / "scripts/hanguard" / name for name in CORE_SOURCES] + list(bindings)
    dependencies = source_dependency_closure(root, seeds)
    source_rows = []
    for source in dependencies:
        copy(source)
        digest = sha256(source)
        expected = sorted(bindings.get(source, set()))
        source_rows.append({"source_path": str(source.relative_to(root)), "sha256": digest,
                            "registered_sha256": expected,
                            "matches_all_registered_hashes": not expected or expected == [digest]})
    source_index = archive / "source_dependencies.json"
    write_json(source_index, {"schema": "hanguard_fusion_source_dependencies_v1",
                             "discovery": "Allowlisted repository Python protocol bindings plus static project import closure; no imports executed.",
                             "sources": source_rows})
    register(source_index)

    predictions: dict[str, Path] = {}
    for name, relative in (("nolora_full", BASE / "analysis/predictions.csv"),
                           ("lora_subset", LORA / "readouts/analysis/predictions.csv")):
        source = root / relative
        if source.exists():
            target = archive / "artifacts" / relative
            details = export_predictions(source, target)
            register(target, source, **details)
            predictions[name] = target

    excluded_file = root / SUBSET / "excluded_base_ids.txt"
    subset_metrics: dict[str, Any] | None = None
    if "nolora_full" in predictions and excluded_file.exists():
        excluded = {line.strip() for line in excluded_file.read_text().splitlines() if line.strip()}
        source = root / BASE / "analysis/predictions.csv"
        target = archive / "artifacts" / SUBSET / "nolora_fixed_predictions.csv"
        details = export_predictions(source, target, excluded)
        register(target, source, **details, excluded_ids_sha256=sha256(excluded_file), selection_policy="posthoc_error_selected_test_subset")
        metrics = prediction_metrics(target)
        subset_metrics = {"rows": details["rows"], "metrics": metrics, "seed": 42,
                          "selection_policy": "posthoc_error_selected_test_subset", "models_retrained": False,
                          "thresholds_changed": False, "same_retained_rows_for_all_models": True}
        claimed = read_json(root / SUBSET / "old_model_subset_metrics.json")["selected_test_subset"]
        if details["rows"] != claimed["rows"]:
            raise ValueError("Retained prediction count disagrees with subset release")
        for method in METHODS:
            if not math.isclose(metrics[method]["accuracy"], claimed["accuracy"][method], abs_tol=1e-12):
                raise ValueError(f"Subset accuracy disagrees with release: {method}")
        metric_target = archive / "artifacts" / SUBSET / "nolora_recomputed_metrics.json"
        write_json(metric_target, subset_metrics)
        register(metric_target, source, transform="recomputed_from_retained_fixed_predictions", excluded_ids_sha256=sha256(excluded_file))
        copy(excluded_file)

    extra = root / ABLATION
    if extra.exists():
        for source in sorted(extra.rglob("predictions.csv")):
            if set(source.relative_to(extra).parts[:-1]) & EXCLUDED_DIRS:
                continue
            target = archive / "artifacts" / source.relative_to(root)
            register(target, source, **export_predictions(source, target))

    results: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for name, relative, scope, policy, no_lora in (
        ("nolora_original", BASE / "analysis/results.json", "full_test_8350", "original_full_test", True),
        ("lora_selected122", LORA / "readouts/analysis/results.json", "selected_test_8228", "posthoc_error_selected_test_subset", False),
    ):
        source = root / relative
        if source.exists():
            value = read_json(source)
            results.extend(result_rows(name, scope, policy, no_lora, value, str(relative)))
            states.append({"study": name, "status": "complete" if value.get("complete") else "partial", "source": str(relative)})
        else:
            states.append({"study": name, "status": "pending", "source": str(relative)})
    if subset_metrics is not None:
        results.extend(result_rows("nolora_fixed_selected122", "selected_test_8228", "posthoc_error_selected_test_subset", True,
                                   subset_metrics, str(SUBSET / "old_model_subset_metrics.json")))
        states.append({"study": "nolora_fixed_selected122", "status": "complete", "models_retrained": False,
                       "source": str(SUBSET / "old_model_subset_metrics.json")})
    # Prefer individual task results to avoid double-counting the aggregate.
    new_results = [path for path in sorted(extra.rglob("results.json"))
                   if not (set(path.relative_to(extra).parts[:-1]) & EXCLUDED_DIRS)] if extra.exists() else []
    task_results = [path for path in new_results if path.parent != extra]
    for source in task_results or new_results:
        value = read_json(source)
        task_name = source.relative_to(extra).parts[0] if source.parent != extra else "aggregate"
        seed_match = re.fullmatch(r"seed_(\d+)", task_name)
        seed = value.get("seed", int(seed_match[1]) if seed_match else 42)
        if any(key in value for key in ("full_test", "selected_subset")):
            for block, scope, policy in (("full_test", "full_test_8350", "original_full_test"),
                                         ("selected_subset", "selected_test_8228", "posthoc_error_selected_test_subset")):
                if isinstance(value.get(block), dict):
                    view = {**value[block], "seed": seed}
                    results.extend(result_rows(f"nolora_additional_{task_name}", scope, policy, True, view, str(source.relative_to(root))))
        else:
            scope = value.get("evaluation_scope", value.get("scope", "see_ablation_protocol"))
            if not isinstance(scope, str):
                scope = "see_ablation_protocol"
            policy = value.get("selection_policy", "see_ablation_protocol")
            results.extend(result_rows("nolora_additional_ablation", scope, policy, True, value, str(source.relative_to(root))))
    aggregate = extra / "results.json"
    done = aggregate.exists() and read_json(aggregate).get("complete") is True
    states.append({"study": "nolora_additional_ablation", "status": "complete" if done else "pending",
                   "source": str(ABLATION), "result_files": len(new_results)})
    reference = root / REFERENCE / "main_results.json"
    states.append({"study": "nolora_reference_analysis", "status": "complete" if reference.exists() and read_json(reference).get("complete") else "pending",
                   "source": str(REFERENCE)})

    summary = archive / "results_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    register(summary, transform="normalized_metrics_percent_and_percentage_points")
    tables = presentation_tables(root, subset_metrics)
    for name, rows in tables.items():
        path = archive / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PRESENTATION_FIELDS[name])
            writer.writeheader()
            writer.writerows(rows)
        register(path, transform="completed_result_tables_with_explicit_pending_states")
    index = archive / "index.json"
    write_json(index, {"schema": "hanguard_fusion_ablation_index_v1", "studies": states,
                       "scope_note": "Full 8350 and outcome-selected 8228 scopes are separate; the latter is descriptive, not an independent blind test."})
    register(index)
    readme = archive / "README.md"
    readme.write_text(archive_readme(states) + "\n" + presentation_readme(tables), encoding="utf-8")
    register(readme)
    write_json(archive / "manifest.json", {
        "schema": "hanguard_fusion_ablation_archive_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_policy": "Small allowlisted JSON and text-free numerical predictions only; no weights, features or raw complete text.",
        "files": [files[key] for key in sorted(files)],
    })
    return verify_archive(archive)


def archive_readme(states: list[dict[str, Any]]) -> str:
    state_lines = "\n".join(f"| `{item['study']}` | {item['status']} | `{item.get('source', '')}` |" for item in states)
    return f"""# 多层融合结果与消融归档（20261019）

主要研究路线采用冻结原始 Qwen3.5-4B 的无 LoRA 读出。以下两种测试范围必须分开报告，所有同范围方法使用完全相同的样本、既有验证集阈值和检查点。

| 范围 | 最佳单层 L19 | 锚点多层融合 | 差值 |
|---|---:|---:|---:|
| 原始完整测试集，8,350 条 | 94.5868% | 95.1976% | +0.6108 个百分点 |
| 固定排除 122 条后的子集，8,228 条 | 94.5066% | 96.6091% | +2.1026 个百分点 |
| 同一子集上的新冻结 LoRA 对照，8,228 条 | 97.2654% | 97.4842% | +0.2188 个百分点 |

122 条子集通过查看旧无 LoRA 模型错误后确定：排除 WildGuard 中“最佳单层正确、锚点融合错误”的 122 条。`posthoc_error_selected_test_subset` 是结果依赖的事后子集，+2.1026 是该固定子集上可复算的描述性增益，不能写成未经筛选测试集上的提升或数据质量整改收益。训练与验证样本没有因此删除。原始完整测试结果同时保留。

新补充消融在 `outputs/hanguard_nolora_ablation_20261019` 登记。缺失结果以 `pending` 表示，不能把计划当成已经完成的实验。新结果沿用固定 122 个 ID，不再按新模型错误更改子集。

| 实验 | 状态 | 来源 |
|---|---|---|
{state_lines}

- [汇总结果 CSV](results_summary.csv)：准确率、F1 等以百分比表示，差值以百分点表示；`evaluation_scope` 与 `threshold_policy` 区分口径。
- [实验状态索引](index.json)：区分已完成与待补充项。
- [SHA256 清单](manifest.json)：保存归档文件哈希、来源路径与来源哈希；转换过的逐样本文件注明转换方式。
- [源码依赖索引](source_dependencies.json)：协议及队列绑定的仓库内小型 Python 源码，以及静态分析得到的本项目 import 依赖；保留源码原相对路径与已登记 SHA256 的匹配情况。
- 已有结果的 CPU 复核报告、逐来源结果、阈值敏感性、错误互补与 32 层验证 probe 表位于 `artifacts/outputs/hanguard_nolora_reference_analysis_20261019/`。对应分析、消融、调度和打包源码快照位于 `artifacts/scripts/hanguard/`。
- `artifacts/` 保留来源相对路径。逐样本 CSV 仅保留 ID、标签、来源、长度及预测数值；完整中文、英文文本、权重、大型特征缓存、运行临时文件均不归档。

在仓库根目录刷新或校验：

```bash
python scripts/hanguard/package_fusion_ablation_results.py
python scripts/hanguard/package_fusion_ablation_results.py --verify
```

刷新只写归档目录，不移动或删除实验原目录，不重写已注册协议、科学源码、模型或数据。归档里的路径和协议可能引用本机未归档的大文件；重放训练仍需原始数据、基座和特征。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--verify", action="store_true", help="Verify archive file sizes and SHA256 without reading source outputs")
    args = parser.parse_args()
    archive = args.output if args.output.is_absolute() else args.root / args.output
    result = verify_archive(archive.resolve()) if args.verify else package(args.root, archive)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
