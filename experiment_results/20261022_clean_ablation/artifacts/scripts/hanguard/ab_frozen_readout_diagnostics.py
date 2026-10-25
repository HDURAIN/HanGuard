"""Group-held-out B readouts for a backbone learned exclusively on cohort A.

This script never opens the release validation/test files or trains a LoRA.
An outer B fold is scored only after probe stopping, layer selection, head
stopping, and thresholds have been fixed using the other B groups.  The same
outer-fold column is consumed for base/adapted and alternative pooling caches.
The output is exploratory diagnostic evidence, not a final test benchmark.

Examples::

    python scripts/hanguard/ab_frozen_readout_diagnostics.py probes \\
      --cache-manifest OUTPUT/features/base_last.manifest.json --output OUTPUT/base_last
    python scripts/hanguard/ab_frozen_readout_diagnostics.py run \\
      --cache-manifest OUTPUT/features/adapted_last.manifest.json --output OUTPUT/adapted_last

``--joint-features`` adds directly trained linear joint-feature readouts with
matched repeat-best controls. A logit stack is deliberately absent: fitting a
meta learner to its base probes' in-sample predictions is not cross-fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.hanguard.frozen_lora_layer_diagnostics import (  # noqa: E402
    ALL_LAYERS, FIXED_FUSION_LAYERS, EqualCapacityLayerProbes,
    accuracy_threshold, assert_capacity_matched_initialization, atomic_torch_save,
    binary_metrics, dump, error_complementarity, group_cluster_bootstrap,
    ordered_base_id_sha256, sha256, sigmoid, state_digest, validate_feature_cache,
)
from scripts.hanguard.repaired_heads import ExperimentHead  # noqa: E402

MAIN_ARMS = ("last", "best_single", "repeat_best", "fusion_fixed", "fusion_selected",
             "fusion_fixed_best_anchor", "fusion_selected_best_anchor")
MATCHED_ARMS = ("repeat_best", "fusion_fixed", "fusion_selected",
                "fusion_fixed_best_anchor", "fusion_selected_best_anchor")
JOINT_ARMS = ("joint_repeat_best", "joint_fixed", "joint_selected")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_outer_folds(metadata: pd.DataFrame) -> np.ndarray:
    """Require preassigned, complete-group outer folds without remaking them."""
    required = {"base_id", "group_id", "source", "label", "outer_fold"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"B cache metadata lacks columns: {sorted(required - set(metadata.columns))}")
    for name in ("base_id", "group_id", "source"):
        if metadata[name].isna().any() or metadata[name].astype(str).str.len().eq(0).any():
            raise ValueError(f"B cache contains empty {name}")
    if metadata.base_id.astype(str).duplicated().any():
        raise ValueError("B base_id must be unique")
    if not metadata.label.isin((0, 1)).all():
        raise ValueError("B labels must be binary")
    values = metadata.outer_fold.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("outer_fold must be integer")
    unique = np.unique(values)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("outer_fold must be contiguous zero-based with at least two folds")
    if metadata.groupby("group_id").outer_fold.nunique().ne(1).any():
        raise ValueError("group_id leakage across outer folds")
    for number in unique:
        if metadata.loc[values == number, "label"].nunique() != 2:
            raise ValueError(f"Outer fold {number} lacks one binary class")
    return values.astype(np.int16)


def indices_from_assignments(metadata: pd.DataFrame, outer_fold: int, assignments: pd.DataFrame) -> dict:
    """Use the pre-registered 90/10 inner split without inspecting held labels."""
    outer = validate_outer_folds(metadata)
    required = {"base_id", "group_id", "role", "outer_fold"}
    if not required.issubset(assignments) or assignments.base_id.astype(str).duplicated().any():
        raise ValueError("Invalid preassigned nested fold file")
    if set(assignments.base_id.astype(str)) != set(metadata.base_id.astype(str)):
        raise ValueError("Nested assignments must cover B exactly")
    if not assignments.outer_fold.eq(outer_fold).all():
        raise ValueError("Nested assignment outer fold differs")
    lookup = assignments.assign(base_id=assignments.base_id.astype(str)).set_index("base_id")
    aligned = lookup.loc[metadata.base_id.astype(str)]
    if aligned.group_id.astype(str).tolist() != metadata.group_id.astype(str).tolist():
        raise ValueError("Nested assignment group identity changed")
    if set(aligned.role) != {"inner_train", "inner_validation", "outer_validation"}:
        raise ValueError("Invalid nested assignment roles")
    fit = np.flatnonzero(aligned.role.to_numpy() == "inner_train")
    dev = np.flatnonzero(aligned.role.to_numpy() == "inner_validation")
    held = np.flatnonzero(aligned.role.to_numpy() == "outer_validation")
    if not np.array_equal(held, np.flatnonzero(outer == outer_fold)):
        raise ValueError("Nested outer_validation rows do not match preassigned outer folds")
    group_sets = [set(metadata.iloc[x].group_id.astype(str)) for x in (fit, dev, held)]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Nested fit/dev/held groups overlap")
    return dict(fit=fit, dev=dev, held=held)


def nested_indices(metadata: pd.DataFrame, outer_fold: int) -> dict:
    protocol_path = Path(metadata.attrs["ab_protocol_path"])
    partition = read_json(protocol_path)
    entry = next(value for value in partition["folds"] if value["fold"] == outer_fold)
    path = Path(entry["assignments_path"])
    relative = f"folds/fold_{outer_fold}_assignments.parquet"
    if path.resolve() != (protocol_path.parent / relative).resolve():
        raise ValueError("Nested assignment path escapes registered partition")
    if sha256(path) != partition["artifact_sha256"][relative]:
        raise ValueError("Nested fold assignments changed")
    return indices_from_assignments(metadata, outer_fold, pd.read_parquet(path))


def load_b_cache(manifest_path: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Verify immutable feature provenance, values, row order and A/B separation."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True or manifest.get("cohort") != "B":
        raise ValueError("Only complete cohort B caches are accepted")
    representation = manifest.get("representation")
    if representation not in ("base", "adapted"):
        raise ValueError("Cache representation must be base or adapted")
    expected_training = "A" if representation == "adapted" else "none"
    if manifest.get("adapter_training_cohort") != expected_training:
        raise ValueError("Adapter must be trained on A only; base must have no trained adapter")
    ab_protocol_path = _resolve_manifest_path(manifest_path, manifest["ab_protocol_path"])
    if sha256(ab_protocol_path) != manifest.get("ab_protocol_sha256"):
        raise ValueError("A/B protocol identity changed")
    if manifest.get("pooling", "last") not in ("last", "mean"):
        raise ValueError("Unsupported frozen cache pooling")
    features_path = _resolve_manifest_path(manifest_path, manifest["path"])
    metadata_path = _resolve_manifest_path(manifest_path, manifest["metadata_path"])
    for path, key in ((features_path, "sha256"), (metadata_path, "metadata_sha256")):
        if sha256(path) != manifest.get(key):
            raise ValueError(f"Cache identity changed: {key}")
    metadata = pd.read_parquet(metadata_path)
    validate_outer_folds(metadata)
    partition = read_json(ab_protocol_path)
    if partition.get("policy_id") != "hanguard_train_internal_ab_grouped_v1":
        raise ValueError("An immutable registered A/B diagnostic partition is required")
    if partition.get("invariants", {}).get("final_test_enabled") is not False:
        raise ValueError("Formal test must remain disabled during diagnosis")
    b_path = ab_protocol_path.parent / "B.parquet"
    if sha256(b_path) != partition["artifact_sha256"]["B.parquet"]:
        raise ValueError("Registered B rows changed")
    b = pd.read_parquet(b_path, columns=["base_id", "group_id", "source", "prompt_harm_label", "outer_fold"])
    if b.base_id.astype(str).tolist() != metadata.base_id.astype(str).tolist():
        raise ValueError("Cache does not contain exactly the registered B row order")
    for name in ("group_id", "source", "outer_fold"):
        if b[name].astype(str).tolist() != metadata[name].astype(str).tolist():
            raise ValueError(f"B cache metadata differs from partition: {name}")
    if not np.array_equal(b.prompt_harm_label.eq("harmful").to_numpy(dtype=int), metadata.label.to_numpy()):
        raise ValueError("B cache source binary labels changed")
    for name in ("lora_data/train.parquet", "lora_data/validation.parquet"):
        a_path = ab_protocol_path.parent / name
        if sha256(a_path) != partition["artifact_sha256"][name]:
            raise ValueError("Registered A rows changed")
        a = pd.read_parquet(a_path, columns=["base_id", "group_id"])
        for column in ("base_id", "group_id"):
            if set(a[column].astype(str)) & set(b[column].astype(str)):
                raise ValueError("A/B identity overlap invalidates frozen diagnostics")
    training_path = _resolve_manifest_path(manifest_path, manifest["training_protocol_path"])
    if sha256(training_path) != manifest.get("training_protocol_sha256"):
        raise ValueError("Training protocol identity changed")
    training = read_json(training_path)
    registration = read_json(training_path.parent / "registration.json")
    if registration.get("protocol_sha256") != manifest["training_protocol_sha256"]:
        raise ValueError("A-only training protocol registration changed")
    if (training.get("training_entrypoint") != "hanguard_ab_diagnostic_v1" or
            training.get("partition_manifest_sha256") != manifest["ab_protocol_sha256"] or
            training.get("adapter_training_cohort") != "A" or
            training.get("fresh_base") is not True or training.get("reuse_old_adapter") is not False or
            training.get("formal_test_enabled") is not False):
        raise ValueError("Cache requires a fresh A-only training protocol with formal test disabled")
    if training.get("data_sha256") != {"train": partition["artifact_sha256"]["lora_data/train.parquet"],
                                        "validation": partition["artifact_sha256"]["lora_data/validation.parquet"]}:
        raise ValueError("Training protocol contains different A rows")
    replay_path = metadata_path.parent / "replay_audit.json"
    if sha256(replay_path) != manifest.get("replay_audit_sha256") or read_json(replay_path).get("passed") is not True:
        raise ValueError("Fresh feature replay failed or changed")
    if representation == "adapted":
        lineage_path = _resolve_manifest_path(manifest_path, manifest["checkpoint_lineage_path"])
        lineage = read_json(lineage_path)
        expected = dict(adapter_training_cohort="A", partition_manifest_sha256=manifest["ab_protocol_sha256"],
                        A_train_sha256=training["data_sha256"]["train"], A_selection_sha256=training["data_sha256"]["validation"],
                        B_sha256=partition["artifact_sha256"]["B.parquet"], adapter_sha256=manifest["adapter_sha256"],
                        protocol_sha256=manifest["training_protocol_sha256"], formal_test_opened=False)
        if any(lineage.get(key) != value for key, value in expected.items()):
            raise ValueError("Selected adapter lineage is inconsistent with A-only training")
        if sha256(lineage_path.parent / "adapter.pt") != manifest["adapter_sha256"]:
            raise ValueError("Selected frozen adapter changed")
    metadata.attrs["ab_protocol_path"] = str(ab_protocol_path)
    metadata.attrs["training_protocol_path"] = str(training_path)
    shape = tuple(manifest["shape"]) if "shape" in manifest else (len(metadata), len(ALL_LAYERS), manifest["hidden_size"])
    if len(shape) != 3 or shape[0] != len(metadata) or shape[1] != len(ALL_LAYERS):
        raise ValueError("Expected [B rows,32 layers,hidden] cache")
    validate_feature_cache(
        features_path, shape, np.float16, manifest["ordered_base_id_sha256"],
        metadata.base_id.astype(str).tolist(),
    )
    return np.load(features_path, mmap_mode="r", allow_pickle=False), metadata, manifest


def _code_identity() -> dict:
    paths = (Path(__file__), ROOT / "scripts/hanguard/repaired_heads.py",
             ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py",
             ROOT / "scripts/hanguard/oof_probe_logit_stack.py")
    return {path.name: sha256(path) for path in paths}


def initialize(args: argparse.Namespace, metadata: pd.DataFrame, manifest: dict) -> dict:
    output = Path(args.output).resolve()
    settings = dict(
        seed=args.seed, probe_epochs=args.probe_epochs,
        head_epochs=args.head_epochs, max_epochs=args.max_epochs,
        probe_lr=args.probe_lr, head_lr=args.head_lr,
        weight_decay=args.weight_decay, width=args.width, dropout=args.dropout,
        batch_size=args.batch_size, joint_features=args.joint_features,
        bootstrap_draws=args.bootstrap_draws,
    )
    for key in ("probe_epochs", "head_epochs", "max_epochs", "width", "batch_size", "bootstrap_draws"):
        if settings[key] < 1:
            raise ValueError(f"Invalid setting: {key}")
    if settings["probe_lr"] <= 0 or settings["head_lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    if not 0 <= settings["dropout"] < 1:
        raise ValueError("Invalid dropout")
    if not max(settings["probe_epochs"], settings["head_epochs"]) <= settings["max_epochs"] <= 40:
        raise ValueError("Initial budgets must fit within max_epochs, capped at 40")
    settings["extension_rule"] = dict(
        milestones=[24, 40], recent_epochs=3, relative_dev_bce_improvement=0.002,
        trigger="any arm/linear probe has its best dev BCE at the latest epoch and improves over the last three epochs",
        effect="extend every matched head/all 32 linear probes to the same next milestone, capped at max_epochs",
        selection_scope="inner dev only; no outer validation labels or metrics",
    )
    protocol = dict(
        schema="hanguard_ab_frozen_readouts_v1", cache_manifest_path=str(Path(args.cache_manifest).resolve()),
        cache_manifest_sha256=sha256(Path(args.cache_manifest)),
        ab_protocol_sha256=manifest["ab_protocol_sha256"], adapter_sha256=manifest.get("adapter_sha256"),
        representation=manifest["representation"], pooling=manifest.get("pooling", "last"),
        ordered_base_id_sha256=ordered_base_id_sha256(metadata.base_id.astype(str).tolist()),
        rows=len(metadata), outer_folds=sorted(map(int, metadata.outer_fold.unique())),
        inner_partition="immutable per-fold inner_train/inner_validation/outer_validation roles from A/B manifest",
        code_sha256=_code_identity(), settings=settings,
        selection_scope="inner dev of each outer-training partition; no formal validation/test access",
        interpretation="screening with one A-only LoRA and one head seed; not cross-LoRA replication",
        stop_directories=list(dict.fromkeys((str(output), str(output.parent),
            str(Path(metadata.attrs["training_protocol_path"]).parent),
            str(Path(metadata.attrs["training_protocol_path"]).parent.parent)))),
    )
    path = output / "protocol.json"
    if path.is_file():
        if read_json(path) != protocol:
            raise ValueError("Existing diagnostic protocol differs; choose a new output directory")
    else:
        dump(path, protocol)
    return protocol


def check_stop(protocol: dict) -> None:
    for folder in protocol["stop_directories"]:
        path = Path(folder) / "STOP"
        if path.exists():
            raise RuntimeError(f"Diagnostic stopped: {path.read_text().strip()}")


def budget_extension(history: Sequence[dict], best_epoch: Mapping[str, int], budget: int, max_epochs: int) -> dict | None:
    """Bounded common-budget extension based entirely on recent inner-dev BCE."""
    if budget >= max_epochs or len(history) < 3:
        return None
    def as_mapping(values: Any) -> dict:
        return values if isinstance(values, dict) else {str(k + 1): value for k, value in enumerate(values)}
    first = as_mapping(history[-3]["dev_bce"])
    last = as_mapping(history[-1]["dev_bce"])
    epoch = history[-1]["epoch"]
    triggers = {}
    for arm, chosen_epoch in best_epoch.items():
        improvement = (float(first[arm]) - float(last[arm])) / max(abs(float(first[arm])), 1e-12)
        if chosen_epoch == epoch and improvement >= 0.002:
            triggers[arm] = dict(best_epoch=int(chosen_epoch), relative_recent_dev_bce_improvement=improvement)
    if not triggers:
        return None
    next_milestone = next((value for value in (24, 40) if value > budget), max_epochs)
    return dict(previous_budget=budget, new_budget=min(next_milestone, max_epochs),
                triggers=triggers, selection_scope="inner_dev_only", common_extension=True)


def _row_identity(metadata: pd.DataFrame, indices: Mapping[str, np.ndarray]) -> dict:
    return {
        name: dict(rows=len(rows), groups=metadata.iloc[rows].group_id.nunique(),
                   ordered_base_id_sha256=ordered_base_id_sha256(metadata.iloc[rows].base_id.astype(str).tolist()))
        for name, rows in indices.items()
    }


def feature_batch(cache: np.ndarray, rows: np.ndarray, layers: Sequence[int], device: str) -> torch.Tensor:
    # Gather only the layers needed by this batch; never materialize the full cache.
    array = np.stack([np.asarray(cache[rows, layer - 1], dtype=np.float32) for layer in layers], axis=1)
    return torch.from_numpy(array).to(device=device)


@torch.no_grad()
def predict_probes(model: nn.Module, cache: np.ndarray, rows: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    values = []
    for begin in range(0, len(rows), batch_size):
        values.append(model(feature_batch(cache, rows[begin:begin + batch_size], ALL_LAYERS, device)).cpu().numpy())
    logits = np.concatenate(values)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite probe logits")
    return logits


def _provenance(output: Path, protocol: dict, fold: int, indices: dict, metadata: pd.DataFrame, stage: str) -> dict:
    return dict(protocol_sha256=sha256(output / "protocol.json"), outer_fold=fold, stage=stage,
                seed=protocol["settings"]["seed"], row_identity=_row_identity(metadata, indices))


def _write_stage(directory: Path, provenance: dict, result: dict, artifacts: Sequence[Path]) -> dict:
    record = dict(provenance=provenance, result=result,
                  artifacts={path.name: sha256(path) for path in artifacts}, complete=True)
    dump(directory / "complete.json", record)
    return record


def _resume_stage(directory: Path, provenance: dict) -> dict | None:
    sentinel = directory / "complete.json"
    if not sentinel.is_file():
        return None
    value = read_json(sentinel)
    if value.get("complete") is not True or value.get("provenance") != provenance:
        raise ValueError(f"Completed stage provenance changed: {directory}")
    if not value.get("artifacts"):
        raise ValueError(f"Completed stage has no bound artifacts: {directory}")
    for name, digest in value["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Completed stage artifact changed: {directory / name}")
    return value


def choose_layers(y_dev: np.ndarray, p_dev: np.ndarray, thresholds: np.ndarray) -> dict:
    """Select the single layer and four-layer rescue set on inner dev only."""
    if p_dev.shape != (len(y_dev), len(ALL_LAYERS)):
        raise ValueError("All 32 aligned dev probe probabilities are required")
    metrics = [binary_metrics(y_dev, p_dev[:, k], thresholds[k]) for k in range(len(ALL_LAYERS))]
    best = max(range(32), key=lambda k: (metrics[k]["accuracy"], -metrics[k]["bce"], k)) + 1
    selected = [best]
    correct = p_dev >= thresholds[None, :]
    correct = correct == y_dev[:, None]
    covered = correct[:, np.asarray(selected) - 1].any(1)
    trace = []
    while len(selected) < 4:
        remaining = [layer for layer in ALL_LAYERS if layer not in selected]
        chosen = max(remaining, key=lambda layer: (
            int((~covered & correct[:, layer - 1]).sum()),
            metrics[layer - 1]["accuracy"], -metrics[layer - 1]["bce"], layer,
        ))
        rescued = int((~covered & correct[:, chosen - 1]).sum())
        covered |= correct[:, chosen - 1]
        selected.append(chosen)
        trace.append(dict(layer=chosen, newly_rescued=rescued))
    return dict(best_single_layer=best, selected_layers=sorted(selected),
                selected_by="inner-dev linear-probe accuracy/BCE; greedy dev error rescue anchored at that best layer",
                best_single_definition="linear-probe-selected layer, then freshly fitted MLP; not best of 32 separately tuned MLPs",
                inner_dev_layer_metrics={str(k + 1): metrics[k] for k in range(32)},
                rescue_trace=trace, dev_probe_oracle_accuracy=float(covered.mean()),
                oracle_interpretation="union of these probes' correct decisions; not attainable deployment accuracy")


def train_probe_fold(cache: np.ndarray, metadata: pd.DataFrame, protocol: dict,
                     output: Path, fold: int, device: str) -> dict:
    s = protocol["settings"]
    indices = nested_indices(metadata, fold)
    directory = output / f"fold_{fold}" / "probes"
    provenance = _provenance(output, protocol, fold, indices, metadata, "probes")
    resumed = _resume_stage(directory, provenance)
    if resumed is not None:
        return resumed
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(s["seed"])
    model = EqualCapacityLayerProbes(cache.shape[-1], seed=s["seed"]).to(device)
    initial = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["probe_lr"], weight_decay=s["weight_decay"])
    best_bce = np.full(32, np.inf)
    best_epoch = np.zeros(32, dtype=np.int16)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    y = metadata.label.to_numpy(dtype=np.int64)
    history, extensions = [], []
    budget, epoch = s["probe_epochs"], 0
    while epoch < budget:
        check_stop(protocol)
        model.train()
        order = np.random.default_rng(s["seed"] * 1009 + epoch).permutation(indices["fit"])
        total = np.zeros(32)
        for start in range(0, len(order), s["batch_size"]):
            check_stop(protocol)
            rows = order[start:start + s["batch_size"]]
            raw = feature_batch(cache, rows, ALL_LAYERS, device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            logits = model(raw)
            losses = F.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits), reduction="none").mean(0)
            optimizer.zero_grad(set_to_none=True)
            # Sum independent objectives: every probe receives the same BCE gradient as if trained alone.
            losses.sum().backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Missing/nonfinite linear probe gradients")
            optimizer.step()
            total += losses.detach().cpu().numpy() * len(rows)
        dev_logits = predict_probes(model, cache, indices["dev"], s["batch_size"], device)
        dev_bce = F.binary_cross_entropy_with_logits(
            torch.from_numpy(dev_logits), torch.tensor(y[indices["dev"]], dtype=torch.float32)[:, None].expand_as(torch.from_numpy(dev_logits)),
            reduction="none",
        ).mean(0).numpy()
        changed = dev_bce < best_bce
        best_bce[changed] = dev_bce[changed]
        best_epoch[changed] = epoch + 1
        for name, parameter in model.state_dict().items():
            best_state[name][changed] = parameter.detach().cpu()[changed]
        history.append(dict(epoch=epoch + 1, train_bce=total.tolist(), dev_bce=dev_bce.tolist()))
        history[-1]["train_bce"] = (total / len(indices["fit"])).tolist()
        print(json.dumps(dict(stage="B_probes", fold=fold, epoch=epoch + 1, epochs=budget)), flush=True)
        epoch += 1
        if epoch == budget:
            extension = budget_extension(history, {str(k + 1): int(best_epoch[k]) for k in range(32)}, budget, s["max_epochs"])
            if extension:
                extensions.append(extension)
                budget = extension["new_budget"]
                print(json.dumps(dict(stage="B_probes_budget_extension", fold=fold, **extension)), flush=True)
    if not np.isfinite(best_bce).all() or (best_epoch == 0).any():
        raise FloatingPointError("No finite dev probe checkpoint")
    model.load_state_dict(best_state)
    dev_p = sigmoid(predict_probes(model, cache, indices["dev"], s["batch_size"], device))
    thresholds = np.asarray([accuracy_threshold(y[indices["dev"]], dev_p[:, k]) for k in range(32)])
    selection = choose_layers(y[indices["dev"]], dev_p, thresholds)
    # All selection is complete before any outer-held predictions/metrics are generated.
    dump(directory / "selection.json", dict(**selection, thresholds=thresholds.tolist(), provenance=provenance))
    held_p = sigmoid(predict_probes(model, cache, indices["held"], s["batch_size"], device))
    frame = metadata.iloc[indices["held"]].copy()
    for k in range(32):
        frame[f"probability_L{k + 1}"] = held_p[:, k]
        frame[f"threshold_L{k + 1}"] = thresholds[k]
        frame[f"prediction_L{k + 1}"] = (held_p[:, k] >= thresholds[k]).astype(np.int8)
    prediction_path = directory / "held_predictions.parquet"
    frame.to_parquet(prediction_path, index=False)
    checkpoint_path = directory / "probes.pt"
    atomic_torch_save(dict(state_dict=best_state, best_epoch=best_epoch.tolist(), provenance=provenance), checkpoint_path)
    dump(directory / "history.json", history)
    result = dict(**selection, best_epoch=best_epoch.tolist(), thresholds=thresholds.tolist(),
                  initial_state_sha256=initial, final_state_sha256=state_digest(model),
                  budgets=dict(initial=s["probe_epochs"], actual=budget, cap=s["max_epochs"],
                               extensions=extensions, best_checkpoint_at_budget=(best_epoch == budget).tolist()),
                  held_layer_metrics={str(k + 1): binary_metrics(y[indices["held"]], held_p[:, k], thresholds[k]) for k in range(32)})
    return _write_stage(directory, provenance, result,
                        (prediction_path, checkpoint_path, directory / "history.json", directory / "selection.json"))


class JointFeatureReadout(nn.Module):
    """L2-regularized linear readout of independently normalized block vectors."""

    def __init__(self, hidden_size: int, branches: int = 4):
        super().__init__()
        self.classifier = nn.Linear(hidden_size * branches, 1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(raw.float(), (raw.shape[-1],))
        return self.classifier(normalized.flatten(1)).squeeze(-1)


def head_specs(best: int, selected: Sequence[int], joint: bool = False) -> dict:
    specs = dict(last=("last", (32,)), best_single=("last", (best,)),
                 repeat_best=("repeat_last", (best,)), fusion_fixed=("fusion", FIXED_FUSION_LAYERS),
                 fusion_selected=("fusion", tuple(selected)),
                 fusion_fixed_best_anchor=("fusion_best_anchor", FIXED_FUSION_LAYERS + (best,)),
                 fusion_selected_best_anchor=("fusion_best_anchor", tuple(selected) + (best,)))
    if joint:
        specs.update(joint_repeat_best=("joint", (best, best, best, best)),
                     joint_fixed=("joint", FIXED_FUSION_LAYERS), joint_selected=("joint", tuple(selected)))
    return specs


def build_heads(hidden_size: int, width: int, best: int, selected: Sequence[int],
                seed: int, dropout: float, device: str, joint: bool = False) -> tuple[dict, dict, dict]:
    specs = head_specs(best, selected, joint)
    heads = {}
    for arm, (mode, layers) in specs.items():
        torch.manual_seed(seed)
        if mode == "joint":
            head = JointFeatureReadout(hidden_size)
        else:
            # Repeat-best uses four branch modules. Their logical capture labels
            # remain the fixed increasing slots; input is explicitly the chosen layer.
            configured = FIXED_FUSION_LAYERS if mode == "repeat_last" else layers[:4] if mode == "fusion_best_anchor" else layers
            head_mode = "fusion" if mode == "fusion_best_anchor" else mode
            head = ExperimentHead(hidden_size, mode=head_mode, width=width, layers=configured, dropout=dropout)
        heads[arm] = head.to(device)
    matched = assert_capacity_matched_initialization({arm: heads[arm] for arm in MATCHED_ARMS})
    single_match = assert_capacity_matched_initialization({arm: heads[arm] for arm in ("last", "best_single")})
    result = dict(fusion=matched, single=single_match)
    if joint:
        result["joint"] = assert_capacity_matched_initialization({arm: heads[arm] for arm in JOINT_ARMS})
    return heads, specs, result


def _head_logits(head: nn.Module, raw: torch.Tensor, positions: Sequence[int], mode: str) -> torch.Tensor:
    if mode == "joint":
        return head(raw[:, list(positions)])
    if mode == "fusion_best_anchor":
        # Preserve the actual fusion's four branch modules, normalization,
        # weights and residual output. Change only the direct projection input.
        # Last position is an explicitly recorded anchor, not another branch.
        source = [raw[:, position] for position in positions[:4]]
        direct = raw[:, positions[4]]
        z = F.gelu(head._project_raw(direct, head.project))
        weights = head.layer_logits.softmax(0)
        value = sum(weights[k] * head._project_raw(source[k], module)
                    for k, module in enumerate(head.layer_values))
        return head.classify_representation(z + head.fusion_output(value))
    pieces = tuple(raw[:, position].unsqueeze(1) for position in positions)
    mask = torch.ones((len(raw), 1), dtype=torch.bool, device=raw.device)
    return head({"layers": pieces, "mask": mask})["logits"]


@torch.no_grad()
def predict_heads(heads: dict, specs: dict, cache: np.ndarray, rows: np.ndarray,
                  batch_size: int, device: str) -> dict:
    union = sorted({layer for _, layers in specs.values() for layer in layers})
    positions = {layer: i for i, layer in enumerate(union)}
    values = {arm: [] for arm in heads}
    for head in heads.values():
        head.eval()
    for begin in range(0, len(rows), batch_size):
        raw = feature_batch(cache, rows[begin:begin + batch_size], union, device)
        for arm, head in heads.items():
            mode, layers = specs[arm]
            values[arm].append(_head_logits(head, raw, [positions[layer] for layer in layers], mode).cpu().numpy())
    result = {arm: np.concatenate(values[arm]) for arm in heads}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite head logits")
    return result


def parameter_audit(head: nn.Module, before: Mapping[str, torch.Tensor], step: int) -> dict:
    """Expose zero-initialized fusion's expected first-step gating and later updates."""
    values = {}
    for name, parameter in head.named_parameters():
        difference = parameter.detach().cpu() - before[name]
        grad = parameter.grad
        values[name] = dict(
            grad_present=grad is not None,
            grad_norm=float(grad.detach().float().norm().cpu()) if grad is not None else None,
            grad_finite=bool(torch.isfinite(grad).all()) if grad is not None else None,
            delta_norm=float(difference.float().norm()),
            delta_max=float(difference.float().abs().max()),
        )
    return dict(step=step, parameters=values,
                note="Zero fusion_output blocks upstream branch gradients at step 1; assess subsequent steps too")


def train_head_fold(cache: np.ndarray, metadata: pd.DataFrame, protocol: dict,
                    output: Path, fold: int, device: str) -> dict:
    s = protocol["settings"]
    indices = nested_indices(metadata, fold)
    probe_directory = output / f"fold_{fold}" / "probes"
    probe_provenance = _provenance(output, protocol, fold, indices, metadata, "probes")
    probes = _resume_stage(probe_directory, probe_provenance)
    if probes is None:
        raise FileNotFoundError("Complete all-layer probes before heads")
    directory = output / f"fold_{fold}" / "heads"
    provenance = _provenance(output, protocol, fold, indices, metadata, "heads")
    provenance["probe_stage_sha256"] = sha256(probe_directory / "complete.json")
    resumed = _resume_stage(directory, provenance)
    if resumed is not None:
        return resumed
    directory.mkdir(parents=True, exist_ok=True)
    selected = probes["result"]["selected_layers"]
    best = probes["result"]["best_single_layer"]
    heads, specs, matched = build_heads(cache.shape[-1], s["width"], best, selected,
                                        s["seed"], s["dropout"], device, s["joint_features"])
    initial_hashes = {arm: state_digest(head) for arm, head in heads.items()}
    initial_states = {arm: {n: p.detach().cpu().clone() for n, p in head.named_parameters()} for arm, head in heads.items()}
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=s["head_lr"], weight_decay=s["weight_decay"])
                  for arm, head in heads.items()}
    best_bce = {arm: np.inf for arm in heads}
    best_epoch = {arm: 0 for arm in heads}
    best_states, audit, history = {}, [], []
    union = sorted({layer for _, layers in specs.values() for layer in layers})
    positions = {layer: k for k, layer in enumerate(union)}
    y = metadata.label.to_numpy(dtype=np.int64)
    global_step, epoch, budget = 0, 0, s["head_epochs"]
    extensions = []
    while epoch < budget:
        check_stop(protocol)
        order = np.random.default_rng(s["seed"] * 1009 + epoch).permutation(indices["fit"])
        total = {arm: 0.0 for arm in heads}
        for step, start in enumerate(range(0, len(order), s["batch_size"])):
            check_stop(protocol)
            rows = order[start:start + s["batch_size"]]
            raw = feature_batch(cache, rows, union, device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            global_step += 1
            for arm, head in heads.items():
                head.train()
                # Pair data order and dropout draws across all arms.
                torch.manual_seed((s["seed"] * 1000003 + epoch * 1009 + step) % (2**63 - 1))
                mode, layers = specs[arm]
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                logits = _head_logits(head, raw, [positions[layer] for layer in layers], mode)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {arm}")
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total[arm] += float(loss.detach()) * len(rows)
                if global_step in (1, 2, 10):
                    audit.append(dict(arm=arm, **parameter_audit(head, initial_states[arm], global_step)))
        dev_logits = predict_heads(heads, specs, cache, indices["dev"], s["batch_size"], device)
        dev_bce = {}
        for arm in heads:
            dev_bce[arm] = binary_metrics(y[indices["dev"]], sigmoid(dev_logits[arm]), 0.5)["bce"]
            if dev_bce[arm] < best_bce[arm]:
                best_bce[arm], best_epoch[arm] = dev_bce[arm], epoch + 1
                best_states[arm] = {name: value.detach().cpu().clone() for name, value in heads[arm].state_dict().items()}
        history.append(dict(epoch=epoch + 1, train_bce={arm: total[arm] / len(indices["fit"]) for arm in heads}, dev_bce=dev_bce))
        dump(directory / "history.json", history)
        print(json.dumps(dict(stage="B_heads", fold=fold, epoch=epoch + 1, epochs=budget)), flush=True)
        epoch += 1
        if epoch == budget:
            extension = budget_extension(history, best_epoch, budget, s["max_epochs"])
            if extension:
                extensions.append(extension)
                budget = extension["new_budget"]
                print(json.dumps(dict(stage="B_heads_budget_extension", fold=fold, **extension)), flush=True)
    for arm, head in heads.items():
        if arm not in best_states:
            raise FloatingPointError(f"No finite checkpoint: {arm}")
        audit.append(dict(arm=arm, checkpoint="last_trained_epoch", **parameter_audit(head, initial_states[arm], global_step)))
        head.load_state_dict(best_states[arm])
    dev_logits = predict_heads(heads, specs, cache, indices["dev"], s["batch_size"], device)
    thresholds = {arm: accuracy_threshold(y[indices["dev"]], sigmoid(logits)) for arm, logits in dev_logits.items()}
    selection = dict(best_single_layer=best, selected_layers=selected,
                     specs={arm: dict(mode=mode, branch_layers=list(layers[:4]) if mode == "fusion_best_anchor" else list(layers),
                                      direct_anchor=layers[-1], cached_inputs=list(layers)) for arm, (mode, layers) in specs.items()},
                     best_epoch=best_epoch, thresholds=thresholds, dev_bce=best_bce, matched_initialization=matched,
                     provenance=provenance)
    dump(directory / "selection.json", selection)
    held_logits = predict_heads(heads, specs, cache, indices["held"], s["batch_size"], device)
    frame = metadata.iloc[indices["held"]].copy()
    metrics = {}
    for arm, logits in held_logits.items():
        probability = sigmoid(logits)
        frame[f"probability_{arm}"] = probability
        frame[f"threshold_{arm}"] = thresholds[arm]
        frame[f"prediction_{arm}"] = (probability >= thresholds[arm]).astype(np.int8)
        metrics[arm] = binary_metrics(y[indices["held"]], probability, thresholds[arm])
    prediction_path = directory / "held_predictions.parquet"
    frame.to_parquet(prediction_path, index=False)
    checkpoint_path = directory / "heads.pt"
    atomic_torch_save(dict(states=best_states, selection=selection), checkpoint_path)
    dump(directory / "gradient_update_audit.json", audit)
    result = dict(**selection, held_metrics=metrics, initial_state_sha256=initial_hashes,
                  parameters={arm: sum(p.numel() for p in head.parameters()) for arm, head in heads.items()},
                  budgets=dict(initial_epochs=s["head_epochs"], common_epochs=budget, cap=s["max_epochs"],
                               extensions=extensions, paired_init=True, paired_order=True, paired_dropout=True,
                               best_checkpoint_at_budget={arm: best_epoch[arm] == budget for arm in heads}))
    return _write_stage(directory, provenance, result,
                        (prediction_path, checkpoint_path, directory / "history.json", directory / "selection.json",
                         directory / "gradient_update_audit.json"))


def _metrics_with_row_thresholds(frame: pd.DataFrame, suffix: str) -> dict:
    p = frame[f"probability_{suffix}"].to_numpy()
    prediction = frame[f"prediction_{suffix}"].to_numpy(dtype=np.int8)
    y = frame.label.to_numpy(dtype=np.int64)
    # BCE/AUROC use probabilities; confusion uses each fold's independently fixed threshold.
    result = binary_metrics(y, p, 0.5)
    decision_metrics = binary_metrics(y, prediction, 0.5)
    for key in ("accuracy", "f1", "recall", "precision", "fpr", "tp", "tn", "fp", "fn"):
        result[key] = decision_metrics[key]
    result["threshold"] = "fold-specific inner-dev threshold"
    result["accuracy_at_0_5"] = binary_metrics(y, p, 0.5)["accuracy"]
    if not (y == 0).any():
        result["fpr"] = None
    if not (y == 1).any():
        result["recall"] = None
    return result


def aggregate_stage(output: Path, protocol: dict, metadata: pd.DataFrame, stage: str) -> pd.DataFrame:
    frames = []
    for fold in protocol["outer_folds"]:
        indices = nested_indices(metadata, fold)
        directory = output / f"fold_{fold}" / stage
        provenance = _provenance(output, protocol, fold, indices, metadata, stage)
        if stage == "heads":
            provenance["probe_stage_sha256"] = sha256(output / f"fold_{fold}" / "probes" / "complete.json")
        if _resume_stage(directory, provenance) is None:
            raise FileNotFoundError(f"Missing completed {stage} fold {fold}")
        frame = pd.read_parquet(directory / "held_predictions.parquet")
        expected = metadata.iloc[indices["held"]]
        if frame.base_id.astype(str).tolist() != expected.base_id.astype(str).tolist():
            raise ValueError("Held prediction row identity changed")
        for column in ("label", "group_id", "source", "outer_fold"):
            if frame[column].astype(str).tolist() != expected[column].astype(str).tolist():
                raise ValueError(f"Held prediction metadata changed: {column}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.base_id.duplicated().any() or set(combined.base_id.astype(str)) != set(metadata.base_id.astype(str)):
        raise ValueError("Outer predictions do not cover every B row exactly once")
    order = {base_id: i for i, base_id in enumerate(metadata.base_id.astype(str))}
    return combined.iloc[np.argsort(combined.base_id.astype(str).map(order).to_numpy())].reset_index(drop=True)


def paired_comparison(frame: pd.DataFrame, candidate: str, baseline: str, draws: int, seed: int) -> dict:
    y = frame.label.to_numpy(dtype=np.int64)
    a = frame[f"prediction_{candidate}"].to_numpy(dtype=np.int8)
    b = frame[f"prediction_{baseline}"].to_numpy(dtype=np.int8)
    ac, bc = a == y, b == y
    transitions = dict(rescued=int((ac & ~bc).sum()), harmed=int((~ac & bc).sum()),
                       both_correct=int((ac & bc).sum()), both_wrong=int((~ac & ~bc).sum()),
                       net_correct=int(ac.sum() - bc.sum()))
    # Pass held hard decisions so the bootstrap uses the frozen per-fold thresholds.
    interval = group_cluster_bootstrap(y, a, b, frame.group_id.astype(str).to_numpy(),
                                        n_resamples=draws, seed=seed)
    return dict(candidate=candidate, baseline=baseline, **transitions, group_bootstrap=interval,
                scope="conditional on this fixed A adapter and split; does not estimate LoRA-seed variance")


def write_report(output: Path, protocol: dict, metadata: pd.DataFrame, heads: bool) -> dict:
    directory = output / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    probes = aggregate_stage(output, protocol, metadata, "probes")
    probes.to_parquet(directory / "probe_outer_predictions.parquet", index=False)
    layer_metrics = {str(layer): _metrics_with_row_thresholds(probes, f"L{layer}") for layer in ALL_LAYERS}
    # This curve is diagnostic; do not choose another arm from outer-held results.
    complement = error_complementarity(
        probes.label.to_numpy(), probes[[f"prediction_L{layer}" for layer in ALL_LAYERS]].to_numpy(),
        layers=ALL_LAYERS, thresholds=np.full(32, 0.5),
    )
    # Decisions must use each fold's threshold, while probability correlation
    # must retain continuous probabilities rather than correlate hard labels.
    probability_matrix = probes[[f"probability_L{layer}" for layer in ALL_LAYERS]].to_numpy()
    centered = probability_matrix - probability_matrix.mean(0, keepdims=True)
    norm = np.sqrt(np.square(centered).sum(0))
    correlation = centered.T @ centered / np.maximum(norm[:, None] * norm[None, :], 1e-30)
    np.fill_diagonal(correlation, 1.0)
    complement["probability_pearson"] = correlation.tolist()
    complement["thresholds"] = "per-row thresholds were fixed on that row's outer-training inner dev"
    result = dict(protocol_sha256=sha256(output / "protocol.json"), rows=len(probes),
                  representation=protocol["representation"], pooling=protocol["pooling"],
                  layer_metrics=layer_metrics, layer_error_complementarity=complement,
                  layer_curve_scope="outer-held diagnostic curve; best layer was chosen independently on each inner dev",
                  stop_conditions=["Any A/B or outer-fold leakage: stop and invalidate results",
                                   "Nonfinite features/loss/gradients or mismatched capacity/init: stop",
                                   "Do not expand training seeds to manufacture a positive screening result"])
    text = ["# A/B 冻结表征诊断", "", f"B 的 {len(probes):,} 条数据均由未训练过其所在 group 的分类头预测。",
            "LoRA 只使用 A；该结果用于定位原因，不是正式测试成绩。", "",
            "所有层选择、停止轮次和阈值都在对应外层训练部分的 inner dev 确定。",
            "置信区间对 group 重采样，并以这一个固定 LoRA 为条件，不能替代多 LoRA 复现。", ""]
    if heads:
        frame = aggregate_stage(output, protocol, metadata, "heads")
        frame.to_parquet(directory / "head_outer_predictions.parquet", index=False)
        arms = MAIN_ARMS + (JOINT_ARMS if protocol["settings"]["joint_features"] else ())
        metrics = {arm: _metrics_with_row_thresholds(frame, arm) for arm in arms}
        comparisons = [paired_comparison(frame, candidate, baseline, protocol["settings"]["bootstrap_draws"], protocol["settings"]["seed"])
                       for candidate, baseline in (("best_single", "last"), ("repeat_best", "best_single"),
                           ("fusion_fixed", "last"), ("fusion_fixed", "best_single"), ("fusion_fixed", "repeat_best"),
                           ("fusion_selected", "best_single"), ("fusion_selected", "repeat_best"),
                           ("fusion_fixed_best_anchor", "repeat_best"), ("fusion_selected_best_anchor", "repeat_best"),
                           ("fusion_fixed_best_anchor", "fusion_fixed"), ("fusion_selected_best_anchor", "fusion_selected"))]
        if protocol["settings"]["joint_features"]:
            comparisons += [paired_comparison(frame, candidate, "joint_repeat_best", protocol["settings"]["bootstrap_draws"], protocol["settings"]["seed"])
                            for candidate in ("joint_fixed", "joint_selected")]
        by_source = {str(source): {arm: _metrics_with_row_thresholds(part, arm) for arm in arms}
                     for source, part in frame.groupby("source", sort=True)}
        subgroup = {}
        if "category_id" in frame:
            subgroup["category_id"] = {str(category): {arm: _metrics_with_row_thresholds(part, arm) for arm in arms}
                                        for category, part in frame.groupby("category_id", sort=True)}
        result.update(head_metrics=metrics, comparisons=comparisons, by_source=by_source, subgroup=subgroup,
                      fold_layer_choices={str(fold): read_json(output / f"fold_{fold}" / "probes" / "selection.json")["selected_layers"]
                                          for fold in protocol["outer_folds"]})
        text += ["| 方法 | Accuracy | BCE | AUROC | Recall | FPR |", "|---|---:|---:|---:|---:|---:|"]
        for arm, metric in metrics.items():
            auc = "—" if metric["auroc"] is None else f"{metric['auroc']:.4f}"
            text.append(f"| {arm} | {100 * metric['accuracy']:.3f}% | {metric['bce']:.4f} | {auc} | {100 * metric['recall']:.3f}% | {100 * metric['fpr']:.3f}% |")
        text += ["", "| 比较 | 修正 | 新增错误 | 净修正 | Accuracy Δ及 group 95% CI |", "|---|---:|---:|---:|---:|"]
        for value in comparisons:
            boot = value["group_bootstrap"]["accuracy"]
            lo, hi = boot["ci95"]
            text.append(f"| {value['candidate']} − {value['baseline']} | {value['rescued']} | {value['harmed']} | {value['net_correct']} | {100 * boot['delta']:.3f} pp [{100 * lo:.3f}, {100 * hi:.3f}] |")
        disagreements = frame.copy()
        for candidate in ("fusion_fixed", "fusion_selected", "fusion_fixed_best_anchor", "fusion_selected_best_anchor"):
            for baseline in ("best_single", "repeat_best"):
                a = frame[f"prediction_{candidate}"].to_numpy() == frame.label.to_numpy()
                b = frame[f"prediction_{baseline}"].to_numpy() == frame.label.to_numpy()
                disagreements[f"transition_{candidate}_vs_{baseline}"] = np.where(a & ~b, "rescued", np.where(~a & b, "harmed", "same_correctness"))
        transitions = [c for c in disagreements if c.startswith("transition_")]
        mask = disagreements[transitions].ne("same_correctness").any(axis=1)
        disagreements.loc[mask].to_parquet(directory / "rescue_harm_rows.parquet", index=False)
    text += ["", "单层 probe 的 oracle 只是这些已训练 probe 的错误并集诊断，不能当作可部署融合收益或严格上界。",
             "若任何诊断促使方法修改，需要将后续结果标明探索性；正式验证、测试仍独立保留。", ""]
    dump(directory / "results.json", result)
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "probes", "heads", "report", "run"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Bound CPU work while another job trains the A backbone")
    parser.add_argument("--fold", type=int, default=None, help="Run one registered outer fold; report requires all folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Shared inner-dev-triggered extension cap, at most 40 (milestones 24/40)")
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--joint-features", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    cache, metadata, manifest = load_b_cache(args.cache_manifest)
    protocol = initialize(args, metadata, manifest)
    check_stop(protocol)
    output = Path(args.output).resolve()
    folds = protocol["outer_folds"] if args.fold is None else [args.fold]
    if any(fold not in protocol["outer_folds"] for fold in folds):
        raise ValueError("Requested unregistered outer fold")
    if args.phase in ("probes", "run"):
        for fold in folds:
            train_probe_fold(cache, metadata, protocol, output, fold, args.device)
    if args.phase in ("heads", "run"):
        for fold in folds:
            train_head_fold(cache, metadata, protocol, output, fold, args.device)
    if args.phase == "report" or (args.phase == "run" and args.fold is None):
        write_report(output, protocol, metadata, heads=True)
    elif args.phase == "probes" and args.fold is None:
        write_report(output, protocol, metadata, heads=False)


if __name__ == "__main__":
    main()
