"""Input-conditioned layer gating on the frozen-LoRA feature cache.

This follow-up keeps the adapter and cached last-token block states fixed.  It
trains three byte-identical heads per seed: a repeated-final-layer control, a
gate over fixed layers 8/16/24/32, and a gate over the validation-selected
layers.  The repeated control executes the same value projections, additive
gate, fusion residual, dropout, and classifier; only its four inputs are copies
of layer 32.  This isolates access to distinct layers from head capacity.

``train`` reads only train/validation caches.  ``lock`` hashes every selected
checkpoint and threshold before ``evaluate`` may open the existing test cache.
The source test set has already been explored, so the result remains a post-hoc
diagnostic rather than a new blind estimate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.hanguard import frozen_lora_layer_diagnostics as frozen  # noqa: E402


DEFAULT_SOURCE = ROOT / "outputs/hanguard_frozen_lora_layer_diagnostics_20261012"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_dynamic_gate_diagnostics_20261012"
FIXED_LAYERS = (8, 16, 24, 32)
ARMS = ("dynamic_repeat_last", "dynamic_fixed", "dynamic_selected")
PRIMARY_THRESHOLD = 0.5
STATIC_ARM_PAIRS = (
    ("dynamic_repeat_last", "repeat_last"),
    ("dynamic_fixed", "fusion_fixed"),
    ("dynamic_selected", "fusion_selected"),
)
EVALUATION_AMENDMENT_SCHEMA = "hanguard_dynamic_gate_evaluation_amendment_v1"
PRIMARY_REPRODUCIBILITY_SCHEMA = "hanguard_dynamic_gate_primary_reproducibility_v1"
EVALUATION_AMENDMENT_FILE = "evaluation_amendment.json"
PRIMARY_REPRODUCIBILITY_FILE = "evaluation_primary_reproducibility.json"
RECOVERY_PARTIAL_SEEDS = (42, 43, 44, 45)
RECOVERY_STATIC_SEEDS = (42, 43, 44)


class DynamicLayerGateHead(nn.Module):
    """Four-branch residual fusion with an input-conditioned additive gate."""

    def __init__(
        self,
        hidden_size: int,
        width: int = 128,
        gate_width: int = 64,
        layer_count: int = 4,
        dropout: float = 0.1,
        fusion_init_scale: float = 1e-3,
    ):
        super().__init__()
        if min(hidden_size, width, gate_width, layer_count) < 1:
            raise ValueError("All model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not math.isfinite(fusion_init_scale) or fusion_init_scale <= 0:
            raise ValueError("fusion_init_scale must be finite and positive")
        self.hidden_size = int(hidden_size)
        self.width = int(width)
        self.gate_width = int(gate_width)
        self.layer_count = int(layer_count)
        self.project = nn.Linear(hidden_size, width)
        self.layer_values = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, width),
                    nn.GELU(),
                    nn.LayerNorm(width),
                )
                for _ in range(layer_count)
            ]
        )
        # Additive attention: each sample's final-layer query is compared with
        # every projected layer value.  layer_bias is the learned global prior.
        self.gate_query = nn.Linear(width, gate_width)
        self.gate_key = nn.Linear(width, gate_width)
        self.gate_score = nn.Linear(gate_width, 1, bias=False)
        self.layer_bias = nn.Parameter(torch.zeros(layer_count))
        self.fusion_output = nn.Linear(width, width)
        # A tiny identity residual keeps the initial predictor close to the
        # final-layer direct path while giving value and gate branches a real
        # gradient on optimization step one.  A zero residual would block both
        # branches until fusion_output itself moved.
        with torch.no_grad():
            self.fusion_output.weight.copy_(
                torch.eye(width, dtype=self.fusion_output.weight.dtype) * fusion_init_scale
            )
        nn.init.zeros_(self.fusion_output.bias)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (self.layer_count, self.hidden_size):
            raise ValueError(
                f"Expected [batch,{self.layer_count},{self.hidden_size}] layer features"
            )
        normalized = F.layer_norm(features.float(), (self.hidden_size,))
        direct = F.gelu(self.project(normalized[:, -1]))
        values = torch.stack(
            [module(normalized[:, index]) for index, module in enumerate(self.layer_values)],
            dim=1,
        )
        query = self.gate_query(direct)[:, None, :]
        keys = self.gate_key(values)
        conditional_logits = self.gate_score(torch.tanh(keys + query)).squeeze(-1)
        gate_logits = conditional_logits + self.layer_bias
        weights = gate_logits.softmax(dim=1)
        fused = (weights[..., None] * values).sum(dim=1)
        representation = direct + self.fusion_output(fused)
        logits = self.classifier(self.dropout(representation)).squeeze(-1)
        return dict(
            logits=logits,
            representation=representation,
            gate_logits=gate_logits,
            gate_weights=weights,
            layer_values=values,
        )


def arm_layers(arm: str, selected_layers: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(map(int, selected_layers))
    if len(selected) != 4 or tuple(sorted(set(selected))) != selected or selected[-1] != 32:
        raise ValueError("Selected layers must be four increasing layers anchored at 32")
    if arm == "dynamic_repeat_last":
        return (32, 32, 32, 32)
    if arm == "dynamic_fixed":
        return FIXED_LAYERS
    if arm == "dynamic_selected":
        return selected
    raise ValueError(f"Unknown dynamic-gate arm: {arm}")


def select_arm_features(
    union_features: torch.Tensor,
    union_layers: Sequence[int],
    arm: str,
    selected_layers: Sequence[int],
) -> torch.Tensor:
    """Select four inputs, explicitly repeating layer 32 for the control."""
    if union_features.ndim != 3 or union_features.shape[1] != len(union_layers):
        raise ValueError("Union features and layer labels do not align")
    positions = {int(layer): index for index, layer in enumerate(union_layers)}
    if len(positions) != len(union_layers):
        raise ValueError("Union layer labels must be unique")
    layers = arm_layers(arm, selected_layers)
    if any(layer not in positions for layer in layers):
        raise ValueError(f"Union cache view lacks layers for {arm}: {layers}")
    return torch.stack([union_features[:, positions[layer]] for layer in layers], dim=1)


def build_matched_heads(
    hidden_size: int,
    width: int,
    gate_width: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, DynamicLayerGateHead], dict]:
    heads = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        heads[arm] = DynamicLayerGateHead(hidden_size, width, gate_width).to(device)
    audit = frozen.assert_capacity_matched_initialization(heads)
    return heads, audit


def gate_diagnostics(
    weights: np.ndarray,
    layer_labels: Sequence[int],
    labels: Sequence[int] | None = None,
) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(layer_labels) or not len(weights):
        raise ValueError("Expected nonempty [rows,layers] gate weights")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("Gate weights must be finite row-wise probabilities")
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1)
    top = weights.argmax(1)
    slots = []
    for index, layer in enumerate(layer_labels):
        slots.append(
            dict(
                slot=index,
                layer=int(layer),
                mean=float(weights[:, index].mean()),
                std=float(weights[:, index].std()),
                minimum=float(weights[:, index].min()),
                maximum=float(weights[:, index].max()),
                top1_fraction=float((top == index).mean()),
            )
        )
    result = dict(
        rows=int(len(weights)),
        slots=slots,
        mean_entropy=float(entropy.mean()),
        normalized_mean_entropy=float(entropy.mean() / math.log(weights.shape[1]))
        if weights.shape[1] > 1
        else 0.0,
        mean_effective_layers=float(np.exp(entropy).mean()),
        mean_l1_from_global_mean=float(np.abs(weights - weights.mean(0)).sum(1).mean()),
        maximum_slot_std=float(weights.std(0).max()),
    )
    if labels is not None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (len(weights),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Binary labels must align with gate rows")
        result["mean_by_label"] = {
            str(label): weights[labels == label].mean(0).tolist() if (labels == label).any() else None
            for label in (0, 1)
        }
    return result


def validation_thresholds(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    """Pre-registered primary threshold plus validation-only sensitivity value."""
    tuned = frozen.accuracy_threshold(labels, probabilities)
    return dict(primary=float(PRIMARY_THRESHOLD), validation_accuracy_sensitivity=float(tuned))


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_identity(source: Path) -> dict:
    selected_path = source / "selected_layers.json"
    protocol_path = source / "protocol.json"
    if not selected_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Frozen-feature study is incomplete")
    selected = _load_json(selected_path)
    # Enforce the original validation-only selection record.
    if selected.get("split") != "validation":
        raise ValueError("Source layer selection was not validation-only")
    arm_layers("dynamic_selected", selected["layers"])
    manifests = {}
    for split in ("train", "validation", "test"):
        path = source / f"features/{split}.manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifests[split] = frozen.sha256(path)
    return dict(
        source_protocol_sha256=frozen.sha256(protocol_path),
        selected_layers_sha256=frozen.sha256(selected_path),
        selected_layers=list(map(int, selected["layers"])),
        cache_manifest_sha256=manifests,
    )


def initialize(args: argparse.Namespace) -> dict:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = _source_identity(source)
    source_protocol = _load_json(source / "protocol.json")
    positive = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width=args.width,
        gate_width=args.gate_width,
        repetitions=args.repetitions,
    )
    if any(value <= 0 for value in positive.values()) or args.weight_decay < 0:
        raise ValueError(f"Positive settings required: {positive}")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    protocol = dict(
        version=1,
        study="input-conditioned dynamic layer gate on one frozen E03_s42 LoRA cache",
        source=str(source),
        **identity,
        adapter_sha256=source_protocol["adapter_sha256"],
        data_dir=source_protocol["data_dir"],
        data_sha256=source_protocol["data_sha256"],
        rows=source_protocol["rows"],
        fixed_layers=list(FIXED_LAYERS),
        arms=list(ARMS),
        architecture=dict(
            width=int(args.width),
            gate_width=int(args.gate_width),
            layer_count=4,
            dropout=float(args.dropout),
            gate="per-sample additive attention over independently projected layer values",
            direct_path="layer-32 projection plus 1e-3 identity-initialized fusion residual",
            fusion_init_scale=1e-3,
        ),
        training=dict(
            seeds=list(map(int, args.seeds)),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            checkpoint_selection="minimum validation BCE independently per arm and seed",
            threshold_selection=(
                "primary decision threshold pre-registered at 0.5; validation accuracy optimum "
                "is sensitivity analysis only"
            ),
            calibration="none in this diagnostic; use out-of-fold calibration before changing the primary threshold",
            common_batch_order=True,
            paired_dropout_stream=True,
        ),
        analysis=dict(repetitions=int(args.repetitions), seed=int(args.analysis_seed)),
        code_sha256={
            "dynamic_gate_diagnostics.py": frozen.sha256(Path(__file__)),
            "frozen_lora_layer_diagnostics.py": frozen.sha256(Path(frozen.__file__)),
        },
        primary_estimand=(
            "dynamic_fixed/dynamic_selected minus dynamic_repeat_last under identical capacity, "
            "initialization, batch order, dropout stream, frozen features, and validation rules"
        ),
        limitations=[
            "The frozen adapter was trained jointly with an E03 final-layer head.",
            "Head seeds are not independent LoRA seeds.",
            "The source test split has already been explored; this is a post-hoc diagnostic.",
            "The selected-layer arm reuses a validation-driven exploratory layer selection.",
            "The primary threshold is fixed at 0.5; no out-of-fold calibration is fitted in this diagnostic.",
            "Per-source and gate-weight analyses are descriptive and unadjusted for multiplicity.",
        ],
    )
    path = output / "protocol.json"
    if path.exists():
        if _load_json(path) != protocol:
            raise ValueError("Requested settings differ from the existing dynamic-gate protocol")
    else:
        frozen.dump(path, protocol)
    return protocol


def load_protocol(output: Path) -> dict:
    output = Path(output).resolve()
    path = output / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run init")
    protocol = _load_json(path)
    source = Path(protocol["source"])
    current = _source_identity(source)
    for key in ("source_protocol_sha256", "selected_layers_sha256", "selected_layers", "cache_manifest_sha256"):
        if current[key] != protocol[key]:
            raise ValueError(f"Frozen-feature source identity changed: {key}")
    return protocol


def _load_cache(protocol: Mapping[str, Any], split: str):
    source = Path(protocol["source"])
    source_protocol = _load_json(source / "protocol.json")
    return frozen._load_cache(source, split, source_protocol)


def _feature_batch(
    cache: np.memmap,
    indices: np.ndarray,
    layers: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    array = np.stack(
        [np.asarray(cache[indices, int(layer) - 1, :], dtype=np.float32) for layer in layers],
        axis=1,
    )
    return torch.from_numpy(array).to(device=device, non_blocking=True)


def _union_layers(selected: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(FIXED_LAYERS) | set(map(int, selected))))


@torch.no_grad()
def predict(
    heads: Mapping[str, DynamicLayerGateHead],
    cache: np.memmap,
    selected: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    union = _union_layers(selected)
    logits = {arm: np.empty(len(cache), dtype=np.float32) for arm in heads}
    weights = {arm: np.empty((len(cache), 4), dtype=np.float32) for arm in heads}
    for head in heads.values():
        head.eval()
    for start in range(0, len(cache), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache)))
        raw = _feature_batch(cache, indices, union, device)
        for arm, head in heads.items():
            output = head(select_arm_features(raw, union, arm, selected))
            logits[arm][indices] = output["logits"].float().cpu().numpy()
            weights[arm][indices] = output["gate_weights"].float().cpu().numpy()
    if any(not np.isfinite(value).all() for value in itertools.chain(logits.values(), weights.values())):
        raise FloatingPointError("Nonfinite dynamic-gate predictions")
    return logits, weights


def _state_copy(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _gate_columns(frame: pd.DataFrame, arm: str, weights: np.ndarray) -> None:
    for slot in range(weights.shape[1]):
        frame[f"gate_{arm}_slot_{slot + 1}"] = weights[:, slot]


def train(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    complete = output / "heads/selection.json"
    if complete.is_file():
        return _load_json(complete)
    train_cache, train_meta, _ = _load_cache(protocol, "train")
    validation_cache, validation_meta, _ = _load_cache(protocol, "validation")
    selected = protocol["selected_layers"]
    settings = protocol["training"]
    device = torch.device(args.device)
    y_train = torch.as_tensor(train_meta.label.to_numpy(), dtype=torch.float32, device=device)
    y_validation = validation_meta.label.to_numpy(dtype=np.int64)
    union = _union_layers(selected)
    entries = []
    for seed in settings["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        selection_path = directory / "selection.json"
        if selection_path.is_file():
            entries.append(_load_json(selection_path))
            continue
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        heads, matched = build_matched_heads(
            train_cache.shape[-1], protocol["architecture"]["width"],
            protocol["architecture"]["gate_width"], seed, device,
        )
        initial_hashes = {arm: frozen.state_digest(head) for arm, head in heads.items()}
        optimizers = {
            arm: torch.optim.AdamW(
                head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
            )
            for arm, head in heads.items()
        }
        best_bce = {arm: float("inf") for arm in ARMS}
        best_epoch = {arm: None for arm in ARMS}
        best_state: dict[str, dict[str, torch.Tensor] | None] = {arm: None for arm in ARMS}
        history = []
        for epoch in range(settings["epochs"]):
            order = np.random.default_rng(seed * 1009 + epoch).permutation(len(train_cache))
            totals = {arm: 0.0 for arm in ARMS}
            rows = 0
            for step, start in enumerate(range(0, len(order), settings["batch_size"])):
                indices = order[start : start + settings["batch_size"]]
                raw = _feature_batch(train_cache, indices, union, device)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                losses = []
                for arm, head in heads.items():
                    head.train()
                    dropout_seed = (seed * 1000003 + epoch * 1009 + step) % (2**63 - 1)
                    torch.manual_seed(dropout_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(dropout_seed)
                    result = head(select_arm_features(raw, union, arm, selected))
                    loss = F.binary_cross_entropy_with_logits(result["logits"], y_train[indices])
                    losses.append(loss)
                    totals[arm] += float(loss.detach()) * len(indices)
                torch.stack(losses).sum().backward()
                for arm, optimizer in optimizers.items():
                    nn.utils.clip_grad_norm_(heads[arm].parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                rows += len(indices)
            validation_logits, _ = predict(
                heads, validation_cache, selected, settings["batch_size"], device
            )
            targets = torch.as_tensor(y_validation, dtype=torch.float32)
            record = dict(epoch=epoch + 1, train_bce={}, validation_bce={})
            for arm in ARMS:
                value = float(
                    F.binary_cross_entropy_with_logits(
                        torch.from_numpy(validation_logits[arm]), targets
                    )
                )
                record["train_bce"][arm] = totals[arm] / rows
                record["validation_bce"][arm] = value
                if value < best_bce[arm]:
                    best_bce[arm] = value
                    best_epoch[arm] = epoch + 1
                    best_state[arm] = _state_copy(heads[arm])
            history.append(record)
            frozen.dump(directory / "history.json", history)
            print(
                json.dumps(
                    dict(state="dynamic_gate_train", seed=seed, epoch=epoch + 1, epochs=settings["epochs"])
                ),
                flush=True,
            )
        for arm in ARMS:
            if best_state[arm] is None:
                raise RuntimeError(f"No finite validation checkpoint for {arm}")
            heads[arm].load_state_dict(best_state[arm])
        validation_logits, validation_weights = predict(
            heads, validation_cache, selected, settings["batch_size"], device
        )
        frame = validation_meta.copy()
        records = {}
        for arm in ARMS:
            checkpoint_path = directory / f"{arm}.pt"
            checkpoint = dict(
                state_dict=best_state[arm],
                arm=arm,
                seed=int(seed),
                selected_layers=list(selected),
                input_layers=list(arm_layers(arm, selected)),
                hidden_size=int(train_cache.shape[-1]),
                width=int(protocol["architecture"]["width"]),
                gate_width=int(protocol["architecture"]["gate_width"]),
                best_epoch=int(best_epoch[arm]),
                best_validation_bce=float(best_bce[arm]),
                protocol_sha256=frozen.sha256(output / "protocol.json"),
            )
            frozen.atomic_torch_save(checkpoint, checkpoint_path)
            probability = frozen.sigmoid(validation_logits[arm])
            thresholds = validation_thresholds(y_validation, probability)
            threshold = thresholds["primary"]
            frame[f"probability_{arm}"] = probability
            frame[f"prediction_{arm}"] = (probability >= threshold).astype(np.int8)
            frame[f"prediction_{arm}_validation_tuned_sensitivity"] = (
                probability >= thresholds["validation_accuracy_sensitivity"]
            ).astype(np.int8)
            _gate_columns(frame, arm, validation_weights[arm])
            records[arm] = dict(
                parameters=sum(parameter.numel() for parameter in heads[arm].parameters()),
                initial_state_sha256=initial_hashes[arm],
                best_epoch=int(best_epoch[arm]),
                validation_bce=float(best_bce[arm]),
                threshold=float(threshold),
                validation_accuracy_sensitivity_threshold=thresholds[
                    "validation_accuracy_sensitivity"
                ],
                validation_metrics=frozen.binary_metrics(y_validation, probability, threshold),
                validation_metrics_accuracy_tuned_sensitivity=frozen.binary_metrics(
                    y_validation, probability, thresholds["validation_accuracy_sensitivity"]
                ),
                validation_gate=gate_diagnostics(
                    validation_weights[arm], arm_layers(arm, selected), y_validation
                ),
                checkpoint_sha256=frozen.sha256(checkpoint_path),
            )
        predictions_path = directory / "validation_predictions.parquet"
        frame.to_parquet(predictions_path, index=False)
        selection = dict(
            split="validation",
            seed=int(seed),
            selected_layers=list(selected),
            common_epoch_budget=int(settings["epochs"]),
            common_batch_order=True,
            paired_dropout_stream=True,
            matched_initialization=matched,
            arms=records,
            validation_predictions_sha256=frozen.sha256(predictions_path),
        )
        frozen.dump(selection_path, selection)
        entries.append(selection)
        del heads, optimizers
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = dict(
        split="validation",
        seeds=settings["seeds"],
        selected_layers=list(selected),
        matched_arms=list(ARMS),
        entries=entries,
        statement=(
            "All three dynamic-gate arms have identical parameter layouts and per-seed initial values; "
            "the repeat control copies layer 32 into all four inputs."
        ),
    )
    frozen.dump(complete, summary)
    return summary


def _barrier_paths(output: Path, protocol: Mapping[str, Any]) -> list[Path]:
    source = Path(protocol["source"])
    paths = [
        output / "protocol.json",
        source / "protocol.json",
        source / "selected_layers.json",
        source / "features/train.manifest.json",
        source / "features/validation.manifest.json",
        source / "features/test.manifest.json",
        output / "heads/selection.json",
        Path(protocol["data_dir"]) / "test.parquet",
    ]
    for seed in protocol["training"]["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        paths.extend([directory / "selection.json", directory / "validation_predictions.parquet"])
        paths.extend(directory / f"{arm}.pt" for arm in ARMS)
    return paths


def _barrier_identity(output: Path, protocol: Mapping[str, Any]) -> dict:
    return dict(
        protocol_sha256=frozen.sha256(output / "protocol.json"),
        source_protocol_sha256=protocol["source_protocol_sha256"],
        selected_layers_sha256=protocol["selected_layers_sha256"],
        adapter_sha256=protocol["adapter_sha256"],
        train_sha256=protocol["data_sha256"]["train"],
        validation_sha256=protocol["data_sha256"]["validation"],
    )


def lock(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    summary = train(args)
    if summary.get("split") != "validation" or len(summary.get("entries", [])) != len(protocol["training"]["seeds"]):
        raise ValueError("All validation selections must be complete before locking test")
    for entry in summary["entries"]:
        if entry.get("split") != "validation" or set(entry.get("arms", {})) != set(ARMS):
            raise ValueError("Incomplete or non-validation head selection")
    return frozen.create_test_barrier(
        output,
        _barrier_paths(output, protocol),
        _barrier_identity(output, protocol),
    )


def _load_head(
    path: Path,
    arm: str,
    seed: int,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> DynamicLayerGateHead:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = dict(
        arm=arm,
        seed=int(seed),
        selected_layers=list(protocol["selected_layers"]),
        protocol_sha256=frozen.sha256(Path(path).parents[2] / "protocol.json"),
    )
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint identity differs for {arm}: {key}")
    head = DynamicLayerGateHead(
        checkpoint["hidden_size"], checkpoint["width"], checkpoint["gate_width"]
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    return head


def _by_source(metadata: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    return {
        str(source): frozen.binary_metrics(
            group.label.to_numpy(), probability[group.index.to_numpy()], threshold
        )
        for source, group in metadata.reset_index(drop=True).groupby("source", sort=True)
    }


def _transition(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict:
    candidate_correct = np.asarray(candidate, dtype=bool) == y
    baseline_correct = np.asarray(baseline, dtype=bool) == y
    return dict(
        both_correct=int((candidate_correct & baseline_correct).sum()),
        candidate_only_correct=int((candidate_correct & ~baseline_correct).sum()),
        baseline_only_correct=int((~candidate_correct & baseline_correct).sum()),
        both_wrong=int((~candidate_correct & ~baseline_correct).sum()),
        net_corrections=int((candidate_correct & ~baseline_correct).sum() - (~candidate_correct & baseline_correct).sum()),
    )


def _comparison(
    y: np.ndarray,
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_threshold: float,
    baseline_threshold: float,
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    candidate = candidate_probability >= candidate_threshold
    baseline = baseline_probability >= baseline_threshold
    return dict(
        transition=_transition(y, candidate, baseline),
        mcnemar_exact=frozen.mcnemar_exact(y, candidate, baseline),
        group_bootstrap=frozen.group_cluster_bootstrap(
            y, candidate_probability, baseline_probability, groups,
            candidate_threshold, baseline_threshold, repetitions, seed,
        ),
        group_permutation=frozen.group_permutation_test(
            y, candidate_probability, baseline_probability, groups,
            candidate_threshold, baseline_threshold, repetitions, seed,
        ),
    )


def _seed_statistics(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else None,
        minimum=float(values.min()),
        maximum=float(values.max()),
        seeds=int(len(values)),
    )


def _source_static_predictions(
    protocol: Mapping[str, Any], seed: int, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    source = Path(protocol["source"])
    frame = pd.read_parquet(source / f"heads/seed_{seed}/test_predictions.parquet")
    if not np.array_equal(frame.base_id.astype(str), metadata.base_id.astype(str)) or not np.array_equal(
        frame.label.to_numpy(), metadata.label.to_numpy()
    ):
        raise ValueError("Source static-head test rows do not align with dynamic-gate rows")
    selection = _load_json(source / f"heads/seed_{seed}/selection.json")
    return frame, selection


def discover_static_comparison_seeds(
    protocol: Mapping[str, Any], dynamic_seeds: Sequence[int]
) -> dict:
    """Find the real seed intersection for the secondary static comparison.

    Dynamic checkpoints are protected by this study's test barrier.  Static
    predictions are pre-existing, secondary source artifacts and may cover a
    smaller seed set; absent seeds are reported rather than synthesized.
    """
    source_heads = Path(protocol["source"]) / "heads"
    dynamic_seeds = [int(seed) for seed in dynamic_seeds]
    source_available = []
    if source_heads.is_dir():
        for directory in source_heads.glob("seed_*"):
            try:
                seed = int(directory.name.removeprefix("seed_"))
            except ValueError:
                continue
            if (
                (directory / "test_predictions.parquet").is_file()
                and (directory / "selection.json").is_file()
            ):
                source_available.append(seed)
    source_available = sorted(set(source_available))
    common = [seed for seed in dynamic_seeds if seed in set(source_available)]
    excluded = [seed for seed in dynamic_seeds if seed not in set(common)]
    missing_artifacts = {}
    for seed in excluded:
        directory = source_heads / f"seed_{seed}"
        missing_artifacts[str(seed)] = [
            name
            for name in ("test_predictions.parquet", "selection.json")
            if not (directory / name).is_file()
        ]
    hashes = {
        str(seed): {
            "test_predictions_sha256": frozen.sha256(
                source_heads / f"seed_{seed}/test_predictions.parquet"
            ),
            "selection_sha256": frozen.sha256(source_heads / f"seed_{seed}/selection.json"),
        }
        for seed in common
    }
    return dict(
        policy="intersection_only; missing static seeds are neither copied nor synthesized",
        dynamic_available_seeds=dynamic_seeds,
        source_static_available_seeds=source_available,
        common_seeds=common,
        n_common_seeds=len(common),
        excluded_dynamic_seeds=excluded,
        missing_source_artifacts=missing_artifacts,
        source_artifact_sha256=hashes,
    )


def aggregate_static_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Aggregate secondary comparisons and retain their exact seed support."""
    aggregates = []
    for candidate, baseline in STATIC_ARM_PAIRS:
        selected = [
            row for row in rows
            if row["candidate"] == candidate and row["baseline"] == baseline
        ]
        seeds = [int(row["seed"]) for row in selected]
        aggregates.append(
            dict(
                candidate=candidate,
                baseline=baseline,
                capacity_matched=False,
                seeds=seeds,
                n_seeds=len(seeds),
                accuracy_delta=(
                    _seed_statistics(
                        [row["group_bootstrap"]["accuracy"]["delta"] for row in selected]
                    )
                    if selected else None
                ),
                f1_delta=(
                    _seed_statistics(
                        [row["group_bootstrap"]["f1"]["delta"] for row in selected]
                    )
                    if selected else None
                ),
            )
        )
    return aggregates


def _artifact_record(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=frozen.sha256(path))


def _verify_artifact_record(name: str, record: Mapping[str, Any]) -> Path:
    path = Path(record.get("path", "")).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or frozen.sha256(path) != record.get("sha256")
    ):
        raise ValueError(f"Evaluation-amendment artifact changed: {name}")
    return path


def _definition_hashes(path: Path) -> dict[str, str]:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    output = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise ValueError(f"Cannot recover source for top-level definition {node.name}")
            output[node.name] = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return output


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode("ascii"))
    result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    result.update(array.tobytes())
    return result.hexdigest()


def _prediction_snapshot(path: Path, expected_rows: int) -> dict:
    frame = pd.read_parquet(path)
    required = {"base_id", "group_id", "label"}
    required.update(f"probability_{arm}" for arm in ARMS)
    required.update(f"prediction_{arm}" for arm in ARMS)
    required.update(f"gate_{arm}_slot_{slot}" for arm in ARMS for slot in range(1, 5))
    if len(frame) != int(expected_rows) or not required.issubset(frame.columns):
        raise ValueError(f"Partial test prediction layout changed: {path}")
    if frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError(f"Partial test prediction row identity changed: {path}")
    numeric = frame[sorted(required - {"base_id", "group_id"})].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not frame.label.isin((0, 1)).all():
        raise ValueError(f"Partial test prediction contains invalid values: {path}")
    return dict(
        artifact=_artifact_record(path),
        rows=len(frame),
        columns=list(frame.columns),
        dtypes=[str(value) for value in frame.dtypes],
        ordered_base_id_sha256=frozen.ordered_base_id_sha256(frame.base_id.astype(str)),
        ordered_group_id_sha256=frozen.ordered_base_id_sha256(frame.group_id.astype(str)),
        label_sha256=_array_sha256(frame.label.to_numpy(dtype=np.int8)),
    )


def _recovery_provenance_paths(output: Path) -> dict[str, Path]:
    directory = Path(output).resolve() / "provenance"
    return {
        "locked_source_snapshot": directory / "dynamic_gate_diagnostics.locked_580130.py",
        "intersection_source_snapshot": directory / "dynamic_gate_diagnostics.intersection_50dc9f.py",
        "locked_to_intersection_patch": directory / "locked_to_intersection.patch",
        "intersection_to_amendment_gate_patch": directory / "intersection_to_amendment_gate.patch",
    }


def create_evaluation_amendment(args: argparse.Namespace) -> dict:
    """Register the post-barrier evaluator repair without changing the barrier."""
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    path = output / EVALUATION_AMENDMENT_FILE
    if path.is_file():
        return verify_evaluation_amendment(output, protocol, require_reproducibility=False)
    if (output / "results.json").exists() or (output / "report.md").exists():
        raise RuntimeError("The evaluation amendment must be registered before results/report exist")
    barrier_path = output / "test_barrier.json"
    frozen.verify_test_barrier(
        output, _barrier_paths(output, protocol), _barrier_identity(output, protocol)
    )
    barrier_before = frozen.sha256(barrier_path)
    protocol_before = frozen.sha256(output / "protocol.json")
    provenance = _recovery_provenance_paths(output)
    provenance_records = {name: _artifact_record(value) for name, value in provenance.items()}
    locked_code = protocol["code_sha256"]["dynamic_gate_diagnostics.py"]
    if provenance_records["locked_source_snapshot"]["sha256"] != locked_code:
        raise ValueError("Locked evaluator snapshot does not reproduce the protocol code hash")
    intersection_code = provenance_records["intersection_source_snapshot"]["sha256"]
    current_code = frozen.sha256(Path(__file__))
    if intersection_code == locked_code or current_code in (locked_code, intersection_code):
        raise ValueError("Evaluation amendment must identify three distinct code stages")

    old_definitions = _definition_hashes(provenance["locked_source_snapshot"])
    intersection_definitions = _definition_hashes(provenance["intersection_source_snapshot"])
    current_definitions = _definition_hashes(Path(__file__))
    primary_definitions = (
        "DynamicLayerGateHead", "select_arm_features", "build_matched_heads",
        "gate_diagnostics", "validation_thresholds", "predict", "_gate_columns",
        "_load_head", "_comparison",
    )
    for name in primary_definitions:
        values = (old_definitions.get(name), intersection_definitions.get(name), current_definitions.get(name))
        if None in values or len(set(values)) != 1:
            raise ValueError(f"Primary prediction definition changed across evaluator repair: {name}")

    settings = protocol["training"]
    expected_partial = list(RECOVERY_PARTIAL_SEEDS)
    partial_paths = {
        seed: output / "heads" / f"seed_{seed}" / "test_predictions.parquet"
        for seed in map(int, settings["seeds"])
    }
    actual_partial = [seed for seed, value in partial_paths.items() if value.is_file()]
    if actual_partial != expected_partial:
        raise ValueError(f"Unexpected partial dynamic prediction set: {actual_partial}")
    partial_predictions = {
        str(seed): _prediction_snapshot(partial_paths[seed], protocol["rows"]["test"])
        for seed in expected_partial
    }
    source_test_manifest = _load_json(Path(protocol["source"]) / "features/test.manifest.json")
    if any(
        snapshot["ordered_base_id_sha256"] != source_test_manifest["ordered_base_id_sha256"]
        for snapshot in partial_predictions.values()
    ):
        raise ValueError("Partial dynamic prediction rows do not match the locked test cache")
    identity_fields = ("rows", "columns", "dtypes", "ordered_base_id_sha256", "ordered_group_id_sha256", "label_sha256")
    if len({tuple(json.dumps(snapshot[field], sort_keys=True) for field in identity_fields)
            for snapshot in partial_predictions.values()}) != 1:
        raise ValueError("Partial dynamic prediction row/schema identities differ across seeds")
    static_scope = discover_static_comparison_seeds(protocol, settings["seeds"])
    if (
        static_scope["common_seeds"] != list(RECOVERY_STATIC_SEEDS)
        or static_scope["excluded_dynamic_seeds"] != [45, 46]
    ):
        raise ValueError(f"Unexpected source-static seed support: {static_scope}")

    artifacts = {
        "protocol": _artifact_record(output / "protocol.json"),
        "test_barrier": _artifact_record(barrier_path),
        "amended_evaluator": _artifact_record(Path(__file__)),
        **{f"provenance/{key}": value for key, value in provenance_records.items()},
    }
    for seed, snapshot in partial_predictions.items():
        artifacts[f"partial_dynamic_prediction/{seed}"] = snapshot["artifact"]
    source = Path(protocol["source"])
    for seed in RECOVERY_STATIC_SEEDS:
        artifacts[f"source_static_prediction/{seed}"] = _artifact_record(
            source / f"heads/seed_{seed}/test_predictions.parquet"
        )
        artifacts[f"source_static_selection/{seed}"] = _artifact_record(
            source / f"heads/seed_{seed}/selection.json"
        )

    value = dict(
        schema=EVALUATION_AMENDMENT_SCHEMA,
        state="locked_before_amended_evaluation",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol_sha256=protocol_before,
        original_test_barrier_sha256=barrier_before,
        locked_evaluation_code_sha256=locked_code,
        intersection_patch_code_sha256=intersection_code,
        amended_evaluation_code_sha256=current_code,
        reason=dict(
            failure=(
                "The locked evaluator assumed a source static test prediction for every dynamic head seed. "
                "Only source static seeds 42/43/44 existed, so evaluation stopped at seed 45 after its "
                "dynamic prediction parquet was written and before results.json/report.md were created."
            ),
            evidence_basis="locked control flow plus filesystem artifact state after the failed run",
            failed_secondary_operation="load source static seed-45 test_predictions.parquet",
            primary_model_or_threshold_change=False,
        ),
        partial_test_access=dict(
            dynamic_forward_completed_seeds=expected_partial,
            dynamic_forward_not_started_seeds=[46],
            predictions=partial_predictions,
            results_json_existed=False,
            report_md_existed=False,
        ),
        repair_scope=dict(
            primary_matched_comparison_seeds=list(map(int, settings["seeds"])),
            secondary_static_seed_policy="exact locked intersection only",
            secondary_static_seed_scope=static_scope,
            training_checkpoints_thresholds_and_original_barrier_unchanged=True,
        ),
        primary_invariance_code_evidence=dict(
            exact_definition_hash_match=True,
            definitions=list(primary_definitions),
            sha256={name: old_definitions[name] for name in primary_definitions},
            locked_to_intersection_patch_sha256=provenance_records[
                "locked_to_intersection_patch"
            ]["sha256"],
            intersection_to_amendment_gate_patch_sha256=provenance_records[
                "intersection_to_amendment_gate_patch"
            ]["sha256"],
        ),
        required_primary_reproducibility_audit=dict(
            path=str((output / PRIMARY_REPRODUCIBILITY_FILE).resolve()),
            seeds=expected_partial,
            comparison="exact row/schema/dtype/probability/decision/gate equality in memory",
            must_complete_before_formal_evaluate=True,
        ),
        artifacts=artifacts,
        rule=(
            "This append-only amendment does not replace or relock the original test barrier. "
            "Formal evaluation must verify this file and an exact 42-45 primary reproducibility audit."
        ),
    )
    if frozen.sha256(barrier_path) != barrier_before or frozen.sha256(output / "protocol.json") != protocol_before:
        raise AssertionError("Original protocol or test barrier changed during amendment creation")
    frozen.dump(path, value)
    path.chmod(0o444)
    return dict(value, amendment_sha256=frozen.sha256(path))


def verify_primary_reproducibility_audit(
    output: Path, protocol: Mapping[str, Any], amendment: Mapping[str, Any]
) -> dict:
    path = Path(output).resolve() / PRIMARY_REPRODUCIBILITY_FILE
    if not path.is_file():
        raise FileNotFoundError("Primary reproducibility audit is required before amended evaluation")
    value = _load_json(path)
    expected_seeds = list(map(int, amendment["required_primary_reproducibility_audit"]["seeds"]))
    test_manifest = _load_json(Path(protocol["source"]) / "features/test.manifest.json")
    if (
        value.get("schema") != PRIMARY_REPRODUCIBILITY_SCHEMA
        or value.get("state") != "exact_primary_reproduction_verified"
        or value.get("protocol_sha256") != frozen.sha256(Path(output) / "protocol.json")
        or value.get("test_barrier_sha256") != frozen.sha256(Path(output) / "test_barrier.json")
        or value.get("evaluation_amendment_sha256")
        != frozen.sha256(Path(output) / EVALUATION_AMENDMENT_FILE)
        or value.get("evaluation_code_sha256") != frozen.sha256(Path(__file__))
        or value.get("source_test_cache_sha256") != test_manifest["sha256"]
        or value.get("seeds") != expected_seeds
        or value.get("all_exact") is not True
    ):
        raise ValueError("Primary reproducibility audit identity changed")
    per_seed = value.get("per_seed", {})
    if set(per_seed) != {str(seed) for seed in expected_seeds}:
        raise ValueError("Primary reproducibility audit seed records changed")
    for seed in expected_seeds:
        key = str(seed)
        record = per_seed[key]
        registered = amendment["partial_test_access"]["predictions"][key]
        checkpoint_sha256 = {
            arm: frozen.sha256(Path(output) / "heads" / f"seed_{seed}" / f"{arm}.pt")
            for arm in ARMS
        }
        numeric = record.get("numeric_columns", {})
        expected_numeric = {
            column for column in registered["columns"]
            if column.startswith(("probability_", "prediction_", "gate_"))
        }
        if (
            record.get("all_exact") is not True
            or int(record.get("rows", -1)) != int(registered["rows"])
            or record.get("columns") != registered["columns"]
            or record.get("dtypes") != registered["dtypes"]
            or record.get("ordered_base_id_sha256") != registered["ordered_base_id_sha256"]
            or record.get("ordered_group_id_sha256") != registered["ordered_group_id_sha256"]
            or record.get("label_sha256") != registered["label_sha256"]
            or record.get("existing_prediction_sha256") != registered["artifact"]["sha256"]
            or record.get("checkpoint_sha256") != checkpoint_sha256
            or set(numeric) != expected_numeric
            or any(
                item.get("exact") is not True
                or item.get("maximum_absolute_difference") != 0.0
                or item.get("existing_array_sha256") != item.get("recomputed_array_sha256")
                for item in numeric.values()
            )
        ):
            raise ValueError(f"Primary reproducibility failed for seed {seed}")
    return dict(value, audit_sha256=frozen.sha256(path))


def verify_evaluation_amendment(
    output: Path,
    protocol: Mapping[str, Any] | None = None,
    *,
    require_reproducibility: bool,
) -> dict:
    output = Path(output).resolve()
    protocol = load_protocol(output) if protocol is None else protocol
    current_code = frozen.sha256(Path(__file__))
    locked_code = protocol["code_sha256"]["dynamic_gate_diagnostics.py"]
    path = output / EVALUATION_AMENDMENT_FILE
    frozen.verify_test_barrier(
        output, _barrier_paths(output, protocol), _barrier_identity(output, protocol)
    )
    if current_code == locked_code and not path.exists():
        return dict(required=False, locked_evaluation_code_sha256=locked_code)
    if not path.is_file():
        raise FileNotFoundError("Evaluator code changed after lock; evaluation_amendment.json is required")
    value = _load_json(path)
    if (
        value.get("schema") != EVALUATION_AMENDMENT_SCHEMA
        or value.get("state") != "locked_before_amended_evaluation"
        or value.get("protocol_sha256") != frozen.sha256(output / "protocol.json")
        or value.get("original_test_barrier_sha256") != frozen.sha256(output / "test_barrier.json")
        or value.get("locked_evaluation_code_sha256") != locked_code
        or value.get("amended_evaluation_code_sha256") != current_code
    ):
        raise ValueError("Evaluation amendment identity changed")
    expected_partial = list(RECOVERY_PARTIAL_SEEDS)
    expected_static = list(RECOVERY_STATIC_SEEDS)
    expected_artifacts = {
        "protocol", "test_barrier", "amended_evaluator",
        "provenance/locked_source_snapshot",
        "provenance/intersection_source_snapshot",
        "provenance/locked_to_intersection_patch",
        "provenance/intersection_to_amendment_gate_patch",
        *(f"partial_dynamic_prediction/{seed}" for seed in expected_partial),
        *(f"source_static_prediction/{seed}" for seed in expected_static),
        *(f"source_static_selection/{seed}" for seed in expected_static),
    }
    artifacts = value.get("artifacts", {})
    if set(artifacts) != expected_artifacts:
        raise ValueError("Evaluation amendment artifact set changed")
    provenance = _recovery_provenance_paths(output)
    expected_paths = {
        "protocol": output / "protocol.json",
        "test_barrier": output / "test_barrier.json",
        "amended_evaluator": Path(__file__),
        **{f"provenance/{name}": path for name, path in provenance.items()},
        **{
            f"partial_dynamic_prediction/{seed}":
            output / "heads" / f"seed_{seed}" / "test_predictions.parquet"
            for seed in expected_partial
        },
        **{
            f"source_static_prediction/{seed}":
            Path(protocol["source"]) / "heads" / f"seed_{seed}" / "test_predictions.parquet"
            for seed in expected_static
        },
        **{
            f"source_static_selection/{seed}":
            Path(protocol["source"]) / "heads" / f"seed_{seed}" / "selection.json"
            for seed in expected_static
        },
    }
    for name, record in artifacts.items():
        verified_path = _verify_artifact_record(name, record)
        if verified_path != expected_paths[name].resolve():
            raise ValueError(f"Evaluation-amendment artifact path changed: {name}")
    registered_scope = value["repair_scope"]["secondary_static_seed_scope"]
    current_scope = discover_static_comparison_seeds(protocol, protocol["training"]["seeds"])
    if current_scope != registered_scope:
        raise ValueError("Source static comparison support changed after evaluation amendment")
    if (
        value.get("partial_test_access", {}).get("dynamic_forward_completed_seeds")
        != expected_partial
        or value.get("partial_test_access", {}).get("dynamic_forward_not_started_seeds") != [46]
        or value.get("repair_scope", {}).get("primary_matched_comparison_seeds")
        != list(map(int, protocol["training"]["seeds"]))
        or value.get("repair_scope", {}).get(
            "training_checkpoints_thresholds_and_original_barrier_unchanged"
        ) is not True
        or registered_scope.get("common_seeds") != expected_static
        or value.get("reason", {}).get("primary_model_or_threshold_change") is not False
    ):
        raise ValueError("Evaluation amendment recovery scope changed")
    invariant = value.get("primary_invariance_code_evidence", {})
    invariant_names = [
        "DynamicLayerGateHead", "select_arm_features", "build_matched_heads",
        "gate_diagnostics", "validation_thresholds", "predict", "_gate_columns",
        "_load_head", "_comparison",
    ]
    locked_definitions = _definition_hashes(provenance["locked_source_snapshot"])
    intersection_definitions = _definition_hashes(provenance["intersection_source_snapshot"])
    current_definitions = _definition_hashes(Path(__file__))
    expected_hashes = {name: locked_definitions[name] for name in invariant_names}
    if (
        invariant.get("exact_definition_hash_match") is not True
        or invariant.get("definitions") != invariant_names
        or invariant.get("sha256") != expected_hashes
        or any(
            locked_definitions[name] != intersection_definitions.get(name)
            or locked_definitions[name] != current_definitions.get(name)
            for name in invariant_names
        )
        or invariant.get("locked_to_intersection_patch_sha256")
        != artifacts["provenance/locked_to_intersection_patch"]["sha256"]
        or invariant.get("intersection_to_amendment_gate_patch_sha256")
        != artifacts["provenance/intersection_to_amendment_gate_patch"]["sha256"]
    ):
        raise ValueError("Evaluation amendment primary-invariance evidence changed")
    registered_predictions = value["partial_test_access"].get("predictions", {})
    if set(registered_predictions) != {str(seed) for seed in expected_partial}:
        raise ValueError("Evaluation amendment partial prediction set changed")
    for seed in expected_partial:
        prediction_path = output / "heads" / f"seed_{seed}" / "test_predictions.parquet"
        current_snapshot = _prediction_snapshot(prediction_path, protocol["rows"]["test"])
        if current_snapshot != registered_predictions[str(seed)]:
            raise ValueError(f"Partial prediction snapshot changed for seed {seed}")
    result = dict(value, required=True, amendment_sha256=frozen.sha256(path))
    if require_reproducibility:
        result["primary_reproducibility"] = verify_primary_reproducibility_audit(
            output, protocol, value
        )
    return result


def _build_test_prediction_frame(
    metadata: pd.DataFrame,
    selection: Mapping[str, Any],
    logits: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame = metadata.copy()
    probabilities = {}
    predictions = {}
    for arm in ARMS:
        probability = frozen.sigmoid(logits[arm])
        threshold = float(selection["arms"][arm]["threshold"])
        prediction = probability >= threshold
        probabilities[arm] = probability
        predictions[arm] = prediction
        frame[f"probability_{arm}"] = probability
        frame[f"prediction_{arm}"] = prediction.astype(np.int8)
        _gate_columns(frame, arm, weights[arm])
    return frame, probabilities, predictions


def _prediction_frame_evidence(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    if list(expected.columns) != list(actual.columns):
        raise ValueError("Recomputed primary prediction columns differ from the partial artifact")
    if [str(value) for value in expected.dtypes] != [str(value) for value in actual.dtypes]:
        raise ValueError("Recomputed primary prediction dtypes differ from the partial artifact")
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True), actual.reset_index(drop=True),
            check_dtype=True, check_exact=True, check_like=False,
        )
    except AssertionError as error:
        raise ValueError("Recomputed primary predictions are not exactly identical") from error
    numeric_columns = [
        column for column in expected.columns
        if column.startswith(("probability_", "prediction_", "gate_"))
    ]
    columns = {}
    for column in numeric_columns:
        left = expected[column].to_numpy()
        right = actual[column].to_numpy()
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        columns[column] = dict(
            exact=bool(np.array_equal(left, right, equal_nan=True)),
            maximum_absolute_difference=float(difference.max(initial=0.0)),
            existing_array_sha256=_array_sha256(left),
            recomputed_array_sha256=_array_sha256(right),
        )
    if not all(value["exact"] for value in columns.values()):
        raise AssertionError("A primary probability, prediction, or gate column changed")
    return dict(
        all_exact=True,
        rows=len(expected),
        columns=list(expected.columns),
        dtypes=[str(value) for value in expected.dtypes],
        ordered_base_id_sha256=frozen.ordered_base_id_sha256(expected.base_id.astype(str)),
        ordered_group_id_sha256=frozen.ordered_base_id_sha256(expected.group_id.astype(str)),
        label_sha256=_array_sha256(expected.label.to_numpy(dtype=np.int8)),
        numeric_columns=columns,
    )


def audit_partial_primary_reproducibility(args: argparse.Namespace) -> dict:
    """Recompute the four completed primary forwards without overwriting them."""
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    amendment = verify_evaluation_amendment(
        output, protocol, require_reproducibility=False
    )
    if amendment.get("required") is not True:
        raise ValueError("This recovery audit requires a post-barrier evaluation amendment")
    path = output / PRIMARY_REPRODUCIBILITY_FILE
    if path.is_file():
        return verify_primary_reproducibility_audit(output, protocol, amendment)
    if (output / "results.json").exists() or (output / "report.md").exists():
        raise RuntimeError("Primary reproducibility must be audited before formal results")
    test_cache, test_meta, test_manifest = _load_cache(protocol, "test")
    selected = protocol["selected_layers"]
    settings = protocol["training"]
    device = torch.device(args.device)
    per_seed = {}
    for seed in amendment["required_primary_reproducibility_audit"]["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        prediction_path = directory / "test_predictions.parquet"
        registered = amendment["partial_test_access"]["predictions"][str(seed)]
        if frozen.sha256(prediction_path) != registered["artifact"]["sha256"]:
            raise ValueError(f"Partial test prediction changed before audit: seed {seed}")
        existing = pd.read_parquet(prediction_path)
        selection = _load_json(directory / "selection.json")
        heads = {
            arm: _load_head(directory / f"{arm}.pt", arm, seed, protocol, device)
            for arm in ARMS
        }
        logits, weights = predict(heads, test_cache, selected, settings["batch_size"], device)
        recomputed, _, _ = _build_test_prediction_frame(
            test_meta, selection, logits, weights
        )
        evidence = _prediction_frame_evidence(existing, recomputed)
        evidence.update(
            existing_prediction_sha256=registered["artifact"]["sha256"],
            checkpoint_sha256={
                arm: frozen.sha256(directory / f"{arm}.pt") for arm in ARMS
            },
        )
        per_seed[str(seed)] = evidence
        del heads
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Recheck all pre-amendment artifacts after the forwards.  The audit only
    # writes its own append-only record; it never rewrites a prediction file.
    verify_evaluation_amendment(output, protocol, require_reproducibility=False)
    value = dict(
        schema=PRIMARY_REPRODUCIBILITY_SCHEMA,
        state="exact_primary_reproduction_verified",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol_sha256=frozen.sha256(output / "protocol.json"),
        test_barrier_sha256=frozen.sha256(output / "test_barrier.json"),
        evaluation_amendment_sha256=frozen.sha256(output / EVALUATION_AMENDMENT_FILE),
        evaluation_code_sha256=frozen.sha256(Path(__file__)),
        source_test_cache_sha256=test_manifest["sha256"],
        seeds=amendment["required_primary_reproducibility_audit"]["seeds"],
        all_exact=all(record["all_exact"] for record in per_seed.values()),
        per_seed=per_seed,
        rule=(
            "Each pre-existing seed-42..45 parquet was compared against an in-memory recomputation; "
            "no partial prediction artifact was overwritten by this audit."
        ),
    )
    if value["all_exact"] is not True:
        raise AssertionError("Primary predictions were not exactly reproduced")
    frozen.dump(path, value)
    path.chmod(0o444)
    return dict(value, audit_sha256=frozen.sha256(path))


def _verify_result_evaluation_provenance(
    result: Mapping[str, Any], amendment: Mapping[str, Any]
) -> None:
    if amendment.get("required") is not True:
        return
    audit = amendment["primary_reproducibility"]
    recovery = result.get("post_barrier_evaluation_amendment", {})
    if (
        result.get("evaluation_amendment_sha256") != amendment["amendment_sha256"]
        or result.get("primary_reproducibility_audit_sha256") != audit["audit_sha256"]
        or recovery.get("applied") is not True
        or recovery.get("original_test_barrier_unchanged") is not True
        or recovery.get("original_test_barrier_sha256")
        != amendment["original_test_barrier_sha256"]
        or recovery.get("locked_evaluation_code_sha256")
        != amendment["locked_evaluation_code_sha256"]
        or recovery.get("intersection_patch_code_sha256")
        != amendment["intersection_patch_code_sha256"]
        or recovery.get("amended_evaluation_code_sha256")
        != amendment["amended_evaluation_code_sha256"]
        or recovery.get("exact_recomputed_partial_seeds") != audit["seeds"]
    ):
        raise ValueError("Result does not disclose the locked evaluation amendment and exact audit")


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    amendment = verify_evaluation_amendment(
        output, protocol, require_reproducibility=True
    )
    result_path = output / "results.json"
    if result_path.is_file():
        existing = _load_json(result_path)
        _verify_result_evaluation_provenance(existing, amendment)
        return existing
    test_cache, test_meta, test_manifest = _load_cache(protocol, "test")
    selected = protocol["selected_layers"]
    settings = protocol["training"]
    analysis = protocol["analysis"]
    static_seed_scope = (
        copy.deepcopy(amendment["repair_scope"]["secondary_static_seed_scope"])
        if amendment.get("required") is True
        else discover_static_comparison_seeds(protocol, settings["seeds"])
    )
    static_seed_set = set(static_seed_scope["common_seeds"])
    device = torch.device(args.device)
    y = test_meta.label.to_numpy(dtype=np.int64)
    groups = test_meta.group_id.astype(str).to_numpy()
    entries = []
    probabilities: dict[tuple[int, str], np.ndarray] = {}
    predictions: dict[tuple[int, str], np.ndarray] = {}
    primary = []
    static_comparisons = []
    prediction_artifacts = {}
    for seed in settings["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        selection = _load_json(directory / "selection.json")
        heads = {
            arm: _load_head(directory / f"{arm}.pt", arm, seed, protocol, device)
            for arm in ARMS
        }
        logits, weights = predict(heads, test_cache, selected, settings["batch_size"], device)
        frame, seed_probabilities, seed_predictions = _build_test_prediction_frame(
            test_meta, selection, logits, weights
        )
        for arm in ARMS:
            probability = seed_probabilities[arm]
            threshold = float(selection["arms"][arm]["threshold"])
            prediction = seed_predictions[arm]
            probabilities[seed, arm] = probability
            predictions[seed, arm] = prediction
            entries.append(
                dict(
                    seed=int(seed),
                    arm=arm,
                    input_layers=list(arm_layers(arm, selected)),
                    parameters=int(selection["arms"][arm]["parameters"]),
                    threshold=threshold,
                    validation_metrics=selection["arms"][arm]["validation_metrics"],
                    validation_metrics_accuracy_tuned_sensitivity=selection["arms"][arm][
                        "validation_metrics_accuracy_tuned_sensitivity"
                    ],
                    validation_gate=selection["arms"][arm]["validation_gate"],
                    test_metrics=frozen.binary_metrics(y, probability, threshold),
                    test_metrics_validation_tuned_sensitivity=frozen.binary_metrics(
                        y,
                        probability,
                        selection["arms"][arm]["validation_accuracy_sensitivity_threshold"],
                    ),
                    test_gate=gate_diagnostics(weights[arm], arm_layers(arm, selected), y),
                    by_source=_by_source(test_meta, probability, threshold),
                )
            )
        test_path = directory / "test_predictions.parquet"
        if test_path.is_file():
            existing_frame = pd.read_parquet(test_path)
            frame_evidence = _prediction_frame_evidence(existing_frame, frame)
            origin = (
                "pre_amendment_partial_prediction"
                if amendment.get("required") is True
                and int(seed) in amendment["partial_test_access"]["dynamic_forward_completed_seeds"]
                else "preexisting_exact_resume_prediction"
            )
        else:
            _write_parquet_atomic(test_path, frame)
            frame_evidence = _prediction_frame_evidence(pd.read_parquet(test_path), frame)
            origin = "written_by_formal_evaluation"
        prediction_artifacts[str(seed)] = dict(
            artifact=_artifact_record(test_path),
            origin=origin,
            exact_in_memory_recomputation=frame_evidence["all_exact"],
            ordered_base_id_sha256=frame_evidence["ordered_base_id_sha256"],
        )
        for candidate_arm in ("dynamic_fixed", "dynamic_selected"):
            comparison = _comparison(
                y,
                probabilities[seed, candidate_arm], probabilities[seed, "dynamic_repeat_last"],
                selection["arms"][candidate_arm]["threshold"],
                selection["arms"]["dynamic_repeat_last"]["threshold"],
                groups, analysis["repetitions"], analysis["seed"] + seed,
            )
            primary.append(
                dict(seed=int(seed), candidate=candidate_arm, baseline="dynamic_repeat_last", **comparison)
            )

        # Secondary architecture comparison is restricted to the exact seed
        # intersection with real, already-evaluated static predictions.
        if seed in static_seed_set:
            static_frame, _ = _source_static_predictions(protocol, seed, test_meta)
            for dynamic_arm, static_arm in STATIC_ARM_PAIRS:
                comparison = _comparison(
                    y,
                    probabilities[seed, dynamic_arm], static_frame[f"probability_{static_arm}"].to_numpy(),
                    selection["arms"][dynamic_arm]["threshold"],
                    PRIMARY_THRESHOLD,
                    groups, analysis["repetitions"], analysis["seed"] + 10000 + seed,
                )
                static_comparisons.append(
                    dict(
                        seed=int(seed), candidate=dynamic_arm, baseline=static_arm,
                        capacity_matched=False, threshold_rule="fixed_0.5_for_both", **comparison,
                    )
                )
        del heads
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregates = []
    for arm in ARMS:
        rows = [row for row in entries if row["arm"] == arm]
        aggregates.append(
            dict(
                arm=arm,
                seeds=[row["seed"] for row in rows],
                test_metrics={
                    metric: _seed_statistics([row["test_metrics"][metric] for row in rows])
                    for metric in ("accuracy", "f1", "auroc", "auprc", "bce", "brier")
                },
                gate={
                    metric: _seed_statistics([row["test_gate"][metric] for row in rows])
                    for metric in (
                        "normalized_mean_entropy", "mean_effective_layers",
                        "mean_l1_from_global_mean", "maximum_slot_std",
                    )
                },
            )
        )
    matched_aggregates = []
    for candidate in ("dynamic_fixed", "dynamic_selected"):
        rows = [row for row in primary if row["candidate"] == candidate]
        matched_aggregates.append(
            dict(
                candidate=candidate,
                baseline="dynamic_repeat_last",
                seeds=[row["seed"] for row in rows],
                accuracy_delta=_seed_statistics(
                    [row["group_bootstrap"]["accuracy"]["delta"] for row in rows]
                ),
                f1_delta=_seed_statistics(
                    [row["group_bootstrap"]["f1"]["delta"] for row in rows]
                ),
                positive_accuracy_seeds=sum(
                    row["group_bootstrap"]["accuracy"]["delta"] > 0 for row in rows
                ),
            )
        )
    for row in matched_aggregates:
        if row["seeds"] != list(map(int, settings["seeds"])):
            raise AssertionError("Primary matched comparison lost a dynamic seed")
    static_aggregates = aggregate_static_comparisons(static_comparisons)
    for row in static_aggregates:
        if row["seeds"] != static_seed_scope["common_seeds"]:
            raise AssertionError("Secondary static comparison seed support changed")
    recovery = dict(applied=False)
    evaluation_amendment_sha256 = None
    primary_reproducibility_audit_sha256 = None
    if amendment.get("required") is True:
        audit = amendment["primary_reproducibility"]
        evaluation_amendment_sha256 = amendment["amendment_sha256"]
        primary_reproducibility_audit_sha256 = audit["audit_sha256"]
        recovery = dict(
            applied=True,
            original_test_barrier_unchanged=True,
            original_test_barrier_sha256=amendment["original_test_barrier_sha256"],
            locked_evaluation_code_sha256=amendment["locked_evaluation_code_sha256"],
            intersection_patch_code_sha256=amendment["intersection_patch_code_sha256"],
            amended_evaluation_code_sha256=amendment["amended_evaluation_code_sha256"],
            failure=amendment["reason"]["failure"],
            partial_dynamic_forward_seeds=amendment["partial_test_access"][
                "dynamic_forward_completed_seeds"
            ],
            exact_recomputed_partial_seeds=audit["seeds"],
            exact_recomputation_passed=audit["all_exact"],
            secondary_static_seed_scope=static_seed_scope["common_seeds"],
            statement=(
                "The original barrier was retained. The evaluator-only amendment restricts the "
                "secondary static comparison to real source artifacts; primary model forwards, "
                "thresholds, and matched comparisons are unchanged."
            ),
        )
    result = dict(
        protocol_sha256=frozen.sha256(output / "protocol.json"),
        barrier_sha256=frozen.sha256(output / "test_barrier.json"),
        evaluation_amendment_sha256=evaluation_amendment_sha256,
        primary_reproducibility_audit_sha256=primary_reproducibility_audit_sha256,
        locked_training_code_sha256=protocol["code_sha256"]["dynamic_gate_diagnostics.py"],
        evaluation_code_sha256=frozen.sha256(Path(__file__)),
        source_test_cache_sha256=test_manifest["sha256"],
        selected_layers=list(selected),
        primary_matched_seed_scope=dict(
            seeds=list(map(int, settings["seeds"])),
            n_seeds=len(settings["seeds"]),
            complete=True,
            policy="all locked dynamic-head seeds",
        ),
        test_prediction_artifacts=prediction_artifacts,
        post_barrier_evaluation_amendment=recovery,
        entries=entries,
        seed_aggregates=aggregates,
        primary_matched_comparisons=primary,
        matched_seed_aggregates=matched_aggregates,
        secondary_static_comparisons=static_comparisons,
        secondary_static_seed_aggregates=static_aggregates,
        secondary_static_seed_scope=static_seed_scope,
        limitations=protocol["limitations"],
    )
    _verify_result_evaluation_provenance(result, amendment)
    frozen.dump(result_path, result)
    write_report(output, result)
    return result


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.3f}"


def write_report(output: Path, result: Mapping[str, Any] | None = None) -> Path:
    output = Path(output).resolve()
    protocol = load_protocol(output)
    if result is None:
        result = _load_json(output / "results.json")
    if "code_sha256" in protocol:
        amendment = verify_evaluation_amendment(
            output, protocol, require_reproducibility=True
        )
        _verify_result_evaluation_provenance(result, amendment)
    aggregates = {row["arm"]: row for row in result["seed_aggregates"]}
    static_scope = result["secondary_static_seed_scope"]
    lines = [
        "# Input-conditioned dynamic gate 诊断",
        "",
        "同一个冻结 E03_s42 LoRA、同一份最后有效 token 的 32 层缓存。三个动态门控头逐参数同构；"
        "dynamic_repeat_last 将第 32 层复制四份，因此是容量、初始化和优化过程匹配的主要对照。",
        "主结果预注册使用固定阈值 0.5；validation 最大 accuracy 阈值仅保存在结果中作为敏感性分析。",
    ]
    recovery = result.get("post_barrier_evaluation_amendment", {})
    if recovery.get("applied") is True:
        lines.extend(
            [
                "",
                "## Post-barrier evaluator amendment",
                "",
                "原 test barrier 未修改或重锁。锁定版 evaluator 错误地假定 5 个 dynamic seeds 都有"
                "原静态头预测；实际仅 seeds 42/43/44 存在，因此旧运行在写出 dynamic seeds "
                "42/43/44/45 的预测后、生成 results.json/report.md 前失败。",
                f"锁定代码 SHA-256：`{recovery['locked_evaluation_code_sha256']}`；"
                f"首次交集修复 SHA-256：`{recovery['intersection_patch_code_sha256']}`；"
                f"带恢复门禁的代码 SHA-256：`{recovery['amended_evaluation_code_sha256']}`。",
                f"不可变 amendment SHA-256：`{result['evaluation_amendment_sha256']}`；"
                f"seed 42–45 精确重算审计 SHA-256："
                f"`{result['primary_reproducibility_audit_sha256']}`。逐行 schema、dtype、概率、"
                "决策和 gate 权重均逐值完全一致；已有预测未被覆盖。",
                f"主等容量比较仍使用全部 seeds={protocol['training']['seeds']}；"
                f"次要静态比较按 amendment 固定为真实共同 seeds="
                f"{recovery['secondary_static_seed_scope']}。",
            ]
        )
    lines.extend(
        [
            "",
            f"## 测试集结果（{len(protocol['training']['seeds'])} 个 head seeds）",
            "",
            "| arm | Accuracy mean±sd | F1 mean±sd | AUROC | BCE | gate entropy/log(4) | effective layers |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        row = aggregates[arm]
        metrics, gate = row["test_metrics"], row["gate"]
        lines.append(
            f"| {arm} | {_percent(metrics['accuracy']['mean'])}±{_percent(metrics['accuracy']['std'])} | "
            f"{_percent(metrics['f1']['mean'])}±{_percent(metrics['f1']['std'])} | "
            f"{metrics['auroc']['mean']:.6f} | {metrics['bce']['mean']:.6f} | "
            f"{gate['normalized_mean_entropy']['mean']:.4f} | {gate['mean_effective_layers']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 与动态 repeat baseline 的容量匹配比较",
            "",
            "差值为候选减 dynamic_repeat_last。每个 seed 的区间按 group_id 整组 bootstrap，"
            "随机化检验也按 group_id 整组交换。"
            f"seed 均值只描述这 {len(protocol['training']['seeds'])} 个 head 初始化；"
            f"主等容量比较完整使用 seeds={protocol['training']['seeds']}。",
            "",
            "| candidate | Accuracy Δ mean±sd (pp) | F1 Δ mean±sd (pp) | positive seeds |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result["matched_seed_aggregates"]:
        lines.append(
            f"| {row['candidate']} | {_percent(row['accuracy_delta']['mean'])}±"
            f"{_percent(row['accuracy_delta']['std'])} | {_percent(row['f1_delta']['mean'])}±"
            f"{_percent(row['f1_delta']['std'])} | {row['positive_accuracy_seeds']}/"
            f"{len(row['seeds'])} |"
        )
    lines.extend(
        [
            "",
            "## 与原全局权重头的次要比较",
            "",
            "动态头多了 query/key/score 参数，所以下表不是容量匹配因果对照，只用于判断动态门控是否值得继续。"
            "两侧均用固定阈值 0.5。",
            f"静态头仅存在于共同 seeds={static_scope['common_seeds']}（n={static_scope['n_common_seeds']}）；"
            f"dynamic-only seeds={static_scope['excluded_dynamic_seeds']} 没有对应静态 test prediction，"
            f"因此未纳入本附加比较，也没有复制或合成预测。主等容量比较仍为完整 "
            f"{len(protocol['training']['seeds'])} seeds。",
            "",
            "| dynamic | global baseline | n | seeds | Accuracy Δ mean±sd (pp) | F1 Δ mean±sd (pp) |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in result["secondary_static_seed_aggregates"]:
        if row["n_seeds"]:
            accuracy = (
                f"{_percent(row['accuracy_delta']['mean'])}±{_percent(row['accuracy_delta']['std'])}"
            )
            f1 = f"{_percent(row['f1_delta']['mean'])}±{_percent(row['f1_delta']['std'])}"
        else:
            accuracy = f1 = "—"
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | {row['n_seeds']} | {row['seeds']} | "
            f"{accuracy} | {f1} |"
        )
    lines.extend(
        [
            "",
            "## 门控解释",
            "",
            "normalized entropy 接近 1 表示接近均匀权重；effective layers 是 exp(entropy)。"
            "mean_l1_from_global_mean 与各 slot 的标准差衡量权重是否随输入变化。门控变化本身不证明有用，"
            "主要证据仍是同容量 dynamic_repeat_last 配对比较。",
            "",
            f"验证集选择层：{protocol['selected_layers']}；固定层：{protocol['fixed_layers']}。",
            "",
            "## 限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in protocol["limitations"])
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if (output / "test_barrier.json").is_file():
        # Resume a locked evaluation without re-entering initialization or
        # training.  The existing protocol/checkpoints remain byte-for-byte
        # protected by verify_test_barrier inside evaluate().
        result = evaluate(args)
        write_report(output, result)
        return result
    initialize(args)
    train(args)
    lock(args)
    result = evaluate(args)
    write_report(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "init", "train", "lock", "amend", "audit-reproducibility",
            "evaluate", "report", "run",
        ),
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gate-width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=314159)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and args.action in (
        "train", "lock", "audit-reproducibility", "evaluate", "run"
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested dynamic-gate action")
    if args.action == "init":
        result = initialize(args)
    elif args.action == "train":
        result = train(args)
    elif args.action == "lock":
        result = lock(args)
    elif args.action == "amend":
        result = create_evaluation_amendment(args)
    elif args.action == "audit-reproducibility":
        result = audit_partial_primary_reproducibility(args)
    elif args.action == "evaluate":
        result = evaluate(args)
    elif args.action == "report":
        result = dict(report=str(write_report(args.output)))
    else:
        result = run_all(args)
    compact = (
        result
        if args.action in ("lock", "amend", "audit-reproducibility", "report")
        else {"state": "complete", "action": args.action}
    )
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
