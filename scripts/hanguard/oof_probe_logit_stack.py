"""True group-cross-fitted layer probes and a calibrated logit stack.

This follow-up consumes the frozen E03 feature cache without running the
backbone again.  Base probe logits used to train the deployment stack are
genuinely out-of-fold: every ``group_id`` is scored by probes trained without
that group.  Strict outer metrics and layer selection additionally use
pair-excluded inner probes, so an outer-heldout label cannot return through
another fold's meta covariate.  Four slot-matched, independently initialized
repeat-L32 probes control parameter count and topology for four-layer stacks.

The train-only phase chooses stack regularization and a learned four-layer set
from strict nested outer-group-OOF predictions.  Validation applies a pre-registered pass/fallback
gate at a fixed calibrated threshold of 0.5.  Test features cannot be opened
until all choices and checkpoints have been hashed into a new barrier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
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

from scripts.hanguard.frozen_lora_layer_diagnostics import (  # noqa: E402
    ALL_LAYERS,
    FIXED_FUSION_LAYERS,
    atomic_torch_save,
    binary_metrics,
    dump,
    group_cluster_bootstrap,
    group_permutation_test,
    mcnemar_exact,
    ordered_base_id_sha256,
    sha256,
    sigmoid,
)


DEFAULT_SOURCE = ROOT / "outputs/hanguard_frozen_lora_layer_diagnostics_20261012"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_oof_probe_logit_stack_20261012"
MATCHED_ARMS = ("repeat32", "fixed4", "selected4")
ALL_ARMS = ("single32",) + MATCHED_ARMS


def _strict_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _code_hashes() -> dict[str, str]:
    paths = {
        "oof_probe_logit_stack.py": Path(__file__),
        "frozen_lora_layer_diagnostics.py": ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def group_equal_weights(groups: Sequence[Any]) -> np.ndarray:
    """Give every group equal total mass while keeping mean row weight one."""
    groups = np.asarray(groups).astype(str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Nonempty one-dimensional groups required")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FloatingPointError("Invalid group-equal weights")
    return weights


def make_group_folds(
    labels: Sequence[int], groups: Sequence[Any], n_splits: int = 5, seed: int = 20261012
) -> np.ndarray:
    """Deterministic stratified folds with hard group isolation."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    if labels.shape != groups.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("Aligned binary labels and groups required")
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError("Insufficient groups for requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(labels), -1, dtype=np.int16)
    for number, (train, heldout) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        if set(groups[train]) & set(groups[heldout]):
            raise AssertionError("group_id leakage across a fold")
        if len(np.unique(labels[heldout])) != 2:
            raise ValueError(f"Fold {number} lacks one class")
        fold[heldout] = number
    if (fold < 0).any() or any(len(np.unique(fold[groups == group])) != 1 for group in np.unique(groups)):
        raise AssertionError("Every complete group must occur in exactly one fold")
    return fold


class BranchProbes(nn.Module):
    """Independent probes initialized by a pre-registered comparison slot."""

    def __init__(
        self, hidden_size: int, branches: int, seed: int, slot_ids: Sequence[int] | None = None
    ):
        super().__init__()
        self.slot_ids = tuple(range(branches) if slot_ids is None else map(int, slot_ids))
        if len(self.slot_ids) != branches or any(slot < 0 for slot in self.slot_ids):
            raise ValueError("One nonnegative slot ID is required per branch")
        bound = 1.0 / math.sqrt(hidden_size)
        weights, biases = [], []
        for slot in self.slot_ids:
            generator = torch.Generator(device="cpu").manual_seed(seed + 104729 * slot)
            weights.append(torch.empty(hidden_size).uniform_(-bound, bound, generator=generator))
            biases.append(torch.empty(1).uniform_(-bound, bound, generator=generator))
        self.weight = nn.Parameter(torch.stack(weights))
        self.bias = nn.Parameter(torch.cat(biases))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != self.weight.shape:
            raise ValueError("Expected [batch,branch,hidden] features")
        normalized = F.layer_norm(features.float(), (features.shape[-1],))
        return torch.einsum("bkd,kd->bk", normalized, self.weight) + self.bias


def layer_slot(layer: int) -> int:
    """Assign each eight-layer depth band to one of four comparison slots."""
    if int(layer) not in ALL_LAYERS:
        raise ValueError(f"Invalid layer: {layer}")
    return min((int(layer) - 1) // 8, 3)


def assert_matched_branch_initialization(
    pool: BranchProbes, repeat: BranchProbes, layer32_index: int | None = None
) -> dict:
    """Every candidate matches its slot control; the four slots are distinct."""
    if repeat.slot_ids != (0, 1, 2, 3):
        raise ValueError("Repeat control must contain slots 0,1,2,3 in order")
    hashes = []
    for slot in range(4):
        weight = repeat.weight[slot].detach().cpu()
        bias = repeat.bias[slot].detach().cpu()
        hashes.append(hashlib.sha256(weight.numpy().tobytes() + bias.numpy().tobytes()).hexdigest())
    if len(set(hashes)) != 4:
        raise ValueError("Repeat-L32 slots do not have independent initializations")
    for branch, slot in enumerate(pool.slot_ids):
        if slot not in range(4):
            raise ValueError("Pool uses an unregistered comparison slot")
        if not torch.equal(pool.weight[branch].detach().cpu(), repeat.weight[slot].detach().cpu()):
            raise ValueError("Pool/repeat slot initial weights differ")
        if not torch.equal(pool.bias[branch].detach().cpu(), repeat.bias[slot].detach().cpu()):
            raise ValueError("Pool/repeat slot initial biases differ")
    if layer32_index is not None and pool.slot_ids[layer32_index] != 3:
        raise ValueError("Layer 32 must occupy comparison slot 3")
    return dict(slot_initialization_sha256=hashes, pool_slot_ids=list(pool.slot_ids))


@torch.no_grad()
def synchronize_matched_l32(
    pool: BranchProbes,
    repeat: BranchProbes,
    layer32_index: int,
    pool_optimizer: torch.optim.Optimizer | None = None,
    repeat_optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Canonicalize the shared L32 nuisance estimator across matched arms.

    The pool and repeat tensors have different branch dimensions.  Even when
    their L32 objectives are mathematically identical, GPU kernels may use
    different reduction paths and accumulate tiny numerical differences.  We
    therefore define pool L32 as the canonical estimator and synchronize the
    corresponding repeat slot, including Adam state, after every update.
    """
    if (
        layer32_index < 0
        or layer32_index >= len(pool.slot_ids)
        or repeat.slot_ids != (0, 1, 2, 3)
        or pool.slot_ids[layer32_index] != 3
        or pool.weight.shape[1] != repeat.weight.shape[1]
    ):
        raise ValueError("Matched L32 synchronization requires pool slot 3 and repeat slot 3")
    if (pool_optimizer is None) != (repeat_optimizer is None):
        raise ValueError("Both matched optimizers must be provided together")
    before_weight = float(
        (pool.weight[layer32_index] - repeat.weight[3]).abs().max().detach().cpu()
    )
    before_bias = float((pool.bias[layer32_index] - repeat.bias[3]).abs().detach().cpu())
    state_pairs = []
    if pool_optimizer is not None:
        for pool_parameter, repeat_parameter in (
            (pool.weight, repeat.weight), (pool.bias, repeat.bias)
        ):
            pool_state = pool_optimizer.state.get(pool_parameter, {})
            repeat_state = repeat_optimizer.state.get(repeat_parameter, {})
            required = {"step", "exp_avg", "exp_avg_sq"}
            if not required.issubset(pool_state) or not required.issubset(repeat_state):
                raise ValueError("Matched AdamW state is incomplete after an optimizer update")
            if not torch.equal(torch.as_tensor(pool_state["step"]), torch.as_tensor(repeat_state["step"])):
                # Adam's step counter is shared by every row in a parameter
                # tensor.  It cannot be repaired for slot 3 without changing
                # the other repeat slots, so fail instead of silently copying.
                raise ValueError("Matched AdamW step counters differ")
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if (key in pool_state) != (key in repeat_state):
                    raise ValueError(f"Matched AdamW state differs for {key}")
                if key in pool_state:
                    if (
                        tuple(pool_state[key].shape) != tuple(pool_parameter.shape)
                        or tuple(repeat_state[key].shape) != tuple(repeat_parameter.shape)
                    ):
                        raise ValueError(f"Matched AdamW state has an invalid shape for {key}")
                    state_pairs.append(
                        (pool_state[key][layer32_index], repeat_state[key][3], key)
                    )
    repeat.weight[3].copy_(pool.weight[layer32_index])
    repeat.bias[3].copy_(pool.bias[layer32_index])
    for source, target, _ in state_pairs:
        target.copy_(source)
    after_weight = float(
        (pool.weight[layer32_index] - repeat.weight[3]).abs().max().detach().cpu()
    )
    after_bias = float((pool.bias[layer32_index] - repeat.bias[3]).abs().detach().cpu())
    if after_weight != 0.0 or after_bias != 0.0:
        raise AssertionError("Canonical L32 parameter synchronization failed")
    if any(not torch.equal(source, target) for source, target, _ in state_pairs):
        raise AssertionError("Canonical L32 AdamW-state synchronization failed")
    return dict(
        raw_parameter_max_absolute_difference=max(before_weight, before_bias),
        canonical_parameter_max_absolute_difference=max(after_weight, after_bias),
    )


def _valid_matched_l32_prediction_diagnostic(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("canonical_l32_logit_max_absolute_difference") == 0.0
        and isinstance(value.get("raw_gpu_l32_logit_max_absolute_difference"), (int, float))
        and math.isfinite(float(value["raw_gpu_l32_logit_max_absolute_difference"]))
        and float(value["raw_gpu_l32_logit_max_absolute_difference"]) >= 0
        and value.get("convention") == "repeat slot 3 is the canonical pool L32 estimator"
    )


def _validate_probe_training_history(value: Any, expected_epochs: int, context: str) -> list[dict]:
    if not isinstance(value, list) or len(value) != int(expected_epochs):
        raise ValueError(f"{context} has incomplete probe-training history")
    numeric = (
        "pool_weighted_bce_sum", "repeat_weighted_bce_sum",
        "raw_l32_parameter_max_absolute_difference",
        "canonical_l32_parameter_max_absolute_difference",
    )
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != index:
            raise ValueError(f"{context} probe-training epoch identity changed")
        if any(
            not isinstance(row.get(key), (int, float))
            or not math.isfinite(float(row[key]))
            or float(row[key]) < 0
            for key in numeric
        ):
            raise ValueError(f"{context} has a nonfinite probe-training diagnostic")
        if row["canonical_l32_parameter_max_absolute_difference"] != 0.0:
            raise ValueError(f"{context} did not retain canonical L32 parameters")
    return value


def _validate_probe_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    candidate_layers: Sequence[int],
    identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
    context: str,
) -> None:
    layers = tuple(map(int, candidate_layers))
    hidden_size = int(protocol["hidden_size"])
    if (
        tuple(checkpoint.get("candidate_layers", ())) != layers
        or checkpoint.get("identity") != dict(identity)
        or checkpoint.get("hidden_size") != hidden_size
    ):
        raise ValueError(f"{context} checkpoint identity changed")
    pool_state = checkpoint.get("pool_state")
    repeat_state = checkpoint.get("repeat_state")
    if not isinstance(pool_state, dict) or not isinstance(repeat_state, dict):
        raise ValueError(f"{context} checkpoint lacks probe states")
    if set(pool_state) != {"weight", "bias"} or set(repeat_state) != {"weight", "bias"}:
        raise ValueError(f"{context} checkpoint state schema changed")
    expected_shapes = {
        "pool": ((len(layers), hidden_size), (len(layers),)),
        "repeat": ((4, hidden_size), (4,)),
    }
    for name, state in (("pool", pool_state), ("repeat", repeat_state)):
        if (
            not torch.is_tensor(state["weight"])
            or not torch.is_tensor(state["bias"])
            or tuple(state["weight"].shape) != expected_shapes[name][0]
            or tuple(state["bias"].shape) != expected_shapes[name][1]
            or not torch.isfinite(state["weight"]).all()
            or not torch.isfinite(state["bias"]).all()
        ):
            raise ValueError(f"{context} checkpoint {name} state changed")
    layer32_position = layers.index(32)
    if not torch.equal(pool_state["weight"][layer32_position], repeat_state["weight"][3]) or not torch.equal(
        pool_state["bias"][layer32_position], repeat_state["bias"][3]
    ):
        raise ValueError(f"{context} checkpoint lacks the canonical matched L32 estimator")
    _validate_probe_training_history(
        checkpoint.get("history"), protocol["probes"]["epochs"], context
    )


def _expected_probe_initialization(
    candidate_layers: Sequence[int], protocol: Mapping[str, Any]
) -> dict:
    pool = BranchProbes(
        int(protocol["hidden_size"]), len(candidate_layers), int(protocol["probes"]["seed"]),
        [layer_slot(layer) for layer in candidate_layers],
    )
    repeat = BranchProbes(
        int(protocol["hidden_size"]), 4, int(protocol["probes"]["seed"]), [0, 1, 2, 3]
    )
    return assert_matched_branch_initialization(
        pool, repeat, list(candidate_layers).index(32)
    )


def _validate_probe_history_sidecar(
    value: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    identity: Mapping[str, Any],
    candidate_layers: Sequence[int],
    protocol: Mapping[str, Any],
    context: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("identity") != dict(identity)
        or value.get("initial") != _expected_probe_initialization(candidate_layers, protocol)
        or value.get("epochs") != checkpoint.get("history")
    ):
        raise ValueError(f"{context} training-history sidecar changed")


def _weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average(np.square(values - mean), axis=0, weights=weights)
    return mean, variance


def fit_stack_model(
    features: np.ndarray,
    labels: Sequence[int],
    weights: Sequence[float],
    regularization_c: float,
    seed: int = 42,
) -> dict:
    """Fit an L2 logistic stack after group-weighted standardization."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(labels) or weights.shape != labels.shape:
        raise ValueError("Stack inputs, labels, and weights must align")
    if regularization_c <= 0 or not np.isfinite(features).all():
        raise ValueError("Positive C and finite stack features required")
    mean, variance = _weighted_mean_variance(features, weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (features - mean) / scale
    model = LogisticRegression(
        C=float(regularization_c), penalty="l2", solver="lbfgs", max_iter=1000,
        tol=1e-10, fit_intercept=True, random_state=seed,
    )
    model.fit(standardized, labels, sample_weight=weights)
    if not bool(model.n_iter_[0] < model.max_iter):
        raise RuntimeError("Stack optimizer did not converge")
    return dict(
        mean=mean.tolist(), scale=scale.tolist(), coef=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]), C=float(regularization_c), iterations=int(model.n_iter_[0]),
    )


def predict_stack_score(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(coef):
        raise ValueError("Stack feature width changed")
    score = ((features - mean) / scale) @ coef + float(model["intercept"])
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite stack score")
    return score


def fit_platt(scores: Sequence[float], labels: Sequence[int], weights: Sequence[float]) -> dict:
    """Fit monotone two-parameter Platt calibration with stable weighted BCE."""
    from scipy.optimize import minimize

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (scores.shape == labels.shape == weights.shape) or not np.isfinite(scores).all():
        raise ValueError("Aligned finite calibration inputs required")

    def objective(parameters):
        log_slope, intercept = parameters
        slope = math.exp(float(log_slope))
        logits = slope * scores + intercept
        loss = np.average(np.logaddexp(0.0, logits) - labels * logits, weights=weights)
        return float(loss + 1e-10 * (log_slope * log_slope + intercept * intercept))

    fitted = minimize(
        objective, np.asarray([0.0, 0.0]), method="L-BFGS-B",
        bounds=[(-8.0, 8.0), (-30.0, 30.0)], options={"maxiter": 1000, "ftol": 1e-13},
    )
    if not fitted.success:
        raise RuntimeError(f"Platt calibration failed: {fitted.message}")
    return dict(
        method="platt", slope=float(math.exp(fitted.x[0])), intercept=float(fitted.x[1]),
        objective=float(fitted.fun), parameter_count=2,
    )


def fit_temperature(scores: Sequence[float], labels: Sequence[int], weights: Sequence[float]) -> dict:
    """Fit positive inverse temperature with its intercept fixed at zero."""
    from scipy.optimize import minimize_scalar

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (scores.shape == labels.shape == weights.shape) or not np.isfinite(scores).all():
        raise ValueError("Aligned finite calibration inputs required")

    def objective(log_slope):
        slope = math.exp(float(log_slope))
        logits = slope * scores
        return float(np.average(np.logaddexp(0.0, logits) - labels * logits, weights=weights))

    fitted = minimize_scalar(objective, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1e-10})
    if not fitted.success:
        raise RuntimeError(f"Temperature calibration failed: {fitted.message}")
    return dict(
        method="temperature", slope=float(math.exp(fitted.x)), intercept=0.0,
        objective=float(fitted.fun), parameter_count=1,
    )


def apply_platt(calibrator: Mapping[str, float], scores: Sequence[float]) -> np.ndarray:
    return sigmoid(float(calibrator["slope"]) * np.asarray(scores) + float(calibrator["intercept"]))


def fit_calibrator(
    method: str, scores: Sequence[float], labels: Sequence[int], weights: Sequence[float]
) -> dict:
    if method == "identity":
        return dict(method="identity", slope=1.0, intercept=0.0, parameter_count=0)
    if method == "temperature":
        return fit_temperature(scores, labels, weights)
    if method == "platt":
        return fit_platt(scores, labels, weights)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator: Mapping[str, Any], scores: Sequence[float]) -> np.ndarray:
    method = calibrator.get("method", "platt")
    if method not in ("identity", "temperature", "platt"):
        raise ValueError(f"Unknown calibration method: {method}")
    return apply_platt(calibrator, scores)


def _weighted_metrics(
    labels: Sequence[int], probabilities: Sequence[float], groups: Sequence[Any], threshold: float = 0.5
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = group_equal_weights(groups)
    prediction = probabilities >= threshold
    tp = weights[(labels == 1) & prediction].sum()
    tn = weights[(labels == 0) & ~prediction].sum()
    fp = weights[(labels == 0) & prediction].sum()
    fn = weights[(labels == 1) & ~prediction].sum()
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return dict(
        accuracy=float((tp + tn) / weights.sum()),
        f1=float(2 * tp / max(1e-12, 2 * tp + fp + fn)),
        bce=float(np.average(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)), weights=weights)),
        brier=float(np.average(np.square(probabilities - labels), weights=weights)),
        auroc=float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        auprc=float(average_precision_score(labels, probabilities, sample_weight=weights)),
        groups=int(len(np.unique(np.asarray(groups).astype(str)))), threshold=float(threshold),
    )


def combined_metrics(labels, probabilities, groups, threshold=0.5) -> dict:
    return dict(
        row=binary_metrics(labels, probabilities, threshold),
        group_equal=_weighted_metrics(labels, probabilities, groups, threshold),
    )


def crossfit_stack_scores(
    features: np.ndarray,
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    regularization_c: float,
    seed: int = 42,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    folds = np.asarray(folds, dtype=np.int64)
    output = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        heldout = folds == fold
        training = ~heldout
        if set(groups[training]) & set(groups[heldout]):
            raise AssertionError("Stack crossfit leaked a group")
        training_weights = group_equal_weights(groups[training])
        model = fit_stack_model(
            features[training], labels[training], training_weights, regularization_c, seed + int(fold)
        )
        output[heldout] = predict_stack_score(model, features[heldout])
    if not np.isfinite(output).all():
        raise FloatingPointError("Incomplete stack OOF scores")
    return output


def crossfit_calibrated_probabilities(
    scores: Sequence[float], labels: Sequence[int], groups: Sequence[Any], folds: Sequence[int],
    method: str = "platt",
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    folds = np.asarray(folds, dtype=np.int64)
    output = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in sorted(np.unique(folds)):
        heldout = folds == fold
        training = ~heldout
        training_weights = group_equal_weights(groups[training])
        calibrator = fit_calibrator(method, scores[training], labels[training], training_weights)
        output[heldout] = apply_calibrator(calibrator, scores[heldout])
    if not np.isfinite(output).all():
        raise FloatingPointError("Incomplete calibrated OOF probabilities")
    return output


def crossfit_platt_probabilities(
    scores: Sequence[float], labels: Sequence[int], groups: Sequence[Any], folds: Sequence[int]
) -> np.ndarray:
    """Compatibility wrapper for the original fixed-Platt helper."""
    return crossfit_calibrated_probabilities(scores, labels, groups, folds, "platt")


def select_stack_regularization(
    features: np.ndarray,
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    seed: int = 42,
    calibration_method: str = "platt",
) -> dict:
    """Choose C by group-equal BCE of cross-fitted, cross-calibrated scores."""
    records = []
    best = None
    for value in c_grid:
        scores = crossfit_stack_scores(features, labels, groups, folds, float(value), seed)
        probability = crossfit_calibrated_probabilities(
            scores, labels, groups, folds, calibration_method
        )
        metrics = combined_metrics(labels, probability, groups, 0.5)
        record = dict(C=float(value), metrics=metrics, raw_scores=scores, probability=probability)
        records.append(record)
        key = (metrics["group_equal"]["bce"], metrics["group_equal"]["brier"], float(value))
        if best is None or key < best[0]:
            best = (key, record)
    selected = best[1]
    weights = group_equal_weights(groups)
    final_stack = fit_stack_model(features, labels, weights, selected["C"], seed)
    final_calibrator = fit_calibrator(calibration_method, selected["raw_scores"], labels, weights)
    return dict(
        C=selected["C"], metrics=selected["metrics"], raw_scores=selected["raw_scores"],
        probability=selected["probability"], stack=final_stack, calibrator=final_calibrator,
        search=[dict(C=row["C"], metrics=row["metrics"]) for row in records],
    )


def select_nested_stack_regularization(
    final_features: np.ndarray,
    outer_records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    seed: int = 42,
    calibration_method: str = "platt",
) -> dict:
    """Select C from strict outer-group-held-out end-to-end predictions."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    folds = np.asarray(folds, dtype=np.int64)
    records = []
    best = None
    for value in c_grid:
        probability = np.full(len(labels), np.nan, dtype=np.float64)
        raw_scores = np.full(len(labels), np.nan, dtype=np.float64)
        for outer in outer_records:
            training = np.asarray(outer["train_indices"], dtype=np.int64)
            heldout = np.asarray(outer["heldout_indices"], dtype=np.int64)
            outer_number = int(outer["outer_fold"])
            if np.any(folds[heldout] != outer_number) or np.any(folds[training] == outer_number):
                raise AssertionError("Nested meta row indices differ from the registered outer fold")
            if set(groups[training]) & set(groups[heldout]):
                raise AssertionError("Nested meta fold leaked an outer-heldout group")
            train_features = np.asarray(outer["train_features"], dtype=np.float64)
            heldout_features = np.asarray(outer["heldout_features"], dtype=np.float64)
            inner_fold = np.asarray(outer["inner_fold"], dtype=np.int64)
            inner_scores = crossfit_stack_scores(
                train_features, labels[training], groups[training], inner_fold,
                float(value), seed + 1000 + outer_number,
            )
            calibrator = fit_calibrator(
                calibration_method, inner_scores, labels[training],
                group_equal_weights(groups[training]),
            )
            stack = fit_stack_model(
                train_features, labels[training], group_equal_weights(groups[training]),
                float(value), seed + 2000 + outer_number,
            )
            score = predict_stack_score(stack, heldout_features)
            raw_scores[heldout] = score
            probability[heldout] = apply_calibrator(calibrator, score)
        if not np.isfinite(probability).all() or not np.isfinite(raw_scores).all():
            raise FloatingPointError("Nested outer OOF prediction is incomplete")
        metrics = combined_metrics(labels, probability, groups, 0.5)
        record = dict(C=float(value), metrics=metrics, raw_scores=raw_scores, probability=probability)
        records.append(record)
        key = (metrics["group_equal"]["bce"], metrics["group_equal"]["brier"], float(value))
        if best is None or key < best[0]:
            best = (key, record)
    selected = best[1]
    weights = group_equal_weights(groups)
    # Deployment stack uses rowwise group-OOF base logits.  Its calibrator is
    # fit to the strict outer-heldout scores above, avoiding indirect
    # self-label return through nonnested level-one covariates.
    final_stack = fit_stack_model(final_features, labels, weights, selected["C"], seed + 4000)
    final_calibrator = fit_calibrator(
        calibration_method, selected["raw_scores"], labels, weights
    )
    final_calibrator["fit_source"] = "strict nested outer-heldout stack scores"
    return dict(
        C=selected["C"], metrics=selected["metrics"], raw_scores=selected["raw_scores"],
        probability=selected["probability"], stack=final_stack, calibrator=final_calibrator,
        search=[dict(C=row["C"], metrics=row["metrics"]) for row in records],
        metric_scope="strict nested outer-group-held-out",
    )


def select_calibration_family(
    features: np.ndarray,
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    seed: int = 42,
    tie_tolerance: float = 1e-6,
) -> dict:
    """Select one shared calibrator family on repeat-L32 train OOF only."""
    methods = ("identity", "temperature", "platt")
    complexity = {"identity": 0, "temperature": 1, "platt": 2}
    fits = {
        method: select_stack_regularization(
            features, labels, groups, folds, c_grid, seed, calibration_method=method
        )
        for method in methods
    }
    best_bce = min(fits[method]["metrics"]["group_equal"]["bce"] for method in methods)
    eligible = [
        method for method in methods
        if fits[method]["metrics"]["group_equal"]["bce"] <= best_bce + tie_tolerance
    ]
    selected = min(eligible, key=lambda method: (complexity[method], methods.index(method)))
    return dict(
        selected=selected, tie_tolerance=float(tie_tolerance), selection_source="repeat32 train group-OOF",
        criterion="minimum group-equal BCE; within tolerance prefer fewer calibration parameters",
        comparisons={
            method: dict(
                parameter_count=complexity[method], C=fits[method]["C"],
                metrics=fits[method]["metrics"], search=fits[method]["search"],
            )
            for method in methods
        },
    )


def select_nested_calibration_family(
    final_features: np.ndarray,
    outer_records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    seed: int = 42,
    tie_tolerance: float = 1e-6,
) -> dict:
    """Select one family using strict nested repeat-L32 outer predictions."""
    methods = ("identity", "temperature", "platt")
    complexity = {"identity": 0, "temperature": 1, "platt": 2}
    fits = {
        method: select_nested_stack_regularization(
            final_features, outer_records, labels, groups, folds, c_grid, seed,
            calibration_method=method,
        )
        for method in methods
    }
    best_bce = min(fits[method]["metrics"]["group_equal"]["bce"] for method in methods)
    eligible = [
        method for method in methods
        if fits[method]["metrics"]["group_equal"]["bce"] <= best_bce + tie_tolerance
    ]
    selected = min(eligible, key=lambda method: (complexity[method], methods.index(method)))
    return dict(
        selected=selected, tie_tolerance=float(tie_tolerance),
        selection_source="repeat32 strict nested train outer-group-OOF",
        criterion="minimum group-equal BCE; within tolerance prefer fewer calibration parameters",
        comparisons={
            method: dict(
                parameter_count=complexity[method], C=fits[method]["C"],
                metrics=fits[method]["metrics"], search=fits[method]["search"],
            )
            for method in methods
        },
    )


def greedy_learned_stack_layers(
    layer_logits: np.ndarray,
    layers: Sequence[int],
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    count: int = 4,
    anchor: int = 32,
    seed: int = 42,
    calibration_method: str = "platt",
) -> dict:
    """Greedy selection using only learned-stack group-OOF calibrated BCE."""
    layers = tuple(map(int, layers))
    layer_logits = np.asarray(layer_logits, dtype=np.float64)
    if layer_logits.shape != (len(labels), len(layers)) or anchor not in layers:
        raise ValueError("OOF layer logits/layer labels are inconsistent")
    if not 1 <= count <= len(layers):
        raise ValueError("Invalid selected layer count")
    index = {layer: position for position, layer in enumerate(layers)}
    selected = [anchor]
    trace = []
    cache = {}
    while len(selected) < count:
        candidates = []
        used_slots = {layer_slot(layer) for layer in selected}
        for layer in layers:
            if layer in selected or layer_slot(layer) in used_slots:
                continue
            trial = tuple(sorted(selected + [layer]))
            if trial not in cache:
                columns = [index[item] for item in trial]
                cache[trial] = select_stack_regularization(
                    layer_logits[:, columns], labels, groups, folds, c_grid,
                    seed + len(trial) * 1000 + layer, calibration_method=calibration_method,
                )
            result = cache[trial]
            candidates.append((result["metrics"]["group_equal"]["bce"], layer, trial, result))
        if not candidates:
            raise ValueError("Candidate pool cannot fill distinct comparison slots")
        bce, layer, trial, result = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(layer)
        trace.append(
            dict(step=len(selected), added_layer=int(layer), added_slot=layer_slot(layer),
                 layers=list(trial), C=result["C"],
                 group_oof_calibrated_bce=float(bce), metrics=result["metrics"])
        )
    return dict(layers=sorted(selected), greedy_order=[anchor] + [row["added_layer"] for row in trace], trace=trace)


def greedy_nested_stack_layers(
    layer_logits: np.ndarray,
    nested_bundles: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    labels: Sequence[int],
    groups: Sequence[Any],
    folds: Sequence[int],
    c_grid: Sequence[float],
    count: int = 4,
    anchor: int = 32,
    seed: int = 42,
    calibration_method: str = "platt",
) -> dict:
    """Greedy layer choice from strict nested outer-group-held-out metrics."""
    layers = tuple(map(int, layers))
    index = {layer: position for position, layer in enumerate(layers)}
    if anchor not in index or layer_logits.shape != (len(labels), len(layers)):
        raise ValueError("Nested layer logits/layer labels are inconsistent")
    selected = [anchor]
    trace = []
    cache = {}
    while len(selected) < count:
        candidates = []
        used_slots = {layer_slot(layer) for layer in selected}
        for layer in layers:
            if layer in selected or layer_slot(layer) in used_slots:
                continue
            trial = tuple(sorted(selected + [layer]))
            if trial not in cache:
                columns = [index[item] for item in trial]
                outer_records = [
                    dict(
                        outer_fold=value["outer_fold"],
                        train_indices=value["train_indices"],
                        heldout_indices=value["heldout_indices"],
                        inner_fold=value["inner_fold"],
                        train_features=value["layer_logits"][:, columns],
                        heldout_features=value["outer_layer_logits"][:, columns],
                    )
                    for value in nested_bundles
                ]
                cache[trial] = select_nested_stack_regularization(
                    layer_logits[:, columns], outer_records, labels, groups, folds, c_grid,
                    seed + len(trial) * 1000 + layer,
                    calibration_method=calibration_method,
                )
            result = cache[trial]
            candidates.append((result["metrics"]["group_equal"]["bce"], layer, trial, result))
        if not candidates:
            raise ValueError("Candidate pool cannot fill distinct comparison slots")
        bce, layer, trial, result = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append(layer)
        trace.append(
            dict(
                step=len(selected), added_layer=int(layer), added_slot=layer_slot(layer),
                layers=list(trial), C=result["C"], group_oof_calibrated_bce=float(bce),
                metrics=result["metrics"], metric_scope=result["metric_scope"],
            )
        )
    return dict(
        layers=sorted(selected), greedy_order=[anchor] + [row["added_layer"] for row in trace],
        trace=trace, selection_scope="strict nested outer-group-held-out",
        slot_constraint="one selected layer per eight-layer comparison slot",
    )


def validation_gate(candidate: Mapping[str, Any], control: Mapping[str, Any], rules: Mapping[str, float]) -> dict:
    """Pre-registered validation fallback rule; all probabilities use 0.5."""
    checks = {
        "group_equal_bce": candidate["group_equal"]["bce"] <= control["group_equal"]["bce"] - rules["minimum_bce_gain"],
        "row_accuracy": candidate["row"]["accuracy"] >= control["row"]["accuracy"] - rules["accuracy_tolerance"],
        "row_auroc": candidate["row"]["auroc"] >= control["row"]["auroc"] - rules["auroc_tolerance"],
    }
    return dict(passed=bool(all(checks.values())), checks=checks, rules=dict(rules))


def _source_paths(source: Path) -> dict[str, Path]:
    source = Path(source).resolve()
    return {
        "protocol": source / "protocol.json",
        "probe_checkpoint": source / "probes/probe.pt",
        "train_cache": source / "features/train.npy",
        "train_manifest": source / "features/train.manifest.json",
        "train_metadata": source / "features/train.metadata.parquet",
        "validation_cache": source / "features/validation.npy",
        "validation_manifest": source / "features/validation.manifest.json",
        "validation_metadata": source / "features/validation.metadata.parquet",
        "test_cache": source / "features/test.npy",
        "test_manifest": source / "features/test.manifest.json",
        "test_metadata": source / "features/test.metadata.parquet",
        "source_barrier": source / "test_barrier.json",
    }


def _metadata_identity(path: Path) -> dict:
    frame = pd.read_parquet(path)
    required = {"base_id", "group_id", "label", "source", "category_id", "text_form"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Metadata lacks columns: {required - set(frame.columns)}")
    if frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError("base_id must be unique and nonempty")
    if frame.base_id.tolist() != sorted(frame.base_id.tolist()):
        raise ValueError("Metadata must remain in base_id ascending order")
    if (
        frame.group_id.isna().any()
        or (frame.group_id.astype(str).str.strip() == "").any()
        or not frame.label.isin((0, 1)).all()
    ):
        raise ValueError("Nonempty group_id and binary labels required")
    if (frame.groupby("group_id").label.nunique() != 1).any():
        raise ValueError("A group_id contains conflicting labels")
    return dict(
        rows=len(frame), sha256=sha256(path), ordered_base_id_sha256=ordered_base_id_sha256(frame.base_id),
        groups=int(frame.group_id.nunique()), positive_rows=int(frame.label.sum()),
    )


def initialize(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    paths = _source_paths(source)
    for key in ("protocol", "train_cache", "train_manifest", "train_metadata", "validation_cache",
                "validation_manifest", "validation_metadata", "source_barrier"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    source_protocol = _strict_json(paths["protocol"])
    if source_protocol.get("code_sha256", {}).get("frozen_lora_layer_diagnostics.py") != sha256(
        ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py"
    ):
        raise ValueError("Source feature protocol code identity changed")
    candidate_layers = tuple(sorted(set(map(int, args.candidate_layers))))
    if not candidate_layers or 32 not in candidate_layers or any(layer not in ALL_LAYERS for layer in candidate_layers):
        raise ValueError("Candidate layers must be unique 1..32 and include layer 32")
    if not set(FIXED_FUSION_LAYERS).issubset(candidate_layers):
        raise ValueError("Candidate layers must contain fixed layers 8/16/24/32")
    c_grid = sorted(set(map(float, args.c_grid)))
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("A positive C grid is required")
    if args.folds < 3 or args.probe_epochs < 1 or args.probe_batch_size < 1:
        raise ValueError("At least three folds and positive probe budgets are required")
    external = None
    if args.oof_logits is not None:
        bundle = Path(args.oof_logits).resolve()
        manifest = Path(args.oof_manifest).resolve() if args.oof_manifest else bundle.with_suffix(".manifest.json")
        if not bundle.is_file() or not manifest.is_file():
            raise FileNotFoundError("External OOF bundle and manifest are both required")
        external = dict(bundle=str(bundle), bundle_sha256=sha256(bundle), manifest=str(manifest),
                        manifest_sha256=sha256(manifest))
    train_frame = pd.read_parquet(paths["train_metadata"], columns=["base_id", "group_id"])
    validation_frame = pd.read_parquet(paths["validation_metadata"], columns=["base_id", "group_id"])
    if set(train_frame.base_id.astype(str)) & set(validation_frame.base_id.astype(str)):
        raise ValueError("train/validation base_id overlap")
    if set(train_frame.group_id.astype(str)) & set(validation_frame.group_id.astype(str)):
        raise ValueError("train/validation group_id overlap")
    protocol = dict(
        version=1,
        source=str(source),
        source_protocol_sha256=sha256(paths["protocol"]),
        source_adapter_sha256=source_protocol["adapter_sha256"],
        source_data_sha256=source_protocol["data_sha256"],
        source_feature_artifacts={
            key: dict(path=str(paths[key]), sha256=sha256(paths[key]))
            for key in ("train_manifest", "train_metadata", "validation_manifest", "validation_metadata")
        },
        train_metadata=_metadata_identity(paths["train_metadata"]),
        validation_metadata=_metadata_identity(paths["validation_metadata"]),
        hidden_size=int(_strict_json(paths["train_manifest"])["hidden_size"]),
        candidate_layers=list(candidate_layers), fixed_layers=list(FIXED_FUSION_LAYERS),
        learned_layer_count=4, learned_layer_anchor=32,
        folds=dict(
            count=int(args.folds), inner_count=int(args.folds) - 1,
            seed=int(args.fold_seed),
            method=(
                "outer StratifiedGroupKFold; within each outer-train, the other global folds are inner folds; "
                "pair-excluded probes are shared across the two valid nested directions"
            ),
        ),
        probes=dict(
            seed=int(args.probe_seed), epochs=int(args.probe_epochs), batch_size=int(args.probe_batch_size),
            lr=float(args.probe_lr), weight_decay=float(args.weight_decay), checkpoint="fixed final epoch",
            objective="group-equal weighted BCE independently summed across branches",
            slot_assignment={str(layer): layer_slot(layer) for layer in candidate_layers},
            initialization=(
                "four distinct seed-derived slots; every candidate layer in an eight-layer depth band "
                "matches the corresponding repeat-L32 slot"
            ),
            matched_l32_convention=(
                "pool L32 is the canonical nuisance estimator; repeat slot 3 parameters and AdamW state "
                "are synchronized after every update, and stored repeat slot-3 logits are copied from "
                "the canonical pool L32 score while raw GPU forward discrepancy is recorded"
            ),
        ),
        stack=dict(
            c_grid=c_grid, penalty="L2", solver="lbfgs",
            selection="minimum strict nested train outer-group-OOF calibrated BCE",
            calibration_candidates=["identity", "temperature", "platt"],
            calibration_selection=(
                "one shared family selected on repeat32 strict nested train outer-group-OOF by group-equal BCE; "
                "within tolerance prefer fewer parameters"
            ),
            calibration_tie_tolerance=1e-6,
            threshold=0.5,
        ),
        validation_gate=dict(minimum_bce_gain=float(args.minimum_bce_gain),
                             accuracy_tolerance=float(args.accuracy_tolerance),
                             auroc_tolerance=float(args.auroc_tolerance)),
        comparisons=dict(
            matched_arms=list(MATCHED_ARMS),
            topology=(
                "four separately parameterized D+1 probe branches per arm, one 4+1 stack, and "
                "one shared-family calibrator with 0/1/2 active parameters; the registered "
                "pool-L32/repeat-slot-3 nuisance estimator is common to both matched arms"
            ),
            repeat_control=(
                "four separately parameterized branches all consume layer 32; slot 3 shares the exact "
                "canonical L32 nuisance estimate across arms"
            ),
            selected_constraint="one layer from each registered eight-layer comparison slot",
            search_pool=(
                "all candidate probes are trained in one vectorized tensor with separable losses and no gradient "
                "clipping; unused search branches are excluded from each arm's parameter count"
            ),
        ),
        bootstrap=dict(repetitions=int(args.repetitions), seed=int(args.analysis_seed)),
        external_oof=external,
        code_sha256=_code_hashes(),
        limitations=[
            "The frozen E03 representation and existing split identities were already explored; this remains diagnostic.",
            "Cross-fitting isolates group_id during base-probe prediction, but it does not provide an independent LoRA seed.",
            "Layer and hyperparameter choice adapts to nested train outer-fold metrics, so the selected train OOF number is descriptive; the independent validation gate controls promotion.",
            "Validation applies only the registered pass/fallback gate; the calibrated decision threshold is fixed at 0.5.",
            "Secondary test comparisons are exploratory and unadjusted for multiplicity.",
        ],
    )
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if _strict_json(protocol_path) != protocol:
            raise ValueError("Requested settings differ from the registered OOF-stack protocol")
    else:
        dump(protocol_path, protocol)
    return protocol


def load_protocol(output: Path) -> dict:
    output = Path(output).resolve()
    path = output / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run init first: {path}")
    protocol = _strict_json(path)
    if protocol.get("code_sha256") != _code_hashes():
        raise ValueError("OOF-stack implementation changed after registration")
    paths = _source_paths(Path(protocol["source"]))
    if sha256(paths["protocol"]) != protocol["source_protocol_sha256"]:
        raise ValueError("Source protocol changed")
    for artifact in protocol["source_feature_artifacts"].values():
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"Source train/validation artifact changed: {artifact['path']}")
    return protocol


def _load_cache(output: Path, protocol: dict, split: str) -> tuple[np.memmap, pd.DataFrame]:
    if split == "test":
        verify_test_barrier(output)
    paths = _source_paths(Path(protocol["source"]))
    manifest = _strict_json(paths[f"{split}_manifest"])
    metadata = pd.read_parquet(paths[f"{split}_metadata"])
    if not manifest.get("complete") or manifest["metadata_sha256"] != sha256(paths[f"{split}_metadata"]):
        raise ValueError(f"Incomplete or changed {split} feature manifest")
    if not manifest.get("sha256") or sha256(paths[f"{split}_cache"]) != manifest["sha256"]:
        raise ValueError(f"{split} feature cache content changed")
    cache = np.load(paths[f"{split}_cache"], mmap_mode="r", allow_pickle=False)
    expected = (len(metadata), len(ALL_LAYERS), protocol["hidden_size"])
    if cache.shape != expected or cache.dtype != np.float16:
        raise ValueError(f"Unexpected {split} cache layout")
    if paths[f"{split}_cache"].stat().st_size != manifest["bytes"]:
        raise ValueError(f"{split} cache extent changed")
    if metadata.base_id.tolist() != sorted(metadata.base_id.tolist()) or metadata.base_id.duplicated().any():
        raise ValueError(f"{split} metadata row identity changed")
    if ordered_base_id_sha256(metadata.base_id) != manifest.get("ordered_base_id_sha256"):
        raise ValueError(f"{split} feature-cache row identity changed")
    if (
        metadata.group_id.isna().any()
        or (metadata.group_id.astype(str).str.strip() == "").any()
        or not metadata.label.isin((0, 1)).all()
    ):
        raise ValueError(f"{split} metadata labels/groups are invalid")
    if split in ("train", "validation"):
        expected = protocol[f"{split}_metadata"]
        if sha256(paths[f"{split}_metadata"]) != expected["sha256"]:
            raise ValueError(f"Registered {split} metadata changed")
        if manifest["ordered_base_id_sha256"] != expected["ordered_base_id_sha256"]:
            raise ValueError(f"Registered {split} row order changed")
    return cache, metadata


def _numpy_features(cache: np.memmap, indices: np.ndarray, layers: Sequence[int]) -> np.ndarray:
    if tuple(layers) == ALL_LAYERS:
        return np.asarray(cache[indices], dtype=np.float32)
    return np.stack([np.asarray(cache[indices, layer - 1, :], dtype=np.float32) for layer in layers], axis=1)


def _train_probe_pair(
    cache: np.memmap,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    candidate_layers: Sequence[int],
    protocol: dict,
    fold_seed: int,
) -> tuple[BranchProbes, BranchProbes, list[dict], dict]:
    settings = protocol["probes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Cross-fitted high-dimensional probes require CUDA")
    random.seed(settings["seed"]); np.random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    pool_slots = [layer_slot(layer) for layer in candidate_layers]
    pool = BranchProbes(
        protocol["hidden_size"], len(candidate_layers), settings["seed"], pool_slots
    ).to(device)
    repeat = BranchProbes(protocol["hidden_size"], 4, settings["seed"], [0, 1, 2, 3]).to(device)
    initial = assert_matched_branch_initialization(pool, repeat, list(candidate_layers).index(32))
    optimizers = [
        torch.optim.AdamW(pool.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]),
        torch.optim.AdamW(repeat.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]),
    ]
    labels = metadata.label.to_numpy(dtype=np.float32)
    row_weights = np.zeros(len(metadata), dtype=np.float64)
    row_weights[train_indices] = group_equal_weights(
        metadata.group_id.iloc[train_indices].astype(str).to_numpy()
    )
    history = []
    layer32_position = list(candidate_layers).index(32)
    for epoch in range(settings["epochs"]):
        order = np.random.default_rng(fold_seed * 1009 + epoch).permutation(train_indices)
        totals = np.zeros(2, dtype=np.float64)
        rows = 0
        raw_l32_parameter_difference = 0.0
        pool.train(); repeat.train()
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start : start + settings["batch_size"]]
            values = torch.from_numpy(_numpy_features(cache, indices, candidate_layers)).to(device)
            y = torch.from_numpy(labels[indices]).to(device)
            weight = torch.from_numpy(row_weights[indices].astype(np.float32)).to(device)
            pool_logits = pool(values)
            repeated = values[:, layer32_position : layer32_position + 1, :].expand(-1, 4, -1)
            repeat_logits = repeat(repeated)
            losses = []
            for logits in (pool_logits, repeat_logits):
                per_row = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction="none")
                losses.append((per_row * weight[:, None]).mean(0).sum())
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            (losses[0] + losses[1]).backward()
            for optimizer in optimizers:
                optimizer.step()
            synchronization = synchronize_matched_l32(
                pool, repeat, layer32_position, optimizers[0], optimizers[1]
            )
            raw_l32_parameter_difference = max(
                raw_l32_parameter_difference,
                synchronization["raw_parameter_max_absolute_difference"],
            )
            totals += np.asarray([float(loss.detach()) for loss in losses]) * len(indices)
            rows += len(indices)
        history.append(dict(epoch=epoch + 1, pool_weighted_bce_sum=float(totals[0] / rows),
                            repeat_weighted_bce_sum=float(totals[1] / rows),
                            raw_l32_parameter_max_absolute_difference=raw_l32_parameter_difference,
                            canonical_l32_parameter_max_absolute_difference=0.0))
    return pool, repeat, history, initial


@torch.no_grad()
def _predict_probe_pair(
    pool: BranchProbes,
    repeat: BranchProbes,
    cache: np.memmap,
    indices: np.ndarray,
    candidate_layers: Sequence[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(indices) == 0 or batch_size < 1:
        raise ValueError("Nonempty probe-prediction rows and a positive batch size are required")
    device = next(pool.parameters()).device
    pool.eval(); repeat.eval()
    pool_output = np.empty((len(indices), len(candidate_layers)), dtype=np.float32)
    repeat_output = np.empty((len(indices), 4), dtype=np.float32)
    position32 = list(candidate_layers).index(32)
    raw_l32_logit_difference = 0.0
    for start in range(0, len(indices), batch_size):
        rows = indices[start : start + batch_size]
        values = torch.from_numpy(_numpy_features(cache, rows, candidate_layers)).to(device)
        pool_batch = pool(values).cpu().numpy()
        pool_output[start : start + len(rows)] = pool_batch
        repeated = values[:, position32 : position32 + 1, :].expand(-1, 4, -1)
        repeat_batch = repeat(repeated).cpu().numpy()
        # Check the raw branch before replacing its registered slot-3 output;
        # otherwise a NaN/Inf in exactly that column would be hidden by the
        # canonical copy below.
        if not np.isfinite(pool_batch).all() or not np.isfinite(repeat_batch).all():
            raise FloatingPointError("Nonfinite raw base-probe predictions")
        raw_l32_logit_difference = max(
            raw_l32_logit_difference,
            float(np.max(np.abs(pool_batch[:, position32] - repeat_batch[:, 3]))),
        )
        # The two forward calls may use different GPU reduction kernels because
        # their branch dimensions are 32 and 4.  Store the registered shared
        # nuisance estimate exactly, while retaining the raw discrepancy below.
        repeat_batch[:, 3] = pool_batch[:, position32]
        repeat_output[start : start + len(rows)] = repeat_batch
    if not np.isfinite(pool_output).all() or not np.isfinite(repeat_output).all():
        raise FloatingPointError("Nonfinite base-probe predictions")
    if not np.array_equal(pool_output[:, position32], repeat_output[:, 3]):
        raise AssertionError("Canonical L32 prediction synchronization failed")
    return pool_output, repeat_output, dict(
        raw_gpu_l32_logit_max_absolute_difference=raw_l32_logit_difference,
        canonical_l32_logit_max_absolute_difference=0.0,
        convention="repeat slot 3 is the canonical pool L32 estimator",
    )


def _checkpoint_payload(pool: BranchProbes, repeat: BranchProbes, candidate_layers, identity, history) -> dict:
    layer32_position = list(candidate_layers).index(32)
    if not torch.equal(pool.weight[layer32_position], repeat.weight[3]) or not torch.equal(
        pool.bias[layer32_position], repeat.bias[3]
    ):
        raise AssertionError("Checkpoint would contain noncanonical matched L32 parameters")
    return dict(
        pool_state={name: value.detach().cpu() for name, value in pool.state_dict().items()},
        repeat_state={name: value.detach().cpu() for name, value in repeat.state_dict().items()},
        candidate_layers=list(candidate_layers), hidden_size=pool.weight.shape[1],
        identity=identity, history=history,
    )


def crossfit_probes(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    if protocol.get("external_oof") is not None:
        return validate_oof_bundle(output, protocol)
    directory = output / "base_probes"
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "oof_logits.npz"
    manifest_path = directory / "oof_logits.manifest.json"
    if bundle_path.is_file() and manifest_path.is_file() and (directory / "full_probes.pt").is_file():
        return validate_oof_bundle(output, protocol)
    cache, metadata = _load_cache(output, protocol, "train")
    labels = metadata.label.to_numpy(dtype=np.int64)
    groups = metadata.group_id.astype(str).to_numpy()
    folds = make_group_folds(labels, groups, protocol["folds"]["count"], protocol["folds"]["seed"])
    fold_frame = metadata[["base_id", "group_id", "label"]].copy()
    fold_frame["fold"] = folds
    folds_path = directory / "folds.parquet"
    fold_frame.to_parquet(folds_path, index=False)
    layers = tuple(protocol["candidate_layers"])
    layer_logits = np.full((len(metadata), len(layers)), np.nan, dtype=np.float32)
    repeat_logits = np.full((len(metadata), 4), np.nan, dtype=np.float32)
    fold_hashes = {}
    fold_prediction_diagnostics = {}
    for fold in range(protocol["folds"]["count"]):
        prediction_diagnostic = None
        prediction_path = directory / f"fold_{fold}_predictions.npz"
        checkpoint_path = directory / f"fold_{fold}_probes.pt"
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fold_identity = dict(
            fold=fold, heldout_rows=len(heldout), training_rows=len(training),
            heldout_base_id_sha256=ordered_base_id_sha256(metadata.base_id.iloc[heldout]),
            protocol_sha256=sha256(output / "protocol.json"),
        )
        if prediction_path.is_file() and checkpoint_path.is_file():
            saved = np.load(prediction_path, allow_pickle=False)
            if not np.array_equal(saved["row_indices"], heldout):
                raise ValueError(f"Fold {fold} saved row identity changed")
            saved_layer = np.asarray(saved["layer_logits"])
            saved_repeat = np.asarray(saved["repeat_last_logits"])
            if (
                saved_layer.shape != (len(heldout), len(layers))
                or saved_repeat.shape != (len(heldout), 4)
                or saved_layer.dtype != np.float32
                or saved_repeat.dtype != np.float32
            ):
                raise ValueError(f"Fold {fold} prediction shape changed")
            if not np.isfinite(saved_layer).all() or not np.isfinite(saved_repeat).all():
                raise FloatingPointError(f"Fold {fold} predictions are nonfinite")
            if not np.array_equal(saved_layer[:, layers.index(32)], saved_repeat[:, 3]):
                raise ValueError(f"Fold {fold} lacks the canonical matched L32 score")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            _validate_probe_checkpoint_payload(
                checkpoint, layers, fold_identity, protocol, f"Fold {fold}"
            )
            history_path = directory / f"fold_{fold}_history.json"
            history_value = _strict_json(history_path)
            _validate_probe_history_sidecar(
                history_value, checkpoint, fold_identity, layers, protocol, f"Fold {fold}"
            )
            prediction_diagnostic = history_value.get("prediction_diagnostic")
            if not _valid_matched_l32_prediction_diagnostic(prediction_diagnostic):
                raise ValueError(f"Fold {fold} lacks a valid matched-L32 prediction diagnostic")
            layer_logits[heldout] = saved_layer
            repeat_logits[heldout] = saved_repeat
        else:
            pool, repeat, history, initial = _train_probe_pair(
                cache, metadata, training, layers, protocol, protocol["probes"]["seed"] + fold
            )
            values, repeated, prediction_diagnostic = _predict_probe_pair(
                pool, repeat, cache, heldout, layers, protocol["probes"]["batch_size"]
            )
            layer32_position = layers.index(32)
            if not np.array_equal(values[:, layer32_position], repeated[:, 3]):
                raise AssertionError("Canonical pool/repeat L32 predictions differ")
            layer_logits[heldout], repeat_logits[heldout] = values, repeated
            temporary = prediction_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, row_indices=heldout, layer_logits=values, repeat_last_logits=repeated)
            temporary.replace(prediction_path)
            atomic_torch_save(_checkpoint_payload(pool, repeat, layers, fold_identity, history), checkpoint_path)
            dump(
                directory / f"fold_{fold}_history.json",
                dict(
                    identity=fold_identity, initial=initial, epochs=history,
                    prediction_diagnostic=prediction_diagnostic,
                ),
            )
            del pool, repeat
            torch.cuda.empty_cache()
        if not _valid_matched_l32_prediction_diagnostic(prediction_diagnostic):
            raise AssertionError(f"Fold {fold} prediction diagnostic was not validly recorded")
        fold_prediction_diagnostics[str(fold)] = prediction_diagnostic
        history_path = directory / f"fold_{fold}_history.json"
        fold_hashes[str(fold)] = dict(
            predictions_sha256=sha256(prediction_path),
            checkpoint_sha256=sha256(checkpoint_path),
            history_sha256=sha256(history_path),
        )
    if not np.isfinite(layer_logits).all() or not np.isfinite(repeat_logits).all():
        raise FloatingPointError("Cross-fitted base logits are incomplete")
    pool_repeat_difference = float(
        np.max(np.abs(layer_logits[:, layers.index(32)] - repeat_logits[:, 3]))
    )
    if pool_repeat_difference != 0.0:
        raise AssertionError("Canonical cross-fitted L32 differs from its matched slot-3 control")
    repeat_distinct = all(
        np.max(np.abs(repeat_logits[:, i] - repeat_logits[:, j])) > 0
        for i in range(4) for j in range(i)
    )
    if not repeat_distinct:
        raise AssertionError("Repeat-L32 control branches are not functionally distinct")
    # Reuse one pair-excluded probe fit for both directions of each nested
    # outer/inner pair.  With five global folds this needs C(5,2)=10 fits,
    # while preserving the exact nested exclusion contract.
    pair_artifacts = {}
    pair_predictions = {}
    pair_prediction_diagnostics = {}
    for first in range(protocol["folds"]["count"]):
        for second in range(first + 1, protocol["folds"]["count"]):
            training = np.flatnonzero((folds != first) & (folds != second))
            held_first = np.flatnonzero(folds == first)
            held_second = np.flatnonzero(folds == second)
            prediction_path = directory / f"nested_pair_{first}_{second}_predictions.npz"
            checkpoint_path = directory / f"nested_pair_{first}_{second}_probes.pt"
            identity = dict(
                excluded_folds=[first, second], training_rows=len(training),
                first_base_id_sha256=ordered_base_id_sha256(metadata.base_id.iloc[held_first]),
                second_base_id_sha256=ordered_base_id_sha256(metadata.base_id.iloc[held_second]),
                protocol_sha256=sha256(output / "protocol.json"),
            )
            if prediction_path.is_file() and checkpoint_path.is_file():
                saved = np.load(prediction_path, allow_pickle=False)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                _validate_probe_checkpoint_payload(
                    checkpoint, layers, identity, protocol, f"Nested pair {first}/{second}"
                )
                if not np.array_equal(saved["indices_first"], held_first) or not np.array_equal(
                    saved["indices_second"], held_second
                ):
                    raise ValueError(f"Nested pair {first}/{second} row identity changed")
                first_layer = np.asarray(saved["layer_logits_first"])
                first_repeat = np.asarray(saved["repeat_logits_first"])
                second_layer = np.asarray(saved["layer_logits_second"])
                second_repeat = np.asarray(saved["repeat_logits_second"])
                history_value = _strict_json(
                    directory / f"nested_pair_{first}_{second}_history.json"
                )
                _validate_probe_history_sidecar(
                    history_value, checkpoint, identity, layers, protocol,
                    f"Nested pair {first}/{second}",
                )
                first_diagnostic = history_value.get("prediction_first")
                second_diagnostic = history_value.get("prediction_second")
                if (
                    not _valid_matched_l32_prediction_diagnostic(first_diagnostic)
                    or not _valid_matched_l32_prediction_diagnostic(second_diagnostic)
                ):
                    raise ValueError(
                        f"Nested pair {first}/{second} lacks valid matched-L32 prediction diagnostics"
                    )
            else:
                pool, repeat, history, initial = _train_probe_pair(
                    cache, metadata, training, layers, protocol,
                    protocol["probes"]["seed"] + 1000 + first * 100 + second,
                )
                first_layer, first_repeat, first_diagnostic = _predict_probe_pair(
                    pool, repeat, cache, held_first, layers, protocol["probes"]["batch_size"]
                )
                second_layer, second_repeat, second_diagnostic = _predict_probe_pair(
                    pool, repeat, cache, held_second, layers, protocol["probes"]["batch_size"]
                )
                temporary = prediction_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream, indices_first=held_first, indices_second=held_second,
                        layer_logits_first=first_layer, repeat_logits_first=first_repeat,
                        layer_logits_second=second_layer, repeat_logits_second=second_repeat,
                    )
                temporary.replace(prediction_path)
                atomic_torch_save(
                    _checkpoint_payload(pool, repeat, layers, identity, history), checkpoint_path
                )
                dump(
                    directory / f"nested_pair_{first}_{second}_history.json",
                    dict(
                        identity=identity, initial=initial, epochs=history,
                        prediction_first=first_diagnostic,
                        prediction_second=second_diagnostic,
                    ),
                )
                del pool, repeat
                torch.cuda.empty_cache()
            for fold_number, layer_value, repeat_value, indices_value in (
                (first, first_layer, first_repeat, held_first),
                (second, second_layer, second_repeat, held_second),
            ):
                if (
                    layer_value.shape != (len(indices_value), len(layers))
                    or repeat_value.shape != (len(indices_value), 4)
                    or layer_value.dtype != np.float32
                    or repeat_value.dtype != np.float32
                    or not np.isfinite(layer_value).all()
                    or not np.isfinite(repeat_value).all()
                ):
                    raise ValueError(f"Nested pair {first}/{second} predictions changed")
                if not np.array_equal(layer_value[:, layers.index(32)], repeat_value[:, 3]):
                    raise AssertionError(f"Nested pair {first}/{second} canonical slot pairing diverged")
                pair_predictions[(first, second, fold_number)] = (layer_value, repeat_value)
            if (
                not _valid_matched_l32_prediction_diagnostic(first_diagnostic)
                or not _valid_matched_l32_prediction_diagnostic(second_diagnostic)
            ):
                raise AssertionError(
                    f"Nested pair {first}/{second} prediction diagnostics were not validly recorded"
                )
            pair_artifacts[f"{first}_{second}"] = dict(
                predictions=str(prediction_path), predictions_sha256=sha256(prediction_path),
                checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
                history=str(directory / f"nested_pair_{first}_{second}_history.json"),
                history_sha256=sha256(directory / f"nested_pair_{first}_{second}_history.json"),
            )
            pair_prediction_diagnostics[f"{first}_{second}"] = dict(
                first=first_diagnostic, second=second_diagnostic
            )
    nested_artifacts = {}
    for outer_fold in range(protocol["folds"]["count"]):
        outer_train = np.flatnonzero(folds != outer_fold)
        other_folds = [value for value in range(protocol["folds"]["count"]) if value != outer_fold]
        inner_fold = np.empty(len(outer_train), dtype=np.int16)
        inner_layer_logits = np.full((len(outer_train), len(layers)), np.nan, dtype=np.float32)
        inner_repeat_logits = np.full((len(outer_train), 4), np.nan, dtype=np.float32)
        for local_inner, global_inner in enumerate(other_folds):
            local_rows = np.flatnonzero(folds[outer_train] == global_inner)
            first, second = sorted((outer_fold, global_inner))
            values, repeated = pair_predictions[(first, second, global_inner)]
            expected_rows = outer_train[local_rows]
            if not np.array_equal(expected_rows, np.flatnonzero(folds == global_inner)):
                raise AssertionError("Nested global/local row mapping changed")
            inner_fold[local_rows] = local_inner
            inner_layer_logits[local_rows] = values
            inner_repeat_logits[local_rows] = repeated
        outer_path = directory / f"nested_outer_{outer_fold}.npz"
        temporary = outer_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, train_indices=outer_train, inner_fold=inner_fold,
                layer_logits=inner_layer_logits, repeat_last_logits=inner_repeat_logits,
                inner_global_folds=np.asarray(other_folds, dtype=np.int16),
                base_probe_excluded_pairs=np.asarray(
                    [[outer_fold, value] for value in other_folds], dtype=np.int16
                ),
            )
        temporary.replace(outer_path)
        nested_artifacts[str(outer_fold)] = dict(path=str(outer_path), sha256=sha256(outer_path))
    full_checkpoint = directory / "full_probes.pt"
    full_identity = dict(
        split="train", rows=len(metadata), protocol_sha256=sha256(output / "protocol.json"),
        base_id_sha256=ordered_base_id_sha256(metadata.base_id),
    )
    if not full_checkpoint.is_file():
        all_rows = np.arange(len(metadata), dtype=np.int64)
        pool, repeat, history, initial = _train_probe_pair(
            cache, metadata, all_rows, layers, protocol, protocol["probes"]["seed"] + 10000
        )
        atomic_torch_save(
            _checkpoint_payload(pool, repeat, layers, full_identity, history), full_checkpoint
        )
        dump(
            directory / "full_history.json",
            dict(identity=full_identity, initial=initial, epochs=history),
        )
        del pool, repeat
        torch.cuda.empty_cache()
    checkpoint = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
    _validate_probe_checkpoint_payload(
        checkpoint, layers, full_identity, protocol, "Full deployment"
    )
    _validate_probe_history_sidecar(
        _strict_json(directory / "full_history.json"), checkpoint, full_identity,
        layers, protocol, "Full deployment",
    )
    temporary = bundle_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, layer_logits=layer_logits, repeat_last_logits=repeat_logits,
                            fold=folds, layers=np.asarray(layers, dtype=np.int16))
    temporary.replace(bundle_path)
    manifest = dict(
        complete=True, bundle=str(bundle_path), bundle_sha256=sha256(bundle_path),
        train_metadata_sha256=sha256(_source_paths(Path(protocol["source"]))["train_metadata"]),
        ordered_base_id_sha256=ordered_base_id_sha256(metadata.base_id), rows=len(metadata), layers=list(layers),
        folds=protocol["folds"], probes=protocol["probes"], fold_artifacts=fold_hashes,
        folds_path=str(folds_path), folds_sha256=sha256(folds_path),
        nested_artifacts=nested_artifacts,
        nested_pair_artifacts=pair_artifacts,
        fold_prediction_diagnostics=fold_prediction_diagnostics,
        nested_pair_prediction_diagnostics=pair_prediction_diagnostics,
        full_checkpoint=str(full_checkpoint), full_checkpoint_sha256=sha256(full_checkpoint),
        full_history=str(directory / "full_history.json"),
        full_history_sha256=sha256(directory / "full_history.json"),
        repeat_branch_pairwise_distinct=bool(repeat_distinct),
        pool32_repeat32_slot3_max_absolute_difference=pool_repeat_difference,
        note="Every row is predicted by probes trained without its complete group_id; full checkpoint is train-only fixed-budget deployment fit.",
    )
    dump(manifest_path, manifest)
    return manifest


def _bundle_locations(output: Path, protocol: dict) -> tuple[Path, Path]:
    external = protocol.get("external_oof")
    if external:
        return Path(external["bundle"]), Path(external["manifest"])
    return Path(output) / "base_probes/oof_logits.npz", Path(output) / "base_probes/oof_logits.manifest.json"


def validate_oof_bundle(output: Path, protocol: dict | None = None) -> dict:
    output = Path(output).resolve()
    protocol = load_protocol(output) if protocol is None else protocol
    bundle_path, manifest_path = _bundle_locations(output, protocol)
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("OOF bundle is not complete; run crossfit or supply --oof-logits")
    manifest = _strict_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("OOF manifest is not marked complete")
    external = protocol.get("external_oof")
    expected_bundle_hash = external["bundle_sha256"] if external else manifest.get("bundle_sha256")
    expected_manifest_hash = external["manifest_sha256"] if external else None
    if sha256(bundle_path) != expected_bundle_hash:
        raise ValueError("OOF bundle hash changed")
    if expected_manifest_hash and sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("External OOF manifest hash changed")
    values = np.load(bundle_path, allow_pickle=False)
    aliases = {
        "layer_logits": ("layer_logits", "oof_logits", "logits"),
        "repeat_last_logits": ("repeat_last_logits", "repeat_logits"),
        "fold": ("fold", "fold_ids", "folds"),
        "layers": ("layers", "layer_ids"),
    }
    arrays = {}
    for canonical, choices in aliases.items():
        key = next((choice for choice in choices if choice in values.files), None)
        if key is not None:
            arrays[canonical] = np.asarray(values[key])
    if "layer_logits" not in arrays:
        raise ValueError(f"OOF bundle lacks layer_logits; available={values.files}")
    if "layers" not in arrays:
        layer_value = manifest.get("layers") or manifest.get("candidate_layers")
        if layer_value is None:
            raise ValueError(f"OOF bundle lacks layer IDs; available={values.files}")
        arrays["layers"] = np.asarray(layer_value, dtype=np.int16)
    rows = protocol["train_metadata"]["rows"]
    if arrays["layer_logits"].shape != (rows, len(arrays["layers"])):
        raise ValueError("OOF layer-logit shape changed")
    if "repeat_last_logits" not in arrays:
        raise ValueError("OOF bundle lacks four independently initialized repeat-L32 slot logits")
    paths = _source_paths(Path(protocol["source"]))
    train_metadata = pd.read_parquet(
        paths["train_metadata"], columns=["base_id", "group_id", "label"]
    )
    expected_fold = make_group_folds(
        train_metadata.label.to_numpy(dtype=np.int64),
        train_metadata.group_id.astype(str).to_numpy(),
        protocol["folds"]["count"], protocol["folds"]["seed"],
    )
    if "fold" not in arrays:
        arrays["fold"] = expected_fold
    if arrays["repeat_last_logits"].shape != (rows, 4) or arrays["fold"].shape != (rows,):
        raise ValueError("OOF repeat/fold shape changed")
    if tuple(map(int, arrays["layers"])) != tuple(protocol["candidate_layers"]):
        raise ValueError("OOF candidate layer order differs from protocol")
    if not np.isfinite(arrays["layer_logits"]).all() or not np.isfinite(arrays["repeat_last_logits"]).all():
        raise FloatingPointError("Nonfinite OOF logits")
    layer32_position = list(map(int, arrays["layers"])).index(32)
    if not np.array_equal(
        arrays["layer_logits"][:, layer32_position], arrays["repeat_last_logits"][:, 3]
    ):
        raise ValueError("OOF bundle lacks the exact canonical matched L32 score")
    manifest_row_hash = manifest.get("ordered_base_id_sha256") or manifest.get("base_id_sha256")
    if manifest_row_hash != protocol["train_metadata"]["ordered_base_id_sha256"]:
        raise ValueError("OOF bundle train row identity changed")
    if set(map(int, np.unique(arrays["fold"]))) != set(range(protocol["folds"]["count"])):
        raise ValueError("OOF fold IDs changed")
    if not np.array_equal(arrays["fold"], expected_fold):
        raise ValueError("OOF fold assignment differs from the registered deterministic split")
    groups = train_metadata.group_id.astype(str).to_numpy()
    if any(len(np.unique(arrays["fold"][groups == group])) != 1 for group in np.unique(groups)):
        raise ValueError("OOF fold assignment splits a group_id")
    if not external:
        if (
            manifest.get("rows") != rows
            or tuple(manifest.get("layers", ())) != tuple(protocol["candidate_layers"])
            or manifest.get("folds") != protocol["folds"]
            or manifest.get("probes") != protocol["probes"]
            or manifest.get("train_metadata_sha256") != protocol["train_metadata"]["sha256"]
        ):
            raise ValueError("Internal OOF manifest differs from its registered protocol")
        if manifest.get("pool32_repeat32_slot3_max_absolute_difference") != 0.0:
            raise ValueError("Internal OOF manifest reports a noncanonical matched L32 score")
        fold_count = int(protocol["folds"]["count"])
        expected_fold_keys = {str(value) for value in range(fold_count)}
        expected_pair_keys = {
            f"{first}_{second}"
            for first in range(fold_count)
            for second in range(first + 1, fold_count)
        }
        fold_diagnostics = manifest.get("fold_prediction_diagnostics", {})
        if set(fold_diagnostics) != expected_fold_keys or any(
            not _valid_matched_l32_prediction_diagnostic(value)
            for value in fold_diagnostics.values()
        ):
            raise ValueError("Internal OOF manifest lacks complete matched-L32 fold diagnostics")
        directory = manifest_path.parent
        folds_path = Path(manifest.get("folds_path", ""))
        if not folds_path.is_absolute():
            folds_path = directory / folds_path
        if not folds_path.is_file() or sha256(folds_path) != manifest.get("folds_sha256"):
            raise ValueError("Internal OOF fold-assignment artifact changed")
        fold_frame = pd.read_parquet(folds_path)
        expected_fold_frame = train_metadata[["base_id", "group_id", "label"]].copy()
        expected_fold_frame["fold"] = expected_fold
        if list(fold_frame.columns) != list(expected_fold_frame.columns) or not fold_frame.equals(
            expected_fold_frame
        ):
            raise ValueError("Internal OOF fold-assignment contents changed")
        fold_artifacts = manifest.get("fold_artifacts", {})
        if set(fold_artifacts) != expected_fold_keys:
            raise ValueError("Internal OOF manifest lacks complete outer-fold artifacts")
        for fold_number, value in fold_artifacts.items():
            if set(value) != {"predictions_sha256", "checkpoint_sha256", "history_sha256"}:
                raise ValueError("Malformed internal fold-artifact identity")
            expected_paths = {
                "predictions_sha256": directory / f"fold_{fold_number}_predictions.npz",
                "checkpoint_sha256": directory / f"fold_{fold_number}_probes.pt",
                "history_sha256": directory / f"fold_{fold_number}_history.json",
            }
            for key, path in expected_paths.items():
                if not path.is_file() or sha256(path) != value.get(key):
                    raise ValueError(f"Internal outer-fold artifact changed: {path}")
            fold_number_int = int(fold_number)
            heldout = np.flatnonzero(expected_fold == fold_number_int)
            prediction = np.load(expected_paths["predictions_sha256"], allow_pickle=False)
            if (
                not np.array_equal(prediction["row_indices"], heldout)
                or not np.array_equal(prediction["layer_logits"], arrays["layer_logits"][heldout])
                or not np.array_equal(
                    prediction["repeat_last_logits"], arrays["repeat_last_logits"][heldout]
                )
            ):
                raise ValueError(f"Internal outer-fold {fold_number} prediction contents changed")
            fold_identity = dict(
                fold=fold_number_int,
                heldout_rows=len(heldout),
                training_rows=rows - len(heldout),
                heldout_base_id_sha256=ordered_base_id_sha256(
                    train_metadata.base_id.iloc[heldout]
                ),
                protocol_sha256=sha256(output / "protocol.json"),
            )
            checkpoint = torch.load(
                expected_paths["checkpoint_sha256"], map_location="cpu", weights_only=True
            )
            _validate_probe_checkpoint_payload(
                checkpoint, protocol["candidate_layers"], fold_identity, protocol,
                f"Fold {fold_number}",
            )
            history = _strict_json(expected_paths["history_sha256"])
            _validate_probe_history_sidecar(
                history, checkpoint, fold_identity, protocol["candidate_layers"], protocol,
                f"Fold {fold_number}",
            )
            if history.get("prediction_diagnostic") != fold_diagnostics[fold_number]:
                raise ValueError(f"Fold {fold_number} prediction diagnostic changed")
        nested = manifest.get("nested_artifacts", {})
        if set(nested) != expected_fold_keys:
            raise ValueError("Internal OOF manifest lacks nested outer artifacts")
        for value in nested.values():
            if set(value) != {"path", "sha256"}:
                raise ValueError("Malformed internal nested outer-artifact identity")
            path = Path(value["path"])
            if not path.is_absolute():
                path = directory / path
            if not path.is_file() or sha256(path) != value.get("sha256"):
                raise ValueError(f"Internal nested artifact changed: {path}")
        pairs = manifest.get("nested_pair_artifacts", {})
        if set(pairs) != expected_pair_keys:
            raise ValueError("Internal OOF manifest lacks pair-excluded artifacts")
        for pair_key, value in pairs.items():
            if set(value) != {
                "predictions", "predictions_sha256", "checkpoint", "checkpoint_sha256",
                "history", "history_sha256",
            }:
                raise ValueError("Malformed pair-excluded artifact identity")
            resolved_pair_paths = {}
            for path_key, hash_key in (
                ("predictions", "predictions_sha256"),
                ("checkpoint", "checkpoint_sha256"),
                ("history", "history_sha256"),
            ):
                path = Path(value[path_key])
                if not path.is_absolute():
                    path = directory / path
                if not path.is_file() or sha256(path) != value.get(hash_key):
                    raise ValueError(f"Internal pair-excluded artifact changed: {path}")
                resolved_pair_paths[path_key] = path
            first, second = map(int, pair_key.split("_"))
            held_first = np.flatnonzero(expected_fold == first)
            held_second = np.flatnonzero(expected_fold == second)
            pair_identity = dict(
                excluded_folds=[first, second],
                training_rows=rows - len(held_first) - len(held_second),
                first_base_id_sha256=ordered_base_id_sha256(
                    train_metadata.base_id.iloc[held_first]
                ),
                second_base_id_sha256=ordered_base_id_sha256(
                    train_metadata.base_id.iloc[held_second]
                ),
                protocol_sha256=sha256(output / "protocol.json"),
            )
            pair_checkpoint = torch.load(
                resolved_pair_paths["checkpoint"], map_location="cpu", weights_only=True
            )
            _validate_probe_checkpoint_payload(
                pair_checkpoint, protocol["candidate_layers"], pair_identity, protocol,
                f"Nested pair {pair_key}",
            )
            pair_history = _strict_json(resolved_pair_paths["history"])
            _validate_probe_history_sidecar(
                pair_history, pair_checkpoint, pair_identity, protocol["candidate_layers"],
                protocol, f"Nested pair {pair_key}",
            )
            pair_prediction = np.load(resolved_pair_paths["predictions"], allow_pickle=False)
            if (
                set(pair_prediction.files) != {
                    "indices_first", "indices_second", "layer_logits_first",
                    "repeat_logits_first", "layer_logits_second", "repeat_logits_second",
                }
                or not np.array_equal(pair_prediction["indices_first"], held_first)
                or not np.array_equal(pair_prediction["indices_second"], held_second)
            ):
                raise ValueError(f"Nested pair {pair_key} prediction identity changed")
            for suffix, heldout in (("first", held_first), ("second", held_second)):
                layer_value = np.asarray(pair_prediction[f"layer_logits_{suffix}"])
                repeat_value = np.asarray(pair_prediction[f"repeat_logits_{suffix}"])
                if (
                    layer_value.shape != (len(heldout), len(protocol["candidate_layers"]))
                    or repeat_value.shape != (len(heldout), 4)
                    or layer_value.dtype != np.float32
                    or repeat_value.dtype != np.float32
                    or not np.isfinite(layer_value).all()
                    or not np.isfinite(repeat_value).all()
                    or not np.array_equal(layer_value[:, layer32_position], repeat_value[:, 3])
                ):
                    raise ValueError(f"Nested pair {pair_key} {suffix} predictions changed")
        pair_diagnostics = manifest.get("nested_pair_prediction_diagnostics", {})
        if set(pair_diagnostics) != expected_pair_keys or any(
            set(value) != {"first", "second"}
            or not _valid_matched_l32_prediction_diagnostic(value["first"])
            or not _valid_matched_l32_prediction_diagnostic(value["second"])
            for value in pair_diagnostics.values()
        ):
            raise ValueError("Internal OOF manifest lacks complete matched-L32 pair diagnostics")
        for pair_key, value in pair_diagnostics.items():
            history_path = Path(pairs[pair_key]["history"])
            if not history_path.is_absolute():
                history_path = directory / history_path
            history = _strict_json(history_path)
            if (
                history.get("prediction_first") != value["first"]
                or history.get("prediction_second") != value["second"]
            ):
                raise ValueError(f"Nested pair {pair_key} prediction diagnostics changed")
        if manifest.get("repeat_branch_pairwise_distinct") is not True:
            raise ValueError("Internal OOF repeat-L32 branches are not registered as distinct")
        full_checkpoint = Path(manifest.get("full_checkpoint", ""))
        full_history = Path(manifest.get("full_history", ""))
        if not full_checkpoint.is_absolute():
            full_checkpoint = directory / full_checkpoint
        if not full_history.is_absolute():
            full_history = directory / full_history
        if (
            not full_checkpoint.is_file()
            or sha256(full_checkpoint) != manifest.get("full_checkpoint_sha256")
            or not full_history.is_file()
            or sha256(full_history) != manifest.get("full_history_sha256")
        ):
            raise ValueError("Internal full deployment probe evidence changed")
        full_identity = dict(
            split="train", rows=rows, protocol_sha256=sha256(output / "protocol.json"),
            base_id_sha256=protocol["train_metadata"]["ordered_base_id_sha256"],
        )
        full_value = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
        _validate_probe_checkpoint_payload(
            full_value, protocol["candidate_layers"], full_identity, protocol, "Full deployment"
        )
        _validate_probe_history_sidecar(
            _strict_json(full_history), full_value, full_identity,
            protocol["candidate_layers"], protocol, "Full deployment",
        )
    return dict(manifest=manifest, arrays=arrays, bundle_path=bundle_path, manifest_path=manifest_path)


def load_nested_bundles(
    output: Path, protocol: dict, bundle: Mapping[str, Any] | None = None
) -> list[dict]:
    """Load the inner-OOF matrices used inside every strict outer fold."""
    bundle = validate_oof_bundle(output, protocol) if bundle is None else bundle
    artifacts = bundle["manifest"].get("nested_artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != protocol["folds"]["count"]:
        raise ValueError("OOF manifest lacks complete nested outer-fold artifacts")
    arrays = bundle["arrays"]
    outer_fold_ids = np.asarray(arrays["fold"], dtype=np.int64)
    groups = pd.read_parquet(
        _source_paths(Path(protocol["source"]))["train_metadata"], columns=["group_id"]
    ).group_id.astype(str).to_numpy()
    result = []
    for outer_fold in range(protocol["folds"]["count"]):
        artifact = artifacts.get(str(outer_fold))
        if not isinstance(artifact, dict):
            raise ValueError(f"Missing nested outer fold {outer_fold}")
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = bundle["manifest_path"].parent / path
        if not path.is_file() or sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Nested outer fold {outer_fold} changed")
        value = np.load(path, allow_pickle=False)
        training = np.flatnonzero(outer_fold_ids != outer_fold)
        heldout = np.flatnonzero(outer_fold_ids == outer_fold)
        if not np.array_equal(value["train_indices"], training):
            raise ValueError(f"Nested outer fold {outer_fold} row identity changed")
        inner_fold = np.asarray(value["inner_fold"], dtype=np.int64)
        layer_logits = np.asarray(value["layer_logits"], dtype=np.float64)
        repeat_logits = np.asarray(value["repeat_last_logits"], dtype=np.float64)
        if (
            inner_fold.shape != (len(training),)
            or layer_logits.shape != (len(training), len(protocol["candidate_layers"]))
            or repeat_logits.shape != (len(training), 4)
            or not np.isfinite(layer_logits).all()
            or not np.isfinite(repeat_logits).all()
        ):
            raise ValueError(f"Nested outer fold {outer_fold} layout changed")
        layer32_position = list(protocol["candidate_layers"]).index(32)
        if not np.array_equal(layer_logits[:, layer32_position], repeat_logits[:, 3]):
            raise ValueError(f"Nested outer fold {outer_fold} lacks the canonical matched L32 score")
        if set(map(int, np.unique(inner_fold))) != set(range(protocol["folds"]["inner_count"])):
            raise ValueError(f"Nested outer fold {outer_fold} inner fold IDs changed")
        expected_inner_global = np.asarray(
            [number for number in range(protocol["folds"]["count"]) if number != outer_fold],
            dtype=np.int16,
        )
        expected_exclusions = np.asarray(
            [[outer_fold, number] for number in expected_inner_global], dtype=np.int16
        )
        if (
            not np.array_equal(value["inner_global_folds"], expected_inner_global)
            or not np.array_equal(value["base_probe_excluded_pairs"], expected_exclusions)
        ):
            raise ValueError(f"Nested outer fold {outer_fold} base-probe exclusions changed")
        if set(groups[training]) & set(groups[heldout]):
            raise ValueError(f"Nested outer fold {outer_fold} leaks a heldout group")
        if any(
            len(np.unique(inner_fold[groups[training] == group])) != 1
            for group in np.unique(groups[training])
        ):
            raise ValueError(f"Nested outer fold {outer_fold} splits an inner group")
        result.append(
            dict(
                outer_fold=outer_fold, train_indices=training, heldout_indices=heldout,
                inner_fold=inner_fold, layer_logits=layer_logits, repeat_last_logits=repeat_logits,
                base_probe_excluded_pairs=expected_exclusions,
                outer_layer_logits=np.asarray(arrays["layer_logits"][heldout], dtype=np.float64),
                outer_repeat_last_logits=np.asarray(
                    arrays["repeat_last_logits"][heldout], dtype=np.float64
                ),
            )
        )
    return result


def _load_full_models(output: Path, protocol: dict) -> tuple[BranchProbes, BranchProbes, Path]:
    bundle = validate_oof_bundle(output, protocol)
    checkpoint_value = bundle["manifest"].get("full_checkpoint")
    if not checkpoint_value:
        raise ValueError("OOF manifest must provide a train-only full_probe_checkpoint")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = bundle["manifest_path"].parent / checkpoint_path
    expected = bundle["manifest"].get("full_checkpoint_sha256")
    if not checkpoint_path.is_file() or (expected and sha256(checkpoint_path) != expected):
        raise ValueError("Full deployment probe checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_identity = dict(
        split="train", rows=protocol["train_metadata"]["rows"],
        protocol_sha256=sha256(Path(output) / "protocol.json"),
        base_id_sha256=protocol["train_metadata"]["ordered_base_id_sha256"],
    )
    _validate_probe_checkpoint_payload(
        checkpoint, protocol["candidate_layers"], expected_identity, protocol,
        "Full deployment",
    )
    pool = BranchProbes(
        protocol["hidden_size"], len(protocol["candidate_layers"]), protocol["probes"]["seed"],
        [layer_slot(layer) for layer in protocol["candidate_layers"]],
    )
    repeat = BranchProbes(protocol["hidden_size"], 4, protocol["probes"]["seed"], [0, 1, 2, 3])
    pool.load_state_dict(checkpoint["pool_state"]); repeat.load_state_dict(checkpoint["repeat_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool.to(device); repeat.to(device)
    return pool, repeat, checkpoint_path


def _arm_features(
    arm: str,
    layer_logits: np.ndarray,
    repeat_logits: np.ndarray,
    layers: Sequence[int],
    selected_layers: Sequence[int],
) -> tuple[np.ndarray, list[int]]:
    position = {int(layer): index for index, layer in enumerate(layers)}
    if arm == "single32":
        chosen = [32]
        return layer_logits[:, [position[32]]], chosen
    if arm == "repeat32":
        return repeat_logits, [32, 32, 32, 32]
    if arm == "fixed4":
        chosen = list(FIXED_FUSION_LAYERS)
        return layer_logits[:, [position[layer] for layer in chosen]], chosen
    if arm == "selected4":
        chosen = list(map(int, selected_layers))
        return layer_logits[:, [position[layer] for layer in chosen]], chosen
    raise ValueError(arm)


def _nested_arm_records(
    nested_bundles: Sequence[Mapping[str, Any]],
    arm: str,
    layers: Sequence[int],
    selected_layers: Sequence[int],
) -> list[dict]:
    result = []
    for value in nested_bundles:
        train_features, _ = _arm_features(
            arm, value["layer_logits"], value["repeat_last_logits"], layers, selected_layers
        )
        heldout_features, _ = _arm_features(
            arm, value["outer_layer_logits"], value["outer_repeat_last_logits"],
            layers, selected_layers,
        )
        result.append(
            dict(
                outer_fold=value["outer_fold"], train_indices=value["train_indices"],
                heldout_indices=value["heldout_indices"], inner_fold=value["inner_fold"],
                train_features=train_features, heldout_features=heldout_features,
            )
        )
    return result


def _serializable_arm_fit(result: dict, layers: Sequence[int], hidden_size: int) -> dict:
    stack = result["stack"]
    original_coef = np.asarray(stack["coef"]) / np.asarray(stack["scale"])
    original_intercept = float(stack["intercept"] - np.sum(original_coef * np.asarray(stack["mean"])))
    calibration_parameters = int(result["calibrator"]["parameter_count"])
    return dict(
        layers=list(map(int, layers)), C=result["C"], metrics=result["metrics"], search=result["search"],
        stack=stack, calibrator=result["calibrator"],
        coefficients=dict(standardized=stack["coef"], original_logit_scale=original_coef.tolist(),
                          original_intercept=original_intercept),
        parameter_count=dict(
            branch_probes=len(layers) * (hidden_size + 1), stack=len(layers) + 1,
            calibrator=calibration_parameters,
            total=len(layers) * (hidden_size + 1) + len(layers) + 1 + calibration_parameters,
        ),
    )


def _predict_arm(arm_model: Mapping[str, Any], raw_features: np.ndarray) -> np.ndarray:
    score = predict_stack_score(arm_model["stack"], raw_features)
    return apply_calibrator(arm_model["calibrator"], score)


def _validate_existing_selection(output: Path, protocol: dict, selection: dict) -> dict:
    if selection.get("protocol_sha256") != sha256(output / "protocol.json"):
        raise ValueError("Existing stack selection belongs to another protocol")
    if set(selection.get("arms", {})) != set(ALL_ARMS):
        raise ValueError("Existing stack selection has incomplete arms")
    for arm in ALL_ARMS:
        model_path = output / "models" / f"{arm}.json"
        if not model_path.is_file() or _strict_json(model_path) != selection["arms"][arm]:
            raise ValueError(f"Existing {arm} model differs from selection")
    artifacts = {
        output / "oof_predictions.parquet": selection.get("oof_predictions_sha256"),
        output / "validation_predictions.parquet": selection.get("validation_predictions_sha256"),
        Path(selection.get("full_probe_checkpoint", "")): selection.get("full_probe_checkpoint_sha256"),
    }
    for path, expected in artifacts.items():
        if not path.is_file() or not expected or sha256(path) != expected:
            raise ValueError(f"Existing selected artifact changed: {path}")
    learned_path = output / "learned_layers.json"
    calibration_path = output / "calibration_selection.json"
    if not learned_path.is_file() or not calibration_path.is_file():
        raise ValueError("Existing selection lacks layer/calibration evidence")
    learned = _strict_json(learned_path)
    if any(learned.get(key) != value for key, value in selection["learned_layers"].items()):
        raise ValueError("Existing learned-layer evidence changed")
    if _strict_json(calibration_path) != selection.get("calibration_selection"):
        raise ValueError("Existing calibration selection changed")
    counts = selection.get("matched_parameter_counts", {})
    if set(counts) != set(MATCHED_ARMS) or len(set(counts.values())) != 1:
        raise ValueError("Existing matched capacities differ")
    bundle_path, _ = _bundle_locations(output, protocol)
    if sha256(bundle_path) != selection.get("oof_bundle_sha256"):
        raise ValueError("Existing selection OOF bundle changed")
    return selection


def fit_stacks(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    selection_path = output / "selection.json"
    if selection_path.is_file():
        return _validate_existing_selection(output, protocol, _strict_json(selection_path))
    bundle = validate_oof_bundle(output, protocol)
    arrays = bundle["arrays"]
    paths = _source_paths(Path(protocol["source"]))
    train_meta = pd.read_parquet(paths["train_metadata"])
    labels = train_meta.label.to_numpy(dtype=np.int64)
    groups = train_meta.group_id.astype(str).to_numpy()
    folds = arrays["fold"].astype(np.int64)
    layers = tuple(map(int, arrays["layers"]))
    c_grid = protocol["stack"]["c_grid"]
    nested_bundles = load_nested_bundles(output, protocol, bundle)
    repeat_features, _ = _arm_features(
        "repeat32", arrays["layer_logits"], arrays["repeat_last_logits"], layers, [32]
    )
    repeat_outer_records = _nested_arm_records(nested_bundles, "repeat32", layers, [32])
    calibration_selection = select_nested_calibration_family(
        repeat_features, repeat_outer_records, labels, groups, folds, c_grid,
        protocol["probes"]["seed"],
        protocol["stack"]["calibration_tie_tolerance"],
    )
    calibration_method = calibration_selection["selected"]
    learned = greedy_nested_stack_layers(
        arrays["layer_logits"], nested_bundles, layers, labels, groups, folds, c_grid,
        count=protocol["learned_layer_count"], anchor=protocol["learned_layer_anchor"],
        seed=protocol["probes"]["seed"], calibration_method=calibration_method,
    )
    fits = {}
    oof_frame = train_meta[["base_id", "group_id", "source", "category_id", "text_form", "label"]].copy()
    for arm in ALL_ARMS:
        features, arm_layers = _arm_features(
            arm, arrays["layer_logits"], arrays["repeat_last_logits"], layers, learned["layers"]
        )
        outer_records = _nested_arm_records(nested_bundles, arm, layers, learned["layers"])
        result = select_nested_stack_regularization(
            features, outer_records, labels, groups, folds, c_grid,
            protocol["probes"]["seed"] + ALL_ARMS.index(arm) * 100,
            calibration_method=calibration_method,
        )
        fits[arm] = _serializable_arm_fit(result, arm_layers, protocol["hidden_size"])
        oof_frame[f"probability_{arm}"] = result["probability"]
        oof_frame[f"prediction_{arm}"] = (result["probability"] >= 0.5).astype(np.int8)
    counts = {arm: fits[arm]["parameter_count"]["total"] for arm in MATCHED_ARMS}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Matched stack capacities differ: {counts}")
    if len({fits[arm]["calibrator"]["method"] for arm in MATCHED_ARMS}) != 1:
        raise AssertionError("Matched arms use different calibration topology")
    # The candidate itself is selected only by strict nested train outer-OOF
    # metrics.  Validation cannot swap fixed/selected based on its own ranking.
    train_candidate = min(
        ("fixed4", "selected4"), key=lambda arm: (fits[arm]["metrics"]["group_equal"]["bce"], arm)
    )
    model_directory = output / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    for arm, value in fits.items():
        dump(model_directory / f"{arm}.json", value)
    oof_path = output / "oof_predictions.parquet"
    oof_frame["fold"] = folds
    oof_frame.to_parquet(oof_path, index=False)
    dump(output / "learned_layers.json", dict(
        **learned, selection_split="train strict nested outer-group-OOF",
        metric="nested outer-heldout calibrated group-equal BCE",
    ))
    dump(output / "calibration_selection.json", calibration_selection)
    # Only now infer validation with probes fit on all training groups.  No
    # validation label influenced base probes, C, Platt, or layer selection.
    pool, repeat, full_checkpoint = _load_full_models(output, protocol)
    validation_cache, validation_meta = _load_cache(output, protocol, "validation")
    validation_indices = np.arange(len(validation_meta), dtype=np.int64)
    validation_layer_logits, validation_repeat_logits, validation_probe_diagnostic = _predict_probe_pair(
        pool, repeat, validation_cache, validation_indices, layers, protocol["probes"]["batch_size"]
    )
    validation_labels = validation_meta.label.to_numpy(dtype=np.int64)
    validation_groups = validation_meta.group_id.astype(str).to_numpy()
    validation_frame = validation_meta[["base_id", "group_id", "source", "category_id", "text_form", "label"]].copy()
    validation_metrics = {}
    for arm in ALL_ARMS:
        features, _ = _arm_features(
            arm, validation_layer_logits, validation_repeat_logits, layers, learned["layers"]
        )
        probability = _predict_arm(fits[arm], features)
        validation_frame[f"probability_{arm}"] = probability
        validation_frame[f"prediction_{arm}"] = (probability >= 0.5).astype(np.int8)
        validation_metrics[arm] = combined_metrics(validation_labels, probability, validation_groups, 0.5)
    gate = validation_gate(validation_metrics[train_candidate], validation_metrics["repeat32"],
                           protocol["validation_gate"])
    selected_arm = train_candidate if gate["passed"] else "repeat32"
    validation_path = output / "validation_predictions.parquet"
    validation_frame.to_parquet(validation_path, index=False)
    selection = dict(
        protocol_sha256=sha256(output / "protocol.json"),
        selection_order=[
            "repeat32 strict nested train outer-OOF chooses one shared identity/temperature/Platt calibration family",
            "strict nested train outer-OOF chooses learned layers, per-arm C, and fixed4-versus-selected4 candidate",
            "validation applies only the registered pass/fallback gate against repeat32",
            "threshold remains fixed at 0.5",
        ],
        learned_layers=learned, train_candidate=train_candidate, validation_gate=gate,
        calibration_selection=calibration_selection,
        selected_arm=selected_arm, validation_metrics=validation_metrics,
        matched_parameter_counts=counts, arms=fits,
        validation_matched_l32_diagnostic=validation_probe_diagnostic,
        oof_bundle_sha256=sha256(bundle["bundle_path"]),
        full_probe_checkpoint=str(full_checkpoint), full_probe_checkpoint_sha256=sha256(full_checkpoint),
        oof_predictions_sha256=sha256(oof_path), validation_predictions_sha256=sha256(validation_path),
    )
    dump(selection_path, selection)
    del pool, repeat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selection


def _barrier_artifacts(output: Path, protocol: dict) -> list[Path]:
    output = Path(output).resolve()
    bundle_path, manifest_path = _bundle_locations(output, protocol)
    paths = [
        output / "protocol.json", bundle_path, manifest_path, output / "learned_layers.json",
        output / "calibration_selection.json",
        output / "oof_predictions.parquet", output / "validation_predictions.parquet", output / "selection.json",
    ]
    paths.extend(output / "models" / f"{arm}.json" for arm in ALL_ARMS)
    manifest = _strict_json(manifest_path)
    checkpoint = Path(manifest["full_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    paths.append(checkpoint)
    for key in ("full_history", "folds_path"):
        if manifest.get(key):
            path = Path(manifest[key])
            paths.append(path if path.is_absolute() else manifest_path.parent / path)
    for fold_number, value in manifest.get("fold_artifacts", {}).items():
        paths.extend(
            (
                manifest_path.parent / f"fold_{fold_number}_predictions.npz",
                manifest_path.parent / f"fold_{fold_number}_probes.pt",
            )
        )
        if "history_sha256" in value:
            paths.append(manifest_path.parent / f"fold_{fold_number}_history.json")
    for value in manifest.get("nested_artifacts", {}).values():
        path = Path(value["path"])
        paths.append(path if path.is_absolute() else manifest_path.parent / path)
    for value in manifest.get("nested_pair_artifacts", {}).values():
        for key in ("predictions", "checkpoint", "history"):
            if key not in value:
                continue
            path = Path(value[key])
            paths.append(path if path.is_absolute() else manifest_path.parent / path)
    return paths


def lock_test_barrier(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    selection = fit_stacks(args)
    if selection["protocol_sha256"] != sha256(output / "protocol.json"):
        raise ValueError("Selection identity changed")
    # A resumed lock must revalidate the full training evidence rather than
    # adopting whatever bytes the nested manifest currently points at.
    bundle = validate_oof_bundle(output, protocol)
    load_nested_bundles(output, protocol, bundle)
    paths = _source_paths(Path(protocol["source"]))
    source_protocol = _strict_json(paths["protocol"])
    if sha256(Path(source_protocol["data_dir"]) / "test.parquet") != protocol["source_data_sha256"]["test"]:
        raise ValueError("Registered test dataset changed")
    if not all(paths[key].is_file() for key in ("test_cache", "test_manifest", "test_metadata")):
        raise FileNotFoundError("Source test cache is incomplete")
    artifacts = {}
    for path in _barrier_artifacts(output, protocol):
        artifacts[str(path.resolve())] = sha256(path)
    # Hash test cache inputs only after every training/validation choice is
    # concrete.  Hashing bytes does not deserialize labels or predictions.
    test_inputs = {str(paths[key].resolve()): sha256(paths[key]) for key in ("test_cache", "test_manifest", "test_metadata")}
    test_manifest = _strict_json(paths["test_manifest"])
    if (
        test_manifest.get("complete") is not True
        or paths["test_cache"].stat().st_size != test_manifest.get("bytes")
        or test_inputs[str(paths["test_cache"].resolve())] != test_manifest.get("sha256")
        or test_inputs[str(paths["test_metadata"].resolve())] != test_manifest.get("metadata_sha256")
    ):
        raise ValueError("Source test cache is incomplete or differs from its manifest")
    value = dict(
        locked=True, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol_sha256=sha256(output / "protocol.json"), artifacts=artifacts, test_inputs=test_inputs,
        selected_arm=selection["selected_arm"], threshold=0.5,
        rule="No test feature or label was deserialized before train selection and validation fallback were locked.",
    )
    barrier = output / "test_barrier.json"
    if barrier.exists():
        prior = _strict_json(barrier)
        comparable = {key: value[key] for key in ("protocol_sha256", "artifacts", "test_inputs", "selected_arm", "threshold")}
        if any(prior.get(key) != expected for key, expected in comparable.items()):
            raise ValueError("Existing test barrier differs from current locked artifacts")
        return prior
    dump(barrier, value)
    return value


def verify_test_barrier(output: Path) -> dict:
    output = Path(output).resolve()
    barrier_path = output / "test_barrier.json"
    if not barrier_path.is_file():
        raise FileNotFoundError("Test barrier is not locked")
    barrier = _strict_json(barrier_path)
    if barrier.get("locked") is not True:
        raise ValueError("Test barrier is not locked")
    for mapping_name in ("artifacts", "test_inputs"):
        for name, expected in barrier.get(mapping_name, {}).items():
            path = Path(name)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Locked {mapping_name} artifact changed: {path}")
    if barrier.get("protocol_sha256") != sha256(output / "protocol.json"):
        raise ValueError("Locked protocol changed")
    selection_path = output / "selection.json"
    if selection_path.is_file():
        selection = _strict_json(selection_path)
        if barrier.get("selected_arm") != selection.get("selected_arm") or barrier.get("threshold") != 0.5:
            raise ValueError("Locked test decision differs from selection")
    return barrier


def _source_metrics(metadata: pd.DataFrame, probability: np.ndarray) -> dict:
    result = {}
    labels = metadata.label.to_numpy(dtype=np.int64)
    groups = metadata.group_id.astype(str).to_numpy()
    sources = metadata.source.astype(str).to_numpy()
    for source in sorted(set(sources)):
        mask = sources == source
        result[source] = combined_metrics(labels[mask], probability[mask], groups[mask], 0.5)
    return result


def assert_test_split_disjointness(protocol: Mapping[str, Any], test_metadata: pd.DataFrame) -> dict:
    """After barrier unlock, hard-check row and cluster isolation from test."""
    paths = _source_paths(Path(protocol["source"]))
    train = pd.read_parquet(paths["train_metadata"], columns=["base_id", "group_id"])
    validation = pd.read_parquet(paths["validation_metadata"], columns=["base_id", "group_id"])
    frames = {"train": train, "validation": validation, "test": test_metadata}
    audit = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        base_overlap = set(frames[left].base_id.astype(str)) & set(frames[right].base_id.astype(str))
        group_overlap = set(frames[left].group_id.astype(str)) & set(frames[right].group_id.astype(str))
        if base_overlap or group_overlap:
            raise ValueError(
                f"Split isolation failed for {left}/{right}: "
                f"base_id={len(base_overlap)}, group_id={len(group_overlap)}"
            )
        audit[f"{left}_vs_{right}"] = dict(base_id_overlap=0, group_id_overlap=0)
    return audit


def paired_arm_comparisons(
    labels: Sequence[int],
    groups: Sequence[Any],
    probabilities: Mapping[str, Sequence[float]],
    repetitions: int,
    seed: int,
) -> dict:
    """Compare each four-layer arm with its explicit repeat-L32 control."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    control_probability = np.asarray(probabilities["repeat32"], dtype=np.float64)
    control_metrics = combined_metrics(labels, control_probability, groups, 0.5)
    control_prediction = control_probability >= 0.5
    comparisons = {}
    for arm in ("fixed4", "selected4"):
        candidate_probability = np.asarray(probabilities[arm], dtype=np.float64)
        candidate_metrics = combined_metrics(labels, candidate_probability, groups, 0.5)
        candidate_prediction = candidate_probability >= 0.5
        comparisons[arm] = dict(
            baseline="repeat32",
            accuracy_delta=(
                candidate_metrics["row"]["accuracy"] - control_metrics["row"]["accuracy"]
            ),
            group_equal_accuracy_delta=(
                candidate_metrics["group_equal"]["accuracy"]
                - control_metrics["group_equal"]["accuracy"]
            ),
            mcnemar=mcnemar_exact(labels, candidate_prediction, control_prediction),
            bootstrap=group_cluster_bootstrap(
                labels=labels, probability_a=candidate_probability,
                probability_b=control_probability, groups=groups,
                threshold_a=0.5, threshold_b=0.5, n_resamples=repetitions,
                seed=seed + ALL_ARMS.index(arm),
            ),
            permutation=group_permutation_test(
                labels=labels, probability_a=candidate_probability,
                probability_b=control_probability, groups=groups,
                threshold_a=0.5, threshold_b=0.5, n_resamples=repetitions,
                seed=seed + ALL_ARMS.index(arm),
            ),
        )
    return comparisons


def evaluate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    barrier = verify_test_barrier(output)
    result_path = output / "results.json"
    if result_path.is_file():
        prior = _strict_json(result_path)
        predictions_path = output / "test_predictions.parquet"
        if (
            prior.get("protocol_sha256") != sha256(output / "protocol.json")
            or prior.get("barrier_sha256") != sha256(output / "test_barrier.json")
            or prior.get("selected_arm") != barrier.get("selected_arm")
            or not predictions_path.is_file()
            or prior.get("test_predictions_sha256") != sha256(predictions_path)
        ):
            raise ValueError("Existing test result differs from the locked evaluation")
        return prior
    selection = _strict_json(output / "selection.json")
    learned_layers = selection["learned_layers"]["layers"]
    layers = tuple(protocol["candidate_layers"])
    pool, repeat, checkpoint_path = _load_full_models(output, protocol)
    test_cache, test_meta = _load_cache(output, protocol, "test")
    split_disjointness = assert_test_split_disjointness(protocol, test_meta)
    indices = np.arange(len(test_meta), dtype=np.int64)
    layer_logits, repeat_logits, test_probe_diagnostic = _predict_probe_pair(
        pool, repeat, test_cache, indices, layers, protocol["probes"]["batch_size"]
    )
    labels = test_meta.label.to_numpy(dtype=np.int64)
    groups = test_meta.group_id.astype(str).to_numpy()
    frame = test_meta[["base_id", "group_id", "source", "category_id", "text_form", "label"]].copy()
    entries = {}
    probabilities = {}
    for arm in ALL_ARMS:
        arm_model = _strict_json(output / "models" / f"{arm}.json")
        features, arm_layers = _arm_features(arm, layer_logits, repeat_logits, layers, learned_layers)
        probability = _predict_arm(arm_model, features)
        probabilities[arm] = probability
        frame[f"probability_{arm}"] = probability
        frame[f"prediction_{arm}"] = (probability >= 0.5).astype(np.int8)
        entries[arm] = dict(
            layers=arm_layers, parameter_count=arm_model["parameter_count"],
            oof_metrics=arm_model["metrics"], validation_metrics=selection["validation_metrics"][arm],
            test_metrics=combined_metrics(labels, probability, groups, 0.5),
            by_source=_source_metrics(test_meta, probability),
        )
    predictions_path = output / "test_predictions.parquet"
    frame.to_parquet(predictions_path, index=False)
    comparisons = paired_arm_comparisons(
        labels, groups, probabilities, protocol["bootstrap"]["repetitions"],
        protocol["bootstrap"]["seed"],
    )
    primary_arm = selection["selected_arm"]
    result = dict(
        protocol_sha256=sha256(output / "protocol.json"), barrier_sha256=sha256(output / "test_barrier.json"),
        full_probe_checkpoint_sha256=sha256(checkpoint_path), selected_layers=learned_layers,
        train_candidate=selection["train_candidate"], validation_gate=selection["validation_gate"],
        selected_arm=primary_arm, fixed_threshold=0.5, arms=entries, comparisons=comparisons,
        primary_test_metrics=entries[primary_arm]["test_metrics"],
        primary_vs_repeat32=(comparisons.get(primary_arm) if primary_arm in comparisons else None),
        test_matched_l32_diagnostic=test_probe_diagnostic,
        split_disjointness=split_disjointness,
        test_predictions_sha256=sha256(predictions_path), limitations=protocol["limitations"],
        note="Only selected_arm is the registered primary result; other test arms are exploratory diagnostics.",
    )
    dump(result_path, result)
    write_report(output, result)
    del pool, repeat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.3f}"


def write_report(output: Path, result: dict | None = None) -> Path:
    output = Path(output).resolve()
    result = _strict_json(output / "results.json") if result is None else result
    selection = _strict_json(output / "selection.json")
    protocol = _strict_json(output / "protocol.json")
    lines = [
        "# Group-aware OOF probe-logit stack",
        "",
        "本轮重新训练按 `group_id` 隔离的 nested cross-fitted base probes。外层 fold 的 stack、校准器和 inner OOF 特征都没有使用外层 heldout group；旧的全 train in-sample probe logits 未用于 stack 训练或校准。",
        "",
        f"候选层池：{', '.join(map(str, protocol['candidate_layers']))}；train strict nested outer-OOF learned-stack 选择：**{', '.join(map(str, result['selected_layers']))}**。",
        f"train strict nested outer-OOF 在 fixed4 与 selected4 中锁定 `{result['train_candidate']}`；validation gate {'通过' if result['validation_gate']['passed'] else '未通过'}，最终主结果为 `{result['selected_arm']}`。所有臂阈值固定为 0.5。",
        f"校准族只在 repeat32 strict nested train outer-OOF 上统一选择：`{selection['calibration_selection']['selected']}`；identity / temperature / Platt 的比较均已保存。",
        "",
        "## 容量与协议",
        "",
        f"`repeat32`、`fixed4`、`selected4` 都包含四个分别参数化的 2,561 参数 probe、一个 5 参数 L2 logistic stack、一个 {selection['arms']['repeat32']['parameter_count']['calibrator']} 参数 `{selection['calibration_selection']['selected']}` calibrator。四个 slot 使用不同的 seed-derived 初始化；每个多层 branch 与对应 repeat-L32 slot 的初值逐字节一致，并使用相同 fold、epoch、batch 顺序和优化预算。为避免 32-branch 与 4-branch GPU reduction 形状造成纯数值漂移，pool L32 被注册为跨臂共同 nuisance estimator：repeat slot 3 的参数和 AdamW state 每步同步，保存 logits 直接取规范 L32 值，原始 GPU forward 差异另行记录。selected4 被约束为每个八层深度带选择一个层。",
        "32 层候选 probe 为节省 I/O 在一个张量中向量化训练，但 loss 与 AdamW state 按 branch 可分，且不做跨 branch gradient clipping；未被某个 arm 使用的搜索 branch 不计入该 arm 参数。",
        "",
        "| 方法 | 层 | 总参数 | Nested outer-OOF group-BCE | Val group-BCE | Test Acc | Test group-Acc | Test AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ALL_ARMS:
        row = result["arms"][arm]
        lines.append(
            f"| {arm} | {','.join(map(str, row['layers']))} | {row['parameter_count']['total']:,} | "
            f"{row['oof_metrics']['group_equal']['bce']:.6f} | "
            f"{row['validation_metrics']['group_equal']['bce']:.6f} | "
            f"{_pct(row['test_metrics']['row']['accuracy'])} | {_pct(row['test_metrics']['group_equal']['accuracy'])} | "
            f"{_pct(row['test_metrics']['row']['auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Strict nested train outer-OOF calibration comparison",
            "",
            "| 校准 | 参数 | repeat32 outer-OOF group-BCE | repeat32 outer-OOF group-Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in ("identity", "temperature", "platt"):
        value = selection["calibration_selection"]["comparisons"][method]
        lines.append(
            f"| {method} | {value['parameter_count']} | "
            f"{value['metrics']['group_equal']['bce']:.6f} | "
            f"{value['metrics']['group_equal']['brier']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 等容量配对比较",
            "",
            "| 候选 − repeat32 | Row Acc Δ | Group Acc Δ | Bootstrap 95% CI | Group permutation p | 净修正 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ("fixed4", "selected4"):
        value = result["comparisons"][arm]
        accuracy = value["bootstrap"]["accuracy"]
        lines.append(
            f"| {arm} | {_pct(value['accuracy_delta'])} | {_pct(value['group_equal_accuracy_delta'])} | "
            f"[{_pct(accuracy['ci95'][0])}, {_pct(accuracy['ci95'][1])}] | "
            f"{value['permutation']['accuracy']['two_sided_p']:.4g} | {value['mcnemar']['net_corrections']} |"
        )
    lines.extend(
        [
            "",
            "## Validation gate",
            "",
            f"预注册规则：group-equal BCE 至少改善 {protocol['validation_gate']['minimum_bce_gain']:.6g}；"
            f"row accuracy 最多下降 {_pct(protocol['validation_gate']['accuracy_tolerance'])}pp；"
            f"AUROC 最多下降 {_pct(protocol['validation_gate']['auroc_tolerance'])}pp。",
            "",
        ]
    )
    for name, passed in result["validation_gate"]["checks"].items():
        lines.append(f"- `{name}`：{'通过' if passed else '未通过'}")
    lines.extend(
        [
            "",
            "主结果只解释预先锁定的 selected_arm；其余 test 臂、来源切片和配对 p 值均是探索性结果，未做多重比较校正。",
            "",
            "## 边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all(args: argparse.Namespace) -> dict:
    initialize(args)
    crossfit_probes(args)
    fit_stacks(args)
    lock_test_barrier(args)
    return evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "crossfit", "stack", "lock", "evaluate", "report", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate-layers", nargs="+", type=int, default=list(ALL_LAYERS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20261012)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--minimum-bce-gain", type=float, default=0.0)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.001)
    parser.add_argument("--auroc-tolerance", type=float, default=0.0005)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--oof-logits", type=Path,
                        help=(
                            "Optional external NPZ; its manifest must also provide strict nested_artifacts "
                            "and a compatible full checkpoint"
                        ))
    parser.add_argument("--oof-manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action in ("crossfit", "run") and args.oof_logits is None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train cross-fitted 2560-dimensional probes")
    if args.action == "init":
        result = initialize(args)
    elif args.action == "crossfit":
        result = crossfit_probes(args)
    elif args.action == "stack":
        result = fit_stacks(args)
    elif args.action == "lock":
        result = lock_test_barrier(args)
    elif args.action == "evaluate":
        result = evaluate(args)
    elif args.action == "report":
        result = dict(report=str(write_report(args.output)))
    else:
        result = run_all(args)
    print(json.dumps({"state": "complete", "action": args.action,
                      "selected_arm": result.get("selected_arm") if isinstance(result, dict) else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
