"""Frozen-LoRA layer probes, capacity controls, and error-complement diagnostics.

The experiment deliberately has two phases.  Training and model selection may
read only ``train`` and ``validation``.  ``test`` features cannot be extracted
or evaluated until :func:`create_test_barrier` has locked every validation-only
selection artifact.

Typical use::

    python scripts/hanguard/frozen_lora_layer_diagnostics.py run \
      --output outputs/hanguard_frozen_lora_layer_diagnostics_20261012

The individual phases are ``init``, ``extract``, ``probes``, ``select``,
``heads``, ``lock``, ``evaluate``, and ``report``.  Extraction writes one raw
float16 last-token vector per transformer block to a resumable ``.npy``
memmap.  No full-token hidden-state cache is ever constructed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE_STUDY = ROOT / "outputs/hanguard_repaired_core_20260928"
DEFAULT_ADAPTER = DEFAULT_SOURCE_STUDY / "runs/E03_s42/adapter.pt"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_frozen_lora_layer_diagnostics_20261012"
ALL_LAYERS = tuple(range(1, 33))
FIXED_FUSION_LAYERS = (8, 16, 24, 32)
MATCHED_ARMS = ("repeat_last", "fusion_fixed", "fusion_selected")
ALL_ARMS = ("last",) + MATCHED_ARMS


def dump(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(piece)
    return result.hexdigest()


def state_digest(state_or_module: Any) -> str:
    """Stable digest of tensor names, dtypes, shapes, and bytes."""
    state = state_or_module.state_dict() if isinstance(state_or_module, nn.Module) else state_or_module
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        result.update(name.encode("utf-8"))
        result.update(str(tensor.dtype).encode("ascii"))
        result.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def ordered_base_id_sha256(base_ids: Sequence[Any]) -> str:
    values = [str(value) for value in base_ids]
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def last_true_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the final true position in every row, for either padding side."""
    mask = torch.as_tensor(mask).bool()
    if mask.ndim != 2 or not mask.any(1).all():
        raise ValueError("Every mask row must contain at least one true token")
    positions = torch.arange(mask.shape[1], device=mask.device)
    return positions[None].expand_as(mask).masked_fill(~mask, -1).max(1).values


def validate_feature_cache(
    path: Path,
    shape: Sequence[int],
    dtype: np.dtype | str,
    expected_base_id_sha256: str,
    base_ids: Sequence[Any],
    chunk_rows: int = 512,
) -> dict:
    """Validate exact NPY extent, layout, ordered row identity, and all values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(map(int, shape))
    expected_dtype = np.dtype(dtype)
    if array.shape != expected_shape or array.dtype != expected_dtype:
        raise ValueError(
            f"Feature cache shape/dtype changed: {array.shape}/{array.dtype}, expected {expected_shape}/{expected_dtype}"
        )
    expected_bytes = int(array.offset + np.prod(expected_shape, dtype=np.int64) * expected_dtype.itemsize)
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Feature cache byte size changed: {path.stat().st_size} != {expected_bytes}")
    actual_id_hash = ordered_base_id_sha256(base_ids)
    if actual_id_hash != expected_base_id_sha256 or len(base_ids) != expected_shape[0]:
        raise ValueError("Feature cache ordered base_id hash/row identity changed")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    for start in range(0, len(array), chunk_rows):
        if not np.isfinite(np.asarray(array[start : start + chunk_rows])).all():
            raise FloatingPointError(f"Nonfinite feature cache values at rows {start}..{start + chunk_rows}")
    return dict(
        shape=list(expected_shape),
        dtype=str(expected_dtype),
        bytes=expected_bytes,
        finite_rows_checked=int(len(array)),
        ordered_base_id_sha256=actual_id_hash,
    )


def validate_lora_load(adapter_state: Mapping[str, torch.Tensor], load_result: Any) -> dict:
    """Reject any missing LoRA tensor or unexpected adapter tensor after load."""
    missing = [str(key) for key in getattr(load_result, "missing_keys", []) if "lora_" in str(key)]
    unexpected = [str(key) for key in getattr(load_result, "unexpected_keys", [])]
    if missing:
        raise ValueError(f"Frozen adapter has missing LoRA keys after load: {missing[:5]}")
    if unexpected:
        raise ValueError(f"Frozen adapter has unexpected keys after load: {unexpected[:5]}")
    if not adapter_state or not any("lora_" in str(key).lower() for key in adapter_state):
        # PEFT checkpoints sometimes omit the literal lora_ prefix in exported
        # keys, so accept a nonempty state only when the load result confirms
        # all generated LoRA parameters were filled.
        if not adapter_state:
            raise ValueError("Frozen adapter state is empty")
    return dict(adapter_tensors=len(adapter_state), missing_lora_keys=0, unexpected_keys=0)


def _code_identity() -> dict[str, str]:
    paths = {
        "frozen_lora_layer_diagnostics.py": Path(__file__),
        "repaired_heads.py": ROOT / "scripts/hanguard/repaired_heads.py",
        "repaired_study.py": ROOT / "scripts/hanguard/repaired_study.py",
        "hanguard_model.py": ROOT / "hanguard_model.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _model_artifact_identity(model_path: Path) -> dict:
    model_path = Path(model_path)
    index = model_path / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Missing model weight index: {index}")
    value = json.loads(index.read_text())
    shards = sorted(set(value.get("weight_map", {}).values()))
    if not shards:
        raise ValueError("Model weight index contains no shards")
    sizes = {}
    for name in shards:
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model weight shard: {path}")
        sizes[name] = path.stat().st_size
    return dict(index_sha256=sha256(index), shard_bytes=sizes)


def atomic_torch_save(value: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0
    output = np.empty_like(logits)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def accuracy_threshold(y: Sequence[int], probability: Sequence[float]) -> float:
    """Exactly maximize validation accuracy; ties use the value nearest 0.5."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError("Nonempty aligned finite predictions required")
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    cumulative = np.r_[0, np.cumsum(1 - 2 * ys)]
    correct = np.r_[ys.sum() + cumulative[starts], (1 - ys).sum()]
    thresholds = np.r_[ps[starts], np.nextafter(ps[-1], np.inf)]
    best = np.flatnonzero(correct == correct.max())
    return float(thresholds[best[np.argmin(abs(thresholds[best] - 0.5))]])


def binary_metrics(y: Sequence[int], probability: Sequence[float], threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if y.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("Aligned finite labels and probabilities required")
    prediction = probability >= threshold
    tp = int(((y == 1) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    result = dict(
        n=int(len(y)),
        accuracy=(tp + tn) / max(1, len(y)),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
        recall=tp / max(1, tp + fn),
        precision=tp / max(1, tp + fp),
        fpr=fp / max(1, fp + tn),
        bce=float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
        brier=float(np.square(probability - y).mean()),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        threshold=float(threshold),
    )
    result["auroc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
    result["auprc"] = float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else None
    return result


def _confusion(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            ((y == 1) & prediction).sum(),
            ((y == 0) & ~prediction).sum(),
            ((y == 0) & prediction).sum(),
            ((y == 1) & ~prediction).sum(),
        ],
        dtype=np.float64,
    )


def _metrics_from_confusion(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    tp, tn, fp, fn = (values[..., i] for i in range(4))
    total = np.maximum(1.0, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    f1 = 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)
    return accuracy, f1


def _group_confusions(
    y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, groups: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(groups).astype(str)
    if not (len(y) == len(candidate) == len(baseline) == len(groups)):
        raise ValueError("Labels, predictions, and groups must align")
    unique, inverse = np.unique(groups, return_inverse=True)
    candidate_counts = np.zeros((len(unique), 4), dtype=np.float64)
    baseline_counts = np.zeros_like(candidate_counts)
    for column, mask in enumerate(
        (
            (y == 1) & candidate,
            (y == 0) & ~candidate,
            (y == 0) & candidate,
            (y == 1) & ~candidate,
        )
    ):
        candidate_counts[:, column] = np.bincount(inverse, weights=mask, minlength=len(unique))
    for column, mask in enumerate(
        (
            (y == 1) & baseline,
            (y == 0) & ~baseline,
            (y == 0) & baseline,
            (y == 1) & ~baseline,
        )
    ):
        baseline_counts[:, column] = np.bincount(inverse, weights=mask, minlength=len(unique))
    return unique, candidate_counts, baseline_counts


def group_cluster_bootstrap(
    labels: Sequence[int],
    probability_a: Sequence[float],
    probability_b: Sequence[float],
    groups: Sequence[Any],
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    n_resamples: int = 2000,
    seed: int = 42,
    *,
    repetitions: int | None = None,
) -> dict:
    """Paired cluster bootstrap, resampling whole ``group_id`` clusters."""
    y = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(probability_a) >= threshold_a
    baseline = np.asarray(probability_b) >= threshold_b
    repetitions = int(n_resamples if repetitions is None else repetitions)
    unique, a, b = _group_confusions(y, candidate, baseline, groups)
    if repetitions < 1 or len(unique) < 1:
        raise ValueError("Positive repetitions and at least one group required")
    observed_a = a.sum(0)
    observed_b = b.sum(0)
    acc_a, f1_a = _metrics_from_confusion(observed_a)
    acc_b, f1_b = _metrics_from_confusion(observed_b)
    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(repetitions, dtype=np.float64)
    f1_samples = np.empty(repetitions, dtype=np.float64)
    # Small chunks bound memory when nearly every row has a unique group.
    cursor = 0
    while cursor < repetitions:
        size = min(8, repetitions - cursor)
        draws = rng.integers(0, len(unique), size=(size, len(unique)))
        a_sum = a[draws].sum(1)
        b_sum = b[draws].sum(1)
        a_acc, a_f1 = _metrics_from_confusion(a_sum)
        b_acc, b_f1 = _metrics_from_confusion(b_sum)
        accuracy_samples[cursor : cursor + size] = a_acc - b_acc
        f1_samples[cursor : cursor + size] = a_f1 - b_f1
        cursor += size

    def summarize(observed: float, samples: np.ndarray) -> dict:
        return dict(
            delta=float(observed),
            ci95=[float(x) for x in np.quantile(samples, [0.025, 0.975])],
            bootstrap_mean=float(samples.mean()),
        )

    result = dict(
        groups=int(len(unique)),
        repetitions=int(repetitions),
        seed=int(seed),
        unit="group_id cluster",
        accuracy=summarize(float(acc_a - acc_b), accuracy_samples),
        f1=summarize(float(f1_a - f1_b), f1_samples),
    )
    result.update(
        observed=result["accuracy"]["delta"],
        ci_low=result["accuracy"]["ci95"][0],
        ci_high=result["accuracy"]["ci95"][1],
    )
    return result


def group_permutation_test(
    labels: Sequence[int],
    probability_a: Sequence[float],
    probability_b: Sequence[float],
    groups: Sequence[Any],
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    n_resamples: int = 2000,
    seed: int = 42,
    *,
    repetitions: int | None = None,
) -> dict:
    """Paired randomization test that swaps complete groups between systems."""
    y = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(probability_a) >= threshold_a
    baseline = np.asarray(probability_b) >= threshold_b
    repetitions = int(n_resamples if repetitions is None else repetitions)
    unique, a, b = _group_confusions(y, candidate, baseline, groups)
    if repetitions < 1 or len(unique) < 1:
        raise ValueError("Positive repetitions and at least one group required")
    acc_a, f1_a = _metrics_from_confusion(a.sum(0))
    acc_b, f1_b = _metrics_from_confusion(b.sum(0))
    observed_accuracy = float(acc_a - acc_b)
    observed_f1 = float(f1_a - f1_b)
    rng = np.random.default_rng(seed)
    null_accuracy = np.empty(repetitions, dtype=np.float64)
    null_f1 = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        size = min(32, repetitions - cursor)
        swap = rng.integers(0, 2, size=(size, len(unique)), dtype=np.int8).astype(bool)
        # Aggregate confusion counts after swapping each paired group as a unit.
        chosen_a = np.where(swap[..., None], b[None], a[None]).sum(1)
        chosen_b = np.where(swap[..., None], a[None], b[None]).sum(1)
        aa, af = _metrics_from_confusion(chosen_a)
        ba, bf = _metrics_from_confusion(chosen_b)
        null_accuracy[cursor : cursor + size] = aa - ba
        null_f1[cursor : cursor + size] = af - bf
        cursor += size

    def summarize(observed: float, null: np.ndarray) -> dict:
        return dict(
            delta=observed,
            two_sided_p=float((1 + (np.abs(null) >= abs(observed) - 1e-15).sum()) / (len(null) + 1)),
            null_mean=float(null.mean()),
        )

    result = dict(
        groups=int(len(unique)),
        repetitions=int(repetitions),
        seed=int(seed),
        unit="group_id cluster swap",
        accuracy=summarize(observed_accuracy, null_accuracy),
        f1=summarize(observed_f1, null_f1),
    )
    result.update(
        observed=result["accuracy"]["delta"],
        p_value=result["accuracy"]["two_sided_p"],
    )
    return result


def mcnemar_exact(y: Sequence[int], candidate: Sequence[bool], baseline: Sequence[bool]) -> dict:
    from scipy.stats import binomtest

    y = np.asarray(y, dtype=np.int64)
    a_correct = np.asarray(candidate, dtype=bool) == y
    b_correct = np.asarray(baseline, dtype=bool) == y
    fixes = int((a_correct & ~b_correct).sum())
    harms = int((~a_correct & b_correct).sum())
    return dict(
        candidate_fixes=fixes,
        candidate_harms=harms,
        net_corrections=fixes - harms,
        two_sided_p=float(binomtest(min(fixes, harms), fixes + harms, 0.5).pvalue) if fixes + harms else 1.0,
    )


def _error_complementarity_matrix(
    y: Sequence[int],
    probabilities: np.ndarray,
    thresholds: Sequence[float] | float | None = None,
    layers: Sequence[int] | None = None,
) -> dict:
    """Return probability, decision, error-overlap, rescue, and oracle matrices."""
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    if probabilities.ndim != 2 or len(probabilities) != len(y) or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite [rows,layers] array")
    count = probabilities.shape[1]
    layers = tuple(range(1, count + 1)) if layers is None else tuple(int(x) for x in layers)
    if len(layers) != count:
        raise ValueError("Layer labels do not match probability columns")
    if thresholds is None:
        thresholds = [accuracy_threshold(y, probabilities[:, i]) for i in range(count)]
    if np.isscalar(thresholds):
        thresholds = np.full(count, float(thresholds))
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (count,):
        raise ValueError("One threshold per probability column is required")
    prediction = probabilities >= thresholds[None]
    correct = prediction == y[:, None]
    error = ~correct
    centered = probabilities - probabilities.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(centered).sum(0))
    correlation = centered.T @ centered / np.maximum(denominator[:, None] * denominator[None, :], 1e-30)
    np.fill_diagonal(correlation, 1.0)
    disagreement_count = np.not_equal(prediction[:, :, None], prediction[:, None, :]).sum(0)
    both_error = np.logical_and(error[:, :, None], error[:, None, :]).sum(0)
    either_error = np.logical_or(error[:, :, None], error[:, None, :]).sum(0)
    error_jaccard = both_error / np.maximum(1, either_error)
    # [i,j] counts examples rescued by i relative to j: i correct, j wrong.
    directed_rescue = np.logical_and(correct[:, :, None], error[:, None, :]).sum(0)
    error_count = error.sum(0)
    directed_rescue_rate = directed_rescue / np.maximum(1, error_count[None, :])
    oracle_accuracy = np.logical_or(correct[:, :, None], correct[:, None, :]).mean(0)
    return dict(
        layers=list(layers),
        thresholds=thresholds.tolist(),
        probability_pearson=correlation.tolist(),
        decision_disagreement_count=disagreement_count.astype(int).tolist(),
        decision_disagreement_rate=(disagreement_count / max(1, len(y))).tolist(),
        both_error_count=both_error.astype(int).tolist(),
        error_jaccard=error_jaccard.tolist(),
        directed_rescue_count=directed_rescue.astype(int).tolist(),
        directed_rescue_rate=directed_rescue_rate.tolist(),
        oracle_pair_accuracy=oracle_accuracy.tolist(),
        matrix_conventions=dict(
            directed_rescue="cell [i,j] is count/rate of layer i correct among errors made by layer j",
            directed_rescue_zero_denominator="rate is 0 when layer j has no errors",
            error_jaccard="intersection(error_i,error_j) / union(error_i,error_j)",
            error_jaccard_zero_union="value is 0 when neither layer makes an error",
            oracle_pair_accuracy="fraction correct by at least one of layers i and j",
        ),
    )


def error_complementarity(
    y: Sequence[int],
    probability_a: np.ndarray,
    probability_b: np.ndarray | None = None,
    threshold_a: float = 0.5,
    threshold_b: float = 0.5,
    *,
    thresholds: Sequence[float] | float | None = None,
    layers: Sequence[int] | None = None,
) -> dict:
    """Pairwise four-cell error accounting or a full layer matrix.

    Passing ``probability_b`` selects the compact two-system API.  Omitting it
    accepts a ``[rows,layers]`` matrix and returns all 32x32 diagnostics.
    """
    if probability_b is None:
        return _error_complementarity_matrix(y, probability_a, thresholds, layers)
    labels = np.asarray(y, dtype=np.int64)
    a = np.asarray(probability_a, dtype=np.float64) >= threshold_a
    b = np.asarray(probability_b, dtype=np.float64) >= threshold_b
    if labels.shape != a.shape or a.shape != b.shape:
        raise ValueError("Labels and paired probabilities must align")
    a_correct = a == labels
    b_correct = b == labels
    both_correct = int((a_correct & b_correct).sum())
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    both_wrong = int((~a_correct & ~b_correct).sum())
    return dict(
        both_correct=both_correct,
        a_only_correct=a_only,
        b_only_correct=b_only,
        both_wrong=both_wrong,
        a_errors=b_only + both_wrong,
        b_errors=a_only + both_wrong,
        shared_errors=both_wrong,
        net_correct_a_minus_b=a_only - b_only,
        decision_disagreement_rate=(a_only + b_only) / max(1, len(labels)),
        error_jaccard=both_wrong / max(1, a_only + b_only + both_wrong),
        oracle_accuracy=(both_correct + a_only + b_only) / max(1, len(labels)),
    )


def select_complementary_layers(
    y: Sequence[int],
    probabilities: np.ndarray | Mapping[int, Sequence[float]],
    thresholds: Sequence[float] | None = None,
    layers: Sequence[int] | None = None,
    *,
    split: str = "validation",
    count: int = 4,
    anchor: int = 32,
    auroc_margin: float = 0.002,
    accuracy_margin: float = 0.02,
    auc_tolerance: float | None = None,
) -> dict:
    """Greedily maximize validation error rescue while anchoring the final layer."""
    from sklearn.metrics import roc_auc_score

    if split != "validation":
        raise ValueError("Layer selection is validation-only")
    y = np.asarray(y, dtype=np.int64)
    if isinstance(probabilities, Mapping):
        if layers is not None:
            raise ValueError("Do not pass layers with a layer-to-probability mapping")
        layers = tuple(int(layer) for layer in probabilities)
        probabilities = np.column_stack([probabilities[layer] for layer in probabilities])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(y):
        raise ValueError("Expected [rows,layers] validation probabilities")
    layer_labels = tuple(range(1, probabilities.shape[1] + 1)) if layers is None else tuple(map(int, layers))
    if len(layer_labels) != probabilities.shape[1] or anchor not in layer_labels:
        raise ValueError("Layer labels must align and contain anchor")
    if not 1 <= count <= len(layer_labels):
        raise ValueError("Invalid requested layer count")
    if thresholds is None:
        thresholds = [accuracy_threshold(y, probabilities[:, i]) for i in range(len(layer_labels))]
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (len(layer_labels),):
        raise ValueError("One validation threshold per layer is required")
    prediction = probabilities >= thresholds[None]
    correct = prediction == y[:, None]
    accuracies = correct.mean(0)
    aurocs = np.asarray([roc_auc_score(y, probabilities[:, i]) for i in range(len(layer_labels))])
    anchor_index = layer_labels.index(anchor)
    if auc_tolerance is not None:
        auroc_margin = float(auc_tolerance)
    eligible = {
        i
        for i in range(len(layer_labels))
        if aurocs[i] >= aurocs[anchor_index] - auroc_margin
        and accuracies[i] >= accuracies[anchor_index] - accuracy_margin
    }
    eligible.add(anchor_index)
    fallback_indices: list[int] = []
    if len(eligible) < count:
        additions = sorted(
            (i for i in range(len(layer_labels)) if i not in eligible),
            key=lambda i: (-aurocs[i], -accuracies[i], layer_labels[i]),
        )
        fallback_indices = additions[: count - len(eligible)]
        eligible.update(fallback_indices)
    selected = [anchor_index]
    oracle = correct[:, anchor_index].copy()
    trace = [
        dict(
            step=1,
            layer=int(anchor),
            newly_rescued=0,
            remaining_oracle_errors=int((~oracle).sum()),
            mean_absolute_probability_correlation=1.0,
            validation_auroc=float(aurocs[anchor_index]),
            validation_accuracy=float(accuracies[anchor_index]),
        )
    ]
    while len(selected) < count:
        candidates = []
        for index in eligible - set(selected):
            rescue = int((~oracle & correct[:, index]).sum())
            residual_error = int((~(oracle | correct[:, index])).sum())
            correlations = []
            for prior in selected:
                value = np.corrcoef(probabilities[:, index], probabilities[:, prior])[0, 1]
                correlations.append(abs(float(value)) if np.isfinite(value) else 1.0)
            mean_abs_correlation = float(np.mean(correlations))
            candidates.append(
                (rescue, -mean_abs_correlation, aurocs[index], accuracies[index], -layer_labels[index], index, residual_error)
            )
        if not candidates:
            raise RuntimeError("Insufficient eligible layers after fallback")
        chosen = max(candidates)
        index = chosen[5]
        trace.append(
            dict(
                step=len(selected) + 1,
                layer=layer_labels[index],
                newly_rescued=int(chosen[0]),
                remaining_oracle_errors=int(chosen[6]),
                mean_absolute_probability_correlation=float(-chosen[1]),
                validation_auroc=float(aurocs[index]),
                validation_accuracy=float(accuracies[index]),
            )
        )
        selected.append(index)
        oracle |= correct[:, index]
    chosen_layers = sorted(layer_labels[i] for i in selected)
    return dict(
        split=split,
        anchor=int(anchor),
        count=int(count),
        layers=chosen_layers,
        greedy_order=[row["layer"] for row in trace],
        eligibility=sorted(layer_labels[i] for i in eligible),
        eligibility_rule=dict(auroc_margin=float(auroc_margin), accuracy_margin=float(accuracy_margin)),
        eligible_layers=sorted(layer_labels[i] for i in eligible),
        fallback_layers=sorted(layer_labels[i] for i in fallback_indices),
        oracle_accuracy=float(oracle.mean()),
        trace=trace,
    )


def assert_capacity_matched_initialization(
    heads_or_states: Mapping[str, nn.Module | Mapping[str, torch.Tensor]],
    arms: Sequence[str] | None = None,
) -> dict:
    """Assert matched arms have identical parameter shapes, counts, and values."""
    arms = tuple(heads_or_states) if arms is None else tuple(arms)
    if len(arms) < 2:
        raise ValueError("At least two heads are required for a capacity match")
    states = {}
    for arm in arms:
        if arm not in heads_or_states:
            raise ValueError(f"Missing capacity-matched arm: {arm}")
        value = heads_or_states[arm]
        states[arm] = value.state_dict() if isinstance(value, nn.Module) else value
    reference = states[arms[0]]
    names = list(reference)
    count = sum(torch.as_tensor(value).numel() for value in reference.values())
    digest = state_digest(reference)
    for arm in arms[1:]:
        state = states[arm]
        if sum(torch.as_tensor(value).numel() for value in state.values()) != count:
            raise ValueError(f"{arm} capacity differs from {arms[0]}")
        if list(state) != names or any(
            torch.as_tensor(state[name]).shape != torch.as_tensor(reference[name]).shape for name in names
        ):
            raise ValueError(f"{arm} capacity/state layout differs from {arms[0]}")
        if state_digest(state) != digest:
            raise ValueError(f"{arm} initial parameters differ from {arms[0]}")
    return dict(arms=list(arms), parameters=int(count), initialization_sha256=digest)


def _named_hashes(paths: Mapping[str, Path | str], kind: str) -> dict[str, str]:
    result = {}
    for name, item in paths.items():
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind} selection: {path}")
        result[str(name)] = sha256(path)
    return result


def create_test_barrier(
    output: Path | None = None,
    required_paths: Sequence[Path | str] | None = None,
    identity: Mapping[str, Any] | None = None,
    *,
    protocol_path: Path | None = None,
    probe_selection_paths: Mapping[str, Path | str] | None = None,
    head_selection_paths: Mapping[str, Path | str] | None = None,
    test_dataset_path: Path | None = None,
) -> dict:
    """Lock validation-only artifacts before any diagnostic test access."""
    # Pure functional form used by protocol tests and external orchestration.
    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        test_dataset_path = Path(test_dataset_path) if test_dataset_path is not None else None
        if not protocol_path.is_file() or test_dataset_path is None or not test_dataset_path.is_file():
            raise FileNotFoundError("Missing protocol or test dataset for barrier")
        return dict(
            locked=True,
            protocol_sha256=sha256(protocol_path),
            probe_selection_sha256=_named_hashes(probe_selection_paths or {}, "probe"),
            head_selection_sha256=_named_hashes(head_selection_paths or {}, "head"),
            test_dataset_sha256=sha256(test_dataset_path),
        )
    if output is None or required_paths is None:
        raise TypeError("output and required_paths are required for a persisted barrier")
    output = Path(output).resolve()
    records = {}
    for item in required_paths:
        path = Path(item)
        path = path if path.is_absolute() else output / path
        if not path.is_file():
            raise FileNotFoundError(f"Cannot lock test barrier; missing {path}")
        try:
            label = str(path.resolve().relative_to(output))
        except ValueError:
            label = str(path.resolve())
        records[label] = sha256(path)
    value = dict(
        state="locked",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        identity=dict(identity or {}),
        artifacts=records,
        rule="All checkpoints, thresholds, and selected layers were fixed using train/validation before test access.",
    )
    path = output / "test_barrier.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("identity") != value["identity"] or existing.get("artifacts") != records:
            raise ValueError("Existing test barrier has a different identity or artifact hashes")
        return existing
    dump(path, value)
    return value


def verify_test_barrier(
    output: Path | Mapping[str, Any],
    required_paths: Sequence[Path | str] | None = None,
    identity: Mapping[str, Any] | None = None,
    *,
    protocol_path: Path | None = None,
    probe_selection_paths: Mapping[str, Path | str] | None = None,
    head_selection_paths: Mapping[str, Path | str] | None = None,
    test_dataset_path: Path | None = None,
) -> dict:
    # Verify an in-memory functional barrier without creating any file.
    if isinstance(output, Mapping):
        barrier = dict(output)
        if barrier.get("locked") is not True:
            raise ValueError("Test barrier is not locked")
        expected = create_test_barrier(
            protocol_path=protocol_path,
            probe_selection_paths=probe_selection_paths,
            head_selection_paths=head_selection_paths,
            test_dataset_path=test_dataset_path,
        )
        if barrier.get("protocol_sha256") != expected["protocol_sha256"]:
            raise ValueError("Locked protocol hash changed")
        if set(barrier.get("probe_selection_sha256", {})) != set(expected["probe_selection_sha256"]):
            raise ValueError("Probe selection set is missing or changed")
        if barrier.get("probe_selection_sha256") != expected["probe_selection_sha256"]:
            raise ValueError("Locked probe selection hash changed")
        if set(barrier.get("head_selection_sha256", {})) != set(expected["head_selection_sha256"]):
            raise ValueError("Head selection set is missing or changed")
        if barrier.get("head_selection_sha256") != expected["head_selection_sha256"]:
            raise ValueError("Locked head selection hash changed")
        if barrier.get("test_dataset_sha256") != expected["test_dataset_sha256"]:
            raise ValueError("Locked test dataset hash changed")
        return True
    output = Path(output).resolve()
    path = output / "test_barrier.json"
    if not path.is_file():
        raise FileNotFoundError("Test barrier is not locked")
    value = json.loads(path.read_text())
    if value.get("state") != "locked":
        raise ValueError("Invalid test barrier state")
    if identity is not None and value.get("identity") != dict(identity):
        raise ValueError("Test barrier identity changed")
    records = value.get("artifacts", {})
    if required_paths is not None:
        expected_labels = set()
        for item in required_paths:
            candidate = Path(item)
            candidate = candidate if candidate.is_absolute() else output / candidate
            try:
                expected_labels.add(str(candidate.resolve().relative_to(output)))
            except ValueError:
                expected_labels.add(str(candidate.resolve()))
        if expected_labels != set(records):
            raise ValueError("Test barrier artifact set changed")
    for label, expected in records.items():
        candidate = Path(label)
        candidate = candidate if candidate.is_absolute() else output / candidate
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"Locked artifact changed after validation selection: {candidate}")
    return value


class EqualCapacityLayerProbes(nn.Module):
    """One independent, identically initialized linear probe per layer."""

    def __init__(self, hidden_size: int, layer_count: int = 32, seed: int = 42):
        super().__init__()
        if hidden_size < 1 or layer_count < 1:
            raise ValueError("Positive dimensions required")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bound = 1 / math.sqrt(hidden_size)
        initial = torch.empty(hidden_size).uniform_(-bound, bound, generator=generator)
        self.weight = nn.Parameter(initial.repeat(layer_count, 1))
        self.bias = nn.Parameter(torch.empty(1).uniform_(-bound, bound, generator=generator).repeat(layer_count))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != self.weight.shape:
            raise ValueError("Expected [batch,layer,hidden] features")
        normalized = F.layer_norm(features.float(), (features.shape[-1],))
        return torch.einsum("bld,ld->bl", normalized, self.weight) + self.bias


def _source_protocol(source_study: Path) -> dict:
    from scripts.hanguard.repaired_study import json_protocol

    protocol, _ = json_protocol(Path(source_study))
    return protocol


def initialize(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_study = Path(args.source_study).resolve()
    adapter = Path(args.adapter).resolve()
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    positive = {
        "token_budget": args.token_budget,
        "max_micro": args.max_micro,
        "pad_multiple": args.pad_multiple,
        "probe_epochs": args.probe_epochs,
        "probe_batch_size": args.probe_batch_size,
        "probe_lr": args.probe_lr,
        "head_epochs": args.head_epochs,
        "head_batch_size": args.head_batch_size,
        "head_lr": args.head_lr,
        "head_width": args.head_width,
        "repetitions": args.repetitions,
    }
    if any(value <= 0 for value in positive.values()) or args.weight_decay < 0:
        raise ValueError(f"Positive experiment settings required: {positive}")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Head seeds must be nonempty and unique")
    source = _source_protocol(source_study)
    model_path = Path(source["model_path"]).resolve()
    value = dict(
        version=1,
        source_study=str(source_study),
        source_protocol_sha256=sha256(source_study / "protocol.json"),
        source_arms_sha256=sha256(source_study / "arms.json"),
        data_dir=str(Path(source["data_dir"]).resolve()),
        model_path=str(model_path),
        model_artifacts=_model_artifact_identity(model_path),
        data_sha256=source["data_sha256"],
        rows=source["rows"],
        model_config_sha256=source.get("model_config_sha256"),
        tokenizer_sha256=source.get("tokenizer_sha256"),
        max_tokens=int(source["max_tokens"]),
        pad_multiple=int(args.pad_multiple),
        adapter_path=str(adapter),
        adapter_sha256=sha256(adapter),
        adapter_source_arm=adapter.parent.name,
        lora_rank=int(source["lora_rank"]),
        lora_alpha=int(source["lora_alpha"]),
        lora_dropout=float(source["lora_dropout"]),
        layers=list(ALL_LAYERS),
        fixed_fusion_layers=list(FIXED_FUSION_LAYERS),
        hidden_size=None,
        extraction=dict(token_budget=int(args.token_budget), max_micro=int(args.max_micro)),
        probes=dict(
            seed=int(args.probe_seed),
            epochs=int(args.probe_epochs),
            batch_size=int(args.probe_batch_size),
            lr=float(args.probe_lr),
            weight_decay=float(args.weight_decay),
            checkpoint_selection="minimum validation BCE independently per layer",
            threshold_selection="maximum validation accuracy independently per layer",
        ),
        heads=dict(
            seeds=list(map(int, args.seeds)),
            epochs=int(args.head_epochs),
            batch_size=int(args.head_batch_size),
            lr=float(args.head_lr),
            weight_decay=float(args.weight_decay),
            width=int(args.head_width),
            checkpoint_selection="minimum validation BCE independently per arm and seed",
            threshold_selection="maximum validation accuracy independently per arm and seed",
        ),
        selection=dict(
            split="validation",
            count=4,
            anchor=32,
            auroc_margin=float(args.selection_auroc_margin),
            accuracy_margin=float(args.selection_accuracy_margin),
        ),
        analysis=dict(repetitions=int(args.repetitions), seed=int(args.analysis_seed)),
        code_sha256=_code_identity(),
        limitations=[
            "The frozen adapter was trained jointly with the E03 final-layer head; this is a conditional representation diagnostic.",
            "Head seeds do not represent independent LoRA training seeds.",
            "The existing split identities have already been explored and are not a new blind test.",
            "Layer-wise matrices and paired p-values are exploratory and are not adjusted for multiple comparisons.",
        ],
    )
    path = output / "protocol.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != value:
            raise ValueError("Requested settings differ from the existing diagnostic protocol")
    else:
        dump(path, value)
    return value


def load_protocol(output: Path) -> dict:
    path = Path(output) / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run init first: {path}")
    value = json.loads(path.read_text())
    if value.get("adapter_sha256") != sha256(Path(value["adapter_path"])):
        raise ValueError("Frozen adapter artifact changed")
    if value.get("code_sha256") != _code_identity():
        raise ValueError("Diagnostic implementation changed after protocol registration")
    if value.get("model_artifacts") != _model_artifact_identity(Path(value["model_path"])):
        raise ValueError("Base model weight index or shard sizes changed")
    return value


def _identity(output: Path, protocol: Mapping[str, Any]) -> dict:
    return dict(
        protocol_sha256=sha256(Path(output) / "protocol.json"),
        adapter_sha256=protocol["adapter_sha256"],
        train_sha256=protocol["data_sha256"]["train"],
        validation_sha256=protocol["data_sha256"]["validation"],
    )


def _load_frozen_model(protocol: dict):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoTokenizer
    from transformers.utils.logging import disable_progress_bar
    from hanguard_model import load_base_model
    from scripts.hanguard.repaired_study import TARGET

    disable_progress_bar()
    torch.set_num_threads(4)
    for filename, key in (("config.json", "model_config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        expected = protocol.get(key)
        if expected and sha256(Path(protocol["model_path"]) / filename) != expected:
            raise ValueError(f"Base model artifact changed: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(protocol["model_path"], local_files_only=True)
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.padding_side = "right"
    model = load_base_model(
        protocol["model_path"], local_files_only=True, device_map={"": 0}, attn_implementation="sdpa"
    )
    model.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=protocol["lora_rank"],
            lora_alpha=protocol["lora_alpha"],
            lora_dropout=protocol["lora_dropout"],
            target_modules=TARGET,
            bias="none",
        ),
    )
    state = torch.load(protocol["adapter_path"], map_location="cpu", weights_only=True)
    incompatible = set_peft_model_state_dict(model, state)
    validate_lora_load(state, incompatible)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    root = model.get_base_model()
    backbone = root.model.language_model
    if len(backbone.layers) != len(ALL_LAYERS):
        raise ValueError(f"Expected 32 blocks, found {len(backbone.layers)}")
    return model, backbone, tokenizer


class LastTokenBlockCapture:
    """Select [B,D] inside each block hook; never retain [B,T,D] outputs."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.last_indices: torch.Tensor | None = None
        self.active = False
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            module.register_forward_hook(self._hook(layer))
            for layer, module in enumerate(backbone.layers, start=1)
        ]

    def _hook(self, layer: int):
        def capture(module, inputs, output):
            if not self.active:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if self.last_indices is None:
                raise RuntimeError("Missing last-token indices")
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            # This indexing is deliberately inside the hook: retained tensors
            # are [B,D], never the block's [B,T,D] output.
            self.values[layer] = hidden[rows, self.last_indices].detach().to(torch.float16)

        return capture

    @torch.inference_mode()
    def __call__(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.values.clear()
        mask = batch["attention_mask"].bool()
        self.last_indices = last_true_indices(mask)
        self.active = True
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0),
                    use_cache=False,
                )
        finally:
            self.active = False
            self.last_indices = None
        if set(self.values) != set(ALL_LAYERS):
            raise RuntimeError(f"Captured layers differ: {sorted(self.values)}")
        stacked = torch.stack([self.values[layer] for layer in ALL_LAYERS], dim=1).cpu()
        self.values.clear()
        return stacked

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


def _load_split_for_extraction(split: str, tokenizer, protocol: dict) -> dict:
    from scripts.hanguard.repaired_study import load_split

    source_protocol = _source_protocol(Path(protocol["source_study"]))
    source_protocol.update(
        data_dir=protocol["data_dir"],
        max_tokens=protocol["max_tokens"],
        data_sha256=protocol["data_sha256"],
        rows=protocol["rows"],
    )
    data = load_split(split, tokenizer, source_protocol)
    frame = pd.read_parquet(Path(protocol["data_dir"]) / f"{split}.parquet").sort_values("base_id").reset_index(drop=True)
    if frame.base_id.tolist() != data["meta"].base_id.tolist():
        raise AssertionError("Metadata order differs from tokenized order")
    recorded_lengths = frame.prompt_tokens.to_numpy(dtype=np.int32)
    if not np.array_equal(recorded_lengths, data["lengths"]):
        raise ValueError("Stored prompt_tokens differ from the no-truncation tokenizer lengths")
    columns = ["base_id", "group_id", "source", "category_id", "text_form", "prompt_tokens"]
    metadata = frame[columns].copy()
    if metadata.group_id.isna().any():
        raise ValueError(f"{split} contains missing group_id values")
    metadata["group_id"] = metadata.group_id.astype(str)
    metadata["label"] = frame.prompt_harm_label.eq("harmful").astype(np.int8)
    data["metadata"] = metadata
    return data


def _make_batch(data: dict, indices: np.ndarray, tokenizer, protocol: dict) -> dict:
    batch = tokenizer.pad(
        [
            {"input_ids": data["encoded"][int(index)], "attention_mask": [1] * len(data["encoded"][int(index)])}
            for index in indices
        ],
        padding=True,
        pad_to_multiple_of=protocol["pad_multiple"],
        return_tensors="pt",
    )
    expected = torch.as_tensor(data["lengths"][indices], dtype=torch.long)
    if not torch.equal(batch["attention_mask"].sum(1), expected):
        raise AssertionError("Tokenizer changed input lengths")
    return batch.to("cuda")


def _cache_paths(output: Path, split: str) -> tuple[Path, Path, Path, Path]:
    directory = Path(output) / "features"
    return (
        directory / f"{split}.npy",
        directory / f"{split}.npy.part",
        directory / f"{split}.progress.json",
        directory / f"{split}.manifest.json",
    )


def _validate_available_split_identities(output: Path) -> None:
    frames = {}
    for split in ("train", "validation", "test"):
        path = Path(output) / "features" / f"{split}.metadata.parquet"
        manifest = Path(output) / "features" / f"{split}.manifest.json"
        if path.is_file() and manifest.is_file() and json.loads(manifest.read_text()).get("complete") is True:
            frame = pd.read_parquet(path, columns=["base_id", "group_id"])
            if frame.group_id.isna().any():
                raise ValueError(f"{split} contains missing group_id")
            frames[split] = frame
    for first, second in itertools.combinations(frames, 2):
        base_overlap = set(frames[first].base_id) & set(frames[second].base_id)
        group_overlap = set(frames[first].group_id.astype(str)) & set(frames[second].group_id.astype(str))
        if base_overlap or group_overlap:
            raise ValueError(
                f"Split leakage between {first}/{second}: {len(base_overlap)} base_id and {len(group_overlap)} group_id"
            )


def extract_splits(args: argparse.Namespace, splits: Sequence[str]) -> None:
    from scripts.hanguard.repaired_study import microbatches

    protocol = load_protocol(args.output)
    for split in splits:
        if split not in ("train", "validation", "test"):
            raise ValueError(split)
        if split == "test":
            verify_test_barrier(args.output, identity=_identity(args.output, protocol))
    pending = []
    for split in splits:
        final, temporary, progress_path, manifest_path = _cache_paths(args.output, split)
        metadata_path = final.parent / f"{split}.metadata.parquet"
        completed = (final.is_file(), manifest_path.is_file(), metadata_path.is_file())
        if all(completed):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("complete") is not True:
                raise ValueError(f"{split} feature manifest is not complete")
            continue
        if completed == (False, False, True) and temporary.is_file() and progress_path.is_file():
            # A crash after writing metadata but before the final cache rename
            # can safely resume from the last flushed feature group.
            metadata_path.unlink()
        elif any(completed):
            raise ValueError(f"Incomplete final {split} cache artifacts; remove or repair them before resuming")
        pending.append(split)
    if not pending:
        _validate_available_split_identities(Path(args.output))
        return
    model, backbone, tokenizer = _load_frozen_model(protocol)
    hidden_size = int(model.config.text_config.hidden_size)
    capture = LastTokenBlockCapture(backbone)
    try:
        for split in pending:
            data = _load_split_for_extraction(split, tokenizer, protocol)
            final, temporary, progress_path, manifest_path = _cache_paths(args.output, split)
            final.parent.mkdir(parents=True, exist_ok=True)
            shape = (len(data["lengths"]), len(ALL_LAYERS), hidden_size)
            identity = dict(
                split=split,
                rows=shape[0],
                layers=shape[1],
                hidden_size=shape[2],
                dtype="float16",
                data_sha256=protocol["data_sha256"][split],
                adapter_sha256=protocol["adapter_sha256"],
                protocol_sha256=sha256(Path(args.output) / "protocol.json"),
            )
            order = np.argsort(data["lengths"], kind="stable")
            groups = microbatches(
                order,
                data["lengths"],
                protocol["extraction"]["token_budget"],
                protocol["extraction"]["max_micro"],
                protocol["pad_multiple"],
            )
            start_group = 0
            if temporary.exists() or progress_path.exists():
                if not temporary.exists() or not progress_path.exists():
                    raise ValueError("Incomplete extraction has only one of cache/progress")
                progress = json.loads(progress_path.read_text())
                if progress.get("identity") != identity:
                    raise ValueError("Extraction resume identity changed")
                start_group = int(progress["next_group"])
                memmap = np.lib.format.open_memmap(temporary, mode="r+", dtype=np.float16, shape=shape)
            else:
                memmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
            began = time.monotonic()
            safe_group = start_group
            for group_number in range(start_group, len(groups)):
                indices = groups[group_number]
                values = capture(_make_batch(data, indices, tokenizer, protocol)).numpy()
                if values.shape != (len(indices), len(ALL_LAYERS), hidden_size) or not np.isfinite(values).all():
                    raise FloatingPointError("Invalid captured layer features")
                memmap[indices] = values
                safe_group = group_number + 1
                if safe_group % max(1, args.status_every) == 0 or safe_group == len(groups):
                    memmap.flush()
                    dump(
                        progress_path,
                        dict(
                            identity=identity,
                            next_group=safe_group,
                            total_groups=len(groups),
                            rows=sum(len(x) for x in groups[:safe_group]),
                            seconds=time.monotonic() - began,
                            updated=time.time(),
                        ),
                    )
                    print(
                        json.dumps(
                            dict(state="extract", split=split, groups=safe_group, total_groups=len(groups)),
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                del values
            memmap.flush()
            del memmap
            metadata_path = final.parent / f"{split}.metadata.parquet"
            data["metadata"].to_parquet(metadata_path, index=False)
            temporary.replace(final)
            progress_path.unlink(missing_ok=True)
            ordered_hash = ordered_base_id_sha256(data["metadata"].base_id.tolist())
            validation = validate_feature_cache(
                final, shape, np.float16, ordered_hash, data["metadata"].base_id.tolist()
            )
            manifest = dict(
                **identity,
                complete=True,
                path=str(final.resolve()),
                bytes=validation["bytes"],
                sha256=sha256(final),
                metadata_path=str(metadata_path.resolve()),
                metadata_sha256=sha256(metadata_path),
                ordered_base_id_sha256=ordered_hash,
                finite_rows_checked=validation["finite_rows_checked"],
                order="base_id ascending; feature writes restored to this row order",
                extraction_order="stable ascending tokenizer length",
                representation="raw block output at final valid token, selected inside each forward hook",
                truncation=False,
            )
            dump(manifest_path, manifest)
            _validate_available_split_identities(Path(args.output))
    finally:
        capture.close()
        del model, backbone
        torch.cuda.empty_cache()


def _load_cache(output: Path, split: str, protocol: Mapping[str, Any]) -> tuple[np.memmap, pd.DataFrame, dict]:
    final, _, _, manifest_path = _cache_paths(output, split)
    metadata_path = final.parent / f"{split}.metadata.parquet"
    if not final.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {split} cache; run extract")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError(f"{split} feature cache lacks a complete sentinel")
    expected = dict(
        data_sha256=protocol["data_sha256"][split],
        adapter_sha256=protocol["adapter_sha256"],
        protocol_sha256=sha256(Path(output) / "protocol.json"),
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{split} cache identity changed: {key}")
    if manifest.get("bytes") != final.stat().st_size:
        raise ValueError(f"{split} cache byte size differs from its manifest")
    if manifest.get("sha256") != sha256(final):
        raise ValueError(f"{split} cache content changed")
    if manifest.get("metadata_sha256") != sha256(metadata_path):
        raise ValueError(f"{split} metadata changed")
    metadata = pd.read_parquet(metadata_path)
    required = {"base_id", "group_id", "source", "category_id", "text_form", "prompt_tokens", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"{split} metadata columns are incomplete")
    if metadata.base_id.isna().any() or metadata.base_id.duplicated().any():
        raise ValueError(f"{split} metadata base_id values must be unique and nonempty")
    if metadata.base_id.tolist() != sorted(metadata.base_id.tolist()):
        raise ValueError(f"{split} metadata is not in base_id ascending order")
    if metadata.group_id.isna().any() or (metadata.group_id.astype(str).str.len() == 0).any():
        raise ValueError(f"{split} metadata contains missing group_id")
    if not metadata.label.isin((0, 1)).all():
        raise ValueError(f"{split} metadata labels are not binary")
    expected_shape = (len(metadata), len(ALL_LAYERS), int(manifest["hidden_size"]))
    validated = validate_feature_cache(
        final,
        expected_shape,
        np.float16,
        manifest["ordered_base_id_sha256"],
        metadata.base_id.tolist(),
    )
    if validated["bytes"] != manifest["bytes"] or validated["finite_rows_checked"] != manifest["rows"]:
        raise ValueError(f"{split} cache completeness validation differs from its manifest")
    features = np.load(final, mmap_mode="r", allow_pickle=False)
    return features, metadata, manifest


def _feature_batch(cache: np.memmap, indices: np.ndarray, layer_indices: Sequence[int] | None = None) -> torch.Tensor:
    if layer_indices is None:
        array = np.asarray(cache[indices], dtype=np.float32)
    else:
        array = np.stack([np.asarray(cache[indices, index, :], dtype=np.float32) for index in layer_indices], axis=1)
    return torch.from_numpy(array).cuda(non_blocking=True)


@torch.no_grad()
def _predict_probes(model: EqualCapacityLayerProbes, cache: np.memmap, batch_size: int) -> np.ndarray:
    model.eval()
    output = np.empty((len(cache), cache.shape[1]), dtype=np.float32)
    for start in range(0, len(cache), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache)))
        output[indices] = model(_feature_batch(cache, indices)).float().cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite probe logits")
    return output


def train_probes(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    protocol = load_protocol(output)
    directory = output / "probes"
    selection_path = directory / "selection.json"
    if selection_path.is_file():
        return json.loads(selection_path.read_text())
    train, train_meta, manifest = _load_cache(output, "train", protocol)
    validation, validation_meta, _ = _load_cache(output, "validation", protocol)
    hidden_size = train.shape[-1]
    settings = protocol["probes"]
    set_seed(settings["seed"])
    model = EqualCapacityLayerProbes(hidden_size, len(ALL_LAYERS), settings["seed"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    y_train = torch.as_tensor(train_meta.label.to_numpy(), dtype=torch.float32, device="cuda")
    y_validation = validation_meta.label.to_numpy(dtype=np.int64)
    best_bce = np.full(len(ALL_LAYERS), np.inf)
    best_epoch = np.full(len(ALL_LAYERS), -1, dtype=np.int64)
    best_weight = model.weight.detach().cpu().clone()
    best_bias = model.bias.detach().cpu().clone()
    history = []
    for epoch in range(settings["epochs"]):
        model.train()
        order = np.random.default_rng(settings["seed"] * 1009 + epoch).permutation(len(train))
        losses = np.zeros(len(ALL_LAYERS), dtype=np.float64)
        rows = 0
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start : start + settings["batch_size"]]
            logits = model(_feature_batch(train, indices))
            targets = y_train[indices, None].expand_as(logits)
            by_layer = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(0)
            loss = by_layer.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses += by_layer.detach().cpu().numpy() * len(indices)
            rows += len(indices)
        validation_logits = _predict_probes(model, validation, settings["batch_size"])
        target = torch.as_tensor(validation_meta.label.to_numpy(), dtype=torch.float32)[:, None]
        val_bce = F.binary_cross_entropy_with_logits(
            torch.from_numpy(validation_logits), target.expand_as(torch.from_numpy(validation_logits)), reduction="none"
        ).mean(0).numpy()
        improved = val_bce < best_bce
        best_bce[improved] = val_bce[improved]
        best_epoch[improved] = epoch + 1
        best_weight[improved] = model.weight.detach().cpu()[improved]
        best_bias[improved] = model.bias.detach().cpu()[improved]
        record = dict(
            epoch=epoch + 1,
            train_bce=(losses / rows).tolist(),
            validation_bce=val_bce.tolist(),
            improved_layers=[int(i + 1) for i in np.flatnonzero(improved)],
        )
        history.append(record)
        dump(directory / "history.json", history)
        print(json.dumps(dict(state="probes", epoch=epoch + 1, epochs=settings["epochs"])), flush=True)
    model.weight.data.copy_(best_weight.to(model.weight.device))
    model.bias.data.copy_(best_bias.to(model.bias.device))
    validation_logits = _predict_probes(model, validation, settings["batch_size"])
    probability = sigmoid(validation_logits)
    thresholds = np.asarray([accuracy_threshold(y_validation, probability[:, i]) for i in range(len(ALL_LAYERS))])
    layer_metrics = [binary_metrics(y_validation, probability[:, i], thresholds[i]) for i in range(len(ALL_LAYERS))]
    layer_by_source = [
        _by_source(validation_meta, probability[:, i], thresholds[i]) for i in range(len(ALL_LAYERS))
    ]
    checkpoint = dict(
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        hidden_size=int(hidden_size),
        layers=list(ALL_LAYERS),
        best_epoch=best_epoch.tolist(),
        best_validation_bce=best_bce.tolist(),
        protocol_sha256=sha256(output / "protocol.json"),
    )
    checkpoint_path = directory / "probe.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    frame = validation_meta.copy()
    for index, layer in enumerate(ALL_LAYERS):
        frame[f"probability_layer_{layer:02d}"] = probability[:, index]
        frame[f"prediction_layer_{layer:02d}"] = (probability[:, index] >= thresholds[index]).astype(np.int8)
    predictions_path = directory / "validation_predictions.parquet"
    frame.to_parquet(predictions_path, index=False)
    selection = dict(
        split="validation",
        layers=list(ALL_LAYERS),
        equal_parameters_per_layer=int(hidden_size + 1),
        identical_initialization=True,
        checkpoint_selection="minimum validation BCE per layer over a common fixed epoch budget",
        threshold_selection="maximum validation accuracy; ties nearest 0.5",
        best_epoch=best_epoch.tolist(),
        validation_bce=best_bce.tolist(),
        thresholds=thresholds.tolist(),
        metrics=layer_metrics,
        by_source=layer_by_source,
        checkpoint_sha256=sha256(checkpoint_path),
        validation_predictions_sha256=sha256(predictions_path),
        train_cache_sha256=manifest["sha256"],
    )
    dump(selection_path, selection)
    return selection


def select_layers(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    protocol = load_protocol(output)
    path = output / "selected_layers.json"
    if path.is_file():
        return json.loads(path.read_text())
    selection = json.loads((output / "probes/selection.json").read_text())
    predictions = pd.read_parquet(output / "probes/validation_predictions.parquet")
    columns = [f"probability_layer_{layer:02d}" for layer in ALL_LAYERS]
    result = select_complementary_layers(
        predictions.label.to_numpy(),
        predictions[columns].to_numpy(),
        selection["thresholds"],
        ALL_LAYERS,
        split="validation",
        count=protocol["selection"]["count"],
        anchor=protocol["selection"]["anchor"],
        auroc_margin=protocol["selection"]["auroc_margin"],
        accuracy_margin=protocol["selection"]["accuracy_margin"],
    )
    result.update(
        probe_selection_sha256=sha256(output / "probes/selection.json"),
        validation_predictions_sha256=sha256(output / "probes/validation_predictions.parquet"),
    )
    dump(path, result)
    return result


def _head_spec(arm: str, selected: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    if arm == "last":
        return "last", (32,)
    if arm == "repeat_last":
        return "repeat_last", FIXED_FUSION_LAYERS
    if arm == "fusion_fixed":
        return "fusion", FIXED_FUSION_LAYERS
    if arm == "fusion_selected":
        return "fusion", tuple(selected)
    raise ValueError(arm)


def _build_heads(hidden_size: int, width: int, selected: Sequence[int], seed: int) -> tuple[dict, dict]:
    from scripts.hanguard.repaired_heads import ExperimentHead

    heads = {}
    for arm in ALL_ARMS:
        # Reset before every arm.  The three matched models therefore receive
        # byte-identical parameter initialization despite different inputs.
        torch.manual_seed(seed)
        mode, layers = _head_spec(arm, selected)
        heads[arm] = ExperimentHead(hidden_size, mode=mode, readout="mlp", width=width, layers=layers).cuda()
    matched = assert_capacity_matched_initialization({arm: heads[arm] for arm in MATCHED_ARMS})
    return heads, matched


def _head_logits(head: nn.Module, raw: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    pieces = tuple(raw[:, position, :].unsqueeze(1) for position in positions)
    mask = torch.ones((len(raw), 1), dtype=torch.bool, device=raw.device)
    return head({"layers": pieces, "mask": mask})["logits"]


@torch.no_grad()
def _predict_heads(
    heads: Mapping[str, nn.Module],
    cache: np.memmap,
    selected: Sequence[int],
    batch_size: int,
) -> dict[str, np.ndarray]:
    union = sorted(set(FIXED_FUSION_LAYERS) | set(selected))
    zero_indices = [layer - 1 for layer in union]
    position = {layer: i for i, layer in enumerate(union)}
    outputs = {arm: np.empty(len(cache), dtype=np.float32) for arm in heads}
    for head in heads.values():
        head.eval()
    for start in range(0, len(cache), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache)))
        raw = _feature_batch(cache, indices, zero_indices)
        for arm, head in heads.items():
            _, layers = _head_spec(arm, selected)
            captured = (32,) if arm == "repeat_last" else layers
            outputs[arm][indices] = _head_logits(head, raw, [position[layer] for layer in captured]).cpu().numpy()
    return outputs


def train_heads(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    protocol = load_protocol(output)
    complete = output / "heads/selection.json"
    if complete.is_file():
        return json.loads(complete.read_text())
    selected = select_layers(args)["layers"]
    train, train_meta, _ = _load_cache(output, "train", protocol)
    validation, validation_meta, _ = _load_cache(output, "validation", protocol)
    settings = protocol["heads"]
    y_train = torch.as_tensor(train_meta.label.to_numpy(), dtype=torch.float32, device="cuda")
    y_validation = validation_meta.label.to_numpy(dtype=np.int64)
    entries = []
    for seed in settings["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        if (directory / "selection.json").is_file():
            entries.append(json.loads((directory / "selection.json").read_text()))
            continue
        heads, matched = _build_heads(train.shape[-1], settings["width"], selected, seed)
        initial_hashes = {arm: state_digest(head) for arm, head in heads.items()}
        optimizers = {
            arm: torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
            for arm, head in heads.items()
        }
        best_bce = {arm: float("inf") for arm in ALL_ARMS}
        best_epoch = {arm: None for arm in ALL_ARMS}
        best_state = {arm: None for arm in ALL_ARMS}
        history = []
        union = sorted(set(FIXED_FUSION_LAYERS) | set(selected))
        union_indices = [layer - 1 for layer in union]
        position = {layer: i for i, layer in enumerate(union)}
        for epoch in range(settings["epochs"]):
            order = np.random.default_rng(seed * 1009 + epoch).permutation(len(train))
            totals = {arm: 0.0 for arm in ALL_ARMS}
            rows = 0
            for step, start in enumerate(range(0, len(order), settings["batch_size"])):
                indices = order[start : start + settings["batch_size"]]
                raw = _feature_batch(train, indices, union_indices)
                losses = []
                for arm, head in heads.items():
                    head.train()
                    # Paired dropout stream for every arm at this batch.
                    torch.manual_seed((seed * 1000003 + epoch * 1009 + step) % (2**63 - 1))
                    _, layers = _head_spec(arm, selected)
                    captured = (32,) if arm == "repeat_last" else layers
                    logits = _head_logits(head, raw, [position[layer] for layer in captured])
                    loss = F.binary_cross_entropy_with_logits(logits.float(), y_train[indices])
                    losses.append(loss)
                    totals[arm] += float(loss.detach()) * len(indices)
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).sum().backward()
                for arm, optimizer in optimizers.items():
                    nn.utils.clip_grad_norm_(heads[arm].parameters(), 5.0, error_if_nonfinite=True)
                    optimizer.step()
                rows += len(indices)
            validation_logits = _predict_heads(heads, validation, selected, settings["batch_size"])
            record = dict(epoch=epoch + 1, train_bce={arm: totals[arm] / rows for arm in ALL_ARMS}, validation_bce={})
            targets = torch.as_tensor(y_validation, dtype=torch.float32)
            for arm in ALL_ARMS:
                value = float(F.binary_cross_entropy_with_logits(torch.from_numpy(validation_logits[arm]), targets))
                record["validation_bce"][arm] = value
                if value < best_bce[arm]:
                    best_bce[arm] = value
                    best_epoch[arm] = epoch + 1
                    best_state[arm] = {name: tensor.detach().cpu().clone() for name, tensor in heads[arm].state_dict().items()}
            history.append(record)
            dump(directory / "history.json", history)
            print(json.dumps(dict(state="heads", seed=seed, epoch=epoch + 1, epochs=settings["epochs"])), flush=True)
        for arm in ALL_ARMS:
            if best_state[arm] is None:
                raise RuntimeError(f"No finite validation checkpoint for {arm}")
            heads[arm].load_state_dict(best_state[arm])
        validation_logits = _predict_heads(heads, validation, selected, settings["batch_size"])
        frame = validation_meta.copy()
        arm_records = {}
        checkpoint_hashes = {}
        for arm in ALL_ARMS:
            checkpoint_path = directory / f"{arm}.pt"
            atomic_torch_save(
                dict(
                    state_dict=best_state[arm],
                    arm=arm,
                    seed=seed,
                    selected_layers=list(selected),
                    best_epoch=best_epoch[arm],
                    best_validation_bce=best_bce[arm],
                    protocol_sha256=sha256(output / "protocol.json"),
                ),
                checkpoint_path,
            )
            probability = sigmoid(validation_logits[arm])
            threshold = accuracy_threshold(y_validation, probability)
            frame[f"probability_{arm}"] = probability
            frame[f"prediction_{arm}"] = (probability >= threshold).astype(np.int8)
            checkpoint_hashes[arm] = sha256(checkpoint_path)
            arm_records[arm] = dict(
                parameters=sum(parameter.numel() for parameter in heads[arm].parameters()),
                initial_state_sha256=initial_hashes[arm],
                best_epoch=best_epoch[arm],
                validation_bce=best_bce[arm],
                threshold=threshold,
                validation_metrics=binary_metrics(y_validation, probability, threshold),
                checkpoint_sha256=checkpoint_hashes[arm],
            )
        predictions_path = directory / "validation_predictions.parquet"
        frame.to_parquet(predictions_path, index=False)
        selection = dict(
            split="validation",
            seed=seed,
            selected_layers=list(selected),
            fixed_layers=list(FIXED_FUSION_LAYERS),
            common_epoch_budget=settings["epochs"],
            common_batch_order=True,
            paired_dropout_stream=True,
            matched_initialization=matched,
            arms=arm_records,
            validation_predictions_sha256=sha256(predictions_path),
        )
        dump(directory / "selection.json", selection)
        entries.append(selection)
        del heads, optimizers
        torch.cuda.empty_cache()
    summary = dict(
        seeds=settings["seeds"],
        selected_layers=list(selected),
        matched_arms=list(MATCHED_ARMS),
        entries=entries,
        statement="repeat_last, fusion_fixed, and fusion_selected have identical parameters and per-seed initial values",
    )
    dump(complete, summary)
    return summary


def _barrier_paths(output: Path, protocol: Mapping[str, Any]) -> list[Path]:
    output = Path(output).resolve()
    paths = [
        output / "protocol.json",
        output / "features/train.manifest.json",
        output / "features/validation.manifest.json",
        output / "probes/probe.pt",
        output / "probes/selection.json",
        output / "probes/validation_predictions.parquet",
        output / "selected_layers.json",
        output / "heads/selection.json",
        Path(protocol["data_dir"]) / "test.parquet",
    ]
    for seed in protocol["heads"]["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        paths.extend([directory / "selection.json", directory / "validation_predictions.parquet"])
        paths.extend(directory / f"{arm}.pt" for arm in ALL_ARMS)
    return paths


def lock_barrier(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    test_path = Path(protocol["data_dir"]) / "test.parquet"
    if sha256(test_path) != protocol["data_sha256"]["test"]:
        raise ValueError("Registered test dataset hash changed before barrier lock")
    return create_test_barrier(output, _barrier_paths(output, protocol), _identity(output, protocol))


def _load_head_checkpoint(
    checkpoint_path: Path, hidden_size: int, width: int, selected: Sequence[int]
) -> nn.Module:
    from scripts.hanguard.repaired_heads import ExperimentHead

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arm = checkpoint["arm"]
    mode, layers = _head_spec(arm, selected)
    head = ExperimentHead(hidden_size, mode=mode, readout="mlp", width=width, layers=layers).cuda()
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head


def _by_source(metadata: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    result = {}
    y = metadata.label.to_numpy(dtype=np.int64)
    for source in sorted(metadata.source.astype(str).unique()):
        mask = metadata.source.astype(str).to_numpy() == source
        result[source] = binary_metrics(y[mask], probabilities[mask], threshold)
    return result


def _transition(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict:
    a = candidate == y
    b = baseline == y
    return dict(
        both_correct=int((a & b).sum()),
        both_wrong=int((~a & ~b).sum()),
        candidate_fixes=int((a & ~b).sum()),
        candidate_harms=int((~a & b).sum()),
        net_corrections=int((a & ~b).sum() - (~a & b).sum()),
    )


def _seed_statistics(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(array.mean()),
        std=float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        minimum=float(array.min()),
        maximum=float(array.max()),
        seeds=int(len(array)),
    )


def _selected_set_oracle(
    labels: Sequence[int], probabilities: np.ndarray, thresholds: Sequence[float], selected_layers: Sequence[int]
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds)
    indices = np.asarray([int(layer) - 1 for layer in selected_layers], dtype=np.int64)
    correct = (probabilities[:, indices] >= thresholds[indices][None]) == labels[:, None]
    anchor_position = list(selected_layers).index(32)
    anchor_correct = correct[:, anchor_position]
    oracle_correct = correct.any(1)
    rescued = ~anchor_correct & oracle_correct
    return dict(
        layers=list(map(int, selected_layers)),
        anchor_layer=32,
        anchor_accuracy=float(anchor_correct.mean()),
        oracle_accuracy=float(oracle_correct.mean()),
        oracle_gain=float(oracle_correct.mean() - anchor_correct.mean()),
        anchor_errors=int((~anchor_correct).sum()),
        anchor_errors_rescued=int(rescued.sum()),
        rescue_fraction_of_anchor_errors=float(rescued.sum() / max(1, (~anchor_correct).sum())),
        remaining_shared_errors=int((~oracle_correct).sum()),
        note="Oracle is an error-complement upper bound, not a deployable selector.",
    )


def evaluate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    protocol = load_protocol(output)
    verify_test_barrier(output, _barrier_paths(output, protocol), _identity(output, protocol))
    result_path = output / "results.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    test, test_meta, test_manifest = _load_cache(output, "test", protocol)
    probe_selection = json.loads((output / "probes/selection.json").read_text())
    probe_checkpoint = torch.load(output / "probes/probe.pt", map_location="cpu", weights_only=True)
    probes = EqualCapacityLayerProbes(probe_checkpoint["hidden_size"], len(ALL_LAYERS)).cuda()
    probes.load_state_dict(probe_checkpoint["state_dict"])
    probe_logits = _predict_probes(probes, test, protocol["probes"]["batch_size"])
    probe_probability = sigmoid(probe_logits)
    thresholds = np.asarray(probe_selection["thresholds"])
    y = test_meta.label.to_numpy(dtype=np.int64)
    probe_metrics = [binary_metrics(y, probe_probability[:, i], thresholds[i]) for i in range(len(ALL_LAYERS))]
    validation_probe_frame = pd.read_parquet(output / "probes/validation_predictions.parquet")
    columns = [f"probability_layer_{layer:02d}" for layer in ALL_LAYERS]
    validation_complementarity = error_complementarity(
        validation_probe_frame.label.to_numpy(),
        validation_probe_frame[columns].to_numpy(),
        thresholds=thresholds,
        layers=ALL_LAYERS,
    )
    test_complementarity = error_complementarity(
        y, probe_probability, thresholds=thresholds, layers=ALL_LAYERS
    )
    dump(output / "analysis/layer_complementarity_validation.json", validation_complementarity)
    dump(output / "analysis/layer_complementarity_test.json", test_complementarity)
    probe_frame = test_meta.copy()
    for i, layer in enumerate(ALL_LAYERS):
        probe_frame[f"probability_layer_{layer:02d}"] = probe_probability[:, i]
        probe_frame[f"prediction_layer_{layer:02d}"] = (probe_probability[:, i] >= thresholds[i]).astype(np.int8)
    probe_test_path = output / "probes/test_predictions.parquet"
    probe_frame.to_parquet(probe_test_path, index=False)
    selected = json.loads((output / "selected_layers.json").read_text())["layers"]
    selected_set_oracle = dict(
        validation=_selected_set_oracle(
            validation_probe_frame.label.to_numpy(),
            validation_probe_frame[columns].to_numpy(),
            thresholds,
            selected,
        ),
        test=_selected_set_oracle(y, probe_probability, thresholds, selected),
    )
    head_entries = []
    head_predictions: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for seed in protocol["heads"]["seeds"]:
        directory = output / "heads" / f"seed_{seed}"
        selection = json.loads((directory / "selection.json").read_text())
        heads = {
            arm: _load_head_checkpoint(directory / f"{arm}.pt", test.shape[-1], protocol["heads"]["width"], selected)
            for arm in ALL_ARMS
        }
        logits = _predict_heads(heads, test, selected, protocol["heads"]["batch_size"])
        frame = test_meta.copy()
        for arm in ALL_ARMS:
            probability = sigmoid(logits[arm])
            threshold = selection["arms"][arm]["threshold"]
            prediction = probability >= threshold
            frame[f"probability_{arm}"] = probability
            frame[f"prediction_{arm}"] = prediction.astype(np.int8)
            head_predictions[(seed, arm)] = (probability, prediction)
            head_entries.append(
                dict(
                    seed=seed,
                    arm=arm,
                    parameters=selection["arms"][arm]["parameters"],
                    selected_layers=list(selected) if arm == "fusion_selected" else None,
                    threshold=threshold,
                    validation_metrics=selection["arms"][arm]["validation_metrics"],
                    test_metrics=binary_metrics(y, probability, threshold),
                    by_source=_by_source(test_meta, probability, threshold),
                )
            )
        frame.to_parquet(directory / "test_predictions.parquet", index=False)
        del heads
    transitions = []
    primary_comparisons = []
    groups = test_meta.group_id.fillna(test_meta.base_id).astype(str).to_numpy()
    repetitions = protocol["analysis"]["repetitions"]
    analysis_seed = protocol["analysis"]["seed"]
    for seed in protocol["heads"]["seeds"]:
        for candidate_arm, baseline_arm in itertools.combinations(ALL_ARMS, 2):
            _, candidate = head_predictions[(seed, candidate_arm)]
            _, baseline = head_predictions[(seed, baseline_arm)]
            transitions.append(
                dict(
                    seed=seed,
                    candidate=candidate_arm,
                    baseline=baseline_arm,
                    **_transition(y, candidate, baseline),
                    mcnemar_exact=mcnemar_exact(y, candidate, baseline),
                )
            )
        for candidate_arm in ("fusion_fixed", "fusion_selected"):
            _, candidate = head_predictions[(seed, candidate_arm)]
            _, baseline = head_predictions[(seed, "repeat_last")]
            primary_comparisons.append(
                dict(
                    seed=seed,
                    candidate=candidate_arm,
                    baseline="repeat_last",
                    transition=_transition(y, candidate, baseline),
                    mcnemar_exact=mcnemar_exact(y, candidate, baseline),
                    group_bootstrap=group_cluster_bootstrap(
                        labels=y,
                        probability_a=candidate.astype(np.float64),
                        probability_b=baseline.astype(np.float64),
                        groups=groups,
                        n_resamples=repetitions,
                        seed=analysis_seed + seed,
                    ),
                    group_permutation=group_permutation_test(
                        labels=y,
                        probability_a=candidate.astype(np.float64),
                        probability_b=baseline.astype(np.float64),
                        groups=groups,
                        n_resamples=repetitions,
                        seed=analysis_seed + seed,
                    ),
                )
            )
    head_seed_aggregates = []
    for arm in ALL_ARMS:
        rows = [row for row in head_entries if row["arm"] == arm]
        head_seed_aggregates.append(
            dict(
                arm=arm,
                seeds=[row["seed"] for row in rows],
                test_metrics={
                    metric: _seed_statistics([row["test_metrics"][metric] for row in rows])
                    for metric in ("accuracy", "f1", "auroc", "auprc", "bce", "brier")
                },
            )
        )
    matched_seed_aggregates = []
    for candidate in ("fusion_fixed", "fusion_selected"):
        rows = [row for row in primary_comparisons if row["candidate"] == candidate]
        matched_seed_aggregates.append(
            dict(
                candidate=candidate,
                baseline="repeat_last",
                seeds=[row["seed"] for row in rows],
                accuracy_delta=_seed_statistics([row["group_bootstrap"]["accuracy"]["delta"] for row in rows]),
                f1_delta=_seed_statistics([row["group_bootstrap"]["f1"]["delta"] for row in rows]),
                positive_accuracy_seeds=sum(row["group_bootstrap"]["accuracy"]["delta"] > 0 for row in rows),
            )
        )
    result = dict(
        protocol_sha256=sha256(output / "protocol.json"),
        test_barrier_sha256=sha256(output / "test_barrier.json"),
        test_cache_sha256=test_manifest["sha256"],
        selected_layers=list(selected),
        probe=dict(
            thresholds=thresholds.tolist(),
            validation_metrics=probe_selection["metrics"],
            validation_by_source=probe_selection.get("by_source"),
            test_metrics=probe_metrics,
            test_by_source=[
                _by_source(test_meta, probe_probability[:, i], thresholds[i]) for i in range(len(ALL_LAYERS))
            ],
            validation_complementarity="analysis/layer_complementarity_validation.json",
            test_complementarity="analysis/layer_complementarity_test.json",
            test_predictions_sha256=sha256(probe_test_path),
            selected_set_oracle=selected_set_oracle,
        ),
        heads=head_entries,
        head_seed_aggregates=head_seed_aggregates,
        pairwise_transitions=transitions,
        primary_matched_comparisons=primary_comparisons,
        matched_seed_aggregates=matched_seed_aggregates,
        limitations=protocol["limitations"],
    )
    dump(result_path, result)
    write_report(output, result)
    return result


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.3f}"


def write_report(output: Path, result: dict | None = None) -> Path:
    output = Path(output)
    if result is None:
        result = json.loads((output / "results.json").read_text())
    lines = [
        "# 同一冻结 LoRA 的全层 probe 与容量匹配融合诊断",
        "",
        "所有表示来自同一个 E03_s42 LoRA。训练、逐层 checkpoint、阈值、融合选层均只使用 train/validation；锁定 test barrier 后才提取 test 表示。",
        "",
        f"验证集错误互补贪心探索选择的四层为：**{', '.join(map(str, result['selected_layers']))}**（固定包含第 32 层；属于验证集探索性选择）。",
        "",
        "## 等容量逐层 probe",
        "",
        "| 层 | Val Acc | Val AUROC | Test Acc | Test F1 | Test AUROC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, validation, test in zip(ALL_LAYERS, result["probe"]["validation_metrics"], result["probe"]["test_metrics"]):
        lines.append(
            f"| {layer} | {_percentage(validation['accuracy'])} | {_percentage(validation['auroc'])} | "
            f"{_percentage(test['accuracy'])} | {_percentage(test['f1'])} | {_percentage(test['auroc'])} |"
        )
    oracle = result["probe"]["selected_set_oracle"]
    lines.extend(
        [
            "",
            f"四层 probe 的逐样本 oracle 上界：validation {_percentage(oracle['validation']['oracle_accuracy'])}%，"
            f"test {_percentage(oracle['test']['oracle_accuracy'])}%；test 可挽救第 32 层 "
            f"{oracle['test']['anchor_errors_rescued']}/{oracle['test']['anchor_errors']} 个错误。"
            "Oracle 仅量化错误互补上限，不是可部署路由器。",
        ]
    )
    lines.extend(
        [
            "",
            "## 冻结特征上的分类头",
            "",
            "repeat_last、fusion_fixed、fusion_selected 每个种子拥有完全相同的参数量与初始参数。last 是同构且按相同预算重新训练的小容量末层头。",
            "",
            "| 方法 | 种子 | 参数量 | Val Acc | Test Acc | Test F1 | Test AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["heads"]:
        validation, test = row["validation_metrics"], row["test_metrics"]
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['parameters']:,} | {_percentage(validation['accuracy'])} | "
            f"{_percentage(test['accuracy'])} | {_percentage(test['f1'])} | {_percentage(test['auroc'])} |"
        )
    if len({row["seed"] for row in result["heads"]}) > 1:
        lines.extend(
            [
                "",
                "按 head seed 汇总（均值 ± 样本标准差；LoRA 始终是同一个冻结 checkpoint）：",
                "",
                "| 方法 | Test Acc | Test F1 | Test AUROC |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in result["head_seed_aggregates"]:
            metrics = row["test_metrics"]
            lines.append(
                f"| {row['arm']} | {_percentage(metrics['accuracy']['mean'])} ± {_percentage(metrics['accuracy']['std'])} | "
                f"{_percentage(metrics['f1']['mean'])} ± {_percentage(metrics['f1']['std'])} | "
                f"{_percentage(metrics['auroc']['mean'])} ± {_percentage(metrics['auroc']['std'])} |"
            )
        aggregates = {row["arm"]: row for row in result["head_seed_aggregates"]}
        capacity_delta = (
            aggregates["repeat_last"]["test_metrics"]["accuracy"]["mean"]
            - aggregates["last"]["test_metrics"]["accuracy"]["mean"]
        )
        fixed_delta = next(
            row["accuracy_delta"]["mean"] for row in result["matched_seed_aggregates"]
            if row["candidate"] == "fusion_fixed"
        )
        selected_delta = next(
            row["accuracy_delta"]["mean"] for row in result["matched_seed_aggregates"]
            if row["candidate"] == "fusion_selected"
        )
        lines.extend(
            [
                "",
                "诊断分解（test accuracy 的 head-seed 均值差）：",
                "",
                f"- repeat_last − last = {_percentage(capacity_delta)} 个百分点，反映增加头容量本身的关联。",
                f"- fusion_fixed − repeat_last = {_percentage(fixed_delta)} 个百分点，反映固定多层信息在等容量条件下的关联。",
                f"- fusion_selected − repeat_last = {_percentage(selected_delta)} 个百分点，反映验证集互补选层在等容量条件下的关联。",
                f"- probe oracle 可挽救第 32 层 {_percentage(oracle['test']['rescue_fraction_of_anchor_errors'])}% 的测试错误；"
                "若 oracle 互补明显而两个等容量融合差值仍接近零，优先排查融合优化/聚合方式；若 oracle 也很低，表示该冻结 LoRA 的层间可用错误互补有限。",
            ]
        )
    lines.extend(
        [
            "",
            "## 与 repeat_last 的配对比较",
            "",
            "区间按 group_id 整组 bootstrap；随机化检验也按 group_id 整组交换。差值为候选融合头减 repeat_last。",
            "这些配对检验和 32 层矩阵均为探索性分析，未做多重比较校正，不能依据单个 p 值作确认性结论。",
            "",
            "| 候选 | 种子 | 准确率差值 [95% CI] | F1 差值 [95% CI] | 净修正 | McNemar p | Group permutation p (Acc/F1) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["primary_matched_comparisons"]:
        bootstrap = row["group_bootstrap"]
        permutation = row["group_permutation"]
        acc, f1 = bootstrap["accuracy"], bootstrap["f1"]
        lines.append(
            f"| {row['candidate']} | {row['seed']} | {_percentage(acc['delta'])} "
            f"[{_percentage(acc['ci95'][0])}, {_percentage(acc['ci95'][1])}] | {_percentage(f1['delta'])} "
            f"[{_percentage(f1['ci95'][0])}, {_percentage(f1['ci95'][1])}] | {row['transition']['net_corrections']} | "
            f"{row['mcnemar_exact']['two_sided_p']:.4g} | {permutation['accuracy']['two_sided_p']:.4g} / "
            f"{permutation['f1']['two_sided_p']:.4g} |"
        )
    if len(result.get("matched_seed_aggregates", [])):
        lines.extend(
            [
                "",
                "| 同种子差值汇总 | Acc 差值均值 ± 标准差 | F1 差值均值 ± 标准差 | Acc 提升种子 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in result["matched_seed_aggregates"]:
            acc, f1 = row["accuracy_delta"], row["f1_delta"]
            lines.append(
                f"| {row['candidate']} − {row['baseline']} | {_percentage(acc['mean'])} ± {_percentage(acc['std'])} | "
                f"{_percentage(f1['mean'])} ± {_percentage(f1['std'])} | "
                f"{row['positive_accuracy_seeds']}/{len(row['seeds'])} |"
            )
    lines.extend(
        [
            "",
            "## 按来源测试表现",
            "",
            "| 方法 | 种子 | 来源 | N | Acc | F1 | AUROC |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["heads"]:
        for source, metrics in row["by_source"].items():
            lines.append(
                f"| {row['arm']} | {row['seed']} | {source} | {metrics['n']} | {_percentage(metrics['accuracy'])} | "
                f"{_percentage(metrics['f1'])} | {_percentage(metrics['auroc'])} |"
            )
    lines.extend(
        [
            "",
            "32×32 概率相关、判定分歧、错误 Jaccard、定向 rescue 和 pair-oracle 矩阵见 `analysis/`；所有预测行均保留 base_id、group_id、source、category_id、text_form。",
            "",
            "## 解释边界",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result["limitations"])
    lines.append("")
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all(args: argparse.Namespace) -> dict:
    initialize(args)
    extract_splits(args, ("train", "validation"))
    train_probes(args)
    select_layers(args)
    train_heads(args)
    lock_barrier(args)
    extract_splits(args, ("test",))
    return evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("init", "extract", "probes", "select", "heads", "lock", "evaluate", "report", "run")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE_STUDY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-micro", type=int, default=32)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--selection-auroc-margin", type=float, default=0.002,
                        help="Validation AUROC eligibility tolerance; 0.002 equals 0.20 percentage point")
    parser.add_argument("--selection-accuracy-margin", type=float, default=0.02)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() and args.action in ("extract", "probes", "heads", "evaluate", "run"):
        raise RuntimeError("CUDA GPU required")
    if args.action == "init":
        result = initialize(args)
    elif args.action == "extract":
        if args.split is None:
            raise ValueError("extract requires --split")
        extract_splits(args, (args.split,))
        result = dict(split=args.split, cache=str(_cache_paths(args.output, args.split)[0]))
    elif args.action == "probes":
        result = train_probes(args)
    elif args.action == "select":
        result = select_layers(args)
    elif args.action == "heads":
        result = train_heads(args)
    elif args.action == "lock":
        result = lock_barrier(args)
    elif args.action == "evaluate":
        result = evaluate(args)
    elif args.action == "report":
        result = dict(report=str(write_report(args.output)))
    else:
        result = run_all(args)
    print(json.dumps(result if args.action in ("select", "lock", "report") else {"state": "complete", "action": args.action}, ensure_ascii=False))


if __name__ == "__main__":
    main()
