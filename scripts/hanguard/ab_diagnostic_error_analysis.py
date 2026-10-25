"""Explain paired B-OOF errors without modifying annotations or opening holdouts.

Consumes the registered A-only study and completed base_last/adapted_last/
adapted_mean readouts. All predictions must exactly cover the registered B
rows and agree with the SHA-bound per-fold prediction artifacts. Outputs only
JSON and parquet; Chinese inputs and official original English are copied as
review context, never treated as new annotations.

    python scripts/hanguard/ab_diagnostic_error_analysis.py \\
      --study outputs/hanguard_ab_diagnosis_20261012 \\
      --output outputs/hanguard_ab_diagnosis_20261012/error_analysis

The practical accuracy difference, declared before results, is 0.005 in
proportion units (0.5 percentage points). All uncertainty is conditional on
the fixed A checkpoint, B groups and trained readouts, not a proof that the
whole pipeline or every annotation is correct.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.ab_frozen_readout_diagnostics import (  # noqa: E402
    MAIN_ARMS, JOINT_ARMS, _metrics_with_row_thresholds, aggregate_stage,
    validate_outer_folds,
)
from scripts.hanguard.frozen_lora_layer_diagnostics import (  # noqa: E402
    dump, ordered_base_id_sha256, sha256,
)

SCHEMA = "hanguard_ab_error_analysis_v1"
PRACTICAL_DELTA = 0.005
REPRESENTATIONS = ("base_last", "adapted_last", "adapted_mean")
PAIRS = (
    ("best_single", "last"), ("repeat_best", "best_single"),
    ("fusion_fixed", "last"), ("fusion_fixed", "repeat_best"),
    ("fusion_selected", "repeat_best"),
    ("fusion_fixed_best_anchor", "repeat_best"),
    ("fusion_selected_best_anchor", "repeat_best"),
    ("fusion_fixed_best_anchor", "fusion_fixed"),
    ("fusion_selected_best_anchor", "fusion_selected"),
    ("joint_fixed", "joint_repeat_best"), ("joint_selected", "joint_repeat_best"),
)
REVIEW_COLUMNS = (
    "base_id", "group_id", "source", "label", "outer_fold", "prompt_tokens", "length_bin",
    "category_id", "type_label_mask", "type_label_conflict", "category_ambiguity_reason",
    "text_form", "curated_marker", "prompt", "original_english", "original_prompt",
    "original_english_sha256", "translation_status", "translation_changed", "revision_reason",
    "intake", "template_id", "recovery_id", "official_source_id", "binary_label_provenance",
)
CONDITIONAL_SCOPE = (
    "Explored train-internal B group-OOF analysis, conditional on this fixed A-only checkpoint, "
    "split and fitted readouts. Bootstrap resamples held groups without retraining; it does not "
    "include LoRA/head fitting variance, fresh-dataset shift or full pipeline uncertainty. "
    "Formal validation/test remain unopened by this analysis."
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_stop(study: Path, output: Path) -> None:
    for directory in (study, study / "lora_study", output, output.parent):
        path = directory / "STOP"
        if path.exists():
            raise RuntimeError(f"Diagnostic stopped: {path.read_text().strip()}")


def load_partition(study: Path) -> tuple[pd.DataFrame, Path, dict, dict]:
    training_path = study / "lora_study" / "protocol.json"
    training = read_json(training_path)
    if (training.get("training_entrypoint") != "hanguard_ab_diagnostic_v1" or
            training.get("formal_test_enabled") is not False or
            training.get("adapter_training_cohort") != "A"):
        raise ValueError("An A-only diagnostic study with formal test disabled is required")
    registration = read_json(training_path.parent / "registration.json")
    if registration.get("protocol_sha256") != sha256(training_path):
        raise ValueError("Training protocol registration changed")
    partition_path = Path(training["partition_manifest_path"]).resolve()
    if sha256(partition_path) != training["partition_manifest_sha256"]:
        raise ValueError("A/B partition manifest changed")
    partition = read_json(partition_path)
    if (partition.get("policy_id") != "hanguard_train_internal_ab_grouped_v1" or
            partition.get("invariants", {}).get("final_test_enabled") is not False):
        raise ValueError("Registered train-internal A/B partition is required")
    b_path = partition_path.parent / "B.parquet"
    if sha256(b_path) != partition["artifact_sha256"]["B.parquet"]:
        raise ValueError("Registered B rows changed")
    b = pd.read_parquet(b_path).sort_values("base_id", kind="stable").reset_index(drop=True)
    if not b.split.eq("train").all() or not b.diagnostic_partition.eq("B").all():
        raise ValueError("Error analysis cannot read non-B or non-training rows")
    if not b.prompt_harm_label.isin(("harmful", "unharmful")).all():
        raise ValueError("B contains an invalid source binary label")
    b["label"] = b.prompt_harm_label.eq("harmful").astype(np.int8)
    validate_outer_folds(b)
    b.attrs["ab_protocol_path"] = str(partition_path)
    identity = dict(training_protocol_path=str(training_path), training_protocol_sha256=sha256(training_path),
                    partition_manifest_path=str(partition_path), partition_manifest_sha256=sha256(partition_path),
                    B_sha256=sha256(b_path), ordered_B_base_id_sha256=ordered_base_id_sha256(b.base_id.astype(str).tolist()))
    return b, partition_path, partition, identity


def align_predictions(predictions: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Align exactly once by base_id and check authoritative binary/group labels."""
    required = {"base_id", "group_id", "source", "label", "outer_fold"}
    if not required.issubset(predictions) or predictions.base_id.astype(str).duplicated().any():
        raise ValueError("Prediction identities/metadata are incomplete or duplicated")
    if set(predictions.base_id.astype(str)) != set(b.base_id.astype(str)):
        raise ValueError("Predictions do not contain exactly B rows")
    aligned = predictions.assign(base_id=predictions.base_id.astype(str)).set_index("base_id").loc[b.base_id.astype(str)].reset_index()
    for column in ("group_id", "source", "label", "outer_fold"):
        if aligned[column].astype(str).tolist() != b[column].astype(str).tolist():
            raise ValueError(f"Prediction metadata differs from B: {column}")
    for column in ("category_id", "text_form"):
        if column in aligned and column in b and aligned[column].astype(str).tolist() != b[column].astype(str).tolist():
            raise ValueError(f"Prediction subgroup metadata differs from B: {column}")
    suffixes = [name.removeprefix("probability_") for name in aligned if name.startswith("probability_")]
    if not suffixes:
        raise ValueError("Prediction file contains no probabilities")
    for suffix in suffixes:
        probability = aligned[f"probability_{suffix}"].to_numpy(dtype=float)
        threshold = aligned[f"threshold_{suffix}"].to_numpy(dtype=float)
        prediction = aligned[f"prediction_{suffix}"].to_numpy()
        if (not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all() or
                not np.isfinite(threshold).all() or not np.isin(prediction, (0, 1)).all()):
            raise ValueError(f"Invalid probabilities, thresholds or decisions: {suffix}")
        if not np.array_equal(probability >= threshold, prediction.astype(bool)):
            raise ValueError(f"Decision/threshold mismatch: {suffix}")
    aligned.attrs = {}
    return aligned


def load_readout(study: Path, name: str, b: pd.DataFrame, identity: dict) -> dict:
    directory = study / "readouts" / name
    protocol_path = directory / "protocol.json"
    protocol = read_json(protocol_path)
    if protocol["ab_protocol_sha256"] != identity["partition_manifest_sha256"]:
        raise ValueError("Readout belongs to another A/B partition")
    expected_representation = "base" if name.startswith("base_") else "adapted"
    expected_pooling = name.split("_", 1)[1]
    if protocol["representation"] != expected_representation or protocol["pooling"] != expected_pooling:
        raise ValueError("Readout directory does not match its representation/pooling protocol")
    cache_manifest = Path(protocol["cache_manifest_path"])
    if sha256(cache_manifest) != protocol["cache_manifest_sha256"]:
        raise ValueError("Readout input cache manifest changed")
    summary_path = directory / "analysis" / "results.json"
    summary = read_json(summary_path)
    if summary.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("Readout summary protocol identity changed")
    result = dict(protocol=protocol, inputs={str(protocol_path): sha256(protocol_path),
                  str(summary_path): sha256(summary_path), str(cache_manifest): sha256(cache_manifest)})
    for stage, filename in (("probes", "probe_outer_predictions.parquet"), ("heads", "head_outer_predictions.parquet")):
        path = directory / "analysis" / filename
        if not path.exists():
            if stage == "heads":
                result[stage] = None
                continue
            raise FileNotFoundError(path)
        frame = align_predictions(pd.read_parquet(path), b)
        # Rebuild from individually SHA-bound complete-fold outputs. The derived
        # analysis parquet itself cannot silently alter probabilities/decisions.
        bound = align_predictions(aggregate_stage(directory, protocol, b, stage), b)
        pd.testing.assert_frame_equal(frame, bound, check_exact=True)
        result[stage] = frame
        result["inputs"][str(path)] = sha256(path)
        for fold in protocol["outer_folds"]:
            sentinel = directory / f"fold_{fold}" / stage / "complete.json"
            result["inputs"][str(sentinel)] = sha256(sentinel)
    return result


def cluster_mean_bootstrap(values: Sequence[float], groups: Sequence[Any], draws: int = 2000, seed: int = 42) -> dict:
    """Bootstrap a paired mean, including four-system differences of gains."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if values.shape != groups.shape or not len(values) or not np.isfinite(values).all() or draws < 1:
        raise ValueError("Nonempty aligned finite paired values/groups and positive draws required")
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for begin in range(0, draws, 8):
        size = min(8, draws - begin)
        index = rng.integers(0, len(unique), size=(size, len(unique)))
        samples[begin:begin + size] = sums[index].sum(1) / counts[index].sum(1)
    return dict(delta=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                bootstrap_mean=float(samples.mean()), groups=len(unique), draws=draws,
                unit="whole group_id clusters", scope=CONDITIONAL_SCOPE)


def practical_gain_interpretation(interval: Mapping[str, Any]) -> dict:
    lo, hi = map(float, interval["ci95"])
    if hi < PRACTICAL_DELTA:
        verdict = "conditional_upper_bound_below_practical_delta"
    elif lo >= PRACTICAL_DELTA:
        verdict = "conditional_interval_supports_at_least_practical_delta"
    elif lo > 0:
        verdict = "positive_conditional_gain_but_practical_magnitude_uncertain"
    else:
        verdict = "inconclusive_interval_covers_zero_and_practical_gain"
    return dict(practical_delta=PRACTICAL_DELTA, units="accuracy proportion; 0.005 = 0.5 percentage points",
                verdict=verdict,
                caveat="A conditional small-effect bound is not proof that data, optimization or the complete pipeline are correct")


def contexts(b: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    result = b.copy()
    if "prompt_tokens" in frame:
        result["prompt_tokens"] = frame.prompt_tokens.to_numpy()
    if result.prompt_tokens.isna().any() or (result.prompt_tokens <= 0).any():
        raise ValueError("Positive actual input token lengths required")
    result["length_bin"] = pd.cut(result.prompt_tokens, [0, 128, 256, 512, 1024, 2048, np.inf],
                                  labels=["1-128", "129-256", "257-512", "513-1024", "1025-2048", "2049+"],
                                  include_lowest=True).astype(str)
    curated = result.source.eq("chinese_curated")
    marker = result.prompt.astype(str).str.contains("请以「", regex=False)
    result["curated_marker"] = np.where(curated, np.where(marker, "marker_present", "marker_absent"), "outside_curated")
    return result[[column for column in REVIEW_COLUMNS if column in result]].reset_index(drop=True)


def correctness(frame: pd.DataFrame, arm: str) -> np.ndarray:
    return (frame[f"prediction_{arm}"].to_numpy() == frame.label.to_numpy()).astype(np.int8)


def transition_counts(frame: pd.DataFrame, candidate: str, baseline: str) -> dict:
    a, b = correctness(frame, candidate).astype(bool), correctness(frame, baseline).astype(bool)
    baseline_errors = int((~b).sum())
    baseline_correct = int(b.sum())
    rescued, harmed = int((a & ~b).sum()), int((~a & b).sum())
    return dict(rows=len(frame), groups=int(frame.group_id.nunique()),
                baseline_errors=baseline_errors, baseline_correct=baseline_correct,
                rescued=rescued, harmed=harmed,
                rescued_fraction_of_baseline_errors=rescued / baseline_errors if baseline_errors else None,
                harmed_fraction_of_baseline_correct=harmed / baseline_correct if baseline_correct else None,
                net_fraction_of_baseline_errors=(rescued - harmed) / baseline_errors if baseline_errors else None,
                common_wrong=int((~a & ~b).sum()), common_correct=int((a & b).sum()),
                net_correct=int(a.sum() - b.sum()), accuracy_gain=float((a.astype(float) - b).mean()))


def probe_headroom(readout: dict) -> dict:
    probes = readout["probes"]
    union_correct = np.stack([correctness(probes, f"L{layer}") for layer in range(1, 33)], axis=1).any(1)
    baselines = {"linear_probe_L32": correctness(probes, "L32").astype(bool)}
    if readout["heads"] is not None:
        baselines.update({f"head_{arm}": correctness(readout["heads"], arm).astype(bool)
                          for arm in ("last", "best_single", "repeat_best")})
    result = dict(full32_individual_probe_union_accuracy=float(union_correct.mean()),
                  all32_individual_probes_wrong=int((~union_correct).sum()), baselines={},
                  scope="Oracle union of already fitted linear probes' individually correct decisions; not deployable accuracy or a strict fusion upper bound")
    for name, correct in baselines.items():
        errors = int((~correct).sum())
        rescue = int((~correct & union_correct).sum())
        result["baselines"][name] = dict(baseline_errors=errors, probe_union_rescuable_errors=rescue,
            probe_union_rescuable_fraction_of_baseline_errors=rescue / errors if errors else None,
            diagnostic_oracle_combined_accuracy=float((correct | union_correct).mean()))
    return result


def subgroup_counts(frame: pd.DataFrame, review: pd.DataFrame, candidate: str, baseline: str) -> dict:
    subgroups = {}
    for column in ("source", "length_bin", "outer_fold", "category_id", "text_form", "curated_marker"):
        if column not in review:
            continue
        entries = {}
        for value, indices in review.groupby(column, dropna=False, observed=True, sort=True).groups.items():
            part = frame.iloc[np.asarray(indices)]
            entries[str(value)] = dict(**transition_counts(part, candidate, baseline),
                label_counts={str(label): int(count) for label, count in part.label.value_counts().sort_index().items()},
                candidate_metrics=_metrics_with_row_thresholds(part, candidate),
                baseline_metrics=_metrics_with_row_thresholds(part, baseline))
        subgroups[column] = entries
    return subgroups


def pair_error_rows(frame: pd.DataFrame, review: pd.DataFrame, representation: str, candidate: str, baseline: str) -> pd.DataFrame:
    a, b = correctness(frame, candidate).astype(bool), correctness(frame, baseline).astype(bool)
    state = np.where(a & ~b, "rescued", np.where(~a & b, "harmed", np.where(~a & ~b, "common_wrong", "common_correct")))
    selected = state != "common_correct"
    result = review.loc[selected].copy()
    result["representation"] = representation
    result["candidate"] = candidate
    result["baseline"] = baseline
    result["error_state"] = state[selected]
    for role, arm in (("candidate", candidate), ("baseline", baseline)):
        for field in ("probability", "threshold", "prediction"):
            result[f"{role}_{field}"] = frame.loc[selected, f"{field}_{arm}"].to_numpy()
    return result


def layer_comparisons(first: dict, second: dict, first_name: str, second_name: str) -> list[dict]:
    a, b = first["probes"], second["probes"]
    return [dict(layer=layer, first=first_name, second=second_name,
                 first_metrics=_metrics_with_row_thresholds(a, f"L{layer}"),
                 second_metrics=_metrics_with_row_thresholds(b, f"L{layer}"),
                 net_correct_second_minus_first=int(correctness(b, f"L{layer}").sum() - correctness(a, f"L{layer}").sum()),
                 scope="Descriptive same-row layer curve; do not choose a new model from these outer-held scores")
            for layer in range(1, 33)]


def cross_representation(first: dict, second: dict, first_name: str, second_name: str, draws: int, seed: int) -> dict:
    result = dict(first=first_name, second=second_name,
                  layer_curve=layer_comparisons(first, second, first_name, second_name),
                  scope=CONDITIONAL_SCOPE)
    a, b = first["heads"], second["heads"]
    if a is None or b is None:
        result["head_comparison_status"] = "not_available_without_completed_heads_for_both_representations"
        return result
    arms = [arm for arm in MAIN_ARMS + JOINT_ARMS if f"prediction_{arm}" in a and f"prediction_{arm}" in b]
    per_head = {}
    for arm in arms:
        delta = correctness(b, arm).astype(float) - correctness(a, arm)
        interval = cluster_mean_bootstrap(delta, b.group_id, draws, seed)
        per_head[arm] = dict(first_metrics=_metrics_with_row_thresholds(a, arm), second_metrics=_metrics_with_row_thresholds(b, arm),
                             second_minus_first_group_bootstrap=interval,
                             interpretation=practical_gain_interpretation(interval))
    gain_changes = []
    for candidate, baseline in PAIRS:
        if candidate not in arms or baseline not in arms:
            continue
        first_gain = correctness(a, candidate).astype(float) - correctness(a, baseline)
        second_gain = correctness(b, candidate).astype(float) - correctness(b, baseline)
        gain_changes.append(dict(candidate=candidate, baseline=baseline,
            first_gain=float(first_gain.mean()), second_gain=float(second_gain.mean()),
            first_error_headroom=transition_counts(a, candidate, baseline),
            second_error_headroom=transition_counts(b, candidate, baseline),
            difference_of_gains_group_bootstrap=cluster_mean_bootstrap(second_gain - first_gain, b.group_id, draws, seed),
            interpretation="A paired conditional gain change, not proof of a unique causal mechanism; a stronger adapted baseline leaves fewer errors to repair, and checkpoint/layer selection can differ"))
    result.update(head_comparison_status="completed_same_row_paired", heads=per_head, gain_changes=gain_changes)
    return result


def analyze(args: argparse.Namespace) -> dict:
    study, output = Path(args.study).resolve(), Path(args.output).resolve()
    if args.bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    check_stop(study, output)
    if output == study or output.is_relative_to(study / "partition") or study.is_relative_to(output):
        raise ValueError("Error-analysis output must be a separate study child or external directory")
    b, partition_path, partition, identity = load_partition(study)
    if output.is_relative_to(partition_path.parent) or partition_path.parent.is_relative_to(output):
        raise ValueError("Error analysis cannot write inside or above the immutable partition")
    loaded, unavailable = {}, []
    for name in REPRESENTATIONS:
        check_stop(study, output)
        path = study / "readouts" / name / "analysis" / "results.json"
        if not path.exists():
            if args.allow_partial:
                unavailable.append(name)
                continue
            raise FileNotFoundError(f"Missing completed readout: {path}")
        loaded[name] = load_readout(study, name, b, identity)
    if not loaded:
        raise ValueError("No completed B readouts are available")
    if "adapted_last" not in loaded or loaded["adapted_last"]["heads"] is None:
        raise ValueError("Completed adapted-last heads are required for the primary diagnosis")
    result = dict(schema=SCHEMA, complete=False, input_identity=identity, rows=len(b), groups=int(b.group_id.nunique()),
                  scope=CONDITIONAL_SCOPE, representations_available=list(loaded), representations_unavailable=unavailable,
                  practical_accuracy_delta=PRACTICAL_DELTA,
                  label_interpretation="category_id=-1 marks unknown/conflicting type, not an incorrect binary label; model errors do not establish annotation errors",
                  subgroup_interpretation="Exploratory descriptions with no multiple-testing correction or causal marker claim",
                  headroom_interpretation="Smaller absolute adapted fusion gain can follow from fewer remaining baseline errors; compare rescue/net gain fractions before hypothesizing lost complementarity",
                  formal_holdout_accessed=False, source_annotations_modified=False,
                  source_sha256={"ab_diagnostic_error_analysis.py": sha256(Path(__file__)),
                                 "ab_frozen_readout_diagnostics.py": sha256(ROOT / "scripts/hanguard/ab_frozen_readout_diagnostics.py")},
                  bootstrap=dict(draws=args.bootstrap_draws, seed=args.seed), within_representation={})
    input_sha = {str(Path(identity["training_protocol_path"])): identity["training_protocol_sha256"],
                 str(partition_path): identity["partition_manifest_sha256"], str(partition_path.parent / "B.parquet"): identity["B_sha256"]}
    error_frames, all_wrong_frames = [], []
    for name, readout in loaded.items():
        input_sha.update(readout["inputs"])
        frame = readout["heads"]
        if frame is None:
            result["within_representation"][name] = dict(heads_available=False, probe_headroom=probe_headroom(readout))
            continue
        review = contexts(b, frame)
        arms = [arm for arm in MAIN_ARMS + JOINT_ARMS if f"prediction_{arm}" in frame]
        entries = []
        for candidate, baseline in PAIRS:
            if candidate not in arms or baseline not in arms:
                continue
            delta = correctness(frame, candidate).astype(float) - correctness(frame, baseline)
            interval = cluster_mean_bootstrap(delta, frame.group_id, args.bootstrap_draws, args.seed)
            entries.append(dict(candidate=candidate, baseline=baseline, **transition_counts(frame, candidate, baseline),
                group_bootstrap=interval, interpretation=practical_gain_interpretation(interval),
                subgroups=subgroup_counts(frame, review, candidate, baseline)))
            error_frames.append(pair_error_rows(frame, review, name, candidate, baseline))
        result["within_representation"][name] = dict(heads_available=True,
            probe_headroom=probe_headroom(readout),
            metrics={arm: _metrics_with_row_thresholds(frame, arm) for arm in arms}, comparisons=entries)
        all_wrong = np.stack([correctness(frame, arm) for arm in arms], axis=1).sum(1) == 0
        wrong = review.loc[all_wrong].copy()
        wrong["representation"] = name
        wrong["definition"] = "all completed registered head arms predict the binary label incorrectly"
        all_wrong_frames.append(wrong)
    cross = []
    for first, second in (("base_last", "adapted_last"), ("adapted_last", "adapted_mean")):
        if first in loaded and second in loaded:
            cross.append(cross_representation(loaded[first], loaded[second], first, second, args.bootstrap_draws, args.seed))
    result["cross_representation"] = cross
    result["input_artifact_sha256"] = input_sha
    existing_path = output / "results.json"
    if existing_path.exists():
        existing = read_json(existing_path)
        identity_keys = ("schema", "input_identity", "input_artifact_sha256", "source_sha256", "bootstrap",
                         "representations_available", "representations_unavailable", "practical_accuracy_delta")
        if existing.get("complete") is not True or any(existing.get(key) != result[key] for key in identity_keys):
            raise ValueError("Existing error analysis has another identity; choose a fresh output directory")
        for name, digest in existing["output_artifact_sha256"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Existing error-analysis output changed: {name}")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    curve = []
    for layer in range(1, 33):
        row = dict(layer=layer)
        for name, readout in loaded.items():
            metric = _metrics_with_row_thresholds(readout["probes"], f"L{layer}")
            for key in ("accuracy", "accuracy_at_0_5", "bce", "auroc", "recall", "fpr", "fp", "fn"):
                row[f"{name}_{key}"] = metric[key]
        curve.append(row)
    curve_frame = pd.DataFrame(curve)
    for suffix in ("parquet", "csv"):
        path = output / f"layer_curve.{suffix}"
        if suffix == "parquet":
            curve_frame.to_parquet(path, index=False)
        else:
            curve_frame.to_csv(path, index=False)
        artifacts.append(path)
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    for state in ("common_wrong", "rescued", "harmed"):
        path = output / f"{state}.parquet"
        errors.loc[errors.error_state.eq(state)].to_parquet(path, index=False)
        artifacts.append(path)
    all_wrong_path = output / "all_registered_heads_common_wrong.parquet"
    pd.concat(all_wrong_frames, ignore_index=True).to_parquet(all_wrong_path, index=False)
    artifacts.append(all_wrong_path)
    # Bind complete per-row probabilities/decisions without copying all original
    # prompts repeatedly into one very large wide table.
    prediction_path = output / "joined_head_predictions.parquet"
    wide = b[["base_id", "group_id", "source", "label", "outer_fold"]].copy()
    for name, readout in loaded.items():
        if readout["heads"] is None:
            continue
        for column in readout["heads"]:
            if column.startswith(("probability_", "threshold_", "prediction_")):
                wide[f"{name}__{column}"] = readout["heads"][column].to_numpy()
    wide.to_parquet(prediction_path, index=False)
    artifacts.append(prediction_path)
    check_stop(study, output)
    for path, digest in input_sha.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Input changed during error analysis: {path}")
    result["output_artifact_sha256"] = {path.name: sha256(path) for path in artifacts}
    result["error_rows"] = {state: int(errors.error_state.eq(state).sum()) for state in ("common_wrong", "rescued", "harmed")}
    result["error_row_unit"] = "comparison × representation × base_id; one source row can appear in several comparisons"
    result["complete"] = True
    dump(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly mark missing secondary representations")
    args = parser.parse_args()
    result = analyze(args)
    print(json.dumps(dict(results_path=str(Path(args.output).resolve() / "results.json"), complete=result["complete"],
                         rows=result["rows"], representations_available=result["representations_available"],
                         error_rows=result["error_rows"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
