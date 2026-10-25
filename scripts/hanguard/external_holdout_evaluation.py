"""One-shot external holdout extraction and evaluation.

The registry contains row pointers and hashes, never copied prompts.  This
script resolves each prompt from the frozen source parquet, verifies its exact
and normalized identity, and extracts all 32 final-token block states with the
same frozen E03_s42 LoRA used by the internal diagnostics.

The workflow is deliberately ordered as ``lock -> extract -> evaluate``.
``lock`` hashes the registry, source data, every evaluated dynamic-gate and
OOF-stack artifact, the adapter, and this implementation before any checkpoint
is deserialized or any model is run.  ``extract`` marks the registry consumed
before the first backbone forward.  Every reported decision uses the fixed
threshold 0.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.hanguard import dynamic_gate_diagnostics as dynamic  # noqa: E402
from scripts.hanguard import frozen_lora_layer_diagnostics as frozen  # noqa: E402
from scripts.hanguard import oof_probe_logit_stack as oof  # noqa: E402


DEFAULT_REGISTRY = ROOT / "outputs/hanguard_external_holdouts_20261012"
DEFAULT_DYNAMIC = ROOT / "outputs/hanguard_dynamic_gate_diagnostics_20261012"
DEFAULT_OOF = ROOT / "outputs/hanguard_oof_probe_logit_stack_20261012"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_external_holdout_evaluation_20261012"
REGISTRY_SCHEMA = "hanguard_external_holdout_registry_v1"
PRIMARY_THRESHOLD = 0.5
OOF_CALIBRATOR_FIT_SOURCE = "strict nested outer-heldout stack scores"
OOF_CALIBRATION_SELECTION_SOURCE = "repeat32 strict nested train outer-group-OOF"
REQUIRED_INDEX_COLUMNS = {
    "base_id", "group_id", "source", "source_file", "source_row",
    "prompt_sha256", "normalized_prompt_sha256", "prompt_harm_label", "label",
}


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize_prompt(value: str) -> str:
    """Registry v1 normalization: NFKC, casefold, no Cf, no whitespace."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(
        character
        for character in value
        if unicodedata.category(character) != "Cf" and not character.isspace()
    )


def normalized_prompt_sha256(value: str) -> str:
    return sha256_text(normalize_prompt(value))


def expected_group_id(normalized_sha256: str) -> str:
    return sha256_text("normalized_prompt\0" + str(normalized_sha256))


def expected_base_id(source: str, source_row: int, prompt_sha256: str) -> str:
    return sha256_text(f"{source}\0{int(source_row)}\0{prompt_sha256}")


def registry_ordered_base_id_sha256(base_ids: Sequence[Any]) -> str:
    """Match the v1 registry builder's ordered-ID digest exactly."""
    return sha256_text("\n".join(map(str, base_ids)))


def _dataset_records(manifest: Mapping[str, Any]) -> dict[str, dict]:
    result = {
        str(name): dict(value)
        for name, value in manifest.items()
        if isinstance(value, Mapping) and "index_file" in value and "source_file" in value
    }
    if not result:
        raise ValueError("External registry contains no dataset index records")
    return result


def validate_registry(registry: Path) -> dict:
    """Validate immutable registry/index/source bytes without reading prompts."""
    registry = Path(registry).resolve()
    manifest_path, registry_lock_path = registry / "manifest.json", registry / "lock.json"
    if not manifest_path.is_file() or not registry_lock_path.is_file():
        raise FileNotFoundError("External registry manifest and lock are both required")
    manifest, registry_lock = _json(manifest_path), _json(registry_lock_path)
    if manifest.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"Unsupported external registry schema: {manifest.get('schema')}")
    if registry_lock.get("state") != "registered_unconsumed":
        raise ValueError("External registry was not locked in the unconsumed state")
    files = registry_lock.get("files", {})
    for name, record in files.items():
        path = registry / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Registered file extent changed: {path}")
        if frozen.sha256(path) != record["sha256"]:
            raise ValueError(f"Registered file hash changed: {path}")
    if files.get("manifest.json", {}).get("sha256") != frozen.sha256(manifest_path):
        raise ValueError("Registry manifest is not the file frozen by lock.json")
    datasets = _dataset_records(manifest)
    snapshots = {}
    for name, record in datasets.items():
        index_path = _resolve(record["index_file"])
        source_path = _resolve(record["source_file"])
        if not index_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"Missing external index/source for {name}")
        if frozen.sha256(index_path) != record["index_sha256"] or index_path.stat().st_size != record["index_bytes"]:
            raise ValueError(f"External index changed: {name}")
        if frozen.sha256(source_path) != record["source_sha256"] or source_path.stat().st_size != record["source_bytes"]:
            raise ValueError(f"External source parquet changed: {name}")
        frame = pd.read_parquet(index_path)
        if not REQUIRED_INDEX_COLUMNS.issubset(frame.columns):
            raise ValueError(f"{name} index lacks {REQUIRED_INDEX_COLUMNS - set(frame.columns)}")
        if len(frame) != int(record.get("selected_rows", record.get("source_rows", -1))):
            raise ValueError(f"{name} selected row count changed")
        if frame.base_id.isna().any() or frame.base_id.duplicated().any():
            raise ValueError(f"{name} base_id must be unique and nonempty")
        if frame.base_id.tolist() != sorted(frame.base_id.tolist()):
            raise ValueError(f"{name} index must stay in base_id order")
        if not frame.label.isin((0, 1)).all() or frame.group_id.isna().any():
            raise ValueError(f"{name} labels/groups are invalid")
        if (frame.groupby("group_id").label.nunique() != 1).any():
            raise ValueError(f"{name} contains conflicting labels inside a group")
        ordered = registry_ordered_base_id_sha256(frame.base_id.astype(str).tolist())
        if ordered != record["ordered_base_id_sha256"]:
            raise ValueError(f"{name} ordered base_id hash changed")
        if set(frame.source_file.astype(str)) != {record["source_file"]}:
            raise ValueError(f"{name} row source_file differs from manifest")
        snapshots[name] = dict(
            index_path=str(index_path), index_sha256=record["index_sha256"],
            source_path=str(source_path), source_sha256=record["source_sha256"],
            rows=int(len(frame)), groups=int(frame.group_id.nunique()),
            ordered_base_id_sha256=ordered,
        )
    return dict(
        registry=str(registry), manifest=manifest, registry_lock=registry_lock,
        manifest_sha256=frozen.sha256(manifest_path),
        registry_lock_sha256=frozen.sha256(registry_lock_path), datasets=snapshots,
    )


def recover_prompts(registry: Path, dataset: str) -> tuple[pd.DataFrame, list[str], dict]:
    """Resolve source rows and prove every exact/normalized identity."""
    snapshot = validate_registry(registry)
    records = _dataset_records(snapshot["manifest"])
    if dataset not in records:
        raise KeyError(f"Unknown registered dataset: {dataset}")
    record = records[dataset]
    index = pd.read_parquet(_resolve(record["index_file"])).reset_index(drop=True)
    source = pd.read_parquet(_resolve(record["source_file"])).reset_index(drop=True)
    rows = index.source_row.to_numpy(dtype=np.int64)
    if (rows < 0).any() or (rows >= len(source)).any():
        raise ValueError(f"{dataset} source_row is out of bounds")
    if "prompt" not in source or "prompt_harm_label" not in source:
        raise ValueError(f"{dataset} source lacks prompt/label columns")
    prompts = source.iloc[rows].prompt.astype(str).tolist()
    source_labels = source.iloc[rows].prompt_harm_label.astype(str).tolist()
    exact = [sha256_text(prompt) for prompt in prompts]
    normalized = [normalized_prompt_sha256(prompt) for prompt in prompts]
    groups = [expected_group_id(value) for value in normalized]
    bases = [
        expected_base_id(source_name, row, digest)
        for source_name, row, digest in zip(index.source.astype(str), rows, exact)
    ]
    checks = {
        "prompt_sha256": np.array_equal(np.asarray(exact), index.prompt_sha256.astype(str)),
        "normalized_prompt_sha256": np.array_equal(
            np.asarray(normalized), index.normalized_prompt_sha256.astype(str)
        ),
        "group_id": np.array_equal(np.asarray(groups), index.group_id.astype(str)),
        "base_id": np.array_equal(np.asarray(bases), index.base_id.astype(str)),
        "prompt_harm_label": np.array_equal(np.asarray(source_labels), index.prompt_harm_label.astype(str)),
        "binary_label": np.array_equal(
            np.asarray([int(value == "harmful") for value in source_labels], dtype=np.int8),
            index.label.to_numpy(dtype=np.int8),
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{dataset} source-row reconstruction failed: {failed}")
    if "category_id" in index and "category_id" in source:
        if not np.array_equal(index.category_id.astype(str), source.iloc[rows].category_id.astype(str)):
            raise ValueError(f"{dataset} category_id changed at source rows")
    metadata = index.copy()
    return metadata, prompts, dict(**snapshot["datasets"][dataset], identity_checks=checks)


def _artifact_hashes(paths: Mapping[str, Path | str]) -> dict[str, dict]:
    output = {}
    for name, value in sorted(paths.items()):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing model artifact {name}: {path}")
        output[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=frozen.sha256(path))
    return output


def _frozen_representation_identity(source: Path) -> tuple[dict, dict[str, Path]]:
    """Describe the frozen backbone/adapter representation used by a study."""
    source = Path(source).resolve()
    protocol_path = source / "protocol.json"
    train_manifest_path = source / "features/train.manifest.json"
    # This is a read-only hash/schema check.  It must happen during lock, before
    # mark_consumed, so stale extractor code cannot consume the one-shot registry.
    protocol = frozen.load_protocol(source)
    train_manifest = _json(train_manifest_path)
    model_path = _resolve(protocol["model_path"])
    adapter_path = _resolve(protocol["adapter_path"])
    if frozen.sha256(adapter_path) != protocol.get("adapter_sha256"):
        raise ValueError("Frozen representation adapter bytes changed")
    if tuple(map(int, protocol.get("layers", ()))) != frozen.ALL_LAYERS:
        raise ValueError("Frozen representation does not contain the registered 32 layers")
    if protocol.get("model_artifacts") != frozen._model_artifact_identity(model_path):
        raise ValueError("Frozen representation base-model identity changed")
    for filename, key in (("config.json", "model_config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        expected = protocol.get(key)
        path = model_path / filename
        if expected is None or not path.is_file() or frozen.sha256(path) != expected:
            raise ValueError(f"Frozen representation {filename} identity changed")
    expected_manifest = dict(
        split="train", layers=len(frozen.ALL_LAYERS), adapter_sha256=protocol["adapter_sha256"],
        protocol_sha256=frozen.sha256(protocol_path), complete=True,
    )
    changed_manifest = [
        key for key, expected in expected_manifest.items() if train_manifest.get(key) != expected
    ]
    if changed_manifest:
        raise ValueError(f"Frozen train feature manifest identity changed: {changed_manifest}")
    if train_manifest.get("representation") != "raw block output at final valid token, selected inside each forward hook":
        raise ValueError("Frozen train feature representation semantics changed")
    identity = dict(
        source=str(source),
        source_protocol_sha256=frozen.sha256(protocol_path),
        adapter_path=str(adapter_path),
        adapter_sha256=protocol["adapter_sha256"],
        model_path=str(model_path),
        model_artifacts=protocol["model_artifacts"],
        model_config_sha256=protocol["model_config_sha256"],
        tokenizer_sha256=protocol["tokenizer_sha256"],
        layers=list(protocol["layers"]),
        hidden_size=int(train_manifest["hidden_size"]),
        representation="raw block output at final valid token",
    )
    artifacts: dict[str, Path] = {
        "frozen/protocol": protocol_path,
        "frozen/train_feature_manifest": train_manifest_path,
        "frozen/adapter": adapter_path,
        "frozen/model_config": model_path / "config.json",
        "frozen/tokenizer": model_path / "tokenizer.json",
        "frozen/model_index": model_path / "model.safetensors.index.json",
    }
    for filename in protocol["model_artifacts"].get("shard_bytes", {}):
        artifacts[f"frozen/model_weight/{filename}"] = model_path / filename
    return identity, artifacts


def assert_same_frozen_representation(dynamic_spec: Mapping[str, Any], oof_spec: Mapping[str, Any]) -> dict:
    """Reject paired studies that do not consume the exact same frozen features."""
    dynamic_identity = dict(dynamic_spec.get("representation", {}))
    oof_identity = dict(oof_spec.get("representation", {}))
    required = (
        "source", "source_protocol_sha256", "adapter_path", "adapter_sha256",
        "model_path", "model_artifacts", "model_config_sha256", "tokenizer_sha256",
        "layers", "hidden_size", "representation",
    )
    missing = [key for key in required if key not in dynamic_identity or key not in oof_identity]
    if missing:
        raise ValueError(f"Study representation identity is incomplete: {missing}")
    changed = [key for key in required if dynamic_identity[key] != oof_identity[key]]
    if changed:
        raise ValueError(f"Dynamic and OOF studies use different frozen representations: {changed}")
    if tuple(dynamic_identity["layers"]) != frozen.ALL_LAYERS:
        raise ValueError("External extraction requires the registered 32-layer representation")
    return dynamic_identity


def selected_calibration_family(
    selection: Mapping[str, Any], arm_models: Mapping[str, Mapping[str, Any]]
) -> str:
    """Read and cross-check the actually selected shared OOF calibrator family."""
    family = selection.get("calibration_selection", {}).get("selected")
    model_families = {
        str(arm): str(model.get("calibrator", {}).get("method"))
        for arm, model in arm_models.items()
    }
    if family not in ("identity", "temperature", "platt"):
        raise ValueError(f"OOF calibration family is missing or invalid: {family}")
    if set(model_families) != set(oof.ALL_ARMS) or set(model_families.values()) != {family}:
        raise ValueError(f"OOF arm calibrators differ from selected family {family}: {model_families}")
    fit_sources = {
        str(arm): model.get("calibrator", {}).get("fit_source")
        for arm, model in arm_models.items()
    }
    if set(fit_sources.values()) != {OOF_CALIBRATOR_FIT_SOURCE}:
        raise ValueError(
            "OOF deployment calibrators were not fit from strict nested outer-heldout scores: "
            f"{fit_sources}"
        )
    return str(family)


def select_dynamic_primary(
    selections: Mapping[int, Mapping[str, Any]], tie_tolerance: float = 1e-12
) -> dict:
    """Choose fixed versus selected using only mean validation BCE across seeds."""
    if not selections or tie_tolerance < 0:
        raise ValueError("Dynamic primary selection requires validation records and a nonnegative tie tolerance")
    candidates = ("dynamic_fixed", "dynamic_selected")
    per_seed = {}
    for seed, selection in sorted(selections.items()):
        if selection.get("split") != "validation" or int(selection.get("seed", -1)) != int(seed):
            raise ValueError(f"Dynamic seed {seed} does not contain a validation-only selection")
        arms = selection.get("arms", {})
        if any(arm not in arms for arm in candidates):
            raise ValueError(f"Dynamic seed {seed} lacks a primary candidate")
        per_seed[str(seed)] = {
            arm: float(arms[arm]["validation_bce"])
            for arm in candidates
        }
        if not all(math.isfinite(value) for value in per_seed[str(seed)].values()):
            raise ValueError(f"Dynamic seed {seed} has nonfinite validation BCE")
    mean_bce = {
        arm: float(np.mean([values[arm] for values in per_seed.values()]))
        for arm in candidates
    }
    best = min(mean_bce.values())
    eligible = [arm for arm in candidates if mean_bce[arm] <= best + tie_tolerance]
    selected = min(eligible, key=lambda arm: candidates.index(arm))
    return dict(
        selected_arm=selected,
        split="validation",
        criterion="minimum arithmetic mean per-seed best-checkpoint validation BCE",
        candidates=list(candidates),
        per_seed_validation_bce=per_seed,
        mean_validation_bce=mean_bce,
        tie_tolerance=float(tie_tolerance),
        tie_break="prefer dynamic_fixed, then dynamic_selected",
    )


def discover_dynamic_study(
    study: Path, primary_arm: str | None = None
) -> tuple[dict, dict[str, Path]]:
    study = Path(study).resolve()
    protocol = dynamic.load_protocol(study)
    if primary_arm not in (None, "auto"):
        raise ValueError("Dynamic primary arm must be selected automatically from validation")
    # Verifies the dynamic study froze its choices before its own test access.
    dynamic.frozen.verify_test_barrier(
        study, dynamic._barrier_paths(study, protocol), dynamic._barrier_identity(study, protocol)
    )
    evaluation_provenance = dynamic.verify_evaluation_amendment(
        study, protocol, require_reproducibility=True
    )
    summary_path = study / "heads/selection.json"
    barrier_path = study / "test_barrier.json"
    if not summary_path.is_file() or not barrier_path.is_file():
        raise FileNotFoundError("Dynamic-gate selections/barrier are incomplete")
    summary = _json(summary_path)
    seeds = list(map(int, protocol["training"]["seeds"]))
    if seeds != [42, 43, 44, 45, 46]:
        raise ValueError("External dynamic evaluation requires the registered five seeds 42..46")
    if list(protocol.get("arms", ())) != list(dynamic.ARMS):
        raise ValueError("Dynamic protocol arm schema differs from the registered study")
    architecture = protocol.get("architecture", {})
    expected_architecture = {"layer_count": 4, "dropout": 0.1, "fusion_init_scale": 1e-3}
    changed_architecture = [
        key for key, expected in expected_architecture.items()
        if architecture.get(key) != expected
    ]
    if changed_architecture or int(architecture.get("width", 0)) < 1 or int(architecture.get("gate_width", 0)) < 1:
        raise ValueError(f"Dynamic evaluation architecture is incompatible: {changed_architecture}")
    selected_layers = list(map(int, protocol.get("selected_layers", ())))
    if len(selected_layers) != 4 or 32 not in selected_layers:
        raise ValueError("Dynamic protocol selected-layer schema is invalid")
    if summary.get("split") != "validation" or summary.get("seeds") != seeds:
        raise ValueError("Dynamic head selection identity differs")
    if (
        summary.get("selected_layers") != selected_layers
        or summary.get("matched_arms") != list(dynamic.ARMS)
    ):
        raise ValueError("Dynamic validation summary schema differs from its protocol")
    paths: dict[str, Path] = {
        "dynamic/protocol": study / "protocol.json",
        "dynamic/test_barrier": barrier_path,
        "dynamic/head_summary": summary_path,
    }
    provenance_spec = dict(
        amendment_required=bool(evaluation_provenance.get("required", False)),
        locked_evaluation_code_sha256=evaluation_provenance["locked_evaluation_code_sha256"],
        current_evaluation_code_sha256=frozen.sha256(Path(dynamic.__file__)),
    )
    if evaluation_provenance.get("required"):
        amendment_path = study / dynamic.EVALUATION_AMENDMENT_FILE
        audit_path = study / dynamic.PRIMARY_REPRODUCIBILITY_FILE
        paths["dynamic/evaluation_amendment"] = amendment_path
        paths["dynamic/evaluation_primary_reproducibility"] = audit_path
        for name, record in evaluation_provenance.get("artifacts", {}).items():
            paths[f"dynamic/amendment_artifact/{name}"] = Path(record["path"])
        audit = evaluation_provenance["primary_reproducibility"]
        provenance_spec.update(
            amendment_sha256=evaluation_provenance["amendment_sha256"],
            primary_reproducibility_sha256=audit["audit_sha256"],
            primary_reproducibility_state=audit["state"],
            exact_primary_reproduction=bool(audit["all_exact"]),
        )
    seed_selections = {}
    validation_evidence = {}
    raw_summary_entries = list(summary.get("entries", []))
    summary_entries = {int(entry["seed"]): entry for entry in raw_summary_entries}
    if (
        len(raw_summary_entries) != len(seeds)
        or len(summary_entries) != len(seeds)
        or set(summary_entries) != set(seeds)
    ):
        raise ValueError("Dynamic validation summary does not contain every registered seed")
    for seed in seeds:
        directory = study / "heads" / f"seed_{seed}"
        selection_path = directory / "selection.json"
        selection = _json(selection_path)
        if selection.get("split") != "validation" or set(selection.get("arms", {})) != set(dynamic.ARMS):
            raise ValueError(f"Dynamic seed {seed} selection is incomplete")
        if (
            int(selection.get("seed", -1)) != seed
            or selection.get("selected_layers") != selected_layers
        ):
            raise ValueError(f"Dynamic seed {seed} selection identity differs")
        if selection != summary_entries[seed]:
            raise ValueError(f"Dynamic seed {seed} differs from its validation summary")
        seed_selections[seed] = selection
        validation_prediction_path = directory / "validation_predictions.parquet"
        if selection.get("validation_predictions_sha256") != frozen.sha256(validation_prediction_path):
            raise ValueError(f"Dynamic seed {seed} validation evidence changed")
        paths[f"dynamic/seed_{seed}/validation_predictions"] = validation_prediction_path
        validation_evidence[str(seed)] = dict(
            selection_sha256=frozen.sha256(selection_path),
            validation_predictions_sha256=frozen.sha256(validation_prediction_path),
        )
        for arm in dynamic.ARMS:
            if float(selection["arms"][arm]["threshold"]) != PRIMARY_THRESHOLD:
                raise ValueError("External evaluation requires fixed dynamic threshold 0.5")
            checkpoint_path = directory / f"{arm}.pt"
            if selection["arms"][arm].get("checkpoint_sha256") != frozen.sha256(checkpoint_path):
                raise ValueError(f"Dynamic seed {seed}/{arm} checkpoint differs from validation selection")
            paths[f"dynamic/seed_{seed}/{arm}"] = checkpoint_path
        paths[f"dynamic/seed_{seed}/selection"] = selection_path
    primary_selection = select_dynamic_primary(seed_selections)
    primary_selection["evidence"] = dict(
        summary_sha256=frozen.sha256(summary_path), seeds=validation_evidence,
    )
    source = Path(protocol["source"]).resolve()
    source_protocol = _json(source / "protocol.json")
    representation, frozen_paths = _frozen_representation_identity(source)
    if (
        protocol.get("source_protocol_sha256") != representation["source_protocol_sha256"]
        or protocol.get("adapter_sha256") != representation["adapter_sha256"]
    ):
        raise ValueError("Dynamic study frozen-representation identity changed")
    paths.update(frozen_paths)
    specification = dict(
        study=str(study), seeds=seeds, arms=list(dynamic.ARMS),
        primary_arm=primary_selection["selected_arm"], primary_selection=primary_selection,
        ensemble="arithmetic mean probability over every registered head seed",
        selected_layers=protocol["selected_layers"], threshold=PRIMARY_THRESHOLD,
        frozen_source=str(source), adapter_sha256=source_protocol["adapter_sha256"],
        representation=representation, evaluation_provenance=provenance_spec,
    )
    return specification, paths


def discover_oof_study(study: Path) -> tuple[dict, dict[str, Path]]:
    study = Path(study).resolve()
    protocol = oof.load_protocol(study)
    barrier = oof.verify_test_barrier(study)
    selection_path = study / "selection.json"
    selection = _json(selection_path)
    protocol_path = study / "protocol.json"
    if selection.get("protocol_sha256") != frozen.sha256(protocol_path):
        raise ValueError("OOF selection belongs to another protocol")
    selected_arm = str(selection["selected_arm"])
    if (
        selected_arm not in oof.MATCHED_ARMS
        or barrier.get("selected_arm") != selected_arm
        or float(barrier.get("threshold")) != PRIMARY_THRESHOLD
    ):
        raise ValueError("OOF selected arm/threshold is invalid")
    if selection.get("train_candidate") not in ("fixed4", "selected4"):
        raise ValueError("OOF train candidate is outside the registered fixed/selected comparison")
    paths: dict[str, Path] = {
        "oof/protocol": study / "protocol.json",
        "oof/test_barrier": study / "test_barrier.json",
        "oof/selection": selection_path,
        "oof/learned_layers": study / "learned_layers.json",
        "oof/calibration_selection": study / "calibration_selection.json",
    }
    arm_models = {}
    for arm in oof.ALL_ARMS:
        model_path = study / "models" / f"{arm}.json"
        paths[f"oof/model/{arm}"] = model_path
        arm_models[arm] = _json(model_path)
    selected_models = selection.get("arms", {})
    if set(selected_models) != set(oof.ALL_ARMS):
        raise ValueError("OOF selection has an incomplete model set")
    changed_models = [arm for arm in oof.ALL_ARMS if selected_models[arm] != arm_models[arm]]
    if changed_models:
        raise ValueError(f"OOF model files differ from selection: {changed_models}")
    learned = _json(study / "learned_layers.json")
    if any(learned.get(key) != value for key, value in selection.get("learned_layers", {}).items()):
        raise ValueError("OOF learned-layer artifact differs from selection")
    calibration_selection = _json(study / "calibration_selection.json")
    if calibration_selection != selection.get("calibration_selection"):
        raise ValueError("OOF calibration-selection artifact differs from selection")
    if calibration_selection.get("selection_source") != OOF_CALIBRATION_SELECTION_SOURCE:
        raise ValueError("OOF calibrator family was not selected by strict nested train outer-OOF")
    checkpoint = _resolve(selection.get("full_probe_checkpoint", ""), study)
    if (
        not checkpoint.is_file()
        or selection.get("full_probe_checkpoint_sha256") != frozen.sha256(checkpoint)
    ):
        raise ValueError("OOF deployment probe checkpoint differs from selection")
    bundle_path, bundle_manifest_path = oof._bundle_locations(study, protocol)
    if selection.get("oof_bundle_sha256") != frozen.sha256(bundle_path):
        raise ValueError("OOF deployment bundle differs from selection")
    bundle_manifest = _json(bundle_manifest_path)
    manifest_checkpoint = Path(bundle_manifest.get("full_checkpoint", ""))
    if not manifest_checkpoint.is_absolute():
        manifest_checkpoint = (bundle_manifest_path.parent / manifest_checkpoint).resolve()
    if (
        manifest_checkpoint != checkpoint
        or bundle_manifest.get("full_checkpoint_sha256") != frozen.sha256(checkpoint)
    ):
        raise ValueError("OOF bundle manifest and selection use different deployment checkpoints")
    paths["oof/full_probe_checkpoint"] = checkpoint
    for number, path in enumerate(oof._barrier_artifacts(study, protocol)):
        paths.setdefault(f"oof/barrier_artifact_{number:02d}", path)
    calibration_family = selected_calibration_family(selection, arm_models)
    representation, _ = _frozen_representation_identity(Path(protocol["source"]))
    if (
        protocol.get("source_protocol_sha256") != representation["source_protocol_sha256"]
        or protocol.get("source_adapter_sha256") != representation["adapter_sha256"]
        or int(protocol.get("hidden_size", -1)) != representation["hidden_size"]
    ):
        raise ValueError("OOF study frozen-representation identity changed")
    candidate_layers = tuple(map(int, protocol["candidate_layers"]))
    if 32 not in candidate_layers or any(layer not in frozen.ALL_LAYERS for layer in candidate_layers):
        raise ValueError("OOF candidate layers do not belong to the 32-layer representation")
    specification = dict(
        study=str(study), arms=list(oof.ALL_ARMS), selected_arm=selected_arm,
        learned_layers=selection["learned_layers"]["layers"], threshold=PRIMARY_THRESHOLD,
        calibrated=True,
        calibration=dict(
            family=calibration_family,
            family_selection_source=calibration_selection["selection_source"],
            fit_source=OOF_CALIBRATOR_FIT_SOURCE,
            external_refit=False,
        ),
        candidate_layers=list(candidate_layers), representation=representation,
        primary_selection=dict(
            selected_arm=selected_arm,
            rule="locked train group-OOF choice followed by validation pass/fallback gate",
            selection_sha256=frozen.sha256(selection_path),
            barrier_sha256=frozen.sha256(study / "test_barrier.json"),
        ),
    )
    return specification, paths


def create_evaluation_lock(
    output: Path,
    registry_snapshot: Mapping[str, Any],
    artifacts: Mapping[str, Path | str],
    evaluation: Mapping[str, Any],
) -> dict:
    """Pure lock constructor; never deserializes a model or produces scores."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    value = dict(
        schema="hanguard_external_evaluation_lock_v1",
        locked_before_prediction=True,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        registry=dict(
            path=registry_snapshot["registry"],
            manifest_sha256=registry_snapshot["manifest_sha256"],
            registry_lock_sha256=registry_snapshot["registry_lock_sha256"],
            datasets=registry_snapshot["datasets"],
        ),
        artifacts=_artifact_hashes(artifacts),
        evaluation=dict(evaluation),
        rule=(
            "All data selection, source bytes, adapters, checkpoints, layer choices, calibration, "
            "ensembling, and threshold 0.5 were frozen before any external model forward."
        ),
    )
    path = output / "evaluation_lock.json"
    if path.exists():
        prior = _json(path)
        comparable = ("schema", "locked_before_prediction", "registry", "artifacts", "evaluation")
        if any(prior.get(key) != value.get(key) for key in comparable):
            raise ValueError("Existing external evaluation lock differs")
        return prior
    frozen.dump(path, value)
    return value


def verify_evaluation_lock(output: Path) -> dict:
    output = Path(output).resolve()
    path = output / "evaluation_lock.json"
    if not path.is_file():
        raise FileNotFoundError("External evaluation must be locked before model access")
    value = _json(path)
    if value.get("schema") != "hanguard_external_evaluation_lock_v1" or value.get("locked_before_prediction") is not True:
        raise ValueError("Invalid external evaluation lock")
    registry = validate_registry(Path(value["registry"]["path"]))
    if registry["manifest_sha256"] != value["registry"]["manifest_sha256"]:
        raise ValueError("Locked external registry manifest changed")
    if registry["registry_lock_sha256"] != value["registry"]["registry_lock_sha256"]:
        raise ValueError("Locked external registry lock changed")
    if registry["datasets"] != value["registry"]["datasets"]:
        raise ValueError("Locked external dataset snapshot changed")
    for name, record in value.get("artifacts", {}).items():
        artifact = Path(record["path"])
        if not artifact.is_file() or artifact.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Locked artifact extent changed: {name}")
        if frozen.sha256(artifact) != record["sha256"]:
            raise ValueError(f"Locked artifact hash changed: {name}")
    return value


def lock_evaluation(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    registry = validate_registry(args.registry)
    consumed_path = Path(args.registry) / "consumed.json"
    if consumed_path.exists():
        consumed = _json(consumed_path)
        existing_lock = output / "evaluation_lock.json"
        if (
            consumed.get("evaluation_output") != str(output)
            or not existing_lock.is_file()
            or consumed.get("evaluation_lock_sha256") != frozen.sha256(existing_lock)
        ):
            raise RuntimeError("External registry was consumed by another frozen evaluation")
        return verify_evaluation_lock(output)
    dynamic_spec, dynamic_paths = discover_dynamic_study(
        args.dynamic_study, getattr(args, "dynamic_primary_arm", "auto")
    )
    oof_spec, oof_paths = discover_oof_study(args.oof_study)
    shared_representation = assert_same_frozen_representation(dynamic_spec, oof_spec)
    artifacts = {
        "implementation/external_holdout_evaluation": Path(__file__),
        "implementation/dynamic_gate_diagnostics": Path(dynamic.__file__),
        "implementation/oof_probe_logit_stack": Path(oof.__file__),
        "implementation/frozen_lora_layer_diagnostics": Path(frozen.__file__),
        **dynamic_paths,
        **oof_paths,
    }
    evaluation = dict(
        datasets=sorted(registry["datasets"]), prompt_mode="direct source-language prompt; no translation",
        dataset_roles={
            name: _dataset_records(registry["manifest"])[name].get("role")
            for name in sorted(registry["datasets"])
        },
        threshold=PRIMARY_THRESHOLD, dynamic=dynamic_spec, oof=oof_spec,
        shared_frozen_representation=shared_representation,
        extraction=dict(
            token_budget=int(args.token_budget), max_micro=int(args.max_micro),
            pad_multiple=int(args.pad_multiple), all_layers=list(frozen.ALL_LAYERS), truncation=False,
        ),
        device=str(args.device),
        prediction_batch_size=int(args.prediction_batch_size),
        primary_comparisons=[
            "dynamic primary probability-mean ensemble vs dynamic repeat probability-mean ensemble",
            "OOF selected arm vs OOF repeat32",
            "OOF selected arm vs dynamic primary probability-mean ensemble",
        ],
        repetitions=int(args.repetitions), analysis_seed=int(args.analysis_seed),
    )
    return create_evaluation_lock(output, registry, artifacts, evaluation)


def _finite_state_dict(state: Mapping[str, Any], name: str) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} state_dict is missing")
    for key, value in state.items():
        tensor = torch.as_tensor(value)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains nonfinite tensor: {key}")


def _preflight_dynamic_deployment(study: Path, representation: Mapping[str, Any]) -> dict:
    """CPU-only semantic check of every locked dynamic deployment checkpoint."""
    study = Path(study).resolve()
    protocol = dynamic.load_protocol(study)
    architecture = protocol["architecture"]
    hidden_size = int(representation["hidden_size"])
    records = {}
    for seed in map(int, protocol["training"]["seeds"]):
        directory = study / "heads" / f"seed_{seed}"
        selection = _json(directory / "selection.json")
        for arm in dynamic.ARMS:
            path = directory / f"{arm}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            expected = dict(
                arm=arm,
                seed=seed,
                selected_layers=list(protocol["selected_layers"]),
                input_layers=list(dynamic.arm_layers(arm, protocol["selected_layers"])),
                hidden_size=hidden_size,
                width=int(architecture["width"]),
                gate_width=int(architecture["gate_width"]),
                best_epoch=int(selection["arms"][arm]["best_epoch"]),
                protocol_sha256=frozen.sha256(study / "protocol.json"),
            )
            changed = [key for key, value in expected.items() if checkpoint.get(key) != value]
            selected_bce = float(selection["arms"][arm]["validation_bce"])
            checkpoint_bce = float(checkpoint.get("best_validation_bce", float("nan")))
            if changed or not math.isfinite(checkpoint_bce) or checkpoint_bce != selected_bce:
                raise ValueError(
                    f"Dynamic deployment checkpoint differs from validation selection: "
                    f"seed={seed}, arm={arm}, fields={changed}"
                )
            _finite_state_dict(checkpoint.get("state_dict", {}), f"dynamic seed={seed}/{arm}")
            head = dynamic.DynamicLayerGateHead(
                hidden_size=hidden_size,
                width=int(architecture["width"]),
                gate_width=int(architecture["gate_width"]),
                layer_count=int(architecture["layer_count"]),
                dropout=float(architecture["dropout"]),
                fusion_init_scale=float(architecture["fusion_init_scale"]),
            )
            head.load_state_dict(checkpoint["state_dict"], strict=True)
            parameters = sum(parameter.numel() for parameter in head.parameters())
            if parameters != int(selection["arms"][arm]["parameters"]):
                raise ValueError(f"Dynamic deployment capacity changed: seed={seed}, arm={arm}")
            records[f"{seed}/{arm}"] = dict(
                checkpoint_sha256=frozen.sha256(path),
                best_epoch=expected["best_epoch"],
                best_validation_bce=selected_bce,
                parameters=parameters,
            )
    return dict(
        protocol_sha256=frozen.sha256(study / "protocol.json"),
        hidden_size=hidden_size,
        checkpoints=records,
    )


def _preflight_oof_deployment(study: Path, representation: Mapping[str, Any]) -> dict:
    """CPU-only semantic check of OOF stack JSON and full probe checkpoint."""
    study = Path(study).resolve()
    protocol = oof.load_protocol(study)
    selection = _json(study / "selection.json")
    learned_layers = list(map(int, selection["learned_layers"]["layers"]))
    expected_layers = {
        "single32": [32],
        "repeat32": [32, 32, 32, 32],
        "fixed4": list(map(int, protocol["fixed_layers"])),
        "selected4": learned_layers,
    }
    model_records = {}
    for arm in oof.ALL_ARMS:
        model_path = study / "models" / f"{arm}.json"
        model = _json(model_path)
        layers = expected_layers[arm]
        if model.get("layers") != layers:
            raise ValueError(f"OOF deployment model layer contract changed: {arm}")
        stack = model.get("stack", {})
        vectors = {
            key: np.asarray(stack.get(key, ()), dtype=np.float64)
            for key in ("mean", "scale", "coef")
        }
        if any(value.shape != (len(layers),) for value in vectors.values()):
            raise ValueError(f"OOF deployment stack width changed: {arm}")
        if (
            not all(np.isfinite(value).all() for value in vectors.values())
            or np.any(vectors["scale"] <= 0)
            or not math.isfinite(float(stack.get("intercept", float("nan"))))
        ):
            raise FloatingPointError(f"OOF deployment stack is invalid: {arm}")
        calibrator = model.get("calibrator", {})
        slope = float(calibrator.get("slope", float("nan")))
        intercept = float(calibrator.get("intercept", float("nan")))
        if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
            raise FloatingPointError(f"OOF deployment calibrator is invalid: {arm}")
        probe = np.zeros((2, len(layers)), dtype=np.float64)
        probability = oof._predict_arm(model, probe)
        if probability.shape != (2,) or not np.isfinite(probability).all():
            raise FloatingPointError(f"OOF deployment prediction smoke test failed: {arm}")
        model_records[arm] = dict(
            model_sha256=frozen.sha256(model_path), layers=layers,
            calibration_method=calibrator["method"],
            calibrator_fit_source=calibrator["fit_source"],
        )

    checkpoint_path = _resolve(selection["full_probe_checkpoint"], study)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    candidate_layers = list(map(int, protocol["candidate_layers"]))
    expected_identity = dict(
        split="train",
        rows=int(protocol["train_metadata"]["rows"]),
        protocol_sha256=frozen.sha256(study / "protocol.json"),
        base_id_sha256=protocol["train_metadata"]["ordered_base_id_sha256"],
    )
    changed = []
    if list(map(int, checkpoint.get("candidate_layers", ()))) != candidate_layers:
        changed.append("candidate_layers")
    if int(checkpoint.get("hidden_size", -1)) != int(representation["hidden_size"]):
        changed.append("hidden_size")
    if checkpoint.get("identity") != expected_identity:
        changed.append("identity")
    if changed:
        raise ValueError(f"OOF deployment checkpoint identity changed: {changed}")
    _finite_state_dict(checkpoint.get("pool_state", {}), "OOF pool")
    _finite_state_dict(checkpoint.get("repeat_state", {}), "OOF repeat")
    pool = oof.BranchProbes(
        int(representation["hidden_size"]), len(candidate_layers), int(protocol["probes"]["seed"]),
        [oof.layer_slot(layer) for layer in candidate_layers],
    )
    repeat = oof.BranchProbes(
        int(representation["hidden_size"]), 4, int(protocol["probes"]["seed"]), [0, 1, 2, 3]
    )
    pool.load_state_dict(checkpoint["pool_state"], strict=True)
    repeat.load_state_dict(checkpoint["repeat_state"], strict=True)
    return dict(
        protocol_sha256=frozen.sha256(study / "protocol.json"),
        full_probe_checkpoint_sha256=frozen.sha256(checkpoint_path),
        hidden_size=int(representation["hidden_size"]),
        models=model_records,
    )


def preflight_deployment_artifacts(
    output: Path, lock: Mapping[str, Any] | None = None
) -> dict:
    """Validate locked model semantics on CPU before consuming a holdout."""
    output = Path(output).resolve()
    lock = verify_evaluation_lock(output) if lock is None else dict(lock)
    representation = lock["evaluation"]["shared_frozen_representation"]
    value = dict(
        schema="hanguard_external_deployment_preflight_v1",
        evaluation_lock_sha256=frozen.sha256(output / "evaluation_lock.json"),
        dynamic=_preflight_dynamic_deployment(
            Path(lock["evaluation"]["dynamic"]["study"]), representation
        ),
        oof=_preflight_oof_deployment(Path(lock["evaluation"]["oof"]["study"]), representation),
        state="cpu_checkpoint_and_schema_validation_passed",
    )
    path = output / "deployment_preflight.json"
    if path.is_file():
        if _json(path) != value:
            raise ValueError("Existing external deployment preflight differs")
    else:
        frozen.dump(path, value)
    return value


def mark_consumed(output: Path) -> dict:
    """Create immutable access markers before the first external model forward."""
    output = Path(output).resolve()
    lock = verify_evaluation_lock(output)
    registry = Path(lock["registry"]["path"])
    core = dict(
        state="consumed_model_access_started",
        evaluation_output=str(output),
        evaluation_lock_sha256=frozen.sha256(output / "evaluation_lock.json"),
        manifest_sha256=lock["registry"]["manifest_sha256"],
        datasets=sorted(lock["registry"]["datasets"]),
    )
    paths = (output / "access_started.json", registry / "consumed.json")
    existing = [path for path in paths if path.exists()]
    if existing:
        for path in existing:
            prior = _json(path)
            if any(prior.get(key) != value for key, value in core.items()):
                raise RuntimeError(f"External holdout was consumed by another evaluation: {path}")
        # Repair a missing mirror only for the exact same frozen evaluation.
        timestamp = _json(existing[0]).get("started_at")
    else:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    value = dict(**core, started_at=timestamp)
    for path in paths:
        if not path.exists():
            frozen.dump(path, value)
    return value


def _cache_paths(output: Path, dataset: str) -> tuple[Path, Path, Path, Path, Path]:
    directory = Path(output) / "features"
    return (
        directory / f"{dataset}.npy",
        directory / f"{dataset}.npy.part",
        directory / f"{dataset}.progress.json",
        directory / f"{dataset}.manifest.json",
        directory / f"{dataset}.metadata.parquet",
    )


def _tokenize_prompts(tokenizer, prompts: Sequence[str], maximum: int) -> tuple[list[list[int]], np.ndarray]:
    encoded = [
        tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        for prompt in prompts
    ]
    lengths = np.asarray([len(value) for value in encoded], dtype=np.int32)
    if not len(lengths) or (lengths < 1).any():
        raise ValueError("Every external prompt must tokenize to at least one token")
    if int(lengths.max()) > int(maximum):
        raise ValueError(f"External prompt exceeds registered model context: {lengths.max()} > {maximum}")
    return encoded, lengths


def _batch(encoded: Sequence[Sequence[int]], indices: np.ndarray, tokenizer, pad_multiple: int) -> dict:
    batch = tokenizer.pad(
        [
            {"input_ids": encoded[int(index)], "attention_mask": [1] * len(encoded[int(index)])}
            for index in indices
        ],
        padding=True,
        pad_to_multiple_of=pad_multiple,
        return_tensors="pt",
    )
    expected = torch.as_tensor([len(encoded[int(index)]) for index in indices], dtype=torch.long)
    if not torch.equal(batch["attention_mask"].sum(1).cpu(), expected):
        raise AssertionError("External tokenizer padding changed prompt lengths")
    return batch.to("cuda")


def load_external_cache(
    output: Path, dataset: str, lock: Mapping[str, Any] | None = None
) -> tuple[np.memmap, pd.DataFrame, dict]:
    output = Path(output).resolve()
    lock = verify_evaluation_lock(output) if lock is None else lock
    final, _, _, manifest_path, metadata_path = _cache_paths(output, dataset)
    if not final.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing extracted external cache for {dataset}")
    manifest = _json(manifest_path)
    expected_dataset = lock["registry"]["datasets"].get(dataset)
    if expected_dataset is None or manifest.get("dataset_snapshot") != expected_dataset:
        raise ValueError(f"{dataset} cache belongs to another registry snapshot")
    if manifest.get("evaluation_lock_sha256") != frozen.sha256(output / "evaluation_lock.json"):
        raise ValueError(f"{dataset} cache evaluation lock changed")
    if manifest.get("complete") is not True or manifest.get("sha256") != frozen.sha256(final):
        raise ValueError(f"{dataset} external feature cache changed")
    if manifest.get("metadata_sha256") != frozen.sha256(metadata_path):
        raise ValueError(f"{dataset} external metadata changed")
    metadata = pd.read_parquet(metadata_path)
    if "prompt" in metadata.columns:
        raise ValueError("External cache metadata must not copy prompt text")
    if metadata.base_id.astype(str).tolist() != sorted(metadata.base_id.astype(str).tolist()):
        raise ValueError(f"{dataset} metadata order changed")
    expected_shape = (len(metadata), len(frozen.ALL_LAYERS), int(manifest["hidden_size"]))
    validation = frozen.validate_feature_cache(
        final, expected_shape, np.float16, manifest["ordered_base_id_sha256"],
        metadata.base_id.astype(str).tolist(),
    )
    if validation["bytes"] != manifest["bytes"]:
        raise ValueError(f"{dataset} cache extent changed")
    return np.load(final, mmap_mode="r", allow_pickle=False), metadata, manifest


def extract(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    lock = verify_evaluation_lock(output)
    if str(args.device) != lock["evaluation"]["device"]:
        raise ValueError("External extraction device differs from the frozen evaluation lock")
    preflight_deployment_artifacts(output, lock)
    mark_consumed(output)
    requested = list(lock["evaluation"]["datasets"]) if not args.dataset else list(args.dataset)
    if set(requested) - set(lock["registry"]["datasets"]):
        raise ValueError("Requested dataset was not frozen in the evaluation lock")
    pending = []
    for dataset in requested:
        final, temporary, progress_path, manifest_path, metadata_path = _cache_paths(output, dataset)
        completed = (final.is_file(), manifest_path.is_file(), metadata_path.is_file())
        if all(completed):
            load_external_cache(output, dataset, lock)
            continue
        if any(completed):
            raise ValueError(f"Incomplete final cache artifacts for {dataset}")
        if temporary.exists() != progress_path.exists():
            raise ValueError(f"Incomplete resumable cache artifacts for {dataset}")
        pending.append(dataset)
    if not pending:
        return dict(datasets=requested, state="already_complete")

    frozen_source = Path(lock["evaluation"]["dynamic"]["frozen_source"])
    source_protocol = frozen.load_protocol(frozen_source)
    model, backbone, tokenizer = frozen._load_frozen_model(source_protocol)
    hidden_size = int(model.config.text_config.hidden_size)
    capture = frozen.LastTokenBlockCapture(backbone)
    extraction_settings = lock["evaluation"]["extraction"]
    from scripts.hanguard.repaired_study import microbatches

    try:
        for dataset in pending:
            metadata, prompts, reconstruction = recover_prompts(
                Path(lock["registry"]["path"]), dataset
            )
            encoded, lengths = _tokenize_prompts(tokenizer, prompts, source_protocol["max_tokens"])
            metadata = metadata.copy()
            metadata["prompt_tokens"] = lengths
            final, temporary, progress_path, manifest_path, metadata_path = _cache_paths(output, dataset)
            final.parent.mkdir(parents=True, exist_ok=True)
            shape = (len(metadata), len(frozen.ALL_LAYERS), hidden_size)
            identity = dict(
                dataset=dataset,
                shape=list(shape),
                dtype="float16",
                evaluation_lock_sha256=frozen.sha256(output / "evaluation_lock.json"),
                dataset_snapshot=lock["registry"]["datasets"][dataset],
                adapter_sha256=lock["evaluation"]["dynamic"]["adapter_sha256"],
                representation="raw block output at final valid token",
                prompt_mode=lock["evaluation"]["prompt_mode"],
                truncation=False,
            )
            order = np.argsort(lengths, kind="stable")
            groups = microbatches(
                order,
                lengths,
                int(extraction_settings["token_budget"]),
                int(extraction_settings["max_micro"]),
                int(extraction_settings["pad_multiple"]),
            )
            start = 0
            if temporary.exists():
                progress = _json(progress_path)
                if progress.get("identity") != identity:
                    raise ValueError(f"{dataset} extraction resume identity changed")
                start = int(progress["next_group"])
                cache = np.lib.format.open_memmap(
                    temporary, mode="r+", dtype=np.float16, shape=shape
                )
            else:
                cache = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=np.float16, shape=shape
                )
            began = time.monotonic()
            for number in range(start, len(groups)):
                indices = groups[number]
                values = capture(
                    _batch(encoded, indices, tokenizer, extraction_settings["pad_multiple"])
                ).numpy()
                if values.shape != (len(indices), len(frozen.ALL_LAYERS), hidden_size):
                    raise ValueError("Captured external feature shape changed")
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite external frozen features")
                cache[indices] = values
                if (number + 1) % max(1, args.status_every) == 0 or number + 1 == len(groups):
                    cache.flush()
                    frozen.dump(
                        progress_path,
                        dict(
                            identity=identity, next_group=number + 1, total_groups=len(groups),
                            rows=sum(len(group) for group in groups[: number + 1]),
                            elapsed_seconds=time.monotonic() - began,
                        ),
                    )
                    print(
                        json.dumps(
                            dict(state="external_extract", dataset=dataset,
                                 groups=number + 1, total_groups=len(groups))
                        ),
                        flush=True,
                    )
            cache.flush()
            del cache
            metadata.to_parquet(metadata_path, index=False)
            temporary.replace(final)
            progress_path.unlink(missing_ok=True)
            ordered = frozen.ordered_base_id_sha256(metadata.base_id.astype(str).tolist())
            validation = frozen.validate_feature_cache(
                final, shape, np.float16, ordered, metadata.base_id.astype(str).tolist()
            )
            manifest = dict(
                **identity,
                complete=True,
                rows=len(metadata), layers=len(frozen.ALL_LAYERS), hidden_size=hidden_size,
                bytes=validation["bytes"], sha256=frozen.sha256(final),
                metadata_sha256=frozen.sha256(metadata_path),
                ordered_base_id_sha256=ordered,
                finite_rows_checked=validation["finite_rows_checked"],
                tokenizer_lengths=dict(minimum=int(lengths.min()), maximum=int(lengths.max())),
                reconstruction=reconstruction,
                prompt_storage="not copied; source_file + source_row with exact hashes",
            )
            frozen.dump(manifest_path, manifest)
    finally:
        capture.close()
        del model, backbone
        torch.cuda.empty_cache()
    return dict(datasets=requested, state="complete")


def group_equal_weights(groups: Sequence[Any]) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Nonempty one-dimensional groups required")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    return weights


def _metrics(
    labels: Sequence[int], probabilities: Sequence[float], sample_weight: Sequence[float] | None = None
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Aligned nonempty finite external predictions required")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("External labels must be binary")
    weights = np.ones(len(labels), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("External metric weights must be aligned, finite, and positive")
    prediction = probabilities >= PRIMARY_THRESHOLD
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    positive, negative = labels == 1, labels == 0
    correct = prediction == labels
    tp = float(weights[positive & prediction].sum())
    tn = float(weights[negative & ~prediction].sum())
    fp = float(weights[negative & prediction].sum())
    fn = float(weights[positive & ~prediction].sum())
    result = dict(
        rows=int(len(labels)),
        threshold=PRIMARY_THRESHOLD,
        accuracy=float(np.average(correct, weights=weights)),
        balanced_accuracy=float(
            balanced_accuracy_score(labels, prediction, sample_weight=weights)
        ),
        f1=float(2 * tp / max(1e-12, 2 * tp + fp + fn)),
        recall=float(tp / max(1e-12, tp + fn)),
        specificity=float(tn / max(1e-12, tn + fp)),
        fpr=float(fp / max(1e-12, fp + tn)),
        bce=float(
            np.average(
                -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)),
                weights=weights,
            )
        ),
        brier=float(np.average(np.square(probabilities - labels), weights=weights)),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities, sample_weight=weights))
        if len(np.unique(labels)) == 2 else None
    )
    result["auprc"] = (
        float(average_precision_score(labels, probabilities, sample_weight=weights))
        if len(np.unique(labels)) == 2 else None
    )
    return result


def collapse_groups(
    labels: Sequence[int], probabilities: Sequence[float], groups: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "probability": np.asarray(probabilities, dtype=np.float64),
            "group": np.asarray(groups).astype(str),
        }
    )
    if frame.group.isna().any() or (frame.groupby("group").label.nunique() != 1).any():
        raise ValueError("Groups must be nonempty and label-consistent")
    collapsed = frame.groupby("group", sort=True).agg(label=("label", "first"), probability=("probability", "mean"))
    return (
        collapsed.label.to_numpy(dtype=np.int64),
        collapsed.probability.to_numpy(dtype=np.float64),
        collapsed.index.to_numpy(dtype=str),
    )


def external_metrics(
    labels: Sequence[int], probabilities: Sequence[float], groups: Sequence[Any]
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    collapsed_y, collapsed_p, collapsed_groups = collapse_groups(labels, probabilities, groups)
    return dict(
        row=_metrics(labels, probabilities),
        group_equal_rows=_metrics(labels, probabilities, group_equal_weights(groups)),
        group_collapsed=_metrics(collapsed_y, collapsed_p),
        groups=int(len(collapsed_groups)),
        duplicate_rows_beyond_first=int(len(labels) - len(collapsed_groups)),
    )


def _collapsed_pair(
    labels: Sequence[int], probability_a: Sequence[float], probability_b: Sequence[float], groups: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.int64),
            "a": np.asarray(probability_a, dtype=np.float64),
            "b": np.asarray(probability_b, dtype=np.float64),
            "group": np.asarray(groups).astype(str),
        }
    )
    if not np.isfinite(frame[["a", "b"]].to_numpy()).all():
        raise ValueError("Paired probabilities must be finite")
    if (frame.groupby("group").label.nunique() != 1).any():
        raise ValueError("Paired groups contain conflicting labels")
    collapsed = frame.groupby("group", sort=True).agg(
        label=("label", "first"), a=("a", "mean"), b=("b", "mean")
    )
    return (
        collapsed.label.to_numpy(dtype=np.int64),
        collapsed.a.to_numpy(dtype=np.float64),
        collapsed.b.to_numpy(dtype=np.float64),
        collapsed.index.to_numpy(dtype=str),
    )


def paired_group_inference(
    labels: Sequence[int],
    probability_a: Sequence[float],
    probability_b: Sequence[float],
    groups: Sequence[Any],
    repetitions: int = 2000,
    seed: int = 42,
) -> dict:
    """Equal-group paired bootstrap/permutation at fixed threshold 0.5."""
    if repetitions < 1:
        raise ValueError("Positive paired repetitions required")
    y, a, b, group_names = _collapsed_pair(labels, probability_a, probability_b, groups)
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("Paired inference requires both classes across groups")
    pred_a, pred_b = a >= PRIMARY_THRESHOLD, b >= PRIMARY_THRESHOLD
    correct_delta = (pred_a == y).astype(np.float64) - (pred_b == y).astype(np.float64)
    # Direct clipped BCE avoids an unstable logit conversion at probabilities 0/1.
    clipped_a, clipped_b = np.clip(a, 1e-12, 1 - 1e-12), np.clip(b, 1e-12, 1 - 1e-12)
    bce_delta = (
        -(y * np.log(clipped_a) + (1 - y) * np.log1p(-clipped_a))
        + (y * np.log(clipped_b) + (1 - y) * np.log1p(-clipped_b))
    )
    def balanced(delta: np.ndarray, indices: np.ndarray | None = None) -> float:
        yy = y if indices is None else y[indices]
        dd = delta if indices is None else delta[indices]
        if not (np.any(yy == 0) and np.any(yy == 1)):
            return float("nan")
        return float(0.5 * (dd[yy == 0].mean() + dd[yy == 1].mean()))

    observed = dict(
        accuracy=float(correct_delta.mean()),
        balanced_accuracy=balanced(correct_delta),
        bce=float(bce_delta.mean()),
        auroc=float(roc_auc_score(y, a) - roc_auc_score(y, b)),
    )
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty(repetitions, dtype=np.float64)
        for name in ("accuracy", "balanced_accuracy", "bce")
    }
    null = {name: np.empty(repetitions, dtype=np.float64) for name in samples}
    cursor = 0
    while cursor < repetitions:
        size = min(16, repetitions - cursor)
        draws = rng.integers(0, len(y), size=(size, len(y)))
        signs = rng.integers(0, 2, size=(size, len(y)), dtype=np.int8) * 2 - 1
        for offset in range(size):
            indices = draws[offset]
            samples["accuracy"][cursor + offset] = correct_delta[indices].mean()
            samples["balanced_accuracy"][cursor + offset] = balanced(correct_delta, indices)
            samples["bce"][cursor + offset] = bce_delta[indices].mean()
            signed_correct = correct_delta * signs[offset]
            signed_bce = bce_delta * signs[offset]
            null["accuracy"][cursor + offset] = signed_correct.mean()
            null["balanced_accuracy"][cursor + offset] = balanced(signed_correct)
            null["bce"][cursor + offset] = signed_bce.mean()
        cursor += size
    inference = {}
    for name in samples:
        finite = samples[name][np.isfinite(samples[name])]
        finite_null = null[name][np.isfinite(null[name])]
        inference[name] = dict(
            delta=observed[name],
            ci95=[float(value) for value in np.quantile(finite, [0.025, 0.975])],
            permutation_two_sided_p=float(
                (1 + (np.abs(finite_null) >= abs(observed[name]) - 1e-15).sum())
                / (len(finite_null) + 1)
            ),
        )
    a_correct, b_correct = pred_a == y, pred_b == y
    return dict(
        groups=int(len(y)), repetitions=int(repetitions), seed=int(seed), threshold=PRIMARY_THRESHOLD,
        delta_definition="candidate minus baseline; negative BCE means candidate improvement",
        observed=observed, inference=inference,
        four_cells=dict(
            both_correct=int((a_correct & b_correct).sum()),
            a_only_correct=int((a_correct & ~b_correct).sum()),
            b_only_correct=int((~a_correct & b_correct).sum()),
            both_wrong=int((~a_correct & ~b_correct).sum()),
        ),
        mcnemar_exact=frozen.mcnemar_exact(y, pred_a, pred_b),
        group_id_sha256=frozen.ordered_base_id_sha256(group_names.tolist()),
    )


def _require_consumed(output: Path) -> dict:
    output = Path(output).resolve()
    lock = verify_evaluation_lock(output)
    preflight_deployment_artifacts(output, lock)
    expected = frozen.sha256(output / "evaluation_lock.json")
    for path in (output / "access_started.json", Path(lock["registry"]["path"]) / "consumed.json"):
        if not path.is_file():
            raise RuntimeError("External access marker is missing; run extract first")
        value = _json(path)
        if value.get("evaluation_lock_sha256") != expected or value.get("state") != "consumed_model_access_started":
            raise RuntimeError(f"External access marker differs: {path}")
    return lock


def verify_completed_evaluation(output: Path) -> dict:
    """Verify results, report, and every prediction artifact before reuse."""
    output = Path(output).resolve()
    lock = verify_evaluation_lock(output)
    result_path = output / "results.json"
    completion_path = output / "completion.json"
    report_path = output / "report.md"
    if not result_path.is_file() or not completion_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("External evaluation completion artifacts are incomplete")
    completion = _json(completion_path)
    if (
        completion.get("schema") != "hanguard_external_holdout_completion_v1"
        or completion.get("state") != "complete_no_retuning_authorized"
    ):
        raise ValueError("External evaluation completion state is invalid")
    lock_sha256 = frozen.sha256(output / "evaluation_lock.json")
    if completion.get("evaluation_lock_sha256") != lock_sha256:
        raise ValueError("Completed evaluation belongs to another lock")
    if completion.get("results_sha256") != frozen.sha256(result_path):
        raise ValueError("Completed external results changed")
    if completion.get("report_sha256") != frozen.sha256(report_path):
        raise ValueError("Completed external report changed")
    preflight_path = output / "deployment_preflight.json"
    if (
        not preflight_path.is_file()
        or completion.get("deployment_preflight_sha256") != frozen.sha256(preflight_path)
    ):
        raise ValueError("Completed external deployment preflight changed")
    result = _json(result_path)
    if (
        result.get("schema") != "hanguard_external_holdout_results_v1"
        or float(result.get("threshold", float("nan"))) != PRIMARY_THRESHOLD
        or result.get("evaluation_lock_sha256") != lock_sha256
        or result.get("registry_manifest_sha256") != lock["registry"]["manifest_sha256"]
    ):
        raise ValueError("External results identity differs from the frozen evaluation")
    locked_datasets = set(lock["evaluation"]["datasets"])
    if set(result.get("datasets", {})) != locked_datasets:
        raise ValueError("External results dataset set differs from the frozen evaluation")
    expected_predictions = completion.get("prediction_sha256", {})
    if set(expected_predictions) != set(result.get("datasets", {})):
        raise ValueError("Completed prediction artifact set changed")
    for dataset, values in result["datasets"].items():
        path = (output / values["predictions"]).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError(f"Completed prediction path is invalid: {dataset}")
        actual = frozen.sha256(path)
        if actual != values.get("predictions_sha256") or actual != expected_predictions.get(dataset):
            raise ValueError(f"Completed external predictions changed: {dataset}")
    return result


def _dynamic_predictions(
    study: Path,
    cache: np.memmap,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    study = Path(study)
    protocol = dynamic.load_protocol(study)
    selected = protocol["selected_layers"]
    probabilities: dict[str, np.ndarray] = {}
    gate_summaries = {}
    by_arm: dict[str, list[np.ndarray]] = {arm: [] for arm in dynamic.ARMS}
    for seed in protocol["training"]["seeds"]:
        directory = study / "heads" / f"seed_{seed}"
        heads = {
            arm: dynamic._load_head(directory / f"{arm}.pt", arm, seed, protocol, device)
            for arm in dynamic.ARMS
        }
        logits, weights = dynamic.predict(heads, cache, selected, batch_size, device)
        for arm in dynamic.ARMS:
            name = f"dynamic_seed_{seed}_{arm}"
            probability = frozen.sigmoid(logits[arm])
            probabilities[name] = probability
            by_arm[arm].append(probability)
            gate_summaries[name] = dynamic.gate_diagnostics(
                weights[arm], dynamic.arm_layers(arm, selected)
            )
        del heads
    for arm, values in by_arm.items():
        probabilities[f"dynamic_ensemble_{arm}"] = np.stack(values).mean(0)
    return probabilities, dict(gates=gate_summaries, seeds=protocol["training"]["seeds"])


def _oof_predictions(
    study: Path,
    cache: np.memmap,
) -> tuple[dict[str, np.ndarray], dict]:
    study = Path(study)
    protocol = oof.load_protocol(study)
    selection = _json(study / "selection.json")
    pool, repeat, checkpoint = oof._load_full_models(study, protocol)
    indices = np.arange(len(cache), dtype=np.int64)
    layers = tuple(protocol["candidate_layers"])
    layer_logits, repeat_logits = oof._predict_probe_pair(
        pool, repeat, cache, indices, layers, protocol["probes"]["batch_size"]
    )
    probabilities = {}
    for arm in oof.ALL_ARMS:
        model = _json(study / "models" / f"{arm}.json")
        features, _ = oof._arm_features(
            arm, layer_logits, repeat_logits, layers, selection["learned_layers"]["layers"]
        )
        probabilities[f"oof_{arm}"] = oof._predict_arm(model, features)
    del pool, repeat
    return probabilities, dict(
        selected_arm=selection["selected_arm"],
        learned_layers=selection["learned_layers"]["layers"],
        full_probe_checkpoint_sha256=frozen.sha256(checkpoint),
    )


def _category_metrics(
    metadata: pd.DataFrame, probabilities: np.ndarray
) -> dict[str, dict] | None:
    if "category_id" not in metadata:
        return None
    result = {}
    for category, group in metadata.reset_index(drop=True).groupby("category_id", sort=True):
        indices = group.index.to_numpy()
        result[str(category)] = external_metrics(
            group.label.to_numpy(dtype=np.int64), probabilities[indices],
            group.group_id.astype(str).to_numpy(),
        )
    return result


def evaluate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    lock = _require_consumed(output)
    if str(args.device) != lock["evaluation"]["device"]:
        raise ValueError("External prediction device differs from the frozen evaluation lock")
    result_path = output / "results.json"
    if result_path.is_file():
        return verify_completed_evaluation(output)
    datasets = lock["evaluation"]["datasets"]
    device = torch.device(args.device)
    batch_size = int(lock["evaluation"]["prediction_batch_size"])
    repetitions = int(lock["evaluation"]["repetitions"])
    analysis_seed = int(lock["evaluation"]["analysis_seed"])
    dynamic_study = Path(lock["evaluation"]["dynamic"]["study"])
    oof_study = Path(lock["evaluation"]["oof"]["study"])
    dynamic_primary = lock["evaluation"]["dynamic"]["primary_arm"]
    oof_selected = lock["evaluation"]["oof"]["selected_arm"]
    dataset_results = {}
    for dataset_index, dataset in enumerate(datasets):
        cache, metadata, cache_manifest = load_external_cache(output, dataset, lock)
        labels = metadata.label.to_numpy(dtype=np.int64)
        groups = metadata.group_id.astype(str).to_numpy()
        dynamic_probabilities, dynamic_info = _dynamic_predictions(
            dynamic_study, cache, batch_size, device
        )
        oof_probabilities, oof_info = _oof_predictions(oof_study, cache)
        probabilities = {**dynamic_probabilities, **oof_probabilities}
        prediction_frame = metadata.copy()
        system_metrics = {}
        for name, probability in probabilities.items():
            if len(probability) != len(metadata) or not np.isfinite(probability).all():
                raise FloatingPointError(f"Invalid external probability: {dataset}/{name}")
            prediction_frame[f"probability_{name}"] = probability
            prediction_frame[f"prediction_{name}"] = (
                probability >= PRIMARY_THRESHOLD
            ).astype(np.int8)
            system_metrics[name] = dict(
                metrics=external_metrics(labels, probability, groups),
                by_category=_category_metrics(metadata, probability),
            )
        prediction_path = output / "predictions" / f"{dataset}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(prediction_path, index=False)

        dynamic_candidate = f"dynamic_ensemble_{dynamic_primary}"
        dynamic_control = "dynamic_ensemble_dynamic_repeat_last"
        oof_candidate = f"oof_{oof_selected}"
        oof_control = "oof_repeat32"
        pairs = [
            ("dynamic_primary_vs_repeat", dynamic_candidate, dynamic_control),
            ("oof_selected_vs_repeat", oof_candidate, oof_control),
            ("oof_selected_vs_dynamic_primary", oof_candidate, dynamic_candidate),
        ]
        comparisons = {}
        for offset, (name, candidate, baseline) in enumerate(pairs):
            comparisons[name] = dict(
                candidate=candidate,
                baseline=baseline,
                **paired_group_inference(
                    labels, probabilities[candidate], probabilities[baseline], groups,
                    repetitions=repetitions,
                    seed=analysis_seed + dataset_index * 1009 + offset,
                ),
            )
        dataset_results[dataset] = dict(
            role=lock["evaluation"]["dataset_roles"].get(dataset),
            dataset_snapshot=lock["registry"]["datasets"][dataset],
            rows=int(len(metadata)), groups=int(metadata.group_id.nunique()),
            label_counts={str(key): int(value) for key, value in metadata.label.value_counts().sort_index().items()},
            cache_sha256=cache_manifest["sha256"],
            predictions=str(prediction_path.relative_to(output)),
            predictions_sha256=frozen.sha256(prediction_path),
            systems=system_metrics,
            primary_systems=dict(
                dynamic=dynamic_candidate, dynamic_control=dynamic_control,
                oof=oof_candidate, oof_control=oof_control,
            ),
            comparisons=comparisons,
            dynamic_info=dynamic_info,
            oof_info=oof_info,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(
        schema="hanguard_external_holdout_results_v1",
        threshold=PRIMARY_THRESHOLD,
        evaluation_lock_sha256=frozen.sha256(output / "evaluation_lock.json"),
        registry_manifest_sha256=lock["registry"]["manifest_sha256"],
        datasets=dataset_results,
        interpretation=dict(
            wildguard_en_official_test="primary external English binary holdout",
            injection_defense="secondary synthetic adversarial group stress; not a primary blind estimate",
        ),
        limitations=[
            "Repository-visible inherited labels are not a strictly hidden blind test.",
            "The English primary holdout mixes architecture behavior with language/domain shift.",
            "Injection-defense labels are generated or inherited and semantic independence is not established.",
            "No external result may be used to retune layers, checkpoints, calibration, thresholds, or ensemble membership.",
        ],
    )
    frozen.dump(result_path, result)
    report_path = write_report(output, result)
    completion = dict(
        schema="hanguard_external_holdout_completion_v1",
        state="complete_no_retuning_authorized",
        evaluation_lock_sha256=result["evaluation_lock_sha256"],
        results_sha256=frozen.sha256(result_path),
        report_sha256=frozen.sha256(report_path),
        deployment_preflight_sha256=frozen.sha256(output / "deployment_preflight.json"),
        prediction_sha256={
            name: value["predictions_sha256"] for name, value in dataset_results.items()
        },
    )
    frozen.dump(output / "completion.json", completion)
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.3f}"


def write_report(output: Path, result: Mapping[str, Any] | None = None) -> Path:
    output = Path(output).resolve()
    result = verify_completed_evaluation(output) if result is None else result
    lines = [
        "# External holdout evaluation",
        "",
        "所有数据选择、checkpoint、层选择、校准和阈值均在首次 external model forward 前写入 evaluation_lock.json。"
        "两个数据集均直接使用 source_file/source_row 回取原文并逐行复核 exact/normalized hash。固定阈值为 0.5。",
        "",
    ]
    for dataset, values in result["datasets"].items():
        systems = values["primary_systems"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"rows={values['rows']}，normalized groups={values['groups']}。",
                "",
                "| system | group-collapsed balanced acc | group-collapsed accuracy | AUROC | BCE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        ordered = [systems["dynamic"], systems["dynamic_control"], systems["oof"], systems["oof_control"]]
        for name in dict.fromkeys(ordered):
            metric = values["systems"][name]["metrics"]["group_collapsed"]
            lines.append(
                f"| {name} | {_pct(metric['balanced_accuracy'])} | {_pct(metric['accuracy'])} | "
                f"{metric['auroc']:.6f} | {metric['bce']:.6f} |"
            )
        lines.extend(
            [
                "",
                "| paired comparison | Δ balanced acc (pp) | 95% group bootstrap CI (pp) | p (group swap) | Δ BCE (candidate−baseline; <0 better) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, comparison in values["comparisons"].items():
            balanced = comparison["inference"]["balanced_accuracy"]
            bce = comparison["inference"]["bce"]
            lines.append(
                f"| {name} | {_pct(balanced['delta'])} | "
                f"[{_pct(balanced['ci95'][0])}, {_pct(balanced['ci95'][1])}] | "
                f"{balanced['permutation_two_sided_p']:.6g} | {bce['delta']:.6f} |"
            )
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(args: argparse.Namespace) -> dict:
    lock_evaluation(args)
    extract(args)
    return evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "extract", "evaluate", "report", "run"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dynamic-study", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--oof-study", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dynamic-primary-arm", default="auto", choices=("auto",),
        help="Fixed to validation-only automatic selection; manual external-arm selection is prohibited",
    )
    parser.add_argument("--dataset", nargs="+", help="Optional locked dataset subset during extraction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-budget", type=int, default=24576)
    parser.add_argument("--max-micro", type=int, default=64)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    parser.add_argument("--status-every", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=271828)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.token_budget, args.max_micro, args.pad_multiple, args.prediction_batch_size,
        args.status_every, args.repetitions,
    )
    if any(value < 1 for value in positive):
        raise ValueError("External extraction/evaluation settings must be positive")
    if args.action in ("extract", "evaluate", "run") and args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for external extraction/evaluation")
    if args.action == "lock":
        result = lock_evaluation(args)
    elif args.action == "extract":
        result = extract(args)
    elif args.action == "evaluate":
        result = evaluate(args)
    elif args.action == "report":
        result = dict(report=str(write_report(args.output)))
    else:
        result = run_all(args)
    compact = result if args.action in ("lock", "report") else {"state": "complete", "action": args.action}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
