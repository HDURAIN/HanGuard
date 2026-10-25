"""Auditably finish the independent E03 LoRA seeds 42/43/44 study.

This module deliberately leaves ``repaired_study.py`` unchanged while jobs are
running.  ``lock`` first proves that all three validation selections are final,
then atomically creates an immutable barrier covering protocol, arms,
selections, checkpoints, validation evidence, inference code, base-model
identity, and the test parquet bytes.  Only ``evaluate``/``run`` may delegate to
the GPU evaluator, and only after the complete barrier verifies.

The final aggregate is E03-only.  It reports the distribution across the three
independent LoRA seeds, source strata, and paired error agreement/discordance;
it never waits for or reads R02.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.hanguard import repaired_study  # noqa: E402


DEFAULT_STUDY = ROOT / "outputs/hanguard_e03_lora_multiseed_20261012"
ARM = "E03"
SEEDS = (42, 43, 44)
SCHEMA = "hanguard_e03_lora_multiseed_test_barrier_v1"
AMENDMENT_SCHEMA = "hanguard_e03_training_code_provenance_amendment_v1"
EXPECTED_HANGUARD_MODEL_SHA256 = "1251a52f0dd95f2b12bb299f8162e5ec811589f514217cff6dbf665680b4cb58"
STALE_REGISTERED_HANGUARD_MODEL_SHA256 = "94bec800c9c89a776c4e8a69418d0f43e1835e87baf4ad4e43f935751abe4fcc"
EXPECTED_HANGUARD_MODEL_MTIME_NS = 1790600835207034393
METRIC_KEYS = (
    "accuracy", "f1", "precision", "recall", "fpr", "auroc", "auprc", "bce", "brier",
)


class AuditError(RuntimeError):
    """Raised when immutable study evidence does not satisfy the contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a derived, non-barrier JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_immutable_json(path: Path, value: Any) -> None:
    """Publish complete bytes atomically, never replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == payload, f"Immutable artifact already exists with different bytes: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_read_only_immutable_json(path: Path, value: Any) -> None:
    """Create an immutable sidecar and remove all write bits (mode 0444)."""
    write_immutable_json(path, value)
    path = Path(path)
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"Failed to make sidecar read-only: {path}")


def artifact_record(path: Path) -> dict:
    path = Path(path).resolve()
    require(path.is_file(), f"Missing artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_artifact(record: Mapping[str, Any], context: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    require(path.is_file(), f"Locked artifact missing: {context}: {path}")
    require(path.stat().st_size == int(record.get("bytes", -1)), f"Locked artifact size changed: {context}")
    require(sha256_file(path) == record.get("sha256"), f"Locked artifact hash changed: {context}")
    return path


def code_paths() -> dict[str, Path]:
    return {
        "finalizer": Path(__file__).resolve(),
        "repaired_study": Path(repaired_study.__file__).resolve(),
        "repaired_heads": (ROOT / "scripts/hanguard/repaired_heads.py").resolve(),
        "hanguard_model": (ROOT / "hanguard_model.py").resolve(),
    }


def load_contract(study: Path) -> tuple[dict, dict, dict]:
    """Read only protocol metadata; this function never resolves or reads test."""
    study = Path(study).resolve()
    protocol_path, arms_path = study / "protocol.json", study / "arms.json"
    registration_path = study / "registration.json"
    require(protocol_path.is_file() and arms_path.is_file(), "Study protocol/arms are missing")
    require(registration_path.is_file(), "Independent-LoRA registration is missing")
    protocol, arms, registration = map(read_json, (protocol_path, arms_path, registration_path))
    require(tuple(protocol.get("seeds", ())) == SEEDS, "Protocol must contain exactly seeds 42/43/44")
    require(protocol.get("all_arms") == [ARM], "Protocol must be E03-only")
    require(set(arms) == {ARM}, "arms.json must contain only E03")
    require(arms[ARM] == {"lora": True, "mode": "last", "readout": "mlp", "objective": "bce"},
            "E03 architecture differs from the registered LoRA final-layer arm")
    require(registration.get("state") == "locked_for_training", "Training registration is not locked")
    require(registration.get("arm") == ARM and tuple(registration.get("seeds", ())) == SEEDS,
            "Training registration arm/seeds changed")
    require(registration.get("protocol_sha256") == sha256_file(protocol_path),
            "Registration protocol hash changed")
    require(registration.get("arms_sha256") == sha256_file(arms_path), "Registration arms hash changed")
    hashes, rows = protocol.get("data_sha256", {}), protocol.get("rows", {})
    require(set(hashes) >= {"train", "validation", "test"}, "Three registered data hashes required")
    require(all(isinstance(rows.get(split), int) and rows[split] > 0 for split in ("train", "validation", "test")),
            "Three positive registered row counts required")
    return protocol, arms, registration


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8").strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().isoformat()


def collect_hanguard_model_git_evidence() -> dict:
    relative = "hanguard_model.py"
    history_commits = sorted(set(_git_text("log", "--all", "--format=%H", "--", relative).splitlines()))
    require(history_commits, "Git history contains no hanguard_model.py revision")
    history = []
    for commit in history_commits:
        blob = _git_text("rev-parse", f"{commit}:{relative}")
        digest = _sha256_bytes(_git_bytes("show", f"{commit}:{relative}"))
        history.append({"commit": commit, "blob_oid": blob, "sha256": digest})
    head_commit = _git_text("rev-parse", "HEAD")
    head_blob = _git_text("rev-parse", f"HEAD:{relative}")
    current_blob = _git_text("hash-object", relative)
    status = _git_text("status", "--porcelain=v1", "--", relative)
    value = {
        "repository": str(ROOT), "head_commit_at_amendment": head_commit,
        "head_blob_oid": head_blob, "head_sha256": _sha256_bytes(_git_bytes("show", f"HEAD:{relative}")),
        "working_tree_blob_oid": current_blob, "working_tree_status": status,
        "all_history_versions": history,
        "all_history_sha256_identical": len({row["sha256"] for row in history}) == 1,
    }
    require(value["head_sha256"] == EXPECTED_HANGUARD_MODEL_SHA256,
            "Git HEAD hanguard_model.py differs from the expected training implementation")
    require(current_blob == head_blob and status == "", "Working hanguard_model.py is not the clean HEAD blob")
    require(value["all_history_sha256_identical"] and
            {row["sha256"] for row in history} == {EXPECTED_HANGUARD_MODEL_SHA256},
            "A relevant Git history version of hanguard_model.py differs")
    return value


def _live_process_observation(status_path: Path) -> dict | None:
    """Capture launch evidence while a pre-existing trainer is still alive."""
    try:
        status_value = read_json(status_path)
        pid = int(status_value["pid"])
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart=", "-o", "args="], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        line = completed.stdout.strip()
        if completed.returncode != 0 or not line:
            return None
        # Linux ps emits a fixed 24-character lstart before the command.
        return {"pid": pid, "ps_lstart": line[:24].strip(), "command": line[24:].strip()}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _run_provenance_evidence(study: Path, seed: int, model_mtime_ns: int) -> dict:
    directory = Path(study) / "runs" / f"E03_s{seed}"
    provenance_path, status_path = directory / "provenance.json", directory / "status.json"
    require(provenance_path.is_file(), f"E03_s{seed} provenance is required before creating the amendment")
    provenance = read_json(provenance_path)
    stat_value = provenance_path.stat()
    require(provenance.get("arm") == ARM and provenance.get("seed") == seed,
            f"E03_s{seed} provenance identity changed")
    require(provenance.get("code_sha256") == sha256_file(Path(repaired_study.__file__)) and
            provenance.get("heads_sha256") == sha256_file(ROOT / "scripts/hanguard/repaired_heads.py"),
            f"E03_s{seed} trainer/head provenance differs from current registered code")
    require(stat_value.st_mtime_ns > model_mtime_ns,
            f"E03_s{seed} provenance predates the locked hanguard_model.py")
    return {
        "seed": seed, "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "provenance_mtime_ns": stat_value.st_mtime_ns,
        "provenance_mtime_local": _iso_from_timestamp(stat_value.st_mtime),
        "trainer_code_sha256": provenance.get("code_sha256"),
        "heads_code_sha256": provenance.get("heads_sha256"),
        "hanguard_model_sha256_recorded_in_run_provenance": None,
        "limitation": "The legacy per-run provenance schema did not record hanguard_model.py SHA-256.",
        "live_process_observation_at_amendment": _live_process_observation(status_path),
    }


def amendment_path(study: Path) -> Path:
    return Path(study).resolve() / "training_code_provenance_amendment.json"


def create_training_code_provenance_amendment(study: Path) -> dict:
    """Create the pre-test 0444 sidecar without reading or hashing test data."""
    study = Path(study).resolve()
    path = amendment_path(study)
    if path.exists():
        return verify_training_code_provenance_amendment(study)
    protocol, _, _ = load_contract(study)
    protocol_path = study / "protocol.json"
    model_path = (ROOT / "hanguard_model.py").resolve()
    model_stat = model_path.stat()
    current_hash = sha256_file(model_path)
    registered_hash = protocol.get("code_sha256", {}).get("hanguard_model.py")
    require(current_hash == EXPECTED_HANGUARD_MODEL_SHA256,
            "Current hanguard_model.py differs from the implementation used by the active trainers")
    require(model_stat.st_mtime_ns == EXPECTED_HANGUARD_MODEL_MTIME_NS,
            "hanguard_model.py mtime drifted from the pre-training file identity")
    require(registered_hash == STALE_REGISTERED_HANGUARD_MODEL_SHA256,
            "Protocol no longer contains the known stale hanguard_model.py registry value")
    registered_at = protocol.get("registered_at", protocol.get("created_at"))
    require(isinstance(registered_at, str), "Protocol registration timestamp is missing")
    registered_time = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
    require(model_stat.st_mtime < registered_time.timestamp(),
            "hanguard_model.py was modified after protocol registration")
    git = collect_hanguard_model_git_evidence()
    runs = [_run_provenance_evidence(study, seed, model_stat.st_mtime_ns) for seed in (42, 43)]
    value = {
        "schema": AMENDMENT_SCHEMA, "state": "locked_pretest", "read_only_mode": "0444",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Training-code provenance correction for E03 LoRA seeds 42/43/44; no test access.",
        "protocol": {
            "path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path),
            "registered_at": registered_at,
            "file_mtime_ns": protocol_path.stat().st_mtime_ns,
            "file_mtime_local": _iso_from_timestamp(protocol_path.stat().st_mtime),
        },
        "hanguard_model": {
            "path": str(model_path),
            "protocol_registered_sha256": registered_hash,
            "current_sha256": current_hash,
            "mtime_ns": model_stat.st_mtime_ns,
            "mtime_local": _iso_from_timestamp(model_stat.st_mtime),
            "mtime_precedes_protocol_registration": True,
        },
        "git_evidence": git, "pre_existing_runs": runs,
        "execution_order_evidence": {
            "repaired_study_sha256": sha256_file(Path(repaired_study.__file__)),
            "fact": (
                "repaired_study.train calls setup before writing provenance; setup imports "
                "hanguard_model.load_base_model. Thus each provenance mtime is a conservative post-import bound."
            ),
        },
        "finding": (
            "The 94bec... protocol value is stale registration metadata. The clean working file, Git HEAD, "
            "and every repository history version resolve to SHA-256 1251a52f..., whose mtime predates protocol "
            "registration and both observed seed42/43 launches."
        ),
        "future_seed_policy": {
            "seed": 44, "required_sha256": EXPECTED_HANGUARD_MODEL_SHA256,
            "required_entrypoint": "finalize_e03_lora_multiseed.py train --seed 44",
            "launch_receipt": "finalization/E03_s44_training_launch_receipt.json",
            "rule": "The wrapper must verify this sidecar and import the locked module before invoking training.",
        },
        "limitations": [
            "Legacy seed42/43 run provenance records trainer/head hashes but omits hanguard_model.py.",
            "Filesystem mtime and the live ps observations are supporting provenance, not cryptographic process attestation.",
        ],
    }
    write_read_only_immutable_json(path, value)
    return verify_training_code_provenance_amendment(study)


def verify_training_code_provenance_amendment(study: Path) -> dict:
    study = Path(study).resolve()
    path = amendment_path(study)
    require(path.is_file(), "training_code_provenance_amendment.json is required before seed44/lock")
    require(stat.S_IMODE(path.stat().st_mode) == 0o444,
            "Training-code provenance amendment must remain mode 0444")
    value = read_json(path)
    require(value.get("schema") == AMENDMENT_SCHEMA and value.get("state") == "locked_pretest",
            "Unexpected training-code provenance amendment")
    protocol, _, _ = load_contract(study)
    protocol_path = study / "protocol.json"
    require(value.get("protocol", {}).get("path") == str(protocol_path.resolve()) and
            value["protocol"].get("sha256") == sha256_file(protocol_path) and
            value["protocol"].get("file_mtime_ns") == protocol_path.stat().st_mtime_ns,
            "Amendment protocol identity changed")
    model_path = (ROOT / "hanguard_model.py").resolve()
    model_stat = model_path.stat()
    model = value.get("hanguard_model", {})
    require(model.get("path") == str(model_path) and
            model.get("protocol_registered_sha256") == STALE_REGISTERED_HANGUARD_MODEL_SHA256 and
            model.get("current_sha256") == EXPECTED_HANGUARD_MODEL_SHA256,
            "Amendment hanguard_model identity changed")
    require(sha256_file(model_path) == EXPECTED_HANGUARD_MODEL_SHA256 and
            model_stat.st_mtime_ns == EXPECTED_HANGUARD_MODEL_MTIME_NS == model.get("mtime_ns"),
            "Current hanguard_model.py hash/mtime drifted after amendment")
    require(protocol.get("code_sha256", {}).get("hanguard_model.py") == STALE_REGISTERED_HANGUARD_MODEL_SHA256,
            "Registered stale hash changed after amendment")
    git = collect_hanguard_model_git_evidence()
    recorded_git = value.get("git_evidence", {})
    # HEAD may advance for unrelated files; the hanguard_model blob and all relevant versions must not.
    for key in ("head_blob_oid", "head_sha256", "working_tree_blob_oid", "working_tree_status",
                "all_history_versions", "all_history_sha256_identical"):
        require(recorded_git.get(key) == git.get(key), f"Git hanguard_model evidence drifted: {key}")
    recorded_runs = {int(row.get("seed", -1)): row for row in value.get("pre_existing_runs", [])}
    require(set(recorded_runs) == {42, 43}, "Amendment must cover pre-existing seeds 42/43")
    for seed, row in recorded_runs.items():
        provenance = Path(row["provenance_path"])
        require(provenance.is_file() and sha256_file(provenance) == row["provenance_sha256"] and
                provenance.stat().st_mtime_ns == row["provenance_mtime_ns"],
                f"E03_s{seed} provenance changed after amendment")
    require(value.get("execution_order_evidence", {}).get("repaired_study_sha256") ==
            sha256_file(Path(repaired_study.__file__)),
            "Trainer implementation changed after the execution-order evidence was recorded")
    expected_policy = {
        "seed": 44, "required_sha256": EXPECTED_HANGUARD_MODEL_SHA256,
        "required_entrypoint": "finalize_e03_lora_multiseed.py train --seed 44",
        "launch_receipt": "finalization/E03_s44_training_launch_receipt.json",
        "rule": "The wrapper must verify this sidecar and import the locked module before invoking training.",
    }
    require(value.get("future_seed_policy") == expected_policy, "Seed44 amended launch policy changed")
    return value


def seed44_receipt_path(study: Path) -> Path:
    return Path(study).resolve() / "finalization/E03_s44_training_launch_receipt.json"


def _seed44_launch_receipt_value(study: Path, amendment: Mapping[str, Any]) -> dict:
    model_module = importlib.import_module("hanguard_model")
    module_path = Path(model_module.__file__).resolve()
    require(module_path == (ROOT / "hanguard_model.py").resolve(), "seed44 resolved a different hanguard_model module")
    require(sha256_file(module_path) == EXPECTED_HANGUARD_MODEL_SHA256,
            "seed44 imported an unexpected hanguard_model implementation")
    return {
        "schema": "hanguard_e03_seed44_training_launch_receipt_v1", "state": "authorized_to_train",
        "seed": 44, "arm": ARM, "created_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "entrypoint": "finalize_e03_lora_multiseed.py train --seed 44",
        "protocol_sha256": sha256_file(Path(study) / "protocol.json"),
        "amendment_sha256": sha256_file(amendment_path(study)),
        "finalizer_sha256": sha256_file(Path(__file__)),
        "imported_module_path": str(module_path),
        "imported_hanguard_model_sha256": EXPECTED_HANGUARD_MODEL_SHA256,
        "rule": "Receipt is written mode 0444 after sidecar verification and module import, before train().",
    }


def create_seed44_launch_receipt(study: Path) -> dict:
    study = Path(study).resolve()
    amendment = verify_training_code_provenance_amendment(study)
    path = seed44_receipt_path(study)
    if path.exists():
        return verify_seed44_launch_receipt(study)
    value = _seed44_launch_receipt_value(study, amendment)
    write_read_only_immutable_json(path, value)
    return verify_seed44_launch_receipt(study)


def verify_seed44_launch_receipt(study: Path) -> dict:
    study = Path(study).resolve()
    amendment = verify_training_code_provenance_amendment(study)
    path = seed44_receipt_path(study)
    require(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444,
            "Seed44 training launch receipt is missing or not mode 0444")
    value = read_json(path)
    require(value.get("schema") == "hanguard_e03_seed44_training_launch_receipt_v1" and
            value.get("state") == "authorized_to_train" and value.get("seed") == 44 and value.get("arm") == ARM,
            "Unexpected seed44 training launch receipt")
    require(value.get("protocol_sha256") == sha256_file(study / "protocol.json") and
            value.get("amendment_sha256") == sha256_file(amendment_path(study)),
            "Seed44 launch receipt protocol/amendment changed")
    require(value.get("finalizer_sha256") == sha256_file(Path(__file__)) and
            value.get("imported_module_path") == str((ROOT / "hanguard_model.py").resolve()) and
            value.get("imported_hanguard_model_sha256") == EXPECTED_HANGUARD_MODEL_SHA256,
            "Seed44 launch receipt code identity changed")
    require(amendment["future_seed_policy"]["required_sha256"] == EXPECTED_HANGUARD_MODEL_SHA256,
            "Seed44 amendment policy changed")
    return value


def train_seed44(study: Path, seed: int) -> dict:
    require(seed == 44, "The amended training wrapper is restricted to pending seed44")
    study = Path(study).resolve()
    receipt = create_seed44_launch_receipt(study)
    # hanguard_model is already imported and pinned in sys.modules by receipt creation.
    repaired_study.train(argparse.Namespace(output=str(study), arm=ARM, seed=44))
    verify_training_code_provenance_amendment(study)
    verify_seed44_launch_receipt(study)
    return receipt


def _validate_validation_csv(path: Path, selection: Mapping[str, Any], expected_rows: int) -> None:
    frame = pd.read_csv(path, dtype={"base_id": str, "source": str}, float_precision="round_trip")
    required = {"base_id", "source", "label", "prompt_tokens", "probability"}
    require(required <= set(frame), f"Validation prediction columns missing: {path}")
    require(len(frame) == expected_rows and frame.base_id.is_unique, f"Validation rows/IDs invalid: {path}")
    labels = frame.label.to_numpy(dtype=np.int64)
    probability = frame.probability.to_numpy(dtype=np.float64)
    require(np.isin(labels, [0, 1]).all() and np.isfinite(probability).all(), f"Invalid validation values: {path}")
    threshold = repaired_study.accuracy_threshold(labels, probability)
    require(float(selection["threshold"]) == threshold, f"Selection threshold is not validation-derived: {path}")
    measured = repaired_study.metrics(labels, probability, threshold)
    recorded = selection.get("metrics", {})
    for key in ("accuracy", "f1", "recall", "precision", "fpr", "auroc"):
        if key in recorded and measured.get(key) is not None:
            require(math.isclose(float(recorded[key]), float(measured[key]), rel_tol=0, abs_tol=1e-12),
                    f"Selection validation metric changed: {path}: {key}")


def validate_all_selections(study: Path, protocol: Mapping[str, Any]) -> dict[str, Path]:
    """Require every seed before any caller is allowed to inspect the test file."""
    study = Path(study).resolve()
    protocol_hash, arms_hash = sha256_file(study / "protocol.json"), sha256_file(study / "arms.json")
    # Check the full set first.  This ordering is a tested no-test-access invariant.
    missing = [
        str(study / "runs" / f"{ARM}_s{seed}" / "selection.json")
        for seed in SEEDS
        if not (study / "runs" / f"{ARM}_s{seed}" / "selection.json").is_file()
    ]
    require(not missing, "All E03 selections must finish before test locking; missing: " + ", ".join(missing))

    artifacts: dict[str, Path] = {}
    current_code = {name: sha256_file(path) for name, path in code_paths().items()}
    for seed in SEEDS:
        directory = study / "runs" / f"{ARM}_s{seed}"
        selection_path = directory / "selection.json"
        provenance_path = directory / "provenance.json"
        validation_path = directory / "validation.csv"
        status_path = directory / "status.json"
        require(provenance_path.is_file() and validation_path.is_file() and status_path.is_file(),
                f"E03_s{seed} final validation evidence is incomplete")
        require(not (directory / "resume.pt").exists(), f"E03_s{seed} still has resumable training state")
        status = read_json(status_path)
        require(status.get("state") in {"selected", "complete"}, f"E03_s{seed} is not in a selected state")
        selection, provenance = read_json(selection_path), read_json(provenance_path)
        common = {
            "arm": ARM, "seed": seed, "protocol_sha256": protocol_hash, "arms_sha256": arms_hash,
            "train_sha256": protocol["data_sha256"]["train"],
            "validation_sha256": protocol["data_sha256"]["validation"],
        }
        for key, expected in common.items():
            require(selection.get(key) == expected, f"E03_s{seed} selection identity changed: {key}")
            require(provenance.get(key) == expected, f"E03_s{seed} provenance identity changed: {key}")
        require(selection.get("fresh_base") is True and provenance.get("fresh_base") is True,
                f"E03_s{seed} is not a fresh-base run")
        require(provenance.get("old_checkpoint_reuse") is False, f"E03_s{seed} reused an old checkpoint")
        require(provenance.get("code_sha256") == current_code["repaired_study"],
                f"E03_s{seed} trainer code differs from current inference code")
        require(provenance.get("heads_sha256") == current_code["repaired_heads"],
                f"E03_s{seed} head code differs from current inference code")
        expected_rows = (int(protocol.get("warm_head_epochs", 0)) + int(protocol["epochs"])) * int(protocol["rows"]["train"])
        require(selection.get("completed_training_rows") == expected_rows,
                f"E03_s{seed} did not complete the registered epoch budget")
        require(math.isfinite(float(selection.get("validation_bce", float("nan")))),
                f"E03_s{seed} lacks finite validation BCE")
        require(math.isfinite(float(selection.get("threshold", float("nan")))),
                f"E03_s{seed} lacks a finite validation threshold")
        checkpoint_hashes = selection.get("checkpoint_hashes", {})
        require(set(checkpoint_hashes) == {"head.pt", "adapter.pt"},
                f"E03_s{seed} must select one head and one adapter checkpoint")
        for filename, digest in checkpoint_hashes.items():
            path = directory / filename
            require(path.is_file() and sha256_file(path) == digest,
                    f"E03_s{seed} selected checkpoint changed: {filename}")
            artifacts[f"runs/E03_s{seed}/{filename}"] = path
        _validate_validation_csv(validation_path, selection, int(protocol["rows"]["validation"]))
        artifacts[f"runs/E03_s{seed}/selection.json"] = selection_path
        artifacts[f"runs/E03_s{seed}/provenance.json"] = provenance_path
        artifacts[f"runs/E03_s{seed}/validation.csv"] = validation_path
    return artifacts


def _model_identity(protocol: Mapping[str, Any]) -> dict:
    model = Path(protocol["model_path"]).resolve()
    config, tokenizer = model / "config.json", model / "tokenizer.json"
    require(config.is_file() and tokenizer.is_file(), "Base model config/tokenizer missing")
    require(sha256_file(config) == protocol["model_config_sha256"], "Base model config changed")
    require(sha256_file(tokenizer) == protocol["tokenizer_sha256"], "Base tokenizer changed")
    index = model / "model.safetensors.index.json"
    require(index.is_file(), "Base model weight index missing")
    weight_map = read_json(index).get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    require(shards, "Base model weight index contains no shards")
    shard_bytes = {}
    for name in shards:
        path = model / name
        require(path.is_file(), f"Base model shard missing: {name}")
        shard_bytes[name] = path.stat().st_size
    return {
        "path": str(model), "config_sha256": sha256_file(config),
        "tokenizer_sha256": sha256_file(tokenizer), "weight_index_sha256": sha256_file(index),
        "weight_shard_bytes": shard_bytes,
    }


def _test_dataset_record(protocol: Mapping[str, Any]) -> dict:
    """Called only after all selections validate; hashes bytes without deserializing rows."""
    path = (Path(protocol["data_dir"]) / "test.parquet").resolve()
    record = artifact_record(path)
    require(record["sha256"] == protocol["data_sha256"]["test"], "Registered test parquet changed")
    record["rows"] = int(protocol["rows"]["test"])
    return record


def _registered_code_comparison(protocol: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict:
    registered = protocol.get("code_sha256", {})
    aliases = {
        "repaired_study": "repaired_study.py",
        "repaired_heads": "repaired_heads.py",
        "hanguard_model": "hanguard_model.py",
    }
    result = {}
    for name, filename in aliases.items():
        expected = registered.get(filename)
        actual = records[name]["sha256"]
        result[name] = {"registered_sha256": expected, "locked_sha256": actual, "matches": expected == actual}
    return result


def lock_test(study: Path) -> dict:
    study = Path(study).resolve()
    protocol, _, _ = load_contract(study)
    verify_training_code_provenance_amendment(study)
    verify_seed44_launch_receipt(study)
    selection_paths = validate_all_selections(study, protocol)
    barrier_path = study / "test_barrier.json"
    if barrier_path.exists():
        return verify_test_barrier(study, protocol=protocol, selection_paths=selection_paths)

    for seed in SEEDS:
        directory = study / "runs" / f"{ARM}_s{seed}"
        require(not (directory / "test.csv").exists() and not (directory / "test_results.json").exists(),
                f"E03_s{seed} test artifacts predate the barrier")

    # This is intentionally the first operation that touches the test file.
    test_dataset = _test_dataset_record(protocol)
    paths = {
        "protocol.json": study / "protocol.json", "arms.json": study / "arms.json",
        "registration.json": study / "registration.json", **selection_paths,
        "training_code_provenance_amendment.json": amendment_path(study),
        "finalization/E03_s44_training_launch_receipt.json": seed44_receipt_path(study),
        **{f"code/{name}": path for name, path in code_paths().items()},
    }
    records = {name: artifact_record(path) for name, path in sorted(paths.items())}
    code_records = {name: records[f"code/{name}"] for name in code_paths()}
    code_comparison = _registered_code_comparison(protocol, code_records)
    barrier = {
        "schema": SCHEMA, "state": "locked", "locked": True,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "E03 independent LoRA seeds 42/43/44 only; no R02 dependency",
        "arm": ARM, "seeds": list(SEEDS),
        "protocol_sha256": records["protocol.json"]["sha256"],
        "arms_sha256": records["arms.json"]["sha256"],
        # Compatibility with repaired_study.evaluate, plus the stronger general artifact table.
        "selection_sha256": {
            f"runs/E03_s{seed}/selection.json": records[f"runs/E03_s{seed}/selection.json"]["sha256"]
            for seed in SEEDS
        },
        "checkpoint_sha256": {
            f"runs/E03_s{seed}/{name}": records[f"runs/E03_s{seed}/{name}"]["sha256"]
            for seed in SEEDS for name in ("head.pt", "adapter.pt")
        },
        "artifacts": records,
        "test_dataset": test_dataset,
        "test_sha256": test_dataset["sha256"],
        "model_identity": _model_identity(protocol),
        "registered_code_comparison": code_comparison,
        "code_provenance_note": (
            "Per-run provenance directly verifies repaired_study.py and repaired_heads.py. The barrier locks "
            "all actual inference code and explicitly exposes any stale protocol-level code hash instead of "
            "silently treating it as matched training provenance."
        ),
        "rule": (
            "All three validation selections and selected checkpoints were complete before the test parquet "
            "was hashed. Test rows may be deserialized only after this immutable barrier verifies in full."
        ),
    }
    write_immutable_json(barrier_path, barrier)
    return verify_test_barrier(study, protocol=protocol, selection_paths=selection_paths)


def verify_test_barrier(
    study: Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    selection_paths: Mapping[str, Path] | None = None,
) -> dict:
    study = Path(study).resolve()
    if protocol is None:
        protocol, _, _ = load_contract(study)
    if selection_paths is None:
        selection_paths = validate_all_selections(study, protocol)
    verify_training_code_provenance_amendment(study)
    verify_seed44_launch_receipt(study)
    path = study / "test_barrier.json"
    require(path.is_file(), "Test barrier is missing; run lock only after all three selections finish")
    value = read_json(path)
    require(value.get("schema") == SCHEMA and value.get("state") == "locked" and value.get("locked") is True,
            "Unexpected or unlocked E03 test barrier")
    require(value.get("arm") == ARM and tuple(value.get("seeds", ())) == SEEDS,
            "Test barrier scope changed")

    expected_paths = {
        "protocol.json": study / "protocol.json", "arms.json": study / "arms.json",
        "registration.json": study / "registration.json", **selection_paths,
        "training_code_provenance_amendment.json": amendment_path(study),
        "finalization/E03_s44_training_launch_receipt.json": seed44_receipt_path(study),
        **{f"code/{name}": code_path for name, code_path in code_paths().items()},
    }
    records = value.get("artifacts", {})
    require(set(records) == set(expected_paths), "Test barrier artifact set changed")
    for name, expected in expected_paths.items():
        actual = verify_artifact(records[name], name)
        require(actual == Path(expected).resolve(), f"Locked artifact path changed: {name}")
    require(value.get("protocol_sha256") == records["protocol.json"]["sha256"], "Barrier protocol hash changed")
    require(value.get("arms_sha256") == records["arms.json"]["sha256"], "Barrier arms hash changed")
    expected_selections = {
        f"runs/E03_s{seed}/selection.json": records[f"runs/E03_s{seed}/selection.json"]["sha256"]
        for seed in SEEDS
    }
    expected_checkpoints = {
        f"runs/E03_s{seed}/{name}": records[f"runs/E03_s{seed}/{name}"]["sha256"]
        for seed in SEEDS for name in ("head.pt", "adapter.pt")
    }
    require(value.get("selection_sha256") == expected_selections, "Barrier selection index changed")
    require(value.get("checkpoint_sha256") == expected_checkpoints, "Barrier checkpoint index changed")
    code_records = {name: records[f"code/{name}"] for name in code_paths()}
    require(value.get("registered_code_comparison") == _registered_code_comparison(protocol, code_records),
            "Barrier registered/current code comparison changed")
    test_record = value.get("test_dataset", {})
    test_path = (Path(protocol["data_dir"]) / "test.parquet").resolve()
    require(verify_artifact(test_record, "test_dataset") == test_path, "Locked test path changed")
    require(test_record.get("sha256") == protocol["data_sha256"]["test"], "Barrier test hash differs from protocol")
    require(test_record.get("rows") == protocol["rows"]["test"], "Barrier test row contract changed")
    require(value.get("test_sha256") == test_record.get("sha256"), "Barrier test hash alias changed")
    require(value.get("model_identity") == _model_identity(protocol), "Locked base-model identity changed")
    return value


def binary_metrics(labels: Sequence[int], probability: Sequence[float], threshold: float) -> dict:
    y, p = np.asarray(labels, dtype=np.int64), np.asarray(probability, dtype=np.float64)
    require(y.ndim == 1 and p.shape == y.shape and len(y) > 0, "Nonempty aligned predictions required")
    require(np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
            "Invalid labels/probabilities")
    require(math.isfinite(float(threshold)), "Finite threshold required")
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "auroc": float(roc_auc_score(y, p)) if two_classes else None,
        "auprc": float(average_precision_score(y, p)) if two_classes else None,
        "bce": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
        "brier": float(np.square(p - y).mean()), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _compare_recorded(computed: Mapping[str, Any], recorded: Mapping[str, Any], context: str) -> None:
    for key in ("accuracy", "f1", "precision", "recall", "fpr", "auroc", "tp", "tn", "fp", "fn"):
        if key not in recorded:
            continue
        left, right = computed.get(key), recorded.get(key)
        if left is None or right is None:
            require(left is right, f"Recorded metric changed: {context}/{key}")
        elif isinstance(left, int):
            require(left == right, f"Recorded metric changed: {context}/{key}")
        else:
            require(math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-10),
                    f"Recorded metric changed: {context}/{key}")


def load_test_metadata(protocol: Mapping[str, Any], barrier: Mapping[str, Any]) -> pd.DataFrame:
    """Deserialize test metadata only after the complete immutable barrier verifies."""
    path = Path(barrier["test_dataset"]["path"])
    require(sha256_file(path) == barrier["test_sha256"] == protocol["data_sha256"]["test"],
            "Test parquet changed before metadata verification")
    frame = pd.read_parquet(path)
    required = {"base_id", "source", "prompt_harm_label"}
    require(required <= set(frame), "Test parquet lacks identity/source/label metadata")
    require(len(frame) == protocol["rows"]["test"] and frame.base_id.is_unique,
            "Test parquet rows/IDs differ from the registered contract")
    require(frame.prompt_harm_label.isin(["harmful", "unharmful"]).all(), "Test parquet labels are invalid")
    result = frame[["base_id", "source"]].copy()
    result["base_id"] = result.base_id.astype(str)
    result["source"] = result.source.astype(str)
    result["label"] = frame.prompt_harm_label.eq("harmful").astype(np.int64)
    return result.sort_values("base_id", kind="stable").reset_index(drop=True)


def load_seed_output(
    study: Path,
    seed: int,
    protocol: Mapping[str, Any],
    barrier: Mapping[str, Any],
    expected_metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    directory = Path(study) / "runs" / f"{ARM}_s{seed}"
    selection = read_json(directory / "selection.json")
    result_path, prediction_path = directory / "test_results.json", directory / "test.csv"
    require(result_path.is_file() and prediction_path.is_file(), f"E03_s{seed} test evaluation is incomplete")
    result = read_json(result_path)
    require(result.get("arm") == ARM and result.get("seed") == seed, f"E03_s{seed} result identity changed")
    require(result.get("test_sha256") == barrier["test_sha256"], f"E03_s{seed} used a different test parquet")
    require(result.get("checkpoint_hashes") == selection["checkpoint_hashes"],
            f"E03_s{seed} result checkpoint identity changed")
    require(float(result.get("threshold")) == float(selection["threshold"]),
            f"E03_s{seed} result threshold changed")
    frame = pd.read_csv(prediction_path, dtype={"base_id": str, "source": str}, float_precision="round_trip")
    columns = {"base_id", "source", "label", "prompt_tokens", "probability"}
    require(columns <= set(frame), f"E03_s{seed} test prediction columns missing")
    require(len(frame) == protocol["rows"]["test"] and frame.base_id.is_unique,
            f"E03_s{seed} test rows/IDs invalid")
    frame = frame.sort_values("base_id", kind="stable").reset_index(drop=True)
    if expected_metadata is not None:
        for column in ("base_id", "source", "label"):
            require(np.array_equal(frame[column].to_numpy(), expected_metadata[column].to_numpy()),
                    f"E03_s{seed} predictions differ from locked test metadata: {column}")
    measured = binary_metrics(frame.label, frame.probability, float(selection["threshold"]))
    _compare_recorded(measured, result.get("metrics", {}), f"E03_s{seed}")
    return frame, {"selection": selection, "saved": result, "metrics": measured}


def _receipt_value(study: Path, seed: int, barrier: Mapping[str, Any]) -> dict:
    directory = Path(study) / "runs" / f"{ARM}_s{seed}"
    return {
        "schema": "hanguard_e03_lora_seed_evaluation_receipt_v1", "arm": ARM, "seed": seed,
        "barrier_sha256": sha256_file(Path(study) / "test_barrier.json"),
        "test_sha256": barrier["test_sha256"],
        "outputs": {
            "test.csv": artifact_record(directory / "test.csv"),
            "test_results.json": artifact_record(directory / "test_results.json"),
        },
    }


def verify_receipt(study: Path, seed: int, barrier: Mapping[str, Any]) -> dict:
    path = Path(study) / "finalization" / f"E03_s{seed}_evaluation_receipt.json"
    require(path.is_file(), f"E03_s{seed} evaluation receipt is missing")
    value = read_json(path)
    expected = _receipt_value(study, seed, barrier)
    require(value == expected, f"E03_s{seed} evaluation receipt/output changed")
    return value


def evaluate_seed(study: Path, seed: int) -> dict:
    study = Path(study).resolve()
    require(seed in SEEDS, f"Evaluation seed must be one of {SEEDS}")
    protocol, _, _ = load_contract(study)
    selections = validate_all_selections(study, protocol)
    barrier = verify_test_barrier(study, protocol=protocol, selection_paths=selections)
    repaired_study.evaluate(argparse.Namespace(output=str(study), arm=ARM, seed=seed))
    # Close the inference-time TOCTOU window before accepting any output.
    selections = validate_all_selections(study, protocol)
    barrier = verify_test_barrier(study, protocol=protocol, selection_paths=selections)
    expected_metadata = load_test_metadata(protocol, barrier)
    load_seed_output(study, seed, protocol, barrier, expected_metadata)
    receipt = _receipt_value(study, seed, barrier)
    write_immutable_json(study / "finalization" / f"E03_s{seed}_evaluation_receipt.json", receipt)
    return verify_receipt(study, seed, barrier)


def summarize(values: Mapping[int, float | None]) -> dict:
    finite = {str(seed): float(value) for seed, value in values.items() if value is not None and math.isfinite(float(value))}
    array = np.asarray(list(finite.values()), dtype=np.float64)
    return {
        "values": finite, "n_seeds": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def aggregate_metrics(per_seed: Mapping[int, Mapping[str, Any]]) -> dict:
    return {key: summarize({seed: row.get(key) for seed, row in per_seed.items()}) for key in METRIC_KEYS}


def paired_diagnostics(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame, threshold_a: float, threshold_b: float,
) -> dict:
    for column in ("base_id", "source", "label", "prompt_tokens"):
        require(np.array_equal(frame_a[column].to_numpy(), frame_b[column].to_numpy()),
                f"Paired seed rows differ: {column}")
    y = frame_a.label.to_numpy(dtype=np.int64)
    pred_a = frame_a.probability.to_numpy(dtype=np.float64) >= threshold_a
    pred_b = frame_b.probability.to_numpy(dtype=np.float64) >= threshold_b
    correct_a, correct_b = pred_a == y, pred_b == y
    a_only = int((correct_a & ~correct_b).sum()); b_only = int((~correct_a & correct_b).sum())
    errors_a, errors_b = ~correct_a, ~correct_b
    error_union = int((errors_a | errors_b).sum()); error_intersection = int((errors_a & errors_b).sum())
    metrics_a = binary_metrics(y, frame_a.probability, threshold_a)
    metrics_b = binary_metrics(y, frame_b.probability, threshold_b)
    return {
        "n": int(len(y)), "decision_agreement": float((pred_a == pred_b).mean()),
        "both_correct": int((correct_a & correct_b).sum()), "both_wrong": error_intersection,
        "a_only_correct": a_only, "b_only_correct": b_only, "net_correct_a_minus_b": a_only - b_only,
        "mcnemar_exact_two_sided_p": float(binomtest(min(a_only, b_only), a_only + b_only, .5).pvalue)
        if a_only + b_only else 1.0,
        "error_jaccard": error_intersection / error_union if error_union else 1.0,
        "a_errors_rescued_by_b": int((errors_a & correct_b).sum()),
        "b_errors_rescued_by_a": int((errors_b & correct_a).sum()),
        "metric_delta_a_minus_b": {
            key: (float(metrics_a[key] - metrics_b[key]) if metrics_a[key] is not None and metrics_b[key] is not None else None)
            for key in METRIC_KEYS
        },
    }


def _source_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    return {
        str(source): binary_metrics(group.label, group.probability, threshold)
        for source, group in frame.groupby("source", sort=True)
    }


def build_aggregate(study: Path) -> tuple[dict, str]:
    study = Path(study).resolve()
    protocol, _, _ = load_contract(study)
    selections = validate_all_selections(study, protocol)
    barrier = verify_test_barrier(study, protocol=protocol, selection_paths=selections)
    expected_metadata = load_test_metadata(protocol, barrier)
    frames: dict[int, pd.DataFrame] = {}; records: dict[int, dict] = {}
    receipts = {}
    for seed in SEEDS:
        frames[seed], records[seed] = load_seed_output(
            study, seed, protocol, barrier, expected_metadata,
        )
        receipts[str(seed)] = verify_receipt(study, seed, barrier)
    reference = frames[SEEDS[0]]
    for seed in SEEDS[1:]:
        for column in ("base_id", "source", "label", "prompt_tokens"):
            require(np.array_equal(reference[column].to_numpy(), frames[seed][column].to_numpy()),
                    f"E03 seed test rows differ: seed {seed}/{column}")

    thresholds = {seed: float(records[seed]["selection"]["threshold"]) for seed in SEEDS}
    per_seed = {seed: records[seed]["metrics"] for seed in SEEDS}
    source_names = sorted(set(reference.source.astype(str)))
    source_by_seed = {seed: _source_metrics(frames[seed], thresholds[seed]) for seed in SEEDS}
    source_aggregate = {
        source: aggregate_metrics({seed: source_by_seed[seed][source] for seed in SEEDS})
        for source in source_names
    }
    pairs = []
    for seed_a, seed_b in combinations(SEEDS, 2):
        overall = paired_diagnostics(frames[seed_a], frames[seed_b], thresholds[seed_a], thresholds[seed_b])
        by_source = {}
        for source in source_names:
            a = frames[seed_a][frames[seed_a].source.astype(str) == source].reset_index(drop=True)
            b = frames[seed_b][frames[seed_b].source.astype(str) == source].reset_index(drop=True)
            by_source[source] = paired_diagnostics(a, b, thresholds[seed_a], thresholds[seed_b])
        pairs.append({"seed_a": seed_a, "seed_b": seed_b, "overall": overall, "by_source": by_source})

    labels = reference.label.to_numpy(dtype=np.int64)
    decisions = np.column_stack([
        frames[seed].probability.to_numpy(dtype=np.float64) >= thresholds[seed] for seed in SEEDS
    ])
    correct = decisions == labels[:, None]
    majority = decisions.sum(axis=1) >= 2
    consensus = {
        "all_three_correct": int(correct.all(axis=1).sum()),
        "all_three_wrong": int((~correct).all(axis=1).sum()),
        "at_least_one_correct": int(correct.any(axis=1).sum()),
        "oracle_accuracy": float(correct.any(axis=1).mean()),
        "unanimous_decision_rate": float(((decisions.sum(axis=1) == 0) | (decisions.sum(axis=1) == 3)).mean()),
        "majority_vote_metrics": binary_metrics(labels, majority.astype(np.float64), .5),
        "note": "Oracle is a descriptive upper bound; majority vote uses each seed's validation-selected threshold.",
    }
    result = {
        "schema": "hanguard_e03_lora_multiseed_results_v1", "scope": "E03 only; seeds 42/43/44",
        "arm": ARM, "seeds": list(SEEDS), "n_independent_lora_seeds": 3,
        "test_barrier_sha256": sha256_file(study / "test_barrier.json"),
        "test_sha256": barrier["test_sha256"], "rows": int(len(reference)),
        "training_code_provenance_amendment_sha256": sha256_file(amendment_path(study)),
        "seed44_training_launch_receipt_sha256": sha256_file(seed44_receipt_path(study)),
        "registered_code_comparison": barrier["registered_code_comparison"],
        "code_provenance_note": barrier["code_provenance_note"],
        "selection": {
            str(seed): {
                "validation_bce": float(records[seed]["selection"]["validation_bce"]),
                "best_epoch": records[seed]["selection"]["best_epoch"], "threshold": thresholds[seed],
                "selection_sha256": sha256_file(study / "runs" / f"E03_s{seed}" / "selection.json"),
            } for seed in SEEDS
        },
        "per_seed": {str(seed): per_seed[seed] for seed in SEEDS},
        "aggregate_across_lora_seeds": aggregate_metrics(per_seed),
        "by_source_per_seed": {str(seed): source_by_seed[seed] for seed in SEEDS},
        "by_source_aggregate": source_aggregate,
        "paired_seed_diagnostics": pairs, "error_consensus": consensus,
        "evaluation_receipts_sha256": {
            str(seed): sha256_file(study / "finalization" / f"E03_s{seed}_evaluation_receipt.json")
            for seed in SEEDS
        },
        "inference_note": (
            "Three independent LoRA initializations give a descriptive seed-stability distribution. "
            "They share the same test rows, so row counts are not tripled and n=3 is too small for a precise "
            "population estimate. Pairwise McNemar values are descriptive and unadjusted."
        ),
        "scope_note": "No R02 artifact, prediction, or completion state is read by this finalizer.",
    }
    aggregate = result["aggregate_across_lora_seeds"]
    def cell(key: str, scale: float = 100.0) -> str:
        row = aggregate[key]
        return f"{scale * row['mean']:.3f} ± {scale * row['sample_std']:.3f}" if row["sample_std"] is not None else f"{scale * row['mean']:.3f}"
    lines = [
        "# E03 独立 LoRA 三种子评估", "",
        "本报告只汇总 E03 的 LoRA seeds 42/43/44；三个种子共用同一测试样本，因此样本量不按三倍计算。", "",
        "| seed | validation threshold | accuracy (%) | F1 (%) | AUROC (%) | BCE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = per_seed[seed]
        lines.append(
            f"| {seed} | {thresholds[seed]:.6f} | {100*row['accuracy']:.3f} | {100*row['f1']:.3f} | "
            f"{100*row['auroc']:.3f} | {row['bce']:.6f} |"
        )
    lines += [
        "", f"三种子均值 ± 样本标准差：accuracy {cell('accuracy')}%，F1 {cell('f1')}%，"
        f"AUROC {cell('auroc')}%，BCE {cell('bce', 1.0)}。", "",
        "| source | rows | accuracy (%) | F1 (%) | AUROC (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in source_names:
        item = source_aggregate[source]
        count = source_by_seed[SEEDS[0]][source]["n"]
        def source_cell(key: str) -> str:
            row = item[key]
            if row["mean"] is None:
                return "—"
            suffix = f" ± {100*row['sample_std']:.3f}" if row["sample_std"] is not None else ""
            return f"{100*row['mean']:.3f}{suffix}"
        lines.append(f"| {source} | {count} | {source_cell('accuracy')} | {source_cell('f1')} | {source_cell('auroc')} |")
    lines += ["", "成对错误互补：", ""]
    for pair in pairs:
        row = pair["overall"]
        lines.append(
            f"- seed {pair['seed_a']} vs {pair['seed_b']}：decision agreement {100*row['decision_agreement']:.3f}%，"
            f"only-correct {row['a_only_correct']}/{row['b_only_correct']}，all-wrong {row['both_wrong']}，"
            f"McNemar p={row['mcnemar_exact_two_sided_p']:.6g}。"
        )
    lines += [
        "", f"三种子全错 {consensus['all_three_wrong']} 条；至少一个种子正确的 oracle 上界 "
        f"{100*consensus['oracle_accuracy']:.3f}%。", "", result["inference_note"], "",
    ]
    mismatches = [name for name, row in result["registered_code_comparison"].items() if not row["matches"]]
    if mismatches:
        lines += [
            "代码登记披露：protocol 中以下旧登记哈希与本次锁定的实际推理代码不同："
            + "、".join(mismatches) + "。逐 run provenance 已直接核验 trainer/head；完整实际推理代码哈希见 test_barrier.json。",
            "",
        ]
    return result, "\n".join(lines)


def aggregate(study: Path) -> dict:
    study = Path(study).resolve()
    result, report = build_aggregate(study)
    result_path, report_path = study / "e03_multiseed_results.json", study / "e03_multiseed_report.md"
    write_json(result_path, result)
    report_path.write_text(report, encoding="utf-8")
    completion = {
        "schema": "hanguard_e03_lora_multiseed_completion_v1", "state": "complete",
        "arm": ARM, "seeds": list(SEEDS),
        "test_barrier_sha256": sha256_file(study / "test_barrier.json"),
        "results_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
        "receipts_sha256": result["evaluation_receipts_sha256"],
    }
    completion_path = study / "e03_multiseed_completion.json"
    if completion_path.exists():
        require(read_json(completion_path) == completion, "Existing E03 completion differs from current evidence")
    else:
        write_immutable_json(completion_path, completion)
    return result


def run(study: Path) -> dict:
    lock_test(study)
    for seed in SEEDS:
        evaluate_seed(study, seed)
    return aggregate(study)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("amend-provenance", "train", "lock", "evaluate", "aggregate", "run"),
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "amend-provenance":
        require(args.seed is None, "amend-provenance does not accept --seed")
        output = create_training_code_provenance_amendment(args.study)
    elif args.action == "train":
        require(args.seed is not None, "--seed 44 is required for the amended train entrypoint")
        output = train_seed44(args.study, args.seed)
    elif args.action == "lock":
        output = lock_test(args.study)
    elif args.action == "evaluate":
        require(args.seed is not None, "--seed is required for evaluate")
        output = evaluate_seed(args.study, args.seed)
    elif args.action == "aggregate":
        output = aggregate(args.study)
    else:
        require(args.seed is None, "run evaluates all registered seeds; omit --seed")
        output = run(args.study)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
