import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.hanguard import finalize_e03_lora_multiseed as finalizer
from scripts.hanguard import repaired_study


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def make_study(
    tmp_path: Path,
    complete_seeds=(42, 43, 44),
    *,
    with_amendment=True,
    with_seed44_receipt=True,
) -> Path:
    study = tmp_path / "study"
    data = tmp_path / "data"; model = tmp_path / "model"
    data.mkdir(); model.mkdir()
    pd.DataFrame({
        "base_id": [f"t{i}" for i in range(6)],
        "source": ["a", "a", "b", "b", "c", "c"],
        "prompt_harm_label": ["unharmful", "harmful"] * 3,
    }).to_parquet(data / "test.parquet", index=False)
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / "weights-1.safetensors").write_bytes(b"model shard")
    dump(model / "model.safetensors.index.json", {"weight_map": {"x": "weights-1.safetensors"}})
    code_hashes = {
        "repaired_study.py": finalizer.sha256_file(Path(repaired_study.__file__)),
        "repaired_heads.py": finalizer.sha256_file(finalizer.ROOT / "scripts/hanguard/repaired_heads.py"),
        "hanguard_model.py": finalizer.sha256_file(finalizer.ROOT / "hanguard_model.py"),
    }
    protocol = {
        "data_dir": str(data), "model_path": str(model), "seeds": [42, 43, 44],
        "all_arms": ["E03"], "warm_head_epochs": 1, "epochs": 1,
        "rows": {"train": 3, "validation": 4, "test": 6},
        "data_sha256": {
            "train": "train-hash", "validation": "validation-hash",
            "test": finalizer.sha256_file(data / "test.parquet"),
        },
        "model_config_sha256": finalizer.sha256_file(model / "config.json"),
        "tokenizer_sha256": finalizer.sha256_file(model / "tokenizer.json"),
        "code_sha256": {**code_hashes, "hanguard_model.py": finalizer.STALE_REGISTERED_HANGUARD_MODEL_SHA256},
        "created_at": "2026-10-12T10:07:41.868542+00:00",
        "registered_at": "2026-10-12T10:07:41.868553+00:00",
    }
    arms = {"E03": {"lora": True, "mode": "last", "readout": "mlp", "objective": "bce"}}
    dump(study / "protocol.json", protocol); dump(study / "arms.json", arms)
    dump(study / "registration.json", {
        "state": "locked_for_training", "arm": "E03", "seeds": [42, 43, 44],
        "protocol_sha256": finalizer.sha256_file(study / "protocol.json"),
        "arms_sha256": finalizer.sha256_file(study / "arms.json"),
    })
    labels = np.array([0, 0, 1, 1])
    probability = np.array([.1, .2, .5, .9])
    threshold = repaired_study.accuracy_threshold(labels, probability)
    for seed in complete_seeds:
        directory = study / "runs" / f"E03_s{seed}"
        directory.mkdir(parents=True)
        (directory / "head.pt").write_bytes(f"head-{seed}".encode())
        (directory / "adapter.pt").write_bytes(f"adapter-{seed}".encode())
        validation = pd.DataFrame({
            "base_id": [f"v{i}" for i in range(4)], "source": ["a", "a", "b", "b"],
            "label": labels, "prompt_tokens": [2, 3, 4, 5], "probability": probability,
        })
        validation.to_csv(directory / "validation.csv", index=False)
        common = {
            "arm": "E03", "seed": seed,
            "protocol_sha256": finalizer.sha256_file(study / "protocol.json"),
            "arms_sha256": finalizer.sha256_file(study / "arms.json"),
            "train_sha256": "train-hash", "validation_sha256": "validation-hash",
        }
        selection = {
            **common, "fresh_base": True, "completed_training_rows": 6,
            "validation_bce": .1 + seed / 10000, "best_epoch": {"epoch": 1},
            "threshold": threshold, "metrics": repaired_study.metrics(labels, probability, threshold),
            "checkpoint_hashes": {
                "head.pt": finalizer.sha256_file(directory / "head.pt"),
                "adapter.pt": finalizer.sha256_file(directory / "adapter.pt"),
            },
        }
        provenance = {
            **common, "fresh_base": True, "old_checkpoint_reuse": False,
            "code_sha256": code_hashes["repaired_study.py"],
            "heads_sha256": code_hashes["repaired_heads.py"],
        }
        dump(directory / "selection.json", selection)
        dump(directory / "provenance.json", provenance)
        dump(directory / "status.json", {"state": "selected", "arm": "E03", "seed": seed})
    if with_amendment:
        finalizer.create_training_code_provenance_amendment(study)
    if with_seed44_receipt:
        assert with_amendment
        finalizer.create_seed44_launch_receipt(study)
    return study


def test_training_code_amendment_is_read_only_and_captures_stale_registry_evidence(tmp_path):
    study = make_study(tmp_path)
    path = study / "training_code_provenance_amendment.json"
    value = finalizer.verify_training_code_provenance_amendment(study)
    assert path.stat().st_mode & 0o777 == 0o444
    assert value["protocol"]["sha256"] == finalizer.sha256_file(study / "protocol.json")
    assert value["hanguard_model"]["protocol_registered_sha256"] == finalizer.STALE_REGISTERED_HANGUARD_MODEL_SHA256
    assert value["hanguard_model"]["current_sha256"] == finalizer.EXPECTED_HANGUARD_MODEL_SHA256
    assert value["hanguard_model"]["mtime_precedes_protocol_registration"] is True
    assert value["git_evidence"]["head_sha256"] == finalizer.EXPECTED_HANGUARD_MODEL_SHA256
    assert value["git_evidence"]["all_history_sha256_identical"] is True
    assert [row["seed"] for row in value["pre_existing_runs"]] == [42, 43]
    before = path.read_bytes()
    assert finalizer.create_training_code_provenance_amendment(study) == value
    assert path.read_bytes() == before


def test_lock_requires_amendment_before_test_access(tmp_path, monkeypatch):
    study = make_study(tmp_path, with_amendment=False, with_seed44_receipt=False)
    touched = False

    def forbidden(_protocol):
        nonlocal touched
        touched = True
        raise AssertionError("test must not be touched")

    monkeypatch.setattr(finalizer, "_test_dataset_record", forbidden)
    with pytest.raises(finalizer.AuditError, match="amendment.json is required"):
        finalizer.lock_test(study)
    assert touched is False
    assert not (study / "test_barrier.json").exists()


def test_seed44_training_wrapper_checks_amendment_and_writes_0444_receipt(tmp_path, monkeypatch):
    study = make_study(
        tmp_path, complete_seeds=(42, 43), with_amendment=True, with_seed44_receipt=False,
    )
    calls = []
    monkeypatch.setattr(repaired_study, "train", lambda args: calls.append((args.arm, args.seed)))
    receipt = finalizer.train_seed44(study, 44)
    path = finalizer.seed44_receipt_path(study)
    assert calls == [("E03", 44)]
    assert path.stat().st_mode & 0o777 == 0o444
    assert receipt["imported_hanguard_model_sha256"] == finalizer.EXPECTED_HANGUARD_MODEL_SHA256
    assert receipt["amendment_sha256"] == finalizer.sha256_file(
        study / "training_code_provenance_amendment.json"
    )


def test_lock_requires_seed44_amended_launch_receipt_before_test_access(tmp_path, monkeypatch):
    study = make_study(tmp_path, with_amendment=True, with_seed44_receipt=False)
    touched = False

    def forbidden(_protocol):
        nonlocal touched
        touched = True
        raise AssertionError("test must not be touched")

    monkeypatch.setattr(finalizer, "_test_dataset_record", forbidden)
    with pytest.raises(finalizer.AuditError, match="launch receipt"):
        finalizer.lock_test(study)
    assert touched is False
    assert not (study / "test_barrier.json").exists()


def test_amendment_rejects_current_model_hash_drift(tmp_path, monkeypatch):
    study = make_study(tmp_path)
    real_sha = finalizer.sha256_file
    model = (finalizer.ROOT / "hanguard_model.py").resolve()

    def drift(path):
        return "0" * 64 if Path(path).resolve() == model else real_sha(path)

    monkeypatch.setattr(finalizer, "sha256_file", drift)
    with pytest.raises(finalizer.AuditError, match="hash/mtime drifted"):
        finalizer.verify_training_code_provenance_amendment(study)


def test_lock_hard_refuses_missing_selection_before_any_test_access(tmp_path, monkeypatch):
    study = make_study(tmp_path, complete_seeds=(42, 43))
    touched = False

    def forbidden(_protocol):
        nonlocal touched
        touched = True
        raise AssertionError("test must not be touched")

    monkeypatch.setattr(finalizer, "_test_dataset_record", forbidden)
    with pytest.raises(finalizer.AuditError, match="missing"):
        finalizer.lock_test(study)
    assert touched is False
    assert not (study / "test_barrier.json").exists()


def test_barrier_locks_all_three_selections_checkpoints_code_and_test(tmp_path):
    study = make_study(tmp_path)
    barrier = finalizer.lock_test(study)
    assert barrier["locked"] is True
    assert tuple(barrier["seeds"]) == (42, 43, 44)
    assert len(barrier["selection_sha256"]) == 3
    assert len(barrier["checkpoint_sha256"]) == 6
    assert {key for key in barrier["artifacts"] if key.startswith("code/")} == {
        "code/finalizer", "code/repaired_study", "code/repaired_heads", "code/hanguard_model",
    }
    assert barrier["test_dataset"]["sha256"] == finalizer.read_json(study / "protocol.json")["data_sha256"]["test"]
    original_barrier = (study / "test_barrier.json").read_bytes()
    (study / "runs/E03_s43/adapter.pt").write_bytes(b"changed")
    with pytest.raises(finalizer.AuditError, match="checkpoint changed"):
        finalizer.verify_test_barrier(study)
    assert (study / "test_barrier.json").read_bytes() == original_barrier


def test_evaluate_never_delegates_when_complete_barrier_no_longer_verifies(tmp_path, monkeypatch):
    study = make_study(tmp_path)
    finalizer.lock_test(study)
    (study / "runs/E03_s44/selection.json").unlink()
    called = False

    def forbidden(_args: argparse.Namespace):
        nonlocal called
        called = True

    monkeypatch.setattr(repaired_study, "evaluate", forbidden)
    with pytest.raises(finalizer.AuditError, match="missing"):
        finalizer.evaluate_seed(study, 42)
    assert called is False


def write_test_output(study: Path, seed: int, probabilities) -> None:
    directory = study / "runs" / f"E03_s{seed}"
    selection = finalizer.read_json(directory / "selection.json")
    labels = np.array([0, 1, 0, 1, 0, 1])
    frame = pd.DataFrame({
        "base_id": [f"t{i}" for i in range(6)],
        "source": ["a", "a", "b", "b", "c", "c"],
        "label": labels, "prompt_tokens": [2, 3, 4, 5, 6, 7],
        "probability": np.asarray(probabilities, dtype=float),
    })
    frame.to_csv(directory / "test.csv", index=False)
    saved = repaired_study.metrics(labels, frame.probability, selection["threshold"])
    dump(directory / "test_results.json", {
        "arm": "E03", "seed": seed, "metrics": saved,
        "test_sha256": finalizer.read_json(study / "protocol.json")["data_sha256"]["test"],
        "threshold": selection["threshold"], "checkpoint_hashes": selection["checkpoint_hashes"],
    })
    barrier = finalizer.read_json(study / "test_barrier.json")
    receipt = finalizer._receipt_value(study, seed, barrier)
    finalizer.write_immutable_json(study / "finalization" / f"E03_s{seed}_evaluation_receipt.json", receipt)


def test_e03_only_aggregate_has_seed_source_and_paired_diagnostics(tmp_path):
    study = make_study(tmp_path)
    finalizer.lock_test(study)
    write_test_output(study, 42, [.1, .9, .2, .8, .2, .8])
    write_test_output(study, 43, [.1, .9, .7, .8, .2, .4])
    write_test_output(study, 44, [.1, .4, .2, .8, .9, .8])
    # A poison reference confirms the E03 finalizer has no R02 dependency.
    (study / "references/R02").mkdir(parents=True)
    (study / "references/R02/DO_NOT_READ").write_bytes(b"irrelevant")

    result = finalizer.aggregate(study)
    assert result["seeds"] == [42, 43, 44]
    assert result["n_independent_lora_seeds"] == 3
    assert result["aggregate_across_lora_seeds"]["accuracy"]["n_seeds"] == 3
    assert set(result["by_source_aggregate"]) == {"a", "b", "c"}
    assert len(result["paired_seed_diagnostics"]) == 3
    assert all("mcnemar_exact_two_sided_p" in row["overall"] for row in result["paired_seed_diagnostics"])
    assert result["scope_note"].startswith("No R02")
    assert (study / "e03_multiseed_completion.json").is_file()
    # Deterministic rerun validates the immutable completion instead of changing it.
    assert finalizer.aggregate(study) == result


def test_run_locks_before_first_gpu_evaluation(tmp_path, monkeypatch):
    study = make_study(tmp_path)
    events = []

    def fake_evaluate(path, seed):
        assert (Path(path) / "test_barrier.json").is_file()
        events.append(seed)
        return {"seed": seed}

    monkeypatch.setattr(finalizer, "evaluate_seed", fake_evaluate)
    monkeypatch.setattr(finalizer, "aggregate", lambda _study: {"done": True})
    assert finalizer.run(study) == {"done": True}
    assert events == [42, 43, 44]
