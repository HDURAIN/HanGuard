"""CPU recovery checks: no premature test, no head retry, preserved evidence."""
import json
import os
from pathlib import Path

import pytest

from scripts.hanguard import frozen_base_recovery_coordinator as recovery
from scripts.hanguard import frozen_base_parallel_queue_v2 as queue
from scripts.hanguard import frozen_base_readout_admission_v2 as admission


def fixture_study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    recovery.base.dump(study / "protocol.json", dict(inputs={}, settings={"seed": 42}, rows={"train": 1}))
    root_sha = recovery.base.sha(study / "protocol.json")
    recovery.base.dump(study / "state.json", dict(state="requires_review", training_started=False,
        formal_validation_opened=False, formal_test_opened=False,
        reason="Own worker identity became unverifiable"))
    recovery.base.dump(study / "parallel_execution/queue_state.json", dict(
        requires_review=True, reason="Own worker identity became unverifiable", phase="extract_train", monitor_pid=900000))
    recovery.base.dump(study / "parallel_execution/termination_evidence.json", dict(workers=[dict(
        worker_id="old_worker", pid=900001, exit_code=0, verified_original_child_exited=True,
        verified_SIGTERM_sent=False, verified_SIGKILL_sent=False)]))
    ledger = study / "parallel_execution/workers/old_worker.json"
    recovery.base.dump(ledger, dict(worker_id="old_worker", root_protocol_sha256=root_sha,
        process_identity=dict(pid=900001, uid=os.getuid(), start_time_ticks=123)))
    recovery.base.dump(ledger.with_suffix(".registration.json"), dict(ledger_sha256=recovery.base.sha(ledger)))
    recovery.base.dump(study / "parallel_resource_protocol.json", dict(bindings={str(study / "protocol.json"): root_sha}))
    recovery.base.dump(study / "parallel_resource_registration.json", dict(
        protocol_sha256=recovery.base.sha(study / "parallel_resource_protocol.json")))
    recovery.base.dump(study / "parallel_features/train/batch_plan.json", dict(chunks=[dict(chunk_id=0)]))
    receipt = recovery.features.chunk_directory(study, "train", 0) / "receipt.json"
    recovery.base.dump(receipt, dict(complete=True, shape=[1, 32, 2560]))
    (study / "coordinator.log").write_text("Original normal-exit race traceback\n")
    monkeypatch.setattr(recovery.base, "load_protocol", lambda *a, **k: dict(rows={"train": 1}, settings={"seed": 42}, inputs={}))
    monkeypatch.setattr(recovery.features, "split_progress", lambda *a: dict(complete=True, completed_rows=1))
    monkeypatch.setattr(admission, "process_birth", lambda pid: None)
    monkeypatch.setattr(queue, "check_control", lambda *a: {})
    return study


def test_recovery_registration_preserves_original_stop_and_receipts(tmp_path, monkeypatch):
    study = fixture_study(tmp_path, monkeypatch)
    original = (study / "parallel_execution/queue_state.json").read_bytes()
    value = recovery.register_recovery(study)
    assert value["policy"]["max_cuda_workers"] == 8
    assert value["policy"]["reserved_gpus"] == []
    assert value["prior_review"]["final_worker_exit_code"] == 0
    assert value["automatic_head_retry"] is False
    assert (study / "recovery_001/queue_state.before.json").read_bytes() == original
    assert (study / "parallel_execution/queue_state.json").read_bytes() == original
    assert str(Path(queue.__file__).resolve()) in value["bindings"]
    assert str(Path(recovery.__file__).resolve()) in value["bindings"]
    assert json.loads((study / "recovery_control_registration.json").read_text())["protocol_sha256"] == recovery.base.sha(study / "recovery_control_protocol.json")
    with pytest.raises(FileExistsError, match="immutable"):
        recovery.register_recovery(study)


@pytest.mark.parametrize("bad_state", ("training_started", "formal_validation_opened", "formal_test_opened"))
def test_no_continuation_if_scientific_phase_already_started(tmp_path, monkeypatch, bad_state):
    study = fixture_study(tmp_path, monkeypatch)
    state = recovery.read(study / "state.json")
    state[bad_state] = True
    recovery.base.dump(study / "state.json", state)
    with pytest.raises(ValueError, match="head training or test"):
        recovery.register_recovery(study)
    assert not (study / "recovery_control_protocol.json").exists()


@pytest.mark.parametrize("bad_field,bad_value", (("exit_code", 1), ("verified_SIGTERM_sent", True),
    ("verified_original_child_exited", False)))
def test_requires_unsignalled_known_successful_exit(tmp_path, monkeypatch, bad_field, bad_value):
    study = fixture_study(tmp_path, monkeypatch)
    evidence = recovery.read(study / "parallel_execution/termination_evidence.json")
    evidence["workers"][0][bad_field] = bad_value
    recovery.base.dump(study / "parallel_execution/termination_evidence.json", evidence)
    with pytest.raises(ValueError, match="successful worker exit"):
        recovery.register_recovery(study)


def test_live_original_worker_blocks_recovery(tmp_path, monkeypatch):
    study = fixture_study(tmp_path, monkeypatch)
    monkeypatch.setattr(admission, "process_birth", lambda pid: None if pid == 900000 else dict(
        pid=pid, uid=os.getuid(), start_time_ticks=123, state="R"))
    with pytest.raises(ValueError, match="live or unrelated"):
        recovery.register_recovery(study)


def test_same_birth_changed_owner_is_unknown_not_pid_reuse():
    expected = dict(pid=1, uid=2, start_time_ticks=3)
    with pytest.raises(ValueError, match="changed owner"):
        recovery.original_worker_exited(expected, birth_reader=lambda _: dict(
            pid=1, uid=999, start_time_ticks=3, state="R"))
    assert recovery.original_worker_exited(expected, birth_reader=lambda _: dict(
        pid=1, uid=999, start_time_ticks=4, state="R")) is True


def fake_pipeline(tmp_path, monkeypatch, locked_after_fit):
    study = fixture_study(tmp_path, monkeypatch)
    recovery.base.dump(study / "recovery_control_protocol.json", {})
    events = []
    locked = False
    monkeypatch.setattr(recovery.features, "merge_split", lambda p, split: events.append(("merge", split)))
    def selection(path):
        if not locked:
            raise ValueError("No selection lock")
    def prepare(path, split):
        if split == "test":
            selection(path)
        events.append(("prepare", split))
    monkeypatch.setattr(recovery.features, "prepare_split", prepare)
    monkeypatch.setattr(recovery.features, "split_complete", lambda *a: True)
    monkeypatch.setattr(recovery.features, "has_work", lambda *a: True)
    monkeypatch.setattr(recovery.base, "validate_selection_lock", selection)
    def workers(path, phase, factory, done, **kwargs):
        assert kwargs["max_workers"] == 8
        command = factory("new_worker", 0)
        assert Path(command[1]).name == "frozen_base_parallel_features.py"
        assert "--study" in command
        assert done()
        events.append(("workers", phase))
    monkeypatch.setattr(queue, "run_workers", workers)
    def single(path, phase, factory):
        nonlocal locked
        command = factory("head_worker", 0)
        assert Path(command[1]).name == "frozen_base_parallel_study.py"
        events.append(("single", phase))
        if phase == "fit":
            locked = locked_after_fit
        else:
            selection(path)
            recovery.base.dump(path / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(queue, "run_single", single)
    return study, events


def test_recovery_merges_train_then_uses_original_gpu_commands_and_test_lock(tmp_path, monkeypatch):
    study, events = fake_pipeline(tmp_path, monkeypatch, True)
    recovery.run(study)
    assert events[0] == ("merge", "train")
    assert ("prepare", "train") not in events
    assert events.index(("single", "fit")) < events.index(("prepare", "test"))
    assert events.index(("merge", "test")) < events.index(("single", "evaluate"))
    assert recovery.read(study / "state.json")["state"] == "completed"


def test_failed_fit_lock_prevents_test_preparation(tmp_path, monkeypatch):
    study, events = fake_pipeline(tmp_path, monkeypatch, False)
    with pytest.raises(ValueError, match="selection lock"):
        recovery.run(study)
    assert ("prepare", "test") not in events
    assert ("single", "evaluate") not in events
