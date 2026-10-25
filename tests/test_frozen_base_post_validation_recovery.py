"""CPU checks for post-validation continuation without refitting or early test."""
import fcntl
from pathlib import Path

import pytest

from scripts.hanguard import frozen_base_post_validation_recovery as recovery


def fake_study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    recovery.base.dump(study / "protocol.json", {})
    recovery.base.dump(study / "state.json", dict(state="requires_review", training_started=False,
        formal_test_opened=False, reason="Foreign process entered an active GPU"))
    recovery.base.dump(study / "parallel_execution/queue_state.json", dict(requires_review=True,
        reason="Foreign process entered an active GPU", phase="extract_validation"))
    recovery.base.dump(study / "recovery_002/control_protocol.json", {})
    monkeypatch.setattr(recovery.queue, "check_control2", lambda *a: {})
    return study


def pipeline(tmp_path, monkeypatch, fit_locks=True):
    study = fake_study(tmp_path, monkeypatch)
    events, selected = [], False
    def merge(path, split):
        events.append(("merge", split))
    monkeypatch.setattr(recovery.features, "merge_split", merge)
    def lock(path):
        if not selected:
            raise ValueError("No completed selection lock")
    monkeypatch.setattr(recovery.base, "validate_selection_lock", lock)
    def extract(path, split, queue):
        lock(path)
        events.append(("extract", split))
        assert split == "test"
        assert queue is recovery.queue
    monkeypatch.setattr(recovery.previous, "extract", extract)
    def single(path, phase, factory):
        nonlocal selected
        events.append(("single", phase))
        cmd = factory("new_head_worker", 0)
        assert Path(cmd[1]).name == "frozen_base_parallel_study.py"
        assert cmd[cmd.index("--phase") + 1] == phase
        assert cmd[cmd.index("--study") + 1] == str(study)
        if phase == "fit":
            selected = fit_locks
        else:
            lock(path)
            recovery.base.dump(path / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(recovery.queue, "run_single", single)
    return study, events


def test_merge_existing_validation_then_original_fit_and_locked_test(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch)
    recovery.run(study)
    assert events == [("merge", "validation"), ("single", "fit"), ("extract", "test"), ("single", "evaluate")]
    assert recovery.read(study / "state.json")["state"] == "completed"
    assert recovery.read(study / "recovery_002/completion.json")["original_registered_gpu_entrypoints"] is True


def test_no_selection_lock_prevents_test_extraction(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch, fit_locks=False)
    with pytest.raises(ValueError, match="selection lock"):
        recovery.run(study)
    assert ("extract", "test") not in events
    assert ("single", "evaluate") not in events


def test_already_started_head_fit_is_never_silently_retried(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch)
    recovery.base.mark_state(study, "requires_review", training_started=True)
    with pytest.raises(ValueError, match="cannot silently repeat"):
        recovery.run(study)
    assert events == []


def test_second_coordinator_cannot_take_current_study(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch)
    with (study / "parallel_coordinator.pid").open("a+") as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another coordinator"):
            recovery.run(study)
    assert events == []


@pytest.mark.parametrize("field", ("training_started", "formal_test_opened"))
def test_registration_rejects_prior_head_or_test_access(tmp_path, monkeypatch, field):
    study = tmp_path / "study"
    study.mkdir()
    recovery.base.dump(study / "state.json", dict(state="requires_review",
        reason="Foreign process entered an active GPU", **{field: True}))
    recovery.base.dump(study / "parallel_execution/queue_state.json", dict(requires_review=True,
        reason="Foreign process entered an active GPU", phase="extract_validation"))
    monkeypatch.setattr(recovery.queue_v2, "check_control", lambda *a: {})
    monkeypatch.setattr(recovery.base, "load_protocol", lambda *a, **k: {})
    with pytest.raises(ValueError, match="No head fit or test"):
        recovery.register(study)
    assert not (study / "recovery_002").exists()
