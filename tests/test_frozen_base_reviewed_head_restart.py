"""Reviewed restart keeps history, original commands and the unopened test gate."""
from pathlib import Path

import pytest

from scripts.hanguard import frozen_base_reviewed_head_restart as restart


def pipeline(tmp_path, monkeypatch, selected_after_fit=True):
    study = tmp_path / "study"
    study.mkdir()
    restart.base.dump(study / "protocol.json", {})
    restart.base.dump(study / "state.json", dict(state="requires_review", training_started=True,
        formal_test_opened=False))
    restart.base.dump(study / "recovery_003/control_protocol.json", {})
    control = dict(runtime_environment={"HANGUARD_RESTART_TEST_TMP": str(tmp_path)})
    monkeypatch.setenv("HANGUARD_RESTART_TEST_TMP", "restore_after_test")
    monkeypatch.setattr(restart.queue, "check_control3", lambda *a: control)
    events, selected = [], False
    def selection(path):
        if not selected:
            raise ValueError("No selected checkpoint lock")
    monkeypatch.setattr(restart.base, "validate_selection_lock", selection)
    def single(path, phase, factory):
        nonlocal selected
        events.append(("single", phase))
        cmd = factory("reviewed_head_worker", 6)
        assert Path(cmd[1]).name == "frozen_base_parallel_study.py"
        assert cmd[cmd.index("--phase") + 1] == phase
        state = restart.read(path / "state.json")
        assert state["training_started"] is True
        assert state["head_fit_attempt"] == 2
        if phase == "fit":
            selected = selected_after_fit
        else:
            selection(path)
            restart.base.dump(path / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(restart.queue, "run_single", single)
    def extract(path, split, queue):
        selection(path)
        assert split == "test" and queue is restart.queue
        events.append(("extract", split))
    monkeypatch.setattr(restart.previous.previous, "extract", extract)
    return study, events


def test_reviewed_fresh_fit_keeps_sticky_started_and_original_test_barrier(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch)
    restart.run(study)
    assert events == [("single", "fit"), ("extract", "test"), ("single", "evaluate")]
    complete = restart.read(study / "recovery_003/completion.json")
    assert complete["head_fit_attempt"] == 2 and complete["partial_checkpoint_reuse"] is False
    assert restart.read(study / "state.json")["training_started"] is True


def test_no_lock_prevents_test_extraction_after_reviewed_fit(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch, selected_after_fit=False)
    with pytest.raises(ValueError, match="checkpoint lock"):
        restart.run(study)
    assert ("extract", "test") not in events


def test_current_partial_heads_never_get_silently_adopted(tmp_path, monkeypatch):
    study, events = pipeline(tmp_path, monkeypatch)
    (study / "heads").mkdir()
    with pytest.raises(ValueError, match="cannot be silently repeated"):
        restart.run(study)
    assert events == []


def test_registration_refuses_any_published_epoch_without_moving_it(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    restart.base.dump(study / "state.json", dict(state="requires_review", training_started=True,
        formal_test_opened=False))
    restart.base.dump(study / "parallel_execution/queue_state.json", dict(phase="fit", requires_review=True,
        reason="Foreign process entered an active GPU"))
    restart.base.dump(study / "recovery_002/scheduler_recovery_entry.json", dict(coordinator_birth={}))
    restart.base.dump(study / "heads/fit_started.json", {})
    restart.base.dump(study / "heads/budget_12/layer_history.json", [{"epoch": 1}])
    monkeypatch.setattr(restart.previous.queue, "check_control2", lambda *a: {})
    monkeypatch.setattr(restart.base, "load_protocol", lambda *a, **k: {})
    monkeypatch.setattr(restart.previous.previous, "original_worker_exited", lambda *a: True)
    with pytest.raises(ValueError, match="zero published epochs"):
        restart.register(study)
    assert (study / "heads/budget_12/layer_history.json").exists()
    assert not (study / "fit_attempts").exists()


def test_private_scratch_paths_stay_in_study(tmp_path):
    environment = restart.scratch_environment(tmp_path)
    assert set(environment) == {"TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"}
    for value in environment.values():
        assert Path(value).is_relative_to(tmp_path.resolve()) and Path(value).is_dir()
