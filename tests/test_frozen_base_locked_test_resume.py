"""CPU-only checks for continuation after selection but before test evaluation."""
import copy
import os
from pathlib import Path
import time

import pytest

from scripts.hanguard import frozen_base_locked_test_resume as resume
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard import frozen_base_parallel_features as features
from scripts.hanguard import frozen_base_parallel_study as original


@pytest.fixture
def study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    write = resume.write
    root = study / "protocol.json"
    write(root, dict(settings=dict(seed=42)))
    write(study / "selection_lock.json", dict(complete=True, immutable="selected_weights_and_thresholds"))
    write(study / "state.json", dict(state="extracting_test", formal_test_opened=True,
                                     training_started=True, head_fit_attempt=3))
    prior = study / resume.cooperative.FOLDER
    prior.mkdir()
    write(prior / "control_protocol.json", dict(unchanged_scientific_settings=dict(seed=42), runtime_environment={}))
    write(prior / "control_registration.json", dict(protocol_sha256=resume.sha(prior / "control_protocol.json")))
    birth = dict(pid=300, uid=os.getuid(), start_time_ticks=99, state="R")
    write(prior / "coordinator_started.json", dict(birth=birth))
    write(prior / "scheduler_entry.json", dict(coordinator_birth=birth))
    (prior / "coordinator.log").write_text("No exception was recorded.\n")
    workers = study / "parallel_execution/workers"
    workers.mkdir(parents=True)
    identity = dict(pid=400, uid=os.getuid(), start_time_ticks=120)
    path = workers / "fit_0043_gpu3.json"
    write(path, dict(schema_version=1, worker_id="fit_0043_gpu3", phase="fit",
                     process_identity=identity, root_protocol_sha256=resume.sha(root)))
    write(path.with_suffix(".registration.json"), dict(ledger_sha256=resume.sha(path)))
    write(path.with_suffix(".complete.json"), dict(complete=True, phase="fit", pid=400,
                                                 protocol_sha256=resume.sha(root)))
    write(study / "parallel_execution/queue_state.json", dict(
        phase="extract_test", active_workers=0, workers=[], requires_review=False,
        runtime_control_sha256=resume.sha(prior / "control_protocol.json"),
        completed_workers=[dict(worker_id="fit_0043_gpu3", pid=400, phase="fit", exit_code=0)]))
    prepared_folder = study / "parallel_features/test"
    prepared = dict(complete=True, test_selection_lock_sha256=resume.sha(study / "selection_lock.json"))
    write(prepared_folder / "prepared_manifest.json", prepared)
    write(prepared_folder / "batch_plan.json", dict(chunks=[]))
    monkeypatch.setattr(resume.cooperative, "check_control", lambda s: resume.read(prior / "control_protocol.json"))
    monkeypatch.setattr(base, "validate_selection_lock", lambda s: resume.read(Path(s) / "selection_lock.json"))
    monkeypatch.setattr(features, "load_prepared", lambda s, split: (copy.deepcopy(prepared), None, None))
    monkeypatch.setattr(resume.queue.admission, "process_birth", lambda pid: None)
    return study


def test_registration_preserves_selected_artifacts_and_original_control(study):
    paths = [study / "selection_lock.json", resume.cooperative.control_path(study), study / "protocol.json"]
    before = {str(p): p.read_bytes() for p in paths}
    control = resume.register(study)
    assert control["scope"] == resume.SCOPE
    assert control["policy"]["active_foreign"] == "record_only"
    assert control["prior_workers_verified_exited"] is True
    assert control["selection_lock_sha256"] == resume.sha(study / "selection_lock.json")
    assert before == {str(p): p.read_bytes() for p in paths}
    assert resume.check_control(study) == control


def test_registration_refuses_a_live_old_coordinator(study, monkeypatch):
    monkeypatch.setattr(resume.queue.admission, "process_birth", lambda pid:
        dict(pid=300, uid=os.getuid(), start_time_ticks=99, state="S"))
    with pytest.raises(ValueError, match="coordinator is still alive"):
        resume.register(study)
    assert not (study / resume.FOLDER).exists()


def test_registration_refuses_a_live_old_worker(study, monkeypatch):
    def birth(pid):
        return dict(pid=400, uid=os.getuid(), start_time_ticks=120, state="S") if pid == 400 else None
    monkeypatch.setattr(resume.queue.admission, "process_birth", birth)
    with pytest.raises(resume.queue.ResourceReviewRequired, match="worker is still live"):
        resume.register(study)
    assert not (study / resume.FOLDER).exists()


@pytest.mark.parametrize("artifact", ["analysis/test_access.json", "analysis/results.json",
                                     "parallel_execution/workers/evaluate_0044_gpu0.json"])
def test_registration_refuses_prior_or_partial_evaluation(study, artifact):
    resume.write(study / artifact, dict(phase="evaluate"))
    with pytest.raises(ValueError, match="evaluation"):
        resume.register(study)
    assert not (study / resume.FOLDER).exists()


def test_registration_refuses_prepared_payload_under_another_lock(study, monkeypatch):
    monkeypatch.setattr(features, "load_prepared", lambda s, split:
        (dict(test_selection_lock_sha256="different"), None, None))
    with pytest.raises(ValueError, match="different selection"):
        resume.register(study)
    assert not (study / resume.FOLDER).exists()


def test_control_rejects_changed_selection_without_rewriting_it(study):
    resume.register(study)
    resume.write(study / "selection_lock.json", dict(complete=True, immutable="changed"))
    with pytest.raises(ValueError, match="identity or scope changed"):
        resume.check_control(study)
    assert resume.read(study / "selection_lock.json")["immutable"] == "changed"


def test_control_poll_does_not_rehash_all_selection_artifacts(study, monkeypatch):
    resume.register(study)
    monkeypatch.setattr(base, "validate_selection_lock", lambda s: pytest.fail("No full lock rehash per poll"))
    resume.check_control(study)


def test_progress_uses_only_plan_receipts_and_births_without_full_barrier(study, monkeypatch):
    folder = study / "parallel_features/test"
    chunk = dict(chunk_id=0, group_start=0, group_stop=1, rows=3)
    prepared = resume.read(folder / "prepared_manifest.json")
    prepared.update(schema=features.SCHEMA, split="test", plan_sha256="original-plan",
                    protocol_sha256=resume.sha(study / "protocol.json"))
    resume.write(folder / "prepared_manifest.json", prepared)
    resume.write(folder / "batch_plan.json", dict(chunks=[chunk]))
    resume.register(study)
    monkeypatch.setattr(base, "validate_selection_lock", lambda s: pytest.fail("No full lock rehash while polling"))
    monkeypatch.setattr(features, "barrier", lambda *a: pytest.fail("No original full barrier while polling"))
    monkeypatch.setattr(features, "load_prepared", lambda *a: pytest.fail("No token payload reload while polling"))
    monkeypatch.setattr(features, "split_progress", lambda *a: pytest.fail("No full-cache progress call"))
    progress = resume.test_progress(study)
    assert not progress["complete"] and progress["pending_chunks"] == 1
    directory = features.chunk_directory(study, "test", 0)
    resume.write(directory / "claim.json", dict(identity=dict(pid=999, uid=os.getuid(), start_time_ticks=90)))
    monkeypatch.setattr(features, "_same_live_process", lambda birth: True)
    assert resume.test_progress(study)["claimed_chunks"] == 1
    receipt = dict(schema=features.SCHEMA, complete=True, split="test", chunk=chunk,
                   plan_sha256=prepared["plan_sha256"], protocol_sha256=prepared["protocol_sha256"])
    resume.write(directory / "receipt.json", receipt)
    progress = resume.test_progress(study)
    assert progress["complete"] and progress["complete_chunks"] == 1
    assert progress["pending_chunks"] == 0 and progress["completed_rows"] == 3


@pytest.mark.parametrize("filename", ["selection_lock.json", "parallel_features/test/batch_plan.json"])
def test_progress_refuses_changed_lock_or_plan_before_reading_receipts(study, filename):
    resume.register(study)
    resume.write(study / filename, dict(changed=True))
    with pytest.raises(ValueError, match="changed"):
        resume.test_progress(study)


def test_phase_guard_refuses_training_before_calling_original_scheduler():
    scheduler = object.__new__(resume.LockedTestScheduler)
    with pytest.raises(ValueError, match="test extraction/evaluation only"):
        scheduler.run_workers("fit", None, None)


def test_admission_and_foreign_arrival_policy_is_inherited_unchanged():
    for method in ("_observe", "_launch", "_verify_command", "_record_arrival"):
        assert getattr(resume.LockedTestScheduler, method) is getattr(resume.cooperative.CooperativeScheduler, method)


def test_run_extracts_then_merges_then_evaluates_locked_heads_once(study, monkeypatch):
    resume.register(study)
    lock_before = (study / "selection_lock.json").read_bytes()
    events = []
    monkeypatch.setattr(features, "prepare_split", lambda *a, **kw: pytest.fail("Never prepare a new split"))
    monkeypatch.setattr(features, "merge_split", lambda s, split: events.append(("merge", split)))
    def mark(s, state, **kwargs):
        events.append(("state", state))
    monkeypatch.setattr(base, "mark_state", mark)
    class Scheduler:
        def __init__(self, s):
            assert Path(s) == study
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def run_workers(self, phase, factory, complete, **kwargs):
            assert phase == "extract_test" and kwargs["split"] == "test" and kwargs["max_workers"] == 8
            command = factory("extract_test_fixture", 0)
            assert command == original.worker_command(study, "extract_test_fixture", split="test")
            events.append(("extract", "test"))
        def run_single(self, phase, factory):
            assert phase == "evaluate"
            command = factory("evaluate_fixture", 0)
            assert command == resume.cooperative.worker_command(study, "evaluate_fixture", phase="evaluate")
            assert events.index(("extract", "test")) < events.index(("merge", "test"))
            events.append(("evaluate", "test"))
            resume.write(study / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(resume, "LockedTestScheduler", Scheduler)
    resume.run(study)
    assert (study / "selection_lock.json").read_bytes() == lock_before
    assert events.count(("evaluate", "test")) == 1
    assert resume.read(study / resume.FOLDER / "completion.json")["complete"] is True
    with pytest.raises(RuntimeError, match="automatic retry disabled"):
        resume.run(study)


def test_new_scheduler_publishes_new_birth_and_refuses_unreviewed_reentry(study, monkeypatch):
    control = resume.register(study)
    scheduler = object.__new__(resume.LockedTestScheduler)
    scheduler.study, scheduler.output = study, study / "parallel_execution"
    scheduler.resume_control = control
    scheduler.clock = time.time
    scheduler._check_control = lambda: None
    scheduler._verify_previous_workers_exited = lambda: None
    states = []
    scheduler._state = lambda state: states.append(state)
    monkeypatch.setattr(resume.queue.admission, "process_birth", lambda pid:
        dict(pid=pid, uid=os.getuid(), start_time_ticks=888, state="R") if pid == os.getpid() else None)
    assert scheduler.__enter__() is scheduler
    scheduler.lock_stream.close()
    scheduler.lock_stream = None
    entry = resume.read(study / resume.FOLDER / "scheduler_entry.json")
    assert entry["coordinator_birth"]["pid"] == os.getpid()
    assert entry["coordinator_birth"]["start_time_ticks"] == 888
    assert states == ["CPU_locked_test_monitor_ready"]
    with pytest.raises(FileExistsError):
        scheduler.__enter__()

