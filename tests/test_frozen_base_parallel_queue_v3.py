import os
from pathlib import Path
import time

import pytest

from scripts.hanguard import frozen_base_parallel_queue_v3 as queue
from test_frozen_base_parallel_queue_v2 import Child, identity, recovery

old = queue.original


@pytest.fixture
def recovery2(recovery, tmp_path, monkeypatch):
    study, control1, worker, _ = recovery
    coordinator2 = tmp_path / "coordinator2.py"
    coordinator2.write_text("# immutable post-validation fixture\n")
    monkeypatch.setattr(queue, "COORDINATOR_PATH", coordinator2)
    snapshot = study / "recovery_002/queue_state.before.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes((study / "parallel_execution/queue_state.json").read_bytes())
    bindings = dict(control1["bindings"])
    for path in (Path(queue.__file__).resolve(), coordinator2, snapshot,
            study / "recovery_control_protocol.json", study / "recovery_control_registration.json"):
        bindings[str(path)] = old._sha(path)
    control2 = dict(schema=queue.SCHEMA, study=str(study),
        root_protocol_sha256=old._sha(study / "protocol.json"),
        resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json"),
        prior_control_sha256=old._sha(study / "recovery_control_protocol.json"),
        bindings=bindings, policy=dict(queue.POLICY), prior_review=dict(
            queue_state_snapshot=str(snapshot), queue_state_sha256=old._sha(snapshot),
            prior_workers_verified_exited=True, resume_authorized=True,
            all_validation_complete=True, head_training_started=False,
            formal_test_opened=False, reason="completed_validation_resource_stop",
            cause_uncertain=True, suspected_exit_inventory_race=True))
    rewrite_control2(study, control2)
    return study, control2, worker, coordinator2


def rewrite_control2(study, control):
    old._write(study / "recovery_002/control_protocol.json", control)
    old._write(study / "recovery_002/control_registration.json", dict(
        protocol_sha256=old._sha(study / "recovery_002/control_protocol.json")))


def bare_scheduler(tmp_path, child, reader, *, sleeper=None):
    scheduler = object.__new__(queue.ParallelScheduler)
    scheduler.study = tmp_path
    scheduler.output = tmp_path / "parallel_execution"
    scheduler.output.mkdir(exist_ok=True)
    (scheduler.output / "workers").mkdir(exist_ok=True)
    scheduler._check_control = lambda: None
    scheduler.clock = time.time
    scheduler.sleep = sleeper or (lambda seconds: None)
    scheduler.identity_reader = reader
    scheduler.birth_reader = lambda pid: None
    scheduler.active = {"fixture": dict(child=child, identity=identity(), worker_id="fixture", gpu=0)}
    scheduler.completed, scheduler.resource_losses = [], []
    scheduler.phase = "fit"
    scheduler.review_required = False
    scheduler.terminate_grace_seconds = 0
    scheduler.protocol = dict(gpu6_pipeline_births=[])
    return scheduler


def test_post_inventory_known_child_exit_retired_before_collision_check(tmp_path):
    child = Child()
    def unavailable(pid):
        raise ValueError("Process command is unavailable")
    scheduler = bare_scheduler(tmp_path, child, unavailable,
        sleeper=lambda seconds: setattr(child, "code", 0))
    scheduler.inventory = lambda: [dict(index=0, pids=[12345], free_gib=40, uuid="GPU0")]
    decisions = scheduler._observe()
    assert decisions[5]["blocking_processes"][0]["classification"] == "unknown"
    assert scheduler.active == {}
    assert scheduler.completed[0]["exit_code"] == 0
    assert scheduler.review_required is False


def test_live_unknown_child_is_not_hidden_by_post_inventory_reap(tmp_path, monkeypatch):
    def unavailable(pid):
        raise ValueError("Process command is unavailable")
    scheduler = bare_scheduler(tmp_path, Child(), unavailable)
    scheduler.inventory = lambda: [dict(index=0, pids=[12345], free_gib=40, uuid="GPU0")]
    signals = []
    monkeypatch.setattr(old.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(queue.ResourceReviewRequired, match="identity became unverifiable"):
        scheduler._observe()
    assert signals == []
    assert scheduler.active
    events = list((tmp_path / "recovery_002/failure_events").glob("*.json"))
    assert len(events) == 1
    assert old._read(events[0])["trigger_evidence"]["exact_original_child_still_live"] is True


def test_same_uid_foreign_process_still_blocks_card(tmp_path):
    scheduler = bare_scheduler(tmp_path, Child(), lambda pid: identity(pid=pid))
    scheduler.inventory = lambda: [dict(index=0, pids=[12345, 999], free_gib=40, uuid="GPU0")]
    decisions = scheduler._observe()
    card = next(value for value in decisions if value["index"] == 0)
    assert card["owned_pids"] == [12345]
    assert card["blocking_processes"][0]["pid"] == 999
    assert card["blocking_processes"][0]["classification"] == "foreign"
    assert scheduler.active


def test_completed_child_and_real_foreign_are_both_handled_without_signal(tmp_path, monkeypatch):
    child = Child(0)
    scheduler = bare_scheduler(tmp_path, child, lambda pid: identity(pid=pid))
    scheduler.inventory = lambda: [dict(index=0, pids=[999], free_gib=40, uuid="GPU0")]
    signals = []
    monkeypatch.setattr(old.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    decisions = scheduler._observe()
    assert scheduler.active == {}
    assert next(value for value in decisions if value["index"] == 0)["eligible"] is False
    assert signals == []


def test_nonzero_post_inventory_exit_remains_failure_and_retains_trigger(tmp_path):
    scheduler = bare_scheduler(tmp_path, Child(9), lambda pid: identity(pid=pid))
    scheduler.inventory = lambda: [dict(index=0, pids=[], free_gib=40, uuid="GPU0")]
    with pytest.raises(queue.ResourceReviewRequired, match="Scientific worker exited unsuccessfully"):
        scheduler._observe()
    events = list((tmp_path / "recovery_002/failure_events").glob("*.json"))
    assert old._read(events[0])["trigger_evidence"]["exit_code"] == 9


def test_phase_complete_recomputed_after_observe_prevents_duplicate_fit(tmp_path, monkeypatch):
    scheduler = bare_scheduler(tmp_path, Child(), lambda pid: identity(pid=pid))
    scheduler.active.clear()
    monkeypatch.setattr(old, "check_protocol", lambda study: {})
    scheduler._poll_workers = lambda: None
    completed, launches = [], []
    def observe():
        scheduler.completed.append(dict(exit_code=0))
        completed.append(True)
        return [dict(index=0, eligible=True, pids=[], blocking_processes=[], pipeline_blocked=False)]
    scheduler._observe = observe
    scheduler._launch = lambda *args: launches.append(args)
    scheduler.run_single("fit", lambda worker, gpu: ["fixture"])
    assert launches == []
    assert len(scheduler.completed) == 1


def test_actual_live_foreign_collision_retains_complete_evidence_before_stop(tmp_path, monkeypatch):
    child = Child()
    scheduler = bare_scheduler(tmp_path, child, lambda pid: identity(pid=pid))
    scheduler.active.clear()
    monkeypatch.setattr(old, "check_protocol", lambda study: {})
    signals = []
    def kill(pid, sig):
        signals.append((pid, sig))
        child.code = -int(sig)
    monkeypatch.setattr(old.os, "kill", kill)
    scheduler._poll_workers = lambda: None
    def observe():
        scheduler.active["fixture"] = dict(child=child, identity=identity(), worker_id="fixture", gpu=0)
        return [dict(index=0, eligible=False, pids=[12345, 999], blocking_processes=[dict(
            pid=999, classification="foreign")], pipeline_blocked=False)]
    scheduler._observe = observe
    with pytest.raises(queue.ResourceReviewRequired, match="Foreign process entered an active GPU"):
        scheduler.run_workers("fit", lambda worker, gpu: [], lambda: False)
    event = old._read(next((tmp_path / "recovery_002/failure_events").glob("*.json")))
    assert event["trigger_evidence"]["blocking_processes"][0]["pid"] == 999
    assert event["active_workers"][0]["process_identity"]["pid"] == 12345
    assert {pid for pid, signal in signals} == {12345}


def test_control2_preserves_prior_protocol_and_eight_card_policy(recovery2):
    study, control2, _, _ = recovery2
    assert queue.check_control2(study) == control2
    assert control2["policy"]["max_cuda_workers"] == 8
    assert control2["policy"]["reserved_gpus"] == []


@pytest.mark.parametrize("field,value", [("head_training_started", True),
    ("formal_test_opened", True), ("all_validation_complete", False),
    ("prior_workers_verified_exited", False), ("resume_authorized", False)])
def test_control2_requires_complete_validation_and_unopened_fit_test(recovery2, field, value):
    study, control2, _, _ = recovery2
    control2["prior_review"][field] = value
    rewrite_control2(study, control2)
    with pytest.raises(ValueError, match="immutable reviewed stop"):
        queue.check_control2(study)


def test_control2_cannot_unbind_old_sources(recovery2):
    study, control2, worker, _ = recovery2
    del control2["bindings"][str(worker)]
    rewrite_control2(study, control2)
    with pytest.raises(ValueError, match="preserve every prior source binding"):
        queue.check_control2(study)


def test_control2_coordinator_source_change_rejected(recovery2):
    study, _, _, coordinator = recovery2
    coordinator.write_text("# changed postvalidation control\n")
    with pytest.raises(ValueError, match="not registered"):
        queue.check_control2(study)


def test_control2_source_review_snapshot_change_rejected(recovery2):
    study, _, _, _ = recovery2
    (study / "recovery_002/queue_state.before.json").write_text("{}\n")
    with pytest.raises(ValueError, match="source changed"):
        queue.check_control2(study)


def test_control2_enter_preserves_old_stop_and_allows_same_birth_next_phase(recovery2):
    study, _, _, _ = recovery2
    old._write(study / "parallel_execution/termination_evidence.json", dict(workers=[], original=True))
    for _ in range(2):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass
    assert old._read(study / "recovery_002/queue_state.before.json")["requires_review"] is True
    assert old._read(study / "recovery_002/termination_evidence.before.json")["original"] is True
    assert not (study / "recovery_001/scheduler_recovery_entry.json").exists()


def test_control2_late_queue_stop_requires_new_review(recovery2):
    study, _, _, _ = recovery2
    with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
        pass
    old._write(study / "parallel_execution/queue_state.json", dict(requires_review=True))
    with pytest.raises(queue.ResourceReviewRequired, match="interrupted again"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass
