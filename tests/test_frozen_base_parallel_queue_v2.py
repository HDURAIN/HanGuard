import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.hanguard import frozen_base_parallel_queue as old
from scripts.hanguard import frozen_base_parallel_queue_v2 as queue


def identity(pid=12345, ticks=99):
    return dict(pid=pid, uid=os.getuid(), start_time_ticks=ticks,
        command_sha256="a" * 64, argv=["cpu-fixture"], cwd=str(old.ROOT))


class Child:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


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
    scheduler.active = {"fixture": dict(child=child, identity=identity(), worker_id="fixture", gpu=0)}
    scheduler.completed = []
    scheduler.resource_losses = []
    scheduler.phase = "train"
    scheduler.review_required = False
    scheduler.terminate_grace_seconds = 0
    return scheduler


@pytest.mark.parametrize("exception", [ValueError("Process command is unavailable"),
    PermissionError("proc denied"), FileNotFoundError("gone"), OSError("race")])
def test_identity_error_known_original_child_exit_zero_is_reaped(tmp_path, exception):
    child = Child()
    def reader(pid):
        raise exception
    scheduler = bare_scheduler(tmp_path, child, reader,
        sleeper=lambda seconds: setattr(child, "code", 0))
    scheduler._poll_workers()
    assert scheduler.completed[0]["exit_code"] == 0
    assert scheduler.active == {}
    assert scheduler.review_required is False


def test_unavailable_cmd_still_live_fails_closed_without_signal(tmp_path, monkeypatch):
    def reader(pid):
        raise ValueError("Process command is unavailable")
    signals, sleeps = [], []
    monkeypatch.setattr(old.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    scheduler = bare_scheduler(tmp_path, Child(), reader, sleeper=sleeps.append)
    with pytest.raises(queue.ResourceReviewRequired, match="identity became unverifiable"):
        scheduler._poll_workers()
    assert signals == []
    assert sleeps == [0.05] * 4
    assert scheduler.review_required is True


def test_identity_error_known_nonzero_child_exit_is_scientific_failure(tmp_path):
    child = Child()
    def reader(pid):
        raise ValueError("Process command is unavailable")
    scheduler = bare_scheduler(tmp_path, child, reader,
        sleeper=lambda seconds: setattr(child, "code", 7))
    with pytest.raises(queue.ResourceReviewRequired, match="Scientific worker exited unsuccessfully"):
        scheduler._poll_workers()
    assert scheduler.completed[0]["exit_code"] == 7
    assert scheduler.review_required is True


def test_reused_pid_still_live_is_never_signalled(tmp_path, monkeypatch):
    signals = []
    monkeypatch.setattr(old.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    scheduler = bare_scheduler(tmp_path, Child(), lambda pid: identity(ticks=999))
    with pytest.raises(queue.ResourceReviewRequired, match="identity changed"):
        scheduler._poll_workers()
    assert signals == []


def test_exit_immediately_after_successful_identity_read_is_reaped(tmp_path):
    child = Child()
    def reader(pid):
        child.code = 0
        return identity()
    scheduler = bare_scheduler(tmp_path, child, reader)
    scheduler._poll_workers()
    assert scheduler.active == {}
    assert scheduler.completed[0]["exit_code"] == 0


def test_real_cpu_subprocess_exit_is_observed_only_through_its_popen(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.05)"])
    def unavailable(pid):
        raise ValueError("Process command is unavailable")
    scheduler = bare_scheduler(tmp_path, child, unavailable, sleeper=time.sleep)
    scheduler.active["fixture"]["identity"] = identity(child.pid)
    try:
        scheduler._poll_workers()
        assert scheduler.completed[0]["exit_code"] == 0
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=2)


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("# immutable scientific fixture\n")
    coordinator = tmp_path / "coordinator.py"
    coordinator.write_text("# immutable CPU control fixture\n")
    data = tmp_path / "data.json"
    data.write_text("{}\n")
    old._write(study / "protocol.json", dict(inputs={str(data): old._sha(data)},
        code_sha256={str(worker): old._sha(worker)}, scope="CPU fixture",
        settings=dict(seed=42, epochs=12)))
    old._write(study / "protocol_registration.json", dict(
        protocol_sha256=old._sha(study / "protocol.json")))
    monkeypatch.setattr(old.admission, "capture_blockers", lambda: [])
    monkeypatch.setattr(old.admission, "capture_display", lambda: dict(verified=False))
    monkeypatch.setattr(queue, "COORDINATOR_PATH", coordinator)
    resource = old.initialize(study)
    old._write(study / "parallel_execution/queue_state.json", dict(
        requires_review=True, active_workers=1, reason="Own worker identity became unverifiable"))
    snapshot = study / "recovery_001/queue_state.before.json"
    snapshot.parent.mkdir()
    snapshot.write_bytes((study / "parallel_execution/queue_state.json").read_bytes())
    bindings = dict(resource["bindings"])
    for path in (Path(queue.__file__).resolve(), coordinator, snapshot):
        bindings[str(path)] = old._sha(path)
    control = dict(schema=queue.SCHEMA, study=str(study),
        root_protocol_sha256=old._sha(study / "protocol.json"),
        resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json"),
        bindings=bindings, policy=dict(queue.POLICY), prior_review=dict(
            queue_state_snapshot=str(snapshot), queue_state_sha256=old._sha(snapshot),
            prior_workers_verified_exited=True, resume_authorized=True,
            reason="normal_exit_identity_read_race"))
    old._write(study / "recovery_control_protocol.json", control)
    old._write(study / "recovery_control_registration.json", dict(
        protocol_sha256=old._sha(study / "recovery_control_protocol.json")))
    return study, control, worker, coordinator


def rewrite_control(study, control):
    old._write(study / "recovery_control_protocol.json", control)
    old._write(study / "recovery_control_registration.json", dict(
        protocol_sha256=old._sha(study / "recovery_control_protocol.json")))


def test_control_preserves_every_original_binding_and_eight_card_policy(recovery):
    study, control, _, _ = recovery
    assert queue.check_control(study) == control
    assert control["policy"]["max_cuda_workers"] == 8
    assert control["policy"]["reserved_gpus"] == []


@pytest.mark.parametrize("field,value", [("max_cuda_workers", 1), ("reserved_gpus", [7]),
    ("never_terminate_foreign_processes", False), ("preserve_registered_science", False)])
def test_recovery_cannot_change_registered_policy(recovery, field, value):
    study, control, _, _ = recovery
    control["policy"][field] = value
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="policy changed"):
        queue.check_control(study)


def test_changed_original_worker_source_fails_registration(recovery):
    study, _, worker, _ = recovery
    worker.write_text("# changed science\n")
    with pytest.raises(ValueError, match="source changed"):
        queue.check_control(study)


def test_changed_coordinator_source_is_rejected(recovery):
    study, _, _, coordinator = recovery
    coordinator.write_text("# changed CPU control\n")
    with pytest.raises(ValueError, match="coordinator is not registered"):
        queue.check_control(study)


def test_review_snapshot_cannot_change(recovery):
    study, _, _, _ = recovery
    (study / "recovery_001/queue_state.before.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Recovery source changed"):
        queue.check_control(study)


def test_review_requires_explicit_prior_worker_exit_proof(recovery):
    study, control, _, _ = recovery
    control["prior_review"]["prior_workers_verified_exited"] = False
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="prior stop review"):
        queue.check_control(study)


def test_enter_preserves_stop_and_allows_subsequent_phase_same_coordinator(recovery):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/termination_evidence.json", dict(workers=[], preserved=True))
    with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
        pass
    assert old._read(study / "recovery_001/queue_state.before.json")["requires_review"] is True
    assert old._read(study / "recovery_001/termination_evidence.before.json")["preserved"] is True
    with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
        pass


def test_unreviewed_changed_current_queue_is_rejected(recovery):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/queue_state.json", dict(requires_review=True, new_stop=True))
    with pytest.raises(queue.ResourceReviewRequired, match="differs from reviewed snapshot"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass


@pytest.mark.parametrize("birth", [dict(pid=12345, uid=os.getuid(), start_time_ticks=99, state="R"),
    dict(pid=12345, uid=os.getuid(), start_time_ticks=99, state="S")])
def test_still_live_prior_worker_blocks_recovery(recovery, birth):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/workers/old_worker.json", dict(
        schema_version=1, process_identity=identity()))
    with pytest.raises(queue.ResourceReviewRequired, match="prior recorded worker is still live"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: birth):
            pass


def test_unknown_prior_worker_birth_blocks_recovery(recovery):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/workers/old_worker.json", dict(
        schema_version=1, process_identity=identity()))
    def denied(pid):
        raise PermissionError("unknown prior birth")
    with pytest.raises(queue.ResourceReviewRequired, match="birth is unverifiable"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=denied):
            pass


def test_same_birth_changed_uid_does_not_count_as_original_exit(recovery):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/workers/old_worker.json", dict(
        schema_version=1, process_identity=identity()))
    with pytest.raises(queue.ResourceReviewRequired, match="owner changed within the same birth"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: dict(
                pid=pid, uid=os.getuid() + 1, start_time_ticks=99, state="R")):
            pass


def test_reused_prior_worker_pid_is_not_signalled(recovery, monkeypatch):
    study, _, _, _ = recovery
    old._write(study / "parallel_execution/workers/old_worker.json", dict(
        schema_version=1, process_identity=identity()))
    signals = []
    monkeypatch.setattr(old.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: dict(
            pid=pid, uid=os.getuid(), start_time_ticks=999, state="R")):
        pass
    assert signals == []


def test_control_checked_before_observe_launch_and_poll(recovery, monkeypatch):
    study, _, _, _ = recovery
    scheduler = queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None)
    def changed():
        raise ValueError("control changed")
    monkeypatch.setattr(scheduler, "_check_control", changed)
    for call in (lambda: scheduler._observe(), lambda: scheduler._poll_workers(),
            lambda: scheduler._launch(0, lambda worker, gpu: [], None)):
        with pytest.raises(ValueError, match="control changed"):
            call()
