"""CPU-only recovery checks; never query a real GPU or mutate an active study."""
import copy
import errno
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import pytest

from scripts.hanguard import frozen_base_test_monitor as monitor
from scripts.hanguard import frozen_base_parallel_features as features
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard import frozen_base_parallel_study as original


@pytest.fixture
def study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    write = monitor.write
    write(study / "protocol.json", dict(settings=dict(seed=42)))
    write(study / "selection_lock.json", dict(complete=True, immutable="selected"))
    prior = study / monitor.prior.FOLDER
    prior.mkdir()
    old = dict(unchanged_scientific_settings=dict(seed=42))
    write(prior / "control_protocol.json", old)
    write(prior / "control_registration.json", dict(protocol_sha256=monitor.sha(prior / "control_protocol.json")))
    birth = dict(pid=300, uid=os.getuid(), start_time_ticks=99, state="R")
    write(prior / "coordinator_started.json", dict(birth=birth))
    write(prior / "scheduler_entry.json", dict(coordinator_birth=birth))
    (prior / "coordinator.log").write_text("subprocess.TimeoutExpired: nvidia-smi\n")
    reason = "Command '['nvidia-smi', '-q', '-x']' timed out after 10 seconds"
    write(study / "state.json", dict(state="requires_review", error_type="TimeoutExpired",
                                     reason=reason, formal_test_opened=True, training_started=True))
    (study / "parallel_execution/workers").mkdir(parents=True)
    write(study / "parallel_execution/queue_state.json", dict(
        phase="extract_test", requires_review=True, active_workers=0, workers=[], reason=reason,
        post_selection_recovery_control_sha256=monitor.sha(prior / "control_protocol.json")))
    prepared = dict(complete=True, test_selection_lock_sha256=monitor.sha(study / "selection_lock.json"))
    write(study / "parallel_features/test/prepared_manifest.json", prepared)
    write(study / "parallel_features/test/batch_plan.json", dict(chunks=[]))
    monkeypatch.setattr(monitor.prior, "check_control", lambda s: copy.deepcopy(old))
    monkeypatch.setattr(features, "load_prepared", lambda s, split: (copy.deepcopy(prepared), None, None))
    monkeypatch.setattr(monitor.queue.admission, "process_birth", lambda pid: None)
    return study


def scheduler(study, monkeypatch, inventory):
    s = object.__new__(monitor.TestMonitorScheduler)
    s.study, s.output = study, study / "parallel_execution"
    s.clock = time.time
    s.telemetry = dict(available=None, consecutive_failures=0, total_failures=0)
    s.raw_inventory, s.inventory = inventory, s._inventory
    s.active, s.completed, s.arrivals = {}, [], set()
    s.admission_in_progress = False
    s.protocol = dict(gpu6_pipeline_births=[])
    s.identity_reader = lambda pid: pytest.fail("No process identity should be queried")
    s.birth_reader = lambda pid: None
    s._check_control = lambda: None
    s._poll_workers = lambda: None
    s._state = lambda state, **details: None
    s._signal_exact = lambda *a: pytest.fail("No running worker may be signalled")
    return s


def test_registration_preserves_root_selection_and_old_controls(study):
    paths = [study / "protocol.json", study / "selection_lock.json", monitor.prior.control_path(study)]
    before = {str(p): p.read_bytes() for p in paths}
    control = monitor.register(study)
    assert control["scope"] == monitor.prior.SCOPE
    assert control["policy"]["active_foreign"] == "record_only"
    assert control["telemetry_policy"]["failure"] == "all_cards_unavailable"
    assert {str(p): p.read_bytes() for p in paths} == before
    assert monitor.check_control(study) == control


@pytest.mark.parametrize("change", [dict(active_workers=1), dict(phase="fit"), dict(workers=[dict(pid=9)])])
def test_registration_rejects_unreviewed_phase_or_workers(study, change):
    path = study / "parallel_execution/queue_state.json"
    value = monitor.read(path)
    value.update(change)
    monitor.write(path, value)
    with pytest.raises(ValueError, match="Expected reviewed"):
        monitor.register(study)
    assert not (study / monitor.FOLDER).exists()


def test_live_old_coordinator_is_rejected(study, monkeypatch):
    monkeypatch.setattr(monitor.queue.admission, "process_birth", lambda pid:
        dict(pid=300, uid=os.getuid(), start_time_ticks=99, state="S"))
    with pytest.raises(ValueError, match="still alive"):
        monitor.register(study)


@pytest.mark.parametrize("artifact", ["analysis/results.json", "analysis/test_access.json",
                                     "parallel_execution/workers/evaluate_0001_gpu0.json"])
def test_registration_refuses_any_evaluation(study, artifact):
    monitor.write(study / artifact, dict(phase="evaluate"))
    with pytest.raises(ValueError, match="evaluation"):
        monitor.register(study)


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired(["nvidia-smi", "-q", "-x"], 10),
    subprocess.CalledProcessError(1, ["/usr/bin/nvidia-smi", "-q", "-x"]),
    FileNotFoundError(errno.ENOENT, "missing", "nvidia-smi"),
    ET.ParseError("bad XML"), ValueError("missing GPU UUID"), KeyError("uuid"), TypeError("bad snapshot")])
def test_inventory_failure_is_unknown_then_recovers_without_signalling(study, monkeypatch, error):
    normal = [dict(index=i, pids=[], free_gib=20) for i in range(8)]
    outcomes = iter([error, normal])
    def inventory():
        value = next(outcomes)
        if isinstance(value, BaseException):
            raise value
        return value
    s = scheduler(study, monkeypatch, inventory)
    # An admitted worker remains untouched even when telemetry is unreadable.
    s.active["owned"] = dict(identity=dict(pid=888), gpu=2)
    first = s._observe()
    assert not any(v["available"] or v["eligible"] for v in first)
    assert "owned" in s.active and s.telemetry["consecutive_failures"] == 1
    second = s._observe()
    assert all(v["available"] and v["eligible"] for v in second)
    assert s.telemetry["available"] and s.telemetry["consecutive_failures"] == 0
    assert s.telemetry["total_failures"] == 1


@pytest.mark.parametrize("error", [RuntimeError("scientific worker failure"),
    subprocess.TimeoutExpired(["other-program"], 10),
    PermissionError(errno.EACCES, "denied", "/study/selection_lock.json")])
def test_non_telemetry_exceptions_are_not_hidden(study, monkeypatch, error):
    def inventory():
        raise error
    s = scheduler(study, monkeypatch, inventory)
    with pytest.raises(type(error)):
        s._inventory()


def test_worker_loop_waits_through_timeout_then_launches_when_inventory_recovers(study, monkeypatch):
    calls = []
    normal = [dict(index=i, pids=[], free_gib=20) for i in range(8)]
    def inventory():
        calls.append("inventory")
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(["nvidia-smi", "-q", "-x"], 10)
        return normal
    s = scheduler(study, monkeypatch, inventory)
    launches, sleeps = [], []
    s.sleep = lambda duration: sleeps.append(duration)
    s._launch = lambda gpu, *a: launches.append(gpu) or True
    s.run_workers("extract_test", lambda *a: None, lambda: bool(launches),
                  work_remaining=lambda: not launches, split="test", max_workers=8)
    assert launches == [monitor.queue.CANDIDATES[0]]
    assert len(sleeps) == 2 and s.telemetry["total_failures"] == 1
    assert s.telemetry["available"] is True


def test_control_exception_is_outside_inventory_recovery(study, monkeypatch):
    s = scheduler(study, monkeypatch, lambda: [])
    def changed():
        raise ValueError("selection changed")
    s._check_control = changed
    with pytest.raises(ValueError, match="selection changed"):
        s._observe()


def test_unknown_inventory_defers_admission_without_signalling(study, monkeypatch):
    def bad():
        raise subprocess.TimeoutExpired(["nvidia-smi", "-q", "-x"], 10)
    s = scheduler(study, monkeypatch, bad)
    def launch(self, *args, **kwargs):
        self._observe()
        pytest.fail("Unknown inventory must not reach CUDA admission")
    monkeypatch.setattr(monitor.cooperative.CooperativeScheduler, "_launch", launch)
    assert s._launch(0, lambda *a: None, "test") is False
    assert s.active == {} and not s.admission_in_progress


def test_postspawn_unknown_only_retires_unpublished_cpu_child(study, monkeypatch):
    s = scheduler(study, monkeypatch, lambda: [])
    existing = dict(worker_id="owned", identity=dict(pid=44), gpu=0, ledger_path="registered")
    s.active["owned"] = existing
    def launch(self, *args, **kwargs):
        self.active["pending"] = dict(worker_id="pending", identity=dict(pid=55), gpu=1)
        raise monitor.AdmissionTelemetryUnavailable("timeout")
    def stop(records, reason):
        assert [r["worker_id"] for r in records] == ["pending"]
        s.active.pop("pending")
        return [dict(worker_id="pending", verified_original_child_exited=True)]
    monkeypatch.setattr(monitor.cooperative.CooperativeScheduler, "_launch", launch)
    s._stop_records = stop
    assert s._launch(1, None, "test") is False
    assert s.active == dict(owned=existing)
    assert len(list((study / monitor.FOLDER / "admission_deferrals").glob("*.json"))) == 1


def test_admission_timeout_reaps_expired_cpu_child_before_polling_cuda_workers(study, monkeypatch):
    def inventory():
        raise subprocess.TimeoutExpired(["nvidia-smi", "-q", "-x"], 10)
    s = scheduler(study, monkeypatch, inventory)
    class Child:
        def __init__(self, code):
            self.code = code
        def poll(self):
            return self.code
    existing = dict(worker_id="owned", identity=dict(pid=44), gpu=0,
                    ledger_path="registered", child=Child(None))
    expired = dict(worker_id="pending", identity=dict(pid=55), gpu=1, child=Child(1))
    s.active["owned"] = existing
    s.terminate_grace_seconds = 0
    s._poll_workers = lambda: pytest.fail("Do not classify expired unpublished child as scientific failure")
    s._signal_exact = lambda record, number: monitor.queue.ParallelScheduler._signal_exact(s, record, number)
    def launch(self, *args, **kwargs):
        self.active["pending"] = expired
        self._observe()
        pytest.fail("No ledger may be published after timeout")
    monkeypatch.setattr(monitor.cooperative.CooperativeScheduler, "_launch", launch)
    assert s._launch(1, None, "test") is False
    assert s.active == dict(owned=existing)
    assert existing["child"].poll() is None
    receipt = monitor.read(next((study / monitor.FOLDER / "admission_deferrals").glob("*.json")))
    assert receipt["workers"][0]["exit_code"] == 1
    assert receipt["workers"][0]["verified_SIGTERM_sent"] is False
    assert receipt["workers"][0]["verified_SIGKILL_sent"] is False


@pytest.mark.parametrize("rows", [[], [dict(index=i, pids=[], free_gib=20) for i in range(7)],
    [dict(index=i, pids=[], free_gib=float("nan")) for i in range(8)]])
def test_malformed_or_incomplete_snapshot_is_unavailable(study, monkeypatch, rows):
    s = scheduler(study, monkeypatch, lambda: rows)
    assert s._inventory() == []
    assert s.telemetry["available"] is False


def test_only_test_phases_are_permitted():
    s = object.__new__(monitor.TestMonitorScheduler)
    for phase in ("fit", "extract_train", "extract_validation", ""):
        with pytest.raises(ValueError, match="test extraction/evaluation only"):
            s.run_workers(phase, None, None)


def test_monitor_state_distinguishes_telemetry_from_empty_card_wait(study, monkeypatch):
    monitor.register(study)
    s = scheduler(study, monkeypatch, lambda: [])
    del s._state
    states = []
    monkeypatch.setattr(monitor.cooperative.CooperativeScheduler, "_state", lambda self, state, **kw:
                        states.append((state, kw)))
    s.telemetry["available"] = False
    s._state("CPU_wait_for_empty_candidates")
    assert states[-1][0] == "CPU_wait_for_GPU_telemetry"
    assert states[-1][1]["gpu_telemetry"]["available"] is False
    s.telemetry["available"] = True
    s._state("CPU_wait_for_empty_candidates")
    assert states[-1][0] == "CPU_wait_for_empty_candidates"


def test_control_poll_uses_no_full_cache_barrier(study, monkeypatch):
    monitor.register(study)
    monkeypatch.setattr(base, "validate_selection_lock", lambda *a: pytest.fail("No 13GB hash per poll"))
    monkeypatch.setattr(features, "load_prepared", lambda *a: pytest.fail("No token payload reload per poll"))
    assert monitor.check_control(study)["scope"] == monitor.SCOPE


def test_new_scheduler_entry_uses_006_birth_and_never_old_entry(study, monkeypatch):
    control = monitor.register(study)
    s = scheduler(study, monkeypatch, lambda: [])
    s.monitor_control = control
    s._verify_previous_workers_exited = lambda: None
    monkeypatch.setattr(monitor.cooperative.CooperativeScheduler, "__enter__",
                        lambda *a: pytest.fail("Do not enter recovery004"))
    assert s.__enter__() is s
    s.lock_stream.close()
    assert (study / monitor.FOLDER / "scheduler_entry.json").exists()
    with pytest.raises(FileExistsError):
        s.__enter__()


def test_run_reuses_existing_test_and_only_evaluates_once(study, monkeypatch):
    monitor.register(study)
    cooperative_folder = study / monitor.cooperative.FOLDER
    cooperative_folder.mkdir()
    monkeypatch.setattr(monitor.cooperative, "check_control", lambda s: dict(runtime_environment={}))
    monkeypatch.setattr(features, "prepare_split", lambda *a: pytest.fail("No new test preparation"))
    events = []
    monkeypatch.setattr(features, "merge_split", lambda s, split: events.append("merge"))
    monkeypatch.setattr(base, "mark_state", lambda *a, **kw: None)
    class Scheduler:
        def __init__(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def run_workers(self, phase, factory, complete, **kwargs):
            assert phase == "extract_test"
            assert factory("fixture", 0) == original.worker_command(study, "fixture", split="test")
            events.append("extract")
        def run_single(self, phase, factory):
            assert phase == "evaluate"
            assert factory("fixture", 0) == monitor.cooperative.worker_command(study, "fixture", phase="evaluate")
            events.append("evaluate")
            monitor.write(study / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(monitor, "TestMonitorScheduler", Scheduler)
    monitor.run(study)
    assert events == ["extract", "merge", "evaluate"]
    with pytest.raises(RuntimeError, match="automatic retry disabled"):
        monitor.run(study)
