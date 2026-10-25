"""CPU-only tests: every inventory, GPU launch and foreign birth is mocked."""
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

from scripts.hanguard import frozen_base_readout_admission as admission


def birth(pid=1531157, uid=1047, ticks=100, state="S"):
    return dict(pid=pid, uid=uid, start_time_ticks=ticks, state=state)


def gpu(index=2, pids=(), free_gib=40):
    return dict(index=index, uuid=f"GPU-{index}", pids=list(pids), free_gib=free_gib)


@pytest.fixture
def study(tmp_path):
    directory = tmp_path / "study"
    directory.mkdir()
    model, source = tmp_path / "model.bin", tmp_path / "scientific.py"
    model.write_bytes(b"frozen model")
    source.write_text("pass\n")
    legacy = admission.queue.guardian.legacy
    legacy.dump(directory / "protocol.json", dict(inputs={str(model): legacy.sha(model)},
        code_sha256={str(source): legacy.sha(source)}, scope="registered scientific scope"))
    legacy.dump(directory / "protocol_registration.json",
                dict(protocol_sha256=legacy.sha(directory / "protocol.json")))
    return directory


def mock_initialization(study, calls, **kwargs):
    calls.append(kwargs)
    (study / admission.queue.ATTEMPT).mkdir()
    admission.queue.guardian.legacy.dump(study / "resource_plan.json", kwargs)


def test_zero_cuda_pids_does_not_release_registered_foreign_pipeline():
    record = admission.capture_blockers(lambda pid: birth(pid))[0]
    candidates, births, error = admission.observations(lambda: [gpu(6)], [record], lambda pid: birth(pid))
    row = next(value for value in candidates if value["index"] == 6)
    assert error is None
    assert row["completely_idle"] is True
    assert row["pipeline_blocked"] is True
    assert row["eligible"] is False
    assert births[0]["birth_status"] == "registered_birth_alive"


@pytest.mark.parametrize("current,expected", ((None, "exited"), (birth(ticks=101), "pid_reused_original_exited"),
                                             (birth(state="Z"), "zombie_or_dead")))
def test_exit_pid_reuse_and_zombie_release_old_birth(current, expected):
    record = admission.capture_blockers(lambda pid: birth(pid))[0]
    result = admission.blocker_status(record, lambda pid: current)
    assert result["blocked"] is False
    assert result["birth_status"] == expected


def test_same_birth_uid_change_is_unknown_and_blocks():
    record = admission.capture_blockers(lambda pid: birth(pid))[0]
    result = admission.blocker_status(record, lambda pid: birth(pid, uid=9999))
    assert result["blocked"] is True
    assert result["birth_status"] == "same_birth_owner_changed_unknown"


def test_permission_failure_is_conservative_both_at_registration_and_monitoring():
    def unreadable(pid):
        raise PermissionError("foreign proc")
    record = admission.capture_blockers(unreadable)[0]
    assert record["registration_status"] == "unverified"
    assert admission.blocker_status(record, unreadable)["blocked"] is True
    assert admission.blocker_status(record, lambda pid: birth(pid))["blocked"] is True
    assert admission.blocker_status(record, lambda pid: None)["blocked"] is False


def test_absent_at_registration_does_not_claim_future_reused_pid():
    record = admission.capture_blockers(lambda pid: None)[0]
    assert admission.blocker_status(record, lambda pid: birth(pid))["blocked"] is False


def test_any_existing_pid_blocks_even_same_uid_low_memory_usage():
    candidates, _, _ = admission.observations(lambda: [gpu(2, [os.getpid()]), gpu(4)], [], lambda pid: None)
    assert next(row for row in candidates if row["index"] == 2)["eligible"] is False
    assert next(row for row in candidates if row["index"] == 4)["eligible"] is True


def test_unverified_nvidia_inventory_never_admits():
    def unreadable():
        raise subprocess.TimeoutExpired("nvidia-smi", 10)
    rows, _, error = admission.observations(unreadable, [], lambda pid: None)
    assert not any(row["eligible"] for row in rows)
    assert error["error_type"] == "TimeoutExpired"


def test_full_inventory_includes_non_compute_graphics_process(monkeypatch):
    monkeypatch.setattr(admission.queue.guardian.inventory_module, "gpu_inventory", lambda: [gpu()])
    xml = "<nvidia_smi_log><gpu><uuid>GPU-2</uuid><processes><process_info><pid>999</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(admission.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout=xml))
    assert admission.full_gpu_inventory()[0]["pids"] == [999]


def test_exact_one_launch_immediate_empty_candidate_no_wait_and_other_card_reserved(study):
    calls, run_calls = [], []
    def run(directory):
        run_calls.append(directory)
        return {"complete": True}
    result = admission.wait_and_run(study, inventory=lambda: [gpu(2), gpu(4), gpu(6)],
        snapshot_reader=lambda pid: None,
        initialize=lambda directory, **kwargs: mock_initialization(directory, calls, **kwargs),
        run=run, sleeper=lambda *args: pytest.fail("Empty admission must not add a sleep"))
    assert result == {"complete": True}
    assert calls == [{"gpu": 2, "reserved_gpu": 4}]
    assert run_calls == [study]
    state = admission.queue.guardian.legacy.read_json(study / "admission_state.json")
    assert state["max_cuda_workers"] == 1
    assert state["launch_attempted"] is True
    assert state["monitor_live"] is False
    assert state["foreign_processes_signalled"] is False


def test_registered_gpu6_parent_gate_waits_until_exit_then_launches(study):
    alive, sleeps, calls = True, [], []
    def snapshots(pid):
        return birth(pid) if alive and pid == 1531157 else None
    def sleeper(directory, seconds):
        nonlocal alive
        sleeps.append(seconds)
        state = admission.queue.guardian.legacy.read_json(directory / "admission_state.json")
        assert state["monitor_live"] is True
        assert state["candidate_observations"][-1]["pipeline_blocked"] is True
        alive = False
    admission.wait_and_run(study, inventory=lambda: [gpu(6)], snapshot_reader=snapshots,
        initialize=lambda directory, **kwargs: mock_initialization(directory, calls, **kwargs),
        run=lambda directory: {"complete": True}, sleeper=sleeper)
    assert len(sleeps) == 1
    assert calls == [{"gpu": 6, "reserved_gpu": 4}]


def test_new_stop_during_cpu_wait_prevents_launch(study):
    def sleeper(directory, seconds):
        (directory / "STOP").write_text("User stopped new study")
    with pytest.raises(admission.queue.guardian.legacy.StopRequested, match="User stopped"):
        admission.wait_and_run(study, inventory=lambda: [gpu(2, [999])],
            snapshot_reader=lambda pid: None, sleeper=sleeper,
            initialize=lambda *args, **kwargs: pytest.fail("STOP must prevent initialization"))
    state = admission.queue.guardian.legacy.read_json(study / "admission_state.json")
    assert state["state"] == "stopped"
    assert state["monitor_live"] is False
    assert state["launch_attempted"] is False


def test_second_monitor_rejected_by_separate_flock(study):
    with (study / "admission_waiter.pid").open("a+") as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another CPU admission monitor"):
            admission.wait_and_run(study, inventory=lambda: pytest.fail("No inventory under second lock"))
    assert not (study / "admission_protocol.json").exists()


def test_guardian_resource_stop_keeps_cpu_monitor_alive_without_retry(study):
    calls, run_calls, monitored = [], [], []
    def run(directory):
        run_calls.append(directory)
        raise admission.queue.guardian.legacy.StopRequested("Foreign process arrived")
    def sleeper(directory, seconds):
        current = admission.queue.guardian.legacy.read_json(directory / "admission_state.json")
        monitored.append(current)
        assert current["state"] == "CPU_monitor_requires_review"
        assert current["monitor_live"] is True
        assert current["requires_review"] is True
        assert current["automatic_training_launch_disabled"] is True
        assert current["candidate_observations"][0]["eligible"] is True
        if len(monitored) == 3:
            (directory / "STOP").write_text("Stop CPU monitor")
    with pytest.raises(admission.queue.guardian.legacy.StopRequested, match="Stop CPU monitor"):
        admission.wait_and_run(study, inventory=lambda: [gpu(2)], snapshot_reader=lambda pid: None,
            initialize=lambda directory, **kwargs: mock_initialization(directory, calls, **kwargs),
            run=run, sleeper=sleeper)
    assert len(calls) == 1
    assert len(run_calls) == 1
    assert len(monitored) == 3


def test_existing_attempt_is_monitored_not_relaunched_on_admission_restart(study):
    admission.admission_protocol(study, lambda pid: None)
    (study / admission.queue.ATTEMPT).mkdir()
    def sleeper(directory, seconds):
        assert admission.queue.guardian.legacy.read_json(directory / "admission_state.json")["requires_review"] is True
        (directory / "STOP").write_text("Stop restart monitor")
    with pytest.raises(admission.queue.guardian.legacy.StopRequested):
        admission.wait_and_run(study, inventory=lambda: [gpu(2)], snapshot_reader=lambda pid: None,
            initialize=lambda *args, **kwargs: pytest.fail("No attempt retry"), sleeper=sleeper)


def test_immutable_protocol_sources_and_input_metadata(study):
    protocol = admission.admission_protocol(study, lambda pid: None)
    again = admission.admission_protocol(study, lambda pid: birth(pid))
    assert again == protocol
    assert str(Path(admission.__file__).resolve()) in protocol["bindings"]
    assert str(Path(admission.queue.__file__).resolve()) in protocol["bindings"]
    admission.check_bindings(study, protocol)
    filename = next(iter(protocol["input_identities"]))
    Path(filename).write_bytes(b"model changed size")
    with pytest.raises(ValueError, match="input metadata changed"):
        admission.check_bindings(study, protocol)


def test_process_birth_reads_stat_handles_parentheses_exit_and_read_error(tmp_path):
    directory = tmp_path / "234"
    directory.mkdir()
    fields = ["S"] + ["0"] * 18 + ["123456"] + ["0"] * 10
    (directory / "stat").write_text("234 (name with ) parentheses) " + " ".join(fields))
    result = admission.process_birth(234, tmp_path)
    assert result == dict(pid=234, uid=os.getuid(), state="S", start_time_ticks=123456)
    assert admission.process_birth(235, tmp_path) is None
    (directory / "stat").write_text("invalid")
    with pytest.raises(IndexError):
        admission.process_birth(234, tmp_path)
