"""CPU-only registration and mocked-GPU ownership tests for the fresh study."""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.hanguard import frozen_base_readout_queue as queue


def gpu_row(pids=(), index=2):
    return dict(index=index, uuid=f"GPU-{index}", free_gib=40, pids=list(pids))


@pytest.fixture
def fresh_study(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    script = repository / "scripts/hanguard/frozen_base_readout_study.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import argparse, json, os, time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('action'); p.add_argument('--output'); a=p.parse_args()\n"
        "root=Path(a.output); time.sleep(.15)\n"
        "(root/'state.json').write_text(json.dumps(dict(formal_validation_opened=True, formal_test_opened=True)))\n"
        "(root/'analysis').mkdir(); (root/'analysis/results.json').write_text(json.dumps(dict(complete=True, cuda=os.environ['CUDA_VISIBLE_DEVICES'])))\n")
    runtime = repository / ".venv-hanguard/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.symlink_to(sys.executable)
    model = repository / "model.safetensors"
    data = repository / "data.parquet"
    model.write_bytes(b"mock model")
    data.write_bytes(b"mock data")
    study = repository / "outputs/fresh"
    study.mkdir(parents=True)
    protocol = dict(inputs={str(path): queue.guardian.legacy.sha(path) for path in (model, data)},
        code_sha256={str(script): queue.guardian.legacy.sha(script)},
        evaluation_scope="Fresh frozen-base train/validation and locked test")
    queue.guardian.legacy.dump(study / "protocol.json", protocol)
    queue.guardian.legacy.dump(study / "protocol_registration.json",
        dict(protocol_sha256=queue.guardian.legacy.sha(study / "protocol.json")))
    monkeypatch.setattr(queue, "ROOT", repository)
    return study, repository, model, data, script


def mocked_supervisor(monkeypatch, inventory, *, identity_reader=None):
    original = queue.ScopedSupervisor
    def construct(*args, **kwargs):
        kwargs.update(inventory=inventory, poll_seconds=.01, terminate_seconds=.5)
        if identity_reader is not None:
            kwargs["identity_reader"] = identity_reader
        return original(*args, **kwargs)
    monkeypatch.setattr(queue, "ScopedSupervisor", construct)


def test_registration_is_CPU_only_and_large_inputs_are_not_polling_bindings(fresh_study, monkeypatch):
    study, repository, model, data, _ = fresh_study
    real_sha = queue.guardian.legacy.sha
    def only_small_sha(path, *args, **kwargs):
        assert Path(path) not in (model, data), "Registration hashed a large scientific input"
        return real_sha(path, *args, **kwargs)
    monkeypatch.setattr(queue.guardian.legacy, "sha", only_small_sha)
    monkeypatch.setattr(queue.guardian.inventory_module, "gpu_inventory",
                        lambda: pytest.fail("Registration inspected GPUs"))
    plan = queue.initialize(study)
    assert queue.check_plan(study) == plan
    assert plan["resource_policy"]["physical_gpu"] == 2
    assert plan["resource_policy"]["reserved_gpu"] == 4
    assert plan["resource_policy"]["max_cuda_workers"] == 1
    assert str(model) in plan["input_identities"]
    assert str(model) not in plan["bindings"]
    resource = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "resource_protocol.json")
    assert str(data) not in resource["bindings"]
    assert resource["formal_test_enabled"] is True
    assert resource["formal_validation_enabled"] is True
    assert plan["stages"][0]["command"][0] == str(repository / ".venv-hanguard/bin/python")
    assert plan["stages"][0]["validator"] == "complete_json"
    assert not (study / queue.ATTEMPT / "resource_state.json").exists()


def test_pipeline_complete_with_mocked_empty_gpu_and_truthful_opened_flags(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    queue.initialize(study)
    mocked_supervisor(monkeypatch, lambda: [gpu_row(), gpu_row([999999], index=4)])
    result = queue.run(study)
    assert result["state"] == "completed"
    assert result["formal_validation_opened"] is True
    assert result["formal_test_opened"] is True
    published = queue.guardian.legacy.read_json(study / "analysis/results.json")
    assert published["cuda"] == "2"
    resource_state = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "resource_state.json")
    assert resource_state["formal_test_opened"] is True
    assert resource_state["formal_validation_opened"] is True
    execution = study / queue.ATTEMPT / "execution"
    registered = queue.guardian.legacy.read_json(execution / "queue_protocol.json")
    assert registered["formal_test_enabled"] is True
    assert "selection-locked test" in registered["evaluation_scope"]
    assert (execution / "queue_receipts/frozen_base_pipeline.json").exists()
    with pytest.raises(RuntimeError, match="already executed; no automatic retry"):
        queue.run(study)


def test_busy_foreign_card_waits_CPU_only_and_stops_at_new_study_STOP(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    queue.initialize(study)
    calls = 0
    def inventory():
        nonlocal calls
        calls += 1
        if calls == 3:
            (study / "STOP").write_text("User stopped the new study")
        return [gpu_row([7654321])]
    mocked_supervisor(monkeypatch, inventory,
        identity_reader=lambda pid: dict(pid=pid, uid=os.getuid() + 1, start_time_ticks=10,
            command_sha256="foreign", argv=["unrelated"], cwd="/tmp"))
    with pytest.raises(queue.guardian.legacy.StopRequested, match="User stopped"):
        queue.run(study)
    assert calls == 3
    assert not (study / "analysis/results.json").exists()
    state = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "resource_state.json")
    assert state["state"] == "stopped"
    assert state["foreign_processes_signalled"] is False
    assert state["formal_test_opened"] is False


def test_old_study_STOP_is_untouched_and_does_not_stop_new_pipeline(fresh_study, monkeypatch):
    study, repository, _, _, _ = fresh_study
    old = repository / "outputs/old_stopped"
    old.mkdir()
    (old / "STOP").write_text("Prior experiment stays stopped")
    queue.initialize(study)
    mocked_supervisor(monkeypatch, lambda: [gpu_row()])
    assert queue.run(study)["state"] == "completed"
    assert (old / "STOP").read_text() == "Prior experiment stays stopped"


def test_foreign_join_stops_only_identity_bound_pipeline(fresh_study, monkeypatch):
    study, _, _, _, script = fresh_study
    script.write_text("import time\ntime.sleep(10)\n")
    protocol = queue.guardian.legacy.read_json(study / "protocol.json")
    protocol["code_sha256"][str(script)] = queue.guardian.legacy.sha(script)
    queue.guardian.legacy.dump(study / "protocol.json", protocol)
    queue.guardian.legacy.dump(study / "protocol_registration.json",
        dict(protocol_sha256=queue.guardian.legacy.sha(study / "protocol.json")))
    queue.initialize(study)
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    original = queue.ScopedSupervisor
    instances = []
    def inventory():
        own = instances[0].child if instances else None
        return [gpu_row([own.pid, foreign.pid] if own is not None else [])]
    def construct(*args, **kwargs):
        instance = original(*args, inventory=inventory, poll_seconds=.01, terminate_seconds=.5, **kwargs)
        instances.append(instance)
        return instance
    monkeypatch.setattr(queue, "ScopedSupervisor", construct)
    try:
        with pytest.raises(queue.guardian.legacy.StopRequested, match="foreign/unknown"):
            queue.run(study)
        assert instances[0].child.poll() is not None
        assert foreign.poll() is None
        trigger = queue.guardian.legacy.read_json(
            study / queue.ATTEMPT / "execution/ownership_stop_trigger.json")
        assert trigger["foreign_processes_signalled"] is False
        assert [row["pid"] for row in trigger["gpu_observation"]["blocking_processes"]] == [foreign.pid]
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


@pytest.mark.parametrize("changed", ("uid", "start_time_ticks", "argv", "cwd", "command_sha256"))
def test_scope_subclass_preserves_exact_owner_checks(fresh_study, changed):
    study, _, _, _, _ = fresh_study
    queue.initialize(study)
    plan = queue.check_plan(study)
    stage = queue.guardian.RegisteredStage.from_registration(plan["stages"][0])
    expected = dict(pid=2345, uid=os.getuid(), start_time_ticks=10,
        argv=list(stage.command), cwd=str(queue.ROOT), command_sha256="birth")
    current = dict(expected)
    current[changed] = {"uid": os.getuid() + 1, "start_time_ticks": 11, "argv": ["other"],
                        "cwd": "/tmp", "command_sha256": "different"}[changed]
    supervisor = queue.ScopedSupervisor(study, [stage], {},
        queue_state_dir=study / "mock-execution", cwd=queue.ROOT,
        inventory=lambda: [gpu_row([2345])], identity_reader=lambda pid: current)
    supervisor.active_stage = stage.name
    supervisor.child = type("Child", (), {"pid": 2345})()
    supervisor.owned_processes[2345] = expected
    observed = supervisor.observation()
    assert observed["owned_pids"] == []
    assert observed["blocking_processes"][0]["classification"] == "foreign"
    assert observed["eligible"] is False


@pytest.mark.parametrize("gpu,reserved", ((2, 2), (4, 4), (-1, 4), (8, 4), (2, 8)))
def test_invalid_card_pairs_are_rejected_before_publication(fresh_study, gpu, reserved):
    study, _, _, _, _ = fresh_study
    with pytest.raises(ValueError, match="GPU0–GPU7"):
        queue.initialize(study, gpu=gpu, reserved_gpu=reserved)
    assert not (study / "resource_plan.json").exists()
    assert not (study / queue.ATTEMPT).exists()


def test_root_protocol_SHA_and_model_metadata_are_checked(fresh_study):
    study, _, model, _, _ = fresh_study
    registration = study / "protocol_registration.json"
    original = registration.read_text()
    queue.guardian.legacy.dump(registration, dict(protocol_sha256="0" * 64))
    with pytest.raises(ValueError, match="Root protocol differs"):
        queue.initialize(study)
    registration.write_text(original)
    queue.initialize(study)
    model.write_bytes(b"changed model contents and size")
    with pytest.raises(ValueError, match="metadata changed"):
        queue.run(study)
    assert not (study / queue.ATTEMPT / "resource_state.json").exists()


def test_resource_plan_source_and_registration_tampering_fail_closed(fresh_study):
    study, _, _, _, _ = fresh_study
    queue.initialize(study)
    plan_path = study / "resource_plan.json"
    plan = queue.guardian.legacy.read_json(plan_path)
    plan["stages"][0]["command"][-1] = "/tmp/another-study"
    queue.guardian.legacy.dump(plan_path, plan)
    with pytest.raises(ValueError, match="binding changed"):
        queue.check_plan(study)
    with pytest.raises(FileExistsError, match="immutable"):
        queue.initialize(study)


def test_scope_status_reads_both_true_flags_without_overriding_them(fresh_study):
    study, _, _, _, _ = fresh_study
    plan = queue.initialize(study)
    stage = queue.guardian.RegisteredStage.from_registration(plan["stages"][0])
    destination = study / "mock-execution"
    destination.mkdir()
    supervisor = queue.ScopedSupervisor(study, [stage], {}, queue_state_dir=destination)
    supervisor.protocol_sha = "mock-sha"
    queue.guardian.legacy.dump(study / "state.json",
        dict(formal_validation_opened=True, formal_test_opened=True))
    value = supervisor._state("running")
    assert value["formal_validation_opened"] is True
    assert value["formal_test_opened"] is True
    assert queue.guardian.legacy.read_json(destination / "queue_state.json") == value


def test_failed_pipeline_summary_preserves_opened_scientific_flags(fresh_study, monkeypatch):
    study, _, _, _, script = fresh_study
    script.write_text(
        "import argparse, json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('action'); p.add_argument('--output'); a=p.parse_args()\n"
        "Path(a.output,'state.json').write_text(json.dumps(dict(formal_validation_opened=True, formal_test_opened=True)))\n"
        "raise SystemExit(7)\n")
    protocol = queue.guardian.legacy.read_json(study / "protocol.json")
    protocol["code_sha256"][str(script)] = queue.guardian.legacy.sha(script)
    queue.guardian.legacy.dump(study / "protocol.json", protocol)
    queue.guardian.legacy.dump(study / "protocol_registration.json",
        dict(protocol_sha256=queue.guardian.legacy.sha(study / "protocol.json")))
    queue.initialize(study)
    mocked_supervisor(monkeypatch, lambda: [gpu_row()])
    with pytest.raises(RuntimeError, match="code 7"):
        queue.run(study)
    summary = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "execution/diagnostic_summary.json")
    assert summary["formal_validation_opened"] is True
    assert summary["formal_test_opened"] is True
    state = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "resource_state.json")
    assert state["formal_validation_opened"] is True
    assert state["formal_test_opened"] is True
    assert state["state"] == "failed"
    assert not (study / queue.ATTEMPT / "execution/queue_receipts/frozen_base_pipeline.json").exists()


def foreign_identity(pid):
    return dict(pid=pid, uid=os.getuid() + 1, start_time_ticks=10, command_sha256="foreign")


def test_CPU_selector_chooses_empty_GPU4_and_reserves_GPU2(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    observations = lambda: [gpu_row([7654321], index=2), gpu_row(index=4)]
    mocked_supervisor(monkeypatch, observations)
    initialized = []
    initialize = queue.initialize
    def record_initialize(*args, **kwargs):
        initialized.append(kwargs)
        return initialize(*args, **kwargs)
    monkeypatch.setattr(queue, "initialize", record_initialize)
    assert queue.wait_and_run(study, inventory=observations, identity_reader=foreign_identity,
                              poll_seconds=.01)["state"] == "completed"
    assert initialized == [dict(gpu=4, reserved_gpu=2)]
    result = queue.guardian.legacy.read_json(study / "analysis/results.json")
    assert result["cuda"] == "4"
    plan = queue.check_plan(study)
    assert plan["resource_policy"]["physical_gpu"] == 4
    assert plan["resource_policy"]["reserved_gpu"] == 2
    waiting = queue.guardian.legacy.read_json(study / "waiting_state.json")
    assert waiting["state"] == "completed"
    assert waiting["selector_cuda_loaded"] is False
    assert waiting["selector_started_GPU_child"] is False
    assert next(row for row in waiting["candidate_observations"] if row["index"] == 2)["process_owners"][0]["pid"] == 7654321
    assert waiting["registered_cuda_workers"] == 1
    assert waiting["max_cuda_workers"] == 1
    assert waiting["formal_test_opened"] is True


def test_both_busy_CPU_wait_then_registers_only_one_worker(fresh_study, monkeypatch):
    study, _, model, data, _ = fresh_study
    inventories = []
    sleep_calls = []
    real_sha = queue.guardian.legacy.sha
    def small_sha(path, *args, **kwargs):
        assert Path(path) not in (model, data)
        return real_sha(path, *args, **kwargs)
    monkeypatch.setattr(queue.guardian.legacy, "sha", small_sha)
    def inventory():
        inventories.append(True)
        if len(inventories) == 1:
            return [gpu_row([12345]), gpu_row([98765], index=4)]
        return [gpu_row(), gpu_row(index=4)]
    def sleep(study_path, seconds):
        sleep_calls.append(seconds)
        assert not (study / queue.ATTEMPT).exists()
        state = queue.guardian.legacy.read_json(study / "waiting_state.json")
        assert state["state"] == "CPU_wait_for_empty_candidate"
        assert state["registered_cuda_workers"] == 0
        assert state["selector_cuda_loaded"] is False
        assert [row["pids"] for row in state["candidate_observations"] if row["available"]] == [[12345], [98765]]
    monkeypatch.setattr(queue, "_sleep_wait", sleep)
    mocked_supervisor(monkeypatch, lambda: [gpu_row(), gpu_row(index=4)])
    assert queue.wait_and_run(study, inventory=inventory, identity_reader=foreign_identity)["state"] == "completed"
    assert len(inventories) == 2
    assert sleep_calls == [5.0]
    assert queue.check_plan(study)["resource_policy"]["physical_gpu"] == 2


def test_waiting_STOP_is_precise_without_registration_or_GPU_child(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    def sleep(study_path, seconds):
        (study_path / "STOP").write_text("Stop CPU waiting too")
    monkeypatch.setattr(queue, "_sleep_wait", sleep)
    monkeypatch.setattr(queue, "initialize", lambda *args, **kwargs: pytest.fail("STOP registered a GPU stage"))
    with pytest.raises(queue.guardian.legacy.StopRequested, match="CPU waiting"):
        queue.wait_and_run(study, inventory=lambda: [gpu_row([999]), gpu_row([888], index=4)],
                           identity_reader=foreign_identity)
    state = queue.guardian.legacy.read_json(study / "waiting_state.json")
    assert state["state"] == "stopped"
    assert state["foreign_processes_signalled"] is False
    assert state["selector_cuda_loaded"] is False
    assert not (study / queue.ATTEMPT).exists()


def test_preexisting_new_STOP_does_not_query_inventory(fresh_study):
    study, _, _, _, _ = fresh_study
    (study / "STOP").write_text("Stop before inventory")
    with pytest.raises(queue.guardian.legacy.StopRequested, match="before inventory"):
        queue.wait_and_run(study, inventory=lambda: pytest.fail("STOP queried GPUs"))
    assert queue.guardian.legacy.read_json(study / "waiting_state.json")["state"] == "stopped"


def test_waiter_lock_rejects_second_selector(fresh_study):
    study, _, _, _, _ = fresh_study
    with (study / "resource_waiter.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another CPU selector"):
            queue.wait_and_run(study, inventory=lambda: pytest.fail("Second waiter queried GPUs"))
    assert not (study / queue.ATTEMPT).exists()


def test_selector_cannot_retry_registered_attempt(fresh_study):
    study, _, _, _, _ = fresh_study
    queue.initialize(study)
    with pytest.raises(RuntimeError, match="already exists; no automatic retry"):
        queue.wait_and_run(study, inventory=lambda: pytest.fail("Retry queried GPUs"))


def test_waiting_model_metadata_change_fails_before_admission(fresh_study):
    study, _, model, _, _ = fresh_study
    def inventory():
        model.write_bytes(b"mutated while waiting")
        return [gpu_row(), gpu_row(index=4)]
    with pytest.raises(ValueError, match="metadata changed while waiting"):
        queue.wait_and_run(study, inventory=inventory)
    assert queue.guardian.legacy.read_json(study / "waiting_state.json")["state"] == "failed"
    assert not (study / queue.ATTEMPT).exists()


def test_waiting_status_is_available_before_resource_registration(fresh_study, monkeypatch, capsys):
    study, _, _, _, _ = fresh_study
    def sleep(study_path, seconds):
        queue.main(["status", "--study", str(study)])
        (study / "STOP").write_text("Enough status testing")
    monkeypatch.setattr(queue, "_sleep_wait", sleep)
    with pytest.raises(queue.guardian.legacy.StopRequested):
        queue.wait_and_run(study, inventory=lambda: [gpu_row([999]), gpu_row([888], index=4)],
                           identity_reader=foreign_identity)
    assert "CPU_wait_for_empty_candidate" in capsys.readouterr().out


def test_new_public_GPU6_is_admitted_and_policy_records_actual_card(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    observations = lambda: [gpu_row(index=6), gpu_row(), gpu_row(index=4)]
    mocked_supervisor(monkeypatch, observations)
    assert queue.wait_and_run(study, inventory=observations)["state"] == "completed"
    plan = queue.check_plan(study)
    assert plan["resource_policy"]["physical_gpu"] == 6
    assert plan["resource_policy"]["reserved_gpu"] == 4
    assert queue.guardian.legacy.read_json(study / "analysis/results.json")["cuda"] == "6"
    execution = queue.guardian.legacy.read_json(study / queue.ATTEMPT / "execution/queue_protocol.json")
    assert execution["resource_policy"]["physical_gpu"] == 6
    assert execution["resource_policy"]["max_cuda_workers"] == 1
    waiting = queue.guardian.legacy.read_json(study / "waiting_state.json")
    assert waiting["candidate_gpus"] == [6, 2, 4, 5, 7, 3, 0, 1]


def test_foreign_process_on_GPU6_blocks_even_same_UID(fresh_study, monkeypatch):
    study, _, _, _, _ = fresh_study
    def own_UID_identity(pid):
        return dict(pid=pid, uid=os.getuid(), start_time_ticks=10, command_sha256="another-job")
    def sleep(study_path, seconds):
        waiting = queue.guardian.legacy.read_json(study / "waiting_state.json")
        card6 = next(row for row in waiting["candidate_observations"] if row["index"] == 6)
        assert card6["eligible"] is False
        assert card6["process_owners"][0]["classification"] == "foreign"
        assert not (study / queue.ATTEMPT).exists()
        (study / "STOP").write_text("Do not share foreign GPU6")
    monkeypatch.setattr(queue, "_sleep_wait", sleep)
    with pytest.raises(queue.guardian.legacy.StopRequested, match="foreign GPU6"):
        queue.wait_and_run(study, inventory=lambda: [gpu_row([333], index=6)],
                           identity_reader=own_UID_identity)
    assert not (study / queue.ATTEMPT).exists()
