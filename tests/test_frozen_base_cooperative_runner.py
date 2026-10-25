"""CPU behavior checks for cooperative admission and the locked-test boundary."""
import copy
import os
from pathlib import Path
import time

import pytest

from scripts.hanguard import frozen_base_cooperative_runner as runner


old = runner.queue


class Child:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def identity(pid=12345, ticks=99):
    return dict(pid=pid, uid=os.getuid(), start_time_ticks=ticks, start_ticks=ticks,
                command_sha256="a" * 64, argv=["cpu-fixture"], cwd=str(old.ROOT))


def bare_scheduler(tmp_path, monkeypatch):
    scheduler = object.__new__(runner.CooperativeScheduler)
    scheduler.study = tmp_path
    scheduler.output = tmp_path / "parallel_execution"
    (scheduler.output / "workers").mkdir(parents=True)
    old._write(runner.control_path(tmp_path), {})
    scheduler.runtime_control = {}
    scheduler._check_control = lambda: None
    scheduler.clock = time.time
    scheduler.sleep = lambda seconds: None
    scheduler.identity_reader = lambda pid: identity(pid)
    scheduler.birth_reader = lambda pid: None
    scheduler.active, scheduler.completed, scheduler.resource_losses = {}, [], []
    scheduler.arrivals = set()
    scheduler.phase = None
    scheduler.review_required = False
    scheduler.terminate_grace_seconds = 0
    scheduler.protocol = dict(gpu6_pipeline_births=[])
    scheduler.sequence = 0
    monkeypatch.setattr(old, "check_protocol", lambda study: scheduler.protocol)
    return scheduler


def test_active_foreign_arrival_records_and_completes_without_any_signal(tmp_path, monkeypatch):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    child, observations = Child(), []
    def inventory():
        observations.append(True)
        pids = [12345, 999] if scheduler.active else []
        return [dict(index=0, pids=pids, free_gib=40, uuid="GPU0")]
    scheduler.inventory = inventory
    launches = []
    def launch(gpu, factory, split, callback):
        launches.append(gpu)
        scheduler.active["fixture"] = dict(child=child, identity=identity(), worker_id="fixture", gpu=gpu)
        return True
    scheduler._launch = launch
    def sleep(seconds):
        # Observe an active foreign process for two polling cycles before exit.
        if len(observations) >= 3:
            child.code = 0
    scheduler.sleep = sleep
    monkeypatch.setattr(old.os, "kill", lambda *args: pytest.fail("No process may be signalled for a foreign arrival"))
    scheduler.stop_owned_workers = lambda reason: pytest.fail("An admitted worker must continue")
    scheduler.run_single("fit", lambda worker, gpu: ["fixture"])
    events = list((tmp_path / runner.FOLDER / "foreign_arrivals").glob("*.json"))
    assert launches == [0]
    assert len(events) == 1
    assert old._read(events[0])["action"] == "record_only_continue_owned_worker"
    assert old._read(events[0])["process"]["pid"] == 999
    assert scheduler.completed[0]["exit_code"] == 0
    assert not scheduler.review_required


@pytest.mark.parametrize("foreign_uid", [None, "other"])
def test_foreign_process_blocks_admission_before_popen(tmp_path, monkeypatch, foreign_uid):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    scheduler.phase = "fit"
    def reader(pid):
        value = identity(pid)
        if foreign_uid == "other":
            value["uid"] += 1
        return value
    scheduler.identity_reader = reader
    scheduler.inventory = lambda: [dict(index=0, pids=[999], free_gib=40, uuid="GPU0")]
    scheduler.popen = lambda *args, **kwargs: pytest.fail("An occupied card must never launch a child")
    assert scheduler._launch(0, lambda wid, gpu: runner.worker_command(tmp_path, wid, phase="fit"), None) is False
    assert scheduler.active == {}


def test_normal_exit_during_inventory_cannot_duplicate_launch(tmp_path, monkeypatch):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    child = Child()
    launches = []
    def launch(gpu, factory, split, callback):
        launches.append(gpu)
        scheduler.active["fixture"] = dict(child=child, identity=identity(), worker_id="fixture", gpu=gpu)
        return True
    scheduler._launch = launch
    def inventory():
        if scheduler.active:
            child.code = 0
        return [dict(index=0, pids=[], free_gib=40, uuid="GPU0")]
    scheduler.inventory = inventory
    scheduler.run_single("fit", lambda worker, gpu: ["fixture"])
    assert launches == [0]
    assert len(scheduler.completed) == 1
    assert scheduler.active == {}


def test_foreign_admission_race_retires_only_pre_cuda_child_and_requeues(tmp_path, monkeypatch):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    scheduler.phase = "fit"
    child = Child()
    child.pid = 12345
    command = runner.worker_command(tmp_path, "fit_0001_gpu0", phase="fit")
    def reader(pid):
        value = identity(pid)
        if pid == child.pid:
            value["argv"] = command
        return value
    scheduler.identity_reader = reader
    inventories = iter([[], [999]])
    scheduler.inventory = lambda: [dict(index=0, pids=next(inventories), free_gib=40, uuid="GPU0")]
    scheduler.popen = lambda *args, **kwargs: child
    signals = []
    def kill(pid, sig):
        signals.append((pid, sig))
        assert pid == child.pid
        child.code = -int(sig)
    monkeypatch.setattr(old.os, "kill", kill)
    assert scheduler._launch(0, lambda wid, gpu: runner.worker_command(tmp_path, wid, phase="fit"), None) is False
    assert [pid for pid, sig in signals] == [child.pid]
    assert scheduler.active == {}
    assert not scheduler.review_required
    assert not (scheduler.output / "workers/fit_0001_gpu0.json").exists()
    assert not (scheduler.output / "workers/fit_0001_gpu0.cuda.json").exists()
    receipt = old._read(scheduler.output / "workers/fit_0001_gpu0.resource_loss.json")
    assert receipt["CUDA_handshake_published"] is False
    assert receipt["termination"]["verified_original_child_exited"] is True


def test_arrival_deduplication_distinguishes_birth_and_worker(tmp_path, monkeypatch):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    record = dict(worker_id="fit_a", gpu=0)
    decision = dict(blocking_processes=[dict(pid=999, start_time_ticks=10, classification="foreign")])
    scheduler._record_arrival(record, decision)
    scheduler._record_arrival(record, decision)
    decision["blocking_processes"][0]["start_time_ticks"] = 11
    scheduler._record_arrival(record, decision)
    scheduler._record_arrival(dict(worker_id="fit_b", gpu=0), decision)
    assert len(list((tmp_path / runner.FOLDER / "foreign_arrivals").glob("*.json"))) == 3


@pytest.mark.parametrize("kind", ["python", "phase", "worker", "extra", "subcommand"])
def test_new_worker_argv_must_match_exact_registered_command(tmp_path, monkeypatch, kind):
    scheduler = bare_scheduler(tmp_path, monkeypatch)
    scheduler.phase = "fit"
    command = runner.worker_command(tmp_path, "fixture", phase="fit")
    if kind == "python":
        command[0] = "/usr/bin/python"
    elif kind == "phase":
        command[-1] = "evaluate"
    elif kind == "worker":
        command[command.index("--worker-id") + 1] = "another"
    elif kind == "extra":
        command.append("--unregistered")
    else:
        command[2] = "run"
    with pytest.raises(ValueError, match="argv"):
        scheduler._verify_command(command, "fixture")


@pytest.fixture
def runtime_control(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    source = tmp_path / "cooperative.py"
    source.write_text("# runtime source fixture\n")
    head = tmp_path / "heads.py"
    head.write_text("# resumable source fixture\n")
    legacy = tmp_path / "legacy.py"
    legacy.write_text("# original scientific source fixture\n")
    monkeypatch.setattr(runner, "__file__", str(source))
    monkeypatch.setattr(runner, "HEAD_SOURCE", head)
    old._write(study / "protocol.json", dict(settings=dict(seed=42, epochs=12)))
    old._write(study / "parallel_resource_protocol.json", {})
    resource = dict(bindings={str(legacy): old._sha(legacy)})
    monkeypatch.setattr(old, "check_protocol", lambda path: resource)
    bindings = dict(resource["bindings"])
    bindings.update({str(p): old._sha(p) for p in (source, head)})
    caches = {}
    for split in ("train", "validation"):
        cache = study / "caches" / f"{split}.npy"
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(b"finite-feature-fixture")
        manifest = cache.with_suffix(".manifest.json")
        old._write(manifest, dict(complete=True, path=str(cache), sha256=old._sha(cache)))
        bindings[str(manifest)] = old._sha(manifest)
        stat = cache.stat()
        caches[split] = dict(sha256=old._sha(cache), stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    control = dict(schema=runner.SCHEMA, study=str(study),
                   root_protocol_sha256=old._sha(study / "protocol.json"),
                   resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json"),
                   unchanged_scientific_settings=dict(seed=42, epochs=12),
                   policy=copy.deepcopy(runner.POLICY), bindings=bindings, cache_identities=caches)
    def rewrite():
        old._write(runner.control_path(study), control)
        old._write(runner.control_path(study).with_name("control_registration.json"),
                   dict(protocol_sha256=old._sha(runner.control_path(study))))
    rewrite()
    return study, control, source, head, legacy, rewrite


@pytest.mark.parametrize("source_index", [2, 3, 4])
def test_changed_actual_runtime_or_legacy_source_is_rejected(runtime_control, source_index):
    study, _, source, head, legacy, _ = runtime_control
    assert runner.check_control(study)
    (source, head, legacy)[source_index - 2].write_text("# changed source\n")
    with pytest.raises(ValueError, match="source"):
        runner.check_control(study)


def test_runtime_control_cannot_discard_original_source_binding(runtime_control):
    study, control, _, _, legacy, rewrite = runtime_control
    del control["bindings"][str(legacy)]
    rewrite()
    with pytest.raises(ValueError, match="discarded original"):
        runner.check_control(study)


@pytest.mark.parametrize("key,value", [("active_foreign", "stop"), ("never_terminate_foreign_processes", False),
                                        ("max_cuda_workers", 1), ("reserved_gpus", [7])])
def test_runtime_control_rejects_policy_drift(runtime_control, key, value):
    study, control, _, _, _, rewrite = runtime_control
    control["policy"][key] = value
    rewrite()
    with pytest.raises(ValueError, match="policy"):
        runner.check_control(study)


@pytest.mark.parametrize("missing", ["train", "validation"])
def test_runtime_control_requires_both_feature_cache_identities(runtime_control, missing):
    study, control, _, _, _, rewrite = runtime_control
    del control["cache_identities"][missing]
    rewrite()
    with pytest.raises(ValueError, match="feature identities"):
        runner.check_control(study)


@pytest.fixture
def bound_worker(tmp_path, monkeypatch):
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_readout_study as base
    study = tmp_path
    old._write(study / "protocol.json", {})
    old._write(study / "parallel_resource_protocol.json", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner, "check_control", lambda path: {})
    monkeypatch.setattr(base, "load_protocol", lambda *args, **kwargs: {})
    monkeypatch.setattr(old, "check_protocol", lambda path: dict(registered_system_display={}))
    command = runner.worker_command(study, "fixture", phase="fit")
    current = dict(identity(os.getpid()), argv=command, cwd=str(runner.ROOT))
    monkeypatch.setattr(old, "process_identity", lambda pid: current)
    cards = [dict(index=0, pids=[os.getpid()])]
    monkeypatch.setattr(old.admission, "full_gpu_inventory", lambda *args: cards)
    ledger = dict(process_identity=copy.deepcopy(current), worker_id="fixture", physical_gpu=0,
                  CUDA_VISIBLE_DEVICES="0", phase="fit", command=command,
                  script_sha256=old._sha(runner.__file__),
                  root_protocol_sha256=old._sha(study / "protocol.json"),
                  resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json"))
    path = features.worker_directory(study) / "fixture.json"
    def rewrite():
        old._write(path, ledger)
        old._write(path.with_suffix(".registration.json"), dict(ledger_sha256=old._sha(path)))
    rewrite()
    return study, ledger, cards, rewrite


def test_worker_binding_rechecks_empty_card_before_cuda(bound_worker):
    study, ledger, cards, _ = bound_worker
    assert runner.bind_worker(study, "fixture", "fit") == ledger
    cards[0]["pids"].append(999)
    with pytest.raises(ValueError, match="foreign work before CUDA"):
        runner.bind_worker(study, "fixture", "fit")


@pytest.mark.parametrize("phase", [None, True, "extract_test", "unknown"])
def test_new_worker_phase_is_explicitly_restricted(bound_worker, phase):
    study, _, _, _ = bound_worker
    with pytest.raises(ValueError, match="phase"):
        runner.worker_command(study, "fixture", phase=phase)
    with pytest.raises(ValueError, match="phase"):
        runner.bind_worker(study, "fixture", phase)


@pytest.mark.parametrize("field,value", [("phase", "evaluate"), ("physical_gpu", 1),
    ("CUDA_VISIBLE_DEVICES", "1"), ("script_sha256", "wrong"), ("resource_protocol_sha256", "wrong"),
    ("root_protocol_sha256", "wrong"), ("worker_id", "another")])
def test_worker_binding_rejects_wrong_registered_ledger_identity(bound_worker, field, value):
    study, ledger, _, rewrite = bound_worker
    ledger[field] = value
    rewrite()
    with pytest.raises(ValueError, match="identity, card, phase"):
        runner.bind_worker(study, "fixture", "fit")


@pytest.mark.parametrize("field,value", [("pid", 999), ("uid", -1), ("start_time_ticks", 101),
    ("command_sha256", "wrong"), ("cwd", "/different"), ("argv", ["unregistered"])])
def test_worker_binding_rejects_wrong_exact_process_birth_or_command(bound_worker, field, value):
    study, ledger, _, rewrite = bound_worker
    ledger["process_identity"][field] = value
    rewrite()
    with pytest.raises(ValueError, match="identity, card, phase"):
        runner.bind_worker(study, "fixture", "fit")


def run_coordinator_fixture(tmp_path, monkeypatch, *, reject_lock=False):
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_readout_study as base
    old._write(runner.control_path(tmp_path), {})
    old._write(tmp_path / "protocol.json", {})
    monkeypatch.setattr(runner, "check_control", lambda study: dict(runtime_environment={}))
    monkeypatch.setattr(runner, "verify_retired", lambda study: None)
    monkeypatch.setattr(old.admission, "process_birth", lambda pid: dict(pid=pid, uid=os.getuid(), start_time_ticks=99))
    monkeypatch.setattr(base, "mark_state", lambda *args, **kwargs: None)
    events = []
    def barrier(study):
        events.append("lock")
        assert events[0] == "fit"
        if reject_lock:
            raise ValueError("No valid selection lock")
    monkeypatch.setattr(base, "validate_selection_lock", barrier)
    def prepare(study, split):
        assert events[-1] == "lock"
        events.append("prepare_test")
    monkeypatch.setattr(features, "prepare_split", prepare)
    monkeypatch.setattr(features, "merge_split", lambda study, split: events.append("merge_test"))
    class Scheduler:
        def __init__(self, study):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def run_single(self, phase, factory):
            events.append(phase)
            if phase == "evaluate":
                old._write(tmp_path / "analysis/results.json", dict(complete=True))
        def run_workers(self, phase, *args, **kwargs):
            events.append(phase)
    monkeypatch.setattr(runner, "CooperativeScheduler", Scheduler)
    return events


def test_coordinator_opens_test_only_after_fit_and_valid_selection_lock(tmp_path, monkeypatch):
    events = run_coordinator_fixture(tmp_path, monkeypatch)
    runner.run(tmp_path)
    assert events == ["fit", "lock", "prepare_test", "extract_test", "merge_test", "evaluate"]


def test_failed_selection_lock_prevents_every_test_operation(tmp_path, monkeypatch):
    events = run_coordinator_fixture(tmp_path, monkeypatch, reject_lock=True)
    with pytest.raises(ValueError, match="No valid selection lock"):
        runner.run(tmp_path)
    assert events == ["fit", "lock"]


def test_evaluate_worker_stop_reaches_original_hook_and_restores_it(tmp_path, monkeypatch):
    import torch
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_resumable_heads as heads

    old._write(runner.control_path(tmp_path), {})
    workers = features.worker_directory(tmp_path)
    workers.mkdir(parents=True)
    (workers / "fixture.STOP").write_text("only this exact worker must stop\n")
    monkeypatch.setattr(runner, "bind_worker", lambda *args: dict(physical_gpu=0))
    monkeypatch.setattr(runner, "check_control", lambda *args: dict(selection_artifacts=[]))
    monkeypatch.setattr(base, "load_protocol", lambda *args, **kwargs: {})
    barrier_checks = []
    monkeypatch.setattr(base, "validate_selection_lock", lambda *args: barrier_checks.append(True))
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "mock CUDA; CPU tensors only")
    cpu_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: cpu_zeros(*args, **dict(kwargs, device="cpu")))
    monkeypatch.setattr(heads.original, "load_cache", lambda *args, **kwargs: pytest.fail("A stopped worker must not read test data"))
    original_stop = heads.original.check_stop
    # Keep the actual evaluate alias: its original-module globals must see the
    # runner's worker-specific hook, including on its first selection-lock check.
    assert heads.evaluate is heads.original.evaluate
    with pytest.raises(RuntimeError, match="Parallel worker stopped:.*only this exact worker"):
        runner.head_worker(tmp_path, "fixture", "evaluate")
    assert barrier_checks == [True]
    assert heads.original.check_stop is original_stop
    assert not (workers / "fixture.complete.json").exists()
