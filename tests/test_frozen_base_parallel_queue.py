import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from scripts.hanguard import frozen_base_parallel_queue as queue


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    script = tmp_path / "worker.py"
    script.write_text("# CPU fixture worker\n")
    data = tmp_path / "data.json"
    data.write_text("{}\n")
    protocol = dict(inputs={str(data): queue._sha(data)},
                    code_sha256={str(script): queue._sha(script)}, scope="CPU fixture")
    queue._write(root / "protocol.json", protocol)
    queue._write(root / "protocol_registration.json",
                 dict(protocol_sha256=queue._sha(root / "protocol.json")))
    queue._write(root / "state.json", dict(formal_test_opened=False))
    monkeypatch.setattr(queue.admission, "capture_blockers", lambda: [])
    monkeypatch.setattr(queue.admission, "capture_display", lambda: dict(verified=False))
    return root, script


def identity(pid, argv=None, ticks=None, uid=None):
    return dict(pid=pid, uid=os.getuid() if uid is None else uid,
        start_time_ticks=pid * 10 if ticks is None else ticks,
        command_sha256=f"{pid:064x}", argv=argv or ["foreign"], cwd=str(queue.ROOT))


class FakeChild:
    def __init__(self, world, pid, command, environment):
        self.world, self.pid, self.command = world, pid, command
        self.gpu = int(environment["CUDA_VISIBLE_DEVICES"])
        self.code = None
        self.started_tick = world.tick

    def poll(self):
        return self.code


class World:
    def __init__(self, *, free=None, finish_after=2):
        self.free = set(range(8) if free is None else free)
        self.children = {}
        self.launched = []
        self.finished = []
        self.tick = 0
        self.now = 1000.0
        self.finish_after = finish_after
        self.on_sleep = None
        self.foreign = {}
        self.signals = []
        self.identities = {}

    def inventory(self):
        result = []
        for gpu in range(8):
            pids = ([] if gpu in self.free else [9000 + gpu])
            pids += [child.pid for child in self.children.values()
                     if child.gpu == gpu and child.code is None]
            pids += self.foreign.get(gpu, [])
            result.append(dict(index=gpu, uuid=f"GPU-{gpu}", pids=pids, free_gib=40))
        return result

    def read_identity(self, pid):
        if pid in self.identities:
            return self.identities[pid]
        if pid in self.children:
            child = self.children[pid]
            return identity(pid, child.command)
        return identity(pid)

    def popen(self, command, cwd, env, stdout, stderr):
        assert cwd == str(queue.ROOT)
        assert "CUDA_VISIBLE_DEVICES" in env
        pid = 1000 + len(self.children)
        child = FakeChild(self, pid, command, env)
        self.children[pid] = child
        self.launched.append(child)
        return child

    def sleep(self, seconds):
        self.now += seconds
        self.tick += 1
        if self.on_sleep:
            self.on_sleep(self)
        for child in self.children.values():
            if child.code is None and self.tick - child.started_tick >= self.finish_after:
                child.code = 0
                self.finished.append(child.pid)

    def kill(self, pid, number):
        self.signals.append((pid, number))
        self.children[pid].code = -int(number)

    def scheduler(self, study):
        return queue.ParallelScheduler(study, inventory=self.inventory,
            identity_reader=self.read_identity, birth_reader=lambda pid: None,
            popen=self.popen, sleep=self.sleep, clock=lambda: self.now,
            terminate_grace_seconds=0)


def factory(study, script):
    def command(worker_id, gpu):
        return [str(queue.ROOT / ".venv-hanguard/bin/python"), str(script), "feature-worker",
                "--study", str(study), "--worker-id", worker_id, "--split", "train"]
    return command


def test_registers_eight_gpus_without_a_spare(study):
    root, _ = study
    protocol = queue.initialize(root)
    assert protocol["max_cuda_workers"] == 8
    assert protocol["reserved_gpus"] == []
    assert set(protocol["candidate_gpus"]) == set(range(8))
    assert protocol["admission_seconds"] == 0
    assert queue.check_protocol(root) == protocol


def test_all_eight_free_gpus_launch_concurrently_and_identity_ledgers(study):
    root, script = study
    world = World()
    with world.scheduler(root) as scheduler:
        scheduler.run_workers("train", factory(root, script), lambda: len(world.finished) == 8,
            work_remaining=lambda: len(world.launched) < 8, split="train")
    assert len(world.launched) == 8
    assert {child.gpu for child in world.launched} == set(range(8))
    assert {child.started_tick for child in world.launched} == {0}
    assert not world.signals
    for child in world.launched:
        worker_id = child.command[child.command.index("--worker-id") + 1]
        ledger = queue._read(root / "parallel_execution/workers" / f"{worker_id}.json")
        assert ledger["process_identity"]["argv"] == child.command
        assert ledger["process_identity"]["start_ticks"] == child.pid * 10
        assert ledger["physical_gpu"] == child.gpu
        assert ledger["CUDA_VISIBLE_DEVICES"] == str(child.gpu)
        assert ledger["split"] == "train"


def test_later_free_cards_are_added_while_first_worker_runs(study):
    root, script = study
    world = World(free={2}, finish_after=4)
    world.on_sleep = lambda value: value.free.update({4, 5, 7}) if value.tick == 1 else None
    with world.scheduler(root) as scheduler:
        scheduler.run_workers("train", factory(root, script), lambda: len(world.finished) == 4,
            work_remaining=lambda: len(world.launched) < 4, split="train")
    assert [child.gpu for child in world.launched] == [2, 4, 5, 7]
    assert world.launched[0].started_tick == 0
    assert all(child.started_tick == 1 for child in world.launched[1:])
    assert not world.signals


def test_busy_same_uid_gpu_is_not_admitted(study):
    root, script = study
    world = World(free={2})
    with world.scheduler(root) as scheduler:
        scheduler.run_workers("train", factory(root, script), lambda: len(world.finished) == 1,
            work_remaining=lambda: not world.launched)
    assert [child.gpu for child in world.launched] == [2]
    assert not world.signals


def test_single_scientific_stage_uses_one_worker(study):
    root, script = study
    world = World()
    with world.scheduler(root) as scheduler:
        scheduler.run_single("heads", factory(root, script))
    assert len(world.launched) == 1
    assert queue._read(root / "parallel_resource_protocol.json")["max_cuda_workers"] == 8


def test_foreign_arrival_stops_only_our_exact_workers_and_disables_retry(study, monkeypatch):
    root, script = study
    world = World(free={2, 4}, finish_after=100)
    world.on_sleep = lambda value: value.foreign.update({2: [99999]})
    monkeypatch.setattr(queue.os, "kill", world.kill)
    with pytest.raises(queue.ResourceReviewRequired, match="Foreign process"):
        with world.scheduler(root) as scheduler:
            scheduler.run_workers("train", factory(root, script), lambda: False,
                work_remaining=lambda: len(world.launched) < 2)
    assert {pid for pid, number in world.signals} == {1000, 1001}
    assert all(number == signal.SIGTERM for pid, number in world.signals)
    state = queue._read(root / "parallel_execution/queue_state.json")
    assert state["requires_review"] is True
    assert state["automatic_training_launch_disabled"] is True
    assert state["formal_test_opened"] is False
    with pytest.raises(queue.ResourceReviewRequired, match="Previously stopped"):
        with world.scheduler(root):
            pass


def test_pid_reuse_is_never_signalled(study, monkeypatch):
    root, script = study
    world = World(free={2}, finish_after=100)
    world.on_sleep = lambda value: value.identities.update({1000: identity(1000, ["foreign"], ticks=999)})
    monkeypatch.setattr(queue.os, "kill", world.kill)
    with pytest.raises(queue.ResourceReviewRequired, match="identity changed"):
        with world.scheduler(root) as scheduler:
            scheduler.run_workers("train", factory(root, script), lambda: False,
                work_remaining=lambda: not world.launched)
    assert world.signals == []


def test_stop_while_waiting_launches_nothing(study):
    root, script = study
    world = World(free=set())
    world.on_sleep = lambda value: (root / "STOP").write_text("User stop\n")
    with pytest.raises(queue.StopRequested, match="User stop"):
        with world.scheduler(root) as scheduler:
            scheduler.run_workers("train", factory(root, script), lambda: False)
    assert not world.launched


def test_unknown_process_blocks_even_with_abundant_memory():
    rows = [dict(index=2, uuid="GPU-2", pids=[1234], free_gib=80)]
    def unknown(pid):
        raise PermissionError("Foreign cmdline unreadable")
    decisions, _ = queue.assess_cards(rows, dict(gpu6_pipeline_births=[]), {},
        identity_reader=unknown, birth_reader=lambda pid: dict(pid=pid, uid=1047, start_time_ticks=5))
    candidate = next(value for value in decisions if value["index"] == 2)
    assert candidate["eligible"] is False
    assert candidate["blocking_processes"][0]["classification"] == "foreign"
    assert candidate["blocking_processes"][0]["uid"] == 1047


def test_unverified_same_uid_identity_remains_unknown():
    rows = [dict(index=2, uuid="GPU-2", pids=[1234], free_gib=80)]
    def unknown(pid):
        raise PermissionError("Command identity unavailable")
    decisions, _ = queue.assess_cards(rows, dict(gpu6_pipeline_births=[]), {},
        identity_reader=unknown, birth_reader=lambda pid: dict(pid=pid, uid=os.getuid(), start_time_ticks=5))
    candidate = next(value for value in decisions if value["index"] == 2)
    assert candidate["eligible"] is False
    assert candidate["blocking_processes"][0]["classification"] == "unknown"


def test_feature_resource_loss_preserves_other_worker_and_redispatches_pending(study, monkeypatch):
    root, script = study
    world = World(free={2, 4}, finish_after=4)
    lost = []
    def tick(value):
        if value.tick == 1:
            value.foreign[2] = [99999]
        if value.tick == 2:
            value.free.add(5)
    world.on_sleep = tick
    monkeypatch.setattr(queue.os, "kill", world.kill)
    def abandon(worker_id, gpu, split, reason):
        assert world.children[1000].poll() is not None
        assert world.children[1001].poll() is None
        lost.append((worker_id, gpu, split, reason))
    with world.scheduler(root) as scheduler:
        scheduler.run_workers("train", factory(root, script), lambda: len(world.finished) == 2,
            work_remaining=lambda: len(world.launched) < 3, split="train", on_resource_loss=abandon)
    assert [child.gpu for child in world.launched] == [2, 4, 5]
    assert world.signals == [(1000, signal.SIGTERM)]
    assert len(lost) == 1 and lost[0][1:3] == (2, "train")
    state = queue._read(root / "parallel_execution/queue_state.json")
    assert state["requires_review"] is False
    assert len(state["resource_losses"]) == 1


def test_unknown_resource_loss_still_stops_phase_without_callback(study, monkeypatch):
    root, script = study
    world = World(free={2}, finish_after=100)
    world.on_sleep = lambda value: value.foreign.update({2: [99999]})
    monkeypatch.setattr(queue.os, "kill", world.kill)
    original_reader = world.read_identity
    def unknown(pid):
        if pid == 99999:
            raise PermissionError("Unknown owner")
        return original_reader(pid)
    called = []
    scheduler = world.scheduler(root)
    scheduler.identity_reader = unknown
    with pytest.raises(queue.ResourceReviewRequired, match="Foreign process"):
        with scheduler:
            scheduler.run_workers("train", factory(root, script), lambda: False,
                work_remaining=lambda: not world.launched, split="train",
                on_resource_loss=lambda *args: called.append(args))
    assert called == []
    assert world.signals == [(1000, signal.SIGTERM)]


def test_gpu6_pipeline_birth_blocks_without_cuda_pid():
    record = dict(pid=1234, expected_uid=1047, uid=1047, start_time_ticks=5,
                  gpu=6, registration_status="verified")
    rows = [dict(index=6, uuid="GPU-6", pids=[], free_gib=80)]
    decisions, _ = queue.assess_cards(rows, dict(gpu6_pipeline_births=[record]), {},
        birth_reader=lambda pid: dict(pid=pid, uid=1047, start_time_ticks=5, state="S"))
    candidate = next(value for value in decisions if value["index"] == 6)
    assert candidate["eligible"] is False
    assert candidate["pipeline_blocked"] is True


def test_policy_and_source_tampering_fail_closed(study):
    root, script = study
    queue.initialize(root)
    script.write_text("# edited after registration\n")
    with pytest.raises(ValueError, match="source changed"):
        queue.check_protocol(root)


def test_reservation_cannot_be_added_to_registered_policy(study):
    root, _ = study
    queue.initialize(root)
    path = root / "parallel_resource_protocol.json"
    protocol = queue._read(path)
    protocol["reserved_gpus"] = [7]
    queue._write(path, protocol)
    queue._write(root / "parallel_resource_registration.json", dict(protocol_sha256=queue._sha(path)))
    with pytest.raises(ValueError, match="policy changed"):
        queue.check_protocol(root)


def test_no_gpu_worker_is_launched_without_registered_script(study):
    root, script = study
    world = World()
    def wrong(worker_id, gpu):
        command = factory(root, script)(worker_id, gpu)
        command[1] = str(queue.ROOT / "scripts/hanguard/frozen_base_readout_heads.py")
        return command
    with pytest.raises(ValueError, match="script is not registered"):
        with world.scheduler(root) as scheduler:
            scheduler.run_workers("train", wrong, lambda: False)
    assert not world.launched


def test_owned_identity_requires_all_fields():
    original = identity(1000, ["exact", "--study", "/a"])
    for field in queue.IDENTITY_KEYS:
        changed = dict(original)
        changed[field] = "changed"
        assert not queue.same_identity(changed, original)


def test_prior_unfinished_workers_are_not_silently_adopted(study):
    root, script = study
    queue.initialize(root)
    queue._write(root / "parallel_execution/queue_state.json",
                 dict(active_workers=2, requires_review=False, monitor_live=True))
    world = World()
    with pytest.raises(queue.ResourceReviewRequired, match="interrupted"):
        with world.scheduler(root):
            pass
    assert not world.launched


def test_real_cpu_child_waits_for_exact_ledger_and_exits_successfully(study):
    root, script = study
    script.write_text("""import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('stage')
p.add_argument('--study')
p.add_argument('--worker-id')
p.add_argument('--split')
a=p.parse_args()
folder=Path(a.study)/'parallel_execution'/'workers'
ledger=folder/(a.worker_id+'.json')
registration=folder/(a.worker_id+'.registration.json')
deadline=time.monotonic()+10
while not ledger.exists() or not registration.exists():
    if time.monotonic()>deadline: raise RuntimeError('Missing ledger')
    time.sleep(.01)
v=json.loads(ledger.read_text())
assert v['process_identity']['pid']==os.getpid()
assert v['physical_gpu']==int(os.environ['CUDA_VISIBLE_DEVICES'])
(folder/(a.worker_id+'.done')).write_text('done')
""")
    protocol = queue._read(root / "protocol.json")
    protocol["code_sha256"][str(script)] = queue._sha(script)
    queue._write(root / "protocol.json", protocol)
    queue._write(root / "protocol_registration.json", dict(protocol_sha256=queue._sha(root / "protocol.json")))
    inventory = lambda: [dict(index=2, uuid="GPU-2", pids=[], free_gib=40)]
    with queue.ParallelScheduler(root, inventory=inventory, birth_reader=lambda pid: None) as scheduler:
        scheduler.run_single("cpu_smoke", factory(root, script))
    ledgers = list((root / "parallel_execution/workers").glob("*.done"))
    assert len(ledgers) == 1
    state = queue._read(root / "parallel_execution/queue_state.json")
    assert state["completed_workers"][0]["exit_code"] == 0
    assert state["requires_review"] is False
