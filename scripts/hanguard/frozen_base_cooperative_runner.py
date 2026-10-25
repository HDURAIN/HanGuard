"""Empty-card admission, continued ownership, and resumable frozen-base heads.

This append-only runtime amendment preserves the scientific protocol and every
old source. Foreign arrivals after admission are recorded, never a reason to
terminate our work. New workers still need a genuinely empty card.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_parallel_queue as queue
from scripts.hanguard import frozen_base_parallel_queue_v2 as queue2

SCHEMA = "hanguard_cooperative_resumable_runtime_v1"
FOLDER = "recovery_004"
HEAD_SOURCE = ROOT / "scripts/hanguard/frozen_base_resumable_heads.py"
POLICY = dict(prelaunch_foreign="block", active_foreign="record_only",
              never_terminate_foreign_processes=True, max_cuda_workers=8,
              reserved_gpus=[], checkpoint_frequency="every_complete_epoch",
              automatic_retry=False)
read, sha, write = queue._read, queue._sha, queue._write


def control_path(study):
    return Path(study) / FOLDER / "control_protocol.json"


def check_control(study):
    study = Path(study).resolve()
    resource = queue.check_protocol(study)
    path = control_path(study)
    control = read(path)
    if read(path.with_name("control_registration.json")).get("protocol_sha256") != sha(path):
        raise ValueError("Runtime amendment registration changed")
    scientific = read(study / "protocol.json")
    if (control.get("schema") != SCHEMA or control.get("study") != str(study)
            or control.get("root_protocol_sha256") != sha(study / "protocol.json")
            or control.get("resource_protocol_sha256") != sha(study / "parallel_resource_protocol.json")
            or control.get("unchanged_scientific_settings") != scientific["settings"]
            or control.get("policy") != POLICY):
        raise ValueError("Runtime scientific binding or policy changed")
    bindings = control.get("bindings", {})
    if any(bindings.get(p) != digest for p, digest in resource["bindings"].items()):
        raise ValueError("Runtime amendment discarded original source bindings")
    for path in (Path(__file__).resolve(), HEAD_SOURCE):
        if bindings.get(str(path)) != sha(path):
            raise ValueError("Runtime source is not registered")
    for filename, digest in bindings.items():
        if sha(filename) != digest:
            raise ValueError(f"Runtime source/evidence changed: {filename}")
    if set(control.get("cache_identities", {})) != {"train", "validation"}:
        raise ValueError("Both training and validation feature identities are required")
    for split, identity in control["cache_identities"].items():
        manifest_path = study / "caches" / f"{split}.manifest.json"
        if bindings.get(str(manifest_path)) != sha(manifest_path):
            raise ValueError("Frozen feature manifest is not registered")
        manifest = read(manifest_path)
        if manifest.get("complete") is not True or manifest.get("sha256") != identity["sha256"]:
            raise ValueError("Frozen feature identity changed")
        stat = Path(manifest["path"]).stat()
        if dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity["stat"]:
            raise ValueError("Frozen feature file metadata changed")
    return control


def verify_retired(study):
    """Same exact-PID/birth retirement check used by the audited scheduler."""
    from types import SimpleNamespace
    proxy = SimpleNamespace(output=Path(study) / "parallel_execution",
                            birth_reader=queue.admission.process_birth)
    queue2.ParallelScheduler._verify_previous_workers_exited(proxy)


def register(study):
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_queue_v4 as prior

    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / FOLDER).exists() or (study / "fit_attempts/002").exists():
            raise FileExistsError("Runtime amendment already registered")
        previous = prior.check_control3(study)
        scientific = base.load_protocol(study, full_hash=False)
        state, stopped = read(study / "state.json"), read(study / "parallel_execution/queue_state.json")
        if (state.get("state") != "requires_review" or state.get("head_fit_attempt") != 2
                or state.get("formal_test_opened") or stopped.get("requires_review") is not True
                or stopped.get("phase") != "fit"
                or stopped.get("reason") != "Foreign process entered an active GPU"):
            raise ValueError("Expected the reviewed second head-fit resource interruption")
        if (study / "selection_lock.json").exists() or (study / "parallel_features/test").exists():
            raise ValueError("A selected or tested run must never be discarded")
        verify_retired(study)
        old_birth = read(study / "recovery_003/scheduler_recovery_entry.json")["coordinator_birth"]
        current = queue.admission.process_birth(old_birth["pid"])
        if current and current.get("start_time_ticks") == old_birth["start_time_ticks"] and current.get("state") not in ("Z", "X"):
            raise ValueError("Old exact coordinator is still alive")
        heads = study / "heads"
        files = {str(p.relative_to(heads)) for p in heads.rglob("*") if p.is_file()}
        if files != {"fit_started.json", "budget_12/layer_history.json"}:
            raise ValueError("Unexpected old head artifact: review before restarting")
        history = read(heads / "budget_12/layer_history.json")
        if [row["epoch"] for row in history] != [1, 2]:
            raise ValueError("Old history differs from the two reviewed completed epochs")
        marker = read(heads / "fit_started.json")
        if marker.get("protocol_sha256") != sha(study / "protocol.json") or marker.get("test_opened") is not False:
            raise ValueError("Old fit identity changed")
        folder = study / FOLDER
        folder.mkdir()
        sources = folder / "source_snapshot"
        sources.mkdir()
        for path in (Path(__file__).resolve(), HEAD_SOURCE):
            shutil.copy2(path, sources / path.name)
        for name, path in {"state.before.json": study / "state.json",
                "queue_state.before.json": study / "parallel_execution/queue_state.json",
                "termination_evidence.before.json": study / "parallel_execution/termination_evidence.json"}.items():
            shutil.copy2(path, folder / name)
        archive = study / "fit_attempts/002"
        archive.mkdir(parents=True)
        heads.rename(archive / "heads")
        manifest = dict(attempt=2, next_attempt=3, completed_layer_epochs=2,
                        fusion_started=False, partial_checkpoint_reuse=False,
                        restart_seed=scientific["settings"]["seed"],
                        reason="user_authorized_resource_policy_amendment",
                        files={str(p.relative_to(archive)): sha(p) for p in (archive / "heads").rglob("*") if p.is_file()})
        write(archive / "archive_manifest.json", manifest, exclusive=True)
        bindings = dict(previous["bindings"])
        evidence = [study / "recovery_003/control_protocol.json", study / "recovery_003/control_registration.json",
                    *sorted((study / "recovery_003/failure_events").glob("*.json"))]
        paths = [Path(__file__).resolve(), HEAD_SOURCE, *sources.iterdir(), *evidence,
                 *folder.glob("*.before.json"), archive / "archive_manifest.json",
                 *[p for p in (archive / "heads").rglob("*") if p.is_file()]]
        caches = {}
        for split in ("train", "validation"):
            path = study / "caches" / f"{split}.manifest.json"
            value = read(path)
            if (value.get("complete") is not True or value.get("rows") != scientific["rows"][split]
                    or value.get("protocol_sha256") != sha(study / "protocol.json")):
                raise ValueError("Verified canonical frozen features are required")
            stat = Path(value["path"]).stat()
            caches[split] = dict(sha256=value["sha256"], stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
            paths.append(path)
        bindings.update({str(p): sha(p) for p in paths})
        scratch = study / "runtime_scratch"
        environment = {key: str(scratch / value) for key, value in dict(TMPDIR="tmp", TRITON_CACHE_DIR="triton",
                        TORCHINDUCTOR_CACHE_DIR="torchinductor", CUDA_CACHE_PATH="cuda").items()}
        for path in environment.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        value = dict(schema=SCHEMA, study=str(study), root_protocol_sha256=sha(study / "protocol.json"),
            resource_protocol_sha256=sha(study / "parallel_resource_protocol.json"),
            unchanged_scientific_settings=scientific["settings"], policy=POLICY, bindings=bindings,
            cache_identities=caches, runtime_environment=environment, head_fit_attempt=3,
            old_attempt_reuse=False, same_seed_restart=True, prior_queue_sha256=sha(folder / "queue_state.before.json"),
            selection_artifacts=[str(control_path(study)), *[str(p) for p in sorted(sources.iterdir())]],
            user_authorization="改成新的策略，继续跑实验", created_unix=time.time())
        write(control_path(study), value, exclusive=True)
        write(folder / "control_registration.json", dict(protocol_sha256=sha(control_path(study))), exclusive=True)
        return check_control(study)


def worker_command(study, worker_id, *, phase):
    if phase not in ("fit", "evaluate"):
        raise ValueError("Invalid readout phase")
    return [str((ROOT / ".venv-hanguard/bin/python").absolute()), str(Path(__file__).resolve()),
            "head-worker", "--study", str(Path(study).resolve()), "--worker-id", worker_id, "--phase", phase]


class CooperativeScheduler(queue2.ParallelScheduler):
    def __init__(self, study, **kwargs):
        self.runtime_control = check_control(study)
        self.arrivals = set()
        queue.ParallelScheduler.__init__(self, study, **kwargs)

    def _check_control(self):
        if check_control(self.study) != self.runtime_control:
            raise ValueError("Runtime amendment changed after scheduler construction")

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_control()
            self._verify_previous_workers_exited()
            path = self.output / "queue_state.json"
            previous = read(path)
            entry = self.study / FOLDER / "scheduler_entry.json"
            if not entry.exists():
                if sha(path) != self.runtime_control["prior_queue_sha256"]:
                    raise ValueError("Queue does not match reviewed interruption")
                write(entry, dict(coordinator_birth=queue.admission.process_birth(os.getpid()),
                    control_sha256=sha(control_path(self.study)), entered_unix=self.clock()), exclusive=True)
            else:
                birth = queue.admission.process_birth(os.getpid())
                recorded = read(entry)
                if (recorded["coordinator_birth"] != birth
                        or recorded["control_sha256"] != sha(control_path(self.study))
                        or previous.get("requires_review") or previous.get("active_workers", 0)):
                    raise queue.ResourceReviewRequired("A stopped coordinator requires explicit review")
            self._state("CPU_cooperative_monitor_ready")
            return self
        except BaseException:
            self.lock_stream.close()
            self.lock_stream = None
            raise

    def _state(self, state, **details):
        details.update(head_fit_attempt=3, active_foreign_policy="record_only",
                       current_fit_started=(self.study / "heads/fit_started.json").exists(),
                       runtime_control_sha256=sha(control_path(self.study)))
        return queue.ParallelScheduler._state(self, state, **details)

    def _observe(self):
        self._check_control()
        decisions = queue.ParallelScheduler._observe(self)
        # Inventory reads can race normal worker exit. Reap before computing done.
        self._poll_workers()
        self._state("parallel_workers_running" if self.active else "CPU_wait_for_empty_candidates",
                    gpu_candidates=decisions)
        return decisions

    def _verify_command(self, command, worker_id):
        command = [str(v) for v in command]
        if len(command) > 1 and Path(command[1]).resolve() == Path(__file__).resolve():
            if self.phase not in ("fit", "evaluate") or command != worker_command(self.study, worker_id, phase=self.phase):
                raise ValueError("New worker argv does not match registered runtime entrypoint")
            self._check_control()
            return command, Path(__file__).resolve()
        return queue.ParallelScheduler._verify_command(self, command, worker_id)

    def _launch(self, gpu, command_factory, split, on_resource_loss=None):
        # A foreign arrival before publishing the ledger is still an admission
        # race: retire only this CPU-only child and wait for another empty card.
        # The inherited pre-handshake branch never invokes this callback.
        prelaunch_only = on_resource_loss or (lambda *args: None)
        return super()._launch(gpu, command_factory, split, prelaunch_only)

    def _record_arrival(self, record, decision):
        for process in decision["blocking_processes"]:
            key = (record["worker_id"], process["pid"], process.get("start_time_ticks"))
            if key in self.arrivals:
                continue
            self.arrivals.add(key)
            path = self.study / FOLDER / "foreign_arrivals" / f"{time.time_ns()}_{process['pid']}.json"
            write(path, dict(action="record_only_continue_owned_worker", worker_id=record["worker_id"],
                            physical_gpu=record["gpu"], process=process, recorded_unix=self.clock(),
                            runtime_control_sha256=sha(control_path(self.study))), exclusive=True)

    def run_workers(self, phase, command_factory, phase_complete, *, work_remaining=None,
                    split=None, max_workers=8, poll_seconds=2, on_resource_loss=None):
        if type(max_workers) is not int or not 1 <= max_workers <= 8 or not 0 < poll_seconds <= 5:
            raise ValueError("Invalid worker ceiling or polling interval")
        if not phase or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in phase) or self.active:
            raise ValueError("Invalid phase or an earlier phase remains active")
        self.phase = phase
        launched = 0
        try:
            while True:
                queue._check_stop(self.study)
                self._check_control()
                self._poll_workers()
                decisions = self._observe()
                done = bool(phase_complete())
                for decision in decisions:
                    record = next((r for r in self.active.values() if r["gpu"] == decision["index"]), None)
                    if record is not None and decision.get("blocking_processes"):
                        self._record_arrival(record, decision)
                if done and not self.active:
                    return self._state("phase_complete", phase_worker_launches=launched)
                has_work = not done and (work_remaining is None or bool(work_remaining()))
                if has_work:
                    for decision in decisions:
                        if len(self.active) >= max_workers:
                            break
                        if work_remaining is not None and not work_remaining():
                            break
                        if (not decision["eligible"] or decision.get("pids")
                                or any(r["gpu"] == decision["index"] for r in self.active.values())):
                            continue
                        if self._launch(decision["index"], command_factory, split, on_resource_loss):
                            launched += 1
                if not self.active and not has_work and not done:
                    self._resource_failure("No workers or pending work but phase incomplete", {})
                self.sleep(poll_seconds)
        except BaseException as exc:
            if not isinstance(exc, queue.StopRequested):
                self.review_required = True
            if self.active:
                self.stop_owned_workers(str(exc))
            raise


def bind_worker(study, worker_id, phase):
    if phase not in ("fit", "evaluate"):
        raise ValueError("Invalid readout phase")
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_features as features

    study = Path(study).resolve()
    features.safe_worker_id(worker_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in tuple(map(str, range(8))):
        raise ValueError("Exactly one physical GPU must be visible")
    check_control(study)
    base.load_protocol(study)
    path = features.worker_directory(study) / f"{worker_id}.json"
    deadline = time.monotonic() + 10
    while not path.exists() or not path.with_suffix(".registration.json").exists():
        features.check_stop(study, worker_id)
        if time.monotonic() >= deadline:
            raise ValueError("No exact scheduler worker ledger")
        time.sleep(.05)
    ledger = read(path)
    current = queue.process_identity(os.getpid())
    if (read(path.with_suffix(".registration.json")).get("ledger_sha256") != sha(path)
            or not queue.same_identity(current, ledger.get("process_identity"))
            or current["uid"] != os.getuid() or ledger.get("worker_id") != worker_id
            or ledger.get("physical_gpu") != int(visible) or ledger.get("CUDA_VISIBLE_DEVICES") != visible
            or ledger.get("phase") != phase or ledger.get("command") != worker_command(study, worker_id, phase=phase)
            or current["argv"] != ledger["command"] or current["cwd"] != str(ROOT)
            or ledger.get("script_sha256") != sha(__file__)
            or ledger.get("root_protocol_sha256") != sha(study / "protocol.json")
            or ledger.get("resource_protocol_sha256") != sha(study / "parallel_resource_protocol.json")):
        raise ValueError("Worker identity, card, phase or registered runtime differs")
    resource = queue.check_protocol(study)
    cards = queue.admission.full_gpu_inventory(resource["registered_system_display"])
    card = next((v for v in cards if v["index"] == int(visible)), None)
    if card is None or any(pid != os.getpid() for pid in card["pids"]):
        raise ValueError("GPU acquired foreign work before CUDA admission")
    return ledger


def head_worker(study, worker_id, phase):
    if phase not in ("fit", "evaluate"):
        raise ValueError("Invalid readout phase")
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_resumable_heads as heads
    import torch

    ledger = bind_worker(study, worker_id, phase)
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA device is required")
    ready = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    if not torch.isfinite(ready).all():
        raise RuntimeError("CUDA allocation readiness failed")
    write(features.worker_directory(study) / f"{worker_id}.cuda.json", dict(passed=True,
          pid=os.getpid(), physical_gpu=ledger["physical_gpu"], actual_cuda_allocation_verified=True,
          device_name=torch.cuda.get_device_name(0), runtime_control_sha256=sha(control_path(study))), exclusive=True)
    base.load_protocol(study, full_hash=True)
    control = check_control(study)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # The new fit loops delegate to the original hook; evaluation and cache
    # loading are original functions too. Wrap that common hook once.
    original_stop = heads.original.check_stop
    def worker_stop(path):
        original_stop(path)
        features.check_stop(path, worker_id)
    heads.original.check_stop = worker_stop
    try:
        if phase == "fit":
            base.mark_state(study, "fitting_readouts", training_started=True)
            heads.fit(study, device="cuda", extra_source_paths=control["selection_artifacts"])
            check_control(study)
            base.validate_selection_lock(study)
        else:
            check_control(study)
            base.validate_selection_lock(study)
            heads.evaluate(study, device="cuda")
    finally:
        heads.original.check_stop = original_stop
    write(features.worker_directory(study) / f"{worker_id}.complete.json", dict(complete=True,
          phase=phase, pid=os.getpid(), protocol_sha256=sha(Path(study) / "protocol.json")), exclusive=True)


def run(study):
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_parallel_study as original

    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = check_control(study)
        if (study / FOLDER / "coordinator_started.json").exists():
            raise RuntimeError("Interrupted coordinator requires review; no blind repeat")
        verify_retired(study)
        os.environ.update(control["runtime_environment"])
        write(study / FOLDER / "coordinator_started.json", dict(
              birth=queue.admission.process_birth(os.getpid()), time=time.time()), exclusive=True)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        base.mark_state(study, "waiting_for_readout_gpu", error_type=None, reason=None,
                        training_started=True, head_fit_attempt=3,
                        recovery_control_protocol_sha256=sha(control_path(study)))
        with CooperativeScheduler(study) as scheduler:
            scheduler.run_single("fit", lambda worker_id, gpu: worker_command(study, worker_id, phase="fit"))
            base.validate_selection_lock(study)
            base.mark_state(study, "validation_selections_locked")
            features.prepare_split(study, "test")
            base.mark_state(study, "extracting_test", formal_test_opened=True)
            scheduler.run_workers("extract_test",
                lambda worker_id, gpu: original.worker_command(study, worker_id, split="test"),
                lambda: features.split_complete(study, "test"),
                work_remaining=lambda: features.has_work(study, "test"), split="test", max_workers=8)
            base.mark_state(study, "merging_test")
            features.merge_split(study, "test")
            base.mark_state(study, "evaluating_locked_test")
            scheduler.run_single("evaluate", lambda worker_id, gpu: worker_command(study, worker_id, phase="evaluate"))
        result = study / "analysis/results.json"
        if not result.exists() or read(result).get("complete") is not True:
            raise ValueError("No completed test evaluation was published")
        write(study / FOLDER / "completion.json", dict(complete=True, results_sha256=sha(result),
              protocol_sha256=sha(study / "protocol.json"), runtime_control_sha256=sha(control_path(study))), exclusive=True)
        base.mark_state(study, "completed", backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "head-worker"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--phase", choices=("fit", "evaluate"))
    args = parser.parse_args(argv)
    if args.action == "init":
        result = register(args.study)
        print(dict(state="cooperative_runtime_registered", policy=result["policy"]), flush=True)
    elif args.action == "head-worker":
        if not args.worker_id or not args.phase:
            parser.error("head-worker requires worker-id and phase")
        head_worker(args.study.resolve(), args.worker_id, args.phase)
    else:
        try:
            run(args.study)
        except BaseException as exc:
            from scripts.hanguard import frozen_base_readout_study as base
            base.mark_state(args.study, "stopped" if isinstance(exc, (queue.StopRequested, KeyboardInterrupt))
                            else "requires_review", error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
