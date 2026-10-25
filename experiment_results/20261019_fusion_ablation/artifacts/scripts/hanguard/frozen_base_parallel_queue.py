"""CPU scheduler for identity-bound workers on all genuinely free public GPUs.

The registered policy permits eight workers and reserves no card. Admission
uses process ownership, never utilization. The only display exception and the
known GPU6 pipeline births are inherited from the separately audited admission
module. This module imports no torch or CUDA package.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_readout_admission_v2 as admission
from scripts.hanguard import ab_readout_parallel_queue as inventory_module

SCHEMA = "hanguard_frozen_base_dynamic_parallel_resource_v1"
CANDIDATES = (2, 4, 5, 7, 3, 0, 1, 6)
IDENTITY_KEYS = ("pid", "uid", "start_time_ticks", "command_sha256", "argv", "cwd")


class ResourceReviewRequired(RuntimeError):
    pass


class StopRequested(RuntimeError):
    pass


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid):
    value = inventory_module.process_identity(pid)
    return dict(value, start_ticks=value["start_time_ticks"])


def same_identity(current, expected):
    return (current is not None and expected is not None and
            all(current.get(key) == expected.get(key) for key in IDENTITY_KEYS))


def _check_stop(study):
    for directory in (Path(study), Path(study) / "parallel_execution"):
        path = directory / "STOP"
        if path.exists():
            raise StopRequested(path.read_text().strip() or "STOP requested")


def initialize(study):
    """Register a CPU-only dynamic eight-card policy exactly once."""
    study = Path(study).resolve()
    _check_stop(study)
    protocol_path = study / "parallel_resource_protocol.json"
    registration_path = study / "parallel_resource_registration.json"
    if protocol_path.exists():
        return check_protocol(study)
    if registration_path.exists() or (study / "parallel_execution/workers").exists():
        raise ValueError("Incomplete resource registration cannot be resumed")
    scientific, root_sha, inputs, sources = admission.queue._root_identity(study)
    bindings = dict(sources)
    for path in (Path(__file__), Path(admission.__file__), Path(inventory_module.__file__),
                 Path(admission.queue.__file__), Path(admission.queue.guardian.__file__),
                 Path(admission.queue.guardian.legacy.__file__),
                 study / "protocol.json", study / "protocol_registration.json"):
        filename, digest = str(path.resolve()), _sha(path)
        if filename in bindings and bindings[filename] != digest:
            raise ValueError(f"Source registration changed: {filename}")
        bindings[filename] = digest
    for filename, digest in bindings.items():
        if _sha(filename) != digest:
            raise ValueError(f"Source registration changed: {filename}")
    metadata = {}
    for filename, digest in inputs.items():
        stat = Path(filename).stat()
        metadata[filename] = dict(sha256=digest, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    protocol = dict(schema=SCHEMA, study=str(study), root_protocol_sha256=root_sha,
        bindings=bindings, input_identities=metadata, candidate_gpus=list(CANDIDATES),
        reserved_gpus=[], max_cuda_workers=8, dynamic_empty_gpu_admission=True,
        admission_seconds=0, automatic_retry=False, automatic_head_retry=False,
        automatic_feature_redispatch=True, preserve_committed_features=True,
        never_reuse_partial_features=True, never_terminate_foreign_processes=True,
        gpu6_pipeline_births=admission.capture_blockers(),
        registered_system_display=admission.capture_display(), created_unix=time.time(),
        scientific_scope=scientific.get("evaluation_scope", scientific.get("scope")))
    _write(protocol_path, protocol, exclusive=True)
    _write(registration_path, dict(protocol_sha256=_sha(protocol_path)), exclusive=True)
    (study / "parallel_execution/workers").mkdir(parents=True)
    return protocol


def check_protocol(study):
    study = Path(study).resolve()
    _check_stop(study)
    path = study / "parallel_resource_protocol.json"
    protocol = _read(path)
    if _read(study / "parallel_resource_registration.json").get("protocol_sha256") != _sha(path):
        raise ValueError("Parallel resource registration digest changed")
    if (protocol.get("schema") != SCHEMA or protocol.get("study") != str(study) or
            protocol.get("candidate_gpus") != list(CANDIDATES) or
            protocol.get("max_cuda_workers") != 8 or protocol.get("reserved_gpus") != [] or
            protocol.get("dynamic_empty_gpu_admission") is not True or
            protocol.get("admission_seconds") != 0 or
            protocol.get("automatic_retry") is not False or
            protocol.get("automatic_head_retry") is not False or
            protocol.get("automatic_feature_redispatch") is not True or
            protocol.get("preserve_committed_features") is not True or
            protocol.get("never_reuse_partial_features") is not True or
            protocol.get("never_terminate_foreign_processes") is not True):
        raise ValueError("Parallel resource policy changed")
    scientific, root_sha, inputs, sources = admission.queue._root_identity(study)
    if root_sha != protocol["root_protocol_sha256"]:
        raise ValueError("Scientific root protocol changed")
    if {name: identity["sha256"] for name, identity in protocol["input_identities"].items()} != inputs:
        raise ValueError("Scientific input identities changed")
    for filename, identity in protocol["input_identities"].items():
        stat = Path(filename).stat()
        if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Scientific input metadata changed: {filename}")
    if any(protocol["bindings"].get(filename) != digest for filename, digest in sources.items()):
        raise ValueError("Scientific source bindings changed")
    for filename, digest in protocol["bindings"].items():
        if _sha(filename) != digest:
            raise ValueError(f"Registered scheduler source changed: {filename}")
    return protocol


def assess_cards(rows, protocol, owned, *, identity_reader=process_identity,
                 birth_reader=admission.process_birth):
    """Unknown identities block; a matching study birth alone is insufficient."""
    blockers = [admission.blocker_status(record, birth_reader)
                for record in protocol["gpu6_pipeline_births"]]
    by_index = {row["index"]: row for row in rows}
    result = []
    for index in CANDIDATES:
        row = by_index.get(index)
        if row is None:
            result.append(dict(index=index, available=False, eligible=False, blocking_processes=[]))
            continue
        blocked, accepted = [], []
        for pid in sorted(set(int(value) for value in row["pids"])):
            try:
                identity = identity_reader(pid)
                if same_identity(identity, owned.get(pid)) and identity.get("uid") == os.getuid():
                    accepted.append(pid)
                    continue
                try:
                    owner = pwd.getpwuid(identity["uid"]).pw_name
                except KeyError:
                    owner = f"uid:{identity['uid']}"
                blocked.append(dict(pid=pid, uid=identity.get("uid"), owner=owner,
                    classification="foreign", start_time_ticks=identity.get("start_time_ticks"),
                    command_sha256=identity.get("command_sha256")))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                value = dict(pid=pid, classification="unknown", error_type=type(exc).__name__)
                try:
                    birth = birth_reader(pid)
                    if birth:
                        value.update(uid=birth["uid"], start_time_ticks=birth["start_time_ticks"])
                        if birth["uid"] != os.getuid():
                            value.update(classification="foreign",
                                         ownership_verified_by="proc_uid_and_process_birth")
                        try:
                            value["owner"] = pwd.getpwuid(birth["uid"]).pw_name
                        except KeyError:
                            value["owner"] = f"uid:{birth['uid']}"
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                blocked.append(value)
        pipeline_blocked = index == 6 and any(value["blocked"] for value in blockers)
        result.append(dict(row, available=True, blocking_processes=blocked,
            owned_pids=accepted, pipeline_blocked=pipeline_blocked,
            completely_idle=not row["pids"],
            eligible=not blocked and not pipeline_blocked and row["free_gib"] >= 4.0))
    return result, blockers


class ParallelScheduler:
    """One CPU coordinator; one exact main worker per admitted physical GPU."""

    def __init__(self, study, *, inventory=None, identity_reader=process_identity,
                 birth_reader=admission.process_birth, popen=subprocess.Popen,
                 sleep=time.sleep, clock=time.time, terminate_grace_seconds=5):
        self.study = Path(study).resolve()
        self.protocol = initialize(self.study)
        self.output = self.study / "parallel_execution"
        self.inventory = inventory or (lambda: admission.full_gpu_inventory(
            self.protocol["registered_system_display"]))
        self.identity_reader, self.birth_reader = identity_reader, birth_reader
        self.popen, self.sleep, self.clock = popen, sleep, clock
        self.terminate_grace_seconds = float(terminate_grace_seconds)
        self.active, self.completed, self.resource_losses = {}, [], []
        self.phase = None
        self.sequence = len(list((self.output / "workers").glob("*.json")))
        self.lock_stream = None
        self.started = self.clock()
        self.review_required = False

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_stream.close()
            raise RuntimeError("A parallel coordinator already holds this study")
        prior = self.output / "queue_state.json"
        if prior.exists():
            previous = _read(prior)
            if previous.get("requires_review") or previous.get("active_workers", 0):
                self.lock_stream.close()
                raise ResourceReviewRequired("Previously stopped or interrupted study requires review; automatic retry disabled")
        self._state("CPU_monitor_ready")
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.active:
            self.stop_owned_workers("coordinator_exit")
        if self.lock_stream:
            self.lock_stream.close()
            self.lock_stream = None
        if exc is not None:
            self._state("requires_review" if self.review_required else "stopped",
                        reason=str(exc), monitor_live=False)
        else:
            self._state("phase_complete", monitor_live=False)

    def _state(self, state, **details):
        scientific = _read(self.study / "state.json") if (self.study / "state.json").exists() else {}
        value = dict(schema=SCHEMA, state=state, monitor_pid=os.getpid(), monitor_live=True,
            updated_unix=self.clock(), phase=self.phase, candidate_gpus=list(CANDIDATES),
            reserved_gpus=[], max_cuda_workers=8, active_workers=len(self.active),
            workers=[dict(worker_id=record["worker_id"], physical_gpu=record["gpu"],
                          pid=record["identity"]["pid"]) for record in self.active.values()],
            completed_workers=self.completed, cuda_loaded_by_scheduler=False,
            resource_losses=self.resource_losses,
            requires_review=self.review_required, automatic_training_launch_disabled=self.review_required,
            formal_validation_opened=scientific.get("formal_validation_opened") is True,
            formal_test_opened=scientific.get("formal_test_opened") is True)
        value.update(details)
        _write(self.output / "queue_state.json", value)
        return value

    def _observe(self):
        owned = {value["identity"]["pid"]: value["identity"] for value in self.active.values()}
        rows = self.inventory()
        decisions, blockers = assess_cards(rows, self.protocol, owned,
            identity_reader=self.identity_reader, birth_reader=self.birth_reader)
        self._state("parallel_workers_running" if self.active else "CPU_wait_for_empty_candidates",
                    gpu_candidates=decisions, gpu6_pipeline_births=blockers)
        return decisions

    def _verify_command(self, command, worker_id):
        command = [str(value) for value in command]
        if (len(command) < 2 or Path(command[0]).absolute() != (ROOT / ".venv-hanguard/bin/python").absolute()
                or "--study" not in command or "--worker-id" not in command):
            raise ValueError("Worker requires registered runtime, exact study and worker ID")
        try:
            if (Path(command[command.index("--study") + 1]).resolve() != self.study or
                    command[command.index("--worker-id") + 1] != worker_id):
                raise ValueError("Worker command belongs to another study or worker")
        except IndexError as exc:
            raise ValueError("Incomplete worker command") from exc
        script = Path(command[1]).resolve()
        if not script.is_file() or str(script) not in self.protocol["bindings"]:
            raise ValueError("Worker script is not registered")
        if _sha(script) != self.protocol["bindings"][str(script)]:
            raise ValueError("Worker script changed")
        return command, script

    def _launch(self, gpu, command_factory, split, on_resource_loss=None):
        self.sequence += 1
        worker_id = f"{self.phase}_{self.sequence:04d}_gpu{gpu}"
        command, script = self._verify_command(command_factory(worker_id, gpu), worker_id)
        _check_stop(self.study)
        check_protocol(self.study)
        # Refresh immediately before creating a child. No minimum idle delay.
        decision = next(value for value in self._observe() if value["index"] == gpu)
        if not decision["eligible"] or decision.get("pids"):
            return False
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        logfile = (self.output / "workers" / f"{worker_id}.log").open("x")
        child = self.popen(command, cwd=str(ROOT), env=environment, stdout=logfile,
                           stderr=subprocess.STDOUT)
        logfile.close()
        identity = None
        for _ in range(20):
            if child.poll() is not None:
                break
            try:
                current = self.identity_reader(child.pid)
                if (current["uid"] == os.getuid() and current["argv"] == command and
                        current["cwd"] == str(ROOT)):
                    identity = dict(current, start_ticks=current["start_time_ticks"])
                    break
            except (OSError, ValueError, KeyError, TypeError):
                pass
            self.sleep(0.05)
        if identity is None:
            # With no ledger, a conforming child times out before CUDA. A PID
            # without complete verified identity is never signalled.
            self.review_required = True
            self._state("requires_review", reason="Child identity could not be verified",
                        unverified_child_pid=child.pid)
            raise ResourceReviewRequired("Child identity capture failed before CUDA handshake")
        record = dict(child=child, identity=identity, gpu=gpu, worker_id=worker_id,
                      command=command, split=split)
        self.active[worker_id] = record
        # The child is CPU-only and waiting for its ledger during this check.
        decision = next(value for value in self._observe() if value["index"] == gpu)
        if decision["blocking_processes"] or decision["pipeline_blocked"]:
            if (on_resource_loss is not None and decision["blocking_processes"] and
                    not decision["pipeline_blocked"] and
                    all(value["classification"] == "foreign"
                        for value in decision["blocking_processes"])):
                evidence = self._stop_records((record,), "GPU changed before CUDA handshake")[0]
                if not evidence["verified_original_child_exited"]:
                    self._resource_failure("Pre-handshake worker termination could not be verified", evidence)
                value = dict(worker_id=worker_id, physical_gpu=gpu, split=split,
                    reason="GPU changed before CUDA handshake", decision=decision,
                    termination=evidence, CUDA_handshake_published=False,
                    no_feature_claim_could_be_acquired=True, updated_unix=self.clock())
                self.resource_losses.append(value)
                _write(self.output / "workers" / f"{worker_id}.resource_loss.json", value,
                       exclusive=True)
                return False
            self._resource_failure("GPU ownership changed before worker handshake", decision)
        ledger = dict(schema_version=1, worker_id=worker_id, phase=self.phase, split=split,
            physical_gpu=gpu, CUDA_VISIBLE_DEVICES=str(gpu), process_identity=identity,
            command=command, root_protocol_sha256=self.protocol["root_protocol_sha256"],
            script_sha256=_sha(script), launched_at=self.clock(),
            resource_protocol_sha256=_sha(self.study / "parallel_resource_protocol.json"))
        path = self.output / "workers" / f"{worker_id}.json"
        _write(path, ledger, exclusive=True)
        _write(path.with_suffix(".registration.json"), dict(ledger_sha256=_sha(path)), exclusive=True)
        record["ledger_path"], record["ledger_sha256"] = path, _sha(path)
        return True

    def _signal_exact(self, record, number):
        if record["child"].poll() is not None:
            return False
        try:
            current = self.identity_reader(record["identity"]["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not same_identity(current, record["identity"]) or current.get("uid") != os.getuid():
            return False
        try:
            os.kill(current["pid"], number)
        except ProcessLookupError:
            return False
        return True

    def _stop_records(self, records, reason):
        evidence = []
        records = list(records)
        for record in records:
            stop = self.output / "workers" / f"{record['worker_id']}.STOP"
            if not stop.exists():
                stop.write_text(reason + "\n")
            sent = self._signal_exact(record, signal.SIGTERM)
            evidence.append(dict(worker_id=record["worker_id"], pid=record["identity"]["pid"],
                                 verified_SIGTERM_sent=sent))
        deadline = self.clock() + self.terminate_grace_seconds
        while self.clock() < deadline and any(value["child"].poll() is None for value in records):
            self.sleep(min(0.1, max(0, deadline - self.clock())))
        for record, entry in zip(records, evidence):
            entry["verified_SIGKILL_sent"] = self._signal_exact(record, signal.SIGKILL)
        # Reap only the recorded subprocesses, never infer termination merely
        # because a new NVIDIA inventory no longer lists a CUDA context.
        deadline = self.clock() + self.terminate_grace_seconds
        while self.clock() < deadline and any(value["child"].poll() is None for value in records):
            self.sleep(min(0.1, max(0, deadline - self.clock())))
        for record, entry in zip(records, evidence):
            entry["exit_code"] = record["child"].poll()
            entry["verified_original_child_exited"] = entry["exit_code"] is not None
            if entry["verified_original_child_exited"]:
                self.active.pop(record["worker_id"], None)
        return evidence

    def stop_owned_workers(self, reason):
        evidence = self._stop_records(self.active.values(), reason)
        _write(self.output / "termination_evidence.json", dict(reason=reason, workers=evidence,
            never_signalled_foreign_processes=True, updated_unix=self.clock()))

    def _feature_resource_loss(self, record, decision, callback):
        reason = "Confirmed foreign process entered a feature GPU"
        evidence = self._stop_records((record,), reason)[0]
        if not evidence["verified_original_child_exited"]:
            self._resource_failure("Lost feature worker termination could not be verified", evidence)
        value = dict(worker_id=record["worker_id"], physical_gpu=record["gpu"],
            split=record["split"], reason=reason, decision=decision,
            termination=evidence, updated_unix=self.clock())
        self.resource_losses.append(value)
        _write(self.output / "workers" / f"{record['worker_id']}.resource_loss.json", value,
               exclusive=True)
        # The callback must verify committed receipts and retire uncommitted
        # files. A callback failure is scientific failure, not a retry signal.
        callback(record["worker_id"], record["gpu"], record["split"], reason)
        self._state("feature_resource_redispatch", last_resource_loss=value)

    def _resource_failure(self, reason, evidence):
        self.review_required = True
        self._state("requires_review", reason=reason, trigger_evidence=evidence)
        self.stop_owned_workers(reason)
        raise ResourceReviewRequired(reason)

    def _record_finished(self, worker_id, record, code):
        entry = dict(worker_id=worker_id, pid=record["identity"]["pid"],
                     physical_gpu=record["gpu"], exit_code=code, phase=self.phase,
                     finished_unix=self.clock())
        self.completed.append(entry)
        del self.active[worker_id]
        if code != 0:
            self._resource_failure("Scientific worker exited unsuccessfully", entry)

    def _poll_workers(self):
        for worker_id, record in list(self.active.items()):
            code = record["child"].poll()
            if code is None:
                try:
                    current = self.identity_reader(record["identity"]["pid"])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if record["child"].poll() is not None:
                        # A normal exit may race the /proc command read. Reap
                        # it immediately rather than call it identity drift.
                        self._record_finished(worker_id, record, record["child"].poll())
                        continue
                    self._resource_failure("Own worker identity became unverifiable", dict(
                        worker_id=worker_id, error_type=type(exc).__name__))
                if not same_identity(current, record["identity"]):
                    if record["child"].poll() is not None:
                        self._record_finished(worker_id, record, record["child"].poll())
                        continue
                    self._resource_failure("Own worker process identity changed", dict(worker_id=worker_id))
                path = record.get("ledger_path")
                if path is not None and _sha(path) != record["ledger_sha256"]:
                    self._resource_failure("Worker ownership ledger changed", dict(worker_id=worker_id))
                continue
            self._record_finished(worker_id, record, code)

    def run_workers(self, phase, command_factory, phase_complete, *, work_remaining=None,
                    split=None, max_workers=8, poll_seconds=2, on_resource_loss=None):
        if type(max_workers) is not int or not 1 <= max_workers <= 8:
            raise ValueError("Worker limit must be 1–8; the registered ceiling is 8")
        if not 0 < poll_seconds <= 5:
            raise ValueError("Polling must be responsive, at most five seconds")
        if not phase or any(value not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for value in phase):
            raise ValueError("Phase must be a safe worker identifier")
        if self.active:
            raise RuntimeError("Prior phase workers remain active")
        if on_resource_loss is not None and split not in ("train", "validation", "test"):
            raise ValueError("Localized resource redispatch applies only to feature splits")
        self.phase = phase
        launched = 0
        try:
            while True:
                _check_stop(self.study)
                check_protocol(self.study)
                self._poll_workers()
                done = bool(phase_complete())
                decisions = self._observe()
                for decision in decisions:
                    record = next((value for value in self.active.values()
                                   if value["gpu"] == decision["index"]), None)
                    if record is not None:
                        if decision["blocking_processes"] or decision["pipeline_blocked"]:
                            confirmed_foreign = (bool(decision["blocking_processes"]) and
                                all(value["classification"] == "foreign"
                                    for value in decision["blocking_processes"]) and
                                not decision["pipeline_blocked"])
                            if on_resource_loss is not None and confirmed_foreign:
                                self._feature_resource_loss(record, decision, on_resource_loss)
                            else:
                                self._resource_failure("Foreign process entered an active GPU", decision)
                if done and not self.active:
                    return self._state("phase_complete", phase_worker_launches=launched)
                has_work = not done and (work_remaining is None or bool(work_remaining()))
                if has_work:
                    for decision in decisions:
                        if len(self.active) >= max_workers:
                            break
                        if work_remaining is not None and not work_remaining():
                            break
                        if (not decision["eligible"] or decision.get("pids") or
                                any(value["gpu"] == decision["index"] for value in self.active.values())):
                            continue
                        if self._launch(decision["index"], command_factory, split, on_resource_loss):
                            launched += 1
                if not self.active and not has_work and not done:
                    self._resource_failure("No active workers or pending work but phase incomplete", {})
                self.sleep(poll_seconds)
        except BaseException as exc:
            if not isinstance(exc, StopRequested):
                self.review_required = True
            if self.active:
                self.stop_owned_workers(str(exc))
            raise

    def run_single(self, phase, command_factory):
        completed_before = len(self.completed)
        return self.run_workers(phase, command_factory,
            lambda: len(self.completed) > completed_before, max_workers=1)


def run_workers(study, phase, command_factory, phase_complete, **kwargs):
    with ParallelScheduler(study) as scheduler:
        return scheduler.run_workers(phase, command_factory, phase_complete, **kwargs)


def run_single(study, phase, command_factory):
    with ParallelScheduler(study) as scheduler:
        return scheduler.run_single(phase, command_factory)
