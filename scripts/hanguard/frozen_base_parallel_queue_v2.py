"""Versioned CPU recovery scheduler around unchanged registered GPU workers.

The only runtime change is a bounded reap of the exact Popen child when /proc
identity reads race its exit. A still-live unverifiable child fails closed.
Recovery requires an independently registered control and an immutable review
of the prior stop; the original scientific/resource protocols are not edited.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from scripts.hanguard import frozen_base_parallel_queue as original

SCHEMA = "hanguard_frozen_base_recovery_control_v1"
POLICY = dict(max_cuda_workers=8, reserved_gpus=[],
              never_terminate_foreign_processes=True, preserve_registered_science=True)
COORDINATOR_PATH = original.ROOT / "scripts/hanguard/frozen_base_recovery_coordinator.py"
initialize = original.initialize
ResourceReviewRequired = original.ResourceReviewRequired
StopRequested = original.StopRequested


def check_control(study, recovery_protocol_path=None):
    study = Path(study).resolve()
    path = Path(recovery_protocol_path or study / "recovery_control_protocol.json").resolve()
    if path != study / "recovery_control_protocol.json":
        raise ValueError("Recovery control must belong to this exact study")
    original._check_stop(study)
    registration = original._read(study / "recovery_control_registration.json")
    if registration.get("protocol_sha256") != original._sha(path):
        raise ValueError("Recovery control registration changed")
    control = original._read(path)
    resource = original.check_protocol(study)
    if (control.get("schema") != SCHEMA or control.get("study") != str(study)
            or control.get("root_protocol_sha256") != original._sha(study / "protocol.json")
            or control.get("resource_protocol_sha256") != original._sha(study / "parallel_resource_protocol.json")
            or control.get("policy") != POLICY):
        raise ValueError("Recovery scientific binding or resource policy changed")
    bindings = control.get("bindings", {})
    if any(bindings.get(name) != digest for name, digest in resource["bindings"].items()):
        raise ValueError("Recovery must preserve every original source binding")
    if bindings.get(str(Path(__file__).resolve())) != original._sha(__file__):
        raise ValueError("Versioned recovery scheduler is not registered")
    if bindings.get(str(COORDINATOR_PATH.resolve())) != original._sha(COORDINATOR_PATH):
        raise ValueError("Versioned CPU recovery coordinator is not registered")
    for name, digest in bindings.items():
        if original._sha(name) != digest:
            raise ValueError(f"Recovery source changed: {name}")
    review = control.get("prior_review", {})
    snapshot = study / "recovery_001" / "queue_state.before.json"
    if (review.get("queue_state_snapshot") != str(snapshot)
            or review.get("queue_state_sha256") != original._sha(snapshot)
            or review.get("prior_workers_verified_exited") is not True
            or review.get("resume_authorized") is not True
            or review.get("reason") != "normal_exit_identity_read_race"):
        raise ValueError("Recovery has no exact immutable prior stop review")
    return control


class ParallelScheduler(original.ParallelScheduler):
    def __init__(self, study, *, recovery_protocol_path=None, **kwargs):
        self.recovery_protocol_path = recovery_protocol_path
        self.control = check_control(study, recovery_protocol_path)
        super().__init__(study, **kwargs)

    def _check_control(self):
        control = check_control(self.study, self.recovery_protocol_path)
        if control != self.control:
            raise ValueError("Recovery control changed after scheduler construction")

    def _verify_previous_workers_exited(self):
        for path in sorted((self.output / "workers").glob("*.json")):
            ledger = original._read(path)
            expected = ledger.get("process_identity")
            if not isinstance(expected, dict) or ledger.get("schema_version") != 1:
                continue
            try:
                birth = self.birth_reader(expected["pid"])
            except Exception as exc:
                raise ResourceReviewRequired("Prior worker birth is unverifiable") from exc
            if birth is None:
                continue
            if (birth.get("pid") != expected.get("pid") or type(birth.get("uid")) is not int
                    or type(birth.get("start_time_ticks")) is not int):
                raise ResourceReviewRequired("Prior worker birth fields are unverifiable")
            if birth["start_time_ticks"] != expected.get("start_time_ticks"):
                # The recorded child is gone and its PID was reused; never signal it.
                continue
            if birth["uid"] != expected.get("uid"):
                raise ResourceReviewRequired("Prior worker owner changed within the same birth")
            if birth.get("state") in ("Z", "X"):
                continue
            raise ResourceReviewRequired("A prior recorded worker is still live")

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_control()
            self._verify_previous_workers_exited()
            previous_path = self.output / "queue_state.json"
            entry_path = self.study / "recovery_001" / "scheduler_recovery_entry.json"
            if not entry_path.exists():
                review = self.control["prior_review"]
                if original._sha(previous_path) != review["queue_state_sha256"]:
                    raise ResourceReviewRequired("Prior queue stop differs from reviewed snapshot")
                previous = original._read(previous_path)
                if previous.get("requires_review") is not True:
                    raise ResourceReviewRequired("Recovery review does not identify the prior stopped queue")
                termination = self.output / "termination_evidence.json"
                preserved = self.study / "recovery_001" / "termination_evidence.before.json"
                if termination.exists():
                    if preserved.exists():
                        if original._sha(preserved) != original._sha(termination):
                            raise ValueError("Preserved termination evidence differs")
                    else:
                        preserved.write_bytes(termination.read_bytes())
                original._write(entry_path, dict(
                    recovery_control_sha256=original._sha(self.study / "recovery_control_protocol.json"),
                    reviewed_queue_state_sha256=review["queue_state_sha256"],
                    prior_workers_verified_exited=True, entered_unix=self.clock(),
                    coordinator_birth=original.admission.process_birth(os.getpid())), exclusive=True)
                original._write(entry_path.with_suffix(".registration.json"),
                    dict(entry_sha256=original._sha(entry_path)), exclusive=True)
            else:
                entry = original._read(entry_path)
                if (original._read(entry_path.with_suffix(".registration.json")).get("entry_sha256")
                        != original._sha(entry_path) or entry.get("recovery_control_sha256")
                        != original._sha(self.study / "recovery_control_protocol.json")):
                    raise ValueError("Recovery entry binding changed")
                birth = original.admission.process_birth(os.getpid())
                if birth is None or any(entry.get("coordinator_birth", {}).get(key) != birth.get(key)
                        for key in ("pid", "uid", "start_time_ticks")):
                    raise ResourceReviewRequired("Recovery coordinator birth changed; a new review is required")
                if previous_path.exists():
                    previous = original._read(previous_path)
                    if previous.get("requires_review") or previous.get("active_workers", 0):
                        raise ResourceReviewRequired("Recovery interrupted again; a new review is required")
            self._state("CPU_recovery_monitor_ready", recovery_control_sha256=original._sha(
                self.study / "recovery_control_protocol.json"))
            return self
        except BaseException:
            self.lock_stream.close()
            self.lock_stream = None
            raise

    def _observe(self):
        self._check_control()
        return super()._observe()

    def _launch(self, gpu, command_factory, split, on_resource_loss=None):
        self._check_control()
        return super()._launch(gpu, command_factory, split, on_resource_loss)

    def _bounded_original_exit(self, record):
        """Only Popen.poll() reaps the exact child; a PID disappearance is not exit."""
        code = record["child"].poll()
        for _ in range(4):
            if code is not None:
                break
            self.sleep(0.05)
            code = record["child"].poll()
        return code

    def _poll_workers(self):
        self._check_control()
        for worker_id, record in list(self.active.items()):
            code = record["child"].poll()
            if code is not None:
                self._record_finished(worker_id, record, code)
                continue
            try:
                current = self.identity_reader(record["identity"]["pid"])
            except Exception as exc:
                code = self._bounded_original_exit(record)
                if code is not None:
                    self._record_finished(worker_id, record, code)
                    continue
                self._resource_failure("Own worker identity became unverifiable", dict(
                    worker_id=worker_id, error_type=type(exc).__name__,
                    exact_original_child_still_live=True, bounded_exit_wait_seconds=0.2))
            if not original.same_identity(current, record["identity"]):
                code = self._bounded_original_exit(record)
                if code is not None:
                    self._record_finished(worker_id, record, code)
                    continue
                self._resource_failure("Own worker process identity changed", dict(
                    worker_id=worker_id, exact_original_child_still_live=True))
            # Exit can also happen immediately after a successful identity read.
            code = record["child"].poll()
            if code is not None:
                self._record_finished(worker_id, record, code)
                continue
            path = record.get("ledger_path")
            if path is not None and original._sha(path) != record["ledger_sha256"]:
                self._resource_failure("Worker ownership ledger changed", dict(worker_id=worker_id))


def run_workers(study, phase, command_factory, phase_complete, *, recovery_protocol_path=None, **kwargs):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_workers(phase, command_factory, phase_complete, **kwargs)


def run_single(study, phase, command_factory, *, recovery_protocol_path=None):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_single(phase, command_factory)
