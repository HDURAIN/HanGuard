"""CPU recovery after completed validation, preserving all registered science.

Inventory may outlive a CUDA worker. Reap exact recorded Popen children after
inventory assessment and re-evaluate phase completion before admitting work.
Live foreign or unverifiable processes still block or stop GPU work.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import time

from scripts.hanguard import frozen_base_parallel_queue_v2 as prior

original = prior.original
SCHEMA = "hanguard_frozen_base_post_validation_recovery_control_v1"
POLICY = prior.POLICY
COORDINATOR_PATH = original.ROOT / "scripts/hanguard/frozen_base_post_validation_recovery.py"
initialize = original.initialize
ResourceReviewRequired = original.ResourceReviewRequired
StopRequested = original.StopRequested


def check_control2(study, recovery_protocol_path=None):
    study = Path(study).resolve()
    path = Path(recovery_protocol_path or study / "recovery_002/control_protocol.json").resolve()
    if path != study / "recovery_002/control_protocol.json":
        raise ValueError("Post-validation recovery control must belong to this exact study")
    control1 = prior.check_control(study)
    registration = original._read(study / "recovery_002/control_registration.json")
    if registration.get("protocol_sha256") != original._sha(path):
        raise ValueError("Post-validation recovery registration changed")
    control = original._read(path)
    if (control.get("schema") != SCHEMA or control.get("study") != str(study)
            or control.get("root_protocol_sha256") != original._sha(study / "protocol.json")
            or control.get("resource_protocol_sha256") != original._sha(study / "parallel_resource_protocol.json")
            or control.get("prior_control_sha256") != original._sha(study / "recovery_control_protocol.json")
            or control.get("policy") != POLICY):
        raise ValueError("Post-validation scientific binding or resource policy changed")
    bindings = control.get("bindings", {})
    if any(bindings.get(name) != digest for name, digest in control1["bindings"].items()):
        raise ValueError("Post-validation recovery must preserve every prior source binding")
    for path in (Path(__file__).resolve(), COORDINATOR_PATH.resolve(),
                 study / "recovery_control_protocol.json", study / "recovery_control_registration.json"):
        if bindings.get(str(path)) != original._sha(path):
            raise ValueError(f"Post-validation implementation is not registered: {path}")
    for name, digest in bindings.items():
        if original._sha(name) != digest:
            raise ValueError(f"Post-validation recovery source changed: {name}")
    review = control.get("prior_review", {})
    snapshot = study / "recovery_002/queue_state.before.json"
    if (review.get("queue_state_snapshot") != str(snapshot)
            or review.get("queue_state_sha256") != original._sha(snapshot)
            or review.get("prior_workers_verified_exited") is not True
            or review.get("resume_authorized") is not True
            or review.get("all_validation_complete") is not True
            or review.get("head_training_started") is not False
            or review.get("formal_test_opened") is not False
            or review.get("reason") != "completed_validation_resource_stop"):
        raise ValueError("Post-validation recovery lacks an immutable reviewed stop")
    return control


class ParallelScheduler(prior.ParallelScheduler):
    def __init__(self, study, *, recovery_protocol_path=None, **kwargs):
        self.control2_path = recovery_protocol_path
        self.control2 = check_control2(study, recovery_protocol_path)
        super().__init__(study, **kwargs)

    def _check_control(self):
        super()._check_control()
        if check_control2(self.study, self.control2_path) != self.control2:
            raise ValueError("Post-validation recovery control changed after construction")

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_control()
            self._verify_previous_workers_exited()
            previous_path = self.output / "queue_state.json"
            entry_path = self.study / "recovery_002/scheduler_recovery_entry.json"
            if not entry_path.exists():
                review = self.control2["prior_review"]
                if original._sha(previous_path) != review["queue_state_sha256"]:
                    raise ResourceReviewRequired("Post-validation queue stop differs from reviewed snapshot")
                if original._read(previous_path).get("requires_review") is not True:
                    raise ResourceReviewRequired("Post-validation recovery does not identify a stopped queue")
                termination = self.output / "termination_evidence.json"
                preserved = self.study / "recovery_002/termination_evidence.before.json"
                if termination.exists():
                    if preserved.exists():
                        if original._sha(preserved) != original._sha(termination):
                            raise ValueError("Preserved post-validation termination evidence differs")
                    else:
                        preserved.write_bytes(termination.read_bytes())
                original._write(entry_path, dict(
                    recovery_control_sha256=original._sha(self.study / "recovery_002/control_protocol.json"),
                    reviewed_queue_state_sha256=review["queue_state_sha256"],
                    prior_workers_verified_exited=True, entered_unix=self.clock(),
                    coordinator_birth=original.admission.process_birth(os.getpid())), exclusive=True)
                original._write(entry_path.with_suffix(".registration.json"), dict(
                    entry_sha256=original._sha(entry_path)), exclusive=True)
            else:
                entry = original._read(entry_path)
                if (original._read(entry_path.with_suffix(".registration.json")).get("entry_sha256")
                        != original._sha(entry_path) or entry.get("recovery_control_sha256")
                        != original._sha(self.study / "recovery_002/control_protocol.json")):
                    raise ValueError("Post-validation recovery entry binding changed")
                birth = original.admission.process_birth(os.getpid())
                if birth is None or any(entry.get("coordinator_birth", {}).get(key) != birth.get(key)
                        for key in ("pid", "uid", "start_time_ticks")):
                    raise ResourceReviewRequired("Post-validation coordinator birth changed; a new review is required")
                if previous_path.exists():
                    previous = original._read(previous_path)
                    if previous.get("requires_review") or previous.get("active_workers", 0):
                        raise ResourceReviewRequired("Post-validation recovery interrupted again; a new review is required")
            self._state("CPU_post_validation_recovery_ready", recovery_control_sha256=original._sha(
                self.study / "recovery_002/control_protocol.json"))
            return self
        except BaseException:
            self.lock_stream.close()
            self.lock_stream = None
            raise

    def _observe(self):
        decisions = super()._observe()
        # Only a known Popen exit retires a worker. CUDA inventory disappearance
        # and unverified process identity are never sufficient by themselves.
        self._poll_workers()
        self._state("parallel_workers_running" if self.active else "CPU_wait_for_empty_candidates",
                    gpu_candidates=decisions)
        return decisions

    def _resource_failure(self, reason, evidence):
        # Queue state is intentionally replaced at exit; retain the exact
        # trigger separately before any stop/reap can change active workers.
        path = self.study / "recovery_002/failure_events" / (
            f"failure_event_{time.time_ns()}_{os.getpid()}.json")
        control_path = self.study / "recovery_002/control_protocol.json"
        original._write(path, dict(reason=reason, trigger_evidence=evidence,
            phase=self.phase, recorded_unix=self.clock(), monitor_pid=os.getpid(),
            recovery_control_sha256=original._sha(control_path) if control_path.exists() else None,
            active_workers=[dict(worker_id=worker_id, physical_gpu=record["gpu"],
                process_identity=record["identity"]) for worker_id, record in self.active.items()]),
            exclusive=True)
        return super()._resource_failure(reason, evidence)

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
                original._check_stop(self.study)
                original.check_protocol(self.study)
                self._poll_workers()
                decisions = self._observe()
                # _observe can reap a worker after its inventory was captured.
                # Recompute before admitting any work, including single fit.
                done = bool(phase_complete())
                for decision in decisions:
                    record = next((value for value in self.active.values()
                                   if value["gpu"] == decision["index"]), None)
                    if record is not None and (decision["blocking_processes"] or decision["pipeline_blocked"]):
                        confirmed_foreign = (bool(decision["blocking_processes"]) and
                            all(value["classification"] == "foreign" for value in decision["blocking_processes"]) and
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


def run_workers(study, phase, command_factory, phase_complete, *, recovery_protocol_path=None, **kwargs):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_workers(phase, command_factory, phase_complete, **kwargs)


def run_single(study, phase, command_factory, *, recovery_protocol_path=None):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_single(phase, command_factory)
