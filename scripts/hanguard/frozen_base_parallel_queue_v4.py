"""One reviewed same-seed restart after a proven foreign-GPU interruption.

All scientific GPU workers and V3 scheduling logic remain unchanged. Only the
independent control and reviewed-entry checks differ. The interrupted output is
immutable, no partial state is adopted, and later head interruptions need a new
review rather than an automatic retry.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import time

from scripts.hanguard import frozen_base_parallel_queue_v3 as prior

original = prior.original
SCHEMA = "hanguard_frozen_base_reviewed_head_restart_control_v1"
POLICY = prior.POLICY
COORDINATOR_PATH = original.ROOT / "scripts/hanguard/frozen_base_reviewed_head_restart.py"
initialize = original.initialize
ResourceReviewRequired = original.ResourceReviewRequired
StopRequested = original.StopRequested


def check_control3(study, recovery_protocol_path=None):
    study = Path(study).resolve()
    folder = study / "recovery_003"
    path = Path(recovery_protocol_path or folder / "control_protocol.json").resolve()
    if path != folder / "control_protocol.json":
        raise ValueError("Reviewed head-restart control must belong to this exact study")
    control2 = prior.check_control2(study)
    if original._read(folder / "control_registration.json").get("protocol_sha256") != original._sha(path):
        raise ValueError("Reviewed head-restart registration changed")
    control = original._read(path)
    scientific = original._read(study / "protocol.json")
    if (control.get("schema") != SCHEMA or control.get("study") != str(study)
            or control.get("root_protocol_sha256") != original._sha(study / "protocol.json")
            or control.get("resource_protocol_sha256") != original._sha(study / "parallel_resource_protocol.json")
            or control.get("prior_control_sha256") != original._sha(study / "recovery_002/control_protocol.json")
            or control.get("unchanged_scientific_settings") != scientific.get("settings")
            or control.get("automatic_head_retry") is not False
            or control.get("actual_gpu_commands_use_original_registered_entrypoints") is not True
            or control.get("policy") != POLICY):
        raise ValueError("Reviewed head-restart scientific binding or resource policy changed")
    bindings = control.get("bindings", {})
    if any(bindings.get(name) != digest for name, digest in control2["bindings"].items()):
        raise ValueError("Reviewed restart must preserve every prior source binding")
    for required in (Path(__file__).resolve(), COORDINATOR_PATH.resolve(),
            study / "recovery_002/control_protocol.json", study / "recovery_002/control_registration.json"):
        if bindings.get(str(required)) != original._sha(required):
            raise ValueError(f"Reviewed head-restart implementation is not registered: {required}")
    for name, digest in bindings.items():
        if original._sha(name) != digest:
            raise ValueError(f"Reviewed head-restart source changed: {name}")
    review = control.get("prior_review", {})
    snapshot = folder / "queue_state.before.json"
    archive_manifest = study / "fit_attempts/001/archive_manifest.json"
    if (review.get("queue_state_snapshot") != str(snapshot)
            or review.get("queue_state_sha256") != original._sha(snapshot)
            or review.get("prior_workers_verified_exited") is not True
            or review.get("resume_authorized") is not True
            or review.get("reviewed_same_seed_restart_by_root") is not True
            or review.get("formal_test_opened") is not False
            or review.get("selection_lock_absent") is not True
            or review.get("partial_checkpoint_reuse") is not False
            or review.get("automatic_head_retry") is not False
            or type(review.get("restart_seed")) is not int or review["restart_seed"] != 42
            or review.get("restart_seed") != scientific.get("settings", {}).get("seed")
            or type(review.get("archived_fit_attempt")) is not int or review["archived_fit_attempt"] != 1
            or type(review.get("new_fit_attempt")) is not int or review["new_fit_attempt"] != 2
            or review.get("archive_manifest") != str(archive_manifest)
            or review.get("archive_manifest_sha256") != original._sha(archive_manifest)
            or bindings.get(str(archive_manifest)) != original._sha(archive_manifest)
            or review.get("reason") != "confirmed_foreign_gpu_interruption_during_fit"):
        raise ValueError("Head restart lacks an exact reviewed same-seed archive")
    archive = study / "fit_attempts/001/heads"
    paths = list(archive.rglob("*"))
    if (not archive.is_dir() or archive.is_symlink() or any(p.is_symlink() for p in paths)
            or {str(p.relative_to(archive)) for p in paths if p.is_file()} != {"fit_started.json"}
            or {str(p.relative_to(archive)) for p in paths if p.is_dir()} != {"budget_12"}):
        raise ValueError("Interrupted archive must have zero completed epochs or checkpoints")
    started = archive / "fit_started.json"
    marker = original._read(started)
    if (bindings.get(str(started)) != original._sha(started)
            or marker.get("protocol_sha256") != control["root_protocol_sha256"]
            or marker.get("test_opened") is not False
            or not isinstance(marker.get("source_sha256"), dict) or not marker["source_sha256"]
            or any(bindings.get(name) != digest for name, digest in marker.get("source_sha256", {}).items())):
        raise ValueError("Interrupted fit marker does not bind the unchanged scientific implementation")
    _validate_foreign_stop(study, control)
    return control


def _validate_foreign_stop(study, control):
    bindings = control["bindings"]
    termination_path = study / "recovery_003/termination_evidence.before.json"
    if bindings.get(str(termination_path)) != original._sha(termination_path):
        raise ValueError("Reviewed foreign stop lacks pinned termination evidence")
    ended = original._read(termination_path).get("workers", [])
    for filename in bindings:
        path = Path(filename)
        if path.parent != study / "recovery_002/failure_events" or path.suffix != ".json":
            continue
        event = original._read(path)
        owned, trigger = event.get("active_workers", []), event.get("trigger_evidence", {})
        if (event.get("phase") != "fit" or event.get("reason") != "Foreign process entered an active GPU"
                or event.get("recovery_control_sha256") != control["prior_control_sha256"] or len(owned) != 1
                or not any(value.get("classification") == "foreign" and
                    type(value.get("uid")) is int and value["uid"] != os.getuid() and
                    type(value.get("start_time_ticks")) is int
                    for value in trigger.get("blocking_processes", []))):
            continue
        worker_id = owned[0].get("worker_id")
        if not worker_id or any(value not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for value in worker_id):
            continue
        ledger = study / "parallel_execution/workers" / f"{worker_id}.json"
        registration, logfile = ledger.with_suffix(".registration.json"), ledger.with_suffix(".log")
        if any(bindings.get(str(p)) != original._sha(p) for p in (ledger, registration, logfile)):
            raise ValueError("Reviewed foreign stop worker evidence is not registered")
        record = original._read(ledger)
        if (original._read(registration).get("ledger_sha256") != original._sha(ledger)
                or record.get("process_identity") != owned[0].get("process_identity")
                or record.get("root_protocol_sha256") != control["root_protocol_sha256"]
                or record.get("resource_protocol_sha256") != control["resource_protocol_sha256"]
                or record.get("phase") != "fit" or logfile.stat().st_size != 0
                or trigger.get("index") != owned[0].get("physical_gpu")):
            raise ValueError("Reviewed foreign stop differs from exact fit-worker provenance")
        if any(value.get("worker_id") == worker_id and value.get("pid") == record["process_identity"]["pid"]
                and value.get("exit_code") == -15 and value.get("verified_SIGTERM_sent") is True
                and value.get("verified_SIGKILL_sent") is False
                and value.get("verified_original_child_exited") is True for value in ended):
            return
    raise ValueError("Reviewed head restart lacks a pinned genuine foreign-GPU interruption")


def _enter_reviewed(scheduler, folder, control, control_path):
    """Keep reviewed admission small while leaving inherited GPU scheduling intact."""
    scheduler.lock_stream = (scheduler.output / "scheduler.lock").open("a+")
    try:
        fcntl.flock(scheduler.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler._check_control()
        scheduler._verify_previous_workers_exited()
        previous_path = scheduler.output / "queue_state.json"
        entry_path = folder / "scheduler_recovery_entry.json"
        if not entry_path.exists():
            review = control["prior_review"]
            if original._sha(previous_path) != review["queue_state_sha256"]:
                raise ResourceReviewRequired("Head-restart queue differs from the reviewed snapshot")
            previous = original._read(previous_path)
            if previous.get("requires_review") is not True or previous.get("phase") != "fit":
                raise ResourceReviewRequired("Reviewed restart does not identify the interrupted fit")
            if (scheduler.study / "heads").exists() or (scheduler.study / "selection_lock.json").exists():
                raise ResourceReviewRequired("A fresh reviewed restart cannot adopt canonical partial heads")
            termination = scheduler.output / "termination_evidence.json"
            preserved = folder / "termination_evidence.before.json"
            if termination.exists():
                if preserved.exists():
                    if original._sha(preserved) != original._sha(termination):
                        raise ValueError("Preserved head termination evidence differs")
                else:
                    preserved.write_bytes(termination.read_bytes())
            original._write(entry_path, dict(recovery_control_sha256=original._sha(control_path),
                reviewed_queue_state_sha256=review["queue_state_sha256"], prior_workers_verified_exited=True,
                coordinator_birth=original.admission.process_birth(os.getpid()),
                entered_unix=scheduler.clock()), exclusive=True)
            original._write(entry_path.with_suffix(".registration.json"), dict(
                entry_sha256=original._sha(entry_path)), exclusive=True)
        else:
            entry = original._read(entry_path)
            if (original._read(entry_path.with_suffix(".registration.json")).get("entry_sha256")
                    != original._sha(entry_path) or entry.get("recovery_control_sha256") != original._sha(control_path)):
                raise ValueError("Reviewed head-restart entry binding changed")
            birth = original.admission.process_birth(os.getpid())
            if birth is None or any(entry.get("coordinator_birth", {}).get(k) != birth.get(k)
                    for k in ("pid", "uid", "start_time_ticks")):
                raise ResourceReviewRequired("Head-restart coordinator birth changed; a new review is required")
            if previous_path.exists():
                previous = original._read(previous_path)
                if previous.get("requires_review") or previous.get("active_workers", 0):
                    raise ResourceReviewRequired("Reviewed fit interrupted again; automatic head retry remains disabled")
        scheduler._state("CPU_reviewed_head_restart_ready", recovery_control_sha256=original._sha(control_path))
        return scheduler
    except BaseException:
        scheduler.lock_stream.close()
        scheduler.lock_stream = None
        raise


class ParallelScheduler(prior.ParallelScheduler):
    def __init__(self, study, *, recovery_protocol_path=None, **kwargs):
        self.control3_path = recovery_protocol_path
        self.control3 = check_control3(study, recovery_protocol_path)
        super().__init__(study, **kwargs)

    def _check_control(self):
        super()._check_control()
        if check_control3(self.study, self.control3_path) != self.control3:
            raise ValueError("Reviewed head-restart control changed after construction")

    def __enter__(self):
        return _enter_reviewed(self, self.study / "recovery_003", self.control3,
                              self.study / "recovery_003/control_protocol.json")

    def _state(self, state, **details):
        details.update(head_fit_attempt=2, current_fit_started=(self.study / "heads/fit_started.json").exists())
        return super()._state(state, **details)

    def _resource_failure(self, reason, evidence):
        path = self.study / "recovery_003/failure_events" / (
            f"failure_event_{time.time_ns()}_{os.getpid()}.json")
        control_path = self.study / "recovery_003/control_protocol.json"
        original._write(path, dict(reason=reason, trigger_evidence=evidence, phase=self.phase,
            recorded_unix=self.clock(), monitor_pid=os.getpid(), head_fit_attempt=2,
            recovery_control_sha256=original._sha(control_path) if control_path.exists() else None,
            active_workers=[dict(worker_id=worker_id, physical_gpu=record["gpu"],
                process_identity=record["identity"]) for worker_id, record in self.active.items()]), exclusive=True)
        return original.ParallelScheduler._resource_failure(self, reason, evidence)


def run_workers(study, phase, command_factory, phase_complete, *, recovery_protocol_path=None, **kwargs):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_workers(phase, command_factory, phase_complete, **kwargs)


def run_single(study, phase, command_factory, *, recovery_protocol_path=None):
    with ParallelScheduler(study, recovery_protocol_path=recovery_protocol_path) as scheduler:
        return scheduler.run_single(phase, command_factory)
