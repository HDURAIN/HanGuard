"""CPU-only locked-test monitor with recoverable NVIDIA telemetry failures.

This append-only continuation changes no selected artifact or GPU entrypoint.
An unreadable inventory makes every card unavailable; admitted CUDA workers keep
running. Scientific, identity and control errors still require review.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_locked_test_resume as prior

cooperative, queue = prior.cooperative, prior.queue
read, sha, write = prior.read, prior.sha, prior.write
FOLDER = "recovery_006"
SCHEMA = "hanguard_locked_test_resilient_telemetry_v1"
SCOPE = prior.SCOPE
TELEMETRY_POLICY = dict(failure="all_cards_unavailable", retry="next_poll",
                        admitted_workers="continue", admission="verified_empty_only")


def control_path(study):
    return Path(study) / FOLDER / "control_protocol.json"


def verify_old_coordinator(study):
    folder = Path(study) / prior.FOLDER
    expected = read(folder / "coordinator_started.json")["birth"]
    recorded = read(folder / "scheduler_entry.json")["coordinator_birth"]
    if (any(expected.get(k) != recorded.get(k) for k in ("pid", "uid", "start_time_ticks"))
            or expected.get("uid") != os.getuid()):
        raise ValueError("Prior test coordinator birth/owner records differ")
    current = queue.admission.process_birth(expected["pid"])
    if (current is not None and current.get("start_time_ticks") == expected["start_time_ticks"]
            and (current.get("uid") != expected["uid"] or current.get("state") not in ("Z", "X"))):
        raise ValueError("Prior exact test coordinator is still alive or unverifiable")
    return expected


def check_control(study):
    study = Path(study).resolve()
    old = prior.check_control(study)
    path = control_path(study)
    value = read(path)
    bindings = value.get("bindings", {})
    required = [Path(__file__).resolve(), prior.control_path(study),
                study / prior.FOLDER / "control_registration.json",
                study / "protocol.json", study / "selection_lock.json",
                study / "parallel_features/test/prepared_manifest.json",
                study / "parallel_features/test/batch_plan.json"]
    if (read(path.with_name("control_registration.json")).get("protocol_sha256") != sha(path)
            or value.get("schema") != SCHEMA or value.get("study") != str(study)
            or value.get("scope") != SCOPE or value.get("policy") != cooperative.POLICY
            or value.get("telemetry_policy") != TELEMETRY_POLICY
            or value.get("unchanged_scientific_settings") != old["unchanged_scientific_settings"]
            or value.get("prior_control_sha256") != sha(prior.control_path(study))
            or value.get("selection_lock_sha256") != sha(study / "selection_lock.json")
            or any(str(p) not in bindings for p in required)):
        raise ValueError("Test monitor identity, scope or registration changed")
    for filename, digest in bindings.items():
        if sha(filename) != digest:
            raise ValueError(f"Test monitor evidence changed: {filename}")
    return value


def register(study):
    from scripts.hanguard import frozen_base_parallel_features as features

    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / FOLDER).exists():
            raise FileExistsError("Test monitor continuation is already registered")
        old = prior.check_control(study)
        state, stopped = read(study / "state.json"), read(study / "parallel_execution/queue_state.json")
        reason = state.get("reason", "")
        if (state.get("state") != "requires_review" or state.get("error_type") != "TimeoutExpired"
                or "nvidia-smi" not in reason or "timed out" not in reason
                or state.get("formal_test_opened") is not True
                or stopped.get("phase") != "extract_test" or stopped.get("active_workers") != 0
                or stopped.get("workers") or stopped.get("requires_review") is not True
                or stopped.get("reason") != reason
                or stopped.get("post_selection_recovery_control_sha256") != sha(prior.control_path(study))):
            raise ValueError("Expected reviewed NVIDIA timeout with no active test workers")
        old_birth = verify_old_coordinator(study)
        cooperative.verify_retired(study)
        prior.verify_no_evaluation(study)
        # One startup barrier validates the existing selection and token payload.
        # No data preparation, head fitting or threshold selection is performed.
        prepared = features.load_prepared(study, "test")[0]
        if prepared.get("test_selection_lock_sha256") != sha(study / "selection_lock.json"):
            raise ValueError("Prepared test payload belongs to a different selection")
        folder = study / FOLDER
        folder.mkdir()
        for name, source in {"state.before.json": study / "state.json",
                "queue_state.before.json": study / "parallel_execution/queue_state.json",
                "coordinator.before.log": study / prior.FOLDER / "coordinator.log"}.items():
            shutil.copy2(source, folder / name)
        paths = [Path(__file__).resolve(), study / "protocol.json", study / "selection_lock.json",
                 prior.control_path(study), study / prior.FOLDER / "control_registration.json",
                 study / prior.FOLDER / "coordinator_started.json", study / prior.FOLDER / "scheduler_entry.json",
                 study / "parallel_features/test/prepared_manifest.json",
                 study / "parallel_features/test/batch_plan.json", *sorted(folder.glob("*.before.*"))]
        value = dict(schema=SCHEMA, study=str(study), scope=SCOPE, policy=cooperative.POLICY,
            telemetry_policy=TELEMETRY_POLICY,
            unchanged_scientific_settings=old["unchanged_scientific_settings"],
            prior_control_sha256=sha(prior.control_path(study)),
            selection_lock_sha256=sha(study / "selection_lock.json"),
            prior_queue_sha256=sha(folder / "queue_state.before.json"),
            prior_coordinator_birth=old_birth, prior_workers_verified_exited=True,
            reason="reviewed_NVIDIA_telemetry_timeout_before_test_extraction",
            bindings={str(p): sha(p) for p in paths}, created_unix=time.time())
        write(control_path(study), value, exclusive=True)
        write(folder / "control_registration.json", dict(protocol_sha256=sha(control_path(study))), exclusive=True)
        return check_control(study)


def is_telemetry_error(exc):
    if isinstance(exc, (subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        command = exc.cmd
        first = command[0] if isinstance(command, (list, tuple)) and command else command
        return isinstance(first, str) and Path(first).name == "nvidia-smi"
    if isinstance(exc, OSError):
        return exc.filename is None or Path(str(exc.filename)).name == "nvidia-smi"
    # These are caught only around the inventory parser, never control checks.
    return isinstance(exc, (ET.ParseError, ValueError, KeyError, TypeError))


class AdmissionTelemetryUnavailable(RuntimeError):
    """A CPU-only admission handshake lost telemetry before ledger publication."""


class TestMonitorScheduler(cooperative.CooperativeScheduler):
    def __init__(self, study, **kwargs):
        self.monitor_control = check_control(study)
        self.telemetry = dict(available=None, consecutive_failures=0, total_failures=0)
        self.admission_in_progress = False
        super().__init__(study, **kwargs)
        self.raw_inventory = self.inventory
        self.inventory = self._inventory

    def _check_control(self):
        if check_control(self.study) != self.monitor_control:
            raise ValueError("Test monitor control changed after construction")

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_control()
            self._verify_previous_workers_exited()
            verify_old_coordinator(self.study)
            prior.verify_no_evaluation(self.study)
            if sha(self.output / "queue_state.json") != self.monitor_control["prior_queue_sha256"]:
                raise ValueError("Queue differs from the reviewed telemetry failure")
            write(self.study / FOLDER / "scheduler_entry.json", dict(
                coordinator_birth=queue.admission.process_birth(os.getpid()),
                control_sha256=sha(control_path(self.study)), entered_unix=self.clock()), exclusive=True)
            self._state("CPU_test_monitor_ready")
            return self
        except BaseException:
            self.lock_stream.close()
            self.lock_stream = None
            raise

    def _state(self, state, **details):
        if self.telemetry.get("available") is False and state == "CPU_wait_for_empty_candidates":
            state = "CPU_wait_for_GPU_telemetry"
        details.update(test_monitor_control_sha256=sha(control_path(self.study)),
                       allowed_resume_phases=SCOPE["allowed_phases"], gpu_telemetry=dict(self.telemetry))
        return super()._state(state, **details)

    def _inventory(self):
        try:
            rows = self.raw_inventory()
            if (not isinstance(rows, (list, tuple))
                    or {row["index"] for row in rows} != set(queue.CANDIDATES)
                    or len(rows) != len(queue.CANDIDATES)):
                raise ValueError("NVIDIA inventory omitted or duplicated a GPU")
            for row in rows:
                if (not isinstance(row["pids"], (list, tuple))
                        or any(type(pid) is not int or pid <= 0 for pid in row["pids"])
                        or not math.isfinite(row["free_gib"]) or row["free_gib"] < 0):
                    raise ValueError("NVIDIA inventory contains an invalid GPU record")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError,
                ET.ParseError, ValueError, KeyError, TypeError) as exc:
            if not is_telemetry_error(exc):
                raise
            self.telemetry.update(available=False,
                consecutive_failures=self.telemetry["consecutive_failures"] + 1,
                total_failures=self.telemetry["total_failures"] + 1,
                last_failure_unix=self.clock(), last_error_type=type(exc).__name__, last_error=str(exc))
            return []
        self.telemetry.update(available=True, consecutive_failures=0, last_success_unix=self.clock())
        return rows

    def _observe(self):
        self._check_control()
        decisions = queue.ParallelScheduler._observe(self)
        # A NVIDIA timeout can exceed the CPU child's ledger-wait deadline.
        # Defer this unpublished admission before ordinary polling could treat
        # that expected CPU-only exit as a failed scientific CUDA worker.
        if self.admission_in_progress and self.telemetry["available"] is False:
            raise AdmissionTelemetryUnavailable("NVIDIA telemetry unavailable during CPU-only admission")
        self._poll_workers()
        self._state("parallel_workers_running" if self.active else "CPU_wait_for_empty_candidates",
                    gpu_candidates=decisions)
        return decisions

    def _launch(self, *args, **kwargs):
        existing = set(self.active)
        self.admission_in_progress = True
        try:
            return super()._launch(*args, **kwargs)
        except AdmissionTelemetryUnavailable as exc:
            # Only an unpublished, verified CPU child may be retired here.
            # Existing CUDA workers are never signalled on telemetry failure.
            new = [r for k, r in self.active.items() if k not in existing]
            if any("ledger_path" in r or (self.output / "workers" / f"{r['worker_id']}.json").exists() for r in new):
                raise ValueError("Telemetry admission recovery encountered a published CUDA ledger") from exc
            if new:
                evidence = self._stop_records(new, str(exc))
                if not all(v["verified_original_child_exited"] for v in evidence):
                    raise queue.ResourceReviewRequired("Unpublished admission child exit could not be verified") from exc
                write(self.study / FOLDER / "admission_deferrals" / f"{time.time_ns()}.json",
                      dict(reason=str(exc), CUDA_handshake_published=False, workers=evidence), exclusive=True)
            return False
        finally:
            self.admission_in_progress = False

    def run_workers(self, phase, *args, **kwargs):
        if phase not in SCOPE["allowed_phases"]:
            raise ValueError("This monitor permits test extraction/evaluation only")
        return super().run_workers(phase, *args, **kwargs)


def run(study):
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_parallel_study as original

    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_control(study)
        started = study / FOLDER / "coordinator_started.json"
        if started.exists():
            raise RuntimeError("Interrupted test monitor requires review; automatic retry disabled")
        verify_old_coordinator(study)
        cooperative.verify_retired(study)
        prior.verify_no_evaluation(study)
        features.load_prepared(study, "test")
        os.environ.update(cooperative.check_control(study)["runtime_environment"])
        write(started, dict(birth=queue.admission.process_birth(os.getpid()), time=time.time()), exclusive=True)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        with TestMonitorScheduler(study) as scheduler:
            base.mark_state(study, "extracting_test", reason=None, error_type=None,
                            test_monitor_control_sha256=sha(control_path(study)))
            scheduler.run_workers("extract_test",
                lambda worker_id, gpu: original.worker_command(study, worker_id, split="test"),
                lambda: prior.test_progress(study)["complete"],
                work_remaining=lambda: prior.test_progress(study)["pending_chunks"] > 0, split="test", max_workers=8)
            check_control(study)
            base.mark_state(study, "merging_test")
            features.merge_split(study, "test")
            prior.verify_no_evaluation(study)
            base.mark_state(study, "evaluating_locked_test")
            scheduler.run_single("evaluate", lambda worker_id, gpu:
                cooperative.worker_command(study, worker_id, phase="evaluate"))
        check_control(study)
        result = study / "analysis/results.json"
        if not result.exists() or read(result).get("complete") is not True:
            raise ValueError("No completed locked test evaluation was published")
        write(study / FOLDER / "completion.json", dict(complete=True, results_sha256=sha(result),
            selection_lock_sha256=sha(study / "selection_lock.json"),
            control_sha256=sha(control_path(study))), exclusive=True)
        base.mark_state(study, "completed", backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        register(args.study)
        print(dict(state="resilient_test_monitor_registered", scope=SCOPE), flush=True)
    else:
        try:
            run(args.study)
        except BaseException as exc:
            from scripts.hanguard import frozen_base_readout_study as base
            base.mark_state(args.study, "requires_review", reason=str(exc),
                            error_type=type(exc).__name__, automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
