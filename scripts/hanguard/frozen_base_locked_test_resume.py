"""Reviewed CPU-only continuation of an already selected, unevaluated test run.

This append-only control never fits heads, prepares another split, or changes a
selection. GPU commands remain the previously registered feature/evaluation
entrypoints, including empty-card admission and record-only foreign arrivals.
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
from scripts.hanguard import frozen_base_cooperative_runner as cooperative

queue = cooperative.queue
read, sha, write = queue._read, queue._sha, queue._write
FOLDER = "recovery_005"
SCHEMA = "hanguard_locked_test_only_resume_v1"
SCOPE = dict(allowed_phases=["extract_test", "evaluate"], fit_allowed=False,
             selection_changes_allowed=False, partial_features_reused=False,
             test_evaluations=1, automatic_retry=False)


def control_path(study):
    return Path(study) / FOLDER / "control_protocol.json"


def test_progress(study):
    """Poll immutable plans and chunk receipts without rehashing frozen caches.

    Registration/startup validate all selected artifacts through load_prepared;
    CUDA workers and final merge/evaluation retain their original full barriers.
    Monitoring only needs the locked plan, complete receipts and process births.
    """
    from scripts.hanguard import frozen_base_parallel_features as features

    study = Path(study).resolve()
    check_control(study)
    folder = study / "parallel_features/test"
    prepared, plan = read(folder / "prepared_manifest.json"), read(folder / "batch_plan.json")
    complete = claimed = pending = rows = 0
    for chunk in plan["chunks"]:
        receipt = features._quick_receipt(study, "test", prepared, chunk)
        if receipt is not None:
            complete += 1
            rows += chunk["rows"]
            continue
        path = features.chunk_directory(study, "test", chunk["chunk_id"]) / "claim.json"
        claim = read(path) if path.exists() else None
        if claim and features._same_live_process(claim["identity"]):
            claimed += 1
        else:
            pending += 1
    return dict(complete=complete == len(plan["chunks"]), complete_chunks=complete,
                claimed_chunks=claimed, pending_chunks=pending, completed_rows=rows)


def check_control(study):
    study = Path(study).resolve()
    prior = cooperative.check_control(study)
    path = control_path(study)
    value = read(path)
    if read(path.with_name("control_registration.json")).get("protocol_sha256") != sha(path):
        raise ValueError("Locked-test continuation registration changed")
    required = [Path(__file__).resolve(), cooperative.control_path(study),
                study / "selection_lock.json", study / "protocol.json",
                study / "parallel_features/test/prepared_manifest.json",
                study / "parallel_features/test/batch_plan.json"]
    bindings = value.get("bindings", {})
    if (value.get("schema") != SCHEMA or value.get("study") != str(study)
            or value.get("scope") != SCOPE or value.get("policy") != cooperative.POLICY
            or value.get("unchanged_scientific_settings") != prior["unchanged_scientific_settings"]
            or value.get("prior_control_sha256") != sha(cooperative.control_path(study))
            or value.get("selection_lock_sha256") != sha(study / "selection_lock.json")
            or any(str(p) not in bindings for p in required)):
        raise ValueError("Locked-test continuation identity or scope changed")
    for filename, digest in bindings.items():
        if sha(filename) != digest:
            raise ValueError(f"Locked-test continuation evidence changed: {filename}")
    return value


def verify_no_evaluation(study):
    """Fail closed on even a partial prior test evaluation, never overwrite it."""
    study = Path(study)
    analysis = study / "analysis"
    if analysis.exists() and any(analysis.iterdir()):
        raise ValueError("Test evaluation artifacts already exist; explicit review required")
    for path in (study / "parallel_execution/workers").glob("*.json"):
        if path.name.startswith("evaluate_") or read(path).get("phase") == "evaluate":
            raise ValueError("A test evaluation worker was already registered")


def verify_old_coordinator(study):
    folder = Path(study) / cooperative.FOLDER
    expected = read(folder / "coordinator_started.json")["birth"]
    recorded = read(folder / "scheduler_entry.json")["coordinator_birth"]
    keys = ("pid", "uid", "start_time_ticks")
    if any(expected.get(k) != recorded.get(k) for k in keys) or expected.get("uid") != os.getuid():
        raise ValueError("Prior coordinator birth/owner records differ")
    current = queue.admission.process_birth(expected["pid"])
    if current is not None and current.get("start_time_ticks") == expected["start_time_ticks"]:
        if current.get("uid") != expected["uid"] or current.get("state") not in ("Z", "X"):
            raise ValueError("Prior exact coordinator is still alive or unverifiable")
    return expected


def verify_completed_fit(study, state):
    completed = [w for w in state.get("completed_workers", [])
                 if w.get("phase") == "fit" and w.get("exit_code") == 0]
    if len(completed) != 1:
        raise ValueError("Expected exactly one successful selected-head fit")
    worker = completed[0]
    folder = Path(study) / "parallel_execution/workers"
    path = folder / f"{worker['worker_id']}.json"
    ledger, receipt = read(path), read(path.with_suffix(".complete.json"))
    if (read(path.with_suffix(".registration.json")).get("ledger_sha256") != sha(path)
            or ledger.get("phase") != "fit" or ledger.get("worker_id") != worker["worker_id"]
            or ledger.get("process_identity", {}).get("pid") != worker["pid"]
            or ledger.get("process_identity", {}).get("uid") != os.getuid()
            or ledger.get("root_protocol_sha256") != sha(Path(study) / "protocol.json")
            or receipt.get("complete") is not True or receipt.get("phase") != "fit"
            or receipt.get("pid") != worker["pid"]
            or receipt.get("protocol_sha256") != sha(Path(study) / "protocol.json")):
        raise ValueError("Successful head-fit evidence differs")
    return [path, path.with_suffix(".registration.json"), path.with_suffix(".complete.json")]


def register(study):
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import frozen_base_parallel_features as features

    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / FOLDER).exists():
            raise FileExistsError("Locked-test continuation is already registered")
        prior = cooperative.check_control(study)
        state, previous = read(study / "state.json"), read(study / "parallel_execution/queue_state.json")
        if (state.get("state") != "extracting_test" or state.get("formal_test_opened") is not True
                or previous.get("phase") != "extract_test" or previous.get("active_workers") != 0
                or previous.get("workers") or previous.get("requires_review")
                or previous.get("runtime_control_sha256") != sha(cooperative.control_path(study))):
            raise ValueError("Expected an interrupted coordinator waiting for test extraction")
        old_birth = verify_old_coordinator(study)
        cooperative.verify_retired(study)
        verify_no_evaluation(study)
        fit_evidence = verify_completed_fit(study, previous)
        # Reads the already prepared token payload under its original lock. No
        # tokenizer, data selection, threshold selection or preparation occurs.
        prepared = features.load_prepared(study, "test")[0]
        if prepared.get("test_selection_lock_sha256") != sha(study / "selection_lock.json"):
            raise ValueError("Prepared test payload belongs to a different selection")
        folder = study / FOLDER
        folder.mkdir()
        for name, source in {"state.before.json": study / "state.json",
                "queue_state.before.json": study / "parallel_execution/queue_state.json",
                "coordinator.before.log": study / cooperative.FOLDER / "coordinator.log"}.items():
            shutil.copy2(source, folder / name)
        paths = [Path(__file__).resolve(), study / "protocol.json", study / "selection_lock.json",
                 cooperative.control_path(study), study / cooperative.FOLDER / "control_registration.json",
                 study / cooperative.FOLDER / "coordinator_started.json",
                 study / cooperative.FOLDER / "scheduler_entry.json",
                 study / "parallel_features/test/prepared_manifest.json",
                 study / "parallel_features/test/batch_plan.json", *fit_evidence,
                 *sorted(folder.glob("*.before.*")),
                 *sorted((study / "parallel_features/test/chunks").glob("*/receipt.json"))]
        value = dict(schema=SCHEMA, study=str(study), scope=SCOPE, policy=cooperative.POLICY,
            unchanged_scientific_settings=prior["unchanged_scientific_settings"],
            prior_control_sha256=sha(cooperative.control_path(study)),
            selection_lock_sha256=sha(study / "selection_lock.json"),
            prior_queue_sha256=sha(folder / "queue_state.before.json"),
            prior_coordinator_birth=old_birth, prior_workers_verified_exited=True,
            reason="reviewed_disappeared_CPU_coordinator_after_selection_before_evaluation",
            bindings={str(p): sha(p) for p in paths}, created_unix=time.time())
        write(control_path(study), value, exclusive=True)
        write(folder / "control_registration.json", dict(protocol_sha256=sha(control_path(study))), exclusive=True)
        return check_control(study)


class LockedTestScheduler(cooperative.CooperativeScheduler):
    def __init__(self, study, **kwargs):
        self.resume_control = check_control(study)
        super().__init__(study, **kwargs)

    def _check_control(self):
        super()._check_control()
        if check_control(self.study) != self.resume_control:
            raise ValueError("Locked-test continuation changed after construction")

    def __enter__(self):
        self.lock_stream = (self.output / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check_control()
            self._verify_previous_workers_exited()
            verify_old_coordinator(self.study)
            verify_no_evaluation(self.study)
            path = self.output / "queue_state.json"
            if sha(path) != self.resume_control["prior_queue_sha256"]:
                raise ValueError("Queue differs from the reviewed stopped coordinator")
            write(self.study / FOLDER / "scheduler_entry.json", dict(
                coordinator_birth=queue.admission.process_birth(os.getpid()),
                control_sha256=sha(control_path(self.study)), entered_unix=self.clock()), exclusive=True)
            self._state("CPU_locked_test_monitor_ready")
            return self
        except BaseException:
            self.lock_stream.close()
            self.lock_stream = None
            raise

    def _state(self, state, **details):
        details.update(post_selection_recovery_control_sha256=sha(control_path(self.study)),
                       allowed_resume_phases=SCOPE["allowed_phases"])
        return super()._state(state, **details)

    def run_workers(self, phase, *args, **kwargs):
        if phase not in SCOPE["allowed_phases"]:
            raise ValueError("This continuation permits test extraction/evaluation only")
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
            raise RuntimeError("Interrupted continuation requires review; automatic retry disabled")
        verify_old_coordinator(study)
        cooperative.verify_retired(study)
        verify_no_evaluation(study)
        features.load_prepared(study, "test")
        os.environ.update(cooperative.check_control(study)["runtime_environment"])
        write(started, dict(birth=queue.admission.process_birth(os.getpid()), time=time.time()), exclusive=True)
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        with LockedTestScheduler(study) as scheduler:
            base.mark_state(study, "extracting_test", post_selection_recovery_control_sha256=sha(control_path(study)))
            scheduler.run_workers("extract_test",
                lambda worker_id, gpu: original.worker_command(study, worker_id, split="test"),
                lambda: test_progress(study)["complete"],
                work_remaining=lambda: test_progress(study)["pending_chunks"] > 0, split="test", max_workers=8)
            check_control(study)
            base.mark_state(study, "merging_test")
            features.merge_split(study, "test")
            verify_no_evaluation(study)
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
        print(dict(state="locked_test_continuation_registered", scope=SCOPE), flush=True)
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
