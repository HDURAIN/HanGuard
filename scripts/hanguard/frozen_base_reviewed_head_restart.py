"""One reviewed, same-seed head restart after a confirmed foreign GPU arrival.

No completed epoch, checkpoint, selection or test access exists in the stopped
attempt. Preserve it in full, then run the original five-arm implementation from
its original initialization. Future interruptions still require a fresh review.
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
from scripts.hanguard import frozen_base_post_validation_recovery as previous
from scripts.hanguard import frozen_base_parallel_queue_v4 as queue

base, features, original = previous.base, previous.features, previous.original
read, exclusive_json = previous.read, previous.previous.exclusive_json
SCHEMA = "hanguard_frozen_base_reviewed_head_restart_control_v1"


def scratch_environment(study):
    folder = Path(study).resolve() / "runtime_scratch"
    values = dict(TMPDIR=str(folder / "tmp"), TRITON_CACHE_DIR=str(folder / "triton"),
                  TORCHINDUCTOR_CACHE_DIR=str(folder / "torchinductor"), CUDA_CACHE_PATH=str(folder / "cuda"))
    for directory in values.values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    return values


def register(study):
    study = Path(study).resolve()
    base.check_stop(study)
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator owns this study") from exc
        if (study / "recovery_003").exists() or (study / "fit_attempts/001").exists():
            raise FileExistsError("This reviewed restart is immutable; review any subsequent stop separately")
        control2 = previous.queue.check_control2(study)
        # Bind the same original input identities. The unchanged GPU worker
        # fully hashes all large inputs before using any model/data/cache.
        scientific = base.load_protocol(study, full_hash=False)
        state, stopped = read(study / "state.json"), read(study / "parallel_execution/queue_state.json")
        if (state.get("state") != "requires_review" or state.get("training_started") is not True
                or stopped.get("phase") != "fit" or stopped.get("requires_review") is not True
                or stopped.get("reason") != "Foreign process entered an active GPU"):
            raise ValueError("A reviewed resource interruption during head fitting is required")
        if state.get("formal_test_opened") or (study / "selection_lock.json").exists() or (study / "parallel_features/test").exists():
            raise ValueError("A selected or tested attempt cannot be discarded and retrained")
        prior_birth = read(study / "recovery_002/scheduler_recovery_entry.json")["coordinator_birth"]
        if not previous.previous.original_worker_exited(prior_birth):
            raise ValueError("Previous exact coordinator is still live")
        heads = study / "heads"
        if (not heads.is_dir() or {p.name for p in heads.iterdir()} != {"fit_started.json", "budget_12"}
                or not (heads / "budget_12").is_dir() or any((heads / "budget_12").iterdir())):
            raise ValueError("Only a stopped attempt with zero published epochs/checkpoints may restart")
        started = read(heads / "fit_started.json")
        if (started.get("protocol_sha256") != base.sha(study / "protocol.json")
                or started.get("test_opened") is not False
                or not started.get("source_sha256")
                or any(scientific["code_sha256"].get(name) != digest for name, digest in started["source_sha256"].items())):
            raise ValueError("Interrupted fit does not bind the original scientific sources")
        for split in ("train", "validation"):
            manifest = read(study / "caches" / f"{split}.manifest.json")
            if (manifest.get("complete") is not True or manifest.get("rows") != scientific["rows"][split]
                    or manifest.get("protocol_sha256") != base.sha(study / "protocol.json")):
                raise ValueError(f"{split}: verified canonical features are required")
        termination = read(study / "parallel_execution/termination_evidence.json")
        ended = termination.get("workers", [])
        if len(ended) != 1 or ended[0].get("exit_code") != -15 or ended[0].get("verified_SIGTERM_sent") is not True or ended[0].get("verified_original_child_exited") is not True or ended[0].get("verified_SIGKILL_sent"):
            raise ValueError("The exact owned head worker's resource stop must be recorded")
        worker_id, pid = ended[0]["worker_id"], ended[0]["pid"]
        ledger_path = study / "parallel_execution/workers" / f"{worker_id}.json"
        ledger, identity = read(ledger_path), read(ledger_path)["process_identity"]
        if (read(ledger_path.with_suffix(".registration.json")).get("ledger_sha256") != base.sha(ledger_path)
                or ledger.get("phase") != "fit" or identity["pid"] != pid or identity["uid"] != os.getuid()
                or not previous.previous.original_worker_exited(identity)):
            raise ValueError("The stopped worker's exact birth/command is not safely retired")
        events = []
        for path in sorted((study / "recovery_002/failure_events").glob("*.json")):
            event = read(path)
            trigger = event.get("trigger_evidence", {})
            active = event.get("active_workers", [])
            if (event.get("phase") == "fit" and event.get("reason") == stopped["reason"]
                    and pid in trigger.get("owned_pids", [])
                    and any(w.get("worker_id") == worker_id and w.get("process_identity") == identity for w in active)
                    and any(b.get("classification") == "foreign" and b.get("uid") != os.getuid()
                            for b in trigger.get("blocking_processes", []))):
                events.append(path)
        if not events:
            raise ValueError("Independent confirmed foreign GPU arrival evidence is required")
        worker_log = ledger_path.with_suffix(".log")
        if not worker_log.exists() or worker_log.stat().st_size != 0:
            raise ValueError("Published head epochs or unexpected fit output require separate review")
        folder = study / "recovery_003"
        folder.mkdir()
        snapshots = {"state.before.json": study / "state.json",
            "queue_state.before.json": study / "parallel_execution/queue_state.json",
            "termination_evidence.before.json": study / "parallel_execution/termination_evidence.json",
            "coordinator.before.log": study / "recovery_002/coordinator.log"}
        for name, source in snapshots.items():
            shutil.copy2(source, folder / name)
        archive = study / "fit_attempts/001"
        archive.mkdir(parents=True)
        # Same-filesystem atomic rename: every old byte is retained, never adopted.
        heads.rename(archive / "heads")
        manifest_path = archive / "archive_manifest.json"
        archive_manifest = dict(attempt=1, new_attempt=2, completed_epochs=0,
            partial_checkpoint_reuse=False, reason="confirmed_foreign_gpu_interruption_during_fit",
            source_worker_id=worker_id, source_worker_identity=identity,
            scientific_protocol_sha256=base.sha(study / "protocol.json"),
            files={"heads/fit_started.json": base.sha(archive / "heads/fit_started.json")},
            empty_directories=["heads/budget_12"], archived_unix=time.time())
        exclusive_json(manifest_path, archive_manifest)
        bindings = dict(control2["bindings"])
        paths = [study / "recovery_002/control_protocol.json", study / "recovery_002/control_registration.json",
            Path(__file__).resolve(), Path(queue.__file__).resolve(), manifest_path,
            archive / "heads/fit_started.json", ledger_path, ledger_path.with_suffix(".registration.json"), worker_log,
            *events, *[folder / name for name in snapshots],
            study / "caches/train.manifest.json", study / "caches/validation.manifest.json"]
        for path in paths:
            bindings[str(path)] = base.sha(path)
        value = dict(schema=SCHEMA, study=str(study),
            root_protocol_sha256=base.sha(study / "protocol.json"),
            resource_protocol_sha256=base.sha(study / "parallel_resource_protocol.json"),
            prior_control_sha256=base.sha(study / "recovery_002/control_protocol.json"),
            bindings=bindings, policy=dict(previous.queue_v2.POLICY),
            unchanged_scientific_settings=scientific["settings"],
            runtime_environment=scratch_environment(study),
            automatic_head_retry=False, actual_gpu_commands_use_original_registered_entrypoints=True,
            prior_review=dict(queue_state_snapshot=str(folder / "queue_state.before.json"),
                queue_state_sha256=base.sha(folder / "queue_state.before.json"),
                prior_workers_verified_exited=True, resume_authorized=True,
                reason="confirmed_foreign_gpu_interruption_during_fit", reviewed_same_seed_restart_by_root=True,
                formal_test_opened=False, selection_lock_absent=True, partial_checkpoint_reuse=False,
                automatic_head_retry=False, restart_seed=scientific["settings"]["seed"],
                archived_fit_attempt=1, new_fit_attempt=2, archive_manifest=str(manifest_path),
                archive_manifest_sha256=base.sha(manifest_path)), created_unix=time.time())
        exclusive_json(folder / "control_protocol.json", value)
        exclusive_json(folder / "control_registration.json", dict(protocol_sha256=base.sha(folder / "control_protocol.json")))
        queue.check_control3(study)
        return value


def run(study):
    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator owns this study") from exc
        control = queue.check_control3(study)
        state = read(study / "state.json")
        if state.get("state") != "requires_review" or (study / "heads").exists() or (study / "selection_lock.json").exists():
            raise ValueError("This reviewed restart cannot be silently repeated")
        os.environ.update(control["runtime_environment"])
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + "\n"); lock.flush()
        base.mark_state(study, "waiting_for_readout_gpu", error_type=None, reason=None,
            training_started=True, head_fit_attempt=2,
            recovery_control_protocol_sha256=base.sha(study / "recovery_003/control_protocol.json"))
        queue.run_single(study, "fit", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="fit"))
        base.validate_selection_lock(study)
        base.mark_state(study, "validation_selections_locked")
        previous.previous.extract(study, "test", queue)
        base.mark_state(study, "evaluating_locked_test")
        queue.run_single(study, "evaluate", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="evaluate"))
        result = study / "analysis/results.json"
        if not result.exists() or read(result).get("complete") is not True:
            raise ValueError("No completed locked test evaluation was published")
        base.dump(study / "recovery_003/completion.json", dict(complete=True, head_fit_attempt=2,
            reviewed_same_seed_restart=True, partial_checkpoint_reuse=False,
            recovery_control_protocol_sha256=base.sha(study / "recovery_003/control_protocol.json"),
            original_scientific_protocol_sha256=base.sha(study / "protocol.json"), results_sha256=base.sha(result)))
        base.mark_state(study, "completed", backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        value = register(args.study)
        print(dict(state="reviewed_same_seed_restart_registered", seed=value["prior_review"]["restart_seed"],
            archived_attempt=1, new_attempt=2, automatic_head_retry=False), flush=True)
    else:
        try:
            run(args.study)
        except BaseException as exc:
            base.mark_state(args.study, "stopped" if (args.study / "STOP").exists() else "requires_review",
                error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
