"""Continue unchanged readouts after all validation chunks were committed.

The previous stop's specific blocking PID was not retained. Recovery rests on
complete verified features, known successful worker exits and unopened head/test
phases, without claiming a proven cause or changing any scientific settings.
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
from scripts.hanguard import frozen_base_recovery_coordinator as previous
from scripts.hanguard import frozen_base_parallel_queue_v2 as queue_v2
from scripts.hanguard import frozen_base_parallel_queue_v3 as queue

base, features, original = previous.base, previous.features, previous.original
read = previous.read
SCHEMA = "hanguard_frozen_base_post_validation_recovery_control_v1"


def register(study):
    study = Path(study).resolve()
    base.check_stop(study)
    if (study / "recovery_002").exists():
        raise FileExistsError("Use a separate reviewed continuation; recovery_002 is immutable")
    control1 = queue_v2.check_control(study)
    scientific = base.load_protocol(study, full_hash=True)
    state, stopped = read(study / "state.json"), read(study / "parallel_execution/queue_state.json")
    if (state.get("state") != "requires_review" or stopped.get("requires_review") is not True
            or state.get("reason") != "Foreign process entered an active GPU"
            or stopped.get("reason") != state["reason"] or stopped.get("phase") != "extract_validation"):
        raise ValueError("This continuation requires the reviewed post-validation resource stop")
    if (state.get("training_started") or state.get("formal_test_opened")
            or (study / "heads/fit_started.json").exists() or (study / "selection_lock.json").exists()
            or (study / "parallel_features/test").exists()):
        raise ValueError("No head fit or test access may precede this continuation")
    prior_birth = read(study / "recovery_001/scheduler_recovery_entry.json")["coordinator_birth"]
    if not previous.original_worker_exited(prior_birth):
        raise ValueError("Previous coordinator is still live")
    termination = read(study / "parallel_execution/termination_evidence.json")
    ended = termination.get("workers", [])
    if not ended or any(w.get("exit_code") != 0 or not w.get("verified_original_child_exited")
            or w.get("verified_SIGTERM_sent") or w.get("verified_SIGKILL_sent") for w in ended):
        raise ValueError("Recorded unsignalled successful validation worker exits are required")
    ledgers = {}
    for path in sorted((study / "parallel_execution/workers").glob("*.json")):
        ledger = read(path)
        if ledger.get("schema_version") != 1 or "process_identity" not in ledger:
            continue
        if read(path.with_suffix(".registration.json")).get("ledger_sha256") != base.sha(path):
            raise ValueError("Previous worker ledger changed")
        if (ledger.get("root_protocol_sha256") != base.sha(study / "protocol.json")
                or ledger["process_identity"].get("uid") != os.getuid()
                or not previous.original_worker_exited(ledger["process_identity"])):
            raise ValueError("Previous exact worker is live, unverifiable or unrelated")
        ledgers[ledger["worker_id"]] = (path, ledger)
    for worker in ended:
        pair = ledgers.get(worker["worker_id"])
        if pair is None or pair[1]["process_identity"]["pid"] != worker["pid"] or pair[1].get("split") != "validation":
            raise ValueError("Successful exit is not bound to this validation worker")
    for split in ("train", "validation"):
        progress = features.split_progress(study, split)
        if not progress["complete"] or progress["completed_rows"] != scientific["rows"][split]:
            raise ValueError(f"{split}: incomplete feature chunks")
    if not (study / "caches/train.manifest.json").exists():
        raise ValueError("The previously verified canonical training cache is required")
    folder = study / "recovery_002"
    folder.mkdir()
    snapshots = {
        "state.before.json": study / "state.json",
        "queue_state.before.json": study / "parallel_execution/queue_state.json",
        "termination_evidence.before.json": study / "parallel_execution/termination_evidence.json",
        "coordinator.before.log": study / "recovery_001/coordinator.log",
    }
    for name, source in snapshots.items():
        shutil.copy2(source, folder / name)
    bindings = dict(control1["bindings"])
    for path in (study / "recovery_control_protocol.json", study / "recovery_control_registration.json",
                 Path(__file__).resolve(), Path(queue.__file__).resolve(),
                 study / "caches/train.manifest.json", *[folder / name for name in snapshots]):
        bindings[str(path)] = base.sha(path)
    for path, _ in ledgers.values():
        bindings[str(path)] = base.sha(path)
        bindings[str(path.with_suffix(".registration.json"))] = base.sha(path.with_suffix(".registration.json"))
    plan = read(study / "parallel_features/validation/batch_plan.json")
    for chunk in plan["chunks"]:
        path = features.chunk_directory(study, "validation", chunk["chunk_id"]) / "receipt.json"
        bindings[str(path)] = base.sha(path)
    for filename, digest in bindings.items():
        if base.sha(filename) != digest:
            raise ValueError(f"Continuation binding changed: {filename}")
    value = dict(schema=SCHEMA, study=str(study),
        root_protocol_sha256=base.sha(study / "protocol.json"),
        resource_protocol_sha256=base.sha(study / "parallel_resource_protocol.json"),
        prior_control_sha256=base.sha(study / "recovery_control_protocol.json"),
        bindings=bindings, policy=dict(queue_v2.POLICY),
        prior_review=dict(queue_state_snapshot=str(folder / "queue_state.before.json"),
            queue_state_sha256=base.sha(folder / "queue_state.before.json"),
            prior_workers_verified_exited=True, resume_authorized=True,
            all_validation_complete=True, head_training_started=False, formal_test_opened=False,
            reason="completed_validation_resource_stop", specific_trigger_not_retained=True,
            cause_uncertain=True, suspected_exit_inventory_race=True),
        unchanged_scientific_settings=scientific["settings"], unchanged_inputs_sha256=scientific["inputs"],
        automatic_head_retry=False, actual_gpu_commands_use_original_registered_entrypoints=True,
        created_unix=time.time())
    previous.exclusive_json(folder / "control_protocol.json", value)
    previous.exclusive_json(folder / "control_registration.json",
        dict(protocol_sha256=base.sha(folder / "control_protocol.json")))
    queue.check_control2(study)
    return value


def run(study):
    study = Path(study).resolve()
    base.check_stop(study)
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator owns this study") from exc
        queue.check_control2(study)
        state = read(study / "state.json")
        if state.get("state") != "requires_review" or state.get("training_started"):
            raise ValueError("An already resumed continuation cannot silently repeat")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        base.mark_state(study, "merging_validation", error_type=None, reason=None,
            recovery_control_protocol_sha256=base.sha(study / "recovery_002/control_protocol.json"))
        # Validates complete feature bytes, every finite value, original row
        # coverage, replay checks and agreement between GPU0 and GPU4.
        features.merge_split(study, "validation")
        base.mark_state(study, "waiting_for_readout_gpu")
        queue.run_single(study, "fit", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="fit"))
        base.validate_selection_lock(study)
        base.mark_state(study, "validation_selections_locked")
        previous.extract(study, "test", queue)
        base.mark_state(study, "evaluating_locked_test")
        queue.run_single(study, "evaluate", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="evaluate"))
        result = study / "analysis/results.json"
        if not result.exists() or read(result).get("complete") is not True:
            raise ValueError("No completed locked test evaluation was published")
        base.dump(study / "recovery_002/completion.json", dict(complete=True,
            recovery_control_protocol_sha256=base.sha(study / "recovery_002/control_protocol.json"),
            original_scientific_protocol_sha256=base.sha(study / "protocol.json"),
            results_sha256=base.sha(result), original_registered_gpu_entrypoints=True))
        base.mark_state(study, "completed", backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        value = register(args.study)
        print(dict(state="post_validation_recovery_registered", study=value["study"],
            max_cuda_workers=8, reserved_gpus=[], feature_reextraction_required=False), flush=True)
    else:
        try:
            run(args.study)
        except BaseException as exc:
            base.mark_state(args.study, "stopped" if (args.study / "STOP").exists() else "requires_review",
                error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
