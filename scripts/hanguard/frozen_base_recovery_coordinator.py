"""Reviewed continuation after the frozen-feature worker exited successfully.

Original scientific and resource protocols remain unchanged. New CPU control
code is registered separately; every GPU command still uses an original,
registered entrypoint. Completed train chunks pass the original merge gates.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard import frozen_base_parallel_features as features
from scripts.hanguard import frozen_base_parallel_study as original

SCHEMA = "hanguard_frozen_base_recovery_control_v1"


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def original_worker_exited(identity, *, birth_reader=None):
    if birth_reader is None:
        from scripts.hanguard.frozen_base_readout_admission_v2 import process_birth
        birth_reader = process_birth
    current = birth_reader(identity["pid"])
    if current is None:
        return True
    ticks = identity.get("start_time_ticks", identity.get("start_ticks"))
    if current["start_time_ticks"] != ticks:
        return True
    if current["uid"] != identity["uid"]:
        raise ValueError("Original process birth has an unknown changed owner")
    return current["state"] in ("Z", "X", "x")


def register_recovery(study):
    """Review prior evidence and pin new CPU control before resuming any work."""
    from scripts.hanguard import frozen_base_parallel_queue_v2 as queue

    study = Path(study).resolve()
    base.check_stop(study)
    if (study / "recovery_control_protocol.json").exists():
        raise FileExistsError("Recovery registration is immutable; review any later attempt separately")
    protocol = base.load_protocol(study, full_hash=True)
    state = read(study / "state.json")
    prior = read(study / "parallel_execution/queue_state.json")
    if (state.get("state") != "requires_review" or prior.get("requires_review") is not True
            or state.get("reason") != "Own worker identity became unverifiable"
            or prior.get("reason") != state["reason"] or prior.get("phase") != "extract_train"):
        raise ValueError("This continuation is specific to the reviewed feature-worker exit race")
    if (state.get("training_started") or state.get("formal_test_opened")
            or state.get("formal_validation_opened") or (study / "heads/fit_started.json").exists()
            or (study / "selection_lock.json").exists() or (study / "parallel_features/test").exists()):
        raise ValueError("No head training or test access may precede this recovery review")
    from scripts.hanguard.frozen_base_readout_admission_v2 import process_birth
    if process_birth(prior["monitor_pid"]) is not None:
        raise ValueError("The prior coordinator still exists; an isolated continuation is required")
    termination = read(study / "parallel_execution/termination_evidence.json")
    final_workers = termination.get("workers", [])
    if not final_workers or any(
            w.get("exit_code") != 0 or w.get("verified_original_child_exited") is not True
            or w.get("verified_SIGTERM_sent") or w.get("verified_SIGKILL_sent") for w in final_workers):
        raise ValueError("No recorded, unsignalled successful worker exit proves this recovery case")
    ledgers = []
    for path in sorted((study / "parallel_execution/workers").glob("*.json")):
        ledger = read(path)
        if "process_identity" not in ledger or "worker_id" not in ledger:
            continue
        if read(path.with_suffix(".registration.json")).get("ledger_sha256") != base.sha(path):
            raise ValueError("Previous worker ledger changed")
        if (ledger.get("root_protocol_sha256") != base.sha(study / "protocol.json")
                or ledger["process_identity"].get("uid") != os.getuid()
                or not original_worker_exited(ledger["process_identity"])):
            raise ValueError("Previous exact worker is live or unrelated")
        ledgers.append((path, ledger))
    ledger_pids = {ledger["process_identity"]["pid"] for _, ledger in ledgers}
    if any(w["pid"] not in ledger_pids for w in final_workers):
        raise ValueError("Recorded successful exit is not bound to this study's worker ledger")
    progress = features.split_progress(study, "train")
    if progress["complete"] is not True or progress["completed_rows"] != protocol["rows"]["train"]:
        raise ValueError("Every train feature chunk must have a complete receipt")
    resource_path = study / "parallel_resource_protocol.json"
    resource = read(resource_path)
    if read(study / "parallel_resource_registration.json").get("protocol_sha256") != base.sha(resource_path):
        raise ValueError("Original resource registration changed")
    folder = study / "recovery_001"
    folder.mkdir(exist_ok=False)
    snapshots = {
        "state.before.json": study / "state.json",
        "queue_state.before.json": study / "parallel_execution/queue_state.json",
        "termination_evidence.before.json": study / "parallel_execution/termination_evidence.json",
        "coordinator.before.log": study / "coordinator.log",
    }
    for name, source in snapshots.items():
        shutil.copy2(source, folder / name)
    bindings = dict(resource["bindings"])
    for path in (Path(__file__).resolve(), Path(queue.__file__).resolve(), *[folder / name for name in snapshots]):
        bindings[str(path)] = base.sha(path)
    for path, _ in ledgers:
        bindings[str(path)] = base.sha(path)
        bindings[str(path.with_suffix(".registration.json"))] = base.sha(path.with_suffix(".registration.json"))
    plan = read(study / "parallel_features/train/batch_plan.json")
    for chunk in plan["chunks"]:
        path = features.chunk_directory(study, "train", chunk["chunk_id"]) / "receipt.json"
        bindings[str(path)] = base.sha(path)
    for filename, digest in bindings.items():
        if base.sha(filename) != digest:
            raise ValueError(f"Recovery binding changed: {filename}")
    value = dict(schema=SCHEMA, study=str(study),
        root_protocol_sha256=base.sha(study / "protocol.json"),
        resource_protocol_sha256=base.sha(resource_path), bindings=bindings,
        policy=dict(max_cuda_workers=8, reserved_gpus=[], never_terminate_foreign_processes=True,
                    preserve_registered_science=True),
        prior_review=dict(queue_state_sha256=base.sha(folder / "queue_state.before.json"),
            queue_state_snapshot=str(folder / "queue_state.before.json"),
            prior_workers_verified_exited=True, resume_authorized=True,
            reason="normal_exit_identity_read_race", final_worker_exit_code=0,
            all_train_chunks_complete=True, head_training_started=False, formal_test_opened=False,
            prior_monitor_pid=prior["monitor_pid"]),
        unchanged_scientific_settings=protocol["settings"],
        unchanged_inputs_sha256=protocol["inputs"],
        recovery_scope="Merge verified train chunks, then original validation/head fitting/locked test phases",
        actual_gpu_commands_use_original_registered_entrypoints=True,
        automatic_head_retry=False, created_unix=time.time())
    exclusive_json(study / "recovery_control_protocol.json", value)
    exclusive_json(study / "recovery_control_registration.json",
                   dict(protocol_sha256=base.sha(study / "recovery_control_protocol.json")))
    queue.check_control(study)
    return value


def extract(study, split, queue):
    base.mark_state(study, f"preparing_{split}")
    features.prepare_split(study, split)
    base.mark_state(study, f"extracting_{split}")
    queue.run_workers(study, f"extract_{split}",
        lambda worker_id, gpu: original.worker_command(study, worker_id, split=split),
        lambda: features.split_complete(study, split),
        work_remaining=lambda: features.has_work(study, split), split=split, max_workers=8,
        on_resource_loss=lambda worker_id, gpu, lost_split, reason:
            features.abandon_worker(study, lost_split, worker_id, reason))
    base.mark_state(study, f"merging_{split}")
    features.merge_split(study, split)


def run(study):
    from scripts.hanguard import frozen_base_parallel_queue_v2 as queue

    study = Path(study).resolve()
    base.check_stop(study)
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator owns this study") from exc
        queue.check_control(study)
        previous = read(study / "state.json")
        if previous.get("state") != "requires_review" or previous.get("training_started"):
            raise ValueError("This recovery cannot silently repeat an already resumed attempt")
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        base.mark_state(study, "merging_train", error_type=None, reason=None,
            recovery_control_protocol_sha256=base.sha(study / "recovery_control_protocol.json"))
        # Merge rechecks every complete feature SHA, finite value, original row
        # index and repeated-forward audit, including agreement across GPUs.
        features.merge_split(study, "train")
        extract(study, "validation", queue)
        base.mark_state(study, "waiting_for_readout_gpu")
        queue.run_single(study, "fit", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="fit"))
        base.validate_selection_lock(study)
        base.mark_state(study, "validation_selections_locked")
        # Test text/labels are prepared only after all readout selections lock.
        extract(study, "test", queue)
        base.mark_state(study, "evaluating_locked_test")
        queue.run_single(study, "evaluate", lambda worker_id, gpu:
            original.worker_command(study, worker_id, phase="evaluate"))
        result = study / "analysis/results.json"
        if not result.exists() or read(result).get("complete") is not True:
            raise ValueError("No completed locked test evaluation was published")
        base.dump(study / "recovery_001/completion.json", dict(complete=True,
            recovery_control_protocol_sha256=base.sha(study / "recovery_control_protocol.json"),
            original_scientific_protocol_sha256=base.sha(study / "protocol.json"),
            results_sha256=base.sha(result), original_registered_gpu_entrypoints=True))
        base.mark_state(study, "completed", backbone_training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "init":
        value = register_recovery(args.study)
        print(json.dumps(dict(state="recovery_registered", study=value["study"],
            max_cuda_workers=8, reserved_gpus=[], train_reextraction_required=False)), flush=True)
    else:
        try:
            run(args.study)
        except BaseException as exc:
            base.mark_state(args.study, "stopped" if (args.study / "STOP").exists() else "requires_review",
                error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise


if __name__ == "__main__":
    main()
