"""Explicit user-authorized shared-GPU amendment to the frozen Guard test.

Scientific inputs, prompts, generation settings, batches and scoring are
unchanged. Only GPU admission is broadened, with a PyTorch allocator budget.
The allocator limit excludes some driver/workspace memory; eight GiB of entry
headroom is therefore required in addition to its fourteen GiB budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import qwen3guard_clean_eval as evaluator
from scripts.hanguard import qwen3guard_clean_queue as queue

CAP_GIB = 14.
HEADROOM_GIB = 8.
MIN_FREE_GIB = CAP_GIB + HEADROOM_GIB
SCHEMA = "qwen3guard_authorized_shared_admission_v1"
_ORIGINAL_STATE = queue.coordinator_state


def amendment_path(output):
    return Path(output) / "shared_admission_amendment.json"


def registration(output, args):
    output = Path(output).resolve()
    sources = {str(path.resolve()): queue.sha(path) for path in
               (Path(__file__), Path(evaluator.__file__), Path(queue.__file__))}
    expected = dict(schema=SCHEMA, protocol_sha256=queue.sha(output / "protocol.json"),
        original_queue_registration_sha256=queue.sha(output / "queue_registration.json"),
        plan_sha256=queue.sha(output / "plan.json"), sources=sources,
        authorization="User explicitly allowed sharing GPUs with other users on 2026-10-25",
        gpu_ids=args.gpus, poll_seconds=args.poll_seconds,
        minimum_free_gib=args.minimum_free_gib, allocator_cap_gib=CAP_GIB,
        admission_headroom_gib=HEADROOM_GIB,
        admission="Other users may already be present; require verified inventory, UUID and free memory",
        other_processes="Never signal, reconfigure or evict another user's process",
        unchanged="Raw texts, all 8350 rows, official template, precision, decoding, fixed batch plan and scoring",
        batch_splitting="Only the original registered OOM-recursive splitting is allowed",
        timing="Shared-GPU timings are operational estimates, not isolated efficiency benchmarks")
    path = amendment_path(output)
    if path.exists():
        if queue.read(path) != expected:
            raise ValueError("Registered shared-admission amendment changed")
    else:
        queue.write(path, expected)
        snapshots = output / "shared_source_snapshot"
        snapshots.mkdir()
        for filename, digest in sources.items():
            target = snapshots / Path(filename).name
            target.write_bytes(Path(filename).read_bytes())
            if queue.sha(target) != digest:
                raise ValueError("Shared-admission source changed while registering")
    return expected


def verify_amendment(output):
    output = Path(output).resolve()
    record = queue.read(amendment_path(output))
    if (record.get("schema") != SCHEMA
            or record.get("protocol_sha256") != queue.sha(output / "protocol.json")
            or record.get("original_queue_registration_sha256") != queue.sha(output / "queue_registration.json")
            or record.get("plan_sha256") != queue.sha(output / "plan.json")
            or record.get("allocator_cap_gib") != CAP_GIB
            or record.get("admission_headroom_gib") != HEADROOM_GIB
            or record.get("minimum_free_gib") != MIN_FREE_GIB):
        raise ValueError("Shared-GPU admission registration is invalid")
    queue.verify_sources(record)
    return record


def shared_eligible(card, minimum_free_gib):
    return card["free_gib"] >= max(MIN_FREE_GIB, minimum_free_gib)


def admission(gpu, minimum_free_gib=MIN_FREE_GIB):
    """Read-only admission; existing foreign PIDs are explicitly allowed."""
    try:
        card = next(row for row in evaluator._gpu_inventory() if row["index"] == gpu)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != card["uuid"]:
            return False, dict(reason="Physical GPU and visible UUID differ")
        if not shared_eligible(card, minimum_free_gib):
            return False, dict(reason="Insufficient shared-GPU headroom", inventory=card)
        return True, dict(card, admission_policy=SCHEMA)
    except Exception as exc:
        return False, dict(reason="Unknown shared-GPU telemetry", error=repr(exc))


def state(output, phase, **details):
    if phase == "waiting_for_empty_gpu":
        phase = "waiting_for_shared_capacity"
    events = []
    for event in details.get("recent_events", []):
        item = dict(event)
        if item.get("event") == "worker_launch":
            item["event"] = "shared_worker_launch"
            item["admission_inventory"] = item.pop("empty_inventory")
        events.append(item)
    if "recent_events" in details:
        details["recent_events"] = events
    _ORIGINAL_STATE(output, phase, **details,
        admission_amendment_sha256=queue.sha(amendment_path(output)),
        shared_gpus_allowed=True, allocator_cap_gib=CAP_GIB,
        admission_headroom_gib=HEADROOM_GIB)


def work(output, gpu):
    output = Path(output).resolve()
    verify_amendment(output)
    evaluator.verify(output)
    accepted, inventory = admission(gpu)
    if not accepted:
        return 75
    # Admission precedes importing torch and setting its allocator limit, which
    # initializes a CUDA context. No other user's process is modified.
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = CAP_GIB * 1024**3 / total
    if not 0 < fraction < 1:
        raise ValueError("The physical GPU is smaller than the registered allocator budget")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    owner = evaluator.identity(os.getpid())
    telemetry = output / "shared_workers" / f"{owner['pid']}_{owner['start_time_ticks']}.json"
    record = dict(identity=owner, gpu=gpu, entry_inventory=inventory,
        amendment_sha256=queue.sha(amendment_path(output)), allocator_fraction=fraction,
        allocator_cap_gib=CAP_GIB, started_unix=time.time())
    evaluator.write(telemetry, record)
    evaluator.admission = admission
    try:
        code = evaluator.work(output, gpu)
        record["exit_code"] = code
        return code
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record.update(finished_unix=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3)
        evaluator.write(telemetry, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--output", type=Path, default=evaluator.DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        if args.gpu is None:
            parser.error("worker needs --gpu")
        return work(args.output, args.gpu)
    args.gpus = list(range(8))
    args.poll_seconds = 1.
    args.minimum_free_gib = MIN_FREE_GIB
    queue.EVALUATOR = Path(__file__).resolve()
    queue.register_queue = registration
    queue.eligible = shared_eligible
    queue.coordinator_state = state
    queue.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
