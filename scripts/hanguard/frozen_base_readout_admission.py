"""CPU admission and live monitoring around the immutable single-card guardian.

Known GPU6 pipelines retain their claim while their registered process birth is
alive. No CUDA is imported here, no foreign process is signalled, and an already
attempted study is only monitored; its GPU worker is never launched again.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import frozen_base_readout_queue as queue

SCHEMA = "hanguard_frozen_base_cpu_admission_v1"
CANDIDATES = (2, 4, 5, 7, 3, 0, 1, 6)
GPU6_BLOCKERS = ((1531157, 1047), (1541914, 1002), (1162730, 1029))


def process_birth(pid, proc_root=Path("/proc")):
    """Read identity without requiring access to another user's cwd/cmdline."""
    directory = Path(proc_root) / str(int(pid))
    try:
        text = (directory / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return dict(pid=int(pid), uid=directory.stat().st_uid, state=fields[0],
                    start_time_ticks=int(fields[19]))
    except FileNotFoundError:
        # A disappearing process is gone; other read failures remain unknown.
        return None


def capture_blockers(snapshot_reader=process_birth):
    records = []
    for pid, expected_uid in GPU6_BLOCKERS:
        record = dict(pid=pid, expected_uid=expected_uid, gpu=6)
        try:
            snapshot = snapshot_reader(pid)
            if snapshot is None:
                record.update(registration_status="absent", uid=None, start_time_ticks=None)
            else:
                record.update(uid=snapshot["uid"], start_time_ticks=snapshot["start_time_ticks"],
                    registration_state=snapshot["state"],
                    registration_status=("verified" if snapshot["uid"] == expected_uid else
                                         "unexpected_owner"))
        except (OSError, ValueError, KeyError, IndexError) as exc:
            record.update(registration_status="unverified", uid=None, start_time_ticks=None,
                          error_type=type(exc).__name__)
        records.append(record)
    return records


def blocker_status(record, snapshot_reader=process_birth):
    result = dict(record)
    if record["registration_status"] == "absent":
        return dict(result, blocked=False, birth_status="absent_at_registration")
    try:
        snapshot = snapshot_reader(record["pid"])
        if snapshot is None:
            return dict(result, blocked=False, birth_status="exited", current=None)
        result["current"] = snapshot
        if snapshot["state"] in ("Z", "X", "x"):
            return dict(result, blocked=False, birth_status="zombie_or_dead")
        if record.get("start_time_ticks") is None:
            return dict(result, blocked=True, birth_status="unverified_birth_alive")
        if snapshot["start_time_ticks"] != record["start_time_ticks"]:
            return dict(result, blocked=False, birth_status="pid_reused_original_exited")
        if snapshot["uid"] != record["uid"]:
            return dict(result, blocked=True, birth_status="same_birth_owner_changed_unknown")
        return dict(result, blocked=True, birth_status="registered_birth_alive")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return dict(result, blocked=True, birth_status="unknown", error_type=type(exc).__name__)


def full_gpu_inventory():
    """Include graphics and other NVIDIA process types as well as compute PIDs."""
    rows = queue.guardian.inventory_module.gpu_inventory()
    completed = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, text=True,
                               capture_output=True, timeout=10)
    root = ET.fromstring(completed.stdout)
    by_uuid = {row["uuid"]: row for row in rows}
    nodes = root.findall("gpu")
    if not nodes:
        raise ValueError("NVIDIA inventory did not include GPU process information")
    for node in nodes:
        uuid = node.findtext("uuid")
        if uuid not in by_uuid or node.find("processes") is None:
            raise ValueError("NVIDIA process inventory is incomplete or has an unknown UUID")
        pids = set(by_uuid[uuid]["pids"])
        for info in node.findall("processes/process_info"):
            pids.add(int(info.findtext("pid")))
        by_uuid[uuid]["pids"] = sorted(pids)
    if {node.findtext("uuid") for node in nodes} != set(by_uuid):
        raise ValueError("NVIDIA process inventory omitted a GPU")
    return rows


def observations(inventory, records, snapshot_reader):
    births = [blocker_status(record, snapshot_reader) for record in records]
    try:
        rows = inventory()
        by_index = {row["index"]: row for row in rows}
        candidates = []
        for index in CANDIDATES:
            row = by_index.get(index)
            if row is None:
                candidates.append(dict(index=index, pids=[], available=False, eligible=False,
                                       reason="GPU unavailable"))
                continue
            pids = sorted(set(int(pid) for pid in row["pids"]))
            pipeline = index == 6 and any(value["blocked"] for value in births)
            candidates.append(dict(row, pids=pids, available=True,
                completely_idle=not pids, pipeline_blocked=pipeline,
                eligible=not pids and not pipeline and row["free_gib"] >= 4.0))
        return candidates, births, None
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, ET.ParseError) as exc:
        candidates = [dict(index=index, pids=[], available=False, eligible=False,
                           reason="Unverified NVIDIA inventory") for index in CANDIDATES]
        return candidates, births, dict(error_type=type(exc).__name__, reason=str(exc))


def _check_stop(study):
    for directory in (study, study / queue.ATTEMPT, study / queue.ATTEMPT / "execution"):
        stop = directory / "STOP"
        if stop.exists():
            raise queue.guardian.legacy.StopRequested(stop.read_text().strip() or "STOP requested")


def _exclusive_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def admission_protocol(study, snapshot_reader):
    """Create once under flock, or verify the original admission registration."""
    protocol_path, registration_path = study / "admission_protocol.json", study / "admission_registration.json"
    if protocol_path.exists():
        registration = queue.guardian.legacy.read_json(registration_path)
        if registration.get("protocol_sha256") != queue.guardian.legacy.sha(protocol_path):
            raise ValueError("Immutable admission protocol changed")
        protocol = queue.guardian.legacy.read_json(protocol_path)
        if protocol.get("schema") != SCHEMA or protocol.get("study") != str(study):
            raise ValueError("Admission protocol belongs to a different study")
        return protocol
    if registration_path.exists():
        raise RuntimeError("Admission registration exists without its protocol")
    if (study / queue.ATTEMPT).exists() or (study / "resource_plan.json").exists():
        raise RuntimeError("Register admission on a fresh study; existing attempts cannot be retried")
    _check_stop(study)
    scientific, root_sha, inputs, sources = queue._root_identity(study)
    bindings = dict(sources)
    for path in (Path(__file__), Path(queue.__file__), Path(queue.guardian.__file__),
                 Path(queue.guardian.legacy.__file__), Path(queue.guardian.inventory_module.__file__),
                 study / "protocol.json", study / "protocol_registration.json"):
        filename, digest = str(path.resolve()), queue.guardian.legacy.sha(path, (study,))
        if filename in bindings and bindings[filename] != digest:
            raise ValueError(f"Registered source differs: {filename}")
        bindings[filename] = digest
    for filename, digest in bindings.items():
        if queue.guardian.legacy.sha(filename, (study,)) != digest:
            raise ValueError(f"Registered source differs: {filename}")
    protocol = dict(schema=SCHEMA, study=str(study), root_protocol_sha256=root_sha,
        scientific_scope=scientific.get("evaluation_scope", scientific.get("scope")),
        bindings=bindings, input_identities=queue._input_metadata(inputs),
        gpu6_pipeline_births=capture_blockers(snapshot_reader), candidate_gpus=list(CANDIDATES),
        max_cuda_workers=1, admission_seconds=0, automatic_retry=False,
        never_terminate_foreign_processes=True, created_unix=time.time())
    _check_stop(study)
    _exclusive_json(protocol_path, protocol)
    _exclusive_json(registration_path, dict(protocol_sha256=queue.guardian.legacy.sha(protocol_path)))
    return protocol


def check_bindings(study, protocol):
    _check_stop(study)
    registered = queue.guardian.legacy.read_json(study / "admission_registration.json")
    if registered.get("protocol_sha256") != queue.guardian.legacy.sha(study / "admission_protocol.json"):
        raise ValueError("Immutable admission protocol changed")
    for filename, digest in protocol["bindings"].items():
        if queue.guardian.legacy.sha(filename, (study,)) != digest:
            raise ValueError(f"Frozen admission binding changed: {filename}")
    for filename, expected in protocol["input_identities"].items():
        stat = Path(filename).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Registered scientific input metadata changed: {filename}")
    _check_stop(study)


def _sleep(study, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _check_stop(study)
        time.sleep(min(.2, max(0, deadline - time.monotonic())))


def _queue_status(study):
    path = study / queue.ATTEMPT / "execution/queue_state.json"
    try:
        return queue.guardian.legacy.read_json(path) if path.exists() else None
    except (OSError, ValueError):
        return None


def wait_and_run(study, *, inventory=None, snapshot_reader=None, poll_seconds=5.0,
                 initialize=None, run=None, sleeper=None):
    """Launch at most once; after a resource stop keep CPU-only status alive."""
    if not 0 < poll_seconds <= 5:
        raise ValueError("CPU admission polling must be positive and at most five seconds")
    inventory = inventory or full_gpu_inventory
    snapshot_reader = snapshot_reader or process_birth
    initialize, run, sleeper = initialize or queue.initialize, run or queue.run, sleeper or _sleep
    study = Path(study).resolve()
    status_stream = sys.stdout
    with (study / "admission_waiter.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another CPU admission monitor owns this study") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        protocol = admission_protocol(study, snapshot_reader)
        monitor_birth = process_birth(os.getpid())
        previous_path = study / "admission_state.json"
        previous = queue.guardian.legacy.read_json(previous_path) if previous_path.exists() else {}
        attempted = ((study / queue.ATTEMPT).exists() or (study / "resource_plan.json").exists()
                     or previous.get("launch_attempted") is True)
        requires_review = attempted
        last_error = previous.get("execution_error")
        publish_lock = threading.Lock()

        def publish(state, *, monitor_live=True, **details):
            candidates, births, observation_error = observations(inventory,
                protocol["gpu6_pipeline_births"], snapshot_reader)
            value = dict(schema=SCHEMA, study=str(study), state=state,
                monitor_pid=os.getpid(), monitor_process_birth=monitor_birth, monitor_live=monitor_live,
                updated_unix=time.time(), updated_utc=datetime.now(timezone.utc).isoformat(),
                candidate_observations=candidates, gpu6_pipeline_birth_status=births,
                inventory_error=observation_error, cuda_loaded_by_admission=False,
                launch_attempted=attempted, automatic_retry=False, requires_review=requires_review,
                automatic_training_launch_disabled=attempted or requires_review,
                max_cuda_workers=1, foreign_processes_signalled=False,
                execution_error=last_error, guardian_state=_queue_status(study),
                **queue.scientific_scope(study), **details)
            with publish_lock:
                queue.guardian.legacy.dump(study / "admission_state.json", value)
                summary = {"state": state, "utc": value["updated_utc"], "monitor_pid": os.getpid(),
                    "eligible_gpus": [row["index"] for row in candidates if row["eligible"]],
                    "requires_review": requires_review, "launch_attempted": attempted}
                print(json.dumps(summary), file=status_stream, flush=True)
            return candidates

        try:
            while True:
                check_bindings(study, protocol)
                candidates = publish("CPU_monitor_requires_review" if requires_review else
                                     "CPU_wait_for_empty_candidate")
                if requires_review:
                    sleeper(study, poll_seconds)
                    continue
                chosen = next((row["index"] for row in candidates if row["eligible"]), None)
                if chosen is None:
                    sleeper(study, poll_seconds)
                    continue
                # Close the CPU observation/registration gap without any added
                # delay. The unchanged guardian independently rechecks launch.
                check_bindings(study, protocol)
                final, _, error = observations(inventory, protocol["gpu6_pipeline_births"], snapshot_reader)
                if error or not any(row["index"] == chosen and row["eligible"] for row in final):
                    continue
                if (study / queue.ATTEMPT).exists() or (study / "resource_plan.json").exists():
                    attempted, requires_review = True, True
                    last_error = dict(error_type="ExistingAttempt", reason="No automatic retry")
                    continue
                attempted = True
                publish("single_launch_committed", physical_gpu=chosen)
                observer_done = threading.Event()
                observer = None
                try:
                    initialize(study, gpu=chosen, reserved_gpu=2 if chosen == 4 else 4)
                    def observe_running():
                        while not observer_done.wait(poll_seconds):
                            publish("guardian_running", physical_gpu=chosen)
                    observer = threading.Thread(target=observe_running,
                                                name="CPU-admission-status", daemon=True)
                    observer.start()
                    # Preserve the guardian's detailed record without printing
                    # its full source/input digest dictionaries every heartbeat.
                    with (study / "admission_guardian.log").open("a") as log:
                        with contextlib.redirect_stdout(log):
                            result = run(study)
                except (Exception, KeyboardInterrupt) as exc:
                    last_error = dict(error_type=type(exc).__name__, reason=str(exc))
                    requires_review = True
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                finally:
                    observer_done.set()
                    if observer is not None:
                        # NVIDIA subprocess calls have finite timeouts. Join
                        # fully so a slow observation cannot overwrite the
                        # final completed/stopped heartbeat after this return.
                        observer.join()
                if requires_review:
                    _check_stop(study)
                    publish("CPU_monitor_requires_review", physical_gpu=chosen)
                    continue
                publish("completed", monitor_live=False, physical_gpu=chosen)
                return result
        except (Exception, KeyboardInterrupt) as exc:
            kind = "stopped" if isinstance(exc, queue.guardian.legacy.StopRequested) else "failed"
            last_error = dict(error_type=type(exc).__name__, reason=str(exc))
            requires_review = attempted
            publish(kind, monitor_live=False)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("wait-and-run",))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    wait_and_run(args.study)


if __name__ == "__main__":
    main()
