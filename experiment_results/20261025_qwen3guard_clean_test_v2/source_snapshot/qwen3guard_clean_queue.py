"""Admit fresh Qwen3Guard evaluation workers only to genuinely empty GPUs.

The coordinator is CPU-only. It never signals an unrelated process and retains
its admitted workers if somebody subsequently joins their GPU. All predictions
and scientific settings belong to qwen3guard_clean_eval.py's fresh protocol.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs/qwen3guard_clean_test_20261025_v2"
EVALUATOR = ROOT / "scripts/hanguard/qwen3guard_clean_eval.py"
SCHEMA = "qwen3guard_clean_empty_gpu_queue_v1"
SYSTEM_DISPLAY = dict(pid=9539, uid=126, start_time_ticks=10027,
    command_sha256="5c4df965a58cfe2ca87303f3e0abf46b4fb972dbe2aa730f8e4e56417a5e16d1")


def verified_system_display(process):
    """Ignore only the independently verified system display's exact G birth."""
    if (process.findtext("pid") != str(SYSTEM_DISPLAY["pid"])
            or process.findtext("type") != "G"
            or process.findtext("process_name") != "/usr/lib/xorg/Xorg"):
        return False
    try:
        return identity(SYSTEM_DISPLAY["pid"]) == SYSTEM_DISPLAY
    except (OSError, ValueError, IndexError):
        return False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid):
    folder = Path("/proc") / str(int(pid))
    try:
        raw = (folder / "cmdline").read_bytes()
        stat = (folder / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] in ("Z", "X", "x"):
            return None
        return dict(pid=int(pid), uid=folder.stat().st_uid,
                    start_time_ticks=int(fields[19]),
                    command_sha256=hashlib.sha256(raw).hexdigest())
    except FileNotFoundError:
        return None


def same_process(expected):
    try:
        return identity(expected["pid"]) == expected
    except (OSError, ValueError, IndexError):
        # Unknown identity is not proof that a process has exited.
        return True


def gpu_inventory():
    """Union compute and full NVIDIA PID inventories; unknown telemetry blocks."""
    def call(arguments):
        return subprocess.run(["nvidia-smi", *arguments], check=True,
                              capture_output=True, text=True, timeout=15).stdout
    cards = {}
    for line in call(["--query-gpu=index,uuid,memory.free,utilization.gpu",
                      "--format=csv,noheader,nounits"]).splitlines():
        index, uuid, free, util = [x.strip() for x in line.split(",")]
        if not uuid.startswith("GPU-") or uuid in cards:
            raise ValueError("Unverified or duplicate NVIDIA GPU UUID")
        cards[uuid] = dict(index=int(index), uuid=uuid, free_gib=float(free) / 1024,
                           utilization=int(util), pids=set())
    if not cards or len({x["index"] for x in cards.values()}) != len(cards):
        raise ValueError("Missing or duplicate NVIDIA GPU index")
    seen = set()
    for node in ET.fromstring(call(["-q", "-x"])).findall("gpu"):
        uuid = node.findtext("uuid")
        process_node = node.find("processes")
        if (uuid not in cards or uuid in seen or process_node is None
                or (process_node.text or "").strip()):
            raise ValueError("Incomplete NVIDIA full-process inventory")
        seen.add(uuid)
        for item in node.findall("processes/process_info"):
            pid = int(item.findtext("pid", "-1"))
            if pid <= 0:
                raise ValueError("Unknown NVIDIA process PID")
            if verified_system_display(item):
                cards[uuid].setdefault("ignored_system_display_pids", []).append(pid)
            else:
                cards[uuid]["pids"].add(pid)
    if seen != set(cards):
        raise ValueError("A GPU is missing from NVIDIA full-process inventory")
    for line in call(["--query-compute-apps=gpu_uuid,pid",
                      "--format=csv,noheader,nounits"]).splitlines():
        if not line.strip():
            continue
        uuid, raw_pid = [x.strip() for x in line.split(",")]
        pid = int(raw_pid)
        if uuid not in cards or pid <= 0:
            raise ValueError("Unverified NVIDIA compute-process identity")
        cards[uuid]["pids"].add(pid)
    return [dict(card, pids=sorted(card["pids"]))
            for card in sorted(cards.values(), key=lambda row: row["index"])]


def eligible(card, minimum_free_gib):
    return not card["pids"] and card["free_gib"] >= minimum_free_gib


def coordinator_state(output, state, **details):
    write(output / "queue_state.json", dict(schema=SCHEMA, state=state,
          automatic_run=state not in ("completed", "failed", "stopped"),
          coordinator_identity=identity(os.getpid()), updated_unix=time.time(), **details))


def register_queue(output, args):
    registration = dict(schema=SCHEMA, protocol_sha256=sha(output / "protocol.json"),
                        sources={str(Path(__file__).resolve()): sha(__file__),
                                 str(EVALUATOR): sha(EVALUATOR)},
                        gpu_ids=args.gpus, poll_seconds=args.poll_seconds,
                        minimum_free_gib=args.minimum_free_gib,
                        admission="No NVIDIA process of any owner at admission; recheck in worker",
                        system_display_exception=SYSTEM_DISPLAY,
                        post_admission="Record arrivals and retain our admitted worker")
    path = output / "queue_registration.json"
    if path.exists():
        if read(path) != registration:
            raise ValueError("Frozen queue protocol or source changed")
    else:
        write(path, registration)
        snapshots = output / "queue_source_snapshot"
        snapshots.mkdir(exist_ok=True)
        for filename, digest in registration["sources"].items():
            dest = snapshots / Path(filename).name
            dest.write_bytes(Path(filename).read_bytes())
            if sha(dest) != digest:
                raise ValueError("Queue source changed during registration")
    return registration


def verify_sources(registration):
    if any(sha(filename) != digest for filename, digest in registration["sources"].items()):
        raise ValueError("Frozen evaluation source changed while queue was running")


def pending_work(progress):
    return max(0, int(progress["total_chunks"]) - int(progress["complete_chunks"]))


def run(args):
    output = args.output.resolve()
    if not (output / "protocol.json").exists():
        raise ValueError("Initialize the new evaluator protocol before launching the queue")
    from scripts.hanguard import qwen3guard_clean_eval as evaluator
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".coordinator.lock").open("a+") as coordinator_lock:
        try:
            fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another coordinator already holds this evaluation")
        registration = register_queue(output, args)
        # A prior coordinator can be gone while one of its workers remains alive.
        prior = read(output / "queue_state.json") if (output / "queue_state.json").exists() else {}
        if any(same_process(x["identity"]) for x in prior.get("active_workers", [])):
            raise RuntimeError("A prior exact worker is still alive; refuse duplicate coordinator")
        lock_dir = ROOT / "outputs/.qwen3guard_gpu_admission"
        lock_dir.mkdir(exist_ok=True)
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        active, failures, events = {}, {}, []
        try:
            while True:
                verify_sources(registration)
                for gpu, child in list(active.items()):
                    code = child["process"].poll()
                    if code is None:
                        continue
                    child["log"].close()
                    child["gpu_lock"].close()
                    events.append(dict(event="worker_exit", gpu=gpu, exit_code=code,
                                       identity=child["identity"], time=time.time()))
                    del active[gpu]
                    if code not in (0, 75):
                        failures[gpu] = failures.get(gpu, 0) + 1
                progress = evaluator.progress(output)
                if evaluator.is_complete(output) and not active:
                    coordinator_state(output, "completed", progress=progress,
                                      active_workers=[], recent_events=events[-40:])
                    return
                if not active and pending_work(progress) == 0:
                    result = evaluator.summarize(output)
                    if not result.get("complete"):
                        raise RuntimeError("Inference finished with invalid Safety outputs; review saved raw outputs")
                    continue
                if (output / "STOP").exists():
                    if not active:
                        coordinator_state(output, "stopped", progress=progress, active_workers=[])
                        return
                    cards, error = [], "STOP requested; waiting for owned workers to finish"
                else:
                    try:
                        cards, error = gpu_inventory(), None
                    except (OSError, ValueError, TypeError, ET.ParseError,
                            subprocess.SubprocessError) as exc:
                        cards, error = [], repr(exc)
                for card in cards:
                    gpu = card["index"]
                    if gpu in active:
                        foreign = sorted(set(card["pids"]) - {active[gpu]["identity"]["pid"]})
                        if foreign != active[gpu].get("observed_foreign", []):
                            events.append(dict(event="post_admission_foreign_pids", gpu=gpu,
                                               pids=foreign, action="continue_owned_worker", time=time.time()))
                            active[gpu]["observed_foreign"] = foreign
                        continue
                    if (gpu not in args.gpus or failures.get(gpu, 0) >= 3
                            or not eligible(card, args.minimum_free_gib)
                            or len(active) >= pending_work(progress)):
                        continue
                    gpu_lock = (lock_dir / (card["uuid"] + ".lock")).open("a+")
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        gpu_lock.close()
                        continue
                    # Recheck after acquiring the cooperative lock; the worker
                    # rechecks again before it imports torch or allocates CUDA.
                    try:
                        current = next(x for x in gpu_inventory() if x["uuid"] == card["uuid"])
                    except (OSError, ValueError, StopIteration, ET.ParseError,
                            subprocess.SubprocessError):
                        gpu_lock.close()
                        continue
                    if not eligible(current, args.minimum_free_gib):
                        gpu_lock.close()
                        continue
                    name = f"gpu{gpu}_{time.time_ns()}"
                    log = (logs / (name + ".log")).open("a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card["uuid"],
                               CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4",
                               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    command = [sys.executable, str(EVALUATOR), "worker", "--output", str(output),
                               "--gpu", str(gpu)]
                    process = subprocess.Popen(command, cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    born = None
                    try:
                        for _ in range(20):
                            born = identity(process.pid)
                            if born is not None or process.poll() is not None:
                                break
                            time.sleep(.01)
                    except (OSError, ValueError, IndexError):
                        # This is our just-created, unreaped direct child, so its
                        # PID cannot have been reused for somebody else's job.
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                        log.close()
                        gpu_lock.close()
                        raise
                    if born is None:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                        log.close()
                        gpu_lock.close()
                        failures[gpu] = failures.get(gpu, 0) + 1
                        continue
                    active[gpu] = dict(process=process, identity=born, log=log,
                                       gpu_lock=gpu_lock, gpu=gpu, command=command,
                                       admitted_unix=time.time(), observed_foreign=[])
                    events.append(dict(event="worker_launch", gpu=gpu, uuid=card["uuid"],
                                       identity=born, empty_inventory=current, time=time.time()))
                workers = [{k: v for k, v in row.items() if k not in ("process", "log", "gpu_lock")}
                           for row in active.values()]
                state = "evaluating" if active else "waiting_for_empty_gpu"
                coordinator_state(output, state, progress=progress, active_workers=workers,
                                  gpu_inventory=cards, inventory_error=error,
                                  failures_by_gpu=failures, recent_events=events[-40:])
                time.sleep(args.poll_seconds)
        except BaseException as exc:
            # Workers own immutable chunk claims and can safely finish them.
            # Do not signal them, and never signal anybody else's process.
            workers = [{k: v for k, v in row.items() if k not in ("process", "log", "gpu_lock")}
                       for row in active.values()]
            coordinator_state(output, "failed", error=repr(exc), active_workers=workers,
                              recent_events=events[-40:])
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "status"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--poll-seconds", type=float, default=5.)
    p.add_argument("--minimum-free-gib", type=float, default=20.)
    args = p.parse_args()
    if not 0 < args.poll_seconds <= 30 or args.minimum_free_gib <= 0:
        p.error("Positive memory and polling interval in (0,30] required")
    if len(args.gpus) != len(set(args.gpus)) or not args.gpus or min(args.gpus) < 0:
        p.error("Unique nonnegative GPU indices required")
    if args.action == "status":
        path = args.output / "queue_state.json"
        print(json.dumps(read(path) if path.exists() else {"state": "not_started"}, ensure_ascii=False))
    else:
        run(args)


if __name__ == "__main__":
    main()
