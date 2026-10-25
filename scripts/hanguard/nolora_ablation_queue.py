"""Run the fixed ablation matrix on empty public GPUs, with a bounded CPU fallback.

Only the small readout heads are trained. Cached backbone states are reused.
GPU admission requires an empty compute-process list; later foreign arrivals
are recorded and do not terminate our job. No foreign process is signalled.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.fragment_clean_lora_study import (  # noqa: E402
    empty_cards, identity, inventory, read, same_process, sha, write,
)

TASKS = ("components", "seed43", "seed44")
POLICY = dict(prelaunch_foreign="block", active_foreign="record_only",
              reserved_gpus=[], minimum_free_gib=4., max_cpu_workers=1,
              cpu_threads=8, gpu_threads=4, never_terminate_foreign=True)


def task_folder(study, task):
    return Path(study) / ("components" if task == "components" else f"seed_{task[4:]}")


def verify_control(study):
    study = Path(study).resolve()
    control = read(study / "queue_protocol.json")
    if read(study / "queue_registration.json")["sha256"] != sha(study / "queue_protocol.json"):
        raise ValueError("Queue registration changed")
    if control["policy"] != POLICY or control["tasks"] != list(TASKS):
        raise ValueError("Queue policy or task matrix changed")
    for filename, digest in control["bindings"].items():
        if sha(filename) != digest:
            raise ValueError(f"Queue source/protocol changed: {filename}")
    return control


def register(study):
    study = Path(study).resolve()
    if (study / "queue_registration.json").exists():
        return verify_control(study)
    paths = [Path(__file__).resolve(), ROOT / "scripts/hanguard/nolora_fusion_ablation.py",
             study / "protocol.json", study / "protocol_registration.json",
             ROOT / "scripts/hanguard/fragment_clean_lora_study.py",
             ROOT / "scripts/hanguard/ab_readout_parallel_queue.py"]
    value = dict(schema="hanguard_nolora_ablation_queue_v1", tasks=list(TASKS),
                 study=str(study), policy=POLICY,
                 bindings={str(p): sha(p) for p in paths}, created_unix=time.time())
    write(study / "queue_protocol.json", value)
    write(study / "queue_registration.json", dict(sha256=sha(study / "queue_protocol.json")))
    return verify_control(study)


def worker(args):
    study = args.output.resolve()
    verify_control(study)
    ledger_path = study / "queue_workers" / f"{args.worker_id}.json"
    deadline = time.monotonic() + 15
    while not ledger_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Coordinator did not register this worker")
        time.sleep(.05)
    ledger = read(ledger_path)
    if (ledger["task"] != args.task or ledger["device"] != args.device or
            ledger["gpu"] != args.gpu or not same_process(ledger["identity"], identity(os.getpid()))):
        raise ValueError("Exact task/worker identity mismatch")
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            raise ValueError("Unexpected visible GPU")
        cards = inventory()
        row = next(r for r in cards if r["index"] == args.gpu)
        if any(pid != os.getpid() for pid in row["pids"]):
            return 75
        if row["free_gib"] < POLICY["minimum_free_gib"]:
            return 75
        import torch
        torch.cuda.init()
        readiness = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del readiness
    write(ledger_path.with_suffix(".admitted.json"), dict(
        admitted=True, task=args.task, device=args.device, gpu=args.gpu,
        identity=ledger["identity"], unix=time.time()))
    from scripts.hanguard import nolora_fusion_ablation as experiment
    experiment.main(["worker", "--output", str(study), "--task", args.task,
                     "--device", args.device, "--cpu-threads",
                     str(POLICY["gpu_threads"] if args.device == "cuda" else POLICY["cpu_threads"])])
    result = task_folder(study, args.task) / "analysis/results.json"
    if not result.is_file() or read(result).get("complete") is not True:
        raise RuntimeError("Task returned without complete result")
    write(ledger_path.with_suffix(".complete.json"), dict(
        complete=True, result=str(result), sha256=sha(result), unix=time.time()))
    return 0


def completed(study, task):
    path = task_folder(study, task) / "analysis/results.json"
    return path.exists() and read(path).get("complete") is True


def run(study):
    study = Path(study).resolve()
    verify_control(study)
    directory = study / "queue_workers"
    directory.mkdir(exist_ok=True)
    active, arrivals = {}, set()
    with (study / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Never duplicate an earlier exact live worker after coordinator loss.
        for path in directory.glob("*.json"):
            if path.name.count(".") != 1:
                continue
            prior = read(path)
            try:
                actual = identity(prior["identity"]["pid"])
            except (OSError, ValueError):
                continue
            if actual.get("start_time_ticks") == prior["identity"]["start_time_ticks"]:
                raise RuntimeError(f"Prior worker is still alive: {actual['pid']}")

        def launch(task, device, gpu=None):
            worker_id = f"{task}_{device}{gpu if gpu is not None else ''}_{time.time_ns()}"
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(study),
                       "--task", task, "--device", device, "--worker-id", worker_id]
            if gpu is not None:
                command += ["--gpu", str(gpu)]
            threads = POLICY["gpu_threads"] if device == "cuda" else POLICY["cpu_threads"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            with (directory / f"{worker_id}.log").open("x") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
            exact = None
            for _ in range(80):
                try:
                    current = identity(child.pid)
                    if current["uid"] == os.getuid() and current["argv"] == command and current["cwd"] == str(ROOT):
                        exact = current
                        break
                except (OSError, ValueError):
                    pass
                if child.poll() is not None:
                    break
                time.sleep(.05)
            if exact is None:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=20)
                raise RuntimeError("Failed to bind a new worker identity")
            ledger = dict(worker_id=worker_id, task=task, device=device, gpu=gpu,
                          identity=exact, command=command, unix=time.time())
            write(directory / f"{worker_id}.json", ledger)
            active[worker_id] = dict(ledger, child=child)

        try:
            while True:
                if (study / "STOP").exists():
                    raise RuntimeError("Study STOP requested")
                for worker_id, record in list(active.items()):
                    code = record["child"].poll()
                    if code is None:
                        continue
                    write(directory / f"{worker_id}.exit.json", dict(exit_code=code, unix=time.time()))
                    del active[worker_id]
                    if code == 75 and not (directory / f"{worker_id}.admitted.json").exists():
                        continue
                    if code != 0 or not (directory / f"{worker_id}.complete.json").exists():
                        raise RuntimeError(f"{worker_id} failed with exit {code}; inspect its log")
                done = [task for task in TASKS if completed(study, task)]
                if len(done) == len(TASKS) and not active:
                    from scripts.hanguard import nolora_fusion_ablation as experiment
                    experiment.main(["summarize", "--output", str(study)])
                    write(study / "queue_state.json", dict(state="completed", done=done,
                        active_workers=[], automatic_run=False, unix=time.time()))
                    packager = ROOT / "scripts/hanguard/package_fusion_ablation_results.py"
                    subprocess.run([sys.executable, str(packager)], cwd=ROOT, check=True)
                    subprocess.run([sys.executable, str(packager), "--verify"], cwd=ROOT, check=True)
                    return
                telemetry_error = None
                try:
                    cards = inventory()
                except Exception as error:
                    cards, telemetry_error = [], repr(error)
                for record in active.values():
                    if record["gpu"] is None:
                        continue
                    row = next((row for row in cards if row["index"] == record["gpu"]), None)
                    for pid in row["pids"] if row else []:
                        key = (record["worker_id"], pid)
                        if pid != record["identity"]["pid"] and key not in arrivals:
                            arrivals.add(key)
                            write(study / "queue_foreign_arrivals" / f"{time.time_ns()}_{pid}.json",
                                  dict(action="record_only_continue_owned", gpu=record["gpu"], pid=pid,
                                       worker_id=record["worker_id"], unix=time.time()))
                pending = [task for task in TASKS if task not in done and
                           task not in [r["task"] for r in active.values()]]
                occupied = {r["gpu"] for r in active.values()}
                for gpu in empty_cards(cards, list(range(8)), POLICY["minimum_free_gib"]):
                    if pending and gpu not in occupied:
                        launch(pending.pop(0), "cuda", gpu)
                if pending and sum(r["device"] == "cpu" for r in active.values()) < POLICY["max_cpu_workers"]:
                    launch(pending.pop(0), "cpu")
                write(study / "queue_state.json", dict(state="running", done=done, pending=pending,
                    active_workers=[{k: v for k, v in r.items() if k != "child"} for r in active.values()],
                    automatic_run=True, gpu_inventory=cards, gpu_inventory_error=telemetry_error,
                    coordinator_pid=os.getpid(), unix=time.time()))
                time.sleep(3)
        except BaseException as error:
            # These Popen children remain unreaped and cannot have reused PIDs.
            for record in active.values():
                if record["child"].poll() is None:
                    record["child"].terminate()
            for record in active.values():
                try:
                    record["child"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    record["child"].kill()
                    record["child"].wait(timeout=10)
            write(study / "queue_state.json", dict(state="requires_review", reason=repr(error),
                automatic_run=False, active_workers=[], unix=time.time()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker-id")
    args = parser.parse_args(argv)
    if args.action == "register":
        register(args.output)
    elif args.action == "run":
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        run(args.output)
    else:
        if not args.task or not args.device or not args.worker_id:
            raise ValueError("worker needs task/device/worker-id")
        raise SystemExit(worker(args))


if __name__ == "__main__":
    main()
