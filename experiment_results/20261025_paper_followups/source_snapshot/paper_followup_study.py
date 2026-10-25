"""Persistent scheduling for the three authorized paper follow-up experiments.

Training can share GPUs under an allocator budget and measured entry headroom.
Efficiency measurements use an empty GPU and are retried if sharing invalidates
them. Existing main-study artifacts and other users' processes are never changed.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import qwen3guard_clean_queue as common

DEFAULT_OUTPUT = ROOT / "outputs/hanguard_paper_followups_20261025"
DEFAULT_PARENT = ROOT / "outputs/hanguard_clean_ablation_20261022"
DEFAULT_MODEL = ROOT / "models/chinese-macbert-base"
ARCHIVE = ROOT / "experiment_results/20261025_paper_followups"
SCHEMA = "hanguard_three_paper_followups_v1"
MODULES = dict(pooling="paper_pooling_controls", external="paper_external_classifier",
               analysis="paper_error_efficiency")
BUDGETS = dict(pooling=dict(cap_gib=4., headroom_gib=4., exclusive=False),
               external=dict(cap_gib=8., headroom_gib=6., exclusive=False),
               analysis=dict(cap_gib=32., headroom_gib=8., exclusive=True))


def module(kind):
    return importlib.import_module("scripts.hanguard." + MODULES[kind])


def script(kind):
    return ROOT / "scripts/hanguard" / (MODULES[kind] + ".py")


def protocol_for(output, kind):
    return Path(output) / kind / "protocol.json"


def initialize(args):
    output, parent = args.output.resolve(), args.parent.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        return verify(output)
    # Child preparation and CPU analysis are run explicitly before root
    # registration, so no training source is still being edited at admission.
    for kind in MODULES:
        if not protocol_for(output, kind).exists():
            raise ValueError(f"Initialize the {kind} child protocol first")
    if not module("analysis").progress(output / "analysis").get("cpu_analysis_complete"):
        raise ValueError("Finish the CPU error analysis before registering GPU followups")
    paths = [Path(__file__).resolve(), *[script(kind) for kind in MODULES],
             Path(common.__file__).resolve()]
    paths += [ROOT / "tests" / name for name in (
        "test_paper_followup_study.py", "test_paper_pooling_controls.py",
        "test_paper_external_classifier.py", "test_paper_error_efficiency.py")]
    protocol = dict(schema=SCHEMA, output=str(output), parent=str(parent),
        parent_protocol_sha256=common.sha(parent / "protocol.json"),
        parent_results_sha256=common.sha(parent / "results.json"),
        children={kind: dict(path=str(protocol_for(output, kind)),
                             sha256=common.sha(protocol_for(output, kind))) for kind in MODULES},
        sources={str(path): common.sha(path) for path in paths},
        budgets=BUDGETS, gpu_ids=list(range(8)), poll_seconds=3.,
        authorization="User requested all three followups and explicitly allowed GPU sharing",
        efficiency="Exclusive GPU only; arrival of another compute process invalidates the attempt",
        shared_training="One followup worker per GPU, process allocator cap plus entry headroom; no foreign signals",
        model=str(args.model.resolve()), created_unix=time.time())
    common.write(output / "protocol.json", protocol)
    snapshots = output / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for filename, digest in protocol["sources"].items():
        target = snapshots / Path(filename).name
        shutil.copyfile(filename, target)
        if common.sha(target) != digest:
            raise ValueError("Followup source changed during registration")
    common.write(output / "protocol_registration.json",
                 dict(protocol_sha256=common.sha(output / "protocol.json")))
    refresh_archive(output)
    return protocol


def verify(output):
    output = Path(output).resolve()
    protocol = common.read(output / "protocol.json")
    if (protocol.get("schema") != SCHEMA or protocol.get("output") != str(output)
            or common.read(output / "protocol_registration.json").get("protocol_sha256") != common.sha(output / "protocol.json")
            or protocol.get("budgets") != BUDGETS):
        raise ValueError("Followup root registration changed")
    common.verify_sources(protocol)
    for record in protocol["children"].values():
        if common.sha(record["path"]) != record["sha256"]:
            raise ValueError("Registered child protocol changed")
    parent = Path(protocol["parent"])
    if (common.sha(parent / "protocol.json") != protocol["parent_protocol_sha256"]
            or common.sha(parent / "results.json") != protocol["parent_results_sha256"]):
        raise ValueError("Completed parent study changed")
    return protocol


def progress(output):
    return {kind: module(kind).progress(Path(output) / kind) for kind in MODULES}


def jobs(status):
    pending = []
    if (status["analysis"].get("cpu_analysis_complete")
            and not status["analysis"].get("benchmark_complete")):
        pending.append(dict(kind="analysis", phase="benchmark"))
    for kind in ("external", "pooling"):
        entries = list(status[kind].get("pending_jobs", []))
        if kind == "pooling":
            entries.sort(key=lambda item: (item.get("task") != "category", item.get("seed", 0)))
        pending += [dict(item, kind=kind) for item in entries]
    return pending


def finalize_children(output, statuses, active):
    """Do not declare success before the per-seed tables have been aggregated."""
    running_kinds = {entry["job"]["kind"] for entry in active.values()}
    for kind in ("pooling", "external"):
        if not statuses[kind].get("pending_jobs") and kind not in running_kinds:
            implementation = module(kind)
            if not implementation.is_complete(Path(output) / kind):
                implementation.summarize(Path(output) / kind)


def job_key(job):
    # Fit and evaluate for the same checkpoint group must never overlap.
    return ":".join(str(job.get(key, "")) for key in ("kind", "task", "seed"))


def eligible(card, job):
    budget = BUDGETS[job["kind"]]
    return (card["free_gib"] >= budget["cap_gib"] + budget["headroom_gib"]
            and (not budget["exclusive"] or not card["pids"]))


def state(output, phase, **details):
    common.write(Path(output) / "queue_state.json", dict(schema=SCHEMA, state=phase,
        automatic_run=phase not in ("complete", "failed", "stopped"),
        coordinator_identity=common.identity(os.getpid()), updated_unix=time.time(), **details))


def configure_numeric(torch, kind):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if kind in ("analysis", "pooling"):
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def worker(args):
    output = args.output.resolve()
    verify(output)
    budget = BUDGETS[args.kind]
    def admitted():
        try:
            cards = common.gpu_inventory()
            card = next(row for row in cards if row["index"] == args.gpu)
        except Exception:
            return None
        if (os.environ.get("CUDA_VISIBLE_DEVICES") != card["uuid"]
                or card["free_gib"] < budget["cap_gib"] + budget["headroom_gib"]):
            return None
        foreign = sorted(set(card["pids"]) - {os.getpid()})
        if budget["exclusive"] and foreign:
            return None
        return card
    card = admitted()
    if card is None:
        return 75
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    configure_numeric(torch, args.kind)
    # Reading device properties initializes the context only after admission.
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = budget["cap_gib"] * 1024**3 / total
    if not 0 < fraction < 1:
        raise ValueError("Allocator budget exceeds physical GPU memory")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    # The first context consumes a little memory, so do not require the full
    # entry headroom again. The target process budget remains unchanged.
    try:
        fresh = next(row for row in common.gpu_inventory() if row["index"] == args.gpu)
    except Exception:
        return 75
    if (fresh["uuid"] != card["uuid"]
            or fresh["free_gib"] < budget["cap_gib"] + budget["headroom_gib"] - 1.):
        return 75
    if budget["exclusive"] and set(fresh["pids"]) - {os.getpid()}:
        return 75
    identity = common.identity(os.getpid())
    child_protocol_sha = common.sha(protocol_for(output, args.kind))
    record = dict(identity, gpu_uuid=card["uuid"], physical_gpu=args.gpu,
        allocator_cap_gib=budget["cap_gib"], exclusive=budget["exclusive"],
        followup_protocol_sha256=child_protocol_sha,
        root_protocol_sha256=common.sha(output / "protocol.json"),
        entry_inventory=card, started_unix=time.time(), kind=args.kind,
        task=args.task, seed=args.seed)
    record_path = output / "admissions" / f"{identity['pid']}_{identity['start_time_ticks']}.json"
    record_path.parent.mkdir(exist_ok=True)
    common.write(record_path, record)
    os.environ["HANGUARD_FOLLOWUP_ADMITTED"] = child_protocol_sha
    os.environ["HANGUARD_FOLLOWUP_ADMISSION_RECORD"] = str(record_path)
    try:
        target = module(args.kind)
        if args.kind == "analysis":
            code = target.benchmark(output / args.kind, gpu=args.gpu)
        elif args.kind == "pooling":
            target.fit(output / args.kind, args.task, args.seed, device="cuda")
            code = 0
        else:
            target.fit(output / args.kind, args.seed, device="cuda")
            code = 0
        record["exit_code"] = int(code or 0)
        return int(code or 0)
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record.update(finished_unix=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3)
        common.write(record_path, record)


def refresh_archive(output):
    output = Path(output).resolve()
    archive = ARCHIVE if output == DEFAULT_OUTPUT.resolve() else output / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    whitelist = ["protocol.json", "protocol_registration.json", "queue_state.json"]
    for kind in MODULES:
        whitelist += [f"{kind}/{name}" for name in ("protocol.json", "registration.json",
            "protocol_registration.json", "results.json", "analysis_results.json", "error_analysis.json", "benchmark_results.json",
            "efficiency_results.json", "efficiency_complete.json", "guard_reference.json", "README.md")]
    whitelist += [f"analysis/{name}" for name in ("metrics_by_seed.csv", "class_metrics_by_seed.csv",
        "confusion_by_seed.csv", "strata_by_seed.csv", "paired_changes_by_seed.csv", "paired_strata_by_seed.csv",
        "seed_summary.csv", "class_summary.csv", "paired_summary.csv", "parameters.csv",
        "strata_summary.csv", "paired_strata_summary.csv", "efficiency.csv")]
    for name in whitelist:
        source = output / name
        if source.exists():
            dest = archive / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    for source in (output / "source_snapshot").glob("*.py"):
        dest = archive / "source_snapshot" / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    statuses = progress(output) if all(protocol_for(output, k).exists() for k in MODULES) else {}
    lines = ["# 论文补充实验（20261025）", "",
        "同一数据划分，复用20261022本轮新特征；不改变已完成的主消融结果。", "",
        "| 项目 | 内容 | 当前进度 |", "|---|---|---|"]
    for kind, label in (("pooling", "全文均值/共享注意力＋普通分类头，二分类与五分类各三seed"),
                        ("external", "Chinese-MacBERT-base，原生两来源五分类三seed"),
                        ("analysis", "分类别错误分析与空卡端到端效率")):
        lines.append(f"| {kind} | {label} | `{json.dumps(statuses.get(kind, {}), ensure_ascii=False)}` |")
    lines += ["", "训练允许有显存余量的共享卡；效率只在无其他计算进程的卡上测量。",
        "进度来自各子协议，未完成项不生成或替换为旧成绩。",
        "", "运行状态：`outputs/hanguard_paper_followups_20261025/queue_state.json`。", ""]
    (archive / "README.md").write_text("\n".join(lines))
    files = {str(path.relative_to(archive)): common.sha(path) for path in sorted(archive.rglob("*"))
             if path.is_file() and path.name != "manifest.json"}
    common.write(archive / "manifest.json", dict(schema=SCHEMA, source=str(output), files=files))


def run(args):
    output = args.output.resolve()
    protocol = verify(output)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    lock_root = ROOT / "outputs/.qwen3guard_gpu_admission"
    lock_root.mkdir(exist_ok=True)
    with (output / ".coordinator.lock").open("a+") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = common.read(output / "queue_state.json") if (output / "queue_state.json").exists() else {}
        if any(common.same_process(item["identity"]) for item in prior.get("active_workers", [])):
            raise RuntimeError("A prior exact followup worker is still alive")
        active, failures, events, archive_at = {}, {}, [], 0.
        try:
            while True:
                verify(output)
                for gpu, entry in list(active.items()):
                    code = entry["process"].poll()
                    if code is None:
                        continue
                    entry["log"].close()
                    entry["gpu_lock"].close()
                    events.append(dict(event="worker_exit", job=entry["job"], gpu=gpu,
                                       exit_code=code, time=time.time()))
                    if code not in (0, 75):
                        failures[job_key(entry["job"])] = dict(exit_code=code, log=entry["log_path"])
                    del active[gpu]
                statuses = progress(output)
                finalize_children(output, statuses, active)
                pending = jobs(statuses)
                actionable = [job for job in pending if job_key(job) not in failures]
                if not actionable and not active:
                    phase = "complete" if not pending and not failures else "failed"
                    state(output, phase, children=statuses, active_workers=[], failures=failures,
                          recent_events=events[-40:])
                    refresh_archive(output)
                    return
                if (output / "STOP").exists() and not active:
                    state(output, "stopped", children=statuses, active_workers=[], failures=failures)
                    refresh_archive(output)
                    return
                try:
                    cards, error = common.gpu_inventory(), None
                except Exception as exc:
                    cards, error = [], repr(exc)
                busy = {job_key(entry["job"]) for entry in active.values()}
                for card in cards:
                    gpu = card["index"]
                    if gpu in active or (output / "STOP").exists():
                        continue
                    job = next((item for item in pending if job_key(item) not in busy
                                and job_key(item) not in failures and eligible(card, item)), None)
                    if job is None:
                        continue
                    device_lock = (lock_root / (card["uuid"] + ".lock")).open("a+")
                    try:
                        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        device_lock.close()
                        continue
                    try:
                        current = next(row for row in common.gpu_inventory() if row["uuid"] == card["uuid"])
                    except Exception:
                        device_lock.close()
                        continue
                    if not eligible(current, job):
                        device_lock.close()
                        continue
                    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                               "--kind", job["kind"], "--gpu", str(gpu)]
                    for key in ("task", "seed"):
                        if key in job:
                            command += ["--" + key, str(job[key])]
                    log_path = logs / (job_key(job).replace(":", "_") + f"_gpu{gpu}_{time.time_ns()}.log")
                    log = log_path.open("a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT)
                    try:
                        owner = common.identity(child.pid)
                        if owner is None:
                            raise RuntimeError("New child identity unavailable")
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                            child.wait(timeout=30)
                        log.close()
                        device_lock.close()
                        raise
                    active[gpu] = dict(process=child, log=log, gpu_lock=device_lock, identity=owner,
                        job=job, log_path=str(log_path), admitted_inventory=current, launched_unix=time.time())
                    busy.add(job_key(job))
                    events.append(dict(event="launch", job=job, gpu=gpu, identity=owner,
                                       inventory=current, time=time.time()))
                workers = [{key: value for key, value in entry.items()
                            if key not in ("process", "log", "gpu_lock")} for entry in active.values()]
                phase = "running" if active else "waiting_for_capacity"
                state(output, phase, children=statuses, active_workers=workers, failures=failures,
                      gpu_inventory=cards, inventory_error=error, recent_events=events[-40:])
                if time.time() - archive_at >= 300:
                    refresh_archive(output)
                    archive_at = time.time()
                time.sleep(protocol["poll_seconds"])
        except BaseException as exc:
            workers = [{key: value for key, value in entry.items()
                        if key not in ("process", "log", "gpu_lock")} for entry in active.values()]
            state(output, "failed", error=repr(exc), active_workers=workers, failures=failures)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "worker", "status", "archive"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--kind", choices=tuple(MODULES))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--task", choices=("binary", "category"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "init":
        initialize(args)
    elif args.action == "run":
        run(args)
    elif args.action == "worker":
        if args.kind is None or args.gpu is None:
            parser.error("worker requires --kind and --gpu")
        return worker(args)
    elif args.action == "archive":
        refresh_archive(args.output)
    else:
        path = args.output / "queue_state.json"
        print(json.dumps(common.read(path) if path.exists() else {"state": "not_started"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
