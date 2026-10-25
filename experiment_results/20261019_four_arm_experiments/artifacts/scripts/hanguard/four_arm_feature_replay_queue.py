"""Wait for an empty GPU to diagnose a frozen-feature mismatch.

This recovery companion never changes the registered experiment. An exact-only
diagnostic and an independent audit of every completed row must both pass before
the original experiment command can resume. No tolerance changes are permitted.
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
from scripts.hanguard.fragment_clean_lora_study import (
    empty_cards, identity, inventory, read, same_process, sha, write,
)

SCHEMA = "four_arm_exact_feature_replay_recovery_v1"
DIAGNOSTIC = ROOT / "scripts/hanguard/four_arm_feature_mismatch_diagnostic.py"
STUDY_SCRIPT = ROOT / "scripts/hanguard/four_arm_experiment_study.py"


def initialize(study, output, audit):
    from scripts.hanguard.four_arm_experiment_study import verify, verify_retired
    study, output, audit = (Path(p).resolve() for p in (study, output, audit))
    p = verify(study)
    verify_retired(study)
    if output.is_relative_to(study) or study.is_relative_to(output):
        raise ValueError("Keep diagnostic evidence separate from the frozen study")
    if (output / "registration.json").exists():
        raise FileExistsError("Recovery queue already registered")
    snapshot = read(audit)
    if snapshot["original_study_protocol_sha256"] != sha(study / "protocol.json"):
        raise ValueError("Completed-row audit belongs to a different experiment")
    sources = [Path(__file__).resolve(), DIAGNOSTIC,
               ROOT / "tests/test_four_arm_feature_replay_queue.py",
               ROOT / "tests/test_four_arm_feature_mismatch_diagnostic.py"]
    value = dict(schema=SCHEMA, study=str(study), output=str(output),
        original_study_protocol_sha256=sha(study / "protocol.json"),
        audit_path=str(audit), audit_sha256=sha(audit),
        sources={str(f): sha(f) for f in sources},
        runtime=p["runtime"], minimum_free_gib=20., poll_seconds=3.,
        policy=dict(prelaunch_foreign="block", active_foreign="continue_owned_worker",
                    automatic_resume="exact diagnostic AND all completed rows match parent",
                    scientific_source_changes=False, tolerance_changes=False),
        registered_unix=time.time())
    write(output / "protocol.json", value)
    write(output / "registration.json", dict(protocol_sha256=sha(output / "protocol.json")))
    for name in value["sources"]:
        source = Path(name)
        target = output / "source_snapshot" / str(source.relative_to(ROOT)).replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write(output / "state.json", dict(state="registered", automatic_run=False,
                                      updated_unix=time.time()))
    return value


def verify(output):
    from scripts.hanguard.four_arm_experiment_study import verify as verify_study
    output = Path(output).resolve()
    p = read(output / "protocol.json")
    if (p["schema"] != SCHEMA or p["output"] != str(output) or
            read(output / "registration.json")["protocol_sha256"] != sha(output / "protocol.json")):
        raise ValueError("Recovery registration changed")
    for name, digest in p["sources"].items():
        if sha(name) != digest:
            raise ValueError(f"Recovery source changed: {name}")
    if sha(p["audit_path"]) != p["audit_sha256"]:
        raise ValueError("Completed-row audit changed")
    study = Path(p["study"])
    if sha(study / "protocol.json") != p["original_study_protocol_sha256"]:
        raise ValueError("Original study registration changed")
    verify_study(study)
    return p


def audit_allows_resume(p):
    a = read(p["audit_path"])
    return bool(a.get("complete") is True and a.get("passed") is True and
        a.get("checked_rows", 0) == a.get("expected_completed_rows", -1) and
        a.get("different_rows") == 0 and a.get("different_elements") == 0 and
        a.get("original_study_protocol_sha256") == p["original_study_protocol_sha256"])


def diagnostic_allows_resume(p, result, admitted, record):
    """Accept only the exact admitted child and its completed strict replay."""
    if not (result.get("complete") is True and result.get("passed") is True
            and result.get("study") == p["study"] and result.get("split") == "train"
            and result.get("diagnostic_only") is True
            and result.get("production_cache_written") is False
            and result.get("test_payloads_read") is False
            and result.get("zero_tolerance_unchanged") is True
            and admitted.get("admitted") is True):
        return False
    for evidence in (result, admitted):
        if (evidence.get("schema") != "hanguard_four_arm_feature_mismatch_diagnostic_v1"
                or evidence.get("study") != p["study"]
                or evidence.get("output") != record["output"]
                or evidence.get("gpu") != record["gpu"]
                or not same_process(record["identity"], evidence.get("identity", {}))
                or evidence.get("run_id") != record["run_id"]
                or evidence.get("physical_gpu") != record["gpu"]
                or evidence.get("protocol_sha256") != p["original_study_protocol_sha256"]
                or evidence.get("diagnostic_script_sha256") != record["diagnostic_script_sha256"]
                or evidence.get("recovery_protocol_sha256") != record["recovery_protocol_sha256"]):
            return False
    return True


def _environment(gpu=None):
    threads = "4" if gpu is not None else "8"
    return dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
        OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
        PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def launch_owned(command, log, gpu=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=_environment(gpu),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    for _ in range(200):
        if child.poll() is not None:
            break
        try:
            exact = identity(child.pid)
            if exact["argv"] == command and exact["uid"] == os.getuid() and exact["cwd"] == str(ROOT):
                return child, exact
        except (OSError, ValueError):
            pass
        time.sleep(.025)
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=20)
    raise RuntimeError("Cannot bind owned child identity")


def check_stops(p):
    study, output = Path(p["study"]), Path(p["output"])
    for directory in (study, study.parent, output):
        if (directory / "STOP").exists():
            raise RuntimeError(f"Stopped by {directory / 'STOP'}")


def _prior_child_is_live(output):
    output = Path(output)
    records = list(output.glob("runs/*/admission.json"))
    production = output / "production_resume_launch.json"
    if production.exists():
        records.append(production)
    for path in records:
        expected = read(path)["identity"]
        try:
            actual = identity(expected["pid"])
        except (OSError, ValueError):
            continue
        if same_process(expected, actual):
            raise RuntimeError("A previous exact recovery child is still alive")


def run(output):
    output = Path(output).resolve()
    p = verify(output)
    child = None
    with (output / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _prior_child_is_live(output)

        def state(name, **extra):
            value = dict(state=name, automatic_run=True, coordinator_pid=os.getpid(),
                         original_study=str(p["study"]), updated_unix=time.time())
            value.update(extra)
            write(output / "state.json", value)

        try:
            passed_run = None
            while passed_run is None:
                check_stops(p)
                try:
                    cards = inventory()
                except Exception as error:
                    state("gpu_inventory_unavailable", reason=repr(error))
                    time.sleep(p["poll_seconds"])
                    continue
                available = empty_cards(cards, list(range(8)), p["minimum_free_gib"])
                if not available:
                    state("waiting_empty_gpu", gpu_inventory=cards)
                    time.sleep(p["poll_seconds"])
                    continue
                verify(output)
                gpu = available[0]
                run_id = f"exact_replay_gpu{gpu}_{time.time_ns()}"
                folder = output / "runs" / run_id
                ledger = folder / "admission.json"
                command = [sys.executable, str(DIAGNOSTIC), "--study", p["study"],
                    "--output", str(folder), "--gpu", str(gpu),
                    "--admission-ledger", str(ledger), "--run-id", run_id]
                child, exact = launch_owned(command, folder / "worker.log", gpu)
                record = dict(run_id=run_id, study=p["study"], output=str(folder),
                    gpu=gpu, identity=exact,
                    protocol_sha256=p["original_study_protocol_sha256"],
                    diagnostic_script_sha256=sha(DIAGNOSTIC),
                    recovery_protocol_sha256=sha(output / "protocol.json"))
                write(ledger, record)
                while child.poll() is None:
                    check_stops(p)
                    state("diagnostic_replay", run_id=run_id, gpu=gpu,
                          worker_identity=exact)
                    time.sleep(p["poll_seconds"])
                code = child.returncode
                write(folder / "exit.json", dict(exit_code=code, exited_unix=time.time()))
                child = None
                if code == 75 and not (folder / "admitted.json").exists():
                    continue
                if code != 0:
                    state("diagnostic_requires_review", automatic_run=False,
                          run_id=run_id, exit_code=code,
                          reason="Original strict feature replay did not pass; production remains stopped")
                    return
                result_path = folder / "result.json"
                admitted_path = folder / "admitted.json"
                if (not result_path.exists() or not admitted_path.exists()
                        or not diagnostic_allows_resume(p, read(result_path), read(admitted_path), record)):
                    raise RuntimeError("Successful diagnostic lacks an identity-bound exact passing result")
                passed_run = run_id
            verify(output)
            if not audit_allows_resume(p):
                state("completed_cache_requires_review", automatic_run=False,
                      diagnostic_run=passed_run,
                      reason="The independent complete-row cache audit does not permit production resume")
                return
            from scripts.hanguard.four_arm_experiment_study import verify_retired
            verify_retired(p["study"])
            command = [sys.executable, str(STUDY_SCRIPT), "run", "--output", p["study"]]
            log = output / f"production_resume_{time.time_ns()}.log"
            child, exact = launch_owned(command, log)
            write(output / "production_resume_launch.json", dict(identity=exact,
                command=command, original_protocol_sha256=p["original_study_protocol_sha256"],
                diagnostic_run=passed_run, audit_sha256=p["audit_sha256"],
                launched_unix=time.time()))
            while child.poll() is None:
                check_stops(p)
                state("original_study_resumed", diagnostic_run=passed_run,
                      production_identity=exact,
                      production_state=read(Path(p["study"]) / "state.json")["state"])
                time.sleep(p["poll_seconds"])
            code = child.returncode
            child = None
            original = read(Path(p["study"]) / "state.json")
            completed = code == 0 and original.get("state") == "completed"
            state("completed" if completed else "original_study_requires_review",
                  automatic_run=False, production_exit_code=code,
                  production_state=original.get("state"))
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            state("requires_review", automatic_run=False,
                  error_type=type(error).__name__, reason=str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "status"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if args.action == "init":
        if args.study is None or args.audit is None:
            parser.error("init requires --study and --audit")
        p = initialize(args.study, args.output, args.audit)
        print(dict(registered=True, output=p["output"]))
    elif args.action == "run":
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        run(args.output)
    else:
        print(read(args.output / "state.json"))


if __name__ == "__main__":
    main()
