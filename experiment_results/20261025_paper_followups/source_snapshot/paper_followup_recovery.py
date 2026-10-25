"""Serialize the remaining frozen MacBERT tests after nonblocking lock contention.

No model, checkpoint, scientific protocol, or original implementation is edited.
The original admitted worker performs each saved-checkpoint evaluation. This
separate execution amendment retains the failed queue snapshot for audit.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import paper_followup_study as study

SCHEMA = "paper_followup_serial_test_recovery_v1"


def register(output):
    output = Path(output).resolve()
    study.verify(output)
    external = study.module("external")
    status = external.progress(output / "external")
    if any(job.get("phase") != "evaluate" for job in status["pending_jobs"]):
        raise ValueError("This recovery can only evaluate already selected models")
    if set(status["selected_seeds"]) != {42, 43, 44}:
        raise ValueError("All three external models must already be selected")
    if not (output / "external/selection_lock.json").exists():
        raise ValueError("The original test barrier must already be locked")
    record = dict(schema=SCHEMA, root_protocol_sha256=study.common.sha(output / "protocol.json"),
        external_protocol_sha256=study.common.sha(output / "external/protocol.json"),
        selection_lock_sha256=study.common.sha(output / "external/selection_lock.json"),
        test_manifest_sha256=study.common.sha(output / "external/data/test/manifest.json"),
        source=str(Path(__file__).resolve()), source_sha256=study.common.sha(__file__),
        mode="Sequential original external workers; saved checkpoints only; no retraining",
        reason="Concurrent nonblocking selection/test-preparation locks caused two evaluation workers to exit",
        gpu_policy=study.BUDGETS["external"], poll_seconds=3.)
    path = output / "serial_test_recovery_registration.json"
    if path.exists():
        if study.common.read(path) != record:
            raise ValueError("Frozen serial recovery registration changed")
    else:
        study.common.write(path, record)
        original_state = output / "queue_state.json"
        if original_state.exists():
            study.common.write(output / "queue_before_serial_recovery.json", study.common.read(original_state))
        (output / "source_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return record


def verify(output, registration):
    study.verify(output)
    if (study.common.sha(__file__) != registration["source_sha256"]
            or study.common.sha(output / "protocol.json") != registration["root_protocol_sha256"]
            or study.common.sha(output / "external/protocol.json") != registration["external_protocol_sha256"]
            or study.common.sha(output / "external/selection_lock.json") != registration["selection_lock_sha256"]
            or study.common.sha(output / "external/data/test/manifest.json") != registration["test_manifest_sha256"]):
        raise ValueError("Frozen recovery source, input, or checkpoint selection changed")


def archive_amendment(output):
    archive = study.ARCHIVE if output == study.DEFAULT_OUTPUT.resolve() else output / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("serial_test_recovery_registration.json", "queue_before_serial_recovery.json",
                 "serial_test_recovery_launch.json"):
        if (output / name).exists():
            shutil.copyfile(output / name, archive / name)
    files = {str(path.relative_to(archive)): study.common.sha(path) for path in sorted(archive.rglob("*"))
             if path.is_file() and path.name != "manifest.json"}
    study.common.write(archive / "manifest.json", dict(schema=study.SCHEMA, source=str(output), files=files))


def run(output=study.DEFAULT_OUTPUT):
    output = Path(output).resolve()
    with (output / ".coordinator.lock").open("a+") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = study.common.read(output / "queue_state.json")
        if any(study.common.same_process(item["identity"]) for item in prior.get("active_workers", [])):
            raise RuntimeError("An exact prior worker is still alive")
        registration = register(output)
        amendment_sha = study.common.sha(output / "serial_test_recovery_registration.json")
        lock_root = ROOT / "outputs/.qwen3guard_gpu_admission"
        lock_root.mkdir(exist_ok=True)
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        events = []
        while True:
            verify(output, registration)
            status = study.progress(output)
            pending = status["external"]["pending_jobs"]
            if not pending:
                study.finalize_children(output, status, {})
                status = study.progress(output)
                if study.jobs(status):
                    raise RuntimeError("Unexpected non-external work remains; use the original queue")
                study.state(output, "complete", children=status, active_workers=[], failures={},
                    recovered_failures=prior.get("failures", {}), recent_events=events[-20:],
                    recovery_registration_sha256=amendment_sha)
                study.refresh_archive(output)
                archive_amendment(output)
                return
            if (output / "STOP").exists():
                study.state(output, "stopped", children=status, active_workers=[],
                            recovery_registration_sha256=amendment_sha)
                return
            job = dict(pending[0], kind="external")
            if job.get("phase") != "evaluate":
                raise ValueError("Refuse to start new training in the test recovery")
            try:
                cards = sorted(study.common.gpu_inventory(), key=lambda card: -card["free_gib"])
                error = None
            except Exception as exc:
                cards, error = [], repr(exc)
            chosen, device_lock = None, None
            for card in cards:
                if not study.eligible(card, job):
                    continue
                handle = (lock_root / (card["uuid"] + ".lock")).open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    latest = next(row for row in study.common.gpu_inventory() if row["uuid"] == card["uuid"])
                    if not study.eligible(latest, job):
                        handle.close()
                        continue
                except Exception:
                    handle.close()
                    continue
                chosen, device_lock = latest, handle
                break
            if chosen is None:
                study.state(output, "waiting_for_recovery_capacity", children=status, active_workers=[],
                    failures={}, inventory_error=error, recovery_registration_sha256=amendment_sha)
                time.sleep(3.)
                continue
            log_path = logs / f"serial_external_s{job['seed']}_gpu{chosen['index']}_{time.time_ns()}.log"
            command = [sys.executable, str(Path(study.__file__).resolve()), "worker", "--output", str(output),
                       "--kind", "external", "--seed", str(job["seed"]), "--gpu", str(chosen["index"])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=chosen["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            with log_path.open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    owner = study.common.identity(child.pid)
                    if owner is None:
                        raise RuntimeError("Cannot verify the exact newly created worker")
                except BaseException:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=30)
                    device_lock.close()
                    raise
                worker = dict(identity=owner, job=job, admitted_inventory=chosen, log_path=str(log_path))
                while child.poll() is None:
                    study.state(output, "recovering_tests", children=study.progress(output),
                        active_workers=[worker], failures={}, recent_events=events[-20:],
                        recovery_registration_sha256=amendment_sha)
                    time.sleep(3.)
                code = child.returncode
            device_lock.close()
            events.append(dict(seed=job["seed"], gpu=chosen["index"], exit_code=code,
                               log=str(log_path), finished_unix=time.time()))
            if code not in (0, 75):
                study.state(output, "failed", active_workers=[],
                    failures={study.job_key(job): dict(exit_code=code, log=str(log_path))},
                    recovery_registration_sha256=amendment_sha, recent_events=events[-20:])
                raise RuntimeError(f"Serial saved-model test failed; inspect {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.DEFAULT_OUTPUT)
    run(parser.parse_args().output)
