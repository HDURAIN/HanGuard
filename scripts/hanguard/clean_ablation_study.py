"""Run the complete frozen-backbone ablation from raw data and pristine weights.

No historical experiment protocol, tokenization, feature, checkpoint, selected
layer, prediction, or exclusion list is an input. All readouts use one fresh
joint feature cache. A process-bound coordinator waits for genuinely empty
public GPUs and never terminates foreign processes.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.fragment_clean_lora_study import (
    empty_cards, identity, inventory, read, same_process, sha, write,
)

SCHEMA = "hanguard_clean_complete_ablation_v1"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_clean_ablation_20261022"
DEFAULT_DATA = ROOT / "data/three_source_remediated_20261012"
DEFAULT_MODEL = ROOT / "models/Qwen3.5-4B"
DEFAULT_ARCHIVE = ROOT / "experiment_results/20261022_clean_ablation"
TASKS = ("binary", "category")
SEEDS = (42, 43, 44)
ARMS = ("baseline", "fusion_only", "category_only", "fusion_category",
        "fusion_repeat_last", "category_fusion_repeat_last", "category_shared",
        "fusion_category_shared", "probe_selected_single")
EXECUTION = dict(prelaunch_foreign="block", active_foreign="record_only",
                 minimum_free_gib=20., max_cuda_workers=8, max_cpu_workers=1,
                 cpu_threads=8, gpu_threads=4, reserved_gpus=[],
                 never_terminate_foreign_processes=True, dummy_reservations=False,
                 poll_seconds=3.)
TRAINING = dict(width=128, dropout=.1, lr=.001, weight_decay=.001,
                grad_clip=5., epochs=12, max_epochs=40, effective_batch=128,
                token_budget=8192, max_micro=32, pad_multiple=1,
                bootstrap_draws=2000, probe_epochs=12)
EXTRACTION = dict(token_budget=16384, max_micro=64, pad_multiple=32,
                  max_tokens=4096, truncation=False, dtype="float16",
                  raw_block_outputs=True, pooling="last_valid_token",
                  group_chunk_size=32)
_SELECTION_VERIFICATION_CACHE = {}


def source_closure():
    """Snapshot reusable implementation code without importing old studies."""
    seeds = [ROOT / "scripts/hanguard" / name for name in (
        "clean_ablation_study.py", "clean_ablation_features.py",
        "clean_ablation_runtime.py", "clean_ablation_io.py", "clean_ablation_heads.py",
        "clean_ablation_training.py", "package_clean_ablation_results.py")]
    seeds += [ROOT / "tests" / name for name in (
        "test_clean_ablation_study.py", "test_clean_ablation_features.py",
        "test_clean_ablation_runtime.py", "test_clean_ablation_io.py", "test_clean_ablation_heads.py",
        "test_clean_ablation_training.py", "test_package_clean_ablation_results.py")]
    pending, seen = list(seeds), set()
    while pending:
        p = pending.pop().resolve()
        if p in seen or not p.is_file() or not p.is_relative_to(ROOT):
            continue
        seen.add(p)
        if p.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
                modules += [f"{node.module}.{a.name}" for a in node.names]
            for module in modules:
                if module and all(x.isidentifier() for x in module.split(".")):
                    candidate = ROOT.joinpath(*module.split("."))
                    pending.extend((candidate.with_suffix(".py"), candidate / "__init__.py"))
    missing = [str(p) for p in seeds if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Complete new implementation before registration: {missing}")
    return sorted(seen)


def dependency_sources():
    """Pin the actual Qwen/FLA/Triton Python implementations, not just versions."""
    paths = set()
    for name, subdir in (("transformers", "models/qwen3_5"), ("fla", ""),
                         ("triton", "runtime")):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"Required frozen runtime package is absent: {name}")
        package = Path(next(iter(spec.submodule_search_locations))) / subdir
        paths.update(p for p in package.rglob("*.py") if p.is_file())
    return sorted(paths)


def initialize(output=DEFAULT_OUTPUT, data=DEFAULT_DATA, model=DEFAULT_MODEL):
    import json
    import pandas as pd
    import pyarrow.parquet as pq
    from scripts.hanguard import remediated_dataset_contract as contract
    from scripts.hanguard.frozen_base_readout_study import runtime_identity

    output, data, model = (Path(p).resolve() for p in (output, data, model))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty study directory")
    if data.is_relative_to(ROOT / "outputs") or data.is_relative_to(ROOT / "experiment_results"):
        raise ValueError("Input must be raw labeled data, not a historical experiment")
    if (model / "adapter_config.json").exists():
        raise ValueError("A pristine foundation model is required")
    contract.release_contract(data)
    config = read(model / "config.json")
    text = config.get("text_config", config)
    if int(text["num_hidden_layers"]) != 32 or config["model_type"] != "qwen3_5":
        raise ValueError("The complete registered comparison uses Qwen3.5 with 32 layers")
    hidden = int(text["hidden_size"])
    rows = {s: pq.ParquetFile(data / f"{s}.parquet").metadata.num_rows
            for s in ("train", "validation", "test")}
    if rows != dict(train=66802, validation=8350, test=8350):
        raise ValueError(f"Raw data membership changed: {rows}")
    frames = {}
    category_rows = {}
    for split in ("train", "validation"):
        frame = pd.read_parquet(data / f"{split}.parquet", columns=[
            "base_id", "group_id", "source", "prompt_harm_label", "category_id", "type_label_mask", "split"])
        cats, mask, harmful = contract.label_arrays(frame, require_type_mask=True)
        if (not frame.base_id.is_unique or frame.base_id.isna().any() or
                frame.group_id.isna().any() or not frame.split.eq(split).all()):
            raise ValueError("Invalid raw data row/split identities")
        category_rows[split] = int((frame.source.isin(["chinese_curated", "jailbench"]).to_numpy() & mask & harmful).sum())
        frames[split] = frame
    # Only identity metadata, never test prompts or labels, is read at registration.
    frames["test"] = pd.read_parquet(data / "test.parquet", columns=["base_id", "group_id"])
    if not frames["test"].base_id.is_unique or frames["test"].isna().any().any():
        raise ValueError("Test identities are invalid")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("base_id", "group_id"):
            if set(frames[a][field].astype(str)) & set(frames[b][field].astype(str)):
                raise ValueError(f"Raw data leakage: {a}/{b}/{field}")
    if category_rows != dict(train=17440, validation=2180):
        raise ValueError("The native known-harmful five-class scope changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 256 * 1024**3:
        raise ValueError("Reserve 256 GiB for fresh joint features and checkpoints")
    codes = {str(p): sha(p) for p in source_closure()}
    dependencies = {str(p): sha(p) for p in dependency_sources()}
    shards = set(read(model / "model.safetensors.index.json")["weight_map"].values())
    model_files = {model / p for p in shards}
    model_files.update(p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".jinja"))
    model_hashes = {str(p): sha(p) for p in sorted(model_files)}
    output.mkdir(parents=True)
    snapshot = output / "data"
    snapshot.mkdir()
    raw_inputs = {}
    for name in ("train.parquet", "validation.parquet", "test.parquet",
                 "manifest.json", "release_qa.json", "split_policy.json"):
        source, target = data / name, snapshot / name
        digest = sha(source)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise RuntimeError("Raw dataset snapshot copy changed bytes")
        raw_inputs[str(source)] = digest
    inputs = dict(model_hashes)
    inputs.update({str(p): sha(p) for p in sorted(snapshot.iterdir())})
    value = dict(schema=SCHEMA, study=str(output), data_dir=str(snapshot),
        raw_data_provenance=raw_inputs, dataset_release=contract.release_contract(snapshot),
        data_sha256={s: inputs[str(snapshot / f"{s}.parquet")] for s in rows},
        model_path=str(model), model_type=config["model_type"], num_layers=32,
        hidden_size=hidden, rows=rows, use_lora=False, adapter_path=None,
        adapter_sha256=None, backbone_trainable_parameters=0,
        fresh_tokenization=True, fresh_features=True, fresh_probes=True,
        fresh_checkpoints=True, historical_experiment_inputs=[],
        selected_test_exclusions=[], posthoc_subset_enabled=False,
        features_dir=str(output / "features"), seeds=list(SEEDS), tasks=list(TASKS),
        arms=list(ARMS), formal_heads=54, linear_probe_heads=192,
        fusion_layers=[8, 16, 24, 32], fusion_weights=[.25] * 4,
        learnable_layer_weights=False, best_layer_anchor=False,
        sentence_feature_layers=list(range(1, 33)), token_feature_layers=[32],
        joint_same_forward=True, training=TRAINING, settings=TRAINING,
        probe_epochs=12, extraction=EXTRACTION, execution=EXECUTION,
        numeric_path=dict(sdpa="math", fla_autotune="canonical_valid_config",
                          cublas_workspace_config=":4096:8", allow_tf32=False,
                          triton_cache_dir=str(output / "runtime/triton_cache"),
                          torchinductor_cache_dir=str(output / "runtime/torchinductor_cache"),
                          cuda_cache_path=str(output / "runtime/cuda_cache"),
                          calibration_atol=0., calibration_rtol=0.),
        calibration=dict(required_distinct_processes=2, training_inputs_only=True,
                         same_joint_output=True, every_worker_replays_baseline=True,
                         different_gpu_validation="every newly admitted GPU replays the same fresh train groups"),
        category_sources=["chinese_curated", "jailbench"],
        category_known_harmful_rows=dict(category_rows, test=2180), gated_native_test_rows=3704,
        checkpoint_selection="minimum validation softmax CE per model",
        layer_selection="mean validation CE over the three fresh probe seeds, per task",
        threshold_selection="validation maximum binary accuracy plus fixed 0.5 reporting",
        budget_extension="common 12/24/40 continuation across all nine heads per task/seed",
        formal_test_barrier="all six probe fits, selected layers and all six nine-head fits locked before test tokenization",
        source_reuse="implementation code only; no historical experimental artifact is read",
        inputs=inputs, input_stats={name: dict(size=Path(name).stat().st_size,
            mtime_ns=Path(name).stat().st_mtime_ns) for name in inputs},
        code_sha256=codes, dependency_sha256=dependencies,
        runtime=runtime_identity(), created_unix=time.time())
    write(output / "protocol.json", value)
    write(output / "protocol_registration.json", dict(protocol_sha256=sha(output / "protocol.json")))
    for name in codes:
        target = output / "source_snapshot" / str(Path(name).relative_to(ROOT)).replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        if sha(target) != codes[name]:
            raise RuntimeError("Registered source snapshot changed")
    write(output / "state.json", dict(state="registered", automatic_run=False,
        formal_test_opened=False, active_workers=[], updated_unix=time.time()))
    return verify(output)


def verify(study, full_hash=False):
    from scripts.hanguard.frozen_base_readout_study import runtime_identity
    study = Path(study).resolve()
    if (study / "STOP").exists() or (study.parent / "STOP").exists():
        raise RuntimeError("Study STOP requested")
    p = read(study / "protocol.json")
    if (p.get("schema") != SCHEMA or p.get("study") != str(study) or
            read(study / "protocol_registration.json")["protocol_sha256"] != sha(study / "protocol.json")):
        raise ValueError("Clean study registration changed")
    if (p.get("historical_experiment_inputs") != [] or p.get("selected_test_exclusions") != []
            or p.get("posthoc_subset_enabled") is not False or p.get("use_lora") is not False
            or p.get("joint_same_forward") is not True or p.get("fusion_weights") != [.25] * 4):
        raise ValueError("Clean joint-feature ablation contract changed")
    if p["runtime"] != runtime_identity():
        raise ValueError("Registered Python/library runtime changed")
    if Path(p["data_dir"]).resolve() != study / "data" or Path(p["features_dir"]).resolve() != study / "features":
        raise ValueError("Fresh data and features must stay inside this study")
    for name, digest in {**p["code_sha256"], **p["dependency_sha256"]}.items():
        if sha(name) != digest:
            raise ValueError(f"Registered implementation changed: {name}")
    model = Path(p["model_path"]).resolve()
    for name, digest in p["inputs"].items():
        path = Path(name).resolve()
        if not (path.is_relative_to(study / "data") or path.is_relative_to(model)):
            raise ValueError("A non-raw or historical experiment artifact entered the input set")
        stat = path.stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != p["input_stats"][name]:
            raise ValueError(f"Input metadata changed: {name}")
        if full_hash and sha(path) != digest:
            raise ValueError(f"Input bytes changed: {name}")
    return p


def verify_preflight(study):
    from scripts.hanguard.clean_ablation_features import SCHEMA as FEATURE_SCHEMA
    study = Path(study).resolve()
    p = verify(study)
    lock = read(study / "feature_preflight_lock.json")
    protocol_sha = sha(study / "protocol.json")
    if (lock.get("complete") is not True or lock.get("protocol_sha256") != protocol_sha
            or len(lock.get("calibrations", [])) < p["calibration"]["required_distinct_processes"]
            or sha(study / "calibration/baseline.json") != lock.get("baseline_sha256")):
        raise ValueError("Fresh feature preflight is incomplete")
    baseline = read(study / "calibration/baseline.json")
    train_plan_sha = sha(study / "features/train/plan.json")
    records = baseline.get("records")
    if (baseline.get("schema") != FEATURE_SCHEMA or baseline.get("passed") is not True
            or baseline.get("protocol_sha256") != protocol_sha
            or baseline.get("train_plan_sha256") != train_plan_sha
            or not isinstance(records, list) or not records
            or any(not isinstance(row, dict) or row.get("repeated_exact") is not True
                   or row.get("same_forward_last_exact") is not True for row in records)):
        raise ValueError("Fresh calibration baseline does not bind exact current joint features")
    births = set()
    for rec in lock["calibrations"]:
        path = Path(rec["path"]).resolve()
        if (not path.is_relative_to(study / "calibration") or sha(path) != rec["sha256"]
                or read(path).get("passed") is not True):
            raise ValueError("Fresh calibration evidence changed")
        report = read(path)
        worker_id = rec.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id or Path(worker_id).name != worker_id:
            raise ValueError("Fresh calibration has no valid worker registration")
        ledger = read(study / "workers" / f"{worker_id}.json")
        if (report.get("schema") != FEATURE_SCHEMA or report.get("worker_id") != worker_id
                or path.name != f"{worker_id}.json"
                or report.get("protocol_sha256") != protocol_sha
                or report.get("train_plan_sha256") != train_plan_sha
                or report.get("records") != records
                or report.get("identity") != rec.get("identity")
                or report.get("identity") != ledger.get("identity")
                or ledger.get("phase") != "calibrate" or ledger.get("device") != "cuda"
                or ledger.get("protocol_sha256") != protocol_sha
                or type(rec.get("gpu")) is not int or rec["gpu"] != ledger.get("gpu")
                or str(report.get("physical_gpu")) != str(rec["gpu"])):
            raise ValueError("Fresh calibration protocol, joint outputs, or exact worker identity changed")
        births.add((rec["identity"]["pid"], rec["identity"]["start_time_ticks"]))
    if len(births) < p["calibration"]["required_distinct_processes"]:
        raise ValueError("Independent process replays are required")
    return lock


def verify_selection_lock(study):
    from scripts.hanguard import clean_ablation_training as training
    study = Path(study).resolve()
    verify(study)
    verify_preflight(study)
    lock = read(study / "selection_lock.json")
    if lock.get("complete") is not True or lock.get("protocol_sha256") != sha(study / "protocol.json"):
        raise ValueError("All clean experiment selections must be locked before test access")
    def signature():
        records = []
        artifacts = lock.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("Selected artifact registration must be a mapping")
        for name in sorted(artifacts):
            path = Path(name).resolve()
            if not path.is_relative_to(study):
                raise ValueError("Selected artifact escaped this clean experiment or changed")
            info = path.stat()
            records.append((name, str(path), info.st_dev, info.st_ino, info.st_size,
                            info.st_mtime_ns, info.st_ctime_ns))
        return tuple(records)
    before = signature()
    cache_key = (sha(study / "selection_lock.json"), before)
    # This filesystem may give a write and an immediate same-size rewrite the
    # same ctime tick. Newly written files are therefore rehashed until their
    # timestamps are older than that tick; stable immutable selections use
    # only stat checks afterwards.
    stable_age_ns = 100_000_000
    stable = all(max(record[-2:]) < time.time_ns() - stable_age_ns for record in before)
    if stable and _SELECTION_VERIFICATION_CACHE.get(str(study)) == cache_key:
        return lock
    expected = training.selection_artifacts(study)
    for key in ("selections", "artifacts"):
        if lock.get(key) != expected[key]:
            raise ValueError("Probe/head selection artifacts changed after the test barrier")
    for name, digest in lock["artifacts"].items():
        if not Path(name).resolve().is_relative_to(study) or sha(name) != digest:
            raise ValueError("Selected artifact escaped this clean experiment or changed")
    if signature() != before:
        raise ValueError("Selected artifact changed during verification")
    _SELECTION_VERIFICATION_CACHE[str(study)] = cache_key
    return lock


def lock_selections(study):
    from scripts.hanguard import clean_ablation_training as training
    study = Path(study).resolve()
    verify_preflight(study)
    if not (study / "selection_lock.json").exists():
        evidence = training.selection_artifacts(study)
        write(study / "selection_lock.json", dict(complete=True, test_opened=False,
            protocol_sha256=sha(study / "protocol.json"), locked_unix=time.time(), **evidence))
    return verify_selection_lock(study)


def worker(args):
    study = args.output.resolve()
    p = verify(study)
    ledger = study / "workers" / f"{args.worker_id}.json"
    deadline = time.monotonic() + 15
    while not ledger.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Exact worker ledger did not appear")
        time.sleep(.05)
    record = read(ledger)
    if (not same_process(record["identity"], identity(os.getpid())) or
            any(record.get(key) != getattr(args, key) for key in ("phase", "task", "seed", "split", "device", "gpu"))):
        raise ValueError("Worker does not match its exact phase/process identity")
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            raise ValueError("Physical CUDA binding differs")
        try:
            cards = inventory()
            card = next((r for r in cards if r["index"] == args.gpu), None)
            if card is not None and (not isinstance(card.get("pids"), list)
                    or any(type(pid) is not int or pid < 1 for pid in card["pids"])
                    or not isinstance(card.get("free_gib"), (int, float))
                    or not math.isfinite(card["free_gib"])):
                raise ValueError("GPU admission telemetry is incomplete or invalid")
        except Exception as error:
            write(ledger.with_suffix(".retry.json"), dict(reason=repr(error), unix=time.time()))
            return 75
        if card is None or card["pids"] or card["free_gib"] < p["execution"]["minimum_free_gib"]:
            return 75
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != p["numeric_path"]["cublas_workspace_config"]:
            raise ValueError("Deterministic cuBLAS workspace is not configured")
        for environment, setting in (("TRITON_CACHE_DIR", "triton_cache_dir"),
                                     ("TORCHINDUCTOR_CACHE_DIR", "torchinductor_cache_dir"),
                                     ("CUDA_CACHE_PATH", "cuda_cache_path")):
            expected = Path(p["numeric_path"][setting]).resolve()
            if (not expected.is_relative_to(study / "runtime")
                    or os.environ.get(environment) != str(expected)):
                raise ValueError("Compiler caches must belong to this new clean study")
        import torch
        torch.cuda.init()
        ready = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del ready
    write(ledger.with_suffix(".admitted.json"), dict(admitted=True,
        identity=record["identity"], protocol_sha256=sha(study / "protocol.json"),
        physical_gpu=args.gpu, admitted_unix=time.time()))
    import torch
    torch.set_num_threads(p["execution"]["gpu_threads" if args.device == "cuda" else "cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.phase in ("calibrate", "extract"):
        from scripts.hanguard import clean_ablation_features as features
        if args.device != "cuda":
            raise ValueError("Real foundation extraction requires an admitted empty GPU")
        if args.phase == "calibrate":
            features.run_calibration(study, args.worker_id, device="cuda", admission_already_checked=True)
        else:
            verify_preflight(study)
            features.extract_worker(study, args.split, args.worker_id, device="cuda", admission_already_checked=True)
    else:
        verify_preflight(study)
        from scripts.hanguard import clean_ablation_training as training
        function = {"probes": training.fit_probes, "fit": training.fit_task,
                    "evaluate": training.evaluate_task}[args.phase]
        function(study, args.task, args.seed, device=args.device)
    write(ledger.with_suffix(".complete.json"), dict(complete=True,
        protocol_sha256=sha(study / "protocol.json"), completed_unix=time.time()))
    return 0


class Scheduler:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.p = verify(study)
        self.active, self.arrivals, self.resume_devices = {}, set(), {}
        self.workers = self.study / "workers"
        self.workers.mkdir(exist_ok=True)

    def state(self, name, **updates):
        previous = read(self.study / "state.json")
        previous.pop("error_type", None)
        previous.pop("reason", None)
        value = dict(previous, state=name, automatic_run=True,
            coordinator_pid=os.getpid(), updated_unix=time.time(),
            active_workers=[{k: v for k, v in r.items() if k != "child"} for r in self.active.values()])
        value.update(updates)
        write(self.study / "state.json", value)

    def cards(self):
        try:
            cards = inventory()
        except Exception as error:
            self.state("gpu_inventory_unavailable", gpu_inventory_error=repr(error))
            return []
        for worker in self.active.values():
            row = next((c for c in cards if c["index"] == worker["gpu"]), None)
            for pid in row["pids"] if row else []:
                key = worker["worker_id"], pid
                if pid != worker["identity"]["pid"] and key not in self.arrivals:
                    self.arrivals.add(key)
                    write(self.study / "foreign_arrivals" / f"{time.time_ns()}_{pid}.json",
                        dict(action="record_only_continue_owned_worker", gpu=worker["gpu"],
                             foreign_pid=pid, worker_id=worker["worker_id"], unix=time.time()))
        return cards

    def launch(self, phase, *, device, gpu=None, task=None, seed=None, split=None):
        name = f"{phase}_{split or task or 'joint'}_{seed or 'none'}_{device}{gpu if gpu is not None else ''}_{time.time_ns()}"
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(self.study),
                   "--worker-id", name, "--phase", phase, "--device", device]
        for key, val in (("gpu", gpu), ("task", task), ("seed", seed), ("split", split)):
            if val is not None:
                command.extend((f"--{key}", str(val)))
        threads = str(self.p["execution"]["gpu_threads" if device == "cuda" else "cpu_threads"])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
            CUBLAS_WORKSPACE_CONFIG=self.p["numeric_path"]["cublas_workspace_config"],
            OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
            PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for environment, setting in (("TRITON_CACHE_DIR", "triton_cache_dir"),
                                     ("TORCHINDUCTOR_CACHE_DIR", "torchinductor_cache_dir"),
                                     ("CUDA_CACHE_PATH", "cuda_cache_path")):
            location = Path(self.p["numeric_path"][setting]).resolve()
            if not location.is_relative_to(self.study / "runtime"):
                raise ValueError("Compiler cache escaped the new study")
            location.mkdir(parents=True, exist_ok=True)
            env[environment] = str(location)
        with (self.workers / f"{name}.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        exact = None
        for _ in range(200):
            if child.poll() is not None:
                break
            try:
                actual = identity(child.pid)
                if actual["argv"] == command and actual["uid"] == os.getuid() and actual["cwd"] == str(ROOT):
                    exact = actual
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.025)
        if exact is None:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=20)
            raise RuntimeError("Cannot register new exact worker identity")
        record = dict(worker_id=name, identity=exact, phase=phase, task=task,
            seed=seed, split=split, device=device, gpu=gpu, command=command,
            protocol_sha256=sha(self.study / "protocol.json"))
        write(self.workers / f"{name}.json", record)
        self.active[name] = dict(record, child=child)
        return name

    def poll(self):
        if (self.study / "STOP").exists() or (self.study.parent / "STOP").exists():
            raise RuntimeError("Study STOP requested")
        for name, record in list(self.active.items()):
            code = record["child"].poll()
            if code is None:
                continue
            write(self.workers / f"{name}.exit.json", dict(exit_code=code, exited_unix=time.time()))
            del self.active[name]
            if code == 75 and not (self.workers / f"{name}.admitted.json").exists():
                continue
            if code != 0 or not (self.workers / f"{name}.complete.json").exists():
                raise RuntimeError(f"Owned worker failed: {name}, exit={code}; inspect its log")

    def stop_owned(self):
        for r in self.active.values():
            if r["child"].poll() is None:
                r["child"].terminate()
        for r in self.active.values():
            try:
                r["child"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                r["child"].kill()
                r["child"].wait(timeout=10)
        self.active.clear()

    def preflight(self):
        if (self.study / "feature_preflight_lock.json").exists():
            return verify_preflight(self.study)
        while True:
            self.poll()
            passed = []
            for path in sorted(self.workers.glob("*.complete.json")):
                name = path.name.removesuffix(".complete.json")
                ledger = read(self.workers / f"{name}.json")
                if ledger["phase"] != "calibrate":
                    continue
                report = self.study / "calibration" / f"{name}.json"
                if not report.exists() or read(report).get("passed") is not True:
                    raise ValueError("Completed calibration lacks a passing fresh report")
                passed.append(dict(worker_id=name, path=str(report), sha256=sha(report),
                                   identity=ledger["identity"], gpu=ledger["gpu"]))
            if len(passed) >= self.p["calibration"]["required_distinct_processes"] and not self.active:
                lock = dict(complete=True, protocol_sha256=sha(self.study / "protocol.json"),
                    baseline_sha256=sha(self.study / "calibration/baseline.json"), calibrations=passed,
                    distinct_physical_gpus=sorted({r["gpu"] for r in passed}), locked_unix=time.time())
                write(self.study / "feature_preflight_lock.json", lock)
                return verify_preflight(self.study)
            cards = self.cards()
            available = empty_cards(cards, list(range(8)), self.p["execution"]["minimum_free_gib"])
            if not self.active and available:
                # Independent processes run sequentially so the first creates
                # the fresh baseline before any other process compares it.
                used = {r["gpu"] for r in passed}
                gpu = next((g for g in available if g not in used), available[0])
                self.launch("calibrate", device="cuda", gpu=gpu)
            self.state("feature_preflight", completed_independent_replays=len(passed),
                       required_independent_replays=self.p["calibration"]["required_distinct_processes"],
                       gpu_inventory=cards)
            time.sleep(self.p["execution"]["poll_seconds"])

    def extract(self, split):
        from scripts.hanguard import clean_ablation_features as features
        self.state(f"preparing_{split}")
        features.prepare(self.study, split)
        while True:
            self.poll()
            progress = features.progress(self.study, split)
            done = features.all_complete(self.study, split)
            if done and not self.active:
                features.finalize(self.study, split)
                return
            cards = self.cards()
            available = empty_cards(cards, list(range(8)), self.p["execution"]["minimum_free_gib"])
            occupied = {r["gpu"] for r in self.active.values()}
            remaining = max(0, progress["total_chunks"] - progress["complete_chunks"] - len(self.active))
            if not done:
                for gpu in available:
                    if gpu not in occupied and remaining > 0:
                        self.launch("extract", device="cuda", gpu=gpu, split=split)
                        remaining -= 1
            self.state(f"extract_{split}", extraction_progress=progress, gpu_inventory=cards)
            time.sleep(self.p["execution"]["poll_seconds"])

    def resume_device(self, phase, task, seed):
        """An epoch resume retains its registered CPU/CUDA numerical identity."""
        if phase not in ("probes", "fit"):
            return None
        folder = "probes" if phase == "probes" else "heads"
        path = (self.study / folder / task / f"s{seed}" / "epoch_resume.pt").resolve()
        if not path.is_relative_to(self.study):
            raise ValueError("Resume checkpoint escaped this clean study")
        if not path.is_file():
            return None
        info = path.stat()
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        cached = self.resume_devices.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1]
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=True)
        kind = saved.get("identity", {}).get("device_type")
        if kind not in ("cpu", "cuda"):
            raise ValueError("Resume checkpoint has no valid registered device identity")
        self.resume_devices[str(path)] = (signature, kind)
        return kind

    def tasks(self, phase):
        from scripts.hanguard import clean_ablation_training as training
        while True:
            self.poll()
            pending = []
            for task in self.p["tasks"]:
                for seed in self.p["seeds"]:
                    base = "probes" if phase == "probes" else "heads"
                    filename = "test_complete.json" if phase == "evaluate" else "selection.json"
                    done = (self.study / base / task / f"s{seed}" / filename).exists()
                    if done and phase == "evaluate":
                        training.verify_evaluation(self.study, task, seed)
                    if not done and not any(r["task"] == task and r["seed"] == seed for r in self.active.values()):
                        pending.append((task, seed, self.resume_device(phase, task, seed)))
            if not pending and not self.active:
                return
            cards = self.cards()
            occupied = {r["gpu"] for r in self.active.values()}
            for gpu in empty_cards(cards, list(range(8)), self.p["execution"]["minimum_free_gib"]):
                index = next((i for i, (_, _, kind) in enumerate(pending) if kind != "cpu"), None)
                if index is not None and gpu not in occupied:
                    task, seed, _ = pending.pop(index)
                    self.launch(phase, device="cuda", gpu=gpu, task=task, seed=seed)
            index = next((i for i, (_, _, kind) in enumerate(pending) if kind != "cuda"), None)
            if index is not None and sum(r["device"] == "cpu" for r in self.active.values()) < self.p["execution"]["max_cpu_workers"]:
                task, seed, _ = pending.pop(index)
                self.launch(phase, device="cpu", task=task, seed=seed)
            self.state(phase, pending_tasks=[dict(task=t, seed=s, resume_device=kind) for t, s, kind in pending], gpu_inventory=cards)
            time.sleep(self.p["execution"]["poll_seconds"])


def verify_retired(study):
    for path in (Path(study) / "workers").glob("*.json"):
        if path.name.count(".") != 1:
            continue
        expected = read(path)["identity"]
        try:
            actual = identity(expected["pid"])
        except (OSError, ValueError):
            continue
        if actual.get("start_time_ticks") == expected["start_time_ticks"]:
            raise RuntimeError("A previous exact worker is still alive; no duplicate writer permitted")


def run(study):
    from scripts.hanguard import clean_ablation_features as features
    from scripts.hanguard import clean_ablation_training as training
    from scripts.hanguard import package_clean_ablation_results as package
    study = Path(study).resolve()
    verify(study, full_hash=True)
    with (study / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(study / "state.json").get("state") == "completed":
            return read(study / "results.json")
        verify_retired(study)
        for split in ("train", "validation", "test"):
            if (study / "features" / split / "plan.json").exists():
                features.release_dead_claims(study, split)
        scheduler = Scheduler(study)
        try:
            scheduler.state("preparing_train")
            features.prepare(study, "train")
            scheduler.state("preparing_validation")
            features.prepare(study, "validation")
            scheduler.preflight()
            scheduler.extract("train")
            scheduler.extract("validation")
            scheduler.tasks("probes")
            training.lock_probe_layers(study)
            scheduler.tasks("fit")
            lock_selections(study)
            scheduler.state("test_unlocked", formal_test_opened=True)
            scheduler.extract("test")
            scheduler.tasks("evaluate")
            result = training.summarize(study)
            if result.get("complete") is not True:
                raise RuntimeError("Fresh complete-ablation results are incomplete")
            scheduler.state("completed", automatic_run=False)
            package.refresh(ROOT, study=study, archive=DEFAULT_ARCHIVE)
            package.verify(DEFAULT_ARCHIVE)
            return result
        except BaseException as error:
            scheduler.stop_owned()
            scheduler.state("requires_review", automatic_run=False,
                            error_type=type(error).__name__, reason=str(error))
            write(study / "failures" / f"{time.time_ns()}.json", read(study / "state.json"))
            try:
                package.refresh(ROOT, study=study, archive=DEFAULT_ARCHIVE)
            except Exception:
                pass
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "run", "worker", "status"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--worker-id")
    p.add_argument("--phase", choices=("calibrate", "extract", "probes", "fit", "evaluate"))
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--gpu", type=int)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--seed", type=int)
    p.add_argument("--split", choices=("train", "validation", "test"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from scripts.hanguard.clean_ablation_io import HistoricalArtifactReadGuard
    # The live CLI coordinator and all its workers reject old result/cache
    # reads in addition to validating registered new-study artifact identities.
    _guard = HistoricalArtifactReadGuard(args.output, DEFAULT_ARCHIVE)
    if args.action == "init":
        p = initialize(args.output, args.data, args.model)
        print(dict(registered=True, study=p["study"], formal_heads=p["formal_heads"], linear_probe_heads=p["linear_probe_heads"]))
    elif args.action == "status":
        print(read(args.output / "state.json"))
    elif args.action == "run":
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        run(args.output)
    else:
        if not args.worker_id or not args.phase or not args.device:
            raise ValueError("Worker requires an explicit ID, phase and device")
        raise SystemExit(worker(args))


if __name__ == "__main__":
    main()
