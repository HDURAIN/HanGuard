"""Register and execute frozen-backbone four-arm classification experiments.

Both binary and five-class heads use the requested 2 x 2 factorial. The original
two-stage system is also evaluated by crossing the binary gates and type heads.
New raw L32 token caches preserve the parent's full original microbatches.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.fragment_clean_lora_study import (  # noqa: E402
    empty_cards, identity, inventory, read, same_process, sha, write,
)

SCHEMA = "hanguard_four_arm_binary_category_system_v1"
DEFAULT_PARENT = ROOT / "outputs/hanguard_frozen_base_full_readouts_20261013_v5"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_four_arm_experiments_20261019"
ARMS = ("baseline", "fusion_only", "category_only", "fusion_category")
TASKS = ("binary", "category")
SEEDS = (42, 43, 44)
POLICY = dict(prelaunch_foreign="block", active_foreign="record_only",
              reserved_gpus=[], minimum_free_gib=20., max_cuda_workers=8,
              max_cpu_workers=1, cpu_threads=8, gpu_threads=4,
              never_terminate_foreign_processes=True, dummy_reservations=False)
TRAINING = dict(width=128, dropout=.1, lr=.001, weight_decay=.001,
                grad_clip=5., epochs=12, max_epochs=40, effective_batch=128,
                token_budget=8192, max_micro=32, pad_multiple=1,
                bootstrap_draws=2000)


def source_files():
    names = (
        "scripts/hanguard/four_arm_experiment_study.py",
        "scripts/hanguard/four_arm_query_heads.py",
        "scripts/hanguard/four_arm_token_cache.py",
        "scripts/hanguard/four_arm_experiment_training.py",
        "scripts/hanguard/package_four_arm_experiments.py",
        "scripts/hanguard/multilabel_heads.py",
        "scripts/hanguard/sparse_token_feature_cache.py",
        "scripts/hanguard/frozen_base_readout_study.py",
        "scripts/hanguard/frozen_base_parallel_features.py",
        "scripts/hanguard/frozen_base_readout_queue.py",
        "scripts/hanguard/sparse_token_readout_queue.py",
        "scripts/hanguard/frozen_base_readout_heads.py",
        "scripts/hanguard/frozen_lora_layer_diagnostics.py",
        "scripts/hanguard/repaired_study.py",
        "scripts/hanguard/repaired_heads.py",
        "scripts/hanguard/remediated_dataset_contract.py",
        "scripts/hanguard/fragment_clean_lora_study.py",
        "scripts/hanguard/ab_readout_parallel_queue.py",
        "scripts/hanguard/ab_diagnostic_queue.py",
        "scripts/hanguard/ab_diagnostic_experiment.py",
        "scripts/hanguard/ab_frozen_readout_diagnostics.py",
        "hanguard_model.py",
        "tests/test_four_arm_query_heads.py",
        "tests/test_four_arm_token_cache.py",
        "tests/test_four_arm_experiment_training.py",
        "tests/test_four_arm_experiment_study.py",
        "tests/test_package_four_arm_experiments.py",
    )
    return [ROOT / name for name in names]


def initialize(output, parent=DEFAULT_PARENT):
    from scripts.hanguard import frozen_base_readout_study as base
    from scripts.hanguard import remediated_dataset_contract as contract
    import pandas as pd

    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty, new output directory")
    p = base.load_protocol(parent, full_hash=True)
    if p.get("use_lora") is not False or p.get("adapter_path") is not None:
        raise ValueError("The parent must be the original frozen foundation")
    data = Path(p["data_dir"])
    contract.release_contract(data)
    inputs = dict(p["inputs"])
    inputs[str(parent / "protocol.json")] = sha(parent / "protocol.json")
    excluded = ROOT / "data/three_source_wildguard122_subset_20261018/excluded_base_ids.txt"
    policy_path = excluded.parent / "filter_policy.json"
    ids = excluded.read_text().splitlines()
    if len(ids) != 122 or len(set(ids)) != 122:
        raise ValueError("The previously fixed 122 exclusions must be retained unchanged")
    inputs.update({str(excluded): sha(excluded), str(policy_path): sha(policy_path)})
    if read(policy_path).get("selection_policy") != "posthoc_error_selected_test_subset":
        raise ValueError("Missing explicit selected-subset provenance")
    token_counts = {}
    row_counts = {}
    native_counts = {}
    for split in ("train", "validation", "test"):
        manifest_path = parent / "caches" / f"{split}.manifest.json"
        manifest = read(manifest_path)
        if manifest.get("complete") is not True or manifest.get("adapter_sha256") is not None:
            raise ValueError("Completed unadapted last-token caches are required")
        inputs[str(manifest_path)] = sha(manifest_path)
        for key, digest in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            if sha(manifest[key]) != manifest[digest]:
                raise ValueError(f"Parent cache changed: {manifest[key]}")
            inputs[manifest[key]] = manifest[digest]
        row_counts[split] = int(manifest["shape"][0])
        # The pre-existing extraction plan contains only public size metadata;
        # no new test token payload is opened before the model-selection lock.
        prepared_manifest = parent / "parallel_features" / split / "prepared_manifest.json"
        prepared = read(prepared_manifest)
        inputs[str(prepared_manifest)] = sha(prepared_manifest)
        token_counts[split] = int(prepared.get("tokens", prepared.get("total_tokens", 0)))
        if split != "test":
            frame = pd.read_parquet(data / f"{split}.parquet", columns=[
                "source", "category_id", "type_label_mask", "prompt_harm_label"])
            categories, mask, harmful = contract.label_arrays(frame, require_type_mask=True)
            native = frame.source.isin(["chinese_curated", "jailbench"]).to_numpy()
            native_counts[split] = int((native & mask & harmful).sum())
    # Exact token counts are also available from the parent's registered
    # lengths file. This reads size/length metadata, not hidden-state payloads.
    if not all(token_counts.values()):
        import numpy as np
        for split in token_counts:
            lengths = parent / "parallel_features" / split / "lengths.npy"
            token_counts[split] = int(np.load(lengths, allow_pickle=False).sum())
            inputs[str(lengths)] = sha(lengths)
    expected_bytes = sum(token_counts.values()) * int(p["hidden_size"]) * 2
    if shutil.disk_usage(output.parent).free < expected_bytes + 24 * 1024**3:
        raise ValueError("Insufficient space for one token cache and head checkpoints")
    if native_counts != dict(train=17440, validation=2180):
        raise ValueError(f"Known native type-supervision scope changed: {native_counts}")
    codes = {str(path): sha(path) for path in source_files()}
    value = dict(schema=SCHEMA, study=str(output), parent_study=str(parent),
        data_dir=str(data), last_token_cache_dir=str(parent / "caches"),
        token_cache_dir=str(output / "token_cache"),
        data_sha256={split: inputs[str(data / f"{split}.parquet")] for split in row_counts},
        model_path=p["model_path"], model_type=p["model_type"], num_layers=32,
        hidden_size=int(p["hidden_size"]), use_lora=False, adapter_path=None,
        adapter_sha256=None, backbone_trainable_parameters=0, rows=row_counts,
        seeds=list(SEEDS), tasks=list(TASKS), arms=list(ARMS),
        fusion_layers=[8, 16, 24, 32], fusion_weights=[.25] * 4,
        learnable_layer_weights=False, best_layer_anchor=False,
        training=TRAINING, settings=TRAINING, execution=POLICY,
        extraction=p["extraction"], token_cache=dict(atol=0., rtol=0.,
            original_microbatches=True, full_tokens=True, truncation=False),
        token_counts=token_counts, expected_token_cache_bytes=expected_bytes,
        category_sources=["chinese_curated", "jailbench"],
        category_known_harmful_rows=dict(native_counts, test=2180),
        gated_native_test_rows=3704,
        excluded_ids_path=str(excluded), excluded_ids_sha256=sha(excluded),
        subset_dir=str(excluded.parent), subset_rows=8228,
        subset_policy="posthoc_error_selected_test_subset", blind_test=False,
        scopes=dict(binary="all three sources, original 8350 and fixed selected 8228 reported separately",
            category="known harmful native two-source rows only, 2180 test examples",
            classic_system="binary baseline/fusion gates crossed with category baseline/query heads on 3704 known native examples",
            category_fusion_system="same gate choices with the full four category heads, separately labeled"),
        checkpoint_selection="minimum validation cross entropy per arm",
        threshold_selection="validation maximum binary accuracy, plus fixed 0.5",
        budget_extension="common 12/24/40 continuation triggered only by the four task arms' validation CE",
        formal_test_barrier="all six task/seed selections and checkpoints locked before any new test token payload",
        inputs=inputs, code_sha256=codes,
        input_stats={name: dict(size=Path(name).stat().st_size,
                              mtime_ns=Path(name).stat().st_mtime_ns) for name in inputs},
        runtime=base.runtime_identity(), created_unix=time.time(),
        limitations=["The 122-row removal list was selected using previous model test errors; that subset is descriptive, not blind.",
                     "The earlier +2.10 pp belongs to a learned-weight, fixed-layer-set model, not these equal-weight arms.",
                     "Binary and five-class accuracy have different label scopes; their gains cannot be added or substituted.",
                     "Actual head parameter counts differ and are reported; previous component and capacity controls remain supplementary."])
    output.mkdir(parents=True)
    write(output / "protocol.json", value)
    write(output / "protocol_registration.json", dict(protocol_sha256=sha(output / "protocol.json")))
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for filename, digest in codes.items():
        target = snapshots / str(Path(filename).relative_to(ROOT)).replace("/", "__")
        shutil.copyfile(filename, target)
        if sha(target) != digest:
            raise RuntimeError("Source snapshot did not preserve registered bytes")
    write(output / "state.json", dict(state="registered", active_workers=[],
        automatic_run=False, formal_test_opened=False, updated_unix=time.time()))
    return verify(output)


def verify(study, full_hash=False):
    study = Path(study).resolve()
    if (study / "STOP").exists():
        raise RuntimeError("Study STOP requested")
    p = read(study / "protocol.json")
    if (p.get("schema") != SCHEMA or read(study / "protocol_registration.json")["protocol_sha256"] != sha(study / "protocol.json")):
        raise ValueError("Registered four-arm protocol changed")
    if p.get("use_lora") is not False or p["fusion_weights"] != [.25] * 4:
        raise ValueError("Backbone/fixed fusion policy changed")
    if "runtime" in p:
        from scripts.hanguard.frozen_base_readout_study import runtime_identity
        if runtime_identity() != p["runtime"]:
            raise ValueError("Pinned Python/library runtime changed")
    for filename, digest in p["code_sha256"].items():
        if sha(filename) != digest:
            raise ValueError(f"Registered implementation changed: {filename}")
    for filename, digest in p["inputs"].items():
        stat = Path(filename).stat()
        if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != p["input_stats"][filename]:
            raise ValueError(f"Registered input metadata changed: {filename}")
        if full_hash and sha(filename) != digest:
            raise ValueError(f"Registered input bytes changed: {filename}")
    return p


def selection_path(study, task, seed):
    return Path(study) / "heads" / task / f"s{seed}" / "selection.json"


def lock_selections(study):
    from scripts.hanguard.frozen_base_readout_study import validate_selection_lock
    study = Path(study).resolve()
    p = verify(study)
    if (study / "selection_lock.json").exists():
        return validate_selection_lock(study)
    artifacts, selections = {}, {}
    for task in p["tasks"]:
        for seed in p["seeds"]:
            path = selection_path(study, task, seed)
            selection = read(path)
            if (selection.get("task") != task or selection.get("seed") != seed or
                    selection.get("protocol_sha256") != sha(study / "protocol.json")):
                raise ValueError("Task selection is not bound to this protocol")
            selections[f"{task}/s{seed}"] = sha(path)
            artifacts[str(path)] = sha(path)
            for key, digest_key in (("checkpoint_path", "checkpoint_sha256"),
                                    ("validation_predictions_path", "validation_predictions_sha256")):
                target = Path(selection[key]).resolve()
                if not target.is_relative_to(study) or sha(target) != selection[digest_key]:
                    raise ValueError("Selected checkpoint/validation predictions changed")
                artifacts[str(target)] = selection[digest_key]
    lock = dict(schema=SCHEMA, complete=True, test_opened=False,
                protocol_sha256=sha(study / "protocol.json"), selections=selections,
                artifacts=artifacts, locked_unix=time.time())
    write(study / "selection_lock.json", lock)
    return validate_selection_lock(study)


def worker(args):
    study = args.output.resolve()
    p = verify(study)
    ledger_path = study / "workers" / f"{args.worker_id}.json"
    deadline = time.monotonic() + 15
    while not ledger_path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Worker identity was not registered")
        time.sleep(.05)
    ledger = read(ledger_path)
    if (not same_process(ledger["identity"], identity(os.getpid())) or
            ledger["phase"] != args.phase or ledger["task"] != args.task or
            ledger["split"] != args.split or ledger["seed"] != args.seed or
            ledger["device"] != args.device or ledger["gpu"] != args.gpu):
        raise ValueError("Worker command/task identity mismatch")
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            raise ValueError("Worker GPU binding differs")
        try:
            cards = inventory()
            row = next((r for r in cards if r["index"] == args.gpu), None)
        except Exception as error:
            write(ledger_path.with_suffix(".admission_retry.json"), dict(reason=repr(error), unix=time.time()))
            return 75
        if row is None or row["pids"] or row["free_gib"] < p["execution"]["minimum_free_gib"]:
            return 75
        import torch
        torch.cuda.init()
        readiness = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del readiness
    write(ledger_path.with_suffix(".admitted.json"), dict(
        admitted=True, identity=ledger["identity"], physical_gpu=args.gpu,
        protocol_sha256=sha(study / "protocol.json"), admitted_unix=time.time()))
    threads = p["execution"]["gpu_threads" if args.device == "cuda" else "cpu_threads"]
    import torch
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.phase == "extract":
        if args.device != "cuda":
            raise ValueError("Real backbone extraction requires an admitted GPU")
        from scripts.hanguard import four_arm_token_cache as cache
        cache.extract_worker(study, args.split, args.worker_id, device="cuda", admission_already_checked=True)
    else:
        from scripts.hanguard import four_arm_experiment_training as training
        function = training.fit_task if args.phase == "fit" else training.evaluate_task
        function(study, args.task, args.seed, device=args.device)
    write(ledger_path.with_suffix(".complete.json"), dict(complete=True, completed_unix=time.time()))
    return 0


class Scheduler:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.protocol = verify(study)
        self.active, self.arrivals = {}, set()
        self.workers = self.study / "workers"
        self.workers.mkdir(exist_ok=True)

    def state(self, name, **updates):
        prior = read(self.study / "state.json")
        value = dict(prior, state=name,
            active_workers=[{k: v for k, v in r.items() if k != "child"} for r in self.active.values()],
            automatic_run=True, coordinator_pid=os.getpid(), updated_unix=time.time())
        value.update(updates)
        write(self.study / "state.json", value)

    def cards(self):
        try:
            cards = inventory()
        except Exception as error:
            self.state("gpu_inventory_unavailable", gpu_inventory_error=repr(error))
            return []
        for record in self.active.values():
            if record["gpu"] is None:
                continue
            row = next((r for r in cards if r["index"] == record["gpu"]), None)
            for pid in row["pids"] if row else []:
                key = record["worker_id"], pid
                if pid != record["identity"]["pid"] and key not in self.arrivals:
                    self.arrivals.add(key)
                    write(self.study / "foreign_arrivals" / f"{time.time_ns()}_{pid}.json", dict(
                        action="record_only_continue_owned_worker", gpu=record["gpu"], pid=pid,
                        worker_id=record["worker_id"], unix=time.time()))
        return cards

    def launch(self, phase, *, device, gpu=None, split=None, task=None, seed=None):
        worker_id = f"{phase}_{split or task}_{seed or 'none'}_{device}{gpu if gpu is not None else ''}_{time.time_ns()}"
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(self.study),
                   "--worker-id", worker_id, "--phase", phase, "--device", device]
        for key, value in (("gpu", gpu), ("split", split), ("task", task), ("seed", seed)):
            if value is not None:
                command.extend([f"--{key}", str(value)])
        threads = self.protocol["execution"]["gpu_threads" if device == "cuda" else "cpu_threads"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                   OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                   PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (self.workers / f"{worker_id}.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        exact = None
        for _ in range(100):
            if child.poll() is not None:
                break
            try:
                current = identity(child.pid)
                if current["argv"] == command and current["uid"] == os.getuid() and current["cwd"] == str(ROOT):
                    exact = current
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.05)
        if exact is None:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=20)
            raise RuntimeError("Cannot register new worker identity")
        record = dict(worker_id=worker_id, identity=exact, phase=phase, device=device,
                      gpu=gpu, split=split, task=task, seed=seed, command=command,
                      protocol_sha256=sha(self.study / "protocol.json"))
        write(self.workers / f"{worker_id}.json", record)
        self.active[worker_id] = dict(record, child=child)

    def poll(self):
        if (self.study / "STOP").exists():
            raise RuntimeError("Study STOP requested")
        for worker_id, record in list(self.active.items()):
            code = record["child"].poll()
            if code is None:
                # This unreaped Popen child cannot have a reused PID. Avoid an
                # exiting /proc identity race; exact identity was bound at birth.
                continue
            write(self.workers / f"{worker_id}.exit.json", dict(exit_code=code, exited_unix=time.time()))
            del self.active[worker_id]
            if code == 75 and not (self.workers / f"{worker_id}.admitted.json").exists():
                continue
            if code != 0 or not (self.workers / f"{worker_id}.complete.json").exists():
                raise RuntimeError(f"{worker_id} failed with exit {code}; inspect its log")

    def stop_owned(self):
        for record in self.active.values():
            if record["child"].poll() is None:
                record["child"].terminate()
        for record in self.active.values():
            try:
                record["child"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                record["child"].kill()
                record["child"].wait(timeout=10)
        self.active.clear()

    def extract(self, split):
        from scripts.hanguard import four_arm_token_cache as cache
        cache.prepare(self.study, split)
        while True:
            self.poll()
            progress = cache.progress(self.study, split)
            done = cache.all_complete(self.study, split)
            if done and not self.active:
                cache.completion_manifest_finalize(self.study, split)
                return
            cards = self.cards()
            available = empty_cards(cards, list(range(8)), self.protocol["execution"]["minimum_free_gib"])
            occupied = {r["gpu"] for r in self.active.values()}
            remaining_slots = max(0, progress["total_chunks"] - progress["complete_chunks"] - len(self.active))
            if not done:
                for gpu in available:
                    if gpu not in occupied and remaining_slots > 0:
                        self.launch("extract", device="cuda", gpu=gpu, split=split)
                        remaining_slots -= 1
            self.state(f"extract_{split}", extraction_progress=progress, gpu_inventory=cards)
            time.sleep(3)

    def tasks(self, phase):
        from scripts.hanguard import four_arm_experiment_training as training
        while True:
            self.poll()
            pending = []
            for task in self.protocol["tasks"]:
                for seed in self.protocol["seeds"]:
                    if phase == "fit":
                        done = selection_path(self.study, task, seed).exists()
                    else:
                        done = (self.study / "heads" / task / f"s{seed}" / "test_complete.json").exists()
                        if done:
                            training.verify_evaluation(self.study, task, seed)
                    if not done and not any(r["task"] == task and r["seed"] == seed for r in self.active.values()):
                        pending.append((task, seed))
            if not pending and not self.active:
                return
            cards = self.cards()
            occupied = {r["gpu"] for r in self.active.values()}
            for gpu in empty_cards(cards, list(range(8)), self.protocol["execution"]["minimum_free_gib"]):
                if pending and gpu not in occupied:
                    task, seed = pending.pop(0)
                    self.launch(phase, device="cuda", gpu=gpu, task=task, seed=seed)
            cpu_count = sum(r["device"] == "cpu" for r in self.active.values())
            if pending and cpu_count < self.protocol["execution"]["max_cpu_workers"]:
                task, seed = pending.pop(0)
                self.launch(phase, device="cpu", task=task, seed=seed)
            self.state(phase, pending_tasks=[dict(task=t, seed=s) for t, s in pending], gpu_inventory=cards)
            time.sleep(3)


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
            raise RuntimeError(f"A prior exact worker is still alive: {expected['pid']}")


def run(study):
    from scripts.hanguard import four_arm_token_cache as cache
    from scripts.hanguard import four_arm_experiment_training as training
    study = Path(study).resolve()
    verify(study, full_hash=True)
    with (study / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(study / "state.json")["state"] == "completed":
            return read(study / "results.json")
        verify_retired(study)
        for split in ("train", "validation", "test"):
            if (study / "token_cache" / split / "plan.json").exists():
                cache.release_dead_claims(study, split)
        scheduler = Scheduler(study)
        try:
            scheduler.extract("train")
            scheduler.extract("validation")
            scheduler.tasks("fit")
            lock_selections(study)
            scheduler.state("test_prepared", formal_test_opened=True)
            scheduler.extract("test")
            scheduler.tasks("evaluate")
            result = training.summarize(study)
            if result.get("complete") is not True:
                raise RuntimeError("The complete six-task experiment report is missing")
            scheduler.state("completed", automatic_run=False)
            packager = ROOT / "scripts/hanguard/package_four_arm_experiments.py"
            subprocess.run([sys.executable, str(packager)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(packager), "--verify"], cwd=ROOT, check=True)
            return result
        except BaseException as error:
            scheduler.stop_owned()
            scheduler.state("requires_review", automatic_run=False, error_type=type(error).__name__, reason=str(error))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "run", "worker", "status"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--worker-id")
    p.add_argument("--phase", choices=("extract", "fit", "evaluate"))
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--gpu", type=int)
    p.add_argument("--split", choices=("train", "validation", "test"))
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "init":
        result = initialize(args.output, args.parent)
        print({"registered": True, "output": str(args.output), "token_cache_bytes": result["expected_token_cache_bytes"]})
    elif args.action == "run":
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        run(args.output)
    elif args.action == "status":
        print(read(args.output / "state.json"))
    else:
        if not args.worker_id or not args.phase or not args.device:
            raise ValueError("worker requires worker-id/phase/device")
        raise SystemExit(worker(args))


if __name__ == "__main__":
    main()
