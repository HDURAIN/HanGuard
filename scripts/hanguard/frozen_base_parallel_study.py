"""Run the frozen-base comparison using every genuinely free public GPU.

Feature extraction preserves each original microbatch and distributes complete
groups across independent, identity-bound workers. The readout implementation,
validation selections and final test barrier remain the same as the serial study.
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

PARALLEL_SOURCES = (
    "scripts/hanguard/frozen_base_parallel_study.py",
    "scripts/hanguard/frozen_base_parallel_features.py",
    "scripts/hanguard/frozen_base_parallel_queue.py",
    "scripts/hanguard/frozen_base_readout_admission_v2.py",
)


def initialize(args):
    """Register all scientific inputs and source bytes before any GPU launch."""
    import pandas as pd
    import pyarrow.parquet as pq

    study = Path(args.output).resolve()
    if study.exists() and any(study.iterdir()):
        raise FileExistsError("Use a fresh study directory; prior runs stay immutable")
    model, data = Path(args.model).resolve(), Path(args.data).resolve()
    release = base.contract.release_contract(data)
    if not release:
        raise ValueError("A completed remediated dataset release is required")
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    depth, hidden = int(text["num_hidden_layers"]), int(text["hidden_size"])
    if depth != 32:
        raise ValueError("This comparison requires 32 transformer blocks")
    shards = sorted(set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values()))
    model_files = [model / name for name in shards]
    model_files += [p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".txt")]
    inputs = {str(p): base.sha(p) for p in sorted(set(model_files))}
    inputs.update({str(data / f"{split}.parquet"): base.sha(data / f"{split}.parquet") for split in base.SPLITS})
    for name in ("manifest.json", "release_qa.json", "split_policy.json"):
        inputs[str(data / name)] = base.sha(data / name)
    columns = ["base_id", "group_id", "source", "prompt_harm_label", "split", "category_id", "type_label_mask"]
    frames = {s: pd.read_parquet(data / f"{s}.parquet", columns=columns) for s in ("train", "validation")}
    for split, frame in frames.items():
        base.validate_metadata(frame, split)
    # Before the selection lock, bind test bytes and identities without opening
    # its text or labels. This is the same boundary as the serial study.
    test_ids = pd.read_parquet(data / "test.parquet", columns=["base_id", "group_id"])
    if not test_ids.base_id.is_unique or test_ids.isna().any().any():
        raise ValueError("Test identity metadata is invalid")
    identities = dict(frames, test=test_ids)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("base_id", "group_id"):
            if set(identities[a][field]) & set(identities[b][field]):
                raise ValueError(f"{a}/{b}: {field} overlap")
    rows = {s: pq.ParquetFile(data / f"{s}.parquet").metadata.num_rows for s in base.SPLITS}
    feature_bytes = sum(rows.values()) * depth * hidden * 2
    # Keep completed chunks for audit as well as the canonical merged cache.
    if shutil.disk_usage(study.parent).free < 2 * feature_bytes + 8 * 1024**3:
        raise ValueError("Insufficient disk for parallel chunks, merged caches and checkpoints")
    source_names = tuple(dict.fromkeys(base.SOURCE_NAMES + PARALLEL_SOURCES))
    sources = {str(ROOT / name): base.sha(ROOT / name) for name in source_names}
    value = dict(
        schema=base.SCHEMA, model_path=str(model), data_dir=str(data),
        dataset_release=release, model_type=config["model_type"], num_layers=depth,
        hidden_size=hidden, rows=rows, inputs=inputs,
        input_stats={p: dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) for p in inputs},
        code_sha256=sources, runtime=base.runtime_identity(),
        settings=dict(seed=42, width=128, dropout=.1, batch_size=256, lr=.001,
                      weight_decay=.001, grad_clip=5., epochs=12, max_epochs=40,
                      layers=[8, 16, 24, 32], bootstrap_draws=2000),
        extraction=dict(token_budget=16384, max_micro=64, pad_multiple=32,
                        max_tokens=4096, pooling="last_valid_token", raw_block_outputs=True,
                        dtype="float16", truncation=False),
        adapter_path=None, adapter_sha256=None, use_lora=False,
        backbone_trainable_parameters=0, fresh_features_required=True,
        methods=["last", "best_single", "fusion_fixed", "repeat_best", "fusion_fixed_best_anchor"],
        best_single_definition="Train 32 independent same-capacity MLPs; select layer/checkpoint by validation BCE only",
        checkpoint_selection="minimum validation BCE independently per layer/arm",
        threshold_selection="validation maximum accuracy; fixed 0.5 also reported",
        evaluation_scope="Fresh full train/validation readout fitting, one locked current-remediated test evaluation",
        formal_test_enabled=True, formal_test_barrier_required=True,
        no_test_layer_or_checkpoint_selection=True,
        execution=dict(single_identity_bound_gpu_worker=False,
            identity_bound_parallel_gpu_workers=True, max_cuda_workers=8,
            candidate_gpus=list(range(8)), reserved_gpus=[],
            dynamic_empty_gpu_admission=True, automatic_retry=False,
            automatic_head_retry=False, automatic_feature_redispatch=True,
            never_terminate_foreign_processes=True,
            cuda_startup="Exact worker ledger and empty card check before early CUDA allocation",
            full_content_hash_required_before_model_or_data_use=True,
            incomplete_attempt_reuse=False),
        parallel_features=dict(group_chunk_size=32, preserve_original_microbatches=True,
            claim_under_flock=True, partial_chunk_reuse=False,
            completed_chunks_preserved_after_resource_loss=True,
            cross_worker_calibration="identical FP16 bytes for common original batches",
            final_cache_layout="original base_id order; one row written exactly once"),
        expected_cache_bytes_without_headers=feature_bytes,
        limitations=[
            "Prior dataset identities were explored and regrouped; this is not pristine external blind confirmation.",
            "One fixed head seed measures this run, not cross-seed reproducibility.",
            "Selected-layer and multi-layer differences must be reported separately from capacity effects.",
            "Full-train readouts cannot be directly substituted for previous B-subset OOF scores.",
        ], created_unix=time.time(),
    )
    study.mkdir(parents=True, exist_ok=True)
    base.dump(study / "protocol.json", value)
    snapshots = study / "source_snapshot"
    snapshots.mkdir()
    for filename, digest in sources.items():
        target = snapshots / str(Path(filename).relative_to(ROOT)).replace("/", "__")
        shutil.copy2(filename, target)
        if base.sha(target) != digest:
            raise ValueError("Source changed while snapshotting")
    base.dump(study / "protocol_registration.json", dict(
        protocol_sha256=base.sha(study / "protocol.json"),
        source_snapshot_sha256={str(p.relative_to(study)): base.sha(p) for p in snapshots.iterdir()},
    ))
    base.dump(study / "state.json", dict(state="registered", training_started=False,
        backbone_training_started=False, formal_validation_opened=False,
        formal_test_opened=False, protocol_sha256=base.sha(study / "protocol.json")))
    return value


def worker_command(study, worker_id, *, split=None, phase=None):
    python = str((ROOT / ".venv-hanguard/bin/python").absolute())
    if split is not None:
        return [python, str(ROOT / "scripts/hanguard/frozen_base_parallel_features.py"),
                "feature-worker", "--study", str(study), "--worker-id", worker_id, "--split", split]
    return [python, str(Path(__file__).resolve()), "head-worker", "--study", str(study),
            "--worker-id", worker_id, "--phase", phase]


def head_worker(study, worker_id, phase):
    if phase not in ("fit", "evaluate"):
        raise ValueError("Invalid readout phase")
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_readout_heads as heads
    import torch

    base.load_protocol(study)
    # Same exact worker holds the real CUDA context during slow input hashing.
    features.cuda_startup(study, worker_id, split=None)
    base.load_protocol(study, full_hash=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    original_stop = heads.check_stop
    def worker_stop(path):
        original_stop(path)
        features.check_stop(path, worker_id)
    heads.check_stop = worker_stop
    try:
        worker_stop(study)
        if phase == "fit":
            base.mark_state(study, "fitting_readouts", training_started=True)
            heads.fit(study, device="cuda")
            base.validate_selection_lock(study)
        else:
            base.validate_selection_lock(study)
            heads.evaluate(study, device="cuda")
    finally:
        heads.check_stop = original_stop
    base.dump(Path(study) / "parallel_execution" / "workers" / f"{worker_id}.complete.json",
              dict(complete=True, phase=phase, pid=os.getpid(), protocol_sha256=base.sha(Path(study) / "protocol.json")))


def _run(study):
    """Coordinate CPU merges and unchanged readouts between parallel phases."""
    from scripts.hanguard import frozen_base_parallel_features as features
    from scripts.hanguard import frozen_base_parallel_queue as queue

    study = Path(study).resolve()
    base.load_protocol(study, full_hash=True)
    queue.initialize(study)
    for split in ("train", "validation"):
        base.mark_state(study, f"preparing_{split}",
            **({"formal_validation_opened": True} if split == "validation" else {}))
        features.prepare_split(study, split)
        base.mark_state(study, f"extracting_{split}")
        queue.run_workers(study, f"extract_{split}",
            lambda worker_id, gpu: worker_command(study, worker_id, split=split),
            lambda: features.split_complete(study, split),
            work_remaining=lambda: features.has_work(study, split), split=split, max_workers=8,
            on_resource_loss=lambda worker_id, gpu, lost_split, reason:
                features.abandon_worker(study, lost_split, worker_id, reason))
        base.mark_state(study, f"merging_{split}")
        features.merge_split(study, split)
    base.mark_state(study, "waiting_for_readout_gpu")
    queue.run_single(study, "fit", lambda worker_id, gpu: worker_command(study, worker_id, phase="fit"))
    base.validate_selection_lock(study)
    base.mark_state(study, "validation_selections_locked")
    # This is the first test prompt/label preparation, after all five selections.
    features.prepare_split(study, "test")
    base.mark_state(study, "extracting_test", formal_test_opened=True)
    queue.run_workers(study, "extract_test",
        lambda worker_id, gpu: worker_command(study, worker_id, split="test"),
        lambda: features.split_complete(study, "test"),
        work_remaining=lambda: features.has_work(study, "test"), split="test", max_workers=8,
        on_resource_loss=lambda worker_id, gpu, lost_split, reason:
            features.abandon_worker(study, lost_split, worker_id, reason))
    base.mark_state(study, "merging_test")
    features.merge_split(study, "test")
    base.mark_state(study, "evaluating_locked_test")
    queue.run_single(study, "evaluate", lambda worker_id, gpu: worker_command(study, worker_id, phase="evaluate"))
    result = study / "analysis/results.json"
    if not result.exists() or json.loads(result.read_text()).get("complete") is not True:
        raise ValueError("Readout evaluation did not publish a completed result")
    base.mark_state(study, "completed", backbone_training_started=False)


def run(study):
    study = Path(study).resolve()
    with (study / "parallel_coordinator.pid").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another coordinator owns this study") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        # A crashed scientific phase is retained rather than silently retried.
        previous = json.loads((study / "state.json").read_text())
        if previous.get("state") != "registered":
            raise RuntimeError("A coordinator attempt already started; review it before any continuation")
        _run(study)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run", "head-worker", "status"))
    parser.add_argument("--output", "--study", dest="output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3.5-4B")
    parser.add_argument("--data", type=Path, default=ROOT / "data/three_source_remediated_20261012")
    parser.add_argument("--worker-id")
    parser.add_argument("--phase", choices=("fit", "evaluate"))
    args = parser.parse_args(argv)
    if args.action == "init":
        protocol = initialize(args)
        print(json.dumps(dict(state="registered", output=str(args.output), rows=protocol["rows"],
            model_path=protocol["model_path"], use_lora=False, max_cuda_workers=8,
            reserved_gpus=[]), ensure_ascii=False), flush=True)
    elif args.action == "run":
        try:
            run(args.output)
        except BaseException as exc:
            if (args.output / "state.json").exists():
                base.mark_state(args.output, "stopped" if (args.output / "STOP").exists()
                    or isinstance(exc, KeyboardInterrupt) else "requires_review",
                    error_type=type(exc).__name__, reason=str(exc), automatic_retry=False)
            raise
    elif args.action == "head-worker":
        if not args.worker_id or not args.phase:
            parser.error("head-worker requires --worker-id and --phase")
        head_worker(args.output.resolve(), args.worker_id, args.phase)
    else:
        print((args.output / "state.json").read_text())


if __name__ == "__main__":
    main()
