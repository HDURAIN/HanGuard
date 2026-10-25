"""Read-only, identity-bound replay of a failed four-arm train feature check.

This separate diagnostic does not amend the study, publish production features,
open test payloads, or relax exact-equality gates. Only small numeric reports are
written to a separate output directory. A scheduler must register this exact
child before CUDA initialization; an occupied or unknown GPU returns exit 75.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import four_arm_experiment_study as study_module
from scripts.hanguard import four_arm_token_cache as tokens
from scripts.hanguard import frozen_base_parallel_features as original
from scripts.hanguard import frozen_base_readout_study as base
from scripts.hanguard.ab_readout_parallel_queue import gpu_inventory, process_identity

SCHEMA = "hanguard_four_arm_feature_mismatch_diagnostic_v1"
TARGET_GROUPS = (860, 861, 862, 863, 864, 865)
IDENTITY_FIELDS = ("pid", "uid", "start_time_ticks", "command_sha256", "argv", "cwd")


def compare_values(actual, expected):
    """Bounded-memory numeric summary; never write example feature vectors."""
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison requires matching finite shapes")
    aa, bb = a.reshape(-1), b.reshape(-1)
    count = len(aa)
    changed = 0
    maximum = absolute = squares = dot = anorm = bnorm = 0.0
    for start in range(0, count, 262144):
        x = aa[start:start + 262144].astype(np.float64)
        y = bb[start:start + 262144].astype(np.float64)
        delta = x - y
        changed += int(np.count_nonzero(delta))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
        absolute += float(np.abs(delta).sum())
        squares += float(np.dot(delta, delta))
        dot += float(np.dot(x, y))
        anorm += float(np.dot(x, x))
        bnorm += float(np.dot(y, y))
    cosine = dot / np.sqrt(anorm * bnorm) if anorm and bnorm else (1.0 if not anorm and not bnorm else 0.0)
    return dict(equal=bool(np.array_equal(a, b)), shape=list(a.shape), count=count,
                different_count=changed, max_abs=maximum,
                mean_abs=absolute / max(count, 1), rmse=float(np.sqrt(squares / max(count, 1))),
                cosine=float(np.clip(cosine, -1, 1)), atol=0, rtol=0)


def comparison(actual, expected, row_lengths=None):
    record = dict(overall=compare_values(actual, expected))
    if row_lengths is None:
        row_lengths = np.ones(len(actual), dtype=np.int64)
    lengths = np.asarray(row_lengths, np.int64)
    if np.any(lengths < 1) or int(lengths.sum()) != len(actual):
        raise ValueError("Comparison row offsets do not cover the feature array")
    offsets = np.r_[0, np.cumsum(lengths)]
    record["rows"] = [dict(row_in_group=i, **compare_values(actual[s:e], expected[s:e]))
                      for i, (s, e) in enumerate(zip(offsets[:-1], offsets[1:]))]
    return record


def target_groups(group_count):
    if group_count <= max(TARGET_GROUPS):
        raise ValueError("Registered train plan lacks requested diagnostic groups")
    return sorted(set((0, group_count // 2, group_count - 1, *TARGET_GROUPS)))


def verify_ledger(args, protocol):
    """No torch CUDA API is called here, including on rejected admission."""
    deadline = time.monotonic() + 15
    while not args.admission_ledger.exists():
        if time.monotonic() >= deadline:
            raise ValueError("Diagnostic exact child ledger was not registered")
        time.sleep(.05)
    ledger = original.read(args.admission_ledger)
    actual = process_identity(os.getpid())
    expected = ledger.get("identity", {})
    if (any(expected.get(key) != actual.get(key) for key in IDENTITY_FIELDS)
            or actual["uid"] != os.getuid()
            or ledger.get("run_id") != args.run_id
            or ledger.get("study") != str(args.study)
            or ledger.get("output") != str(args.output)
            or ledger.get("gpu") != args.gpu
            or ledger.get("protocol_sha256") != base.sha(args.study / "protocol.json")
            or ledger.get("diagnostic_script_sha256") != base.sha(Path(__file__).resolve())):
        raise ValueError("Diagnostic exact child/command/source identity differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("Diagnostic CUDA binding differs from registered physical GPU")
    protected = [args.study, Path(protocol["parent_study"]), Path(protocol["model_path"]),
                 Path(protocol["data_dir"])]
    if any(args.output.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Diagnostic output must be separate from immutable study/model/data")
    return actual


def empty_gpu(gpu, minimum_free_gib):
    """Unknown telemetry blocks admission, even when no PID is reported."""
    try:
        row = next((row for row in gpu_inventory() if row["index"] == gpu), None)
        if row is None or row["pids"] or row["free_gib"] < minimum_free_gib:
            return None
        return row
    except Exception:
        return None


def verify_train_inputs(protocol):
    """Fully hash model and train/validation inputs; never hash test payloads."""
    model = Path(protocol["model_path"]).resolve()
    data = Path(protocol["data_dir"]).resolve()
    parent = Path(protocol["parent_study"]).resolve()
    checked = {}
    for filename, digest in protocol["inputs"].items():
        path = Path(filename).resolve()
        allowed = (path.is_relative_to(model) or path in (data / "train.parquet", data / "validation.parquet", parent / "protocol.json")
                   or any(path.is_relative_to(parent / folder) and
                          (path.name.startswith("train.") or path.name.startswith("validation."))
                          for folder in ("caches",)))
        if allowed:
            if base.sha(path) != digest:
                raise ValueError(f"Pinned diagnostic input changed: {path}")
            checked[str(path)] = digest
    if not any(Path(name).is_relative_to(model) for name in checked):
        raise ValueError("No pinned model artifacts were verified")
    return checked


def parent_chunks(parent, prepared, plan, merged, numbers):
    """Verify original raw chunk bytes and compare selected rows with merge."""
    audit_path = parent / "caches/train.replay_audit.json"
    manifest = original.read(parent / "caches/train.manifest.json")
    if base.sha(audit_path) != manifest["replay_audit_sha256"]:
        raise ValueError("Parent merge audit bytes changed")
    audit = original.read(audit_path)
    receipts = {r["chunk_id"]: r for r in audit["complete_chunk_receipts"]}
    chunks = {}
    for number in numbers:
        chunk = next(c for c in plan["chunks"] if c["group_start"] <= number < c["group_stop"])
        cid = chunk["chunk_id"]
        if cid not in chunks:
            pinned = receipts[cid]
            expected_receipt = original.chunk_directory(parent, "train", cid) / "receipt.json"
            if Path(pinned["receipt_path"]).resolve() != expected_receipt.resolve() or base.sha(expected_receipt) != pinned["receipt_sha256"]:
                raise ValueError("Parent chunk receipt differs from merge audit")
            receipt = original._quick_receipt(parent, "train", prepared, chunk)
            for field, hash_field in (("path", "sha256"), ("row_indices_path", "row_indices_sha256")):
                path = Path(receipt[field]).resolve()
                expected_root = expected_receipt.parent / f'attempt_{receipt["attempt"]:03d}'
                if not path.is_relative_to(expected_root.resolve()) or base.sha(path) != receipt[hash_field]:
                    raise ValueError("Parent raw chunk artifact changed")
            rows = np.load(receipt["row_indices_path"], allow_pickle=False)
            feature = np.load(receipt["path"], mmap_mode="r", allow_pickle=False)
            expected_rows = np.concatenate([np.asarray(g, np.int64)
                for g in plan["groups"][chunk["group_start"]:chunk["group_stop"]]])
            if (not np.array_equal(rows, expected_rows) or len(np.unique(rows)) != len(rows)
                    or feature.shape != (len(rows), *plan["shape"][1:]) or feature.dtype != np.float16):
                raise ValueError("Parent raw chunk row/feature identity changed")
            chunks[cid] = dict(feature=feature, positions={int(row): i for i, row in enumerate(rows)},
                               receipt_path=str(expected_receipt), receipt_sha256=pinned["receipt_sha256"])
    result = {}
    for number in numbers:
        chunk = next(c for c in plan["chunks"] if c["group_start"] <= number < c["group_stop"])
        record = chunks[chunk["chunk_id"]]
        group = np.asarray(plan["groups"][number], np.int64)
        selected = np.asarray([record["positions"][int(row)] for row in group])
        result[number] = dict(chunk_id=chunk["chunk_id"], receipt_path=record["receipt_path"],
            receipt_sha256=record["receipt_sha256"],
            comparison=comparison(np.asarray(record["feature"][selected]), np.asarray(merged[group])))
    return result


@torch.inference_mode()
def simultaneous_capture(backbone, batch):
    """Activate the unchanged old and new hooks in one identical forward."""
    raw = tokens.RawLastTokenCapture(backbone)
    old = base.LastBlockCapture(backbone)
    try:
        raw.mask = batch["attention_mask"].bool()
        old.indices = base.last_true_indices(batch["attention_mask"])
        kind = batch["input_ids"].device.type
        with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
            backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0), use_cache=False)
        if raw.values is None or set(old.values) != set(range(old.count)):
            raise ValueError("Simultaneous hooks omitted expected outputs")
        return raw.values.numpy(), torch.stack([old.values[i] for i in range(old.count)], dim=1).cpu().numpy()
    finally:
        raw.mask = old.indices = None
        raw.close()
        old.close()


def replay_group(backbone, batch, lengths, expected):
    """Keep hook modes separate so retained old GPU tensors cannot alter raw mode."""
    with ExitStack() as cleanup:
        raw = tokens.RawLastTokenCapture(backbone)
        cleanup.callback(raw.close)
        first_raw, second_raw = raw(batch).numpy(), raw(batch).numpy()
    with ExitStack() as cleanup:
        old = base.LastBlockCapture(backbone)
        cleanup.callback(old.close)
        first_old, second_old = old(batch).numpy(), old(batch).numpy()
    both_raw, both_old = simultaneous_capture(backbone, batch)
    ends = np.cumsum(lengths, dtype=np.int64) - 1
    last = expected[:, -1, :]
    records = {
        "raw_full_token_repeat": comparison(first_raw, second_raw, lengths),
        "raw_last_vs_parent": comparison(first_raw[ends], last),
        "raw_repeat_last_vs_parent": comparison(second_raw[ends], last),
        "old_all_layers_repeat": comparison(first_old, second_old),
        "old_all_layers_vs_parent": comparison(first_old, expected),
        "old_repeat_all_layers_vs_parent": comparison(second_old, expected),
        "raw_vs_old_last": comparison(first_raw[ends], first_old[:, -1, :]),
        "simultaneous_raw_last_vs_old_last": comparison(both_raw[ends], both_old[:, -1, :]),
        "simultaneous_raw_vs_separate_raw": comparison(both_raw, first_raw, lengths),
        "simultaneous_old_vs_separate_old": comparison(both_old, first_old),
        "simultaneous_raw_last_vs_parent": comparison(both_raw[ends], last),
        "simultaneous_old_all_layers_vs_parent": comparison(both_old, expected),
    }
    return dict(passed=all(r["overall"]["equal"] for r in records.values()), comparisons=records)


def run(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("Unsafe diagnostic run identifier")
    protocol = study_module.verify(args.study, full_hash=False)
    actual = verify_ledger(args, protocol)
    ledger = original.read(args.admission_ledger)
    source_sha = base.sha(Path(__file__).resolve())
    card = empty_gpu(args.gpu, protocol["execution"]["minimum_free_gib"])
    if card is None:
        return 75
    # This is the admitted real replay worker, never a GPU placeholder.
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible physical GPU is required")
    ready = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del ready
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "result.json"
    if report_path.exists():
        raise FileExistsError("Diagnostic report already exists; use a fresh run output")
    original.dump(args.output / "admitted.json", dict(schema=SCHEMA, run_id=args.run_id,
        study=str(args.study), output=str(args.output), gpu=args.gpu,
        admitted=True, identity=actual, physical_gpu=args.gpu,
        protocol_sha256=base.sha(args.study / "protocol.json"),
        diagnostic_script_sha256=source_sha,
        recovery_protocol_sha256=ledger.get("recovery_protocol_sha256"), admitted_unix=time.time(),
        admission_inventory=card))
    torch.set_num_threads(protocol["execution"]["gpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = dict(schema=SCHEMA, run_id=args.run_id, study=str(args.study),
        output=str(args.output), gpu=args.gpu, split="train",
        diagnostic_only=True, production_cache_written=False, test_payloads_read=False,
        zero_tolerance_unchanged=True, protocol_sha256=base.sha(args.study / "protocol.json"),
        diagnostic_script_sha256=source_sha, recovery_protocol_sha256=ledger.get("recovery_protocol_sha256"),
        identity=actual, physical_gpu=args.gpu, runtime=base.runtime_identity(), groups=[], complete=False,
        causal_conclusion=None)
    try:
        checked = verify_train_inputs(protocol)
        report["fully_hashed_inputs"] = checked
        parent = Path(protocol["parent_study"]).resolve()
        pp = base.load_protocol(parent, full_hash=False)
        if (pp.get("use_lora") is not False or pp.get("adapter_path") is not None
                or Path(pp["model_path"]).resolve() != Path(protocol["model_path"]).resolve()):
            raise ValueError("Parent is not the same pinned unadapted model")
        prepared, plan, data = original.load_prepared(parent, "train", protocol=pp)
        numbers = target_groups(len(plan["groups"]))
        previous = tokens._legacy_last(protocol, "train")
        report["parent_raw_chunk_vs_merged"] = parent_chunks(parent, prepared, plan, previous, numbers)
        original.dump(report_path, report)
        model, backbone, tokenizer = base.load_frozen_backbone(pp)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            for number in numbers:
                study_module.verify(args.study, full_hash=False)
                group = np.asarray(plan["groups"][number], np.int64)
                batch = base.legacy.make_batch(data, group, tokenizer, pp["extraction"], device="cuda")
                lengths = data["lengths"][group]
                item = dict(group=number, rows=len(group), row_indices=group.tolist(),
                    min_tokens=int(lengths.min()), max_tokens=int(lengths.max()),
                    padded_tokens=int(batch["input_ids"].shape[1]),
                    **replay_group(backbone, batch, lengths, np.asarray(previous[group])))
                report["groups"].append(item)
                original.dump(report_path, report)
                del batch
            report["complete"] = True
            report["passed"] = (all(g["passed"] for g in report["groups"]) and all(
                record["comparison"]["overall"]["equal"] for record in report["parent_raw_chunk_vs_merged"].values()))
            report["completed_unix"] = time.time()
            original.dump(report_path, report)
            return 0 if report["passed"] else 1
        finally:
            del backbone, model
    except Exception as error:
        report.update(complete=False, passed=False, error=repr(error), traceback=traceback.format_exc())
        original.dump(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--admission-ledger", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    for name in ("study", "output", "admission_ledger"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
