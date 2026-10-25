"""Registered, resumable 2 x 2 frozen-base experiments.

The binary and native five-category objectives are separate experiments.  The
four arms in each experiment share batches, optimizer settings and validation
selection; no test feature is read until all six task/seed fits are hash locked.
Only identities, labels and predictions are exported, never prompt text.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.hanguard.four_arm_query_heads import ARMS, LAYERS, FourArmHead
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    accuracy_threshold, binary_metrics, group_cluster_bootstrap,
)

SCHEMA = "hanguard_four_arm_experiments_v1"
RESUME_SCHEMA = "hanguard_four_arm_epoch_resume_v1"
NATIVE_SOURCES = ("chinese_curated", "jailbench")
TASKS = ("binary", "category")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _check_stop(study):
    study = Path(study)
    for path in (study / "STOP", study / "heads" / "STOP"):
        if path.exists():
            raise RuntimeError(f"Study stopped: {path}")


def settings(protocol):
    raw = protocol.get("training", protocol.get("settings", {}))
    value = dict(width=int(raw.get("width", raw.get("head_width", 128))),
                 lr=float(raw.get("lr", .001)),
                 weight_decay=float(raw.get("weight_decay", raw.get("wd", .001))),
                 dropout=float(raw.get("dropout", .1)),
                 epochs=int(raw.get("epochs", 12)), max_epochs=int(raw.get("max_epochs", 40)),
                 token_budget=int(raw.get("token_budget", 8192)),
                 max_micro=int(raw.get("max_micro", 32)),
                 effective_batch=int(raw.get("effective_batch", 128)),
                 grad_clip=float(raw.get("grad_clip", raw.get("clip", 5))),
                 pad_multiple=int(raw.get("pad_multiple", 1)),
                 bootstrap_draws=int(raw.get("bootstrap_draws", 2000)))
    if (min(value[k] for k in ("width", "epochs", "max_micro", "effective_batch",
                              "token_budget", "pad_multiple", "bootstrap_draws")) < 1
            or not value["epochs"] <= value["max_epochs"] <= 40
            or value["lr"] <= 0 or value["weight_decay"] < 0
            or value["grad_clip"] <= 0 or not 0 <= value["dropout"] < 1):
        raise ValueError("Invalid four-arm training settings")
    return value


def read_protocol(study):
    study = Path(study).resolve()
    value = read_json(study / "protocol.json")
    registration_path = study / "protocol_registration.json"
    if registration_path.exists() and read_json(registration_path).get("protocol_sha256") != sha(study / "protocol.json"):
        raise ValueError("Registered four-arm protocol changed")
    for name, digest in value.get("code_sha256", {}).items():
        if sha(name) != digest:
            raise ValueError(f"Registered source changed: {name}")
    if set(value.get("tasks", TASKS)) != set(TASKS):
        raise ValueError("Both binary and category tasks must be registered")
    seeds = value.get("seeds", [42, 43, 44])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Distinct nonnegative seeds are required")
    if set(value.get("data_sha256", {})) != {"train", "validation", "test"}:
        raise ValueError("Register all three original data hashes")
    if value.get("fusion_layers", value.get("layers", list(LAYERS))) != list(LAYERS):
        raise ValueError("Fusion layers must remain L8/L16/L24/L32")
    settings(value)
    return value


def task_folder(study, task, seed):
    if task not in TASKS:
        raise ValueError("Unknown task")
    return Path(study).resolve() / "heads" / task / f"s{int(seed)}"


def _sources():
    paths = [Path(__file__), Path(__file__).with_name("four_arm_query_heads.py"),
             Path(__file__).with_name("four_arm_token_cache.py"),
             Path(__file__).with_name("multilabel_heads.py"),
             Path(__file__).with_name("frozen_lora_layer_diagnostics.py")]
    return {str(path.resolve()): sha(path) for path in paths if path.exists()}


def verify_selection_lock(study):
    """Require every registered task/seed and every bound fit artifact."""
    study = Path(study).resolve()
    protocol = read_protocol(study)
    lock = read_json(study / "selection_lock.json")
    if lock.get("complete") is not True or lock.get("protocol_sha256") != sha(study / "protocol.json"):
        raise ValueError("All-task selection barrier is incomplete or belongs to another protocol")
    expected = {f"{task}/s{seed}" for task in TASKS for seed in protocol.get("seeds", [42, 43, 44])}
    if set(lock.get("selections", {})) != expected:
        raise ValueError("Every task and seed must be locked before test access")
    artifacts = lock.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Selection artifacts must be a path-to-hash mapping")
    for key in sorted(expected):
        folder = study / "heads" / key
        selection_path = folder / "selection.json"
        if sha(selection_path) != lock["selections"][key]:
            raise ValueError(f"Selection changed after root lock: {key}")
        selection = _verify_local_selection(study, key.split('/')[0], int(key.split('/s')[1]))
        for path in (selection_path, Path(selection["checkpoint_path"]),
                     Path(selection["validation_predictions_path"])):
            if artifacts.get(str(path.resolve())) != sha(path):
                raise ValueError(f"Root lock does not bind fit artifact: {path}")
    for name, expected_sha in artifacts.items():
        path = Path(name)
        if not path.is_absolute() or sha(path) != expected_sha:
            raise ValueError(f"Root-locked artifact changed: {name}")
    return lock


def label_arrays(frame):
    """Keep every binary row; native harmful masks alone supervise type CE."""
    required = {"base_id", "source", "prompt_harm_label", "category_id", "type_label_mask"}
    if not required.issubset(frame) or frame.base_id.astype(str).duplicated().any():
        raise ValueError("Original unique row identities and label fields are required")
    if not frame.prompt_harm_label.isin(["harmful", "unharmful"]).all():
        raise ValueError("Unknown binary label")
    binary = frame.prompt_harm_label.eq("harmful").to_numpy(dtype=np.int64)
    numeric = pd.to_numeric(frame.category_id, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if not np.equal(numeric[finite], np.floor(numeric[finite])).all():
        raise ValueError("Nonintegral category label")
    categories = np.where(finite, numeric, -1).astype(np.int64)
    masks = frame.type_label_mask.fillna(False)
    if not masks.isin([True, False]).all():
        raise ValueError("Type masks must be boolean")
    native = frame.source.isin(NATIVE_SOURCES).to_numpy()
    typed = native & masks.to_numpy(dtype=bool) & np.isin(categories, np.arange(1, 6)) & (binary == 1)
    # A safe binary label is category 0 by task definition, irrespective of
    # historic category fields. Unknown harmful categories remain excluded.
    final_labels = np.where(binary == 0, 0, categories)
    known = native & ((binary == 0) | typed)
    return dict(binary=binary, categories=categories, typed=typed,
                known_native=known, final_labels=final_labels)


def _load_split(study, split):
    """Load last-token and all-token features in original parent row order."""
    study = Path(study).resolve()
    protocol = read_protocol(study)
    _check_stop(study)
    if split == "test":
        verify_selection_lock(study)
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    data_path = Path(protocol["data_dir"]) / f"{split}.parquet"
    if sha(data_path) != protocol["data_sha256"][split]:
        raise ValueError(f"Original dataset changed: {split}")
    columns = ["base_id", "group_id", "source", "prompt_harm_label", "category_id", "type_label_mask"]
    frame = pd.read_parquet(data_path, columns=columns)
    labels = label_arrays(frame)
    cache_dir = Path(protocol["last_token_cache_dir"])
    manifest_path = cache_dir / f"{split}.manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("complete") is not True or manifest.get("adapter_sha256") is not None
            or manifest.get("representation", "base") != "base"
            or manifest.get("input_sha256") != protocol["data_sha256"][split]):
        raise ValueError("Expected complete, original-data, no-LoRA last-token cache")
    for name, digest in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
        if sha(manifest[name]) != manifest[digest]:
            raise ValueError(f"Frozen parent cache changed: {split}/{name}")
    last = np.load(manifest["path"], mmap_mode="r", allow_pickle=False)
    meta = pd.read_parquet(manifest["metadata_path"])
    if (last.ndim != 3 or last.shape[:2] != (len(frame), 32) or last.dtype != np.float16
            or list(last.shape) != manifest["shape"]):
        raise ValueError("Expected FP16 last-token cache [rows,32,hidden]")
    if (meta.base_id.astype(str).tolist() != frame.base_id.astype(str).tolist()
            or meta.source.tolist() != frame.source.tolist()
            or not np.array_equal(meta.label.to_numpy(), labels["binary"])):
        raise ValueError("Frozen parent row identity, source or binary label changed")
    from scripts.hanguard import four_arm_token_cache
    tokens = four_arm_token_cache.load(study, split)
    token_meta = tokens.meta
    if (len(token_meta) != len(meta)
            or token_meta.base_id.astype(str).tolist() != meta.base_id.astype(str).tolist()
            or token_meta.source.tolist() != meta.source.tolist()):
        raise ValueError("All-token and last-token cache order differs")
    lengths = np.asarray(tokens.lengths, dtype=np.int64)
    if lengths.shape != (len(meta),) or (lengths < 1).any() or not np.array_equal(lengths, meta.prompt_tokens):
        raise ValueError("All-token cache lengths differ from original inputs")
    return dict(last=last, tokens=tokens, meta=meta, frame=frame, lengths=lengths,
                labels=labels, hidden_size=int(last.shape[-1]),
                identity=dict(dataset_sha256=protocol["data_sha256"][split],
                              last_manifest_sha256=sha(manifest_path),
                              token_manifest_sha256=sha(Path(protocol["token_cache_dir"]) / split / "manifest.json"),
                              rows=len(meta)))


def macro_batches(indices, seed, epoch, effective_batch):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.random.default_rng(int(seed) * 1009 + int(epoch)).permutation(indices)
    return [order[start:start + effective_batch] for start in range(0, len(order), effective_batch)]


def micro_batches(indices, lengths, token_budget, max_micro, pad_multiple=1):
    """Preserve macro order, cap padded token count, and never truncate a row."""
    current, longest = [], 0
    for row in np.asarray(indices, dtype=np.int64):
        length = int(lengths[row])
        padded = (length + pad_multiple - 1) // pad_multiple * pad_multiple
        if padded > token_budget:
            raise ValueError("One full input exceeds the registered token budget")
        proposed = max(longest, padded)
        if current and (len(current) >= max_micro or (len(current) + 1) * proposed > token_budget):
            yield np.asarray(current, dtype=np.int64)
            current, longest = [], 0
        current.append(int(row))
        longest = max(longest, padded)
    if current:
        yield np.asarray(current, dtype=np.int64)


def feature_batch(data, rows, device, pad_multiple=1):
    raw = np.stack([np.asarray(data["last"][rows, layer - 1], dtype=np.float32) for layer in LAYERS], axis=1)
    maximum = int(data["lengths"][rows].max())
    maximum = (maximum + pad_multiple - 1) // pad_multiple * pad_multiple
    token_array = np.zeros((len(rows), maximum, data["hidden_size"]), dtype=np.float16)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for offset, row in enumerate(rows):
        values = np.asarray(data["tokens"].get_row(int(row)))
        length = int(data["lengths"][row])
        if values.shape != (length, data["hidden_size"]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid cached full-token features at row {row}")
        token_array[offset, :length] = values
        mask[offset, :length] = True
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite last-token features")
    return (torch.from_numpy(raw).to(device), torch.from_numpy(token_array).to(device),
            torch.from_numpy(mask).to(device))


def _step_seed(seed, epoch, step, micro):
    return (int(seed) * 1000003 + int(epoch) * 1009 + int(step) * 97 + int(micro)) % (2**63 - 1)


def _set_seed(seed, device):
    # CPU seeding also seeds the current device in CUDA-enabled PyTorch builds;
    # torch.manual_seed itself does not allocate features or load a backbone.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed)


def _rng_state(device):
    state = np.random.get_state()
    return dict(device_type=torch.device(device).type, python=random.getstate(),
                numpy=[state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == "cuda" else None)


def _restore_rng(state, device):
    if state["device_type"] != torch.device(device).type:
        raise ValueError("Resume RNG device type differs")
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _task_indices(data, task):
    return (np.arange(len(data["meta"]), dtype=np.int64) if task == "binary"
            else np.flatnonzero(data["labels"]["typed"]))


def _targets(data, task):
    return data["labels"]["binary"] if task == "binary" else data["labels"]["categories"] - 1


def build_heads(hidden_size, task, seed, cfg, device):
    return {arm: FourArmHead(hidden_size, width=cfg["width"], num_classes=2 if task == "binary" else 5,
                            arm=arm, dropout=cfg["dropout"], seed=seed).to(device) for arm in ARMS}


def train_macro(heads, optimizers, data, rows, task, cfg, device, *, seed, epoch, step):
    """Each micro loss is sample-weighted to equal one whole effective batch."""
    if not len(rows):
        raise ValueError("Empty macro batch")
    losses = {arm: 0. for arm in ARMS}
    for arm in ARMS:
        heads[arm].train()
        optimizers[arm].zero_grad(set_to_none=True)
    for micro, indices in enumerate(micro_batches(rows, data["lengths"], cfg["token_budget"],
                                                  cfg["max_micro"], cfg["pad_multiple"])):
        last, tokens, mask = feature_batch(data, indices, device, cfg["pad_multiple"])
        target = torch.as_tensor(_targets(data, task)[indices], dtype=torch.long, device=device)
        for arm in ARMS:
            _set_seed(_step_seed(seed, epoch, step, micro), device)
            logits = heads[arm](last, tokens if heads[arm].has_category else None,
                               mask if heads[arm].has_category else None)
            loss = F.cross_entropy(logits, target, reduction="sum") / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {task}/{arm} loss")
            loss.backward()
            losses[arm] += float(loss.detach())
    norms = {}
    for arm in ARMS:
        norm = torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        norms[arm] = float(norm)
        optimizers[arm].step()
    return losses, norms


@torch.no_grad()
def predict(heads, data, rows, cfg, device, study=None):
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        raise ValueError("Nonempty evaluation rows required")
    output = {arm: np.empty((len(rows), heads[arm].num_classes), dtype=np.float32) for arm in ARMS}
    position = {int(row): index for index, row in enumerate(rows)}
    order = rows[np.argsort(data["lengths"][rows], kind="stable")]
    for head in heads.values():
        head.eval()
    for indices in micro_batches(order, data["lengths"], cfg["token_budget"], cfg["max_micro"], cfg["pad_multiple"]):
        if study is not None:
            _check_stop(study)
        last, tokens, mask = feature_batch(data, indices, device, cfg["pad_multiple"])
        locations = [position[int(row)] for row in indices]
        for arm in ARMS:
            values = heads[arm](last, tokens if heads[arm].has_category else None,
                               mask if heads[arm].has_category else None).float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite predictions: {arm}")
            output[arm][locations] = values
    return output


def extension_rule(history, best_epochs, budget, maximum):
    if budget >= maximum or len(history) < 3:
        return None
    first, last = history[-3], history[-1]
    triggers = {}
    for arm in ARMS:
        improvement = (first["validation_ce"][arm] - last["validation_ce"][arm]) / max(abs(first["validation_ce"][arm]), 1e-12)
        if best_epochs[arm] == last["epoch"] and improvement >= .002:
            triggers[arm] = dict(best_epoch=int(best_epochs[arm]), relative_recent_ce_improvement=float(improvement))
    if not triggers:
        return None
    new_budget = min(next((n for n in (24, 40) if n > budget), maximum), maximum)
    return dict(previous_budget=int(budget), new_budget=int(new_budget), triggers=triggers,
                all_four_arms_extended=True, selection_scope="validation_cross_entropy_only",
                continuation="same_optimizer_and_epoch_state")


def _safe_meta(data, rows):
    columns = [name for name in ("base_id", "group_id", "source", "prompt_tokens") if name in data["meta"]]
    result = data["meta"].iloc[rows][columns].copy().reset_index(drop=True)
    result["binary_label"] = data["labels"]["binary"][rows]
    result["category_id"] = data["labels"]["categories"][rows]
    result["known_native_category"] = data["labels"]["known_native"][rows]
    return result


def _probabilities(logits):
    return torch.from_numpy(np.asarray(logits)).double().softmax(-1).numpy()


def _prediction_frame(data, rows, logits):
    frame = _safe_meta(data, rows)
    for arm in ARMS:
        probs = _probabilities(logits[arm])
        for category in range(probs.shape[1]):
            frame[f"{arm}_p{category}"] = probs[:, category]
        frame[f"{arm}_argmax"] = probs.argmax(1)
    return frame


def _save_frame(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _verify_local_selection(study, task, seed):
    folder = task_folder(study, task, seed)
    result = read_json(folder / "selection.json")
    if (result.get("task") != task or result.get("seed") != seed
            or result.get("protocol_sha256") != sha(Path(study) / "protocol.json")
            or set(result.get("arms", {})) != set(ARMS)):
        raise ValueError("Local selection identity differs")
    for path_key, sha_key, expected_path in (
            ("checkpoint_path", "checkpoint_sha256", folder / "selected_heads.pt"),
            ("validation_predictions_path", "validation_predictions_sha256", folder / "validation_predictions.parquet")):
        if Path(result[path_key]).resolve() != expected_path.resolve() or sha(expected_path) != result[sha_key]:
            raise ValueError(f"Selected artifact changed: {path_key}")
    return result


def fit_task(study, task, seed, device="cpu"):
    study = Path(study).resolve()
    protocol = read_protocol(study)
    if task not in TASKS or seed not in protocol.get("seeds", [42, 43, 44]):
        raise ValueError("Task/seed is not registered")
    folder = task_folder(study, task, seed)
    folder.mkdir(parents=True, exist_ok=True)
    _check_stop(study)
    if (folder / "selection.json").exists():
        return _verify_local_selection(study, task, seed)
    if (study / "selection_lock.json").exists():
        raise ValueError("Cannot change fits after the root test barrier has been locked")
    cfg = settings(protocol)
    train, validation = _load_split(study, "train"), _load_split(study, "validation")
    for name in ("base_id", "group_id"):
        if set(train["meta"][name].astype(str)) & set(validation["meta"][name].astype(str)):
            raise ValueError(f"Train/validation leakage: {name}")
    tr_rows, val_rows = _task_indices(train, task), _task_indices(validation, task)
    if not len(tr_rows) or not len(val_rows):
        raise ValueError("Each task needs train and validation supervision")
    identity = dict(protocol_sha256=sha(study / "protocol.json"), task=task, seed=seed,
                    settings=cfg, sources=_sources(), training_identity=train["identity"],
                    validation_identity=validation["identity"], device_type=torch.device(device).type)
    heads = build_heads(train["hidden_size"], task, seed, cfg, device)
    optimizers = {arm: torch.optim.AdamW(heads[arm].parameters(), lr=cfg["lr"],
                                       weight_decay=cfg["weight_decay"]) for arm in ARMS}
    completed, budget, history, extensions = 0, cfg["epochs"], [], []
    best_ce, best_epochs = {a: float("inf") for a in ARMS}, {a: 0 for a in ARMS}
    best_states = {a: _cpu(heads[a].state_dict()) for a in ARMS}
    resume_path = folder / "epoch_resume.pt"
    started = time.monotonic()
    prior_seconds = 0.
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved.get("schema") != RESUME_SCHEMA or saved.get("identity") != identity:
            raise ValueError("Resume checkpoint protocol, data, source or device identity differs")
        completed, budget = saved["completed_epochs"], saved["budget"]
        history, extensions = saved["history"], saved["extensions"]
        if (not 1 <= completed <= budget <= cfg["max_epochs"] or len(history) != completed
                or [r["epoch"] for r in history] != list(range(1, completed + 1))):
            raise ValueError("Invalid atomic epoch history")
        best_ce, best_epochs, best_states = saved["best_ce"], saved["best_epochs"], saved["best_states"]
        for arm in ARMS:
            if not 1 <= best_epochs[arm] <= completed or not np.isfinite(best_ce[arm]):
                raise ValueError("Invalid validation-best state")
            heads[arm].load_state_dict(saved["current_states"][arm], strict=True)
            optimizers[arm].load_state_dict(saved["optimizer_states"][arm])
        prior_seconds = float(saved.get("wall_seconds", 0.))
        _restore_rng(saved["rng"], device)
    dump(folder / "fit_started.json", dict(
        task=task, seed=seed, protocol_sha256=identity["protocol_sha256"],
        settings=cfg, device=str(device), training_rows=len(tr_rows), validation_rows=len(val_rows),
        resumed_after_epoch=completed, started_unix=time.time(),
        data_identity=dict(train=train["identity"], validation=validation["identity"])))

    def commit():
        atomic_save(resume_path, _cpu(dict(
            schema=RESUME_SCHEMA, identity=identity, completed_epochs=completed,
            budget=budget, history=history, extensions=extensions,
            current_states={a: heads[a].state_dict() for a in ARMS},
            optimizer_states={a: optimizers[a].state_dict() for a in ARMS},
            best_ce=best_ce, best_epochs=best_epochs, best_states=best_states,
            rng=_rng_state(device), wall_seconds=prior_seconds + time.monotonic() - started)))

    while True:
        while completed < budget:
            _check_stop(study)
            totals = {a: 0. for a in ARMS}
            batches = macro_batches(tr_rows, seed, completed, cfg["effective_batch"])
            for step, rows in enumerate(batches):
                _check_stop(study)
                losses, _ = train_macro(heads, optimizers, train, rows, task, cfg, device,
                                        seed=seed, epoch=completed, step=step)
                for arm in ARMS:
                    totals[arm] += losses[arm] * len(rows)
                if step == 0 or (step + 1) % 25 == 0:
                    dump(folder / "progress.json", dict(
                        task=task, seed=seed, state="training", device=str(device),
                        completed_epochs=completed, current_epoch=completed + 1,
                        budget=budget, macro_step=step + 1, macro_steps=len(batches),
                        wall_seconds=prior_seconds + time.monotonic() - started))
            val_logits = predict(heads, validation, val_rows, cfg, device, study)
            target = torch.tensor(_targets(validation, task)[val_rows], dtype=torch.long)
            validation_ce = {a: float(F.cross_entropy(torch.from_numpy(val_logits[a]).double(), target)) for a in ARMS}
            completed += 1
            for arm in ARMS:
                if validation_ce[arm] < best_ce[arm]:
                    best_ce[arm], best_epochs[arm] = validation_ce[arm], completed
                    best_states[arm] = _cpu(heads[arm].state_dict())
            history.append(dict(epoch=completed, train_ce={a: totals[a] / len(tr_rows) for a in ARMS},
                                validation_ce=validation_ce))
            commit()
            dump(folder / "history.json", history)
            dump(folder / "progress.json", dict(task=task, seed=seed, completed_epochs=completed,
                                                budget=budget, state="training", device=str(device),
                                                best_epochs=best_epochs, validation_ce=validation_ce,
                                                wall_seconds=prior_seconds + time.monotonic() - started))
            print(json.dumps(dict(task=task, seed=seed, epoch=completed, budget=budget,
                                  validation_ce=validation_ce)), flush=True)
        extension = extension_rule(history, best_epochs, budget, cfg["max_epochs"])
        if extension is None:
            break
        extensions.append(extension)
        budget = extension["new_budget"]
        commit()
        print(json.dumps(dict(task=task, seed=seed, extension=extension)), flush=True)
    for arm in ARMS:
        heads[arm].load_state_dict(best_states[arm])
    val_logits = predict(heads, validation, val_rows, cfg, device, study)
    validation_predictions_path = folder / "validation_predictions.parquet"
    _save_frame(validation_predictions_path, _prediction_frame(validation, val_rows, val_logits))
    checkpoint_path = folder / "selected_heads.pt"
    atomic_save(checkpoint_path, dict(schema=SCHEMA, task=task, seed=seed,
                                     protocol_sha256=identity["protocol_sha256"],
                                     states=best_states, metadata={a: heads[a].metadata() for a in ARMS}))
    arms = {}
    for arm in ARMS:
        entry = dict(best_epoch=best_epochs[arm], validation_ce=best_ce[arm],
                     parameters=heads[arm].parameter_count(), architecture=heads[arm].metadata())
        if task == "binary":
            probability = _probabilities(val_logits[arm])[:, 1]
            entry["threshold"] = accuracy_threshold(validation["labels"]["binary"][val_rows], probability)
            entry["validation_metrics"] = binary_metrics(validation["labels"]["binary"][val_rows], probability, entry["threshold"])
        else:
            entry["validation_metrics"] = multiclass_metrics(
                validation["labels"]["categories"][val_rows], val_logits[arm].argmax(1) + 1, range(1, 6))
        arms[arm] = entry
    selection = dict(schema=SCHEMA, task=task, seed=seed, protocol_sha256=identity["protocol_sha256"],
                     checkpoint_path=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                     validation_predictions_path=str(validation_predictions_path),
                     validation_predictions_sha256=sha(validation_predictions_path), arms=arms,
                     training_rows=len(tr_rows), validation_rows=len(val_rows), final_budget=budget,
                     extensions=extensions, settings=cfg, source_sha256=identity["sources"],
                     data_sha256=protocol["data_sha256"], device=str(device),
                     wall_seconds=prior_seconds + time.monotonic() - started,
                     formal_test_opened=False, selection_scope="validation_only",
                     objective="two_class_softmax_ce" if task == "binary" else "native_known_harmful_five_class_softmax_ce")
    dump(folder / "selection.json", selection)
    dump(folder / "progress.json", dict(task=task, seed=seed, state="selected",
                                        completed_epochs=completed, budget=budget, device=str(device)))
    return selection


def multiclass_metrics(labels, predicted, classes):
    labels, predicted = np.asarray(labels, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    classes = list(classes)
    if labels.ndim != 1 or predicted.shape != labels.shape or not len(labels):
        raise ValueError("Nonempty aligned class decisions required")
    if not np.isin(labels, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError("Class outside the declared task")
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    positions = {value: index for index, value in enumerate(classes)}
    for target, output in zip(labels, predicted):
        confusion[positions[int(target)], positions[int(output)]] += 1
    per_class = []
    for index, category in enumerate(classes):
        tp, support, count = int(confusion[index, index]), int(confusion[index].sum()), int(confusion[:, index].sum())
        per_class.append(dict(category_id=category, support=support,
                              precision=tp / max(1, count), recall=tp / max(1, support),
                              f1=2 * tp / max(1, support + count)))
    return dict(rows=len(labels), accuracy=float((labels == predicted).mean()),
                macro_f1=float(np.mean([item["f1"] for item in per_class])),
                classes=classes, per_class=per_class, confusion_matrix=confusion.tolist())


def paired_changes(labels, baseline, candidate):
    labels, baseline, candidate = np.asarray(labels), np.asarray(baseline), np.asarray(candidate)
    if labels.shape != baseline.shape or candidate.shape != labels.shape:
        raise ValueError("Paired labels and decisions must align")
    before, after = baseline == labels, candidate == labels
    rescued, harmed = int((~before & after).sum()), int((before & ~after).sum())
    return dict(rows=len(labels), rescued=rescued, harmed=harmed, net_correct=rescued - harmed,
                delta_accuracy=(rescued - harmed) / max(1, len(labels)))


def multiclass_bootstrap(labels, candidate, baseline, groups, draws=2000, seed=42):
    labels, candidate, baseline = map(np.asarray, (labels, candidate, baseline))
    groups = np.asarray(groups).astype(str)
    if not len(labels) or labels.shape != candidate.shape or labels.shape != baseline.shape or groups.shape != labels.shape:
        raise ValueError("Nonempty aligned bootstrap inputs required")
    _, inverse = np.unique(groups, return_inverse=True)
    size = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=size)
    delta = np.bincount(inverse, weights=(candidate == labels).astype(float) - (baseline == labels), minlength=size)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for begin in range(0, draws, 16):
        choice = rng.integers(0, size, size=(min(16, draws - begin), size))
        values[begin:begin + len(choice)] = delta[choice].sum(1) / counts[choice].sum(1)
    return dict(unit="group_id cluster", groups=size, repetitions=draws, seed=seed,
                accuracy=dict(delta=float(delta.sum() / counts.sum()),
                              ci95=np.quantile(values, [.025, .975]).tolist()))


def _subset_mask(meta, protocol):
    path = protocol.get("excluded_ids_path", protocol.get("excluded_test_ids_path", protocol.get("selected122_excluded_ids_path")))
    if path is None:
        raise ValueError("The fixed selected122 exclusion IDs must be registered")
    path = Path(path)
    expected = protocol.get("excluded_ids_sha256", protocol.get("excluded_test_ids_sha256", protocol.get("selected122_excluded_ids_sha256")))
    if expected is None or sha(path) != expected:
        raise ValueError("Fixed selected122 identity list changed or is unregistered")
    excluded = set(path.read_text().splitlines()) - {""}
    if not excluded.issubset(set(meta.base_id.astype(str))):
        raise ValueError("Selected122 exclusions are not all members of original test")
    return ~meta.base_id.astype(str).isin(excluded).to_numpy()


def verify_evaluation(study, task, seed):
    folder = task_folder(study, task, seed)
    receipt = read_json(folder / "test_complete.json")
    if (receipt.get("complete") is not True or receipt.get("task") != task or receipt.get("seed") != seed
            or receipt.get("protocol_sha256") != sha(Path(study) / "protocol.json")
            or receipt.get("rootselection_lock_sha256") != sha(Path(study) / "selection_lock.json")):
        raise ValueError("Test completion receipt identity differs")
    expected = {str(folder / "test_results.json"), str(folder / "test_predictions.parquet")}
    if set(receipt.get("artifacts", {})) != expected:
        raise ValueError("Test receipt must bind both results and row predictions")
    for name, digest in receipt["artifacts"].items():
        if sha(name) != digest:
            raise ValueError(f"Completed test artifact changed: {name}")
    result = read_json(folder / "test_results.json")
    if result.get("predictions_sha256") != sha(folder / "test_predictions.parquet"):
        raise ValueError("Test result and prediction digest disagree")
    return result


def evaluate_task(study, task, seed, device="cpu"):
    study = Path(study).resolve()
    protocol, lock = read_protocol(study), verify_selection_lock(study)
    selection = _verify_local_selection(study, task, seed)
    folder = task_folder(study, task, seed)
    results_path = folder / "test_results.json"
    if (folder / "test_complete.json").exists():
        return verify_evaluation(study, task, seed)
    # This is the first point at which this module can request the test cache.
    data = _load_split(study, "test")
    cfg = settings(protocol)
    checkpoint = torch.load(selection["checkpoint_path"], map_location="cpu", weights_only=True)
    heads = build_heads(data["hidden_size"], task, seed, cfg, device)
    if checkpoint.get("protocol_sha256") != selection["protocol_sha256"]:
        raise ValueError("Selected checkpoint protocol differs")
    for arm in ARMS:
        heads[arm].load_state_dict(checkpoint["states"][arm], strict=True)
    rows = np.arange(len(data["meta"]), dtype=np.int64)
    logits = predict(heads, data, rows, cfg, device, study)
    frame = _prediction_frame(data, rows, logits)
    path = folder / "test_predictions.parquet"
    _save_frame(path, frame)
    result = dict(schema=SCHEMA, task=task, seed=seed, protocol_sha256=sha(study / "protocol.json"),
                  rootselection_lock_sha256=sha(study / "selection_lock.json"),
                  predictions_path=str(path), predictions_sha256=sha(path), arms={},
                  interpretation="Full original test is primary; selected122 is an explicitly outcome-selected fixed subset.")
    if task == "binary":
        labels = data["labels"]["binary"]
        masks = dict(full_original_test=np.ones(len(rows), dtype=bool), fixed_selected122_subset=_subset_mask(data["meta"], protocol))
        for arm in ARMS:
            p = _probabilities(logits[arm])[:, 1]
            threshold = selection["arms"][arm]["threshold"]
            result["arms"][arm] = {scope: dict(
                validation_threshold=binary_metrics(labels[mask], p[mask], threshold),
                fixed_threshold_05=binary_metrics(labels[mask], p[mask], .5),
                by_source={str(source): binary_metrics(labels[mask & data["meta"].source.eq(source).to_numpy()],
                                                       p[mask & data["meta"].source.eq(source).to_numpy()], threshold)
                           for source in sorted(data["meta"].source.unique())
                           if (mask & data["meta"].source.eq(source).to_numpy()).any()}) for scope, mask in masks.items()}
        contrasts = (("fusion_only", "baseline"), ("category_only", "baseline"),
                     ("fusion_category", "baseline"), ("fusion_category", "category_only"),
                     ("fusion_category", "fusion_only"))
        result["paired_contrasts"] = {}
        for scope, mask in masks.items():
            entries = {}
            for candidate, baseline in contrasts:
                pc, pb = _probabilities(logits[candidate])[:, 1], _probabilities(logits[baseline])[:, 1]
                tc, tb = selection["arms"][candidate]["threshold"], selection["arms"][baseline]["threshold"]
                entries[f"{candidate}_minus_{baseline}"] = dict(
                    changes=paired_changes(labels[mask], pb[mask] >= tb, pc[mask] >= tc),
                    bootstrap=group_cluster_bootstrap(labels[mask], pc[mask], pb[mask],
                        data["meta"].group_id.to_numpy()[mask], tc, tb,
                        n_resamples=cfg["bootstrap_draws"], seed=seed))
            result["paired_contrasts"][scope] = entries
    else:
        mask = data["labels"]["typed"]
        y = data["labels"]["categories"][mask]
        for arm in ARMS:
            predicted = logits[arm].argmax(1) + 1
            result["arms"][arm] = dict(native_known_harmful=multiclass_metrics(y, predicted[mask], range(1, 6)),
                                      by_source={source: multiclass_metrics(
                                          data["labels"]["categories"][mask & data["meta"].source.eq(source).to_numpy()],
                                          predicted[mask & data["meta"].source.eq(source).to_numpy()], range(1, 6))
                                          for source in NATIVE_SOURCES
                                          if (mask & data["meta"].source.eq(source).to_numpy()).any()})
        result["paired_contrasts"] = {}
        for candidate, baseline in (("fusion_only", "baseline"), ("category_only", "baseline"),
                                    ("fusion_category", "baseline"), ("fusion_category", "category_only")):
            pc, pb = logits[candidate].argmax(1) + 1, logits[baseline].argmax(1) + 1
            result["paired_contrasts"][f"{candidate}_minus_{baseline}"] = dict(
                changes=paired_changes(y, pb[mask], pc[mask]),
                bootstrap=multiclass_bootstrap(y, pc[mask], pb[mask], data["meta"].group_id.to_numpy()[mask],
                                              cfg["bootstrap_draws"], seed))
    dump(results_path, result)
    dump(folder / "test_complete.json", dict(
        schema=SCHEMA, complete=True, task=task, seed=seed,
        protocol_sha256=sha(study / "protocol.json"),
        rootselection_lock_sha256=sha(study / "selection_lock.json"),
        artifacts={str(path): sha(path), str(results_path): sha(results_path)}))
    return result


def pipeline_metrics(binary, category, selections, *, draws=2000, seed=42):
    if binary.base_id.astype(str).tolist() != category.base_id.astype(str).tolist():
        raise ValueError("Binary and type system predictions must have identical row order")
    for name in ("source", "binary_label", "category_id", "known_native_category"):
        if not np.array_equal(binary[name].to_numpy(), category[name].to_numpy()):
            raise ValueError(f"Binary/type truth or provenance mismatch: {name}")
    mask = binary.known_native_category.to_numpy(dtype=bool)
    labels = np.where(binary.binary_label.to_numpy() == 0, 0, binary.category_id.to_numpy())[mask]
    specs = dict(
        classic_gate_type_factorial=dict(baseline=("baseline", "baseline"),
                                        fusion_only=("fusion_only", "baseline"),
                                        category_only=("baseline", "category_only"),
                                        fusion_category=("fusion_only", "category_only")),
        fusion_in_both_stages_factorial=dict(baseline=("baseline", "baseline"),
                                             fusion_only=("fusion_only", "fusion_only"),
                                             category_only=("baseline", "category_only"),
                                             fusion_category=("fusion_only", "fusion_category")))
    output = {}
    for table, arms in specs.items():
        entries, predictions = {}, {}
        for arm, (gate, type_head) in arms.items():
            threshold = selections["binary"]["arms"][gate]["threshold"]
            type_prediction = category[[f"{type_head}_p{k}" for k in range(5)]].to_numpy().argmax(1) + 1
            predicted = np.where(binary[f"{gate}_p1"].to_numpy() >= threshold, type_prediction, 0)[mask]
            predictions[arm] = predicted
            entries[arm] = dict(binary_arm=gate, category_arm=type_head, threshold=threshold,
                                metrics=multiclass_metrics(labels, predicted, range(6)))
        changes = {}
        for arm in ARMS[1:]:
            changes[f"{arm}_minus_baseline"] = dict(
                changes=paired_changes(labels, predictions["baseline"], predictions[arm]),
                bootstrap=multiclass_bootstrap(labels, predictions[arm], predictions["baseline"],
                                              binary.group_id.to_numpy()[mask], draws, seed))
        output[table] = dict(scope="native_two_source_known_safe_or_known_type", rows=int(mask.sum()),
                             arms=entries, paired_contrasts=changes)
    return output


def _aggregate(values):
    values = np.asarray(values, dtype=float)
    return dict(seeds=len(values), mean=float(values.mean()),
                sample_std=float(values.std(ddof=1)) if len(values) > 1 else None,
                values=values.tolist())


def summarize(study):
    study = Path(study).resolve()
    protocol = read_protocol(study)
    verify_selection_lock(study)
    seeds, cfg = protocol.get("seeds", [42, 43, 44]), settings(protocol)
    per_seed, rows = {}, []
    for seed in seeds:
        binary = verify_evaluation(study, "binary", seed)
        category = verify_evaluation(study, "category", seed)
        for result in (binary, category):
            if (result.get("rootselection_lock_sha256") != sha(study / "selection_lock.json")
                    or sha(result["predictions_path"]) != result["predictions_sha256"]):
                raise ValueError("Evaluation predictions or root lock changed")
        selections = {t: _verify_local_selection(study, t, seed) for t in TASKS}
        systems = pipeline_metrics(pd.read_parquet(binary["predictions_path"]),
                                   pd.read_parquet(category["predictions_path"]), selections,
                                   draws=cfg["bootstrap_draws"], seed=seed)
        per_seed[str(seed)] = dict(binary=binary, category=category, systems=systems)
        for arm in ARMS:
            for scope in ("full_original_test", "fixed_selected122_subset"):
                metrics = binary["arms"][arm][scope]["validation_threshold"]
                rows.append(dict(seed=seed, task="binary", scope=scope, arm=arm,
                                 accuracy=metrics["accuracy"], f1=metrics["f1"], auroc=metrics["auroc"],
                                 fpr=metrics["fpr"], macro_f1=None))
            metrics = category["arms"][arm]["native_known_harmful"]
            rows.append(dict(seed=seed, task="category", scope="native_known_harmful", arm=arm,
                             accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
            for name, system in systems.items():
                metrics = system["arms"][arm]["metrics"]
                rows.append(dict(seed=seed, task=name, scope="native_known_final_six_class", arm=arm,
                                 accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
    frame = pd.DataFrame(rows)
    aggregates = []
    for (task, scope, arm), subset in frame.groupby(["task", "scope", "arm"], sort=False):
        aggregates.append(dict(task=task, scope=scope, arm=arm,
                               metrics={metric: _aggregate(subset[metric].dropna())
                                        for metric in ("accuracy", "f1", "macro_f1", "auroc", "fpr")
                                        if metric in subset and subset[metric].notna().any()}))
    deltas = []
    for (task, scope), subset in frame.groupby(["task", "scope"], sort=False):
        baseline = subset[subset.arm.eq("baseline")].set_index("seed")
        for arm in ARMS[1:]:
            candidate = subset[subset.arm.eq(arm)].set_index("seed")
            deltas.append(dict(task=task, scope=scope, candidate=arm, baseline="baseline",
                               metrics={metric: _aggregate(candidate[metric].dropna() - baseline[metric].dropna())
                                        for metric in ("accuracy", "f1", "macro_f1", "auroc", "fpr")
                                        if metric in candidate and candidate[metric].notna().all()
                                        and baseline[metric].notna().all()}))
    output = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(study / "protocol.json"),
                  rootselection_lock_sha256=sha(study / "selection_lock.json"),
                  seeds=seeds, arms=list(ARMS), per_seed=per_seed, aggregate=aggregates,
                  paired_delta_aggregate=deltas,
                  outcome_selected_subset=True,
                  interpretation="Original full test and fixed outcome-selected subset are reported separately; objective accuracies are never added.")
    analysis = study / "analysis"
    analysis.mkdir(exist_ok=True)
    frame.to_csv(analysis / "metrics_by_seed.csv", index=False)
    dump(analysis / "results.json", output)
    dump(study / "results.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate", "summarize"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.study)
    else:
        if args.task is None or args.seed is None:
            parser.error("--task and --seed are required for fitting/evaluation")
        (fit_task if args.command == "fit" else evaluate_task)(args.study, args.task, args.seed, args.device)


if __name__ == "__main__":
    main()
