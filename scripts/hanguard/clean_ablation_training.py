"""Fresh resumable nine-configuration frozen-base experiments.

The binary and native five-category objectives are separate experiments.  The
nine configurations in each experiment share batches, optimizer settings and validation
selection; no test feature is read until all fresh probes and six task/seed fits are hash locked.
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

from scripts.hanguard.clean_ablation_heads import ARMS, MAIN_ARMS, LAYERS, CleanAblationHead
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    accuracy_threshold, binary_metrics, group_cluster_bootstrap,
)

SCHEMA = "hanguard_clean_ablation_v1"
RESUME_SCHEMA = "hanguard_clean_ablation_epoch_resume_v1"
NATIVE_SOURCES = ("chinese_curated", "jailbench")
TASKS = ("binary", "category")
CONTRASTS = (("fusion_only", "baseline"), ("category_only", "baseline"),
             ("fusion_category", "baseline"), ("fusion_category", "category_only"),
             ("fusion_category", "fusion_only"), ("fusion_only", "fusion_repeat_last"),
             ("fusion_category", "category_fusion_repeat_last"),
             ("category_only", "category_shared"),
             ("fusion_category", "fusion_category_shared"),
             ("fusion_only", "probe_selected_single"),
             ("fusion_category", "probe_selected_single"))


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
        raise ValueError("Invalid clean-ablation training settings")
    return value


def read_protocol(study):
    study = Path(study).resolve()
    value = read_json(study / "protocol.json")
    registration_path = study / "protocol_registration.json"
    if registration_path.exists() and read_json(registration_path).get("protocol_sha256") != sha(study / "protocol.json"):
        raise ValueError("Registered clean-ablation protocol changed")
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
    if any(key in value for key in ("last_token_cache_dir", "excluded_ids_path", "excluded_test_ids_path", "selected122_excluded_ids_path")):
        raise ValueError("Clean study cannot register old caches or outcome-selected exclusions")
    settings(value)
    return value


def task_folder(study, task, seed):
    if task not in TASKS:
        raise ValueError("Unknown task")
    return Path(study).resolve() / "heads" / task / f"s{int(seed)}"


def _sources():
    paths = [Path(__file__), Path(__file__).with_name("clean_ablation_heads.py"),
             Path(__file__).with_name("clean_ablation_features.py"),
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
    probe_selection = verify_probe_selection(study)
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
    for name, expected_sha in probe_selection["artifacts"].items():
        if artifacts.get(name) != expected_sha:
            raise ValueError("Root lock must bind every fresh probe artifact")
    probe_path = study / "probe_selection.json"
    if artifacts.get(str(probe_path)) != sha(probe_path):
        raise ValueError("Root lock must bind fresh selected probe layers")
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
    """Read only the new study's jointly generated cache and raw release."""
    study = Path(study).resolve()
    protocol = read_protocol(study)
    _check_stop(study)
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    if split == "test":
        verify_selection_lock(study)
    data_path = Path(protocol["data_dir"]) / f"{split}.parquet"
    if sha(data_path) != protocol["data_sha256"][split]:
        raise ValueError(f"Raw dataset changed: {split}")
    columns = ["base_id", "group_id", "source", "prompt_harm_label", "category_id", "type_label_mask"]
    frame = pd.read_parquet(data_path, columns=columns).sort_values("base_id", kind="stable").reset_index(drop=True)
    labels = label_arrays(frame)
    from scripts.hanguard import clean_ablation_features
    cache = clean_ablation_features.load(study, split)
    meta = cache.meta
    if (meta.base_id.astype(str).tolist() != frame.base_id.astype(str).tolist()
            or meta.source.tolist() != frame.source.tolist()
            or not np.array_equal(meta.label.to_numpy(), labels["binary"])):
        raise ValueError("New joint cache identity or labels differ from raw release")
    lengths = np.asarray(cache.lengths, dtype=np.int64)
    if lengths.shape != (len(meta),) or (lengths < 1).any():
        raise ValueError("Invalid full token lengths")
    sample = np.asarray(cache.last[np.asarray([0]), 31])
    if sample.ndim != 2 or len(sample) != 1:
        raise ValueError("New joint cache must contain all 32 last-token layers")
    return dict(last=cache.last, tokens=cache, meta=meta, frame=frame, lengths=lengths,
                labels=labels, hidden_size=int(sample.shape[-1]),
                identity=dict(dataset_sha256=protocol["data_sha256"][split],
                              joint_manifest_sha256=sha(cache.manifest_path), rows=len(meta)))


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
    raw = np.asarray(data["last"][rows], dtype=np.float32)
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
    return {arm: CleanAblationHead(hidden_size, width=cfg["width"], num_classes=2 if task == "binary" else 5,
                            arm=arm, dropout=cfg["dropout"], seed=seed,
                            selected_layer=cfg.get("selected_layer", 32)).to(device) for arm in ARMS}


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
                all_nine_arms_extended=True, selection_scope="validation_cross_entropy_only",
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
            or set(result.get("arms", {})) != set(ARMS)
            or result.get("probe_selection_sha256") != sha(Path(study) / "probe_selection.json")
            or result.get("selected_probe_layer") != read_json(Path(study) / "probe_selection.json")["selected_layers"][task]):
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
    probe_lock = verify_probe_selection(study)
    cfg["selected_layer"] = probe_lock["selected_layers"][task]
    train, validation = _load_split(study, "train"), _load_split(study, "validation")
    for name in ("base_id", "group_id"):
        if set(train["meta"][name].astype(str)) & set(validation["meta"][name].astype(str)):
            raise ValueError(f"Train/validation leakage: {name}")
    tr_rows, val_rows = _task_indices(train, task), _task_indices(validation, task)
    if not len(tr_rows) or not len(val_rows):
        raise ValueError("Each task needs train and validation supervision")
    identity = dict(protocol_sha256=sha(study / "protocol.json"), task=task, seed=seed,
                    settings=cfg, probe_selection_sha256=sha(study / "probe_selection.json"), sources=_sources(), training_identity=train["identity"],
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
                                     states=best_states, metadata={a: heads[a].metadata() for a in ARMS},
                                     probe_selection_sha256=sha(study / "probe_selection.json"),
                                     selected_probe_layer=cfg["selected_layer"]))
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
                     objective="two_class_softmax_ce" if task == "binary" else "native_known_harmful_five_class_softmax_ce",
                     probe_selection_sha256=sha(study / "probe_selection.json"),
                     selected_probe_layer=cfg["selected_layer"],
                     train_length_edges=np.quantile(train["lengths"][tr_rows], [.25, .5, .75]).tolist())
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



def group_delta_interval(delta, groups, draws=2000, seed=42):
    delta, groups = np.asarray(delta, dtype=float), np.asarray(groups).astype(str)
    if not len(delta) or delta.shape != groups.shape:
        raise ValueError("Aligned nonempty group deltas required")
    _, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for begin in range(0, draws, 16):
        choice = rng.integers(0, len(count), size=(min(16, draws - begin), len(count)))
        values[begin:begin + len(choice)] = sums[choice].sum(1) / count[choice].sum(1)
    return dict(unit="group_id cluster", repetitions=draws, groups=len(count), seed=seed,
                delta=float(delta.mean()), ci95=np.quantile(values, [.025, .975]).tolist())


def task_diagnostics(data, task, logits, selection, cfg, seed):
    rows = _task_indices(data, task)
    labels = _targets(data, task)[rows]
    predictions = {}
    for arm in ARMS:
        if task == "binary":
            predictions[arm] = (_probabilities(logits[arm])[rows, 1] >= selection["arms"][arm]["threshold"]).astype(int)
        else:
            predictions[arm] = logits[arm][rows].argmax(1)
    groups = data["meta"].group_id.to_numpy()[rows]
    hit = {arm: (value == labels).astype(float) for arm, value in predictions.items()}
    interaction = (hit["fusion_category"] - hit["category_only"]) - (hit["fusion_only"] - hit["baseline"])
    output = dict(interaction_accuracy=group_delta_interval(interaction, groups, cfg["bootstrap_draws"], seed),
                  length_edges_from_training=selection["train_length_edges"], stratified_changes={})
    source = data["meta"].source.to_numpy()[rows]
    length_bucket = np.searchsorted(selection["train_length_edges"], data["lengths"][rows], side="right")
    strata = {f"source:{value}": source == value for value in sorted(set(source))}
    strata.update({f"label:{value}": labels == value for value in sorted(set(labels))})
    strata.update({f"length_quartile:{value + 1}": length_bucket == value for value in range(4)})
    for candidate, baseline in CONTRASTS:
        name = f"{candidate}_minus_{baseline}"
        output["stratified_changes"][name] = {key: paired_changes(labels[mask], predictions[baseline][mask], predictions[candidate][mask])
                                             for key, mask in strata.items() if mask.any()}
    return output


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
    cfg["selected_layer"] = verify_probe_selection(study)["selected_layers"][task]
    checkpoint = torch.load(selection["checkpoint_path"], map_location="cpu", weights_only=True)
    heads = build_heads(data["hidden_size"], task, seed, cfg, device)
    if (checkpoint.get("protocol_sha256") != selection["protocol_sha256"]
            or checkpoint.get("probe_selection_sha256") != sha(study / "probe_selection.json")
            or checkpoint.get("selected_probe_layer") != cfg["selected_layer"]):
        raise ValueError("Selected checkpoint protocol or fresh probe layer differs")
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
                  interpretation="Only the complete untouched raw-release test is evaluated; no historical exclusion list or scores are read.")
    if task == "binary":
        labels = data["labels"]["binary"]
        masks = dict(full_original_test=np.ones(len(rows), dtype=bool))
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
        contrasts = CONTRASTS
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
        for candidate, baseline in CONTRASTS:
            pc, pb = logits[candidate].argmax(1) + 1, logits[baseline].argmax(1) + 1
            result["paired_contrasts"][f"{candidate}_minus_{baseline}"] = dict(
                changes=paired_changes(y, pb[mask], pc[mask]),
                bootstrap=multiclass_bootstrap(y, pc[mask], pb[mask], data["meta"].group_id.to_numpy()[mask],
                                              cfg["bootstrap_draws"], seed))
    result["diagnostics"] = task_diagnostics(data, task, logits, selection, cfg, seed)
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
        corresponding_both_stages={arm: (arm, arm) for arm in MAIN_ARMS},
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
        for candidate, baseline in CONTRASTS:
            if candidate not in MAIN_ARMS or baseline not in MAIN_ARMS:
                continue
            changes[f"{candidate}_minus_{baseline}"] = dict(
                changes=paired_changes(labels, predictions[baseline], predictions[candidate]),
                bootstrap=multiclass_bootstrap(labels, predictions[candidate], predictions[baseline],
                                              binary.group_id.to_numpy()[mask], draws, seed))
        hits = {arm: (prediction == labels).astype(float) for arm, prediction in predictions.items()}
        interaction = (hits["fusion_category"] - hits["category_only"]) - (hits["fusion_only"] - hits["baseline"])
        output[table] = dict(scope="native_two_source_known_safe_or_known_type", rows=int(mask.sum()),
                             arms=entries, paired_contrasts=changes,
                             interaction_accuracy=group_delta_interval(interaction, binary.group_id.to_numpy()[mask], draws, seed))
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
            for scope in ("full_original_test",):
                metrics = binary["arms"][arm][scope]["validation_threshold"]
                rows.append(dict(seed=seed, task="binary", scope=scope, arm=arm,
                                 accuracy=metrics["accuracy"], f1=metrics["f1"], auroc=metrics["auroc"],
                                 fpr=metrics["fpr"], macro_f1=None))
            metrics = category["arms"][arm]["native_known_harmful"]
            rows.append(dict(seed=seed, task="category", scope="native_known_harmful", arm=arm,
                             accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
            for name, system in systems.items():
                if arm not in system["arms"]:
                    continue
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
            if candidate.empty:
                continue
            deltas.append(dict(task=task, scope=scope, candidate=arm, baseline="baseline",
                               metrics={metric: _aggregate(candidate[metric].dropna() - baseline[metric].dropna())
                                        for metric in ("accuracy", "f1", "macro_f1", "auroc", "fpr")
                                        if metric in candidate and candidate[metric].notna().all()
                                        and baseline[metric].notna().all()}))
    mechanism_deltas, interactions = [], []
    for task in TASKS:
        for candidate, baseline in CONTRASTS:
            sub = frame[frame.task.eq(task)]
            a = sub[sub.arm.eq(candidate)].set_index("seed")
            b = sub[sub.arm.eq(baseline)].set_index("seed")
            mechanism_deltas.append(dict(task=task, candidate=candidate, baseline=baseline,
                metrics={metric: _aggregate(a[metric].dropna() - b[metric].dropna())
                         for metric in ("accuracy", "f1", "macro_f1", "auroc", "fpr")
                         if metric in a and a[metric].notna().all() and b[metric].notna().all()}))
        interactions.append(dict(task=task,
            accuracy=_aggregate([per_seed[str(seed)][task]["diagnostics"]["interaction_accuracy"]["delta"] for seed in seeds])))
    for name in per_seed[str(seeds[0])]["systems"]:
        interactions.append(dict(task=name,
            accuracy=_aggregate([per_seed[str(seed)]["systems"][name]["interaction_accuracy"]["delta"] for seed in seeds])))
    output = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(study / "protocol.json"),
                  rootselection_lock_sha256=sha(study / "selection_lock.json"),
                  seeds=seeds, arms=list(ARMS), per_seed=per_seed, aggregate=aggregates,
                  paired_delta_aggregate=deltas, mechanism_delta_aggregate=mechanism_deltas,
                  interaction_aggregate=interactions, outcome_selected_subset=False,
                  interpretation="All features, probe selections, heads and predictions are newly generated. The complete original test is primary; task accuracies are never added.")
    analysis = study / "analysis"
    analysis.mkdir(exist_ok=True)
    frame.to_csv(analysis / "metrics_by_seed.csv", index=False)
    dump(analysis / "results.json", output)
    dump(study / "results.json", output)
    return output


class LayerProbes(torch.nn.Module):
    """32 independent linear classifiers, with the same initial weights."""
    def __init__(self, hidden_size, num_classes, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            layer = torch.nn.Linear(hidden_size, num_classes)
        self.weight = torch.nn.Parameter(layer.weight.detach()[None].repeat(32, 1, 1))
        self.bias = torch.nn.Parameter(layer.bias.detach()[None].repeat(32, 1))

    def forward(self, raw):
        x = F.layer_norm(raw.float(), (raw.shape[-1],))
        return torch.einsum('bld,lcd->blc', x, self.weight) + self.bias[None]


def probe_folder(study, task, seed):
    return Path(study).resolve() / "probes" / task / f"s{int(seed)}"


def verify_probe_fit(study, task, seed):
    folder = probe_folder(study, task, seed)
    result = read_json(folder / "selection.json")
    if (result.get("task") != task or result.get("seed") != seed
            or result.get("protocol_sha256") != sha(Path(study) / "protocol.json")
            or len(result.get("layers", {})) != 32 or result.get("formal_test_opened") is not False):
        raise ValueError("Fresh probe selection identity differs")
    for filename, key in (("selected_probes.pt", "checkpoint_sha256"),
                          ("validation_predictions.parquet", "validation_predictions_sha256")):
        if sha(folder / filename) != result[key]:
            raise ValueError("Fresh probe selected artifact changed")
    return result


@torch.no_grad()
def _probe_predict(probes, data, rows, device, batch_size=128):
    probes.eval()
    output = []
    for begin in range(0, len(rows), batch_size):
        idx = rows[begin:begin + batch_size]
        raw = torch.from_numpy(np.asarray(data["last"][idx], dtype=np.float32)).to(device)
        output.append(probes(raw).float().cpu().numpy())
    return np.concatenate(output)


def fit_probes(study, task, seed, device="cpu"):
    study = Path(study).resolve()
    protocol = read_protocol(study)
    if task not in TASKS or seed not in protocol.get("seeds", [42, 43, 44]):
        raise ValueError("Probe task/seed is unregistered")
    folder = probe_folder(study, task, seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "selection.json").exists():
        return verify_probe_fit(study, task, seed)
    if (study / "probe_selection.json").exists() or (study / "selection_lock.json").exists():
        raise ValueError("Cannot change probes after selected layer lock")
    cfg = settings(protocol)
    epochs = int(protocol.get("probe_epochs", 12))
    if not 1 <= epochs <= 40:
        raise ValueError("Probe epoch budget must be positive and at most 40")
    train, validation = _load_split(study, "train"), _load_split(study, "validation")
    for name in ("base_id", "group_id"):
        if set(train["meta"][name].astype(str)) & set(validation["meta"][name].astype(str)):
            raise ValueError(f"Train/validation leakage in probe: {name}")
    tr_rows, val_rows = _task_indices(train, task), _task_indices(validation, task)
    if not len(tr_rows) or not len(val_rows):
        raise ValueError("Probe requires both training and validation supervision")
    probes = LayerProbes(train["hidden_size"], 2 if task == "binary" else 5, seed).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    identity = dict(protocol_sha256=sha(study / "protocol.json"), task=task, seed=seed,
                    settings=cfg, epochs=epochs, sources=_sources(),
                    training_identity=train["identity"], validation_identity=validation["identity"],
                    device_type=torch.device(device).type)
    completed, history = 0, []
    best_ce = np.full(32, np.inf)
    best_epochs = np.zeros(32, dtype=np.int64)
    best_states = _cpu(probes.state_dict())
    resume = folder / "epoch_resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("schema") != "clean_probe_resume_v1" or saved.get("identity") != identity:
            raise ValueError("Probe resume identity differs")
        completed, history = int(saved["completed_epochs"]), saved["history"]
        if not 1 <= completed <= epochs or len(history) != completed:
            raise ValueError("Invalid probe resume epoch history")
        best_ce, best_epochs = np.asarray(saved["best_ce"]), np.asarray(saved["best_epochs"])
        best_states = saved["best_states"]
        probes.load_state_dict(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], device)
    val_targets = torch.as_tensor(_targets(validation, task)[val_rows], dtype=torch.long)
    for epoch in range(completed, epochs):
        _check_stop(study)
        probes.train()
        totals = np.zeros(32)
        batches = macro_batches(tr_rows, seed, epoch, cfg["effective_batch"])
        for step, rows in enumerate(batches):
            _check_stop(study)
            _set_seed(_step_seed(seed, epoch, step, 0), device)
            raw = torch.from_numpy(np.asarray(train["last"][rows], dtype=np.float32)).to(device)
            target = torch.as_tensor(_targets(train, task)[rows], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = probes(raw)
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     target[:, None].expand(-1, 32).reshape(-1), reduction="none").reshape(len(rows), 32).mean(0)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite fresh probe loss")
            losses.sum().backward()
            # Clip each classifier separately; classifiers cannot influence one
            # another through the vectorized gradient clipping operation.
            norms = (probes.weight.grad.square().sum((1, 2)) + probes.bias.grad.square().sum(1)).sqrt()
            if not torch.isfinite(norms).all():
                raise FloatingPointError("Nonfinite fresh probe gradients")
            scale = (cfg["grad_clip"] / (norms + 1e-6)).clamp(max=1.)
            probes.weight.grad.mul_(scale[:, None, None])
            probes.bias.grad.mul_(scale[:, None])
            optimizer.step()
            totals += losses.detach().cpu().numpy() * len(rows)
        logits = _probe_predict(probes, validation, val_rows, device, cfg["effective_batch"])
        val_loss = F.cross_entropy(torch.from_numpy(logits).double().reshape(-1, logits.shape[-1]),
                                  val_targets[:, None].expand(-1, 32).reshape(-1), reduction="none").reshape(len(val_rows), 32).mean(0).numpy()
        for layer in range(32):
            if val_loss[layer] < best_ce[layer]:
                best_ce[layer], best_epochs[layer] = val_loss[layer], epoch + 1
                for name, value in probes.state_dict().items():
                    best_states[name][layer] = value[layer].detach().cpu()
        history.append(dict(epoch=epoch + 1, train_ce=(totals / len(tr_rows)).tolist(), validation_ce=val_loss.tolist()))
        atomic_save(resume, dict(schema="clean_probe_resume_v1", identity=identity, completed_epochs=epoch + 1,
                                history=history, best_ce=best_ce.tolist(), best_epochs=best_epochs.tolist(),
                                best_states=best_states, current_state=_cpu(probes.state_dict()),
                                optimizer=_cpu(optimizer.state_dict()), rng=_rng_state(device)))
        dump(folder / "progress.json", dict(state="training_probes", task=task, seed=seed,
                                             completed_epochs=epoch + 1, budget=epochs,
                                             validation_ce=val_loss.tolist(), device=str(device)))
        print(json.dumps(dict(task=task, seed=seed, probe_epoch=epoch + 1, budget=epochs,
                              best_layer=int(np.argmin(best_ce)) + 1)), flush=True)
    probes.load_state_dict(best_states)
    logits = _probe_predict(probes, validation, val_rows, device, cfg["effective_batch"])
    frame = _safe_meta(validation, val_rows)
    probability_columns = {}
    for layer in range(32):
        probs = _probabilities(logits[:, layer])
        for category in range(probs.shape[-1]):
            probability_columns[f"L{layer + 1}_p{category}"] = probs[:, category]
    frame = pd.concat([frame, pd.DataFrame(probability_columns)], axis=1)
    prediction_path = folder / "validation_predictions.parquet"
    _save_frame(prediction_path, frame)
    checkpoint_path = folder / "selected_probes.pt"
    atomic_save(checkpoint_path, dict(schema=SCHEMA, task=task, seed=seed,
                                     protocol_sha256=identity["protocol_sha256"], states=best_states))
    layers = {str(layer + 1): dict(best_epoch=int(best_epochs[layer]), validation_ce=float(best_ce[layer]),
                                  parameters=int(probes.weight[layer].numel() + probes.bias[layer].numel()))
              for layer in range(32)}
    result = dict(schema=SCHEMA, task=task, seed=seed, protocol_sha256=identity["protocol_sha256"],
                  layers=layers, epochs=epochs, checkpoint_sha256=sha(checkpoint_path),
                  validation_predictions_sha256=sha(prediction_path), training_rows=len(tr_rows),
                  validation_rows=len(val_rows), formal_test_opened=False, selection_scope="validation_only",
                  source_sha256=identity["sources"])
    dump(folder / "selection.json", result)
    dump(folder / "progress.json", dict(state="selected", task=task, seed=seed, completed_epochs=epochs))
    return result


def probe_validation_analysis(study):
    """Label association and error complementarity from fresh validation only."""
    study = Path(study).resolve()
    protocol = read_protocol(study)
    result = dict(scope="validation_only", formal_test_opened=False, tasks={})
    for task in TASKS:
        task_results = {}
        for seed in protocol.get("seeds", [42, 43, 44]):
            verify_probe_fit(study, task, seed)
            frame = pd.read_parquet(probe_folder(study, task, seed) / "validation_predictions.parquet")
            labels = frame.binary_label.to_numpy() if task == "binary" else frame.category_id.to_numpy() - 1
            classes = 2 if task == "binary" else 5
            probs = np.stack([frame[[f"L{layer}_p{c}" for c in range(classes)]].to_numpy() for layer in range(1, 33)], axis=1)
            decisions = probs.argmax(2)
            correct = decisions == labels[:, None]
            layer_stats = []
            for layer in range(32):
                p = probs[:, layer]
                ce = -np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-300)).mean()
                metrics = multiclass_metrics(labels, decisions[:, layer], range(classes))
                layer_stats.append(dict(layer=layer + 1, validation_ce=float(ce), metrics=metrics))
            oracle = np.empty((32, 32))
            rescued = np.empty((32, 32), dtype=np.int64)
            error_phi = [[None] * 32 for _ in range(32)]
            for a in range(32):
                for b in range(32):
                    oracle[a, b] = (correct[:, a] | correct[:, b]).mean()
                    rescued[a, b] = (~correct[:, a] & correct[:, b]).sum()
                    xa, xb = correct[:, a].astype(float), correct[:, b].astype(float)
                    denom = xa.std() * xb.std()
                    error_phi[a][b] = float(((xa - xa.mean()) * (xb - xb.mean())).mean() / denom) if denom else None
            task_results[str(seed)] = dict(layers=layer_stats,
                pairwise_oracle_accuracy=oracle.tolist(),
                pairwise_rescue_count=rescued.tolist(), pairwise_error_phi=error_phi,
                error_phi_undefined_value=None,
                interpretation="Linear label predictability and pairwise correct-error associations are diagnostic; they do not prove causal feature relevance.")
        result["tasks"][task] = task_results
    return result


def lock_probe_layers(study):
    study = Path(study).resolve()
    protocol = read_protocol(study)
    if (study / "probe_selection.json").exists():
        return verify_probe_selection(study)
    layers, selected, artifacts, selections = {}, {}, {}, {}
    for task in TASKS:
        losses = []
        for seed in protocol.get("seeds", [42, 43, 44]):
            local = verify_probe_fit(study, task, seed)
            losses.append([local["layers"][str(i)]["validation_ce"] for i in range(1, 33)])
            folder = probe_folder(study, task, seed)
            selections[f"{task}/s{seed}"] = sha(folder / "selection.json")
            for filename in ("selection.json", "selected_probes.pt", "validation_predictions.parquet"):
                path = folder / filename
                artifacts[str(path)] = sha(path)
        means = np.asarray(losses).mean(0)
        layers[task] = means.tolist()
        selected[task] = int(np.argmin(means)) + 1
    diagnostics_path = study / "probes" / "validation_analysis.json"
    dump(diagnostics_path, probe_validation_analysis(study))
    artifacts[str(diagnostics_path)] = sha(diagnostics_path)
    result = dict(schema="clean_probe_layer_selection_v1", complete=True,
                  protocol_sha256=sha(study / "protocol.json"), selected_layers=selected,
                  mean_validation_ce=layers, probe_selections=selections, artifacts=artifacts,
                  criterion="minimum across-seed mean validation CE; ties choose lowest layer index",
                  formal_test_opened=False)
    dump(study / "probe_selection.json", result)
    return result


def verify_probe_selection(study):
    study = Path(study).resolve()
    protocol = read_protocol(study)
    result = read_json(study / "probe_selection.json")
    expected = {f"{task}/s{seed}" for task in TASKS for seed in protocol.get("seeds", [42, 43, 44])}
    if (result.get("complete") is not True or result.get("protocol_sha256") != sha(study / "protocol.json")
            or set(result.get("probe_selections", {})) != expected or result.get("formal_test_opened") is not False):
        raise ValueError("All fresh probe fits must be locked before fitting selected-layer head")
    for task in TASKS:
        losses = []
        for seed in protocol.get("seeds", [42, 43, 44]):
            local = verify_probe_fit(study, task, seed)
            folder = probe_folder(study, task, seed)
            if sha(folder / "selection.json") != result["probe_selections"][f"{task}/s{seed}"]:
                raise ValueError("Fresh probe selection changed after layer lock")
            losses.append([local["layers"][str(i)]["validation_ce"] for i in range(1, 33)])
            for filename in ("selection.json", "selected_probes.pt", "validation_predictions.parquet"):
                path = folder / filename
                if result.get("artifacts", {}).get(str(path)) != sha(path):
                    raise ValueError("Probe layer lock omitted or changed an artifact")
        means = np.asarray(losses).mean(0)
        if (not np.array_equal(means, result["mean_validation_ce"][task])
                or result["selected_layers"][task] != int(np.argmin(means)) + 1):
            raise ValueError("Selected layer does not follow preregistered fresh validation criterion")
    for name, digest in result["artifacts"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(study / "probes") or sha(path) != digest:
            raise ValueError("Fresh probe artifact is outside study or changed")
    return result


def selection_artifacts(study):
    """Return the complete artifact mapping for the coordinator's test lock."""
    study = Path(study).resolve()
    protocol = read_protocol(study)
    probe = verify_probe_selection(study)
    artifacts = dict(probe["artifacts"])
    artifacts[str(study / "probe_selection.json")] = sha(study / "probe_selection.json")
    selections = {}
    for task in TASKS:
        for seed in protocol.get("seeds", [42, 43, 44]):
            local = _verify_local_selection(study, task, seed)
            folder = task_folder(study, task, seed)
            selections[f"{task}/s{seed}"] = sha(folder / "selection.json")
            for path in (folder / "selection.json", Path(local["checkpoint_path"]),
                         Path(local["validation_predictions_path"])):
                artifacts[str(path.resolve())] = sha(path)
    return dict(selections=selections, artifacts=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probes", "lock-probes", "fit", "evaluate", "summarize"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.study)
    elif args.command == "lock-probes":
        lock_probe_layers(args.study)
    else:
        if args.task is None or args.seed is None:
            parser.error("--task and --seed are required for fitting/evaluation")
        {"fit": fit_task, "evaluate": evaluate_task, "probes": fit_probes}[args.command](args.study, args.task, args.seed, args.device)


if __name__ == "__main__":
    main()
