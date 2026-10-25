"""Epoch-resumable execution of the registered frozen-base head comparison.

The original readout implementation remains immutable. This module changes only
durability: a complete epoch commits current weights, optimizer state and each
candidate's validation-best snapshot together. Budget extensions still refit
every candidate from the original common seed.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.hanguard import frozen_base_readout_heads as original

ROOT, ARMS, FUSION_ARMS, SCHEMA = original.ROOT, original.ARMS, original.FUSION_ARMS, original.SCHEMA
RESUME_SCHEMA = "hanguard_frozen_base_epoch_resume_v1"
FIT_SCHEMA = "hanguard_frozen_base_resumable_fit_v1"

# Reuse the scientific implementation, cache validation and test barrier.
VectorizedLayerMLPs = original.VectorizedLayerMLPs
settings, read_json = original.settings, original.read_json
load_cache, verify_selection_lock, evaluate = original.load_cache, original.verify_selection_lock, original.evaluate
choose_best_layer, build_fusion_heads = original.choose_best_layer, original.build_fusion_heads
epoch_batches, _step_seed, _batch = original.epoch_batches, original._step_seed, original._batch
binary_metrics, sigmoid = original.binary_metrics, original.sigmoid
dump, sha256, state_digest = original.dump, original.sha256, original.state_digest
atomic_torch_save = original.atomic_torch_save
accuracy_threshold, budget_extension = original.accuracy_threshold, original.budget_extension
assert_disjoint, parameter_audit = original.assert_disjoint, original.parameter_audit
_head_logits = original._head_logits


def check_stop(study):
    """The runner may wrap this hook with its identity-bound worker stop check."""
    original.check_stop(study)


def _legacy_sources():
    return (Path(original.__file__).resolve(), ROOT / "scripts/hanguard/repaired_heads.py",
            ROOT / "scripts/hanguard/ab_frozen_readout_diagnostics.py",
            ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py")


def _bindings(study, extra_source_paths=()):
    study = Path(study).resolve()
    manifests = {str(study / "caches" / f"{split}.manifest.json"):
                 sha256(study / "caches" / f"{split}.manifest.json")
                 for split in ("train", "validation")}
    selection_sources = {str(p): sha256(p) for p in _legacy_sources()}
    for value in extra_source_paths:
        path = Path(value).resolve()
        if not path.is_relative_to(study) or not path.is_file():
            raise ValueError("Extra selection source must be an existing study-local artifact")
        selection_sources[str(path)] = sha256(path)
    actual_sources = dict(selection_sources)
    actual_sources[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    return manifests, selection_sources, actual_sources


def _resume_identity(study, s, budget, stage, best=None):
    study = Path(study).resolve()
    marker = study / "heads/fit_started.json"
    extras = ()
    if marker.exists():
        started = read_json(marker)
        if started.get("schema") == FIT_SCHEMA:
            extras = started.get("extra_source_paths", ())
    manifests, _, sources = _bindings(study, extras)
    # Round-trip tuples to lists once, so loaded JSON and Python settings agree.
    normalized = {k: list(v) if isinstance(v, tuple) else v for k, v in s.items()}
    return dict(protocol_sha256=sha256(study / "protocol.json"), settings=normalized,
                train_validation_manifest_sha256=manifests, source_sha256=sources,
                stage=stage, budget=int(budget), best_single_layer=best)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _rng_state(device):
    numpy_state = np.random.get_state()
    return dict(device_type=torch.device(device).type, python=random.getstate(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                       int(numpy_state[3]), float(numpy_state[4])],
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=[value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                if torch.device(device).type == "cuda" else [])


def _restore_rng(state, device):
    if state["device_type"] != torch.device(device).type:
        raise ValueError("Resume RNG device type differs from the saved execution")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_resume(path, payload):
    """Atomically replace one complete epoch; an interrupted write keeps the old one."""
    atomic_torch_save(_cpu_copy(payload), Path(path))


def _load_resume(path, identity, steps_per_epoch):
    path = Path(path)
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != RESUME_SCHEMA or value.get("identity") != identity:
        raise ValueError(f"Resume checkpoint identity mismatch: {path}")
    completed = value.get("completed_epochs")
    if (type(completed) is not int or not 1 <= completed <= identity["budget"] or
            len(value.get("history", [])) != completed or
            [row.get("epoch") for row in value["history"]] != list(range(1, completed + 1)) or
            value.get("global_step") != completed * steps_per_epoch):
        raise ValueError(f"Invalid completed epoch/step history in resume checkpoint: {path}")
    return value


@torch.no_grad()
def _predict_layers(model, cache, batch_size, device, study):
    model.eval()
    output = []
    for start in range(0, len(cache), batch_size):
        check_stop(study)
        output.append(model(_batch(cache, np.arange(start, min(start + batch_size, len(cache))), device)).cpu().numpy())
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite layer predictions")
    return result


@torch.no_grad()
def _predict_heads(heads, specs, cache, batch_size, device, study):
    layers = sorted({k for _, ks in specs.values() for k in ks})
    positions = {k: i for i, k in enumerate(layers)}
    output = {arm: [] for arm in heads}
    for head in heads.values():
        head.eval()
    for start in range(0, len(cache), batch_size):
        check_stop(study)
        raw = _batch(cache, np.arange(start, min(start + batch_size, len(cache))), device, layers)
        for arm, head in heads.items():
            mode, ks = specs[arm]
            output[arm].append(_head_logits(head, raw, [positions[k] for k in ks], mode).cpu().numpy())
    result = {arm: np.concatenate(values) for arm, values in output.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("Nonfinite head predictions")
    return result


def _train_layers(study, folder, train, validation, y_train, y_val, s, budget, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = _resume_identity(study, s, budget, "layers")
    path = folder / "layers_resume.pt"
    batches_per_epoch = (len(train) + s["batch_size"] - 1) // s["batch_size"]
    saved = _load_resume(path, identity, batches_per_epoch)
    if saved is None and (folder / "layer_history.json").exists():
        raise RuntimeError("Layer history without a complete epoch checkpoint cannot resume")
    model = VectorizedLayerMLPs(train.shape[-1], s["width"], train.shape[1], s["seed"], s["dropout"]).to(device)
    initial = _cpu_copy(model.state_dict())
    initial_sha = state_digest(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["wd"])
    best_bce = np.full(train.shape[1], np.inf)
    best_epoch = np.zeros(train.shape[1], dtype=int)
    best_state = _cpu_copy(model.state_dict())
    history, audit, global_step, completed = [], [], 0, 0
    if saved is not None:
        if saved["initial_state_sha256"] != initial_sha:
            raise ValueError("Layer resume initialization differs")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        best_state, history, audit = saved["best_state"], saved["history"], saved["audit"]
        best_bce, best_epoch = np.asarray(saved["best_bce"]), np.asarray(saved["best_epoch"], dtype=int)
        global_step, completed = saved["global_step"], saved["completed_epochs"]
        if (best_bce.shape != (train.shape[1],) or best_epoch.shape != best_bce.shape or
                not np.isfinite(best_bce).all() or (best_epoch < 1).any() or (best_epoch > completed).any()):
            raise ValueError("Invalid per-layer best epoch in resume checkpoint")
        _restore_rng(saved["rng"], device)
    for epoch in range(completed, budget):
        check_stop(study)
        total = np.zeros(train.shape[1])
        for step, rows in enumerate(epoch_batches(len(train), s["seed"], epoch, s["batch_size"])):
            check_stop(study)
            model.train()
            torch.manual_seed(_step_seed(s["seed"], epoch, step))
            logits = model(_batch(train, rows, device))
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            losses = F.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits), reduction="none").mean(0)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite layer loss")
            optimizer.zero_grad(set_to_none=True)
            losses.sum().backward()
            norms = model.clip_independent_gradients(s["clip"])
            optimizer.step()
            global_step += 1
            total += losses.detach().cpu().numpy() * len(rows)
            if global_step in (1, 2, 10):
                audit.append(dict(step=global_step, per_layer_grad_norm=norms.detach().cpu().tolist(),
                                  all_gradients_finite=True,
                                  parameter_delta_norm={k: float((v.detach().cpu() - initial[k]).norm())
                                                        for k, v in model.state_dict().items()}))
        validation_logits = _predict_layers(model, validation, s["batch_size"], device, study)
        bce = np.asarray([binary_metrics(y_val, sigmoid(validation_logits[:, k]), .5)["bce"]
                          for k in range(train.shape[1])])
        improved = bce < best_bce
        best_bce[improved], best_epoch[improved] = bce[improved], epoch + 1
        for key, value in model.state_dict().items():
            best_state[key][improved] = value.detach().cpu()[improved]
        history.append(dict(epoch=epoch + 1, train_bce=(total / len(train)).tolist(), dev_bce=bce.tolist()))
        _save_resume(path, dict(schema=RESUME_SCHEMA, identity=identity, completed_epochs=epoch + 1,
                               global_step=global_step, current_state=model.state_dict(),
                               optimizer_state=optimizer.state_dict(), best_state=best_state,
                               best_bce=best_bce.tolist(), best_epoch=best_epoch.tolist(),
                               history=history, audit=audit, initial_state_sha256=initial_sha,
                               rng=_rng_state(device)))
        dump(folder / "layer_history.json", history)
        print(json.dumps(dict(stage="all_layer_mlps", epoch=epoch + 1, budget=budget)), flush=True)
    # Derived artifacts can be rebuilt even when the preceding process exited
    # immediately after committing its last epoch.
    dump(folder / "layer_history.json", history)
    model.load_state_dict(best_state)
    validation_logits = _predict_layers(model, validation, s["batch_size"], device, study)
    dump(folder / "layer_gradient_audit.json", audit)
    atomic_torch_save(dict(state=best_state, best_epoch=best_epoch.tolist(), best_bce=best_bce.tolist(),
                           initial_state_sha256=initial_sha), folder / "all_layer_mlps.pt")
    return model, validation_logits, best_bce, best_epoch, history


def _train_fusion(study, folder, train, validation, y_train, y_val, s, budget, best, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = _resume_identity(study, s, budget, "fusion", int(best))
    path = folder / "fusion_resume.pt"
    batches_per_epoch = (len(train) + s["batch_size"] - 1) // s["batch_size"]
    saved = _load_resume(path, identity, batches_per_epoch)
    if saved is None and (folder / "fusion_history.json").exists():
        raise RuntimeError("Fusion history without a complete epoch checkpoint cannot resume")
    heads, specs, matched = build_fusion_heads(train.shape[-1], s, best, device)
    initial = {a: _cpu_copy(dict(h.named_parameters())) for a, h in heads.items()}
    initial_sha = {a: state_digest(h.state_dict()) for a, h in heads.items()}
    optimizers = {a: torch.optim.AdamW(h.parameters(), lr=s["lr"], weight_decay=s["wd"]) for a, h in heads.items()}
    best_bce, best_epoch = {a: np.inf for a in heads}, {a: 0 for a in heads}
    best_states, history, audit, global_step, completed = {}, [], [], 0, 0
    layers = sorted(set(s["layers"]) | {best})
    positions = {k: i for i, k in enumerate(layers)}
    if saved is not None:
        if saved["initial_state_sha256"] != initial_sha or saved["specs"] != specs or saved["matched"] != matched:
            raise ValueError("Fusion resume initialization/specs differ")
        for arm, head in heads.items():
            head.load_state_dict(saved["current_states"][arm], strict=True)
            optimizers[arm].load_state_dict(saved["optimizer_states"][arm])
        best_bce, best_epoch, best_states = saved["best_bce"], saved["best_epoch"], saved["best_states"]
        history, audit = saved["history"], saved["audit"]
        global_step, completed = saved["global_step"], saved["completed_epochs"]
        if (set(best_bce) != set(heads) or set(best_epoch) != set(heads) or set(best_states) != set(heads) or
                any(not np.isfinite(best_bce[a]) or not 1 <= best_epoch[a] <= completed for a in heads)):
            raise ValueError("Invalid fusion best epoch in resume checkpoint")
        _restore_rng(saved["rng"], device)
    for epoch in range(completed, budget):
        check_stop(study)
        total = {a: 0. for a in heads}
        for step, rows in enumerate(epoch_batches(len(train), s["seed"], epoch, s["batch_size"])):
            check_stop(study)
            raw = _batch(train, rows, device, layers)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            global_step += 1
            for arm, head in heads.items():
                head.train()
                torch.manual_seed(_step_seed(s["seed"], epoch, step))
                mode, ks = specs[arm]
                logits = _head_logits(head, raw, [positions[k] for k in ks], mode)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {arm}")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), s["clip"], error_if_nonfinite=True)
                optimizers[arm].step()
                total[arm] += float(loss.detach()) * len(rows)
                if global_step in (1, 2, 10):
                    audit.append(dict(arm=arm, **parameter_audit(head, initial[arm], global_step)))
        validation_logits = _predict_heads(heads, specs, validation, s["batch_size"], device, study)
        bce = {a: binary_metrics(y_val, sigmoid(p), .5)["bce"] for a, p in validation_logits.items()}
        for arm in heads:
            if bce[arm] < best_bce[arm]:
                best_bce[arm], best_epoch[arm] = bce[arm], epoch + 1
                best_states[arm] = _cpu_copy(heads[arm].state_dict())
        history.append(dict(epoch=epoch + 1, train_bce={a: t / len(train) for a, t in total.items()}, dev_bce=bce))
        _save_resume(path, dict(schema=RESUME_SCHEMA, identity=identity, completed_epochs=epoch + 1,
                               global_step=global_step,
                               current_states={a: h.state_dict() for a, h in heads.items()},
                               optimizer_states={a: opt.state_dict() for a, opt in optimizers.items()},
                               best_states=best_states, best_bce=best_bce, best_epoch=best_epoch,
                               history=history, audit=audit, specs=specs, matched=matched,
                               initial_state_sha256=initial_sha, rng=_rng_state(device)))
        dump(folder / "fusion_history.json", history)
        print(json.dumps(dict(stage="fusion_heads", epoch=epoch + 1, budget=budget)), flush=True)
    dump(folder / "fusion_history.json", history)
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
    dump(folder / "fusion_gradient_audit.json", audit)
    validation_logits = _predict_heads(heads, specs, validation, s["batch_size"], device, study)
    return heads, specs, matched, validation_logits, best_bce, best_epoch, history, best_states


def fit(study, device="cuda", extra_source_paths=()):
    study = Path(study).resolve()
    check_stop(study)
    folder = study / "heads"
    if (study / "selection_lock.json").exists():
        return verify_selection_lock(study)
    protocol = read_json(study / "protocol.json")
    protocol_sha = sha256(study / "protocol.json")
    s = settings(protocol)
    bindings, source_sha, runtime_source_sha = _bindings(study, extra_source_paths)
    extra_sources = sorted({str(Path(value).resolve()) for value in extra_source_paths})
    marker = dict(schema=FIT_SCHEMA, protocol_sha256=protocol_sha, test_opened=False,
                  source_sha256=runtime_source_sha, selection_source_sha256=source_sha,
                  train_validation_manifest_sha256=bindings, settings=_resume_identity(study, s, s["epochs"], "layers")["settings"],
                  extra_source_paths=extra_sources)
    started_path = folder / "fit_started.json"
    if started_path.exists():
        if read_json(started_path) != marker:
            raise RuntimeError("Incomplete fit identity differs; existing history cannot automatically retry")
        candidates = list(folder.glob("budget_*/layers_resume.pt")) + list(folder.glob("budget_*/fusion_resume.pt"))
        if not candidates:
            raise RuntimeError("Incomplete fit has no complete epoch checkpoint; automatic retry refused")
        if not (folder / f"budget_{s['epochs']}" / "layers_resume.pt").is_file():
            raise RuntimeError("Initial layer checkpoint is missing; automatic retry refused")
    train, train_meta, _ = load_cache(study, "train")
    validation, val_meta, _ = load_cache(study, "validation")
    if train.shape[1:] != validation.shape[1:] or max(s["layers"]) > train.shape[1]:
        raise ValueError("Train/validation dimensions or fixed layers differ")
    assert_disjoint(train_meta, val_meta)
    folder.mkdir(parents=True, exist_ok=True)
    if not started_path.exists():
        if any(folder.iterdir()):
            raise RuntimeError("Unregistered existing head artifacts cannot automatically retry")
        dump(started_path, marker)
    started, attempts, budget = time.monotonic(), [], s["epochs"]
    y_train, y_val = train_meta.label.to_numpy(dtype=int), val_meta.label.to_numpy(dtype=int)
    while True:
        attempt = folder / f"budget_{budget}"
        attempt.mkdir(exist_ok=True)
        attempt_start = time.monotonic()
        model, layer_logits, layer_bce, layer_epochs, layer_history = _train_layers(
            study, attempt, train, validation, y_train, y_val, s, budget, device)
        best = choose_best_layer(layer_bce)
        heads, specs, matched, fusion_logits, fusion_bce, fusion_epochs, fusion_history, fusion_states = _train_fusion(
            study, attempt, train, validation, y_train, y_val, s, budget, best, device)
        extension_layer = budget_extension(layer_history, {str(k + 1): int(e) for k, e in enumerate(layer_epochs)}, budget, s["max_epochs"])
        extension_fusion = budget_extension(fusion_history, fusion_epochs, budget, s["max_epochs"])
        extension = extension_layer or extension_fusion
        attempts.append(dict(budget=budget, seconds=time.monotonic() - attempt_start,
                             best_single_layer=best, layer_extension=extension_layer, fusion_extension=extension_fusion,
                             discarded_for_common_budget_extension=bool(extension)))
        dump(folder / "attempts.json", attempts)
        if not extension:
            break
        budget = extension["new_budget"]
        del model, heads
    states = dict(last=model.layer_state(train.shape[1] - 1),
                  best_single=model.layer_state(best - 1), **fusion_states)
    specs.update(last=("last", (train.shape[1],)), best_single=("last", (best,)))
    validation_logits = dict(last=layer_logits[:, -1], best_single=layer_logits[:, best - 1], **fusion_logits)
    thresholds = {a: accuracy_threshold(y_val, sigmoid(p)) for a, p in validation_logits.items()}
    checkpoint = folder / "selected_heads.pt"
    atomic_torch_save(dict(states=states, specs=specs, hidden_size=train.shape[-1]), checkpoint)
    epochs = dict(last=int(layer_epochs[-1]), best_single=int(layer_epochs[best - 1]), **fusion_epochs)
    val_metrics = {a: binary_metrics(y_val, sigmoid(p), thresholds[a]) for a, p in validation_logits.items()}
    selection = dict(schema=SCHEMA, complete=True, no_lora=True, best_single_layer=best,
                     best_single_definition="minimum validation BCE among 32 independent equally trained MLP checkpoints; lowest layer index breaks ties",
                     layer_validation_bce=layer_bce.tolist(), layer_best_epochs=layer_epochs.tolist(),
                     thresholds=thresholds, best_epochs=epochs, final_common_epoch_budget=budget,
                     total_seconds=time.monotonic() - started, attempts=attempts,
                     validation_metrics=val_metrics, matched_initialization=matched,
                     specs={a: dict(mode=m, layers=list(ks)) for a, (m, ks) in specs.items()},
                     checkpoint_sha256=sha256(checkpoint), protocol_sha256=protocol_sha,
                     train_validation_manifest_sha256=bindings, source_sha256=source_sha,
                     selection_scope="validation only; test cache unopened", test_opened=False,
                     execution_schema=FIT_SCHEMA,
                     extension_rule="12/24/40 validation BCE milestones; any arm may trigger common re-fit; all final arms have the same budget")
    selection_path = folder / "selection.json"
    dump(selection_path, selection)
    val_frame = val_meta.copy()
    for arm, logits in validation_logits.items():
        val_frame[f"probability_{arm}"] = sigmoid(logits)
    val_frame.to_parquet(folder / "validation_predictions.parquet", index=False)
    for filename, digest in {**bindings, **runtime_source_sha}.items():
        if sha256(filename) != digest:
            raise ValueError(f"Input/source changed during fit: {filename}")
    if sha256(study / "protocol.json") != protocol_sha:
        raise ValueError("Protocol changed during fit")
    final_files = [checkpoint, selection_path, folder / "validation_predictions.parquet",
                   folder / "attempts.json", started_path, attempt / "all_layer_mlps.pt", attempt / "layer_history.json",
                   attempt / "fusion_history.json", attempt / "layer_gradient_audit.json", attempt / "fusion_gradient_audit.json",
                   attempt / "layers_resume.pt", attempt / "fusion_resume.pt"]
    artifacts = {str(p.relative_to(study)): sha256(p) for p in final_files}
    artifacts.update(bindings)
    artifacts.update(source_sha)
    for split in ("train", "validation"):
        manifest = read_json(study / "caches" / f"{split}.manifest.json")
        for key, digest_key in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            artifacts[manifest[key]] = manifest[digest_key]
    lock = dict(schema=SCHEMA, complete=True, test_opened=False, protocol_sha256=protocol_sha,
                artifacts=artifacts)
    check_stop(study)
    dump(study / "selection_lock.json", lock)
    return lock
