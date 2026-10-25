"""Fresh frozen-foundation readouts with validation-only selection and a test barrier.

No backbone or adapter is loaded here. All 32 independent MLPs are vectorized,
with paired initialization, minibatches and dropout draws. Test cache access is
permitted only after every checkpoint and selection has been hash locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.ab_frozen_readout_diagnostics import (  # noqa: E402
    _head_logits, budget_extension, parameter_audit,
)
from scripts.hanguard.frozen_lora_layer_diagnostics import (  # noqa: E402
    accuracy_threshold, assert_capacity_matched_initialization, atomic_torch_save,
    binary_metrics, dump, group_cluster_bootstrap, ordered_base_id_sha256,
    sha256, sigmoid, state_digest, validate_feature_cache,
)
from scripts.hanguard.repaired_heads import ExperimentHead  # noqa: E402

ARMS = ("last", "best_single", "repeat_best", "fusion_fixed", "fusion_fixed_best_anchor")
FUSION_ARMS = ARMS[2:]
SCHEMA = "hanguard_fresh_frozen_base_readouts_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def check_stop(study):
    study = Path(study).resolve()
    for folder in (study, study / "heads"):
        if (folder / "STOP").exists():
            raise RuntimeError(f"Study stopped: {folder / 'STOP'}")


def epoch_batches(rows, seed, epoch, batch_size):
    """Shared row order for every layer and every head, including restarts."""
    order = np.random.default_rng(seed * 1009 + epoch).permutation(rows)
    return [order[i:i + batch_size] for i in range(0, rows, batch_size)]


def _step_seed(seed, epoch, step):
    return (seed * 1000003 + epoch * 1009 + step) % (2**63 - 1)


def settings(protocol):
    s = protocol["settings"]
    result = dict(seed=int(s.get("seed", 42)), width=int(s.get("width", 128)),
                  dropout=float(s.get("dropout", .1)), batch_size=int(s.get("batch_size", s.get("batch", 256))),
                  lr=float(s.get("lr", .001)), wd=float(s.get("weight_decay", s.get("wd", .001))),
                  clip=float(s.get("grad_clip", s.get("clip", 5))), epochs=int(s.get("epochs", 12)),
                  max_epochs=int(s.get("max_epochs", 40)), layers=tuple(s.get("layers", (8, 16, 24, 32))),
                  bootstrap_draws=int(s.get("bootstrap_draws", 1000)))
    if (min(result[k] for k in ("width", "batch_size", "epochs")) < 1 or
            not result["epochs"] <= result["max_epochs"] <= 40 or
            result["lr"] <= 0 or result["wd"] < 0 or result["clip"] <= 0 or
            not 0 <= result["dropout"] < 1):
        raise ValueError("Invalid readout settings")
    if (len(result["layers"]) != 4 or min(result["layers"]) < 1 or
            tuple(sorted(set(result["layers"]))) != result["layers"]):
        raise ValueError("Exactly four increasing fixed fusion layers required")
    return result


def load_cache(study, split):
    """Load one registered frozen cache; fitting never requests the test split."""
    study = Path(study).resolve()
    check_stop(study)
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown cache split")
    if split == "test":
        verify_selection_lock(study)
    manifest_path = study / "caches" / f"{split}.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ValueError("Cache is incomplete")
    if manifest.get("adapter_sha256") is not None or manifest.get("representation", "base") != "base":
        raise ValueError("Frozen base cache must contain no adapter")
    protocol = read_json(study / "protocol.json")
    if (manifest.get("split") != split or
            manifest.get("protocol_sha256") != sha256(study / "protocol.json")):
        raise ValueError("Cache split/protocol identity differs from fresh study")
    for key, digest_key in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
        path = Path(manifest[key])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Cache {key} must be an existing absolute path")
        if sha256(path) != manifest[digest_key]:
            raise ValueError(f"Cache artifact changed: {key}")
    array = np.load(manifest["path"], mmap_mode="r", allow_pickle=False)
    shape = tuple(manifest["shape"])
    if (array.shape != shape or len(shape) != 3 or shape[1] != 32 or
            array.dtype != np.float16):
        raise ValueError("Expected FP16 [rows,layers,hidden] NPY cache")
    expected_shape = (protocol.get("rows", {}).get(split, shape[0]),
                      protocol.get("num_layers", shape[1]), protocol.get("hidden_size", shape[2]))
    if shape != expected_shape:
        raise ValueError("Cache dimensions differ from registered rows/model")
    if "data_dir" in protocol and "inputs" in protocol:
        input_path = str(Path(protocol["data_dir"]) / f"{split}.parquet")
        if (manifest.get("input_sha256") != protocol["inputs"].get(input_path) or
                sha256(input_path) != protocol["inputs"].get(input_path)):
            raise ValueError("Cache input differs from registered split data")
    metadata = pd.read_parquet(manifest["metadata_path"])
    required = {"base_id", "group_id", "source", "label", "prompt_tokens"}
    if not required.issubset(metadata) or len(metadata) != shape[0] or not len(metadata):
        raise ValueError("Missing or misaligned cache metadata")
    if metadata[list(required)].isna().any().any() or metadata.base_id.astype(str).duplicated().any():
        raise ValueError("Cache metadata contains missing or duplicated identities")
    if not metadata.label.isin([0, 1]).all() or not (metadata.prompt_tokens > 0).all():
        raise ValueError("Binary labels and positive full-input token lengths required")
    ordered = ordered_base_id_sha256(metadata.base_id.astype(str).tolist())
    if manifest.get("ordered_base_id_sha256") != ordered:
        raise ValueError("Cache row identity/order changed")
    validate_feature_cache(Path(manifest["path"]), shape, np.float16, ordered,
                           metadata.base_id.astype(str).tolist())
    return array, metadata, manifest


def assert_disjoint(left, right):
    for key in ("base_id", "group_id"):
        if set(left[key].astype(str)) & set(right[key].astype(str)):
            raise ValueError(f"Split leakage: {key}")


def _batch(cache, rows, device, layers=None):
    if layers is None:
        array = np.array(cache[rows], dtype=np.float32, copy=True)
    else:
        array = np.stack([np.asarray(cache[rows, k - 1], dtype=np.float32) for k in layers], axis=1)
    return torch.from_numpy(array).to(device)


class VectorizedLayerMLPs(nn.Module):
    """Independent repaired last-token MLPs; leading parameter axis is layer."""
    def __init__(self, hidden_size, width, layer_count, seed=42, dropout=.1):
        super().__init__()
        torch.manual_seed(seed)
        reference = ExperimentHead(hidden_size, mode="last", width=width, layers=(1,), dropout=dropout)
        self.layer_count, self.width, self.dropout = layer_count, width, dropout
        for name, tensor in reference.state_dict().items():
            self.register_parameter(name.replace(".", "_"), nn.Parameter(tensor.unsqueeze(0).repeat(
                layer_count, *([1] * tensor.ndim))))

    def forward(self, raw):
        if raw.ndim != 3 or raw.shape[1] != self.layer_count:
            raise ValueError("Aligned [batch,layers,hidden] features required")
        normalized = F.layer_norm(raw.float(), (raw.shape[-1],))
        projected = torch.einsum("bld,lwd->blw", normalized, self.project_weight) + self.project_bias
        z = F.gelu(projected)
        if self.training and self.dropout:
            # Every independent head receives the same dropout draw as a
            # separately seeded [batch,width] repaired MLP.
            mask = F.dropout(torch.ones((len(raw), self.width), device=z.device),
                             p=self.dropout, training=True)
            z = z * mask[:, None]
        return torch.einsum("blw,low->blo", z, self.classifier_weight).squeeze(-1) + self.classifier_bias[:, 0]

    def layer_state(self, index, state=None):
        values = self.state_dict() if state is None else state
        return {key.replace("_", ".", 1): value[index].detach().cpu().clone()
                for key, value in values.items()}

    def clip_independent_gradients(self, maximum):
        norms = sum(p.grad.detach().square().flatten(1).sum(1) for p in self.parameters()
                    if p.grad is not None).sqrt()
        if not torch.isfinite(norms).all():
            raise FloatingPointError("Nonfinite per-layer gradients")
        scale = (maximum / (norms + 1e-6)).clamp(max=1)
        for p in self.parameters():
            if p.grad is not None:
                p.grad.mul_(scale.reshape(-1, *([1] * (p.ndim - 1))))
        return norms


def choose_best_layer(validation_bce):
    values = np.asarray(validation_bce, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite per-layer validation BCE required")
    # Lowest one-based index breaks exact ties; no test predictions are accepted.
    return int(np.argmin(values)) + 1


def build_fusion_heads(hidden, s, best, device):
    specs = dict(repeat_best=("repeat_last", (best,)),
                 fusion_fixed=("fusion", s["layers"]),
                 fusion_fixed_best_anchor=("fusion_best_anchor", s["layers"] + (best,)))
    heads = {}
    for arm, (mode, _) in specs.items():
        torch.manual_seed(s["seed"])
        head = ExperimentHead(hidden, mode="repeat_last" if mode == "repeat_last" else "fusion",
                              width=s["width"], layers=s["layers"], dropout=s["dropout"])
        heads[arm] = head.to(device)
    matched = assert_capacity_matched_initialization(heads)
    return heads, specs, matched


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
    model = VectorizedLayerMLPs(train.shape[-1], s["width"], train.shape[1], s["seed"], s["dropout"]).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["wd"])
    best_bce = np.full(train.shape[1], np.inf)
    best_epoch = np.zeros(train.shape[1], dtype=int)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history, audit, global_step = [], [], 0
    for epoch in range(budget):
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
        dump(folder / "layer_history.json", history)
        print(json.dumps(dict(stage="all_layer_mlps", epoch=epoch + 1, budget=budget)), flush=True)
    model.load_state_dict(best_state)
    validation_logits = _predict_layers(model, validation, s["batch_size"], device, study)
    dump(folder / "layer_gradient_audit.json", audit)
    atomic_torch_save(dict(state=best_state, best_epoch=best_epoch.tolist(), best_bce=best_bce.tolist(),
                           initial_state_sha256=state_digest(initial)), folder / "all_layer_mlps.pt")
    return model, validation_logits, best_bce, best_epoch, history


def _train_fusion(study, folder, train, validation, y_train, y_val, s, budget, best, device):
    heads, specs, matched = build_fusion_heads(train.shape[-1], s, best, device)
    initial = {a: {n: p.detach().cpu().clone() for n, p in h.named_parameters()} for a, h in heads.items()}
    optimizers = {a: torch.optim.AdamW(h.parameters(), lr=s["lr"], weight_decay=s["wd"]) for a, h in heads.items()}
    best_bce, best_epoch = {a: np.inf for a in heads}, {a: 0 for a in heads}
    best_states, history, audit, global_step = {}, [], [], 0
    layers = sorted(set(s["layers"]) | {best})
    positions = {k: i for i, k in enumerate(layers)}
    for epoch in range(budget):
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
                best_states[arm] = {k: v.detach().cpu().clone() for k, v in heads[arm].state_dict().items()}
        history.append(dict(epoch=epoch + 1, train_bce={a: t / len(train) for a, t in total.items()}, dev_bce=bce))
        dump(folder / "fusion_history.json", history)
        print(json.dumps(dict(stage="fusion_heads", epoch=epoch + 1, budget=budget)), flush=True)
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
    dump(folder / "fusion_gradient_audit.json", audit)
    validation_logits = _predict_heads(heads, specs, validation, s["batch_size"], device, study)
    return heads, specs, matched, validation_logits, best_bce, best_epoch, history, best_states


def verify_selection_lock(study):
    study = Path(study).resolve()
    check_stop(study)
    path = study / "selection_lock.json"
    if not path.exists():
        raise RuntimeError("Test barrier: all checkpoint and selection hashes must be locked first")
    value = read_json(path)
    if value.get("complete") is not True or value.get("test_opened") is not False:
        raise ValueError("Invalid selection lock")
    for filename, digest in value["artifacts"].items():
        if sha256(study / filename) != digest:
            raise ValueError(f"Selection/checkpoint changed after lock: {filename}")
    if sha256(study / "protocol.json") != value["protocol_sha256"]:
        raise ValueError("Study protocol changed after selection")
    return value


def fit(study, device="cuda"):
    study = Path(study).resolve()
    check_stop(study)
    folder = study / "heads"
    if (study / "selection_lock.json").exists():
        return verify_selection_lock(study)
    if (folder / "fit_started.json").exists():
        raise RuntimeError("Incomplete fit attempt cannot automatically retry")
    protocol = read_json(study / "protocol.json")
    protocol_sha = sha256(study / "protocol.json")
    s = settings(protocol)
    train, train_meta, _ = load_cache(study, "train")
    validation, val_meta, _ = load_cache(study, "validation")
    if train.shape[1:] != validation.shape[1:] or max(s["layers"]) > train.shape[1]:
        raise ValueError("Train/validation dimensions or fixed layers differ")
    assert_disjoint(train_meta, val_meta)
    bindings = {str(study / "caches" / f"{split}.manifest.json"): sha256(study / "caches" / f"{split}.manifest.json")
                for split in ("train", "validation")}
    source_paths = (Path(__file__), ROOT / "scripts/hanguard/repaired_heads.py",
                    ROOT / "scripts/hanguard/ab_frozen_readout_diagnostics.py",
                    ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py")
    source_sha = {str(path): sha256(path) for path in source_paths}
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / "fit_started.json", dict(protocol_sha256=protocol_sha, test_opened=False, source_sha256=source_sha))
    started, attempts, budget = time.monotonic(), [], s["epochs"]
    y_train, y_val = train_meta.label.to_numpy(dtype=int), val_meta.label.to_numpy(dtype=int)
    while True:
        attempt = folder / f"budget_{budget}"
        attempt.mkdir()
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
        # The validation-selected layer can change with budget. Restart every
        # arm with identical seeds/order so the final comparison has one budget.
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
                     extension_rule="12/24/40 validation BCE milestones; any arm may trigger common re-fit; all final arms have the same budget")
    selection_path = folder / "selection.json"
    dump(selection_path, selection)
    val_frame = val_meta.copy()
    for arm, logits in validation_logits.items():
        val_frame[f"probability_{arm}"] = sigmoid(logits)
    val_frame.to_parquet(folder / "validation_predictions.parquet", index=False)
    for filename, digest in {**bindings, **source_sha}.items():
        if sha256(filename) != digest:
            raise ValueError(f"Input/source changed during fit: {filename}")
    if sha256(study / "protocol.json") != protocol_sha:
        raise ValueError("Protocol changed during fit")
    final_files = [checkpoint, selection_path, folder / "validation_predictions.parquet",
                   folder / "attempts.json", attempt / "all_layer_mlps.pt", attempt / "layer_history.json",
                   attempt / "fusion_history.json", attempt / "layer_gradient_audit.json", attempt / "fusion_gradient_audit.json"]
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


def evaluate(study, device="cuda"):
    study = Path(study).resolve()
    lock = verify_selection_lock(study)  # Must precede every access to the test cache.
    selection = read_json(study / "heads" / "selection.json")
    for filename, digest in {**selection["train_validation_manifest_sha256"], **selection["source_sha256"]}.items():
        if sha256(filename) != digest:
            raise ValueError(f"Locked input/source changed: {filename}")
    protocol = read_json(study / "protocol.json")
    s = settings(protocol)
    cache, metadata, manifest = load_cache(study, "test")
    for split in ("train", "validation"):
        prior = read_json(study / "caches" / f"{split}.manifest.json")
        if sha256(prior["metadata_path"]) != prior["metadata_sha256"]:
            raise ValueError("Selected training/validation metadata changed")
        assert_disjoint(pd.read_parquet(prior["metadata_path"]), metadata)
    output = study / "analysis"
    if (output / "complete.json").exists():
        receipt = read_json(output / "complete.json")
        if receipt["selection_lock_sha256"] != sha256(study / "selection_lock.json"):
            raise ValueError("Evaluation belongs to another selection")
        for filename, digest in receipt["artifacts"].items():
            if sha256(output / filename) != digest:
                raise ValueError("Completed evaluation changed")
        return read_json(output / "results.json")
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "test_access.json", dict(selection_lock_sha256=sha256(study / "selection_lock.json"),
                                          cache_manifest_sha256=sha256(study / "caches" / "test.manifest.json"),
                                          test_opened=True, opened_unix=time.time()))
    checkpoint = torch.load(study / "heads" / "selected_heads.pt", map_location="cpu", weights_only=True)
    if cache.shape[2] != checkpoint["hidden_size"]:
        raise ValueError("Test hidden size differs from selected checkpoint")
    heads, specs = {}, checkpoint["specs"]
    for arm in ARMS:
        mode, ks = specs[arm]
        head = ExperimentHead(cache.shape[-1], mode="last" if mode == "last" else
                              "repeat_last" if mode == "repeat_last" else "fusion",
                              width=s["width"], layers=ks if mode == "last" else s["layers"], dropout=s["dropout"])
        head.load_state_dict(checkpoint["states"][arm], strict=True)
        heads[arm] = head.to(device)
    logits = _predict_heads(heads, specs, cache, s["batch_size"], device, study)
    frame = metadata.copy()
    for arm, values in logits.items():
        frame[f"probability_{arm}"] = sigmoid(values)
        frame[f"threshold_{arm}"] = selection["thresholds"][arm]
        frame[f"prediction_{arm}"] = (frame[f"probability_{arm}"] >= selection["thresholds"][arm]).astype(np.int8)
    y = frame.label.to_numpy(dtype=int)
    metrics = {a: binary_metrics(y, frame[f"probability_{a}"], selection["thresholds"][a]) for a in ARMS}
    fixed = {a: binary_metrics(y, frame[f"probability_{a}"], .5) for a in ARMS}
    sources = {str(source): {a: binary_metrics(part.label, part[f"probability_{a}"], selection["thresholds"][a])
                            for a in ARMS} for source, part in frame.groupby("source")}
    comparisons = []
    for candidate, baseline in (("best_single", "last"), ("repeat_best", "best_single"),
                                ("fusion_fixed", "last"), ("fusion_fixed", "best_single"),
                                ("fusion_fixed", "repeat_best"), ("fusion_fixed_best_anchor", "best_single"),
                                ("fusion_fixed_best_anchor", "repeat_best"), ("fusion_fixed_best_anchor", "fusion_fixed")):
        check_stop(study)
        a = frame[f"prediction_{candidate}"].to_numpy() == y
        b = frame[f"prediction_{baseline}"].to_numpy() == y
        comparisons.append(dict(candidate=candidate, baseline=baseline,
                                rescued=int((a & ~b).sum()), harmed=int((~a & b).sum()), net_correct=int(a.sum() - b.sum()),
                                group_bootstrap=group_cluster_bootstrap(y, frame[f"probability_{candidate}"],
                                    frame[f"probability_{baseline}"], frame.group_id,
                                    selection["thresholds"][candidate], selection["thresholds"][baseline],
                                    n_resamples=s["bootstrap_draws"], seed=s["seed"])))
    frame.to_parquet(output / "predictions.parquet", index=False)
    frame.to_csv(output / "predictions.csv", index=False)
    prediction_artifacts = {name: sha256(output / name)
                            for name in ("predictions.parquet", "predictions.csv", "test_access.json")}
    result = dict(schema=SCHEMA, complete=True, no_lora=True, rows=len(frame), groups=frame.group_id.nunique(),
                  metrics=metrics, fixed_05_metrics=fixed, by_source=sources, comparisons=comparisons,
                  protocol_sha256=lock["protocol_sha256"], model_path=protocol.get("model_path"),
                  data_version=protocol.get("dataset_release", {}).get("dataset_version"),
                  data_dir=protocol.get("data_dir"),
                  parameters={a: sum(p.numel() for p in h.parameters()) for a, h in heads.items()},
                  best_single_layer=selection["best_single_layer"],
                  final_common_epoch_budget=selection["final_common_epoch_budget"],
                  best_epochs=selection["best_epochs"], fit_seconds=selection["total_seconds"],
                  limitations=protocol.get("limitations", []), artifacts=prediction_artifacts,
                  selection_lock_sha256=sha256(study / "selection_lock.json"),
                  test_cache_manifest_sha256=sha256(study / "caches" / "test.manifest.json"),
                  scope="One frozen foundation and one head seed; held-out release test after validation-only selection")
    dump(output / "results.json", result)
    verify_selection_lock(study)
    check_stop(study)
    files = ("predictions.parquet", "predictions.csv", "results.json", "test_access.json")
    dump(output / "complete.json", dict(complete=True, selection_lock_sha256=result["selection_lock_sha256"],
                                       artifacts={name: sha256(output / name) for name in files}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    result = fit(args.study, args.device) if args.phase == "fit" else evaluate(args.study, args.device)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
