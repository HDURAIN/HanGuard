"""Registered frozen-base fusion ablations on full and fixed selected tests.

The old scientific implementation and artifacts are read-only. New head seeds
reuse verified foundation features, and six component controls use exactly the
old optimizer, batches, validation checkpoint rule and bounded epoch budgets.
The 122 excluded identities are fixed before these runs and are never reselected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
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
from scripts.hanguard import frozen_base_readout_heads as original
from scripts.hanguard import frozen_base_resumable_heads as resumable
from scripts.hanguard.ab_frozen_readout_diagnostics import budget_extension
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    accuracy_threshold, atomic_torch_save, binary_metrics, dump,
    group_cluster_bootstrap, sha256, sigmoid,
)
from scripts.hanguard.repaired_heads import ExperimentHead

DEFAULT_PARENT = ROOT / "outputs/hanguard_frozen_base_full_readouts_20261013_v5"
DEFAULT_SUBSET = ROOT / "data/three_source_wildguard122_subset_20261018"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_nolora_ablation_20261019"
SCHEMA = "hanguard_nolora_fusion_ablation_v1"
TASKS = ("seed43", "seed44", "components")
COMPONENTS = ("uniform_weights", "no_anchor", "shared_projection", "top4", "two_layers", "mean_pool")
META = ("base_id", "group_id", "source", "label", "prompt_tokens")


def read(path):
    return json.loads(Path(path).read_text())


def task_directory(output, task):
    if task not in TASKS:
        raise ValueError("Unknown ablation task")
    return Path(output) / (f"seed_{task[4:]}" if task.startswith("seed") else task)


def check_stop(output):
    for folder in (Path(output), Path(output).parent):
        if (folder / "STOP").exists():
            raise RuntimeError(f"Stopped: {folder / 'STOP'}")


def source_paths(extra=()):
    paths = [Path(__file__), Path(original.__file__), Path(resumable.__file__),
             ROOT / "scripts/hanguard/repaired_heads.py",
             ROOT / "scripts/hanguard/ab_frozen_readout_diagnostics.py",
             ROOT / "scripts/hanguard/frozen_lora_layer_diagnostics.py"]
    queue = ROOT / "scripts/hanguard/nolora_ablation_queue.py"
    if queue.exists():
        paths.append(queue)
        paths.extend(ROOT / "scripts/hanguard" / filename for filename in
                     ("fragment_clean_lora_study.py", "ab_readout_parallel_queue.py", "ab_diagnostic_queue.py"))
    paths.extend(Path(p).resolve() for p in extra)
    return sorted(set(p.resolve() for p in paths))


def component_specs(selection):
    losses = np.asarray(selection["layer_validation_bce"], dtype=float)
    if losses.shape != (32,) or not np.isfinite(losses).all():
        raise ValueError("All 32 parent validation BCE values are required")
    top4 = sorted((np.argsort(losses, kind="stable")[:4] + 1).tolist())
    fixed = [8, 16, 24, 32]
    result = {arm: dict(kind=arm, layers=top4 if arm == "top4" else
                      [16, 32] if arm == "two_layers" else fixed,
                      anchor=int(selection["best_single_layer"])) for arm in COMPONENTS}
    result["anchor_replay"] = dict(kind="full", layers=fixed, anchor=int(selection["best_single_layer"]))
    return result


def excluded_ids(output):
    protocol = read(Path(output) / "protocol.json")
    path = Path(protocol["excluded_ids_path"])
    if sha256(path) != protocol["bindings"][str(path)]:
        raise ValueError("Fixed exclusion identities changed")
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)) or len(values) != protocol["excluded_rows"]:
        raise ValueError("Fixed exclusion list is duplicated or has changed size")
    return set(values)


def initialize(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT, subset=DEFAULT_SUBSET, extra_sources=()):
    output, parent, subset = (Path(p).resolve() for p in (output, parent, subset))
    if (output / "protocol_registration.json").exists():
        return verify(output, full_hash=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A partially registered ablation directory requires review")
    original.verify_selection_lock(parent)
    parent_protocol = read(parent / "protocol.json")
    selection = read(parent / "heads/selection.json")
    if parent_protocol.get("use_lora", False) or not selection.get("no_lora"):
        raise ValueError("A frozen foundation without LoRA is required")
    policy = read(subset / "filter_policy.json")
    if (policy.get("selection_policy") != "posthoc_error_selected_test_subset" or
            policy.get("no_text_changes") is not True or policy.get("no_label_changes") is not True):
        raise ValueError("Expected the previously fixed outcome-selected test subset")
    ids_path = subset / "excluded_base_ids.txt"
    ids = [s.strip() for s in ids_path.read_text().splitlines() if s.strip()]
    if len(ids) != len(set(ids)) or len(ids) != int(policy["selected_rows"]):
        raise ValueError("Invalid fixed exclusion identity list")
    sources = {str(p): sha256(p) for p in source_paths(extra_sources)}
    bindings = dict(sources)
    parent_budget = int(selection["final_common_epoch_budget"])
    artifacts = [parent / name for name in ("protocol.json", "selection_lock.json", "heads/selection.json",
                  "heads/selected_heads.pt", f"heads/budget_{parent_budget}/all_layer_mlps.pt",
                  "heads/validation_predictions.parquet", "analysis/predictions.parquet", "analysis/results.json")]
    artifacts += [p for p in subset.iterdir() if p.is_file()]
    for filename, digest in parent_protocol.get("inputs", {}).items():
        if sha256(filename) != digest:
            raise ValueError(f"Registered parent data changed: {filename}")
        bindings[str(Path(filename).resolve())] = digest
    manifests = {}
    for split in ("train", "validation", "test"):
        path = parent / "caches" / f"{split}.manifest.json"
        manifest = read(path)
        if (manifest.get("complete") is not True or manifest.get("adapter_sha256") is not None or
                manifest.get("representation", "base") != "base" or
                manifest.get("protocol_sha256") != sha256(parent / "protocol.json")):
            raise ValueError("Incomplete or adapted parent cache")
        for key, digest_key in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
            filename = str(Path(manifest[key]).resolve())
            if sha256(filename) != manifest[digest_key]:
                raise ValueError(f"Parent frozen cache changed: {filename}")
            bindings[filename] = manifest[digest_key]
        artifacts.append(path)
        manifests[split] = manifest
    for path in artifacts:
        bindings[str(path)] = sha256(path)
    # Retained training and validation must still be byte-identical to the
    # foundation experiment; the selected test release changes no fit input.
    if parent_protocol.get("data_dir"):
        for split in ("train", "validation"):
            if sha256(subset / f"{split}.parquet") != sha256(Path(parent_protocol["data_dir"]) / f"{split}.parquet"):
                raise ValueError("Selected release changed a training/validation split")
    output.mkdir(parents=True, exist_ok=True)
    specs = component_specs(selection)
    for task in TASKS:
        child = task_directory(output, task)
        (child / "caches").mkdir(parents=True)
        clone = copy.deepcopy(parent_protocol)
        clone["settings"]["seed"] = int(task[4:]) if task.startswith("seed") else 42
        clone.update(ablation_parent=str(parent), ablation_task=task,
                     evaluation_scope="Original full test plus one previously fixed outcome-selected subset",
                     fresh_features_required=False, reused_frozen_features=True)
        if task == "components":
            clone["component_specs"] = specs
            clone["methods"] = list(specs)
            clone["parent_best_single_layer"] = int(selection["best_single_layer"])
        dump(child / "protocol.json", clone)
        for split, manifest in manifests.items():
            clone_manifest = copy.deepcopy(manifest)
            clone_manifest["protocol_sha256"] = sha256(child / "protocol.json")
            clone_manifest["reuse_parent_manifest"] = str(parent / "caches" / f"{split}.manifest.json")
            dump(child / "caches" / f"{split}.manifest.json", clone_manifest)
        for path in [child / "protocol.json", *sorted((child / "caches").glob("*.manifest.json"))]:
            bindings[str(path)] = sha256(path)
    stats = {name: dict(bytes=Path(name).stat().st_size, mtime_ns=Path(name).stat().st_mtime_ns)
             for name in bindings}
    protocol = dict(schema=SCHEMA, parent=str(parent), subset=str(subset), tasks=list(TASKS),
                    excluded_ids_path=str(ids_path), excluded_rows=len(ids),
                    subset_policy="posthoc_error_selected_test_subset",
                    selected_subset_is_independent_test=False,
                    full_rows=int(parent_protocol["rows"]["test"]),
                    subset_rows=int(parent_protocol["rows"]["test"]) - len(ids),
                    source_sha256=sources, bindings=bindings, input_stats=stats,
                    component_specs=specs, settings=parent_protocol["settings"],
                    selection_rule="validation BCE checkpoint; validation accuracy threshold; test never selects anything",
                    budget_rule="common 12/24/40 refits on the original validation BCE extension rule",
                    seed42_imported_without_retraining=True,
                    limitations=["The fixed 122 removals were selected from seed 42 test errors; their +2.10 pp is descriptive.",
                                 "Full test and fixed selected subset are reported together for every arm and seed."],
                    created_unix=time.time())
    dump(output / "protocol.json", protocol)
    dump(output / "protocol_registration.json", dict(protocol_sha256=sha256(output / "protocol.json")))
    return verify(output, full_hash=False)


def verify(output, full_hash=False):
    output = Path(output).resolve()
    check_stop(output)
    protocol = read(output / "protocol.json")
    if (protocol.get("schema") != SCHEMA or
            read(output / "protocol_registration.json").get("protocol_sha256") != sha256(output / "protocol.json")):
        raise ValueError("Ablation registration changed")
    for filename, digest in protocol["source_sha256"].items():
        if sha256(filename) != digest:
            raise ValueError(f"Registered ablation source changed: {filename}")
    for filename, digest in protocol["bindings"].items():
        stat = Path(filename).stat()
        if dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != protocol["input_stats"][filename]:
            raise ValueError(f"Registered ablation artifact metadata changed: {filename}")
        if full_hash and sha256(filename) != digest:
            raise ValueError(f"Registered ablation artifact content changed: {filename}")
    excluded_ids(output)
    return protocol


class ComponentHead(ExperimentHead):
    """A control of the original four-branch, best-single anchored head."""
    def __init__(self, hidden_size, width, dropout, spec):
        kind, layers = spec["kind"], tuple(spec["layers"])
        if kind not in (*COMPONENTS, "full"):
            raise ValueError("Unknown component control")
        super().__init__(hidden_size, mode="last" if kind == "mean_pool" else "fusion",
                         width=width, layers=layers, dropout=dropout)
        self.kind, self.anchor = kind, int(spec["anchor"])
        if kind == "uniform_weights":
            self.layer_logits.requires_grad_(False)
        elif kind == "no_anchor":
            del self.project
        elif kind == "shared_projection":
            shared = self.layer_values[0]
            self.layer_values = nn.ModuleList([shared for _ in layers])

    def forward(self, raw, available_layers=None):
        available = tuple(range(1, raw.shape[1] + 1)) if available_layers is None else tuple(available_layers)
        positions = {layer: index for index, layer in enumerate(available)}
        pieces = [raw[:, positions[layer]] for layer in self.layers]
        if self.kind == "mean_pool":
            z = F.gelu(self._project_raw(torch.stack(pieces, dim=1).mean(1), self.project))
        else:
            weights = self.layer_logits.softmax(0)
            values = sum(weights[i] * self._project_raw(piece, module)
                         for i, (piece, module) in enumerate(zip(pieces, self.layer_values)))
            z = self.fusion_output(values)
            if self.kind != "no_anchor":
                z = z + F.gelu(self._project_raw(raw[:, positions[self.anchor]], self.project))
        return self.classify_representation(z)


def build_components(hidden_size, settings, specs, device):
    heads = {}
    for arm, spec in specs.items():
        torch.manual_seed(settings["seed"])
        heads[arm] = ComponentHead(hidden_size, settings["width"], settings["dropout"], spec).to(device)
    return heads


def component_layers(specs):
    return sorted({layer for spec in specs.values() for layer in
                   (*spec["layers"],) + (() if spec["kind"] in ("no_anchor", "mean_pool") else (spec["anchor"],))})


@torch.no_grad()
def predict_components(heads, cache, batch_size, device, study, layers):
    result = {arm: [] for arm in heads}
    for head in heads.values():
        head.eval()
    for start in range(0, len(cache), batch_size):
        check_stop(study)
        raw = original._batch(cache, np.arange(start, min(start + batch_size, len(cache))), device, layers)
        for arm, head in heads.items():
            result[arm].append(head(raw, layers).cpu().numpy())
    result = {arm: np.concatenate(values) for arm, values in result.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("Nonfinite component predictions")
    return result


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def execution_info(device):
    is_cuda = torch.device(device).type == "cuda"
    return dict(training_device=str(device), cpu_threads=torch.get_num_threads(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                torch_cuda_version=torch.version.cuda,
                cuda_device_name=torch.cuda.get_device_name(0) if is_cuda else None,
                cuda_device_capability=list(torch.cuda.get_device_capability(0)) if is_cuda else None,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark)


def train_component_budget(child, folder, train, validation, y_train, y_val, s, specs, budget, device):
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_sha256=sha256(child / "protocol.json"), source_sha256=sha256(__file__),
                    settings=s, specs=specs, budget=budget,
                    manifests={split: sha256(child / "caches" / f"{split}.manifest.json")
                               for split in ("train", "validation")})
    # JSON normalization makes tuple-valued settings stable after persistence.
    identity = json.loads(json.dumps(identity))
    heads = build_components(train.shape[-1], s, specs, device)
    optimizers = {arm: torch.optim.AdamW((p for p in head.parameters() if p.requires_grad),
                                       lr=s["lr"], weight_decay=s["wd"]) for arm, head in heads.items()}
    layers = component_layers(specs)
    history, audit, completed = [], [], 0
    best_bce, best_epoch = {arm: np.inf for arm in heads}, {arm: 0 for arm in heads}
    best_states = {}
    checkpoint = folder / "resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("identity") != identity:
            raise ValueError("Component epoch checkpoint identity changed")
        completed, history, audit = saved["completed_epochs"], saved["history"], saved["audit"]
        if (not 1 <= completed <= budget or len(history) != completed or
                [row["epoch"] for row in history] != list(range(1, completed + 1))):
            raise ValueError("Invalid component epoch checkpoint")
        best_bce, best_epoch, best_states = saved["best_bce"], saved["best_epoch"], saved["best_states"]
        for arm in heads:
            heads[arm].load_state_dict(saved["current_states"][arm], strict=True)
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
    for epoch in range(completed, budget):
        check_stop(child)
        total = {arm: 0.0 for arm in heads}
        for step, rows in enumerate(original.epoch_batches(len(train), s["seed"], epoch, s["batch_size"])):
            check_stop(child)
            raw = original._batch(train, rows, device, layers)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            for arm, head in heads.items():
                head.train()
                torch.manual_seed(original._step_seed(s["seed"], epoch, step))
                logits = head(raw, layers)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite component loss: {arm}")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(head.parameters(), s["clip"], error_if_nonfinite=True)
                optimizers[arm].step()
                total[arm] += float(loss.detach()) * len(rows)
                if epoch == 0 and step in (0, 1, 9):
                    audit.append(dict(arm=arm, step=step + 1, grad_norm=float(norm),
                                      all_gradients_finite=True))
        validation_logits = predict_components(heads, validation, s["batch_size"], device, child, layers)
        bce = {arm: binary_metrics(y_val, sigmoid(logits), .5)["bce"]
               for arm, logits in validation_logits.items()}
        for arm in heads:
            if bce[arm] < best_bce[arm]:
                best_bce[arm], best_epoch[arm] = bce[arm], epoch + 1
                best_states[arm] = _cpu(heads[arm].state_dict())
        history.append(dict(epoch=epoch + 1, train_bce={a: v / len(train) for a, v in total.items()}, dev_bce=bce))
        atomic_torch_save(dict(identity=identity, completed_epochs=epoch + 1, history=history, audit=audit,
                               current_states={a: h.state_dict() for a, h in heads.items()},
                               optimizers={a: o.state_dict() for a, o in optimizers.items()},
                               best_bce=best_bce, best_epoch=best_epoch, best_states=best_states), checkpoint)
        dump(folder / "history.json", history)
        print(json.dumps(dict(stage="components", epoch=epoch + 1, budget=budget)), flush=True)
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm], strict=True)
    logits = predict_components(heads, validation, s["batch_size"], device, child, layers)
    dump(folder / "gradient_audit.json", audit)
    return heads, logits, best_bce, best_epoch, history, best_states


def fit_components(output, device="cuda"):
    output = Path(output).resolve()
    verify(output)
    child = task_directory(output, "components")
    if (child / "selection_lock.json").exists():
        return original.verify_selection_lock(child)
    p = read(child / "protocol.json")
    specs, s = p["component_specs"], original.settings(p)
    train, train_meta, _ = original.load_cache(child, "train")
    validation, val_meta, _ = original.load_cache(child, "validation")
    original.assert_disjoint(train_meta, val_meta)
    folder = child / "heads"
    folder.mkdir(exist_ok=True)
    marker = dict(schema=SCHEMA, task="components", protocol_sha256=sha256(child / "protocol.json"),
                  source_sha256=sha256(__file__), test_opened=False)
    if (folder / "fit_started.json").exists():
        if read(folder / "fit_started.json") != marker:
            raise ValueError("Component fit registration changed")
    else:
        dump(folder / "fit_started.json", marker)
    started, budget, attempts = time.monotonic(), s["epochs"], []
    while True:
        attempt = folder / f"budget_{budget}"
        begin = time.monotonic()
        heads, logits, losses, epochs, history, states = train_component_budget(
            child, attempt, train, validation, train_meta.label.to_numpy(dtype=int),
            val_meta.label.to_numpy(dtype=int), s, specs, budget, device)
        extension = budget_extension(history, epochs, budget, s["max_epochs"])
        attempts.append(dict(budget=budget, seconds=time.monotonic() - begin,
                             extension=extension, discarded_for_common_budget_extension=bool(extension)))
        dump(folder / "attempts.json", attempts)
        if extension is None:
            break
        budget = extension["new_budget"]
        del heads
    thresholds = {arm: accuracy_threshold(val_meta.label, sigmoid(logit)) for arm, logit in logits.items()}
    checkpoint = folder / "selected_heads.pt"
    atomic_torch_save(dict(states=states, specs=specs, hidden_size=train.shape[-1]), checkpoint)
    parameter_counts = {arm: dict(total=sum(v.numel() for v in head.parameters()),
                                 trainable=sum(v.numel() for v in head.parameters() if v.requires_grad))
                        for arm, head in heads.items()}
    selection = dict(schema=SCHEMA, task="components", complete=True, no_lora=True,
                     thresholds=thresholds, best_epochs=epochs, validation_bce=losses,
                     validation_metrics={arm: binary_metrics(val_meta.label, sigmoid(logit), thresholds[arm])
                                         for arm, logit in logits.items()},
                     specs=specs, parameters=parameter_counts, final_common_epoch_budget=budget,
                     total_seconds=time.monotonic() - started, selection_scope="validation only",
                     execution=execution_info(device),
                     checkpoint_sha256=sha256(checkpoint), test_opened=False,
                     shared_projection_definition="Shared Linear-GELU-LayerNorm module across four distinct inputs",
                     mean_pool_definition="Mean raw 8/16/24/32 features, then the single-layer LayerNorm and MLP")
    dump(folder / "selection.json", selection)
    frame = val_meta.loc[:, list(META)].copy()
    for arm, logit in logits.items():
        frame[f"probability_{arm}"] = sigmoid(logit)
    frame.to_parquet(folder / "validation_predictions.parquet", index=False)
    files = [checkpoint, folder / "selection.json", folder / "fit_started.json", folder / "attempts.json",
             folder / "validation_predictions.parquet", attempt / "resume.pt", attempt / "history.json",
             attempt / "gradient_audit.json"]
    if (child / "execution.json").exists():
        files.append(child / "execution.json")
    artifacts = {str(path.relative_to(child)): sha256(path) for path in files}
    protocol = verify(output)
    artifacts.update(protocol["source_sha256"])
    artifacts[str(output / "protocol.json")] = sha256(output / "protocol.json")
    artifacts[str(output / "protocol_registration.json")] = sha256(output / "protocol_registration.json")
    for split in ("train", "validation"):
        manifest_path = child / "caches" / f"{split}.manifest.json"
        manifest = read(manifest_path)
        artifacts[str(manifest_path)] = sha256(manifest_path)
        artifacts[manifest["path"]] = manifest["sha256"]
        artifacts[manifest["metadata_path"]] = manifest["metadata_sha256"]
    lock = dict(schema=SCHEMA, complete=True, test_opened=False,
                protocol_sha256=sha256(child / "protocol.json"), artifacts=artifacts)
    dump(child / "selection_lock.json", lock)
    return original.verify_selection_lock(child)


def selected_mask(frame, ids):
    values = frame.base_id.astype(str)
    if values.duplicated().any() or not set(ids).issubset(set(values)):
        raise ValueError("Evaluation rows do not cover the exact fixed exclusions once each")
    return ~values.isin(ids).to_numpy()


def scope_report(frame, arms, thresholds, draws, seed, comparison_baselines=("best_single",)):
    y = frame.label.to_numpy(dtype=int)
    metrics = {arm: binary_metrics(y, frame[f"probability_{arm}"], thresholds[arm]) for arm in arms}
    comparisons = []
    pairs = []
    for arm in arms:
        for baseline in comparison_baselines:
            if arm != baseline and baseline in arms and (arm, baseline) not in pairs:
                pairs.append((arm, baseline))
    for arm, baseline in pairs:
        a = frame[f"probability_{arm}"].to_numpy() >= thresholds[arm]
        b = frame[f"probability_{baseline}"].to_numpy() >= thresholds[baseline]
        good_a, good_b = a == y, b == y
        comparison = dict(candidate=arm, baseline=baseline,
                          rescued=int((good_a & ~good_b).sum()), harmed=int((~good_a & good_b).sum()),
                          net_correct=int(good_a.sum() - good_b.sum()),
                          delta_accuracy_pp=float(100 * (good_a.mean() - good_b.mean())),
                          group_bootstrap=group_cluster_bootstrap(
                              y, frame[f"probability_{arm}"], frame[f"probability_{baseline}"], frame.group_id,
                              thresholds[arm], thresholds[baseline], n_resamples=draws, seed=seed))
        comparisons.append(comparison)
    return dict(rows=len(frame), groups=int(frame.group_id.nunique()), metrics=metrics,
                fixed_05_metrics={arm: binary_metrics(y, frame[f"probability_{arm}"], .5) for arm in arms},
                comparisons=comparisons,
                by_source={str(source): {arm: binary_metrics(part.label, part[f"probability_{arm}"], thresholds[arm])
                                        for arm in arms} for source, part in frame.groupby("source")})


def report_predictions(output, frame, arms, thresholds, seed, components=False):
    protocol = verify(output)
    mask = selected_mask(frame, excluded_ids(output))
    if len(frame) != protocol["full_rows"] or int(mask.sum()) != protocol["subset_rows"]:
        raise ValueError("Full or fixed-subset evaluation size changed")
    baselines = ("best_single", "fusion_fixed_best_anchor", "anchor_replay") if components else ("best_single",)
    draws = int(protocol["settings"].get("bootstrap_draws", 2000))
    return dict(full_test=scope_report(frame, arms, thresholds, draws, seed, baselines),
                selected_subset=scope_report(frame.loc[mask], arms, thresholds, draws, seed, baselines),
                selected_subset_policy=protocol["subset_policy"],
                selected_subset_is_independent_test=False,
                excluded_ids_sha256=protocol["bindings"][protocol["excluded_ids_path"]])


@contextmanager
def augmented_seed_results(output, child, seed):
    """Add both scopes before the original evaluator seals its result hash."""
    previous = original.dump

    def write(path, value):
        if Path(path) == child / "analysis/results.json":
            frame = pd.read_parquet(child / "analysis/predictions.parquet")
            thresholds = read(child / "heads/selection.json")["thresholds"]
            value = dict(value, **report_predictions(output, frame, original.ARMS, thresholds, seed),
                         ablation_schema=SCHEMA, task=f"seed{seed}", seed=seed,
                         execution=read(child / "execution.json"))
        return previous(path, value)

    original.dump = write
    try:
        yield
    finally:
        original.dump = previous


def evaluate_components(output, device="cuda"):
    output = Path(output).resolve()
    p = verify(output)
    child = task_directory(output, "components")
    original.verify_selection_lock(child)
    folder = child / "analysis"
    if (folder / "complete.json").exists():
        receipt = read(folder / "complete.json")
        for name, digest in receipt["artifacts"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Completed component evaluation changed")
        if receipt["selection_lock_sha256"] != sha256(child / "selection_lock.json"):
            raise ValueError("Completed component evaluation belongs to a different selection")
        return read(folder / "results.json")
    # No test cache or test predictions are interpreted before this lock check.
    cache, metadata, _ = original.load_cache(child, "test")
    for split in ("train", "validation"):
        prior = read(child / "caches" / f"{split}.manifest.json")
        original.assert_disjoint(pd.read_parquet(prior["metadata_path"]), metadata)
    folder.mkdir(exist_ok=True)
    dump(folder / "test_access.json", dict(test_opened=True, opened_unix=time.time(),
                                           selection_lock_sha256=sha256(child / "selection_lock.json")))
    selection = read(child / "heads/selection.json")
    checkpoint = torch.load(child / "heads/selected_heads.pt", map_location="cpu", weights_only=True)
    specs, s = checkpoint["specs"], original.settings(read(child / "protocol.json"))
    if cache.shape[-1] != checkpoint["hidden_size"]:
        raise ValueError("Component test hidden size changed")
    heads = build_components(cache.shape[-1], s, specs, device)
    for arm in heads:
        heads[arm].load_state_dict(checkpoint["states"][arm], strict=True)
    logits = predict_components(heads, cache, s["batch_size"], device, child, component_layers(specs))
    frame = metadata.loc[:, list(META)].copy()
    parent = Path(p["parent"])
    baseline = pd.read_parquet(parent / "analysis/predictions.parquet")
    if (baseline.base_id.astype(str).tolist() != frame.base_id.astype(str).tolist() or
            not np.array_equal(baseline.label.to_numpy(dtype=int), frame.label.to_numpy(dtype=int))):
        raise ValueError("Parent baseline and component test rows differ")
    thresholds = dict(read(parent / "heads/selection.json")["thresholds"], **selection["thresholds"])
    for arm in original.ARMS:
        frame[f"probability_{arm}"] = baseline[f"probability_{arm}"].to_numpy()
    for arm, logit in logits.items():
        frame[f"probability_{arm}"] = sigmoid(logit)
    arms = (*original.ARMS, *specs)
    for arm in arms:
        frame[f"threshold_{arm}"] = thresholds[arm]
        frame[f"prediction_{arm}"] = (frame[f"probability_{arm}"] >= thresholds[arm]).astype(np.int8)
    frame.to_parquet(folder / "predictions.parquet", index=False)
    frame.to_csv(folder / "predictions.csv", index=False)
    result = dict(schema=SCHEMA, task="components", seed=42, complete=True,
                  parameters=selection["parameters"], specs=specs,
                  imported_reference_arms=list(original.ARMS), trained_arms=list(specs),
                  ablation_arms=list(COMPONENTS), common_budget_reference_arm="anchor_replay",
                  execution=selection["execution"],
                  final_common_epoch_budget=selection["final_common_epoch_budget"],
                  best_epochs=selection["best_epochs"], fit_seconds=selection["total_seconds"],
                  **report_predictions(output, frame, arms, thresholds, 42, components=True))
    dump(folder / "results.json", result)
    original.verify_selection_lock(child)
    dump(folder / "complete.json", dict(complete=True, selection_lock_sha256=sha256(child / "selection_lock.json"),
                                         artifacts={name: sha256(folder / name) for name in
                                                    ("results.json", "predictions.parquet", "predictions.csv", "test_access.json")}))
    return result


def run_worker(output, task, device="cuda", cpu_threads=8):
    output = Path(output).resolve()
    verify(output, full_hash=True)
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    child = task_directory(output, task)
    info_path = child / "execution.json"
    info = execution_info(device)
    if info_path.exists():
        if read(info_path) != info:
            raise ValueError("A resumed task must retain its execution device, threads, and numerical settings")
    else:
        dump(info_path, info)
    if task == "components":
        fit_components(output, device)
        result = evaluate_components(output, device)
    else:
        seed = int(task[4:])
        resumable.fit(child, device=device)
        with augmented_seed_results(output, child, seed):
            original.evaluate(child, device=device)
        result = read(child / "analysis/results.json")
    verify(output)
    return result


def imported_seed42(output):
    p = verify(output)
    folder = Path(output) / "seed_42_imported/analysis"
    if (folder / "complete.json").exists():
        receipt = read(folder / "complete.json")
        for name, digest in receipt["artifacts"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Imported baseline report changed")
        return read(folder / "results.json")
    parent = Path(p["parent"])
    original.verify_selection_lock(parent)
    frame = pd.read_parquet(parent / "analysis/predictions.parquet")
    columns = list(META) + [f"{prefix}_{arm}" for arm in original.ARMS
                           for prefix in ("probability", "threshold", "prediction")]
    frame = frame.loc[:, columns].copy()
    thresholds = read(parent / "heads/selection.json")["thresholds"]
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "predictions.parquet", index=False)
    frame.to_csv(folder / "predictions.csv", index=False)
    parent_result = read(parent / "analysis/results.json")
    result = dict(schema=SCHEMA, complete=True, task="seed42_imported", seed=42,
                  imported=True, parent=str(parent), parameters=parent_result.get("parameters"),
                  parent_selection_lock_sha256=sha256(parent / "selection_lock.json"),
                  **report_predictions(output, frame, original.ARMS, thresholds, 42))
    dump(folder / "results.json", result)
    dump(folder / "complete.json", dict(complete=True,
        artifacts={name: sha256(folder / name) for name in ("results.json", "predictions.parquet", "predictions.csv")}))
    return result


def summarize(output):
    output = Path(output).resolve()
    p = verify(output)
    tasks = {"seed42_imported": imported_seed42(output)}
    for task in TASKS:
        child = task_directory(output, task)
        if (child / "analysis/complete.json").exists():
            receipt = read(child / "analysis/complete.json")
            if receipt.get("selection_lock_sha256") != sha256(child / "selection_lock.json"):
                raise ValueError("Task report selection identity changed")
            for filename, digest in receipt["artifacts"].items():
                if sha256(child / "analysis" / filename) != digest:
                    raise ValueError("Task report artifact changed")
            tasks[task] = read(child / "analysis/results.json")
    rows = []
    for task, result in tasks.items():
        for scope in ("full_test", "selected_subset"):
            for arm, metrics in result[scope]["metrics"].items():
                rows.append(dict(task=task, seed=result["seed"], scope=scope, arm=arm,
                                 rows=metrics["n"], accuracy=metrics["accuracy"],
                                 accuracy_pct=100 * metrics["accuracy"], f1=metrics["f1"],
                                 auroc=metrics.get("auroc"), bce=metrics["bce"]))
    table = pd.DataFrame(rows)
    table.to_csv(output / "summary.csv", index=False)
    seed_tasks = [key for key in ("seed42_imported", "seed43", "seed44") if key in tasks]
    stats = {}
    for scope in ("full_test", "selected_subset"):
        stats[scope] = {}
        for arm in original.ARMS:
            values = [100 * tasks[key][scope]["metrics"][arm]["accuracy"] for key in seed_tasks]
            stats[scope][arm] = dict(seeds=[tasks[key]["seed"] for key in seed_tasks], accuracy_pct=values,
                                    mean_accuracy_pct=float(np.mean(values)),
                                    sample_std_accuracy_pp=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        gains = [100 * (tasks[key][scope]["metrics"]["fusion_fixed_best_anchor"]["accuracy"] -
                        tasks[key][scope]["metrics"]["best_single"]["accuracy"]) for key in seed_tasks]
        stats[scope]["anchor_gain_vs_best_single"] = dict(
            seeds=[tasks[key]["seed"] for key in seed_tasks], delta_accuracy_pp=gains,
            mean_delta_accuracy_pp=float(np.mean(gains)),
            sample_std_delta_accuracy_pp=float(np.std(gains, ddof=1)) if len(gains) > 1 else None)
    result = dict(schema=SCHEMA, complete=all(task in tasks for task in TASKS), tasks=tasks,
                  seed_statistics=stats, pending_tasks=[task for task in TASKS if task not in tasks],
                  subset_policy=p["subset_policy"], selected_subset_is_independent_test=False,
                  limitations=p["limitations"], protocol_sha256=sha256(output / "protocol.json"),
                  summary_csv_sha256=sha256(output / "summary.csv"), updated_unix=time.time())
    dump(output / "results.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    init.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    init.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    init.add_argument("--source", action="append", type=Path, default=[])
    worker = sub.add_parser("worker")
    worker.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    worker.add_argument("--task", choices=TASKS, required=True)
    worker.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    worker.add_argument("--cpu-threads", type=int, default=8)
    report = sub.add_parser("summarize")
    report.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "init":
        result = initialize(args.output, args.parent, args.subset, args.source)
    elif args.command == "worker":
        if args.cpu_threads < 1:
            raise ValueError("cpu-threads must be positive")
        result = run_worker(args.output, args.task, args.device, args.cpu_threads)
    else:
        result = summarize(args.output)
    print(json.dumps(dict(command=args.command, output=str(args.output.resolve()),
                          complete=result.get("complete"), pending_tasks=result.get("pending_tasks")), ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
