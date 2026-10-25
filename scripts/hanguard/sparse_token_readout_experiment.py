"""Exploratory, group-held-out token selection with one frozen A-only adapter.

No LoRA fitting, release validation/test access, new layer selection, or
post-hoc threshold tuning. Existing inner-dev layer choices are frozen inputs.
The already-explored B cohort is not a new blind test.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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
from scripts.hanguard.ab_frozen_readout_diagnostics import (
    budget_extension, indices_from_assignments, parameter_audit,
    _metrics_with_row_thresholds,
)
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    accuracy_threshold, atomic_torch_save, binary_metrics, dump,
    group_cluster_bootstrap, ordered_base_id_sha256, sha256, sigmoid, state_digest,
)
from scripts.hanguard.repaired_heads import ExperimentHead
from scripts.hanguard.sparse_token_readout_heads import ARMS, build_matched_heads, parameter_count

ALL_ARMS = ("last_mlp",) + ARMS
FOLD_ARTIFACTS = ("selection.json", "heads.pt", "history.json", "gradient_update_audit.json",
                  "held_predictions.parquet", "held_attention.parquet")
PAIRS = (
    ("global_query_entmax15", "global_query_softmax"),
    ("global_query_softmax", "static_query_softmax"),
    ("global_query_entmax15", "gated_attention_softmax"),
    ("global_query_entmax15", "repeat_last"),
    ("global_query_entmax15", "last_mlp"),
    ("gated_attention_softmax", "repeat_last"),
    ("gated_attention_softmax", "last_mlp"),
)


def read(path):
    return json.loads(Path(path).read_text())


def code_identity():
    names = ("sparse_token_readout_experiment.py", "sparse_token_readout_heads.py",
             "sparse_token_feature_cache.py", "ab_frozen_readout_diagnostics.py",
             "frozen_lora_layer_diagnostics.py", "repaired_heads.py")
    return {name: sha256(ROOT / "scripts/hanguard" / name) for name in names}


def runtime_identity():
    return dict(python=str(Path(sys.executable).absolute()), prefix=str(Path(sys.prefix).absolute()),
        torch=torch.__version__, cuda=torch.version.cuda,
        libraries={name: importlib.metadata.version(name) for name in
                   ("transformers", "peft", "triton", "causal-conv1d", "flash-linear-attention", "tokenizers")})


def initialize(args):
    study, output = Path(args.source_study).resolve(), Path(args.output).resolve()
    partition_path = study / "partition/manifest.json"
    partition = read(partition_path)
    if partition["invariants"]["final_test_enabled"] is not False:
        raise ValueError("Formal holdout must remain closed")
    training_path = study / "lora_study/protocol.json"
    training = read(training_path)
    lineage_path = study / "lora_study/runs/E03_s42/diagnostic_lineage.json"
    lineage = read(lineage_path)
    adapter_path = lineage_path.parent / "adapter.pt"
    adapter_sha = sha256(adapter_path)
    if (lineage["adapter_training_cohort"] != "A" or
            lineage["adapter_sha256"] != adapter_sha or
            lineage["protocol_sha256"] != sha256(training_path) or
            lineage["partition_manifest_sha256"] != sha256(partition_path) or
            lineage["B_sha256"] != partition["artifact_sha256"]["B.parquet"] or
            lineage.get("formal_test_opened") is not False):
        raise ValueError("A-only adapter lineage differs")
    b_path = study / "partition/B.parquet"
    if sha256(b_path) != partition["artifact_sha256"]["B.parquet"]:
        raise ValueError("B changed")
    b = pd.read_parquet(b_path)
    if len(b) != 20048 or b.base_id.duplicated().any():
        raise ValueError("Expected unique, unchanged B")
    probe_root = study / "readouts/adapted_last"
    probe_protocol_sha = sha256(probe_root / "protocol.json")
    cache_registration = output / "token_cache/registration.json"
    reference_metadata = study / "lora_study/features/adapted/B.metadata.parquet"
    inputs = {str(p): sha256(p) for p in (partition_path, training_path, lineage_path, b_path,
              probe_root / "protocol.json", cache_registration, reference_metadata)}
    layers = {}
    for fold in range(5):
        selection_path = probe_root / f"fold_{fold}/probes/selection.json"
        complete_path = selection_path.with_name("complete.json")
        selection, complete = read(selection_path), read(complete_path)
        if (complete.get("complete") is not True or
                complete["artifacts"]["selection.json"] != sha256(selection_path) or
                selection["provenance"]["protocol_sha256"] != probe_protocol_sha or
                selection["provenance"]["outer_fold"] != fold):
            raise ValueError("Frozen inner-dev layer selection changed")
        layers[str(fold)] = int(selection["best_single_layer"])
        assignment = study / f"partition/folds/fold_{fold}_assignments.parquet"
        if sha256(assignment) != partition["artifact_sha256"][f"folds/fold_{fold}_assignments.parquet"]:
            raise ValueError("Inner/outer groups changed")
        for path in (selection_path, complete_path, assignment):
            inputs[str(path)] = sha256(path)
    legacy_audit = getattr(args, "legacy_audit", None)
    legacy_audit_identity = None
    if legacy_audit is not None:
        legacy_audit = Path(legacy_audit).resolve()
        if read(legacy_audit).get("passed") is not False:
            raise ValueError("The preserved legacy failure record differs")
        inputs[str(legacy_audit)] = sha256(legacy_audit)
        legacy_audit_identity = dict(path=str(legacy_audit), sha256=sha256(legacy_audit))
    settings = dict(seed=42, width=128, dropout=.1, logical_batch_size=256,
                    micro_token_budget=8192, micro_max_rows=64, lr=.001,
                    weight_decay=.001, grad_clip=5., epochs=12, max_epochs=40,
                    bootstrap_draws=2000, alpha=1.5, projection_chunk_size=8192)
    protocol = dict(schema="hanguard_sparse_token_readout_v3", source_study=str(study),
        adapter_sha256=adapter_sha, adapter_path=str(adapter_path),
        training_protocol_sha256=sha256(training_path),
        partition_manifest_path=str(partition_path), partition_manifest_sha256=sha256(partition_path),
        B_path=str(b_path), B_sha256=sha256(b_path), rows=20048,
        token_cache=str(output / "token_cache"), fold_layers=layers,
        cache_layers=sorted(set(layers.values())), arms=list(ALL_ARMS), settings=settings,
        input_sha256=inputs, code_sha256=code_identity(),
        runtime=runtime_identity(),
        estimand="Within one newly extracted common token cache, paired readout gains at previously selected, prespecified layers",
        cache_validity="Frozen inputs/runtime/GPU; same-original-group sampled full-token repeated-forward equality, same-forward capture equality, full cache finite/journal/SHA and sampled exact fresh replay",
        legacy_equivalence="Strict comparison failed for the longest predefined old-cache sample; failure preserved. Legacy scores are descriptive context and never a new baseline.",
        legacy_numeric_equivalence_passed=False, preserved_legacy_failure=legacy_audit_identity,
        primary_comparisons=[list(pair) for pair in PAIRS[:5]],
        interval_scope="Prespecified exploratory comparisons; individual 95% intervals, no multiplicity adjustment",
        global_query_context="Projected, normalized last valid token of the chosen causal layer; prefix-inclusive context",
        interpretation_rule="Judge entmax against fresh last MLP, matched repeat and existing gated attention; also inspect dynamic-softmax/static contrasts, fixed-0.5 sensitivity, recall/FPR and length slices. A point gain alone does not establish the mechanism or warrant a seed sweep.",
        checkpoint_selection="minimum inner-dev BCE independently per arm",
        threshold_selection="maximum inner-dev accuracy; fixed 0.5 is secondary only",
        extension_rule="common 12->24->40 only if original inner-dev improvement rule triggers",
        formal_holdout_accessed=False,
        scope="Exploratory previously-exposed train-internal B group OOF, one fixed A adapter and head seed",
        effective_capacity_caveat="Five local heads share counts/init; repeat query and static conditional route are intentionally inactive",
        no_span_truth="Attention support is a readout choice, not gold harmful evidence",
        stop_directories=[str(output), str(study), str(training_path.parent)])
    path = output / "protocol.json"
    if path.exists():
        if read(path) != protocol:
            raise ValueError("Protocol changed; use a fresh independent directory")
    else:
        dump(path, protocol)
    return protocol


def load_protocol(output):
    protocol = read(output / "protocol.json")
    if protocol["code_sha256"] != code_identity():
        raise ValueError("Source drift after registration")
    if protocol.get("runtime") != runtime_identity():
        raise ValueError("Scientific runtime differs from registration")
    for path, digest in protocol["input_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Input drift: {path}")
    if sha256(Path(protocol["adapter_path"])) != protocol["adapter_sha256"]:
        raise ValueError("Frozen adapter changed")
    check_stop(protocol)
    return protocol


def check_stop(protocol):
    for directory in protocol["stop_directories"]:
        marker = Path(directory) / "STOP"
        if marker.exists():
            raise RuntimeError(f"Stopped: {marker.read_text()}")


def split_microbatches(rows, lengths, budget, max_rows):
    rows = np.asarray(rows, dtype=np.int64)
    current, maximum = [], 0
    for row in rows:
        length = int(lengths[row])
        if length < 1 or length > budget:
            raise ValueError("Cannot truncate or admit empty/oversize input")
        if current and (len(current) >= max_rows or (len(current) + 1) * max(maximum, length) > budget):
            yield np.asarray(current, dtype=np.int64)
            current, maximum = [], 0
        current.append(int(row))
        maximum = max(maximum, length)
    if current:
        yield np.asarray(current, dtype=np.int64)


class TokenStore:
    def __init__(self, path, protocol):
        from scripts.hanguard.sparse_token_feature_cache import validate
        validate(Path(path))
        self.manifest = read(Path(path) / "manifest.json")
        expected = dict(adapter_sha256=protocol["adapter_sha256"], B_sha256=protocol["B_sha256"],
            training_protocol_sha256=protocol["training_protocol_sha256"],
            ab_protocol_sha256=protocol["partition_manifest_sha256"],
            layers=protocol["cache_layers"], fold_best_layers=protocol["fold_layers"],
            hidden_size=2560, rows=protocol["rows"], dtype="float16")
        if any(self.manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Token cache layer/adapter/protocol identity differs")
        self.manifest_sha256 = sha256(Path(path) / "manifest.json")
        self.path = Path(path)
        self.metadata = pd.read_parquet(self.path / "B.metadata.parquet")
        self.offsets = np.load(self.path / "offsets.npy", allow_pickle=False)
        self.lengths = np.diff(self.offsets)
        self.layers = list(self.manifest["layers"])
        self.hidden = self.manifest["hidden_size"]
        if self.manifest["shape"] != [int(self.offsets[-1]), len(self.layers), self.hidden]:
            raise ValueError("Token cache shape differs from registered axis semantics")
        self.raw = np.memmap(self.path / "tokens.bin", dtype=np.float16, mode="r",
                            shape=(int(self.offsets[-1]), len(self.layers), self.hidden))
        reference = pd.read_parquet(Path(protocol["source_study"]) /
                                   "lora_study/features/adapted/B.metadata.parquet")
        for column in ("base_id", "group_id", "source", "label", "outer_fold", "prompt_tokens"):
            if not np.array_equal(self.metadata[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"Token cache row identity differs: {column}")
        if not np.array_equal(self.lengths, self.metadata.prompt_tokens):
            raise ValueError("Ragged offsets changed actual input lengths")

    def batch(self, rows, layer, device):
        maximum = int(self.lengths[rows].max())
        raw = np.zeros((len(rows), maximum, self.hidden), dtype=np.float16)
        mask = np.zeros((len(rows), maximum), dtype=bool)
        position = self.layers.index(layer)
        for i, row in enumerate(rows):
            begin, end = map(int, self.offsets[int(row):int(row) + 2])
            raw[i, :end - begin] = self.raw[begin:end, position]
            mask[i, :end - begin] = True
        return torch.from_numpy(raw).to(device), torch.from_numpy(mask).to(device)


def make_heads(settings, device):
    heads, audit = build_matched_heads(2560, settings["width"], settings["seed"],
        settings["dropout"], device, settings["projection_chunk_size"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings["seed"])
        single = ExperimentHead(2560, mode="last", width=settings["width"], dropout=settings["dropout"])
    template = heads[ARMS[0]]
    single.project.load_state_dict(template.project.state_dict())
    single.classifier.load_state_dict(template.classifier.state_dict())
    heads = {"last_mlp": single.to(device), **heads}
    audit["global_initialization_shared"] = True
    audit["single_parameters"] = parameter_count(single)
    return heads, audit


def forward(head, arm, raw, mask):
    if arm == "last_mlp":
        return head({"layers": (raw,), "mask": mask})
    return head(raw, mask)


@torch.no_grad()
def predict(heads, store, rows, layer, settings, device, evidence=False):
    for head in heads.values():
        head.eval()
    logits = {arm: [] for arm in heads}
    detail = []
    for batch in split_microbatches(rows, store.lengths, settings["micro_token_budget"], settings["micro_max_rows"]):
        raw, mask = store.batch(batch, layer, device)
        record = {int(row): {"row": int(row)} for row in batch}
        for arm, head in heads.items():
            result = forward(head, arm, raw, mask)
            logits[arm].append(result["logits"].cpu().numpy())
            if evidence and arm != "last_mlp":
                weights = result["attention_weights"].cpu().numpy()
                for i, row in enumerate(batch):
                    value = weights[i, :int(store.lengths[row])].astype(np.float32)
                    if not np.isfinite(value).all() or abs(float(value.sum()) - 1.) > 2e-5:
                        raise FloatingPointError("Invalid attention probability simplex")
                    record[int(row)][f"weights_{arm}"] = value.tolist()
                    record[int(row)][f"support_{arm}"] = int((value > 0).sum())
                    record[int(row)][f"effective_tokens_{arm}"] = float(1. / np.square(value).sum())
                    record[int(row)][f"top_positions_{arm}"] = np.argsort(-value, kind="stable")[:8].tolist()
        if evidence:
            detail.extend(record.values())
    values = {arm: np.concatenate(parts) for arm, parts in logits.items()}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Nonfinite prediction")
    return values, detail


def validate_fold(directory, protocol, fold):
    record = read(directory / "complete.json")
    if (record.get("complete") is not True or
            record["protocol_sha256"] != protocol["_sha"] or record["fold"] != fold or
            record["layer"] != protocol["fold_layers"][str(fold)] or
            record.get("cache_manifest_sha256") != sha256(Path(protocol["token_cache"]) / "manifest.json")):
        raise ValueError("Fold publication identity differs")
    if set(record.get("artifacts", {})) != set(FOLD_ARTIFACTS):
        raise ValueError("Fold publication omits required scientific artifacts")
    for name, digest in record["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Fold artifact changed: {name}")
    return record


def preflight(output, device):
    """No fitted state escapes this numerical check; targets are synthetic."""
    torch.set_num_threads(4)
    protocol = load_protocol(output)
    store = TokenStore(protocol["token_cache"], protocol)
    settings = protocol["settings"]
    order = np.argsort(store.lengths, kind="stable")
    rows = np.unique(order[[0, len(order) // 2, int(len(order) * .95), len(order) - 1]])
    layer = protocol["fold_layers"]["0"]
    raw, mask = store.batch(rows, layer, device)
    heads, capacity = make_heads(settings, device)
    initial, audits, supports = {}, [], {}
    for arm, head in heads.items():
        head.eval()
        with torch.no_grad():
            value = forward(head, arm, raw, mask)
            initial[arm] = value["logits"].detach().clone()
            padded_raw = F.pad(raw, (0, 0, 0, 7), value=float("nan"))
            padded_mask = F.pad(mask, (0, 7), value=False)
            padded = forward(head, arm, padded_raw, padded_mask)
            torch.testing.assert_close(value["logits"], padded["logits"], rtol=2e-5, atol=2e-6)
            if arm != "last_mlp":
                weights = value["attention_weights"]
                if (not torch.isfinite(weights).all() or torch.any(weights[~mask] != 0) or
                        not torch.allclose(weights.sum(1), torch.ones(len(rows), device=device), atol=2e-5)):
                    raise FloatingPointError("Preflight attention padding/simplex failed")
                supports[arm] = value["support_size"].cpu().tolist()
        torch.testing.assert_close(initial[arm], initial["last_mlp"], rtol=2e-5, atol=2e-6)
        before = {name: parameter.detach().cpu().clone() for name, parameter in head.named_parameters()}
        optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        synthetic_target = (torch.arange(len(rows), device=device) % 2).float()
        for step in (1, 2):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(forward(head, arm, raw, mask)["logits"], synthetic_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Preflight synthetic loss failed")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            audit = parameter_audit(head, before, step)
            if any(not entry["grad_present"] or not entry["grad_finite"]
                   for entry in audit["parameters"].values()):
                raise FloatingPointError("Preflight gradient failed")
            audits.append(dict(arm=arm, **audit))
        head.eval()
        with torch.no_grad():
            activated = forward(head, arm, raw, mask)
            padded = forward(head, arm, padded_raw, padded_mask)
            torch.testing.assert_close(activated["logits"], padded["logits"], rtol=2e-5, atol=2e-6)
            if arm != "last_mlp":
                torch.testing.assert_close(activated["attention_weights"],
                    padded["attention_weights"][:, :mask.shape[1]], rtol=2e-5, atol=2e-6)
                if audit["parameters"]["pool_output.weight"]["delta_norm"] <= 0:
                    raise FloatingPointError("Preflight local residual did not activate")
    load_protocol(output)
    value = dict(complete=True, passed=True, protocol_sha256=sha256(output / "protocol.json"),
        cache_manifest_sha256=store.manifest_sha256, rows=rows.tolist(), lengths=store.lengths[rows].tolist(),
        capacity=capacity, initial_logits_shared=True, padding_invariance=True,
        padding_checked_after_local_branch_update=True,
        initial_supports=supports, gradient_audits=audits, synthetic_targets=True,
        no_preflight_parameters_reused=True)
    dump(output / "head_preflight.json", value)
    print(json.dumps(dict(stage="token_head_preflight", passed=True, capacity=capacity)), flush=True)
    return value


def train_fold(output, fold, device):
    torch.set_num_threads(4)
    protocol = load_protocol(output)
    protocol["_sha"] = sha256(output / "protocol.json")
    final = output / f"fold_{fold}"
    if (final / "complete.json").exists():
        return validate_fold(final, protocol, fold)
    if final.exists():
        raise ValueError("Uncommitted final directory; inspect before retry")
    store = TokenStore(protocol["token_cache"], protocol)
    gate = read(output / "head_preflight.json")
    if (gate.get("passed") is not True or gate.get("complete") is not True or
            gate.get("protocol_sha256") != protocol["_sha"] or
            gate.get("cache_manifest_sha256") != store.manifest_sha256):
        raise ValueError("A same-protocol, same-cache GPU head preflight is required")
    metadata, settings = store.metadata, protocol["settings"]
    assignments = pd.read_parquet(Path(protocol["source_study"]) /
                                  f"partition/folds/fold_{fold}_assignments.parquet")
    indices = indices_from_assignments(metadata, fold, assignments)
    layer = protocol["fold_layers"][str(fold)]
    attempts = output / "fold_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    number = len(list(attempts.glob(f"fold_{fold}_*"))) + 1
    work = attempts / f"fold_{fold}_{number:03d}"
    work.mkdir()
    heads, matched = make_heads(settings, device)
    initial = {arm: state_digest(head) for arm, head in heads.items()}
    before = {arm: {n: p.detach().cpu().clone() for n, p in head.named_parameters()} for arm, head in heads.items()}
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=settings["lr"],
                                        weight_decay=settings["weight_decay"]) for arm, head in heads.items()}
    y = metadata.label.to_numpy(dtype=np.int64)
    best_bce = {arm: np.inf for arm in heads}
    best_epoch = {arm: 0 for arm in heads}
    states, history, audits, extensions = {}, [], [], []
    epoch, budget, global_step = 0, settings["epochs"], 0
    started = time.monotonic()
    while epoch < budget:
        check_stop(protocol)
        order = np.random.default_rng(settings["seed"] * 1009 + epoch).permutation(indices["fit"])
        total = {arm: 0. for arm in heads}
        for step, begin in enumerate(range(0, len(order), settings["logical_batch_size"])):
            logical = order[begin:begin + settings["logical_batch_size"]]
            for arm in heads:
                heads[arm].train()
                optimizers[arm].zero_grad(set_to_none=True)
            global_step += 1
            for micro, rows in enumerate(split_microbatches(logical, store.lengths,
                    settings["micro_token_budget"], settings["micro_max_rows"])):
                check_stop(protocol)
                raw, mask = store.batch(rows, layer, device)
                target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
                for arm, head in heads.items():
                    torch.manual_seed((settings["seed"] * 1000003 + epoch * 1009 +
                                       step * 4099 + micro) % (2**63 - 1))
                    logits = forward(head, arm, raw, mask)["logits"]
                    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {arm}")
                    (loss / len(logical)).backward()
                    total[arm] += float(loss.detach())
            for arm, head in heads.items():
                nn.utils.clip_grad_norm_(head.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                optimizers[arm].step()
                if global_step in (1, 2, 10):
                    audits.append(dict(arm=arm, **parameter_audit(head, before[arm], global_step)))
            if step % 15 == 0:
                print(json.dumps(dict(stage="token_heads", fold=fold, epoch=epoch + 1,
                                      step=step, logical_steps=int(np.ceil(len(order) / 256)))), flush=True)
        dev_logits, _ = predict(heads, store, indices["dev"], layer, settings, device)
        dev_bce = {}
        for arm, values in dev_logits.items():
            dev_bce[arm] = binary_metrics(y[indices["dev"]], sigmoid(values), .5)["bce"]
            if dev_bce[arm] < best_bce[arm]:
                best_bce[arm], best_epoch[arm] = dev_bce[arm], epoch + 1
                states[arm] = {n: v.detach().cpu().clone() for n, v in heads[arm].state_dict().items()}
        history.append(dict(epoch=epoch + 1,
            train_bce={arm: total[arm] / len(order) for arm in heads}, dev_bce=dev_bce))
        dump(work / "history.json", history)
        epoch += 1
        print(json.dumps(dict(stage="token_epoch_complete", fold=fold, epoch=epoch,
                              budget=budget, elapsed_seconds=time.monotonic() - started)), flush=True)
        if epoch == budget:
            extension = budget_extension(history, best_epoch, budget, settings["max_epochs"])
            if extension:
                extensions.append(extension)
                budget = extension["new_budget"]
                print(json.dumps(dict(stage="token_budget_extension", fold=fold, **extension)), flush=True)
    for arm, head in heads.items():
        if arm not in states:
            raise FloatingPointError("No finite selected head")
        audits.append(dict(arm=arm, checkpoint="last_trained",
                           **parameter_audit(head, before[arm], global_step)))
        head.load_state_dict(states[arm])
    dev_logits, _ = predict(heads, store, indices["dev"], layer, settings, device)
    thresholds = {arm: accuracy_threshold(y[indices["dev"]], sigmoid(values))
                  for arm, values in dev_logits.items()}
    selection = dict(fold=fold, layer=layer, protocol_sha256=protocol["_sha"], best_epoch=best_epoch,
                     dev_bce=best_bce, thresholds=thresholds, capacity=matched,
                     budgets=dict(initial=settings["epochs"], actual=budget, cap=settings["max_epochs"], extensions=extensions))
    # Freeze all choices before generating the outer-fold predictions.
    dump(work / "selection.json", selection)
    atomic_torch_save(dict(states=states, selection=selection), work / "heads.pt")
    dump(work / "gradient_update_audit.json", audits)
    held_logits, details = predict(heads, store, indices["held"], layer, settings, device, evidence=True)
    frame = metadata.iloc[indices["held"]].copy()
    for arm, values in held_logits.items():
        p = sigmoid(values)
        frame[f"probability_{arm}"] = p
        frame[f"threshold_{arm}"] = thresholds[arm]
        frame[f"prediction_{arm}"] = (p >= thresholds[arm]).astype(np.int8)
    frame.to_parquet(work / "held_predictions.parquet", index=False)
    evidence = pd.DataFrame(details)
    evidence["base_id"] = metadata.iloc[evidence.row.to_numpy()].base_id.to_numpy()
    evidence.to_parquet(work / "held_attention.parquet", index=False)
    record = dict(complete=True, protocol_sha256=protocol["_sha"], fold=fold, layer=layer,
        rows=len(frame), cache_manifest_sha256=store.manifest_sha256,
        initial_state_sha256=initial, capacity=matched,
        row_identity={role: dict(rows=len(rows), groups=metadata.iloc[rows].group_id.nunique(),
            ordered_base_id_sha256=ordered_base_id_sha256(metadata.iloc[rows].base_id.tolist()))
            for role, rows in indices.items()},
        artifacts={name: sha256(work / name) for name in FOLD_ARTIFACTS})
    dump(work / "complete.json", record)
    load_protocol(output)
    os.rename(work, final)
    return record


def report(output):
    protocol = load_protocol(output)
    protocol["_sha"] = sha256(output / "protocol.json")
    frames, folds, attention_frames = [], [], []
    reference = pd.read_parquet(Path(protocol["source_study"]) /
                               "lora_study/features/adapted/B.metadata.parquet")
    for fold in range(5):
        directory = output / f"fold_{fold}"
        record = validate_fold(directory, protocol, fold)
        assignments = pd.read_parquet(Path(protocol["source_study"]) /
                                      f"partition/folds/fold_{fold}_assignments.parquet")
        indices = indices_from_assignments(reference, fold, assignments)
        held = pd.read_parquet(directory / "held_predictions.parquet")
        attention = pd.read_parquet(directory / "held_attention.parquet")
        expected_ids = reference.iloc[indices["held"]].base_id.tolist()
        expected_digest = ordered_base_id_sha256(expected_ids)
        if (held.base_id.tolist() != expected_ids or attention.base_id.tolist() != expected_ids or
                not (held.outer_fold == fold).all() or record["rows"] != len(expected_ids) or
                record["row_identity"]["held"]["ordered_base_id_sha256"] != expected_digest):
            raise ValueError("Per-fold outer predictions differ from the registered held groups")
        folds.append(record)
        frames.append(held)
        attention_frames.append(attention)
    frame = pd.concat(frames).sort_values("base_id").reset_index(drop=True)
    if frame.base_id.tolist() != reference.base_id.tolist() or len(frame) != 20048:
        raise ValueError("Incomplete or duplicated OOF rows")
    for column in ("group_id", "source", "label", "outer_fold"):
        if frame[column].tolist() != reference[column].tolist():
            raise ValueError("OOF label/group identity changed")
    result = dict(schema="hanguard_sparse_token_readout_analysis_v1", complete=False,
        protocol_sha256=protocol["_sha"], rows=len(frame), groups=frame.group_id.nunique(),
        scope=protocol["scope"], formal_holdout_accessed=False,
        metrics={arm: _metrics_with_row_thresholds(frame, arm) for arm in ALL_ARMS},
        comparisons=[], folds=folds)
    result["interval_scope"] = protocol.get("interval_scope", "Exploratory individual conditional intervals")
    result["estimand"] = protocol.get("estimand")
    result["legacy_equivalence"] = protocol.get("legacy_equivalence")
    result["legacy_numeric_equivalence_passed"] = protocol.get("legacy_numeric_equivalence_passed")
    result["preserved_legacy_failure"] = protocol.get("preserved_legacy_failure")
    y = frame.label.to_numpy()
    for candidate, baseline in PAIRS:
        predictions = [frame[f"prediction_{arm}"].to_numpy() for arm in (candidate, baseline)]
        a, b = (values == y for values in predictions)
        bootstrap = group_cluster_bootstrap(y, *predictions, frame.group_id.astype(str),
            n_resamples=protocol["settings"]["bootstrap_draws"], seed=42)
        fixed = [(frame[f"probability_{arm}"].to_numpy() >= .5).astype(np.int8)
                 for arm in (candidate, baseline)]
        result["comparisons"].append(dict(candidate=candidate, baseline=baseline,
            rescued=int((a & ~b).sum()), harmed=int((~a & b).sum()),
            net_correct=int(a.sum() - b.sum()), group_bootstrap=bootstrap,
            fixed_0_5_group_bootstrap=group_cluster_bootstrap(y, *fixed, frame.group_id.astype(str),
                n_resamples=protocol["settings"]["bootstrap_draws"], seed=42),
            by_source={str(source): dict(rows=len(part),
                net_correct=int(((part[f"prediction_{candidate}"] == part.label).astype(int) -
                                 (part[f"prediction_{baseline}"] == part.label).astype(int)).sum()))
                for source, part in frame.groupby("source")}))
    evidence = pd.concat(attention_frames).sort_values("base_id").reset_index(drop=True)
    if evidence.base_id.tolist() != frame.base_id.tolist():
        raise ValueError("Evidence records do not match held predictions")
    result["attention_summary"] = {}
    for arm in ARMS:
        support = evidence[f"support_{arm}"].to_numpy()
        effective = evidence[f"effective_tokens_{arm}"].to_numpy()
        lengths = frame.prompt_tokens.to_numpy()
        result["attention_summary"][arm] = dict(
            median_support=float(np.median(support)), median_support_fraction=float(np.median(support / lengths)),
            median_effective_tokens=float(np.median(effective)),
            fully_dense_fraction=float(np.mean(support == lengths)),
            by_length={name: dict(rows=int(mask.sum()), median_support=float(np.median(support[mask])),
                median_support_fraction=float(np.median(support[mask] / lengths[mask])),
                median_effective_tokens=float(np.median(effective[mask])))
                for name, mask in (("1_64", lengths <= 64), ("65_256", (lengths > 64) & (lengths <= 256)),
                                   ("257_512", (lengths > 256) & (lengths <= 512)), ("over_512", lengths > 512))
                if mask.any()})
    result["metrics_by_source"] = {str(source): {arm: _metrics_with_row_thresholds(part, arm)
        for arm in ALL_ARMS} for source, part in frame.groupby("source")}
    result["metrics_by_length"] = {name: dict(rows=int(mask.sum()), metrics={arm:
        _metrics_with_row_thresholds(frame.loc[mask], arm) for arm in ALL_ARMS})
        for name, mask in (("1_64", frame.prompt_tokens <= 64),
                          ("65_256", (frame.prompt_tokens > 64) & (frame.prompt_tokens <= 256)),
                          ("257_512", (frame.prompt_tokens > 256) & (frame.prompt_tokens <= 512)),
                          ("over_512", frame.prompt_tokens > 512)) if mask.any()}
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    target = directory / "results.json"
    if target.exists():
        raise FileExistsError("Never overwrite published analysis")
    frame.to_parquet(directory / "held_predictions.parquet", index=False)
    result["artifacts"] = {"held_predictions.parquet": sha256(directory / "held_predictions.parquet")}
    result["limitations"] = [
        "Same nominal local-head capacity does not imply same effective query capacity.",
        "The fresh last MLP uses the new token cache and matched microbatch path at the prespecified old-selected layer; it is not an exhaustive best-single search on the new features. Old scores are context only.",
        "Primary choices use inner dev only, but the B dataset was already explored.",
        "Conditional group bootstrap does not include adapter/head-seed or fresh-dataset uncertainty.",
        "Sparse attention support is not a gold harmful-span label or proof of semantic extraction."]
    result["complete"] = True
    load_protocol(output)
    dump(target, result)
    print(json.dumps(dict(stage="token_readout_report_complete", results_path=str(target))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "preflight", "fold", "report"))
    parser.add_argument("--source-study", default=str(ROOT / "outputs/hanguard_ab_diagnosis_20261012"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--legacy-audit", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "init":
        value = initialize(args)
        print(json.dumps(dict(initialized=True, fold_layers=value["fold_layers"], settings=value["settings"])))
    elif args.command == "preflight":
        preflight(args.output, args.device)
    elif args.command == "fold":
        if args.fold is None:
            parser.error("--fold required")
        train_fold(args.output, args.fold, args.device)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
