"""Cross-LoRA-seed replication of fixed four-layer dynamic fusion.

Three independently trained E03 adapters (seeds 42/43/44) are frozen.  For
each adapter this study extracts only layers 8/16/24/32 and trains matched
``dynamic_fixed`` and ``dynamic_repeat_last`` heads over at least five head
seeds.  Checkpoints are chosen by validation BCE and all primary decisions use
threshold 0.5.  Test parquet/features cannot be deserialized until every
adapter, cache, head checkpoint, validation selection, dataset, and code file
has been hashed into ``test_barrier.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.hanguard import dynamic_gate_diagnostics as dynamic  # noqa: E402
from scripts.hanguard import frozen_lora_layer_diagnostics as frozen  # noqa: E402


DEFAULT_SOURCE = ROOT / "outputs/hanguard_e03_lora_multiseed_20261012"
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_multiseed_fixed_layer_replication_20261012"
ADAPTER_SEEDS = (42, 43, 44)
DEFAULT_HEAD_SEEDS = (42, 43, 44, 45, 46)
FIXED_LAYERS = (8, 16, 24, 32)
ARMS = ("dynamic_repeat_last", "dynamic_fixed")
PRIMARY_THRESHOLD = 0.5
HEAD_DROPOUT = 0.1
FUSION_INIT_SCALE = 1e-3


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _code_identity() -> dict[str, str]:
    paths = {
        "multiseed_fixed_layer_replication.py": Path(__file__),
        "dynamic_gate_diagnostics.py": Path(dynamic.__file__),
        "frozen_lora_layer_diagnostics.py": Path(frozen.__file__),
        "repaired_study.py": ROOT / "scripts/hanguard/repaired_study.py",
        "hanguard_model.py": ROOT / "hanguard_model.py",
    }
    return {name: frozen.sha256(path) for name, path in paths.items()}


def _code_paths() -> dict[str, Path]:
    return {
        "code/multiseed_fixed_layer_replication.py": Path(__file__),
        "code/dynamic_gate_diagnostics.py": Path(dynamic.__file__),
        "code/frozen_lora_layer_diagnostics.py": Path(frozen.__file__),
        "code/repaired_study.py": ROOT / "scripts/hanguard/repaired_study.py",
        "code/hanguard_model.py": ROOT / "hanguard_model.py",
    }


def select_arm_features(features: torch.Tensor, arm: str) -> torch.Tensor:
    """Map a [batch,4,hidden] fixed-layer cache to one matched arm."""
    if features.ndim != 3 or features.shape[1] != len(FIXED_LAYERS):
        raise ValueError("Features must contain exactly the four registered layers")
    if arm == "dynamic_fixed":
        return features
    if arm == "dynamic_repeat_last":
        return features[:, -1:, :].expand(-1, len(FIXED_LAYERS), -1)
    raise ValueError(f"Unknown replication arm: {arm}")


def build_matched_replication_heads(
    hidden_size: int,
    width: int,
    gate_width: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, dynamic.DynamicLayerGateHead], dict]:
    """Construct independent byte-identical heads for the paired arm inputs."""
    heads = {}
    for arm in ARMS:
        torch.manual_seed(int(seed))
        heads[arm] = dynamic.DynamicLayerGateHead(
            hidden_size, width, gate_width,
            dropout=HEAD_DROPOUT, fusion_init_scale=FUSION_INIT_SCALE,
        ).to(device)
    matched = frozen.assert_capacity_matched_initialization(heads)
    matched["parameter_counts"] = {
        arm: sum(parameter.numel() for parameter in head.parameters())
        for arm, head in heads.items()
    }
    matched["initial_state_sha256"] = {
        arm: frozen.state_digest(head) for arm, head in heads.items()
    }
    return heads, matched


def choose_validation_checkpoints(history: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """Select the earliest minimum-validation-BCE epoch independently per arm."""
    if not history or any(record.get("split") != "validation" for record in history):
        raise ValueError("Checkpoint choice requires validation-only history")
    output = {}
    for arm in ARMS:
        candidates = []
        for record in history:
            values = record.get("validation_bce", {})
            if arm not in values or not math.isfinite(float(values[arm])):
                raise ValueError(f"Missing finite validation BCE for {arm}")
            candidates.append((float(values[arm]), int(record["epoch"])))
        value, epoch = min(candidates, key=lambda item: (item[0], item[1]))
        output[arm] = dict(epoch=epoch, validation_bce=value)
    return output


def validate_four_layer_cache(
    path: Path,
    rows: int,
    hidden_size: int,
    expected_base_id_sha256: str,
    base_ids: Sequence[Any],
) -> dict:
    return frozen.validate_feature_cache(
        path,
        (int(rows), len(FIXED_LAYERS), int(hidden_size)),
        np.float16,
        expected_base_id_sha256,
        base_ids,
    )


def _gain_distribution(
    gains: Mapping[int, Mapping[str, float]], unit: str, inference: str
) -> dict:
    if not gains:
        raise ValueError("At least one adapter gain is required")
    seeds = sorted(map(int, gains))
    metric_names = tuple(next(iter(gains.values())).keys())
    if any(set(value) != set(metric_names) for value in gains.values()):
        raise ValueError("Adapter gain metrics must align")
    metrics = {}
    for name in metric_names:
        values = np.asarray([float(gains[seed][name]) for seed in seeds], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Adapter gains must be finite")
        improvement = values < 0 if name in ("bce", "loss") else values > 0
        metrics[name] = dict(
            values=values.tolist(),
            mean=float(values.mean()),
            median=float(np.median(values)),
            std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            minimum=float(values.min()),
            maximum=float(values.max()),
            positive_fraction=float((values > 0).mean()),
            improved_fraction=float(improvement.mean()),
        )
    return dict(
        unit=unit,
        units=len(seeds),
        seeds=seeds,
        metrics=metrics,
        inference=inference,
    )


def adapter_gain_distribution(gains: Mapping[int, Mapping[str, float]]) -> dict:
    """Describe gains with the independently trained LoRA adapter as the unit."""
    result = _gain_distribution(
        gains,
        unit="independent E03 LoRA seed",
        inference="descriptive distribution; n=3 adapters is too small for a reliable population test",
    )
    result.update(adapters=result["units"], adapter_seeds=result["seeds"])
    return result


def head_seed_gain_distribution(gains: Mapping[int, Mapping[str, float]]) -> dict:
    result = _gain_distribution(
        gains,
        unit="head seed conditional on one frozen E03 adapter",
        inference="conditional optimization variability; head seeds are not independent LoRA replications",
    )
    result.update(head_seeds=result["seeds"])
    return result


def _artifact_records(paths: Mapping[str, Path | str]) -> dict[str, dict]:
    records = {}
    for name, value in sorted(paths.items()):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing barrier artifact {name}: {path}")
        records[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=frozen.sha256(path))
    return records


def create_replication_test_barrier(
    output: Path,
    artifact_paths: Mapping[str, Path | str],
    identity: Mapping[str, Any],
    test_dataset_path: Path,
) -> dict:
    """Freeze every pre-test choice; hashing test bytes does not deserialize rows."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_dataset_path = Path(test_dataset_path).resolve()
    if not test_dataset_path.is_file():
        raise FileNotFoundError(test_dataset_path)
    value = dict(
        schema="hanguard_multiseed_fixed_layer_test_barrier_v1",
        state="locked",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        identity=dict(identity),
        artifacts=_artifact_records(artifact_paths),
        test_dataset=dict(
            path=str(test_dataset_path), bytes=test_dataset_path.stat().st_size,
            sha256=frozen.sha256(test_dataset_path),
        ),
        test_dataset_sha256=frozen.sha256(test_dataset_path),
        rule=(
            "Only train/validation selected checkpoints are locked. Test parquet and test features "
            "may be deserialized only after this barrier exists and verifies."
        ),
    )
    path = output / "test_barrier.json"
    if path.exists():
        prior = _json(path)
        keys = ("schema", "state", "identity", "artifacts", "test_dataset", "test_dataset_sha256")
        if any(prior.get(key) != value.get(key) for key in keys):
            raise ValueError("Existing replication test barrier differs")
        return prior
    frozen.dump(path, value)
    return value


def verify_replication_test_barrier(
    output: Path,
    artifact_paths: Mapping[str, Path | str],
    identity: Mapping[str, Any],
    test_dataset_path: Path,
) -> dict:
    output = Path(output).resolve()
    path = output / "test_barrier.json"
    if not path.is_file():
        raise FileNotFoundError("Replication test barrier is not locked")
    value = _json(path)
    if (
        value.get("schema") != "hanguard_multiseed_fixed_layer_test_barrier_v1"
        or value.get("state") != "locked"
        or value.get("identity") != dict(identity)
    ):
        raise ValueError("Replication test barrier identity changed")
    expected_names = set(artifact_paths)
    if set(value.get("artifacts", {})) != expected_names:
        raise ValueError("Replication barrier artifact set changed")
    for name, path_value in artifact_paths.items():
        path_value = Path(path_value).resolve()
        record = value["artifacts"][name]
        if (
            not path_value.is_file()
            or str(path_value) != record.get("path")
            or path_value.stat().st_size != int(record.get("bytes", -1))
            or frozen.sha256(path_value) != record.get("sha256")
        ):
            raise ValueError(f"Locked artifact changed: {name}")
    test_dataset_path = Path(test_dataset_path).resolve()
    record = value.get("test_dataset", {})
    if (
        not test_dataset_path.is_file()
        or str(test_dataset_path) != record.get("path")
        or test_dataset_path.stat().st_size != int(record.get("bytes", -1))
        or frozen.sha256(test_dataset_path) != record.get("sha256")
    ):
        raise ValueError("Locked test dataset changed")
    return value


def _adapter_records(source: Path, source_protocol: Mapping[str, Any]) -> dict[str, dict]:
    source = Path(source).resolve()
    protocol_hash = frozen.sha256(source / "protocol.json")
    output = {}
    for seed in ADAPTER_SEEDS:
        directory = source / "runs" / f"E03_s{seed}"
        selection_path, adapter_path = directory / "selection.json", directory / "adapter.pt"
        if not selection_path.is_file() or not adapter_path.is_file():
            raise FileNotFoundError(f"E03_s{seed} validation selection/adapter is incomplete")
        selection = _json(selection_path)
        if (
            selection.get("arm") != "E03"
            or int(selection.get("seed", -1)) != seed
            or selection.get("protocol_sha256") != protocol_hash
            or selection.get("train_sha256") != source_protocol["data_sha256"]["train"]
            or selection.get("validation_sha256") != source_protocol["data_sha256"]["validation"]
            or not math.isfinite(float(selection.get("validation_bce", float("nan"))))
        ):
            raise ValueError(f"E03_s{seed} is not a valid validation-selected adapter")
        adapter_hash = frozen.sha256(adapter_path)
        if selection.get("checkpoint_hashes", {}).get("adapter.pt") != adapter_hash:
            raise ValueError(f"E03_s{seed} adapter differs from its validation selection")
        output[str(seed)] = dict(
            adapter_path=str(adapter_path.resolve()), adapter_sha256=adapter_hash,
            adapter_bytes=adapter_path.stat().st_size,
            selection_path=str(selection_path.resolve()), selection_sha256=frozen.sha256(selection_path),
            selection_bytes=selection_path.stat().st_size,
            validation_bce=float(selection["validation_bce"]), best_epoch=selection["best_epoch"],
        )
    return output


def initialize(args: argparse.Namespace) -> dict:
    output, source = Path(args.output).resolve(), Path(args.source).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_protocol_path, source_arms_path = source / "protocol.json", source / "arms.json"
    if not source_protocol_path.is_file() or not source_arms_path.is_file():
        raise FileNotFoundError("Source multiseed E03 protocol/arms are incomplete")
    source_protocol, source_arms = _json(source_protocol_path), _json(source_arms_path)
    if tuple(map(int, source_protocol.get("seeds", ()))) != ADAPTER_SEEDS:
        raise ValueError("Source study must contain E03 adapter seeds 42/43/44")
    if source_protocol.get("all_arms") != ["E03"] or set(source_arms) != {"E03"}:
        raise ValueError("Source study must be the E03-only independent-LoRA-seed study")
    head_seeds = tuple(map(int, args.head_seeds))
    if len(head_seeds) < 5 or len(set(head_seeds)) != len(head_seeds):
        raise ValueError("At least five unique head seeds are required")
    data_dir, model_path = Path(source_protocol["data_dir"]), Path(source_protocol["model_path"])
    for filename, key in (("config.json", "model_config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        if frozen.sha256(model_path / filename) != source_protocol[key]:
            raise ValueError(f"Source base model artifact changed: {filename}")
    for split in ("train", "validation"):
        path = data_dir / f"{split}.parquet"
        if frozen.sha256(path) != source_protocol["data_sha256"][split]:
            raise ValueError(f"Source {split} dataset changed")
    adapters = _adapter_records(source, source_protocol)
    model_config = _json(model_path / "config.json")
    hidden_size = int(model_config.get("text_config", {}).get("hidden_size", 0))
    if hidden_size < 1:
        raise ValueError("Base model config lacks a positive text hidden size")
    value = dict(
        schema="hanguard_multiseed_fixed_layer_replication_v1",
        source_study=str(source),
        source_protocol_sha256=frozen.sha256(source_protocol_path),
        source_arms_sha256=frozen.sha256(source_arms_path),
        adapter_seeds=list(ADAPTER_SEEDS), adapters=adapters,
        data_dir=str(data_dir.resolve()), data_sha256=source_protocol["data_sha256"],
        rows=source_protocol["rows"], model_path=str(model_path.resolve()),
        hidden_size=hidden_size,
        model_config_sha256=source_protocol["model_config_sha256"],
        tokenizer_sha256=source_protocol["tokenizer_sha256"],
        model_artifacts=frozen._model_artifact_identity(model_path),
        lora=dict(
            rank=int(source_protocol["lora_rank"]), alpha=int(source_protocol["lora_alpha"]),
            dropout=float(source_protocol["lora_dropout"]), source_arm="E03",
        ),
        layers=list(FIXED_LAYERS), arms=list(ARMS),
        architecture=dict(
            width=int(args.width), gate_width=int(args.gate_width), dropout=HEAD_DROPOUT,
            fusion_init_scale=FUSION_INIT_SCALE,
            topology="DynamicLayerGateHead; four slots for both arms",
            repeat_control="layer 32 copied into all four slots",
        ),
        training=dict(
            head_seeds=list(head_seeds), epochs=int(args.epochs), batch_size=int(args.batch_size),
            lr=float(args.lr), weight_decay=float(args.weight_decay),
            checkpoint_selection="minimum validation BCE independently per adapter/head seed/arm",
            threshold=PRIMARY_THRESHOLD, common_batch_order=True, paired_dropout_stream=True,
        ),
        extraction=dict(
            token_budget=int(args.token_budget), max_micro=int(args.max_micro),
            pad_multiple=int(args.pad_multiple), max_tokens=int(source_protocol["max_tokens"]),
            layers=list(FIXED_LAYERS), dtype="float16", truncation=False,
        ),
        analysis=dict(
            repetitions=int(args.repetitions), seed=int(args.analysis_seed),
            adapter_primary="arithmetic mean probability over every registered head seed",
            cross_adapter_unit="independent E03 LoRA seed",
        ),
        code_sha256=_code_identity(),
        stage_rule=(
            "Extract train/validation and select every head checkpoint first; create test_barrier.json; "
            "only then deserialize or extract test rows."
        ),
        primary_estimand=(
            "dynamic_fixed minus dynamic_repeat_last at fixed threshold 0.5, paired within adapter/head "
            "seed; adapter-level ensemble gain is summarized across three independent E03 LoRA seeds"
        ),
        limitations=[
            "The repository split identities have been explored, so test is an internal diagnostic.",
            "Three LoRA seeds support a descriptive stability check, not a precise population inference.",
            "Each LoRA adapter is paired with multiple head seeds; head seeds are not independent adapters.",
        ],
    )
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if _json(protocol_path) != value:
            raise ValueError("Requested settings differ from the registered replication protocol")
    else:
        frozen.dump(protocol_path, value)
    registration = dict(
        schema="hanguard_multiseed_fixed_layer_registration_v1",
        state="registered_pretest",
        protocol_sha256=frozen.sha256(protocol_path),
        source_protocol_sha256=value["source_protocol_sha256"],
        adapter_sha256={seed: record["adapter_sha256"] for seed, record in adapters.items()},
        rule="Protocol, adapters, data identities, architecture, and budgets are immutable before feature extraction.",
    )
    registration_path = output / "registration.json"
    if registration_path.exists():
        if _json(registration_path) != registration:
            raise ValueError("Replication protocol registration changed")
    else:
        frozen.dump(registration_path, registration)
    return value


def load_protocol(output: Path) -> dict:
    output = Path(output).resolve()
    path = output / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run init first: {path}")
    protocol = _json(path)
    if protocol.get("schema") != "hanguard_multiseed_fixed_layer_replication_v1":
        raise ValueError("Unexpected replication protocol schema")
    if protocol.get("code_sha256") != _code_identity():
        raise ValueError("Replication implementation changed after registration")
    registration_path = output / "registration.json"
    if not registration_path.is_file():
        raise FileNotFoundError("Replication protocol registration is missing")
    registration = _json(registration_path)
    if (
        registration.get("schema") != "hanguard_multiseed_fixed_layer_registration_v1"
        or registration.get("state") != "registered_pretest"
        or registration.get("protocol_sha256") != frozen.sha256(path)
    ):
        raise ValueError("Replication protocol changed after registration")
    if (
        tuple(protocol.get("adapter_seeds", ())) != ADAPTER_SEEDS
        or tuple(protocol.get("layers", ())) != FIXED_LAYERS
        or tuple(protocol.get("arms", ())) != ARMS
        or len(set(protocol.get("training", {}).get("head_seeds", ()))) < 5
        or float(protocol.get("training", {}).get("threshold", float("nan"))) != PRIMARY_THRESHOLD
    ):
        raise ValueError("Replication protocol structural contract changed")
    source = Path(protocol["source_study"])
    if frozen.sha256(source / "protocol.json") != protocol["source_protocol_sha256"]:
        raise ValueError("Source E03 multiseed protocol changed")
    if frozen.sha256(source / "arms.json") != protocol["source_arms_sha256"]:
        raise ValueError("Source E03 arms changed")
    if frozen._model_artifact_identity(Path(protocol["model_path"])) != protocol["model_artifacts"]:
        raise ValueError("Base model weight index or shard sizes changed")
    for filename, key in (("config.json", "model_config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
        if frozen.sha256(Path(protocol["model_path"]) / filename) != protocol[key]:
            raise ValueError(f"Base model artifact changed: {filename}")
    model_config = _json(Path(protocol["model_path"]) / "config.json")
    if int(model_config.get("text_config", {}).get("hidden_size", -1)) != int(protocol["hidden_size"]):
        raise ValueError("Base model hidden size changed")
    for seed, record in protocol["adapters"].items():
        for kind in ("adapter", "selection"):
            artifact = Path(record[f"{kind}_path"])
            if (
                not artifact.is_file()
                or artifact.stat().st_size != int(record[f"{kind}_bytes"])
                or frozen.sha256(artifact) != record[f"{kind}_sha256"]
            ):
                raise ValueError(f"Frozen E03_s{seed} {kind} changed")
    for split in ("train", "validation"):
        path = Path(protocol["data_dir"]) / f"{split}.parquet"
        if frozen.sha256(path) != protocol["data_sha256"][split]:
            raise ValueError(f"Registered {split} dataset changed")
    return protocol


class SelectedTokenBlockCapture:
    """Retain final-valid-token states only for the four registered blocks."""

    def __init__(self, backbone, layers: Sequence[int] = FIXED_LAYERS):
        self.backbone = backbone
        self.layers = tuple(map(int, layers))
        if self.layers != FIXED_LAYERS:
            raise ValueError("Only preregistered layers 8/16/24/32 may be captured")
        self.last_indices: torch.Tensor | None = None
        self.active = False
        self.values: dict[int, torch.Tensor] = {}
        self.handles = [
            backbone.layers[layer - 1].register_forward_hook(self._hook(layer))
            for layer in self.layers
        ]

    def _hook(self, layer: int):
        def capture(module, inputs, output):
            if not self.active:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if self.last_indices is None:
                raise RuntimeError("Missing last-token positions")
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            self.values[layer] = hidden[rows, self.last_indices].detach().to(torch.float16)
        return capture

    @torch.inference_mode()
    def __call__(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.values.clear()
        self.last_indices = frozen.last_true_indices(batch["attention_mask"])
        self.active = True
        try:
            device_type = batch["input_ids"].device.type
            with torch.autocast(
                device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
            ):
                self.backbone(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0),
                    use_cache=False,
                )
        finally:
            self.active = False
            self.last_indices = None
        if set(self.values) != set(self.layers):
            raise RuntimeError(f"Captured blocks changed: {sorted(self.values)}")
        result = torch.stack([self.values[layer] for layer in self.layers], dim=1).cpu()
        self.values.clear()
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


def _adapter_load_protocol(protocol: Mapping[str, Any], adapter_seed: int) -> dict:
    adapter = protocol["adapters"][str(int(adapter_seed))]
    return dict(
        model_path=protocol["model_path"], model_config_sha256=protocol["model_config_sha256"],
        tokenizer_sha256=protocol["tokenizer_sha256"], model_artifacts=protocol["model_artifacts"],
        adapter_path=adapter["adapter_path"], adapter_sha256=adapter["adapter_sha256"],
        lora_rank=protocol["lora"]["rank"], lora_alpha=protocol["lora"]["alpha"],
        lora_dropout=protocol["lora"]["dropout"], max_tokens=protocol["extraction"]["max_tokens"],
    )


def _load_split(split: str, tokenizer, protocol: Mapping[str, Any]) -> dict:
    from scripts.hanguard.repaired_study import load_split

    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    split_protocol = dict(protocol)
    split_protocol["max_tokens"] = int(protocol["extraction"]["max_tokens"])
    data = load_split(split, tokenizer, split_protocol)
    frame = pd.read_parquet(Path(protocol["data_dir"]) / f"{split}.parquet").sort_values("base_id").reset_index(drop=True)
    if frame.base_id.astype(str).tolist() != data["meta"].base_id.astype(str).tolist():
        raise ValueError(f"{split} tokenized row order changed")
    if not np.array_equal(frame.prompt_tokens.to_numpy(dtype=np.int32), data["lengths"]):
        raise ValueError(f"{split} registered prompt lengths changed")
    columns = ["base_id", "group_id", "source", "category_id", "text_form", "prompt_tokens"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{split} metadata schema changed")
    metadata = frame[columns].copy()
    if metadata.group_id.isna().any():
        raise ValueError(f"{split} contains missing group identities")
    metadata["group_id"] = metadata.group_id.astype(str)
    metadata["label"] = frame.prompt_harm_label.eq("harmful").astype(np.int8)
    if metadata.base_id.duplicated().any() or metadata.group_id.str.strip().eq("").any():
        raise ValueError(f"{split} row/group identities are invalid")
    data["metadata"] = metadata
    return data


def _make_batch(data: Mapping[str, Any], indices: np.ndarray, tokenizer, protocol: Mapping[str, Any]):
    batch = tokenizer.pad(
        [
            {"input_ids": data["encoded"][int(index)], "attention_mask": [1] * len(data["encoded"][int(index)])}
            for index in indices
        ],
        padding=True, pad_to_multiple_of=protocol["extraction"]["pad_multiple"], return_tensors="pt",
    )
    expected = torch.as_tensor(data["lengths"][indices], dtype=torch.long)
    if not torch.equal(batch["attention_mask"].sum(1), expected):
        raise AssertionError("Tokenizer padding changed full prompt lengths")
    return batch.to("cuda")


def _cache_paths(output: Path, adapter_seed: int, split: str) -> tuple[Path, Path, Path, Path]:
    directory = Path(output) / "features" / f"adapter_s{int(adapter_seed)}"
    return (
        directory / f"{split}.npy", directory / f"{split}.npy.part",
        directory / f"{split}.progress.json", directory / f"{split}.manifest.json",
    )


def _metadata_path(output: Path, split: str) -> Path:
    return Path(output) / "features" / f"{split}.metadata.parquet"


def _validate_split_isolation(output: Path, protocol: Mapping[str, Any]) -> None:
    frames = {}
    for split in ("train", "validation", "test"):
        path = _metadata_path(output, split)
        if not path.is_file():
            continue
        if split == "test":
            verify_test_barrier(output, protocol)
        frames[split] = pd.read_parquet(path, columns=["base_id", "group_id"])
    names = tuple(frames)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            base_overlap = set(frames[first].base_id.astype(str)) & set(frames[second].base_id.astype(str))
            group_overlap = set(frames[first].group_id.astype(str)) & set(frames[second].group_id.astype(str))
            if base_overlap or group_overlap:
                raise ValueError(
                    f"Split leakage {first}/{second}: {len(base_overlap)} base IDs, {len(group_overlap)} groups"
                )


def _barrier_identity(output: Path, protocol: Mapping[str, Any]) -> dict:
    return dict(
        protocol_sha256=frozen.sha256(Path(output) / "protocol.json"),
        adapter_seeds=protocol["adapter_seeds"],
        adapter_sha256={seed: value["adapter_sha256"] for seed, value in protocol["adapters"].items()},
        head_seeds=protocol["training"]["head_seeds"], layers=protocol["layers"],
        train_sha256=protocol["data_sha256"]["train"],
        validation_sha256=protocol["data_sha256"]["validation"],
        test_sha256=protocol["data_sha256"]["test"], threshold=PRIMARY_THRESHOLD,
    )


def _head_directory(output: Path, adapter_seed: int, head_seed: int) -> Path:
    return Path(output) / "heads" / f"adapter_s{adapter_seed}" / f"seed_{head_seed}"


def _validate_head_selection(
    output: Path, protocol: Mapping[str, Any], adapter_seed: int, head_seed: int
) -> dict:
    directory = _head_directory(output, adapter_seed, head_seed)
    selection_path = directory / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = _json(selection_path)
    if (
        selection.get("split") != "validation"
        or int(selection.get("adapter_seed", -1)) != int(adapter_seed)
        or int(selection.get("head_seed", -1)) != int(head_seed)
        or selection.get("layers") != list(FIXED_LAYERS)
        or set(selection.get("arms", {})) != set(ARMS)
    ):
        raise ValueError(f"Head selection identity changed: {adapter_seed}/{head_seed}")
    for arm in ARMS:
        record = selection["arms"][arm]
        checkpoint = directory / f"{arm}.pt"
        if (
            float(record.get("threshold", float("nan"))) != PRIMARY_THRESHOLD
            or not math.isfinite(float(record.get("validation_bce", float("nan"))))
            or not checkpoint.is_file()
            or record.get("checkpoint_sha256") != frozen.sha256(checkpoint)
        ):
            raise ValueError(f"Validation-selected head changed: {adapter_seed}/{head_seed}/{arm}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_checkpoint = dict(
            adapter_seed=int(adapter_seed), head_seed=int(head_seed), arm=arm,
            layers=list(FIXED_LAYERS), hidden_size=int(protocol["hidden_size"]),
            width=int(protocol["architecture"]["width"]),
            gate_width=int(protocol["architecture"]["gate_width"]),
            dropout=float(protocol["architecture"]["dropout"]),
            fusion_init_scale=float(protocol["architecture"]["fusion_init_scale"]),
            adapter_sha256=protocol["adapters"][str(adapter_seed)]["adapter_sha256"],
            protocol_sha256=frozen.sha256(Path(output) / "protocol.json"),
            best_epoch=int(record["best_epoch"]),
            best_validation_bce=float(record["validation_bce"]),
        )
        changed = [key for key, value in expected_checkpoint.items() if payload.get(key) != value]
        if changed:
            raise ValueError(
                f"Checkpoint contract changed: {adapter_seed}/{head_seed}/{arm}/{changed}"
            )
    matched = selection.get("matched_initialization", {})
    counts = matched.get("parameter_counts", {})
    hashes = matched.get("initial_state_sha256", {})
    if (
        set(counts) != set(ARMS) or len(set(counts.values())) != 1
        or set(hashes) != set(ARMS) or len(set(hashes.values())) != 1
    ):
        raise ValueError(f"Matched capacity/initialization audit changed: {adapter_seed}/{head_seed}")
    predictions = directory / "validation_predictions.parquet"
    if (
        not predictions.is_file()
        or selection.get("validation_predictions_sha256") != frozen.sha256(predictions)
    ):
        raise ValueError(f"Validation prediction evidence changed: {adapter_seed}/{head_seed}")
    return selection


def _barrier_artifacts(output: Path, protocol: Mapping[str, Any]) -> dict[str, Path]:
    output, source = Path(output).resolve(), Path(protocol["source_study"])
    model_path = Path(protocol["model_path"])
    artifacts: dict[str, Path] = {
        "protocol": output / "protocol.json", "registration": output / "registration.json",
        "source/protocol": source / "protocol.json",
        "source/arms": source / "arms.json", **_code_paths(),
        "data/train": Path(protocol["data_dir"]) / "train.parquet",
        "data/validation": Path(protocol["data_dir"]) / "validation.parquet",
        "model/config": model_path / "config.json",
        "model/tokenizer": model_path / "tokenizer.json",
        "model/weight_index": model_path / "model.safetensors.index.json",
    }
    for filename in protocol["model_artifacts"].get("shard_bytes", {}):
        artifacts[f"model/weight/{filename}"] = model_path / filename
    for seed, record in protocol["adapters"].items():
        artifacts[f"adapter/{seed}/weights"] = Path(record["adapter_path"])
        artifacts[f"adapter/{seed}/selection"] = Path(record["selection_path"])
        for split in ("train", "validation"):
            final, _, _, manifest = _cache_paths(output, int(seed), split)
            artifacts[f"features/{seed}/{split}"] = final
            artifacts[f"features/{seed}/{split}_manifest"] = manifest
    for split in ("train", "validation"):
        artifacts[f"features/{split}_metadata"] = _metadata_path(output, split)
    for adapter_seed in protocol["adapter_seeds"]:
        for head_seed in protocol["training"]["head_seeds"]:
            directory = _head_directory(output, adapter_seed, head_seed)
            artifacts[f"head/{adapter_seed}/{head_seed}/selection"] = directory / "selection.json"
            artifacts[f"head/{adapter_seed}/{head_seed}/validation_predictions"] = directory / "validation_predictions.parquet"
            for arm in ARMS:
                artifacts[f"head/{adapter_seed}/{head_seed}/{arm}"] = directory / f"{arm}.pt"
    return artifacts


def load_feature_cache(
    output: Path, protocol: Mapping[str, Any], adapter_seed: int, split: str
) -> tuple[np.memmap, pd.DataFrame, dict]:
    output = Path(output).resolve()
    if split == "test":
        verify_test_barrier(output, protocol)
    final, _, _, manifest_path = _cache_paths(output, adapter_seed, split)
    metadata_path = _metadata_path(output, split)
    if not final.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing adapter_s{adapter_seed}/{split} feature cache")
    manifest, metadata = _json(manifest_path), pd.read_parquet(metadata_path)
    if (
        len(metadata) != int(protocol["rows"][split])
        or metadata.base_id.duplicated().any()
        or metadata.base_id.astype(str).tolist() != sorted(metadata.base_id.astype(str).tolist())
        or not metadata.label.isin((0, 1)).all()
        or metadata.group_id.isna().any()
        or metadata.group_id.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Feature metadata rows/labels/groups changed: {split}")
    if (metadata.groupby("group_id").label.nunique() != 1).any():
        raise ValueError(f"Feature metadata contains conflicting labels inside a group: {split}")
    expected_identity = dict(
        split=split, adapter_seed=int(adapter_seed), layers=list(FIXED_LAYERS),
        data_sha256=protocol["data_sha256"][split],
        adapter_sha256=protocol["adapters"][str(adapter_seed)]["adapter_sha256"],
        protocol_sha256=frozen.sha256(output / "protocol.json"),
    )
    if manifest.get("identity") != expected_identity or manifest.get("complete") is not True:
        raise ValueError(f"Feature cache identity changed: adapter_s{adapter_seed}/{split}")
    if manifest.get("metadata_sha256") != frozen.sha256(metadata_path):
        raise ValueError(f"Feature metadata changed: {split}")
    if manifest.get("sha256") != frozen.sha256(final):
        raise ValueError(f"Feature cache content changed: adapter_s{adapter_seed}/{split}")
    ordered = frozen.ordered_base_id_sha256(metadata.base_id.astype(str).tolist())
    if int(manifest.get("hidden_size", -1)) != int(protocol["hidden_size"]):
        raise ValueError(f"Feature hidden size changed: adapter_s{adapter_seed}/{split}")
    validation = validate_four_layer_cache(
        final, len(metadata), int(protocol["hidden_size"]), ordered,
        metadata.base_id.astype(str).tolist(),
    )
    if ordered != manifest.get("ordered_base_id_sha256") or validation["bytes"] != manifest.get("bytes"):
        raise ValueError(f"Feature row/extent identity changed: adapter_s{adapter_seed}/{split}")
    return np.load(final, mmap_mode="r", allow_pickle=False), metadata, manifest


def extract_splits(args: argparse.Namespace, splits: Sequence[str]) -> dict:
    from scripts.hanguard.repaired_study import microbatches

    output, protocol = Path(args.output).resolve(), load_protocol(args.output)
    splits = tuple(splits)
    if any(split not in ("train", "validation", "test") for split in splits):
        raise ValueError(splits)
    if "test" in splits:
        verify_test_barrier(output, protocol)
    completed = []
    for adapter_seed in protocol["adapter_seeds"]:
        pending = []
        for split in splits:
            final, temporary, progress, manifest = _cache_paths(output, adapter_seed, split)
            metadata = _metadata_path(output, split)
            if final.is_file() and manifest.is_file() and metadata.is_file():
                load_feature_cache(output, protocol, adapter_seed, split)
                completed.append(f"{adapter_seed}/{split}")
                continue
            if final.exists() or manifest.exists():
                raise ValueError(f"Incomplete final cache for adapter_s{adapter_seed}/{split}")
            if temporary.exists() != progress.exists():
                raise ValueError(f"Incomplete resumable cache for adapter_s{adapter_seed}/{split}")
            pending.append(split)
        if not pending:
            continue
        model, backbone, tokenizer = frozen._load_frozen_model(
            _adapter_load_protocol(protocol, adapter_seed)
        )
        hidden_size = int(model.config.text_config.hidden_size)
        if hidden_size != int(protocol["hidden_size"]):
            raise ValueError("Loaded base model hidden size differs from protocol")
        capture = SelectedTokenBlockCapture(backbone)
        try:
            for split in pending:
                data = _load_split(split, tokenizer, protocol)
                metadata = data["metadata"]
                metadata_path = _metadata_path(output, split)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                if metadata_path.is_file():
                    existing = pd.read_parquet(metadata_path)
                    if not existing.equals(metadata):
                        raise ValueError(f"Shared {split} metadata differs across adapters")
                else:
                    metadata.to_parquet(metadata_path, index=False)
                final, temporary, progress_path, manifest_path = _cache_paths(output, adapter_seed, split)
                final.parent.mkdir(parents=True, exist_ok=True)
                shape = (len(metadata), len(FIXED_LAYERS), hidden_size)
                identity = dict(
                    split=split, adapter_seed=int(adapter_seed), layers=list(FIXED_LAYERS),
                    data_sha256=protocol["data_sha256"][split],
                    adapter_sha256=protocol["adapters"][str(adapter_seed)]["adapter_sha256"],
                    protocol_sha256=frozen.sha256(output / "protocol.json"),
                )
                order = np.argsort(data["lengths"], kind="stable")
                groups = microbatches(
                    order, data["lengths"], protocol["extraction"]["token_budget"],
                    protocol["extraction"]["max_micro"], protocol["extraction"]["pad_multiple"],
                )
                start = 0
                if temporary.exists():
                    progress = _json(progress_path)
                    if progress.get("identity") != identity:
                        raise ValueError("Feature extraction resume identity changed")
                    start = int(progress["next_group"])
                    cache = np.lib.format.open_memmap(temporary, mode="r+", dtype=np.float16, shape=shape)
                else:
                    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
                began = time.monotonic()
                for number in range(start, len(groups)):
                    indices = groups[number]
                    values = capture(_make_batch(data, indices, tokenizer, protocol)).numpy()
                    if values.shape != (len(indices), len(FIXED_LAYERS), hidden_size):
                        raise ValueError("Selected block capture shape changed")
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Nonfinite selected block features")
                    cache[indices] = values
                    if (number + 1) % max(1, int(args.status_every)) == 0 or number + 1 == len(groups):
                        cache.flush()
                        frozen.dump(progress_path, dict(
                            identity=identity, next_group=number + 1, total_groups=len(groups),
                            elapsed_seconds=time.monotonic() - began,
                        ))
                        print(json.dumps(dict(
                            state="multiseed_extract", adapter_seed=adapter_seed, split=split,
                            group=number + 1, groups=len(groups),
                        )), flush=True)
                cache.flush(); del cache
                temporary.replace(final); progress_path.unlink(missing_ok=True)
                ordered = frozen.ordered_base_id_sha256(metadata.base_id.astype(str).tolist())
                validation = validate_four_layer_cache(
                    final, len(metadata), hidden_size, ordered, metadata.base_id.astype(str).tolist()
                )
                manifest = dict(
                    identity=identity, complete=True, rows=len(metadata), hidden_size=hidden_size,
                    dtype="float16", representation="raw block output at final valid token",
                    bytes=validation["bytes"], sha256=frozen.sha256(final),
                    metadata_sha256=frozen.sha256(metadata_path),
                    ordered_base_id_sha256=ordered,
                    finite_rows_checked=validation["finite_rows_checked"],
                )
                frozen.dump(manifest_path, manifest)
                completed.append(f"{adapter_seed}/{split}")
        finally:
            capture.close()
            del model, backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _validate_split_isolation(output, protocol)
    return dict(state="complete", caches=completed)


def _feature_batch(cache: np.memmap, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(cache[indices], dtype=np.float32)).to(device=device, non_blocking=True)


@torch.no_grad()
def _predict_heads(
    heads: Mapping[str, dynamic.DynamicLayerGateHead], cache: np.memmap,
    batch_size: int, device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    logits = {arm: np.empty(len(cache), dtype=np.float32) for arm in ARMS}
    weights = {arm: np.empty((len(cache), len(FIXED_LAYERS)), dtype=np.float32) for arm in ARMS}
    for head in heads.values():
        head.eval()
    for start in range(0, len(cache), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache)), dtype=np.int64)
        raw = _feature_batch(cache, indices, device)
        for arm, head in heads.items():
            result = head(select_arm_features(raw, arm))
            logits[arm][indices] = result["logits"].float().cpu().numpy()
            weights[arm][indices] = result["gate_weights"].float().cpu().numpy()
    if any(not np.isfinite(value).all() for value in (*logits.values(), *weights.values())):
        raise FloatingPointError("Nonfinite replication-head predictions")
    return logits, weights


def _state_copy(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def train_heads(args: argparse.Namespace) -> dict:
    output, protocol = Path(args.output).resolve(), load_protocol(args.output)
    device = torch.device(args.device)
    training = protocol["training"]
    summaries = []
    for adapter_seed in protocol["adapter_seeds"]:
        train_cache, train_metadata, _ = load_feature_cache(output, protocol, adapter_seed, "train")
        validation_cache, validation_metadata, _ = load_feature_cache(output, protocol, adapter_seed, "validation")
        y_train = torch.as_tensor(train_metadata.label.to_numpy(), dtype=torch.float32, device=device)
        y_validation = validation_metadata.label.to_numpy(dtype=np.int64)
        for head_seed in training["head_seeds"]:
            directory = _head_directory(output, adapter_seed, head_seed)
            selection_path = directory / "selection.json"
            if selection_path.is_file():
                summaries.append(
                    _validate_head_selection(output, protocol, adapter_seed, head_seed)
                )
                continue
            directory.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(head_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(head_seed)
            heads, initialization = build_matched_replication_heads(
                train_cache.shape[-1], protocol["architecture"]["width"],
                protocol["architecture"]["gate_width"], head_seed, device,
            )
            optimizers = {
                arm: torch.optim.AdamW(
                    head.parameters(), lr=training["lr"], weight_decay=training["weight_decay"]
                )
                for arm, head in heads.items()
            }
            best = {arm: dict(value=float("inf"), epoch=None, state=None) for arm in ARMS}
            history = []
            for epoch_index in range(training["epochs"]):
                epoch = epoch_index + 1
                order = np.random.default_rng(head_seed * 1009 + epoch_index).permutation(len(train_cache))
                totals, rows = {arm: 0.0 for arm in ARMS}, 0
                for step, start in enumerate(range(0, len(order), training["batch_size"])):
                    indices = order[start : start + training["batch_size"]]
                    raw = _feature_batch(train_cache, indices, device)
                    for optimizer in optimizers.values():
                        optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for arm, head in heads.items():
                        head.train()
                        dropout_seed = (head_seed * 1000003 + epoch_index * 1009 + step) % (2**63 - 1)
                        torch.manual_seed(dropout_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(dropout_seed)
                        logits = head(select_arm_features(raw, arm))["logits"]
                        loss = F.binary_cross_entropy_with_logits(logits, y_train[indices])
                        losses.append(loss); totals[arm] += float(loss.detach()) * len(indices)
                    torch.stack(losses).sum().backward()
                    for arm, optimizer in optimizers.items():
                        torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), 5.0, error_if_nonfinite=True)
                        optimizer.step()
                    rows += len(indices)
                validation_logits, _ = _predict_heads(
                    heads, validation_cache, training["batch_size"], device
                )
                targets = torch.as_tensor(y_validation, dtype=torch.float32)
                record = dict(split="validation", epoch=epoch, train_bce={}, validation_bce={})
                for arm in ARMS:
                    value = float(F.binary_cross_entropy_with_logits(
                        torch.from_numpy(validation_logits[arm]), targets
                    ))
                    record["train_bce"][arm] = totals[arm] / rows
                    record["validation_bce"][arm] = value
                    if value < best[arm]["value"]:
                        best[arm] = dict(value=value, epoch=epoch, state=_state_copy(heads[arm]))
                history.append(record)
                frozen.dump(directory / "history.json", history)
                print(json.dumps(dict(
                    state="multiseed_head_train", adapter_seed=adapter_seed,
                    head_seed=head_seed, epoch=epoch, epochs=training["epochs"],
                )), flush=True)
            selected = choose_validation_checkpoints(history)
            for arm in ARMS:
                if selected[arm]["epoch"] != best[arm]["epoch"] or best[arm]["state"] is None:
                    raise AssertionError("Validation checkpoint tracking diverged")
                heads[arm].load_state_dict(best[arm]["state"])
            validation_logits, validation_weights = _predict_heads(
                heads, validation_cache, training["batch_size"], device
            )
            prediction_frame = validation_metadata.copy()
            arm_records = {}
            for arm in ARMS:
                checkpoint_path = directory / f"{arm}.pt"
                checkpoint = dict(
                    state_dict=best[arm]["state"], adapter_seed=int(adapter_seed),
                    head_seed=int(head_seed), arm=arm, layers=list(FIXED_LAYERS),
                    hidden_size=int(train_cache.shape[-1]), width=protocol["architecture"]["width"],
                    gate_width=protocol["architecture"]["gate_width"],
                    dropout=protocol["architecture"]["dropout"],
                    fusion_init_scale=protocol["architecture"]["fusion_init_scale"],
                    adapter_sha256=protocol["adapters"][str(adapter_seed)]["adapter_sha256"],
                    best_epoch=int(best[arm]["epoch"]),
                    best_validation_bce=float(best[arm]["value"]),
                    protocol_sha256=frozen.sha256(output / "protocol.json"),
                )
                frozen.atomic_torch_save(checkpoint, checkpoint_path)
                probability = frozen.sigmoid(validation_logits[arm])
                prediction_frame[f"probability_{arm}"] = probability
                prediction_frame[f"prediction_{arm}"] = (probability >= PRIMARY_THRESHOLD).astype(np.int8)
                arm_records[arm] = dict(
                    parameters=sum(parameter.numel() for parameter in heads[arm].parameters()),
                    best_epoch=int(best[arm]["epoch"]), validation_bce=float(best[arm]["value"]),
                    threshold=PRIMARY_THRESHOLD,
                    validation_metrics=_metrics(y_validation, probability),
                    gate=dynamic.gate_diagnostics(
                        validation_weights[arm], FIXED_LAYERS if arm == "dynamic_fixed" else (32, 32, 32, 32),
                        y_validation,
                    ),
                    checkpoint_sha256=frozen.sha256(checkpoint_path),
                )
            predictions_path = directory / "validation_predictions.parquet"
            prediction_frame.to_parquet(predictions_path, index=False)
            selection = dict(
                split="validation", adapter_seed=int(adapter_seed), head_seed=int(head_seed),
                layers=list(FIXED_LAYERS), arms=arm_records, matched_initialization=initialization,
                common_batch_order=True, paired_dropout_stream=True,
                validation_predictions_sha256=frozen.sha256(predictions_path),
            )
            frozen.dump(selection_path, selection)
            summaries.append(selection)
            del heads, optimizers
            if device.type == "cuda":
                torch.cuda.empty_cache()
    expected = len(protocol["adapter_seeds"]) * len(training["head_seeds"])
    if len(summaries) != expected:
        raise RuntimeError("Replication head grid is incomplete")
    summary = dict(
        split="validation", entries=summaries, adapter_seeds=protocol["adapter_seeds"],
        head_seeds=training["head_seeds"], arms=list(ARMS), threshold=PRIMARY_THRESHOLD,
        statement="Every arm pair has identical topology/initialization/training streams; only cached layer inputs differ.",
    )
    frozen.dump(output / "heads/selection.json", summary)
    return summary


def lock_test(args: argparse.Namespace) -> dict:
    output, protocol = Path(args.output).resolve(), load_protocol(args.output)
    if (output / "test_barrier.json").is_file():
        return verify_test_barrier(output, protocol)
    summary_path = output / "heads/selection.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Train all validation-selected heads before locking test")
    summary = _json(summary_path)
    expected = len(protocol["adapter_seeds"]) * len(protocol["training"]["head_seeds"])
    if summary.get("split") != "validation" or len(summary.get("entries", [])) != expected:
        raise ValueError("Head grid is not a complete validation-only selection")
    entries = {
        (int(entry.get("adapter_seed", -1)), int(entry.get("head_seed", -1))): entry
        for entry in summary["entries"]
    }
    expected_keys = {
        (int(adapter_seed), int(head_seed))
        for adapter_seed in protocol["adapter_seeds"]
        for head_seed in protocol["training"]["head_seeds"]
    }
    if set(entries) != expected_keys:
        raise ValueError("Head summary has missing or duplicate adapter/head seed cells")
    for adapter_seed, head_seed in sorted(expected_keys):
        selection = _validate_head_selection(output, protocol, adapter_seed, head_seed)
        if selection != entries[(adapter_seed, head_seed)]:
            raise ValueError("Head summary differs from a validation selection file")
    if _metadata_path(output, "test").exists() or any(
        any(path.exists() for path in _cache_paths(output, adapter_seed, "test"))
        for adapter_seed in protocol["adapter_seeds"]
    ):
        raise RuntimeError("Test feature artifacts existed before the replication barrier")
    test_dataset = Path(protocol["data_dir"]) / "test.parquet"
    if frozen.sha256(test_dataset) != protocol["data_sha256"]["test"]:
        raise ValueError("Registered test dataset changed before barrier creation")
    artifacts = _barrier_artifacts(output, protocol)
    artifacts["heads/summary"] = summary_path
    # _barrier_artifacts must remain identical for later verification.
    barrier = create_replication_test_barrier(
        output, artifacts, _barrier_identity(output, protocol),
        test_dataset,
    )
    return barrier


def _verified_barrier_artifacts(output: Path, protocol: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = _barrier_artifacts(output, protocol)
    artifacts["heads/summary"] = Path(output) / "heads/selection.json"
    return artifacts


def verify_test_barrier(output: Path, protocol: Mapping[str, Any] | None = None) -> dict:
    output = Path(output).resolve()
    protocol = load_protocol(output) if protocol is None else protocol
    barrier = verify_replication_test_barrier(
        output, _verified_barrier_artifacts(output, protocol), _barrier_identity(output, protocol),
        Path(protocol["data_dir"]) / "test.parquet",
    )
    if barrier.get("test_dataset_sha256") != protocol["data_sha256"]["test"]:
        raise ValueError("Barrier test dataset differs from registered protocol")
    return barrier


def _load_head(
    path: Path, protocol: Mapping[str, Any], adapter_seed: int, head_seed: int,
    arm: str, device: torch.device,
) -> dynamic.DynamicLayerGateHead:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    selection = _json(Path(path).parent / "selection.json")
    record = selection.get("arms", {}).get(arm, {})
    expected = dict(
        adapter_seed=int(adapter_seed), head_seed=int(head_seed), arm=arm,
        layers=list(FIXED_LAYERS), hidden_size=int(protocol["hidden_size"]),
        width=int(protocol["architecture"]["width"]),
        gate_width=int(protocol["architecture"]["gate_width"]),
        dropout=float(protocol["architecture"]["dropout"]),
        fusion_init_scale=float(protocol["architecture"]["fusion_init_scale"]),
        adapter_sha256=protocol["adapters"][str(adapter_seed)]["adapter_sha256"],
        best_epoch=int(record.get("best_epoch", -1)),
        best_validation_bce=float(record.get("validation_bce", float("nan"))),
        protocol_sha256=frozen.sha256(Path(path).parents[3] / "protocol.json"),
    )
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Replication head identity changed: {adapter_seed}/{head_seed}/{arm}/{key}")
    head = dynamic.DynamicLayerGateHead(
        protocol["hidden_size"], protocol["architecture"]["width"],
        protocol["architecture"]["gate_width"],
        dropout=protocol["architecture"]["dropout"],
        fusion_init_scale=protocol["architecture"]["fusion_init_scale"],
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True); head.eval()
    return head


def _metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("Aligned finite probabilities are required")
    prediction = probabilities >= PRIMARY_THRESHOLD
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    result = frozen.binary_metrics(labels, probabilities, PRIMARY_THRESHOLD)
    result.update(
        balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
        bce=float(log_loss(labels, clipped, labels=[0, 1])),
        auroc=float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
    )
    return result


def _gain(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    return {
        "accuracy": float(candidate["accuracy"] - baseline["accuracy"]),
        "balanced_accuracy": float(candidate["balanced_accuracy"] - baseline["balanced_accuracy"]),
        "bce": float(candidate["bce"] - baseline["bce"]),
        "auroc": float(candidate["auroc"] - baseline["auroc"]),
    }


def verify_completed_results(output: Path, protocol: Mapping[str, Any] | None = None) -> dict:
    output = Path(output).resolve()
    protocol = load_protocol(output) if protocol is None else protocol
    verify_test_barrier(output, protocol)
    result_path, report_path = output / "results.json", output / "report.md"
    completion_path = output / "completion.json"
    if not result_path.is_file() or not report_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("Replication result completion artifacts are incomplete")
    completion = _json(completion_path)
    if (
        completion.get("schema") != "hanguard_multiseed_fixed_layer_completion_v1"
        or completion.get("state") != "complete_no_test_retuning"
        or completion.get("protocol_sha256") != frozen.sha256(output / "protocol.json")
        or completion.get("barrier_sha256") != frozen.sha256(output / "test_barrier.json")
        or completion.get("results_sha256") != frozen.sha256(result_path)
        or completion.get("report_sha256") != frozen.sha256(report_path)
    ):
        raise ValueError("Replication completion identity changed")
    result = _json(result_path)
    if set(result.get("adapters", {})) != set(completion.get("prediction_sha256", {})):
        raise ValueError("Replication completed adapter set changed")
    for seed, values in result["adapters"].items():
        prediction = (output / values["predictions"]).resolve()
        if not prediction.is_relative_to(output) or not prediction.is_file():
            raise ValueError(f"Replication prediction path changed: {seed}")
        digest = frozen.sha256(prediction)
        if digest != values.get("predictions_sha256") or digest != completion["prediction_sha256"][seed]:
            raise ValueError(f"Replication predictions changed: adapter_s{seed}")
    return result


def evaluate(args: argparse.Namespace) -> dict:
    output, protocol = Path(args.output).resolve(), load_protocol(args.output)
    barrier = verify_test_barrier(output, protocol)
    result_path = output / "results.json"
    if result_path.is_file():
        return verify_completed_results(output, protocol)
    device = torch.device(args.device)
    repetitions, analysis_seed = protocol["analysis"]["repetitions"], protocol["analysis"]["seed"]
    adapter_results, adapter_gains = {}, {}
    for adapter_index, adapter_seed in enumerate(protocol["adapter_seeds"]):
        cache, metadata, cache_manifest = load_feature_cache(output, protocol, adapter_seed, "test")
        labels = metadata.label.to_numpy(dtype=np.int64)
        groups = metadata.group_id.astype(str).to_numpy()
        probabilities_by_arm = {arm: [] for arm in ARMS}
        seed_results = {}
        prediction_frame = metadata.copy()
        for head_seed in protocol["training"]["head_seeds"]:
            directory = _head_directory(output, adapter_seed, head_seed)
            heads = {
                arm: _load_head(directory / f"{arm}.pt", protocol, adapter_seed, head_seed, arm, device)
                for arm in ARMS
            }
            logits, weights = _predict_heads(
                heads, cache, protocol["training"]["batch_size"], device
            )
            metrics, seed_probability = {}, {}
            for arm in ARMS:
                probability = frozen.sigmoid(logits[arm]); seed_probability[arm] = probability
                probabilities_by_arm[arm].append(probability)
                prediction_frame[f"probability_seed_{head_seed}_{arm}"] = probability
                metrics[arm] = _metrics(labels, probability)
            candidate, baseline = seed_probability["dynamic_fixed"], seed_probability["dynamic_repeat_last"]
            seed_results[str(head_seed)] = dict(
                metrics=metrics, gain=_gain(metrics["dynamic_fixed"], metrics["dynamic_repeat_last"]),
                mcnemar=frozen.mcnemar_exact(
                    labels, candidate >= PRIMARY_THRESHOLD, baseline >= PRIMARY_THRESHOLD
                ),
                error_complementarity=frozen.error_complementarity(
                    labels, candidate, baseline, PRIMARY_THRESHOLD, PRIMARY_THRESHOLD
                ),
                gate={
                    arm: dynamic.gate_diagnostics(
                        weights[arm], FIXED_LAYERS if arm == "dynamic_fixed" else (32, 32, 32, 32), labels
                    )
                    for arm in ARMS
                },
            )
            del heads
        ensemble = {arm: np.stack(values).mean(0) for arm, values in probabilities_by_arm.items()}
        ensemble_metrics = {arm: _metrics(labels, probability) for arm, probability in ensemble.items()}
        gain = _gain(ensemble_metrics["dynamic_fixed"], ensemble_metrics["dynamic_repeat_last"])
        adapter_gains[int(adapter_seed)] = gain
        for arm, probability in ensemble.items():
            prediction_frame[f"probability_ensemble_{arm}"] = probability
            prediction_frame[f"prediction_ensemble_{arm}"] = (probability >= PRIMARY_THRESHOLD).astype(np.int8)
        prediction_path = output / "predictions" / f"adapter_s{adapter_seed}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(prediction_path, index=False)
        candidate, baseline = ensemble["dynamic_fixed"], ensemble["dynamic_repeat_last"]
        adapter_results[str(adapter_seed)] = dict(
            rows=len(metadata), groups=int(metadata.group_id.nunique()),
            cache_sha256=cache_manifest["sha256"], head_seeds=seed_results,
            ensemble=dict(
                metrics=ensemble_metrics, gain=gain,
                gain_direction="dynamic_fixed minus dynamic_repeat_last; negative BCE is improvement",
                bootstrap=frozen.group_cluster_bootstrap(
                    labels, candidate, baseline, groups, PRIMARY_THRESHOLD, PRIMARY_THRESHOLD,
                    repetitions=repetitions, seed=analysis_seed + adapter_index,
                ),
                permutation=frozen.group_permutation_test(
                    labels, candidate, baseline, groups, PRIMARY_THRESHOLD, PRIMARY_THRESHOLD,
                    repetitions=repetitions, seed=analysis_seed + adapter_index,
                ),
                mcnemar=frozen.mcnemar_exact(
                    labels, candidate >= PRIMARY_THRESHOLD, baseline >= PRIMARY_THRESHOLD
                ),
                error_complementarity=frozen.error_complementarity(
                    labels, candidate, baseline, PRIMARY_THRESHOLD, PRIMARY_THRESHOLD
                ),
            ),
            head_seed_gain_distribution=head_seed_gain_distribution({
                int(seed): values["gain"] for seed, values in seed_results.items()
            }),
            predictions=str(prediction_path.relative_to(output)),
            predictions_sha256=frozen.sha256(prediction_path),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = dict(
        schema="hanguard_multiseed_fixed_layer_results_v1",
        protocol_sha256=frozen.sha256(output / "protocol.json"),
        test_barrier_sha256=frozen.sha256(output / "test_barrier.json"),
        threshold=PRIMARY_THRESHOLD, layers=list(FIXED_LAYERS),
        adapters=adapter_results, cross_adapter=adapter_gain_distribution(adapter_gains),
        test_dataset_sha256=barrier["test_dataset_sha256"],
        interpretation=(
            "The cross-adapter distribution uses three independently trained E03 LoRA seeds. "
            "Five head seeds per adapter measure conditional head variance and are not treated as LoRA replicates."
        ),
    )
    frozen.dump(result_path, result)
    report_path = write_report(output, result)
    frozen.dump(output / "completion.json", dict(
        schema="hanguard_multiseed_fixed_layer_completion_v1",
        state="complete_no_test_retuning",
        protocol_sha256=result["protocol_sha256"],
        barrier_sha256=result["test_barrier_sha256"],
        results_sha256=frozen.sha256(result_path), report_sha256=frozen.sha256(report_path),
        prediction_sha256={
            seed: values["predictions_sha256"] for seed, values in adapter_results.items()
        },
    ))
    return result


def write_report(output: Path, result: Mapping[str, Any] | None = None) -> Path:
    output = Path(output).resolve()
    result = verify_completed_results(output) if result is None else result
    lines = [
        "# Cross-LoRA-seed fixed-layer replication", "",
        "固定抽取 L8/L16/L24/L32；dynamic_fixed 与四槽均重复 L32 的 control 使用完全相同的 topology、容量、初始化和训练流。主阈值固定为 0.5。", "",
        "| E03 adapter seed | Δ accuracy (pp) | Δ balanced accuracy (pp) | Δ BCE | group bootstrap accuracy 95% CI (pp) | group permutation p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, values in result["adapters"].items():
        ensemble = values["ensemble"]
        gain, bootstrap, permutation = ensemble["gain"], ensemble["bootstrap"]["accuracy"], ensemble["permutation"]["accuracy"]
        lines.append(
            f"| {seed} | {100 * gain['accuracy']:.3f} | {100 * gain['balanced_accuracy']:.3f} | "
            f"{gain['bce']:.6f} | [{100 * bootstrap['ci95'][0]:.3f}, {100 * bootstrap['ci95'][1]:.3f}] | "
            f"{permutation['two_sided_p']:.6g} |"
        )
    cross = result["cross_adapter"]["metrics"]
    lines.extend([
        "", "## Cross-adapter descriptive distribution", "",
        f"Adapter-level ensemble Δ accuracy: mean={100 * cross['accuracy']['mean']:.3f}pp, "
        f"range=[{100 * cross['accuracy']['minimum']:.3f}, {100 * cross['accuracy']['maximum']:.3f}]pp, "
        f"improved adapters={cross['accuracy']['improved_fraction']:.3f}.",
        "",
        f"Adapter-level ensemble Δ BCE (fixed−repeat; lower is better): mean={cross['bce']['mean']:.6f}, "
        f"range=[{cross['bce']['minimum']:.6f}, {cross['bce']['maximum']:.6f}], "
        f"improved adapters={cross['bce']['improved_fraction']:.3f}.",
        "", result["interpretation"], "",
    ])
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("init", "extract-pretest", "train", "lock", "extract-test", "evaluate", "report", "run-pretest", "run-test"),
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-seeds", nargs="+", type=int, default=list(DEFAULT_HEAD_SEEDS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gate-width", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=24576)
    parser.add_argument("--max-micro", type=int, default=64)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--status-every", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=424242)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.epochs, args.batch_size, args.lr, args.width, args.gate_width,
        args.token_budget, args.max_micro, args.pad_multiple, args.status_every, args.repetitions,
    )
    if any(value <= 0 for value in positive) or args.weight_decay < 0:
        raise ValueError("Replication settings must be positive (weight decay may be zero)")
    if args.action in ("extract-pretest", "train", "extract-test", "evaluate", "run-pretest", "run-test"):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested replication stage")
    if args.action == "init":
        result = initialize(args)
    elif args.action == "extract-pretest":
        result = extract_splits(args, ("train", "validation"))
    elif args.action == "train":
        result = train_heads(args)
    elif args.action == "lock":
        result = lock_test(args)
    elif args.action == "extract-test":
        result = extract_splits(args, ("test",))
    elif args.action == "evaluate":
        result = evaluate(args)
    elif args.action == "report":
        result = dict(report=str(write_report(args.output)))
    elif args.action == "run-pretest":
        initialize(args); extract_splits(args, ("train", "validation")); result = train_heads(args)
    else:
        lock_test(args); extract_splits(args, ("test",)); result = evaluate(args)
    compact = result if args.action in ("init", "lock", "report") else {"state": "complete", "action": args.action}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
