"""Independent CPU checks for capture, immutable inputs, and test barriers.

The backbone here only copies integer inputs and adds constants. No real model
or GPU is loaded, and no experiment head is trained by these tests.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts.hanguard import frozen_base_readout_study as study_module


class ConstantBlock(nn.Module):
    def __init__(self, increment, tuple_output=False):
        super().__init__()
        self.increment = increment
        self.tuple_output = tuple_output

    def forward(self, hidden):
        result = hidden + self.increment
        return (result,) if self.tuple_output else result


class CopyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ConstantBlock(1), ConstantBlock(2, True)])

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        assert use_cache is False
        assert torch.equal(position_ids, (attention_mask.cumsum(-1) - 1).clamp_min(0))
        hidden = input_ids.float()[..., None].repeat(1, 1, 3)
        for layer in self.layers:
            hidden = layer(hidden)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return hidden


def write_json(path, value):
    study_module.dump(path, value)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    source = tmp_path / "implementation.py"
    source.write_text("implementation version one\n")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model-one")
    runtime = {"python": "CPU fixture", "versions": {"torch": "fixture"}}
    monkeypatch.setattr(study_module, "runtime_identity", lambda: runtime)
    stat = model.stat()
    protocol = dict(schema=study_module.SCHEMA, use_lora=False, adapter_path=None,
                    runtime=runtime, code_sha256={str(source): study_module.sha(source)},
                    inputs={str(model): study_module.sha(model)},
                    input_stats={str(model): dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)})
    write_json(root / "protocol.json", protocol)
    write_json(root / "protocol_registration.json",
               dict(protocol_sha256=study_module.sha(root / "protocol.json")))
    return root, source, model, protocol


def make_lock(root):
    if not (root / "protocol.json").exists():
        write_json(root / "protocol.json", {"fixture": True})
    head = root / "selected_head.pt"
    head.write_bytes(b"selected checkpoint")
    lock = dict(complete=True, test_opened=False, protocol_sha256=study_module.sha(root / "protocol.json"),
                artifacts={head.name: study_module.sha(head)})
    write_json(root / "selection_lock.json", lock)
    return head, lock


def test_capture_uses_last_valid_position_for_both_padding_sides_and_holes():
    backbone = CopyBackbone()
    capture = study_module.LastBlockCapture(backbone)
    try:
        batch = dict(input_ids=torch.tensor([[10, 20, 99, 99], [99, 99, 30, 40], [50, 99, 60, 99]]),
                     attention_mask=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]))
        result = capture(batch)
        expected = torch.tensor([[[21] * 3, [23] * 3], [[41] * 3, [43] * 3],
                                 [[61] * 3, [63] * 3]], dtype=torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "cpu"
        assert torch.equal(result, expected)
        assert torch.isfinite(result).all()
        assert capture.indices is None and capture.values == {}
    finally:
        capture.close()
    assert all(not block._forward_hooks for block in backbone.layers)


def test_capture_clears_temporary_state_when_backbone_raises(monkeypatch):
    backbone = CopyBackbone()
    capture = study_module.LastBlockCapture(backbone)
    monkeypatch.setattr(backbone, "forward", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    try:
        with pytest.raises(RuntimeError, match="fixture failure"):
            capture(dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]])))
        assert capture.indices is None and capture.values == {}
    finally:
        capture.close()


def test_exact_replay_checks_same_batch_finiteness_and_saved_values(monkeypatch):
    monkeypatch.setattr(study_module.legacy, "make_batch", lambda data, indices, tokenizer, extraction: indices)
    data = dict(lengths=np.array([1, 2, 3]), meta=pd.DataFrame({"base_id": ["a", "b", "c"]}))
    groups = [np.array([0, 1]), np.array([2])]
    expected = torch.arange(18, dtype=torch.float16).reshape(3, 2, 3)
    capture = lambda rows: expected[rows].clone()
    good = study_module.exact_replay(capture, data, None, {}, groups, expected.numpy())
    assert good["passed"] is True
    assert good["batch_shape_equivalence_claimed"] is False
    corrupt = expected.numpy().copy()
    corrupt[0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="saved features"):
        study_module.exact_replay(capture, data, None, {}, groups, corrupt)
    with pytest.raises(RuntimeError, match="finite"):
        study_module.exact_replay(lambda rows: torch.full((len(rows), 2, 3), float("nan")),
                                  data, None, {}, groups)


@pytest.mark.parametrize("lengths", ([0], [4097]))
def test_tokenization_forbids_empty_inputs_and_silent_truncation(lengths):
    frame = pd.DataFrame(dict(prompt=["fixture prompt"], prompt_harm_label=["harmful"],
                              base_id=["a"], group_id=["g"], source=["wildguard_zh"]))
    calls = []
    def tokenizer(prompts, **kwargs):
        calls.append(kwargs)
        return {"input_ids": [[1] * lengths[0]]}
    with pytest.raises(ValueError, match="truncation is forbidden"):
        study_module.make_data(frame, tokenizer, dict(extraction=dict(max_tokens=4096)))
    assert calls == [dict(add_special_tokens=False, truncation=False)]


@pytest.mark.parametrize("problem", ("missing_ledger", "birth", "same_uid_foreign", "unknown_card"))
def test_guard_child_admission_rejects_direct_runs_pid_reuse_and_foreign_owners(tmp_path, monkeypatch, problem):
    from scripts.hanguard import ab_readout_parallel_queue as inventory
    expected = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=123,
                    command_sha256="registered command", argv=["registered runner"], cwd=str(tmp_path))
    actual = dict(expected)
    ledger = tmp_path / "resource_attempt_001/execution/child_identity_frozen_base_pipeline.json"
    if problem != "missing_ledger":
        write_json(ledger, expected)
    if problem == "birth":
        actual["start_time_ticks"] += 1
    monkeypatch.setattr(inventory, "process_identity", lambda pid: actual)
    cards = [] if problem == "unknown_card" else [dict(index=2, pids=[os.getpid()] +
        ([7654321] if problem == "same_uid_foreign" else []))]
    monkeypatch.setattr(inventory, "gpu_inventory", lambda: cards)
    with pytest.raises(ValueError):
        study_module.verify_guardian_child(tmp_path, "2")


def test_guard_child_accepts_only_current_registered_birth_on_empty_card(tmp_path, monkeypatch):
    from scripts.hanguard import ab_readout_parallel_queue as inventory
    expected = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=123,
                    command_sha256="registered command", argv=["registered runner"], cwd=str(tmp_path))
    write_json(tmp_path / "resource_attempt_001/execution/child_identity_frozen_base_pipeline.json", expected)
    monkeypatch.setattr(inventory, "process_identity", lambda pid: dict(expected))
    monkeypatch.setattr(inventory, "gpu_inventory", lambda: [dict(index=2, pids=[])])
    assert study_module.verify_guardian_child(tmp_path, "2") == expected


@pytest.mark.parametrize("full_hash_ok", (True, False))
def test_worker_initializes_guarded_CUDA_before_full_hash_and_uses_data_only_after_hash(tmp_path, monkeypatch, full_hash_ok):
    """Exercise the startup order with CPU fakes, never touching CUDA."""
    from scripts.hanguard import frozen_base_readout_heads as heads
    root = tmp_path / "study"
    root.mkdir()
    protocol = {"fixture": "registered frozen base"}
    write_json(root / "protocol.json", protocol)
    events, gate = [], {"full_hash_passed": False, "locked": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")

    def load_protocol(study, *, full_hash=False):
        assert Path(study) == root
        if full_hash:
            assert "cuda_init" in events and "bind_guard_child" in events
            assert "cuda_allocate" in events and "cuda_synchronize" in events
            if not full_hash_ok:
                events.append("full_hash_failed")
                raise ValueError("Full content hash fixture failure")
            gate["full_hash_passed"] = True
        events.append("full_hash" if full_hash else "light_protocol")
        return protocol

    def bind(study):
        assert events == ["light_protocol"]
        events.append("bind_guard_child")

    def cuda_init():
        assert events[-1] == "bind_guard_child"
        events.append("cuda_init")

    def count():
        assert "cuda_init" in events
        events.append("device_count")
        return 1

    def name(index):
        assert index == 0 and "cuda_init" in events
        events.append("device_name")
        return "CPU mocked CUDA device"

    def zero(*args, **kwargs):
        assert args == (1,) and kwargs == {"device": "cuda"}
        assert "cuda_init" in events and not gate["full_hash_passed"]
        events.append("cuda_allocate")
        return torch.tensor([0.], device="cpu")

    def synchronize():
        assert events[-1] == "cuda_allocate"
        events.append("cuda_synchronize")

    def extract(study, selected_protocol, split):
        assert gate["full_hash_passed"], "Model/data use preceded full hash verification"
        if split == "test":
            assert gate["locked"], "Test extraction preceded selection locking"
        events.append(f"extract_{split}")

    def fit(study, device):
        assert device == "cuda" and gate["full_hash_passed"]
        events.append("fit_heads")

    def lock(study):
        assert "fit_heads" in events
        gate["locked"] = True
        events.append("selection_lock")

    def evaluate(study, device):
        assert device == "cuda" and gate["locked"]
        events.append("evaluate")
        write_json(root / "analysis/results.json", dict(complete=True))

    monkeypatch.setattr(study_module, "load_protocol", load_protocol)
    monkeypatch.setattr(study_module, "bind_gpu", bind)
    monkeypatch.setattr(study_module.torch.cuda, "init", cuda_init)
    monkeypatch.setattr(study_module.torch.cuda, "device_count", count)
    monkeypatch.setattr(study_module.torch.cuda, "get_device_name", name)
    monkeypatch.setattr(study_module.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(study_module.torch, "zeros", zero)
    monkeypatch.setattr(study_module.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(study_module, "extract_split", extract)
    monkeypatch.setattr(study_module, "validate_selection_lock", lock)
    monkeypatch.setattr(study_module, "mark_state", lambda study, state, **kwargs: events.append(f"state_{state}"))
    monkeypatch.setattr(heads, "fit", fit)
    monkeypatch.setattr(heads, "evaluate", evaluate)
    if not full_hash_ok:
        with pytest.raises(ValueError, match="Full content hash fixture failure"):
            study_module.run(root)
        assert "cuda_allocate" in events and "cuda_synchronize" in events
        assert not any(event.startswith("extract_") for event in events)
        assert "fit_heads" not in events and "evaluate" not in events
        return
    study_module.run(root)
    assert events.index("bind_guard_child") < events.index("cuda_init") < events.index("full_hash")
    assert events.index("full_hash") < events.index("extract_train") < events.index("fit_heads")
    assert events.index("selection_lock") < events.index("extract_test") < events.index("evaluate")
    startup = json.loads((root / "cuda_startup_preflight.json").read_text())
    assert startup["passed"] is True and startup["pid"] == os.getpid()
    assert startup["physical_gpu"] == 6 and startup["visible_device_count"] == 1
    assert startup["model_or_data_used_before_full_hash"] is False
    assert startup["actual_cuda_allocation_verified"] is True


def test_CUDA_startup_rejects_multiple_visible_devices_before_any_model_or_data(tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    calls = []
    monkeypatch.setattr(study_module, "load_protocol", lambda *args, **kwargs: {})
    monkeypatch.setattr(study_module, "bind_gpu", lambda _: None)
    monkeypatch.setattr(study_module.torch.cuda, "init", lambda: calls.append("cuda_init"))
    monkeypatch.setattr(study_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(study_module.torch, "zeros", lambda *args, **kwargs: pytest.fail("Allocated CUDA before rejecting visibility"))
    monkeypatch.setattr(study_module, "extract_split", lambda *args: pytest.fail("Used model/data after invalid CUDA visibility"))
    with pytest.raises(ValueError, match="one registered CUDA device"):
        study_module.run(root)
    assert calls == ["cuda_init"]


@pytest.mark.parametrize("level", ("study", "parent"))
def test_STOP_rejects_protocol_loading_before_training(registered, level):
    root, _, _, _ = registered
    (root if level == "study" else root.parent).joinpath("STOP").write_text("User stop fixture")
    with pytest.raises(RuntimeError, match="User stop fixture"):
        study_module.load_protocol(root)


def test_protocol_and_source_hashes_fail_closed(registered):
    root, source, _, protocol = registered
    assert study_module.load_protocol(root) == protocol
    source.write_text("implementation version two\n")
    with pytest.raises(ValueError, match="implementation changed"):
        study_module.load_protocol(root)


def test_protocol_registration_rejects_modified_protocol(registered):
    root, _, _, protocol = registered
    write_json(root / "protocol.json", dict(protocol, extra="changed"))
    with pytest.raises(ValueError, match="protocol changed"):
        study_module.load_protocol(root)


def test_full_content_hash_catches_equal_size_equal_mtime_model_change(registered):
    root, _, model, _ = registered
    before = model.stat()
    model.write_bytes(b"model-two")
    os.utime(model, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert model.stat().st_size == before.st_size
    with pytest.raises(ValueError, match="input bytes changed"):
        study_module.load_protocol(root, full_hash=True)


def test_runtime_change_rejects_protocol(registered, monkeypatch):
    root, _, _, _ = registered
    monkeypatch.setattr(study_module, "runtime_identity", lambda: {"python": "different"})
    with pytest.raises(ValueError, match="runtime changed"):
        study_module.load_protocol(root)


@pytest.mark.parametrize("change", ("checkpoint", "protocol", "incomplete", "empty", "outside"))
def test_test_barrier_rejects_tampered_or_invalid_selection(tmp_path, change):
    root = tmp_path / "study"
    root.mkdir()
    head, lock = make_lock(root)
    assert study_module.validate_selection_lock(root) == lock
    if change == "checkpoint":
        head.write_bytes(b"altered checkpoint")
    elif change == "protocol":
        write_json(root / "protocol.json", {"altered": True})
    else:
        if change == "incomplete":
            lock["complete"] = False
        elif change == "empty":
            lock["artifacts"] = {}
        elif change == "outside":
            outside = root.parent / "outside.pt"
            outside.write_bytes(b"outside checkpoint")
            lock["artifacts"] = {str(outside): study_module.sha(outside)}
        write_json(root / "selection_lock.json", lock)
    with pytest.raises(ValueError):
        study_module.validate_selection_lock(root)


def test_test_payload_is_never_read_before_selection_barrier(registered, monkeypatch):
    root, _, _, protocol = registered
    read_calls = []
    def forbidden(*args, **kwargs):
        read_calls.append((args, kwargs))
        pytest.fail("Test payload was read before locking selection")
    monkeypatch.setattr(study_module.pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="locked"):
        study_module.load_frame(root, protocol, "test")
    with pytest.raises(ValueError, match="locked"):
        study_module.extract_split(root, protocol, "test")
    assert read_calls == []


def make_cache(root, shape=(2, 2, 3)):
    root.mkdir()
    folder = root / "caches"
    folder.mkdir()
    data_dir = root / "data"
    data_dir.mkdir()
    data_file = data_dir / "train.parquet"
    data_file.write_bytes(b"dataset bytes binding")
    protocol = dict(rows={"train": 2}, num_layers=2, hidden_size=3, data_dir=str(data_dir),
                    inputs={str(data_file): study_module.sha(data_file)})
    write_json(root / "protocol.json", protocol)
    array_path = folder / "train.npy"
    np.save(array_path, np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
    metadata_path = folder / "train.metadata.parquet"
    meta = pd.DataFrame(dict(base_id=["a", "b"], group_id=["ga", "gb"], source=["wildguard_zh"] * 2,
                             label=[0, 1], prompt_tokens=[1, 2], split=["train"] * 2,
                             category_id=[0, -1], type_label_mask=[False, False]))
    meta.to_parquet(metadata_path, index=False)
    replay_path = folder / "train.replay_audit.json"
    write_json(replay_path, dict(passed=True))
    manifest = dict(complete=True, protocol_sha256=study_module.sha(root / "protocol.json"),
                    adapter_sha256=None, adapter_path=None, backbone_trainable_parameters=0,
                    cohort="train", split="train", shape=list(shape), rows=shape[0], num_layers=shape[1],
                    hidden_size=shape[2], layers=list(range(1, shape[1] + 1)),
                    raw_block_outputs=True, truncation=False, pooling="last_valid_token", dtype="float16",
                    path=str(array_path), sha256=study_module.sha(array_path),
                    metadata_path=str(metadata_path), metadata_sha256=study_module.sha(metadata_path),
                    replay_audit_path=str(replay_path), replay_audit_sha256=study_module.sha(replay_path),
                    ordered_base_id_sha256=study_module.ordered_base_id_sha256(meta.base_id.tolist()),
                    input_sha256=protocol["inputs"][str(data_file)])
    write_json(folder / "train.manifest.json", manifest)
    return protocol, manifest


def test_fresh_cache_identity_and_numeric_gate(tmp_path):
    root = tmp_path / "study"
    protocol, _ = make_cache(root)
    assert study_module.cache_complete(root, protocol, "train") is True
    assert study_module.cache_complete(root, protocol, "validation") is False


@pytest.mark.parametrize("change", ("cohort", "protocol", "adapter", "shape", "input", "truncation", "raw"))
def test_cache_rejects_other_split_model_or_data_even_with_self_consistent_files(tmp_path, change):
    root = tmp_path / "study"
    protocol, manifest = make_cache(root, shape=(2, 3, 3) if change == "shape" else (2, 2, 3))
    if change == "cohort":
        manifest["cohort"] = "validation"
    elif change == "protocol":
        manifest["protocol_sha256"] = "0" * 64
    elif change == "adapter":
        manifest["adapter_sha256"] = "a" * 64
    elif change == "input":
        manifest["input_sha256"] = "b" * 64
    elif change == "truncation":
        manifest["truncation"] = True
    elif change == "raw":
        manifest["raw_block_outputs"] = False
    write_json(root / "caches/train.manifest.json", manifest)
    with pytest.raises(ValueError):
        study_module.cache_complete(root, protocol, "train")


def test_cache_rejects_nonfinite_values_and_extra_file_bytes(tmp_path):
    root = tmp_path / "study"
    protocol, manifest = make_cache(root)
    path = Path(manifest["path"])
    array = np.load(path)
    array[0, 0, 0] = np.nan
    np.save(path, array)
    manifest["sha256"] = study_module.sha(path)
    write_json(root / "caches/train.manifest.json", manifest)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        study_module.cache_complete(root, protocol, "train")
    array[0, 0, 0] = 0
    np.save(path, array)
    with path.open("ab") as stream:
        stream.write(b"unexpected trailing data")
    manifest["sha256"] = study_module.sha(path)
    write_json(root / "caches/train.manifest.json", manifest)
    with pytest.raises(ValueError, match="byte size"):
        study_module.cache_complete(root, protocol, "train")
