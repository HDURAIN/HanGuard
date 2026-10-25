"""CPU checks for unchanged science, parallel phase routing and the test barrier."""
import argparse
import fcntl
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.hanguard import frozen_base_parallel_study as parallel
from scripts.hanguard import frozen_base_parallel_features as features
from scripts.hanguard import frozen_base_parallel_queue as queue
from scripts.hanguard import frozen_base_readout_heads as heads


def tiny_inputs(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    parallel.base.dump(model / "config.json", dict(model_type="qwen3_5",
        text_config=dict(num_hidden_layers=32, hidden_size=3)))
    parallel.base.dump(model / "model.safetensors.index.json", dict(weight_map={"weight": "model.safetensors"}))
    (model / "model.safetensors").write_bytes(b"CPU registration fixture")
    for split in parallel.base.SPLITS:
        pd.DataFrame(dict(base_id=[split + "_id"], group_id=[split + "_group"],
            source=["wildguard_zh"], prompt=["CPU text"], prompt_harm_label=["unharmful"],
            category_id=[0], type_label_mask=[False], split=[split])).to_parquet(data / f"{split}.parquet", index=False)
    parallel.base.dump(data / "manifest.json", dict(dataset_contract=parallel.base.contract.CONTRACT,
        dataset_version="CPU_registration_fixture"))
    parallel.base.dump(data / "split_policy.json", {})
    parallel.base.dump(data / "release_qa.json", dict(release_status="qa_complete_with_documented_limitations",
        manifest_sha256=parallel.base.sha(data / "manifest.json"),
        split_sha256={s: parallel.base.sha(data / f"{s}.parquet") for s in parallel.base.SPLITS}))
    return model, data


def test_parallel_registration_keeps_serial_scientific_comparison(tmp_path, monkeypatch):
    model, data = tiny_inputs(tmp_path)
    original = pd.read_parquet
    test_columns = []
    def observed_read(path, **kwargs):
        if Path(path).name == "test.parquet":
            test_columns.append(kwargs.get("columns"))
        return original(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", observed_read)
    serial = parallel.base.initialize(argparse.Namespace(output=tmp_path / "serial", model=model, data=data))
    new = parallel.initialize(argparse.Namespace(output=tmp_path / "parallel", model=model, data=data))
    changed = {key for key in set(serial) | set(new) if serial.get(key) != new.get(key)}
    assert changed == {"code_sha256", "execution", "parallel_features", "created_unix"}
    assert new["execution"]["max_cuda_workers"] == 8
    assert new["execution"]["reserved_gpus"] == []
    assert new["execution"]["automatic_feature_redispatch"] is True
    assert new["execution"]["automatic_head_retry"] is False
    assert new["parallel_features"]["group_chunk_size"] == 32
    assert test_columns == [["base_id", "group_id"], ["base_id", "group_id"]]
    assert parallel.base.load_protocol(tmp_path / "parallel", full_hash=True) == new


def fake_phases(tmp_path, monkeypatch, *, fitted_lock=True):
    study = tmp_path / "study"
    study.mkdir()
    parallel.base.dump(study / "state.json", dict(state="registered", training_started=False))
    parallel.base.dump(study / "protocol.json", {})
    events, locked = [], False
    monkeypatch.setattr(parallel.base, "load_protocol", lambda *args, **kwargs: {})
    monkeypatch.setattr(queue, "initialize", lambda *args: events.append(("resource",)))
    def require_lock(path):
        events.append(("lock",))
        if not locked:
            raise ValueError("No completed selection lock")
        return {}
    monkeypatch.setattr(parallel.base, "validate_selection_lock", require_lock)
    def prepare(path, split):
        if split == "test":
            require_lock(path)
        events.append(("prepare", split))
    monkeypatch.setattr(features, "prepare_split", prepare)
    monkeypatch.setattr(features, "split_complete", lambda *args: True)
    monkeypatch.setattr(features, "has_work", lambda *args: True)
    monkeypatch.setattr(features, "merge_split", lambda path, split: events.append(("merge", split)))
    monkeypatch.setattr(features, "abandon_worker", lambda *args: events.append(("abandon",)))
    def workers(path, phase, factory, done, **kwargs):
        events.append(("workers", phase, kwargs["max_workers"]))
        assert kwargs["max_workers"] == 8
        command = factory("fixture_worker", 2)
        assert "--study" in command and command[command.index("--study") + 1] == str(study)
        assert command[command.index("--split") + 1] == kwargs["split"]
        assert "on_resource_loss" in kwargs
        assert done() is True
    monkeypatch.setattr(queue, "run_workers", workers)
    def single(path, phase, factory):
        nonlocal locked
        events.append(("single", phase))
        command = factory("fixture_head", 4)
        assert command[command.index("--phase") + 1] == phase
        assert "--study" in command
        if phase == "fit":
            locked = fitted_lock
        else:
            require_lock(path)
            parallel.base.dump(path / "analysis/results.json", dict(complete=True))
    monkeypatch.setattr(queue, "run_single", single)
    return study, events


def test_orchestrator_prepares_test_only_after_readout_lock(tmp_path, monkeypatch):
    study, events = fake_phases(tmp_path, monkeypatch)
    parallel.run(study)
    assert events.index(("single", "fit")) < events.index(("prepare", "test"))
    assert events.index(("merge", "train")) < events.index(("single", "fit"))
    assert events.index(("merge", "validation")) < events.index(("single", "fit"))
    assert events.index(("merge", "test")) < events.index(("single", "evaluate"))
    assert json.loads((study / "state.json").read_text())["state"] == "completed"


def test_missing_selection_lock_prevents_even_test_preparation(tmp_path, monkeypatch):
    study, events = fake_phases(tmp_path, monkeypatch, fitted_lock=False)
    with pytest.raises(ValueError, match="selection lock"):
        parallel.run(study)
    assert ("prepare", "test") not in events
    assert ("single", "evaluate") not in events


def test_no_second_coordinator_or_implicit_restart(tmp_path, monkeypatch):
    study, _ = fake_phases(tmp_path, monkeypatch)
    with (study / "parallel_coordinator.pid").open("a+") as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another coordinator"):
            parallel.run(study)
    parallel.base.mark_state(study, "requires_review")
    with pytest.raises(RuntimeError, match="already started"):
        parallel.run(study)


def test_head_worker_cuda_handshake_precedes_full_hash_and_restores_stop(tmp_path, monkeypatch):
    import torch
    study = tmp_path / "study"
    study.mkdir()
    parallel.base.dump(study / "protocol.json", {})
    parallel.base.dump(study / "state.json", {})
    events = []
    def protocol(path, **kwargs):
        events.append("full_hash" if kwargs.get("full_hash") else "protocol")
        return {}
    monkeypatch.setattr(parallel.base, "load_protocol", protocol)
    monkeypatch.setattr(features, "cuda_startup", lambda *a, **k: events.append("cuda_handshake"))
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    original_stop = heads.check_stop
    monkeypatch.setattr(features, "check_stop", lambda *a: None)
    def fit(path, **kwargs):
        events.append("fit")
        heads.check_stop(path)
    monkeypatch.setattr(heads, "fit", fit)
    monkeypatch.setattr(parallel.base, "validate_selection_lock", lambda *a: events.append("lock"))
    parallel.head_worker(study, "fixture", "fit")
    assert events.index("cuda_handshake") < events.index("full_hash") < events.index("fit")
    assert heads.check_stop is original_stop


def test_head_worker_stop_checked_before_training_and_restored(tmp_path, monkeypatch):
    import torch
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(parallel.base, "load_protocol", lambda *a, **k: {})
    monkeypatch.setattr(features, "cuda_startup", lambda *a, **k: {})
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    original_stop = heads.check_stop
    def stopped(*args):
        raise RuntimeError("own worker STOP")
    monkeypatch.setattr(features, "check_stop", stopped)
    monkeypatch.setattr(heads, "fit", lambda *a, **k: pytest.fail("Stopped worker cannot train"))
    with pytest.raises(RuntimeError, match="own worker STOP"):
        parallel.head_worker(study, "fixture", "fit")
    assert heads.check_stop is original_stop
