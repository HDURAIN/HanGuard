"""Numerical and isolation checks for the fresh frozen-base head experiment."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from scripts.hanguard import frozen_base_readout_heads as runner
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    dump, ordered_base_id_sha256, sha256, state_digest,
)
from scripts.hanguard.repaired_heads import ExperimentHead


@pytest.fixture(autouse=True)
def small_cpu_pool():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def cache_fixture(tmp_path, include_test=True):
    study = tmp_path / "study"
    (study / "caches").mkdir(parents=True)
    dump(study / "protocol.json", dict(num_layers=32, hidden_size=6, settings=dict(
        seed=42, width=5, dropout=0., batch_size=8, lr=.001,
        weight_decay=.007, grad_clip=.5, epochs=1, max_epochs=1,
        layers=[8, 16, 24, 32], bootstrap_draws=16)))
    for split, count in (("train", 24), ("validation", 12), ("test", 12)):
        if split == "test" and not include_test:
            continue
        values = np.random.default_rng(count + len(split)).normal(size=(count, 32, 6)).astype(np.float16)
        labels = np.arange(count) % 2
        metadata = pd.DataFrame(dict(base_id=[f"{split}_{k}" for k in range(count)],
            group_id=[f"{split}_group_{k // 2}" for k in range(count)],
            source=np.where(np.arange(count) % 3, "source_a", "source_b"),
            label=labels, prompt_tokens=np.arange(count) + 1))
        path = study / "caches" / f"{split}.npy"
        meta = study / "caches" / f"{split}.parquet"
        np.save(path, values)
        metadata.to_parquet(meta, index=False)
        dump(study / "caches" / f"{split}.manifest.json", dict(
            complete=True, split=split, shape=list(values.shape), path=str(path),
            metadata_path=str(meta), sha256=sha256(path), metadata_sha256=sha256(meta),
            protocol_sha256=sha256(study / "protocol.json"), representation="base",
            adapter_sha256=None, ordered_base_id_sha256=ordered_base_id_sha256(metadata.base_id)))
    return study


def solo_logits(head, raw):
    return head(dict(layers=(raw[:, None],), mask=torch.ones((len(raw), 1), dtype=torch.bool)))["logits"]


def test_vectorized_matches_independent_mlp_initialization_forward_backward_and_step():
    hidden, width, depth = 7, 5, 3
    vector = runner.VectorizedLayerMLPs(hidden, width, depth, seed=42, dropout=0.)
    solos = []
    for k in range(depth):
        torch.manual_seed(42)
        head = ExperimentHead(hidden, mode="last", width=width, layers=(k + 1,), dropout=0.)
        assert state_digest(head) == state_digest(vector.layer_state(k))
        solos.append(head)
    raw = torch.randn(9, depth, hidden)
    target = torch.arange(9).remainder(2).float()
    logits = vector(raw)
    loss = F.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits), reduction="none").mean(0).sum()
    loss.backward()
    solo_optimizers = []
    for k, head in enumerate(solos):
        expected = solo_logits(head, raw[:, k])
        torch.testing.assert_close(logits[:, k], expected, rtol=1e-6, atol=1e-7)
        F.binary_cross_entropy_with_logits(expected, target).backward()
        for name, p in head.named_parameters():
            q = getattr(vector, name.replace(".", "_"))
            torch.testing.assert_close(q.grad[k], p.grad, rtol=1e-5, atol=1e-7)
        solo_optimizers.append(torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.007))
        torch.nn.utils.clip_grad_norm_(head.parameters(), .05)
    vector.clip_independent_gradients(.05)
    for k, head in enumerate(solos):
        for name, p in head.named_parameters():
            torch.testing.assert_close(getattr(vector, name.replace(".", "_")).grad[k], p.grad, rtol=1e-5, atol=1e-7)
    torch.optim.AdamW(vector.parameters(), lr=.003, weight_decay=.007).step()
    for optimizer in solo_optimizers:
        optimizer.step()
    for k, head in enumerate(solos):
        for name, p in head.state_dict().items():
            torch.testing.assert_close(vector.layer_state(k)[name], p, rtol=1e-5, atol=1e-7)


def test_vectorized_paired_dropout_matches_solo():
    vector = runner.VectorizedLayerMLPs(7, 5, 3, seed=42, dropout=.3)
    raw = torch.randn(9, 3, 7)
    torch.manual_seed(1729)
    actual = vector(raw)
    for k in range(3):
        head = ExperimentHead(7, mode="last", width=5, layers=(k + 1,), dropout=.3)
        head.load_state_dict(vector.layer_state(k))
        torch.manual_seed(1729)
        torch.testing.assert_close(actual[:, k], solo_logits(head, raw[:, k]), rtol=1e-6, atol=1e-7)


def test_capacity_initialization_and_best_anchor_inputs():
    s = runner.settings(dict(settings=dict(width=5, dropout=0.)))
    heads, specs, matched = runner.build_fusion_heads(7, s, best=21, device="cpu")
    assert len({state_digest(h) for h in heads.values()}) == 1
    assert len({sum(p.numel() for p in h.parameters()) for h in heads.values()}) == 1
    assert matched["parameters"] == sum(p.numel() for p in heads["fusion_fixed"].parameters())
    assert specs["repeat_best"] == ("repeat_last", (21,))
    assert specs["fusion_fixed_best_anchor"] == ("fusion_best_anchor", (8, 16, 24, 32, 21))


def test_shared_batches_and_validation_only_layer_selection():
    a = runner.epoch_batches(35, 42, 0, 8)
    b = runner.epoch_batches(35, 42, 0, 8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert sorted(np.concatenate(a).tolist()) == list(range(35))
    assert not np.array_equal(np.concatenate(a), np.concatenate(runner.epoch_batches(35, 42, 1, 8)))
    assert runner.choose_best_layer([.3, .1, .1, .2]) == 2
    with pytest.raises(ValueError):
        runner.choose_best_layer([np.nan, .1])


def test_settings_match_root_names():
    s = runner.settings(dict(settings=dict(weight_decay=.007, grad_clip=.5, bootstrap_draws=16)))
    assert (s["wd"], s["clip"], s["bootstrap_draws"]) == (.007, .5, 16)
    with pytest.raises(ValueError):
        runner.settings(dict(settings=dict(layers=[0, 8, 16, 32])))


def test_cache_rejects_stale_protocol_wrong_split_and_bad_metadata(tmp_path):
    study = cache_fixture(tmp_path)
    array, meta, _ = runner.load_cache(study, "train")
    assert isinstance(array, np.memmap) and len(meta) == 24
    path = study / "caches/train.manifest.json"
    manifest = runner.read_json(path)
    manifest["protocol_sha256"] = "stale"
    dump(path, manifest)
    with pytest.raises(ValueError, match="split/protocol"):
        runner.load_cache(study, "train")
    manifest["protocol_sha256"] = sha256(study / "protocol.json")
    manifest["split"] = "validation"
    dump(path, manifest)
    with pytest.raises(ValueError, match="split/protocol"):
        runner.load_cache(study, "train")
    manifest["split"] = "train"
    metadata = pd.read_parquet(manifest["metadata_path"])
    metadata.loc[0, "label"] = 3
    metadata.to_parquet(manifest["metadata_path"], index=False)
    manifest["metadata_sha256"] = sha256(manifest["metadata_path"])
    dump(path, manifest)
    with pytest.raises(ValueError, match="Binary labels"):
        runner.load_cache(study, "train")


def test_cache_rejects_extra_npy_bytes_even_if_self_hash_updated(tmp_path):
    study = cache_fixture(tmp_path)
    path = study / "caches/train.manifest.json"
    manifest = runner.read_json(path)
    with Path(manifest["path"]).open("ab") as stream:
        stream.write(b"extra")
    manifest["sha256"] = sha256(manifest["path"])
    dump(path, manifest)
    with pytest.raises(ValueError):
        runner.load_cache(study, "train")


def test_selection_barrier_precedes_every_test_load(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, include_test=False)
    with pytest.raises(RuntimeError, match="Test barrier"):
        runner.load_cache(study, "test")
    calls = []
    monkeypatch.setattr(runner, "load_cache", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="Test barrier"):
        runner.evaluate(study, "cpu")
    assert calls == []


def test_fit_without_test_cache_independent_snapshots_and_lock_tamper(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, include_test=False)
    actual_load = runner.load_cache
    calls = []
    def guarded_load(study, split):
        assert split != "test"
        calls.append(split)
        return actual_load(study, split)
    monkeypatch.setattr(runner, "load_cache", guarded_load)
    lock = runner.fit(study, "cpu")
    assert lock["complete"] and calls == ["train", "validation"]
    selection = runner.read_json(study / "heads/selection.json")
    assert selection["best_single_layer"] == runner.choose_best_layer(selection["layer_validation_bce"])
    assert len(selection["layer_best_epochs"]) == 32
    checkpoint = torch.load(study / "heads/selected_heads.pt", weights_only=True)
    layers = torch.load(study / "heads/budget_1/all_layer_mlps.pt", weights_only=True)
    vector = runner.VectorizedLayerMLPs(6, 5, 32, dropout=0.)
    assert state_digest(checkpoint["states"]["last"]) == state_digest(vector.layer_state(31, layers["state"]))
    best = selection["best_single_layer"] - 1
    assert state_digest(checkpoint["states"]["best_single"]) == state_digest(vector.layer_state(best, layers["state"]))
    assert runner.verify_selection_lock(study)["complete"]
    with (study / "heads/selected_heads.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="after lock"):
        runner.verify_selection_lock(study)


def test_each_layer_keeps_its_own_validation_best_epoch(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, include_test=False)
    train, train_meta, _ = runner.load_cache(study, "train")
    validation, val_meta, _ = runner.load_cache(study, "validation")
    s = runner.settings(runner.read_json(study / "protocol.json"))
    folder = study / "snapshot_test"
    folder.mkdir()
    snapshots = []
    def controlled_validation(model, cache, batch_size, device, study):
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        epoch = min(len(snapshots), 3)
        magnitudes = np.full(32, epoch, dtype=np.float32)
        magnitudes[0] = [3., .5, 1.][epoch - 1]
        magnitudes[1] = [.5, 4., 2.][epoch - 1]
        return (2 * val_meta.label.to_numpy() - 1)[:, None] * magnitudes[None]
    monkeypatch.setattr(runner, "_predict_layers", controlled_validation)
    model, _, _, best_epochs, _ = runner._train_layers(study, folder, train, validation,
        train_meta.label.to_numpy(), val_meta.label.to_numpy(), s, 3, "cpu")
    assert best_epochs.tolist() == [1, 2] + [3] * 30
    for layer, chosen_epoch in enumerate(best_epochs):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value[layer], snapshots[chosen_epoch - 1][key][layer], rtol=0, atol=0)


def test_common_budget_extension_refits_all_candidates_with_new_layer_selection(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, include_test=False)
    protocol = runner.read_json(study / "protocol.json")
    protocol["settings"]["max_epochs"] = 2
    dump(study / "protocol.json", protocol)
    for split in ("train", "validation"):
        path = study / "caches" / f"{split}.manifest.json"
        manifest = runner.read_json(path)
        manifest["protocol_sha256"] = sha256(study / "protocol.json")
        dump(path, manifest)
    actual_layers, actual_fusion = runner._train_layers, runner._train_fusion
    executed = []
    def track_layers(*args):
        executed.append(("layers", args[7]))
        return actual_layers(*args)
    def track_fusion(*args):
        executed.append(("fusion", args[7]))
        return actual_fusion(*args)
    monkeypatch.setattr(runner, "_train_layers", track_layers)
    monkeypatch.setattr(runner, "_train_fusion", track_fusion)
    monkeypatch.setattr(runner, "budget_extension", lambda history, chosen, budget, maximum:
        dict(new_budget=2) if budget == 1 else None)
    runner.fit(study, "cpu")
    assert executed == [("layers", 1), ("fusion", 1), ("layers", 2), ("fusion", 2)]
    selection = runner.read_json(study / "heads/selection.json")
    assert selection["final_common_epoch_budget"] == 2
    assert [a["discarded_for_common_budget_extension"] for a in selection["attempts"]] == [True, False]
    assert selection["best_single_layer"] == runner.choose_best_layer(selection["layer_validation_bce"])


def test_cpu_full_fit_and_locked_evaluation(tmp_path):
    study = cache_fixture(tmp_path)
    runner.fit(study, "cpu")
    result = runner.evaluate(study, "cpu")
    assert result["complete"] and result["rows"] == 12
    assert set(result["metrics"]) == set(runner.ARMS)
    assert set(result["fixed_05_metrics"]) == set(runner.ARMS)
    assert len(result["comparisons"]) == 8
    assert all(p["group_bootstrap"]["repetitions"] == 16 for p in result["comparisons"])
    frame = pd.read_parquet(study / "analysis/predictions.parquet")
    assert len(frame) == 12
    assert runner.evaluate(study, "cpu") == result
    dump(study / "protocol.json", {"modified": True})
    with pytest.raises(ValueError, match="protocol"):
        runner.evaluate(study, "cpu")
