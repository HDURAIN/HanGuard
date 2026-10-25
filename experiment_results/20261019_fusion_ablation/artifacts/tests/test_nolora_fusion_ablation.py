"""CPU-only contract checks for append-only no-LoRA fusion ablations."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import nolora_fusion_ablation as study
from scripts.hanguard.ab_frozen_readout_diagnostics import _head_logits
from scripts.hanguard.frozen_lora_layer_diagnostics import ordered_base_id_sha256


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def parent_fixture(tmp_path):
    parent = tmp_path / "parent"
    (parent / "caches").mkdir(parents=True)
    protocol = dict(num_layers=32, hidden_size=6, use_lora=False,
                    rows=dict(train=24, validation=12, test=12),
                    settings=dict(seed=42, width=5, dropout=.2, batch_size=8, lr=.001,
                                  weight_decay=.001, grad_clip=5., epochs=2, max_epochs=2,
                                  layers=[8, 16, 24, 32], bootstrap_draws=4))
    study.dump(parent / "protocol.json", protocol)
    for split, rows in protocol["rows"].items():
        values = np.random.default_rng(rows + len(split)).normal(size=(rows, 32, 6)).astype(np.float16)
        meta = pd.DataFrame(dict(base_id=[f"{split}_{i}" for i in range(rows)],
                                 group_id=[f"{split}_group_{i // 2}" for i in range(rows)],
                                 source=["a" if i % 4 < 2 else "b" for i in range(rows)],
                                 label=np.arange(rows) % 2, prompt_tokens=np.arange(rows) + 1))
        path, meta_path = parent / f"caches/{split}.npy", parent / f"caches/{split}.metadata.parquet"
        np.save(path, values)
        meta.to_parquet(meta_path, index=False)
        study.dump(parent / f"caches/{split}.manifest.json", dict(
            complete=True, split=split, shape=list(values.shape), path=str(path),
            metadata_path=str(meta_path), sha256=study.sha256(path), metadata_sha256=study.sha256(meta_path),
            protocol_sha256=study.sha256(parent / "protocol.json"), adapter_sha256=None,
            representation="base", ordered_base_id_sha256=ordered_base_id_sha256(meta.base_id)))
    study.resumable.fit(parent, device="cpu")
    study.original.evaluate(parent, device="cpu")
    subset = tmp_path / "subset"
    subset.mkdir()
    (subset / "excluded_base_ids.txt").write_text("test_0\ntest_1\n")
    study.dump(subset / "filter_policy.json", dict(selection_policy="posthoc_error_selected_test_subset",
               no_text_changes=True, no_label_changes=True, selected_rows=2))
    return parent, subset


def setup_study(tmp_path):
    parent, subset = parent_fixture(tmp_path)
    output = tmp_path / "ablation"
    study.initialize(output, parent, subset)
    return output, parent, subset


def test_full_forward_and_gradients_equal_original_anchor():
    torch.manual_seed(42)
    head = study.ComponentHead(6, 5, 0, dict(kind="full", layers=[8, 16, 24, 32], anchor=19))
    torch.manual_seed(42)
    old = study.ExperimentHead(6, mode="fusion", width=5, layers=(8, 16, 24, 32), dropout=0)
    assert head.state_dict().keys() == old.state_dict().keys()
    # Nonzero residual checks both branches; the production initialization is zero.
    with torch.no_grad():
        head.fusion_output.weight.normal_()
    old.load_state_dict(head.state_dict())
    raw = torch.randn(4, 32, 6)
    actual = head(raw)
    expected = _head_logits(old, raw, [7, 15, 23, 31, 18], "fusion_best_anchor")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name_a, a), (name_b, b) in zip(head.named_parameters(), old.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_component_controls_change_only_requested_routes():
    spec = dict(kind="uniform_weights", layers=[8, 16, 24, 32], anchor=19)
    uniform = study.ComponentHead(6, 5, 0, spec)
    assert not uniform.layer_logits.requires_grad
    torch.testing.assert_close(uniform.layer_logits.softmax(0), torch.full((4,), .25))
    shared = study.ComponentHead(6, 5, 0, dict(spec, kind="shared_projection"))
    assert shared.layer_values[0] is shared.layer_values[3]
    assert len(list(shared.parameters())) == len({id(p) for p in shared.parameters()})
    no_anchor = study.ComponentHead(6, 5, 0, dict(spec, kind="no_anchor"))
    assert not hasattr(no_anchor, "project")
    raw = torch.randn(3, 32, 6)
    one = no_anchor(raw)
    raw[:, 18] += 100
    torch.testing.assert_close(one, no_anchor(raw), rtol=0, atol=0)
    mean = study.ComponentHead(6, 5, 0, dict(spec, kind="mean_pool"))
    single = study.ExperimentHead(6, mode="last", width=5, layers=(19,), dropout=0)
    assert sum(p.numel() for p in mean.parameters()) == sum(p.numel() for p in single.parameters())
    raw = torch.randn(3, 32, 6)
    expected = mean.classify_representation(torch.nn.functional.gelu(mean._project_raw(
        raw[:, [7, 15, 23, 31]].mean(1), mean.project)))
    torch.testing.assert_close(mean(raw), expected, rtol=0, atol=0)


def test_top4_is_validation_only_and_stable_on_ties():
    values = np.ones(32)
    values[[20, 3, 9, 11]] = .1
    specs = study.component_specs(dict(layer_validation_bce=values.tolist(), best_single_layer=4))
    assert specs["top4"]["layers"] == [4, 10, 12, 21]
    assert specs["two_layers"]["layers"] == [16, 32]
    assert specs["no_anchor"]["anchor"] == 4
    assert study.component_specs(dict(layer_validation_bce=[1.] * 32, best_single_layer=1))["top4"]["layers"] == [1, 2, 3, 4]


@pytest.mark.parametrize("kind", [*study.COMPONENTS, "full"])
def test_every_component_has_finite_nonzero_training_gradients(kind):
    torch.manual_seed(42)
    head = study.ComponentHead(6, 5, .1, dict(kind=kind, layers=[8, 16, 24, 32], anchor=19))
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=.01)
    raw, y = torch.randn(8, 32, 6), torch.arange(8).remainder(2).float()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(raw), y)
        loss.backward()
        gradients = [p.grad for p in head.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        optimizer.step()
    assert head.classifier.weight.grad.abs().sum() > 0
    if kind != "mean_pool":
        assert head.layer_values[0][0].weight.grad.abs().sum() > 0
    if kind == "uniform_weights":
        assert head.layer_logits.grad is None
        torch.testing.assert_close(head.layer_logits.detach(), torch.zeros(4), rtol=0, atol=0)


def test_fixed_mask_uses_identity_and_rejects_missing_or_duplicate_rows():
    frame = pd.DataFrame(dict(base_id=["a", "b", "c"], label=[0, 1, 0]))
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(study.selected_mask(frame, {"b"}), expected)
    frame.label = 1 - frame.label
    np.testing.assert_array_equal(study.selected_mask(frame, {"b"}), expected)
    with pytest.raises(ValueError):
        study.selected_mask(frame, {"missing"})
    with pytest.raises(ValueError):
        study.selected_mask(pd.concat([frame, frame.iloc[:1]]), {"b"})


def test_init_preserves_parent_and_registers_reused_cache_identity(tmp_path):
    parent, subset = parent_fixture(tmp_path)
    files = [p for p in parent.rglob("*") if p.is_file()]
    before = {str(p): study.sha256(p) for p in files}
    output = tmp_path / "new"
    p = study.initialize(output, parent, subset)
    assert p["subset_rows"] == 10
    assert p["selected_subset_is_independent_test"] is False
    for task in study.TASKS:
        child = study.task_directory(output, task)
        protocol = study.read(child / "protocol.json")
        assert protocol["settings"]["seed"] == (int(task[4:]) if task.startswith("seed") else 42)
        for split in ("train", "validation", "test"):
            old = study.read(parent / f"caches/{split}.manifest.json")
            new = study.read(child / f"caches/{split}.manifest.json")
            assert new["path"] == old["path"] and new["sha256"] == old["sha256"]
            assert new["protocol_sha256"] == study.sha256(child / "protocol.json")
    assert before == {str(p): study.sha256(p) for p in files}
    assert study.initialize(output, parent, subset)["bindings"] == p["bindings"]


def test_test_barrier_and_end_to_end_component_report(tmp_path, monkeypatch):
    output, parent, _ = setup_study(tmp_path)
    child = study.task_directory(output, "components")
    with pytest.raises(RuntimeError, match="Test barrier"):
        study.evaluate_components(output, "cpu")
    load = study.original.load_cache
    seen = []
    def tracked_cache(folder, split):
        seen.append(split)
        if split == "test":
            assert (child / "selection_lock.json").exists()
        return load(folder, split)
    with monkeypatch.context() as patch:
        patch.setattr(study.original, "load_cache", tracked_cache)
        study.fit_components(output, "cpu")
        assert seen == ["train", "validation"]
        result = study.evaluate_components(output, "cpu")
    assert result["full_test"]["rows"] == 12
    assert result["selected_subset"]["rows"] == 10
    assert set(result["trained_arms"]) == set(study.COMPONENTS) | {"anchor_replay"}
    assert result["common_budget_reference_arm"] == "anchor_replay"
    assert result["parameters"]["mean_pool"]["total"] == 41
    assert result["parameters"]["uniform_weights"]["total"] - result["parameters"]["uniform_weights"]["trainable"] == 4
    frame = pd.read_csv(child / "analysis/predictions.csv")
    assert "text" not in frame and "prompt" not in frame
    assert study.evaluate_components(output, "cpu") == result
    summary = study.summarize(output)
    assert set(summary["tasks"]) == {"seed42_imported", "components"}
    assert summary["pending_tasks"] == ["seed43", "seed44"]
    assert summary["tasks"]["seed42_imported"]["full_test"]["rows"] == 12
    assert study.summarize(output)["seed_statistics"] == summary["seed_statistics"]


def test_seed_worker_keeps_original_evaluation_receipt_valid(tmp_path):
    output, _, _ = setup_study(tmp_path)
    result = study.run_worker(output, "seed43", "cpu", 2)
    child = study.task_directory(output, "seed43")
    assert result["complete"] is True and result["seed"] == 43
    assert result["full_test"]["rows"] == 12 and result["selected_subset"]["rows"] == 10
    receipt = study.read(child / "analysis/complete.json")
    for name, digest in receipt["artifacts"].items():
        assert study.sha256(child / "analysis" / name) == digest
    assert study.run_worker(output, "seed43", "cpu", 2) == result
    summary = study.summarize(output)
    assert summary["seed_statistics"]["full_test"]["last"]["seeds"] == [42, 43]


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_component_resume_discards_partial_epoch_exactly(tmp_path, monkeypatch):
    output, _, _ = setup_study(tmp_path)
    child = study.task_directory(output, "components")
    train, train_meta, _ = study.original.load_cache(child, "train")
    validation, val_meta, _ = study.original.load_cache(child, "validation")
    protocol = study.read(child / "protocol.json")
    s, specs = study.original.settings(protocol), protocol["component_specs"]
    common = (train, validation, train_meta.label.to_numpy(), val_meta.label.to_numpy(), s, specs, 2, "cpu")
    continuous, resumed = child / "continuous", child / "resumed"
    study.train_component_budget(child, continuous, *common)
    original_batch = study.original._batch
    calls = []
    def interrupted(cache, *args, **kwargs):
        if cache is train and (resumed / "resume.pt").exists():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("mid epoch interruption")
        return original_batch(cache, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(study.original, "_batch", interrupted)
        with pytest.raises(RuntimeError, match="mid epoch"):
            study.train_component_budget(child, resumed, *common)
    assert torch.load(resumed / "resume.pt", weights_only=True)["completed_epochs"] == 1
    study.train_component_budget(child, resumed, *common)
    assert_tree_equal(torch.load(continuous / "resume.pt", weights_only=True),
                      torch.load(resumed / "resume.pt", weights_only=True))


def test_modified_exclusion_list_rejected(tmp_path):
    output, _, subset = setup_study(tmp_path)
    (subset / "excluded_base_ids.txt").write_text("test_2\ntest_3\n")
    with pytest.raises(ValueError, match="changed"):
        study.verify(output)


def test_cli_has_queue_worker_contract():
    args = study.parser().parse_args(["worker", "--output", "/tmp/unused", "--task", "seed44",
                                     "--device", "cpu", "--cpu-threads", "8"])
    assert args.task == "seed44" and args.cpu_threads == 8
