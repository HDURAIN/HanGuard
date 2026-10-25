"""Recovery checks for the validation-only, epoch-checkpointed readout fit."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import frozen_base_resumable_heads as runner
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    dump, ordered_base_id_sha256, sha256, state_digest,
)


class InterruptedFit(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def small_cpu_pool():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prior)


def cache_fixture(tmp_path, *, epochs=3, maximum=None):
    study = tmp_path / "study"
    (study / "caches").mkdir(parents=True)
    dump(study / "protocol.json", dict(num_layers=32, hidden_size=6, settings=dict(
        seed=42, width=5, dropout=.3, batch_size=8, lr=.001,
        weight_decay=.007, grad_clip=.5, epochs=epochs,
        max_epochs=epochs if maximum is None else maximum,
        layers=[8, 16, 24, 32], bootstrap_draws=16)))
    for split, count in (("train", 24), ("validation", 12)):
        values = np.random.default_rng(count + len(split)).normal(
            size=(count, 32, 6)).astype(np.float16)
        metadata = pd.DataFrame(dict(
            base_id=[f"{split}_{k}" for k in range(count)],
            group_id=[f"{split}_group_{k // 2}" for k in range(count)],
            source=np.where(np.arange(count) % 3, "source_a", "source_b"),
            label=np.arange(count) % 2, prompt_tokens=np.arange(count) + 1))
        path = study / "caches" / f"{split}.npy"
        meta = study / "caches" / f"{split}.parquet"
        np.save(path, values)
        metadata.to_parquet(meta, index=False)
        dump(study / "caches" / f"{split}.manifest.json", dict(
            complete=True, split=split, shape=list(values.shape), path=str(path),
            metadata_path=str(meta), sha256=sha256(path), metadata_sha256=sha256(meta),
            protocol_sha256=sha256(study / "protocol.json"), representation="base",
            adapter_sha256=None,
            ordered_base_id_sha256=ordered_base_id_sha256(metadata.base_id)))
    return study


def stage_args(study, folder):
    train, train_meta, _ = runner.load_cache(study, "train")
    validation, val_meta, _ = runner.load_cache(study, "validation")
    s = runner.settings(runner.read_json(study / "protocol.json"))
    folder.mkdir(parents=True, exist_ok=True)
    return (study, folder, train, validation, train_meta.label.to_numpy(),
            val_meta.label.to_numpy(), s)


def assert_tree_equal(left, right):
    """Compare optimizer tensors as well as the independently selected models."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def interrupt_after_epoch(monkeypatch, epoch=1):
    actual = runner._save_resume
    def interrupted(*args, **kwargs):
        actual(*args, **kwargs)
        payloads = [value for value in (*args, *kwargs.values())
                    if isinstance(value, dict) and "completed_epochs" in value]
        assert len(payloads) == 1, "_save_resume should receive one epoch payload"
        if payloads[0]["completed_epochs"] == epoch:
            raise InterruptedFit("simulated exit after atomic epoch save")
    monkeypatch.setattr(runner, "_save_resume", interrupted)


@pytest.mark.parametrize("stage", ["layers", "fusion"])
def test_nonzero_dropout_continuous_matches_original_scientific_training(tmp_path, stage):
    """Recovery code must preserve the original experiment, beyond self-consistency."""
    study = cache_fixture(tmp_path)
    legacy_args = stage_args(study, study / "legacy")
    new_args = stage_args(study, study / "continuous")
    suffix = (3, "cpu") if stage == "layers" else (3, 21, "cpu")
    legacy_function = (runner.original._train_layers if stage == "layers"
                       else runner.original._train_fusion)
    new_function = runner._train_layers if stage == "layers" else runner._train_fusion
    expected = legacy_function(*legacy_args, *suffix)
    actual = new_function(*new_args, *suffix)
    if stage == "layers":
        assert_tree_equal(expected[0].state_dict(), actual[0].state_dict())
    else:
        for arm in runner.FUSION_ARMS:
            assert_tree_equal(expected[0][arm].state_dict(), actual[0][arm].state_dict())
    for left, right in zip(expected[1:], actual[1:]):
        assert_tree_equal(left, right)
    for audit in (("layer_gradient_audit.json",) if stage == "layers"
                  else ("fusion_gradient_audit.json",)):
        assert_tree_equal(runner.read_json(legacy_args[1] / audit),
                          runner.read_json(new_args[1] / audit))


@pytest.mark.parametrize("stage", ["layers", "fusion"])
def test_nonzero_dropout_epoch_resume_matches_continuous_current_best_optimizer_and_history(
        tmp_path, monkeypatch, stage):
    study = cache_fixture(tmp_path)
    complete_args = stage_args(study, study / "continuous")
    resumed_args = stage_args(study, study / "resumed")
    function = runner._train_layers if stage == "layers" else runner._train_fusion
    suffix = (3, "cpu") if stage == "layers" else (3, 21, "cpu")
    continuous = function(*complete_args, *suffix)
    with monkeypatch.context() as patch:
        interrupt_after_epoch(patch)
        with pytest.raises(InterruptedFit):
            function(*resumed_args, *suffix)
    checkpoint_path = resumed_args[1] / f"{stage}_resume.pt"
    first = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert first["completed_epochs"] == 1
    assert first["global_step"] == 3
    assert first["rng"]
    # Unrelated CPU RNG consumption before resume must not affect the fit.
    torch.manual_seed(999)
    torch.rand(117)
    resumed = function(*resumed_args, *suffix)
    expected_checkpoint = torch.load(complete_args[1] / f"{stage}_resume.pt",
                                     map_location="cpu", weights_only=True)
    actual_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert actual_checkpoint["completed_epochs"] == 3
    assert actual_checkpoint["global_step"] == 9
    assert_tree_equal(expected_checkpoint, actual_checkpoint)
    if stage == "layers":
        assert_tree_equal(continuous[0].state_dict(), resumed[0].state_dict())
        for expected, actual in zip(continuous[1:], resumed[1:]):
            assert_tree_equal(expected, actual)
    else:
        for arm in runner.FUSION_ARMS:
            assert_tree_equal(continuous[0][arm].state_dict(), resumed[0][arm].state_dict())
        for expected, actual in zip(continuous[1:], resumed[1:]):
            assert_tree_equal(expected, actual)


@pytest.mark.parametrize("stage", ["layers", "fusion"])
def test_mid_epoch_failure_discards_uncommitted_steps(tmp_path, monkeypatch, stage):
    study = cache_fixture(tmp_path)
    continuous_args = stage_args(study, study / "continuous")
    resumed_args = stage_args(study, study / "resumed")
    function = runner._train_layers if stage == "layers" else runner._train_fusion
    suffix = (3, "cpu") if stage == "layers" else (3, 21, "cpu")
    function(*continuous_args, *suffix)
    path = resumed_args[1] / f"{stage}_resume.pt"
    original_batch = runner._batch
    training_calls_after_commit = []
    def stop_during_next_epoch(cache, *args, **kwargs):
        if path.exists() and cache is resumed_args[2]:
            training_calls_after_commit.append(1)
            if len(training_calls_after_commit) == 2:
                raise InterruptedFit("simulated mid-epoch process loss")
        return original_batch(cache, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_batch", stop_during_next_epoch)
        with pytest.raises(InterruptedFit):
            function(*resumed_args, *suffix)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["completed_epochs"] == 1
    assert checkpoint["global_step"] == 3
    function(*resumed_args, *suffix)
    assert_tree_equal(torch.load(continuous_args[1] / f"{stage}_resume.pt", weights_only=True),
                      torch.load(path, weights_only=True))


@pytest.mark.parametrize("changed", ["protocol", "settings", "cache", "source", "budget"])
def test_layers_reject_changed_recovery_identity(tmp_path, monkeypatch, changed):
    study = cache_fixture(tmp_path)
    args = stage_args(study, study / "resume_identity")
    with monkeypatch.context() as patch:
        interrupt_after_epoch(patch)
        with pytest.raises(InterruptedFit):
            runner._train_layers(*args, 3, "cpu")
    call_args, budget = list(args), 3
    if changed == "protocol":
        protocol = runner.read_json(study / "protocol.json")
        protocol["identity_change"] = True
        dump(study / "protocol.json", protocol)
    elif changed == "settings":
        call_args[6] = {**args[6], "lr": .02}
    elif changed == "cache":
        path = study / "caches/train.manifest.json"
        manifest = runner.read_json(path)
        manifest["identity_change"] = True
        dump(path, manifest)
    elif changed == "source":
        path = args[1] / "layers_resume.pt"
        checkpoint = torch.load(path, weights_only=True)
        sources = checkpoint["identity"]["source_sha256"]
        sources[next(iter(sources))] = "0" * 64
        torch.save(checkpoint, path)
    elif changed == "budget":
        budget = 2
    with pytest.raises((ValueError, RuntimeError)):
        runner._train_layers(*call_args, budget, "cpu")


def test_fusion_rejects_changed_validation_selected_anchor(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path)
    args = stage_args(study, study / "anchor_identity")
    with monkeypatch.context() as patch:
        interrupt_after_epoch(patch)
        with pytest.raises(InterruptedFit):
            runner._train_fusion(*args, 3, 21, "cpu")
    with pytest.raises((ValueError, RuntimeError)):
        runner._train_fusion(*args, 3, 22, "cpu")


def test_each_layer_retains_its_own_best_epoch_across_resume(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path)
    args = stage_args(study, study / "independent_best")
    snapshots = []
    labels = args[5]
    def controlled_validation(model, cache, batch_size, device, study):
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        epoch = min(len(snapshots), 3)
        magnitudes = np.full(32, epoch, dtype=np.float32)
        magnitudes[0] = [3., .5, 1.][epoch - 1]
        magnitudes[1] = [.5, 4., 2.][epoch - 1]
        return (2 * labels - 1)[:, None] * magnitudes[None]
    monkeypatch.setattr(runner, "_predict_layers", controlled_validation)
    with monkeypatch.context() as patch:
        interrupt_after_epoch(patch)
        with pytest.raises(InterruptedFit):
            runner._train_layers(*args, 3, "cpu")
    model, _, _, best_epochs, _ = runner._train_layers(*args, 3, "cpu")
    assert best_epochs.tolist() == [1, 2] + [3] * 30
    checkpoint = torch.load(args[1] / "layers_resume.pt", weights_only=True)
    for layer, epoch in enumerate(best_epochs):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value[layer], snapshots[epoch - 1][key][layer], rtol=0, atol=0)
            torch.testing.assert_close(checkpoint["best_state"][key][layer], value[layer], rtol=0, atol=0)
    # The continuation state must be the final epoch, never the per-layer best.
    assert_tree_equal(checkpoint["current_state"], snapshots[2])


def test_fit_resume_never_opens_test_and_skips_completed_layer_training(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, epochs=2)
    original_load = runner.load_cache
    loads = []
    def train_validation_only(study, split):
        assert split in ("train", "validation"), "fit must never read the test split"
        loads.append(split)
        return original_load(study, split)
    monkeypatch.setattr(runner, "load_cache", train_validation_only)
    actual_save = runner._save_resume
    def interrupt_fusion(*args, **kwargs):
        actual_save(*args, **kwargs)
        if any(isinstance(value, (str, Path)) and Path(value).name == "fusion_resume.pt"
               for value in (*args, *kwargs.values())):
            raise InterruptedFit("fusion epoch complete; layers already complete")
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_save_resume", interrupt_fusion)
        with pytest.raises(InterruptedFit):
            runner.fit(study, "cpu")
    assert not (study / "selection_lock.json").exists()
    layer_path = study / "heads/budget_2/layers_resume.pt"
    prior_layer_sha = sha256(layer_path)
    original_batch = runner._batch
    def no_repeated_layer_training(cache, rows, device, layers=None):
        if layers is None:
            # Completed layer stage may predict validation, but may not train.
            assert len(cache) == 12
        return original_batch(cache, rows, device, layers)
    monkeypatch.setattr(runner, "_batch", no_repeated_layer_training)
    lock = runner.fit(study, "cpu")
    assert lock["complete"] and lock["test_opened"] is False
    assert sha256(layer_path) == prior_layer_sha
    assert loads == ["train", "validation", "train", "validation"]
    assert not (study / "caches/test.manifest.json").exists()
    assert runner.verify_selection_lock(study)["complete"]
    # Once selection is locked, fit is idempotent and cannot resume more epochs.
    monkeypatch.setattr(runner, "load_cache", lambda *_: pytest.fail("locked fit accessed cache"))
    assert runner.fit(study, "cpu") == lock


def test_preexisting_attempt_without_full_checkpoint_is_not_automatically_reused(tmp_path):
    study = cache_fixture(tmp_path)
    folder = study / "heads"
    folder.mkdir()
    dump(folder / "fit_started.json", dict(protocol_sha256=sha256(study / "protocol.json"),
                                           test_opened=False, source_sha256={}))
    dump(folder / "layer_history.json", [dict(epoch=1)])
    with pytest.raises((RuntimeError, ValueError)):
        runner.fit(study, "cpu")
    assert not (study / "selection_lock.json").exists()
    assert not list(folder.rglob("*_resume.pt"))


def test_budget_extension_refits_every_arm_from_common_initialization(tmp_path, monkeypatch):
    study = cache_fixture(tmp_path, epochs=1, maximum=2)
    original_layers, original_fusion = runner._train_layers, runner._train_fusion
    calls = []
    def layers(*args):
        calls.append(("layers", args[7]))
        return original_layers(*args)
    def fusion(*args):
        calls.append(("fusion", args[7]))
        return original_fusion(*args)
    monkeypatch.setattr(runner, "_train_layers", layers)
    monkeypatch.setattr(runner, "_train_fusion", fusion)
    monkeypatch.setattr(runner, "budget_extension", lambda history, chosen, budget, maximum:
                        dict(new_budget=2) if budget == 1 else None)
    runner.fit(study, "cpu")
    assert calls == [("layers", 1), ("fusion", 1), ("layers", 2), ("fusion", 2)]
    for stage in ("layers", "fusion"):
        first = torch.load(study / f"heads/budget_1/{stage}_resume.pt", weights_only=True)
        second = torch.load(study / f"heads/budget_2/{stage}_resume.pt", weights_only=True)
        assert first["global_step"] == 3
        assert second["global_step"] == 6
        assert first["initial_state_sha256"] == second["initial_state_sha256"]
    selection = runner.read_json(study / "heads/selection.json")
    assert selection["final_common_epoch_budget"] == 2
    assert selection["best_single_layer"] == runner.choose_best_layer(selection["layer_validation_bce"])
    assert [item["discarded_for_common_budget_extension"] for item in selection["attempts"]] == [True, False]


def test_failed_checkpoint_serialization_preserves_previous_complete_epoch(tmp_path, monkeypatch):
    path = tmp_path / "layers_resume.pt"
    first = dict(completed_epochs=1, current_state={"weight": torch.arange(5.)})
    runner._save_resume(path, first)
    prior_sha = sha256(path)
    def fail_serialization(payload, temporary):
        Path(temporary).write_bytes(b"incomplete serialized checkpoint")
        raise OSError("simulated loss while writing next epoch")
    monkeypatch.setattr(torch, "save", fail_serialization)
    with pytest.raises(OSError, match="simulated loss"):
        runner._save_resume(path, dict(completed_epochs=2,
                                       current_state={"weight": torch.arange(5.) + 1}))
    assert sha256(path) == prior_sha
    assert_tree_equal(torch.load(path, weights_only=True), first)


def test_study_local_runtime_sources_are_locked_and_pass_original_test_barrier(tmp_path):
    from scripts.hanguard import frozen_base_readout_study as base

    study = cache_fixture(tmp_path, epochs=1)
    protocol = runner.read_json(study / "protocol.json")
    protocol["code_sha256"] = {str(path): sha256(path) for path in runner._legacy_sources()}
    dump(study / "protocol.json", protocol)
    for split in ("train", "validation"):
        path = study / f"caches/{split}.manifest.json"
        manifest = runner.read_json(path)
        manifest["protocol_sha256"] = sha256(study / "protocol.json")
        dump(path, manifest)
    source_folder = study / "runtime_sources"
    source_folder.mkdir()
    snapshot = source_folder / "resumable_heads.py"
    snapshot.write_bytes(Path(runner.__file__).read_bytes())
    control = source_folder / "control.json"
    dump(control, dict(reviewed_epoch_resume=True, seed=42))
    lock = runner.fit(study, "cpu", extra_source_paths=(snapshot, control))
    selection = runner.read_json(study / "heads/selection.json")
    for path in (snapshot, control):
        assert selection["source_sha256"][str(path)] == sha256(path)
        assert lock["artifacts"][str(path)] == sha256(path)
    # The registered old protocol only permits its original external sources;
    # new runtime code is instead represented by its locked study-local copy.
    assert str(Path(runner.__file__).resolve()) not in lock["artifacts"]
    assert base.validate_selection_lock(study) == lock
    snapshot.write_text(snapshot.read_text() + "\n# changed after selection\n")
    with pytest.raises(ValueError):
        runner.verify_selection_lock(study)
    with pytest.raises(ValueError):
        base.validate_selection_lock(study)
