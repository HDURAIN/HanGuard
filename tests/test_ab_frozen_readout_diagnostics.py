"""Tests for nested label isolation and the anchor/capacity diagnostic controls."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard.ab_frozen_readout_diagnostics import (
    MAIN_ARMS, _head_logits, _resume_stage, budget_extension, build_heads, build_parser, check_stop,
    choose_layers, indices_from_assignments, initialize, load_b_cache,
    nested_indices, paired_comparison, predict_heads, train_head_fold,
    train_probe_fold, validate_outer_folds, write_report,
)
from scripts.hanguard.frozen_lora_layer_diagnostics import (
    dump, ordered_base_id_sha256, sha256, state_digest,
)


@contextmanager
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture_cache(tmp_path: Path):
    partition = tmp_path / "partition"
    (partition / "lora_data").mkdir(parents=True)
    (partition / "folds").mkdir()
    rows = 120
    ids = [f"B_{i:03d}" for i in range(rows)]
    groups = np.repeat([f"G_{i:02d}" for i in range(rows // 2)], 2)
    labels = np.tile([0, 1], rows // 2)
    outer = np.repeat(np.arange(2), rows // 2)
    metadata = pd.DataFrame(dict(base_id=ids, group_id=groups, source="synthetic",
                                 label=labels, outer_fold=outer))
    b = metadata.rename(columns={"label": "prompt_harm_label"}).copy()
    b["prompt_harm_label"] = np.where(labels, "harmful", "unharmful")
    b.to_parquet(partition / "B.parquet", index=False)
    for name in ("train", "validation"):
        pd.DataFrame(dict(base_id=[f"A_{name}"], group_id=[f"A_group_{name}"])).to_parquet(
            partition / "lora_data" / f"{name}.parquet", index=False)
    folds = []
    for fold in range(2):
        held = outer == fold
        available = np.flatnonzero(~held)
        roles = np.full(rows, "inner_train", dtype=object)
        roles[available[:12]] = "inner_validation"
        roles[held] = "outer_validation"
        frame = metadata[["base_id", "group_id"]].assign(role=roles, outer_fold=fold)
        path = partition / "folds" / f"fold_{fold}_assignments.parquet"
        frame.to_parquet(path, index=False)
        folds.append(dict(fold=fold, assignments_path=str(path)))
    names = ["B.parquet", "lora_data/train.parquet", "lora_data/validation.parquet",
             "folds/fold_0_assignments.parquet", "folds/fold_1_assignments.parquet"]
    ab_path = partition / "manifest.json"
    artifact_sha = {name: sha256(partition / name) for name in names}
    dump(ab_path, dict(policy_id="hanguard_train_internal_ab_grouped_v1", invariants=dict(final_test_enabled=False),
                       folds=folds, artifact_sha256=artifact_sha))
    training_directory = tmp_path / "training"
    run = training_directory / "runs/E03_s42"
    run.mkdir(parents=True)
    training_path = training_directory / "protocol.json"
    dump(training_path, dict(training_entrypoint="hanguard_ab_diagnostic_v1", partition_manifest_sha256=sha256(ab_path),
         adapter_training_cohort="A", fresh_base=True, reuse_old_adapter=False, formal_test_enabled=False,
         data_sha256=dict(train=artifact_sha["lora_data/train.parquet"], validation=artifact_sha["lora_data/validation.parquet"])))
    dump(training_directory / "registration.json", dict(protocol_sha256=sha256(training_path)))
    adapter_path = run / "adapter.pt"
    adapter_path.write_bytes(b"fresh-A-only-synthetic-adapter")
    lineage_path = run / "diagnostic_lineage.json"
    dump(lineage_path, dict(adapter_training_cohort="A", partition_manifest_sha256=sha256(ab_path),
         A_train_sha256=artifact_sha["lora_data/train.parquet"], A_selection_sha256=artifact_sha["lora_data/validation.parquet"],
         B_sha256=artifact_sha["B.parquet"], adapter_sha256=sha256(adapter_path), protocol_sha256=sha256(training_path), formal_test_opened=False))
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    feature_path = cache_directory / "features.npy"
    rng = np.random.default_rng(551)
    cache = rng.normal(size=(rows, 32, 8)).astype(np.float16)
    cache[:, 7, 0] += np.where(labels, 4, -4)
    cache[:, 7, 1] -= np.where(labels, 4, -4)
    np.save(feature_path, cache)
    metadata_path = cache_directory / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    manifest = dict(complete=True, cohort="B", path=str(feature_path), metadata_path=str(metadata_path),
                    shape=list(cache.shape), sha256=sha256(feature_path), metadata_sha256=sha256(metadata_path),
                    ordered_base_id_sha256=ordered_base_id_sha256(ids), representation="adapted", pooling="last",
                    adapter_training_cohort="A", adapter_sha256=sha256(adapter_path),
                    checkpoint_lineage_path=str(lineage_path),
                    training_protocol_path=str(training_path), training_protocol_sha256=sha256(training_path),
                    ab_protocol_path=str(ab_path), ab_protocol_sha256=sha256(ab_path))
    replay_path = cache_directory / "replay_audit.json"
    dump(replay_path, dict(passed=True))
    manifest["replay_audit_sha256"] = sha256(replay_path)
    manifest_path = cache_directory / "manifest.json"
    dump(manifest_path, manifest)
    return manifest_path


def arguments(manifest_path, output):
    return build_parser().parse_args([
        "run", "--cache-manifest", str(manifest_path), "--output", str(output), "--device", "cpu",
        "--probe-epochs", "2", "--head-epochs", "2", "--max-epochs", "2", "--batch-size", "48", "--width", "8",
        "--bootstrap-draws", "25", "--dropout", "0", "--joint-features",
    ])


def test_cache_requires_registered_b_and_rejects_old_full_train_adapter(tmp_path):
    path = fixture_cache(tmp_path)
    cache, metadata, manifest = load_b_cache(path)
    assert cache.shape == (120, 32, 8)
    assert nested_indices(metadata, 0)["dev"].shape == (12,)
    manifest["adapter_training_cohort"] = "full_train"
    dump(path, manifest)
    with pytest.raises(ValueError, match="trained on A only"):
        load_b_cache(path)
    manifest["adapter_training_cohort"] = "A"
    manifest["cohort"] = "test"
    dump(path, manifest)
    with pytest.raises(ValueError, match="cohort B"):
        load_b_cache(path)


def test_cache_rejects_changed_adapter_and_failed_replay(tmp_path):
    path = fixture_cache(tmp_path)
    manifest = __import__("json").loads(path.read_text())
    adapter = Path(manifest["checkpoint_lineage_path"]).parent / "adapter.pt"
    original = adapter.read_bytes()
    adapter.write_bytes(b"an old full-data adapter")
    with pytest.raises(ValueError, match="frozen adapter changed"):
        load_b_cache(path)
    adapter.write_bytes(original)
    replay = Path(manifest["metadata_path"]).parent / "replay_audit.json"
    dump(replay, dict(passed=False))
    manifest["replay_audit_sha256"] = sha256(replay)
    dump(path, manifest)
    with pytest.raises(ValueError, match="replay failed"):
        load_b_cache(path)


def test_stop_in_training_study_is_honored_before_readout_training(tmp_path):
    path = fixture_cache(tmp_path)
    _, metadata, manifest = load_b_cache(path)
    args = arguments(path, tmp_path / "readouts" / "base_last")
    protocol = initialize(args, metadata, manifest)
    (tmp_path / "training/STOP").write_text("Direction invalid: terminate diagnosis")
    with pytest.raises(RuntimeError, match="Direction invalid"):
        check_stop(protocol)


def test_prespecified_inner_roles_reject_group_and_heldout_leakage(tmp_path):
    path = fixture_cache(tmp_path)
    _, metadata, _ = load_b_cache(path)
    assignments = pd.read_parquet(tmp_path / "partition/folds/fold_0_assignments.parquet")
    assignments.loc[60, "role"] = "inner_train"
    with pytest.raises(AssertionError, match="groups overlap"):
        indices_from_assignments(metadata, 0, assignments)
    assignments = pd.read_parquet(tmp_path / "partition/folds/fold_0_assignments.parquet")
    assignments.loc[0, "role"] = "inner_train"
    with pytest.raises(ValueError, match="do not match"):
        indices_from_assignments(metadata, 0, assignments)
    metadata.loc[0, "outer_fold"] = 1
    with pytest.raises(ValueError, match="group_id leakage"):
        validate_outer_folds(metadata)


def test_best_anchor_controls_change_only_direct_path_and_match_repeat_parameters():
    with small_cpu_pool(), torch.no_grad():
        heads, specs, matches = build_heads(8, 8, best=8, selected=(2, 8, 13, 19),
                                             seed=29, dropout=0, device="cpu", joint=True)
        assert set(MAIN_ARMS).issubset(heads)
        assert matches["fusion"]["parameters"] > matches["single"]["parameters"]
        assert len({state_digest(heads[arm]) for arm in matches["fusion"]["arms"]}) == 1
        raw = torch.randn(11, 32, 8)
        last = _head_logits(heads["last"], raw, [31], "last")
        best = _head_logits(heads["best_single"], raw, [7], "last")
        fixed = _head_logits(heads["fusion_fixed"], raw, [7, 15, 23, 31], "fusion")
        anchored = _head_logits(heads["fusion_fixed_best_anchor"], raw, [7, 15, 23, 31, 7], "fusion_best_anchor")
        repeat = _head_logits(heads["repeat_best"], raw, [7], "repeat_last")
        # Zero residual initialization makes the direct anchor's prediction exact.
        torch.testing.assert_close(fixed, last)
        torch.testing.assert_close(anchored, best)
        torch.testing.assert_close(repeat, best)
        # Same actual features and anchor must also agree after activating branches.
        original = heads["fusion_fixed"]
        original.fusion_output.weight.copy_(torch.eye(8))
        anchor = heads["fusion_fixed_best_anchor"]
        anchor.load_state_dict(original.state_dict())
        expected = _head_logits(original, raw, [7, 15, 23, 31], "fusion")
        actual = _head_logits(anchor, raw, [7, 15, 23, 31, 31], "fusion_best_anchor")
        torch.testing.assert_close(actual, expected)


def test_selected_layers_do_not_force_layer_32_and_choose_dev_best():
    labels = np.tile([0, 1], 20)
    p = np.full((len(labels), 32), .5)
    p[:, 7] = np.where(labels, .99, .01)
    selected = choose_layers(labels, p, np.full(32, .5))
    assert selected["best_single_layer"] == 8
    assert 8 in selected["selected_layers"]
    assert len(selected["selected_layers"]) == 4
    # L32 can enter by the registered tie-break, but is not a forced anchor.
    assert "L32" not in selected["selected_by"]


def test_budget_extension_is_shared_bounded_and_uses_inner_dev_only():
    history = [dict(epoch=epoch, dev_bce=dict(single=.3, fusion=value))
               for epoch, value in ((10, .2), (11, .199), (12, .198))]
    extension = budget_extension(history, dict(single=10, fusion=12), budget=12, max_epochs=40)
    assert extension["new_budget"] == 24
    assert extension["common_extension"] is True
    assert set(extension["triggers"]) == {"fusion"}
    assert extension["selection_scope"] == "inner_dev_only"
    assert budget_extension(history, dict(single=10, fusion=11), 12, 40) is None
    assert budget_extension(history, dict(single=10, fusion=12), 12, 12) is None
    assert budget_extension(history, dict(single=10, fusion=12), 12, 15)["new_budget"] == 15


def test_outer_labels_cannot_change_probe_layer_head_checkpoint_or_threshold(tmp_path):
    with small_cpu_pool():
        path = fixture_cache(tmp_path)
        cache, metadata, manifest = load_b_cache(path)
        args = arguments(path, tmp_path / "first")
        protocol = initialize(args, metadata, manifest)
        first_probe = train_probe_fold(cache, metadata, protocol, args.output, 0, "cpu")
        first_head = train_head_fold(cache, metadata, protocol, args.output, 0, "cpu")
        changed = metadata.copy()
        changed.loc[changed.outer_fold == 0, "label"] ^= 1
        args2 = arguments(path, tmp_path / "second")
        protocol2 = initialize(args2, changed, manifest)
        second_probe = train_probe_fold(cache, changed, protocol2, args2.output, 0, "cpu")
        second_head = train_head_fold(cache, changed, protocol2, args2.output, 0, "cpu")
        for key in ("best_single_layer", "selected_layers", "best_epoch", "thresholds", "final_state_sha256"):
            assert first_probe["result"][key] == second_probe["result"][key], key
        for key in ("best_single_layer", "selected_layers", "best_epoch", "thresholds", "dev_bce"):
            assert first_head["result"][key] == second_head["result"][key], key
        first_states = torch.load(args.output / "fold_0/heads/heads.pt", weights_only=False)["states"]
        second_states = torch.load(args2.output / "fold_0/heads/heads.pt", weights_only=False)["states"]
        assert {arm: state_digest(state) for arm, state in first_states.items()} == {
            arm: state_digest(state) for arm, state in second_states.items()}
        # Different evaluation labels do affect held metrics, proving the altered labels are exercised.
        assert first_head["result"]["held_metrics"] != second_head["result"]["held_metrics"]


def test_full_cpu_diagnostic_covers_rows_and_refuses_mutated_resume(tmp_path):
    with small_cpu_pool():
        path = fixture_cache(tmp_path)
        cache, metadata, manifest = load_b_cache(path)
        args = arguments(path, tmp_path / "out")
        protocol = initialize(args, metadata, manifest)
        for fold in (0, 1):
            train_probe_fold(cache, metadata, protocol, args.output, fold, "cpu")
            train_head_fold(cache, metadata, protocol, args.output, fold, "cpu")
        report = write_report(args.output, protocol, metadata, heads=True)
        assert report["rows"] == 120
        assert set(MAIN_ARMS).issubset(report["head_metrics"])
        frame = pd.read_parquet(args.output / "analysis/head_outer_predictions.parquet")
        assert frame.base_id.tolist() == metadata.base_id.tolist()
        assert frame.base_id.is_unique
        assert all(value["rescued"] - value["harmed"] == value["net_correct"] for value in report["comparisons"])
        assert train_probe_fold(cache, metadata, protocol, args.output, 0, "cpu")["complete"]
        history = args.output / "fold_0/probes/history.json"
        history.write_text("[]\n")
        with pytest.raises(ValueError, match="artifact changed"):
            train_probe_fold(cache, metadata, protocol, args.output, 0, "cpu")
