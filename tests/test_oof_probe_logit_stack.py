import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.hanguard.oof_probe_logit_stack as oof

from scripts.hanguard.oof_probe_logit_stack import (
    ALL_LAYERS,
    BranchProbes,
    _predict_probe_pair,
    _validate_probe_checkpoint_payload,
    _valid_matched_l32_prediction_diagnostic,
    apply_platt,
    assert_matched_branch_initialization,
    crossfit_stack_scores,
    fit_platt,
    greedy_learned_stack_layers,
    group_equal_weights,
    layer_slot,
    make_group_folds,
    paired_arm_comparisons,
    select_calibration_family,
    select_nested_stack_regularization,
    select_stack_regularization,
    synchronize_matched_l32,
    validation_gate,
    verify_test_barrier,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grouped_sample(seed=7, group_count=60, rows_per_group=3):
    rng = np.random.default_rng(seed)
    group_label = np.tile([0, 1], group_count // 2)
    groups = np.repeat([f"g{i:03d}" for i in range(group_count)], rows_per_group)
    labels = np.repeat(group_label, rows_per_group)
    folds = make_group_folds(labels, groups, n_splits=5, seed=19)
    return rng, labels, groups, folds


def test_group_equal_weights_give_each_group_the_same_total_mass():
    groups = np.asarray(["a", "a", "a", "b", "c", "c"])
    weights = group_equal_weights(groups)
    totals = [weights[groups == group].sum() for group in np.unique(groups)]
    assert np.allclose(totals, totals[0])
    assert weights.mean() == pytest.approx(1.0)


def test_group_folds_are_deterministic_and_never_split_a_group():
    _, labels, groups, folds = _grouped_sample()
    assert np.array_equal(folds, make_group_folds(labels, groups, n_splits=5, seed=19))
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    for fold in np.unique(folds):
        heldout_groups = set(groups[folds == fold])
        training_groups = set(groups[folds != fold])
        assert heldout_groups.isdisjoint(training_groups)


def test_matched_branch_probes_begin_byte_identically_and_are_independent():
    pool = BranchProbes(hidden_size=8, branches=5, seed=42, slot_ids=[0, 1, 2, 3, 3])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    identity = assert_matched_branch_initialization(pool, repeat, layer32_index=4)
    assert len(identity["slot_initialization_sha256"]) == 4
    assert len(set(identity["slot_initialization_sha256"])) == 4
    for slot in range(4):
        assert torch.equal(pool.weight[slot], repeat.weight[slot])
    assert not torch.equal(repeat.weight[0], repeat.weight[1])
    assert [layer_slot(layer) for layer in (8, 16, 24, 32)] == [0, 1, 2, 3]
    assert pool.weight.untyped_storage().data_ptr() != repeat.weight.untyped_storage().data_ptr()
    features = torch.randn(6, 5, 8)
    logits = pool(features)
    assert logits.shape == (6, 5)


def test_matched_l32_sync_canonicalizes_parameters_and_adam_state():
    pool = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    pool_optimizer = torch.optim.AdamW(pool.parameters(), lr=0.01)
    repeat_optimizer = torch.optim.AdamW(repeat.parameters(), lr=0.01)
    pool.weight.grad = torch.ones_like(pool.weight)
    pool.bias.grad = torch.ones_like(pool.bias)
    repeat.weight.grad = torch.zeros_like(repeat.weight)
    repeat.bias.grad = torch.zeros_like(repeat.bias)
    pool_optimizer.step(); repeat_optimizer.step()
    diagnostic = synchronize_matched_l32(
        pool, repeat, 3, pool_optimizer, repeat_optimizer
    )
    assert diagnostic["raw_parameter_max_absolute_difference"] > 0
    assert diagnostic["canonical_parameter_max_absolute_difference"] == 0
    assert torch.equal(pool.weight[3], repeat.weight[3])
    assert torch.equal(pool.bias[3], repeat.bias[3])
    for pool_parameter, repeat_parameter in (
        (pool.weight, repeat.weight), (pool.bias, repeat.bias)
    ):
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                pool_optimizer.state[pool_parameter][key][3],
                repeat_optimizer.state[repeat_parameter][key][3],
            )
        assert torch.equal(
            pool_optimizer.state[pool_parameter]["step"],
            repeat_optimizer.state[repeat_parameter]["step"],
        )


def test_matched_l32_sync_rejects_step_mismatch_before_mutating_repeat():
    pool = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    pool_optimizer = torch.optim.AdamW(pool.parameters(), lr=0.01)
    repeat_optimizer = torch.optim.AdamW(repeat.parameters(), lr=0.01)
    pool.weight.grad = torch.ones_like(pool.weight)
    pool.bias.grad = torch.ones_like(pool.bias)
    repeat.weight.grad = torch.zeros_like(repeat.weight)
    repeat.bias.grad = torch.zeros_like(repeat.bias)
    pool_optimizer.step(); repeat_optimizer.step()
    repeat_optimizer.state[repeat.weight]["step"].add_(1)
    before_weight = repeat.weight[3].detach().clone()
    before_bias = repeat.bias[3].detach().clone()
    with pytest.raises(ValueError, match="step counters"):
        synchronize_matched_l32(pool, repeat, 3, pool_optimizer, repeat_optimizer)
    assert torch.equal(repeat.weight[3], before_weight)
    assert torch.equal(repeat.bias[3], before_bias)


def test_probe_prediction_stores_one_exact_canonical_l32_score():
    pool = BranchProbes(hidden_size=8, branches=32, seed=42, slot_ids=[(i // 8) for i in range(32)])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    with torch.no_grad():
        repeat.weight[3].add_(0.1)
    cache = np.random.default_rng(7).normal(size=(9, 32, 8)).astype(np.float16)
    layer_logits, repeat_logits, diagnostic = _predict_probe_pair(
        pool, repeat, cache, np.arange(9), ALL_LAYERS, batch_size=4
    )
    assert diagnostic["raw_gpu_l32_logit_max_absolute_difference"] > 0
    assert diagnostic["canonical_l32_logit_max_absolute_difference"] == 0
    assert np.array_equal(layer_logits[:, 31], repeat_logits[:, 3])


def test_probe_prediction_rejects_nonfinite_raw_slot_before_canonical_copy():
    pool = BranchProbes(hidden_size=8, branches=32, seed=42, slot_ids=[(i // 8) for i in range(32)])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    with torch.no_grad():
        repeat.weight[3, 0] = float("nan")
    cache = np.random.default_rng(7).normal(size=(3, 32, 8)).astype(np.float16)
    with pytest.raises(FloatingPointError, match="raw base-probe"):
        _predict_probe_pair(pool, repeat, cache, np.arange(3), ALL_LAYERS, batch_size=2)


def test_checkpoint_validation_requires_complete_canonical_history():
    layers = (8, 16, 24, 32)
    pool = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    repeat = BranchProbes(hidden_size=8, branches=4, seed=42, slot_ids=[0, 1, 2, 3])
    identity = {"fold": 0}
    history = [
        {
            "epoch": epoch,
            "pool_weighted_bce_sum": 1.0,
            "repeat_weighted_bce_sum": 1.0,
            "raw_l32_parameter_max_absolute_difference": 1e-7,
            "canonical_l32_parameter_max_absolute_difference": 0.0,
        }
        for epoch in (1, 2)
    ]
    checkpoint = {
        "pool_state": pool.state_dict(),
        "repeat_state": repeat.state_dict(),
        "candidate_layers": list(layers),
        "hidden_size": 8,
        "identity": identity,
        "history": history,
    }
    protocol = {"hidden_size": 8, "probes": {"epochs": 2}}
    _validate_probe_checkpoint_payload(checkpoint, layers, identity, protocol, "test")
    checkpoint["history"] = history[:1]
    with pytest.raises(ValueError, match="incomplete"):
        _validate_probe_checkpoint_payload(checkpoint, layers, identity, protocol, "test")


def test_prediction_diagnostic_rejects_nonfinite_or_noncanonical_values():
    valid = {
        "raw_gpu_l32_logit_max_absolute_difference": 1e-6,
        "canonical_l32_logit_max_absolute_difference": 0.0,
        "convention": "repeat slot 3 is the canonical pool L32 estimator",
    }
    assert _valid_matched_l32_prediction_diagnostic(valid)
    assert not _valid_matched_l32_prediction_diagnostic({**valid, "raw_gpu_l32_logit_max_absolute_difference": float("nan")})
    assert not _valid_matched_l32_prediction_diagnostic({**valid, "canonical_l32_logit_max_absolute_difference": 1e-8})


def test_crossfit_resume_manifest_and_barrier_cover_canonical_evidence(tmp_path, monkeypatch):
    output = tmp_path / "output"
    source = tmp_path / "source"
    features = source / "features"
    features.mkdir(parents=True)
    rows = 30
    metadata = pd.DataFrame(
        {
            "base_id": [f"b{index:03d}" for index in range(rows)],
            "group_id": [f"g{index:03d}" for index in range(rows)],
            "label": np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2),
        }
    )
    metadata_path = features / "train.metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    candidate_layers = [8, 16, 24, 32]
    protocol = {
        "source": str(source),
        "hidden_size": 8,
        "candidate_layers": candidate_layers,
        "folds": {"count": 3, "inner_count": 2, "seed": 19},
        "probes": {"seed": 42, "epochs": 2, "batch_size": 7},
        "train_metadata": {
            "rows": rows,
            "sha256": oof.sha256(metadata_path),
            "ordered_base_id_sha256": oof.ordered_base_id_sha256(metadata.base_id),
        },
        "external_oof": None,
    }
    output.mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    cache = np.random.default_rng(3).normal(size=(rows, 32, 8)).astype(np.float16)

    def fake_train(_cache, _metadata, _indices, layers, value, _seed):
        pool = BranchProbes(8, len(layers), 42, [layer_slot(layer) for layer in layers])
        repeat = BranchProbes(8, 4, 42, [0, 1, 2, 3])
        initial = assert_matched_branch_initialization(pool, repeat, list(layers).index(32))
        history = [
            {
                "epoch": epoch,
                "pool_weighted_bce_sum": 1.0,
                "repeat_weighted_bce_sum": 1.0,
                "raw_l32_parameter_max_absolute_difference": 0.0,
                "canonical_l32_parameter_max_absolute_difference": 0.0,
            }
            for epoch in range(1, value["probes"]["epochs"] + 1)
        ]
        return pool, repeat, history, initial

    monkeypatch.setattr(oof, "load_protocol", lambda _output: protocol)
    monkeypatch.setattr(oof, "_load_cache", lambda _output, _protocol, _split: (cache, metadata))
    monkeypatch.setattr(oof, "_train_probe_pair", fake_train)
    args = SimpleNamespace(output=output)
    oof.crossfit_probes(args)
    base = output / "base_probes"
    (base / "oof_logits.npz").unlink()
    (base / "oof_logits.manifest.json").unlink()
    # Rebuild only the aggregate outputs while exercising every per-fit resume
    # check against the immutable checkpoints, histories, and raw diagnostics.
    oof.crossfit_probes(args)
    bundle = oof.validate_oof_bundle(output, protocol)
    oof.load_nested_bundles(output, protocol, bundle)
    barrier_paths = {path.name for path in oof._barrier_artifacts(output, protocol)}
    assert "folds.parquet" in barrier_paths
    assert "full_history.json" in barrier_paths
    assert "fold_0_history.json" in barrier_paths
    assert "nested_pair_0_1_history.json" in barrier_paths


def test_crossfitted_stack_and_platt_are_finite_and_predictive():
    rng, labels, groups, folds = _grouped_sample()
    signal = 2.5 * (2 * labels - 1) + rng.normal(0, 0.4, len(labels))
    features = np.column_stack([signal, rng.normal(size=len(labels)), signal * 0.4])
    result = select_stack_regularization(features, labels, groups, folds, [0.03, 0.3], seed=11)
    assert np.isfinite(result["probability"]).all()
    assert result["metrics"]["group_equal"]["auroc"] > 0.98
    assert result["calibrator"]["slope"] > 0
    raw = crossfit_stack_scores(features, labels, groups, folds, 0.3, seed=11)
    calibrator = fit_platt(raw, labels, group_equal_weights(groups))
    assert np.logical_and(apply_platt(calibrator, raw) > 0, apply_platt(calibrator, raw) < 1).all()


def test_calibration_family_selection_compares_all_methods_and_locks_one_topology():
    rng, labels, groups, folds = _grouped_sample()
    signed = 2 * labels - 1
    features = np.column_stack([signed + rng.normal(0, 0.7, len(labels))] * 4)
    result = select_calibration_family(features, labels, groups, folds, [0.1], seed=8)
    assert set(result["comparisons"]) == {"identity", "temperature", "platt"}
    assert result["selected"] in result["comparisons"]
    assert result["comparisons"][result["selected"]]["parameter_count"] in (0, 1, 2)


def _outer_records(features, labels, groups, folds):
    records = []
    for outer in sorted(np.unique(folds)):
        training = np.flatnonzero(folds != outer)
        heldout = np.flatnonzero(folds == outer)
        other = [value for value in sorted(np.unique(folds)) if value != outer]
        mapping = {value: index for index, value in enumerate(other)}
        inner = np.asarray([mapping[value] for value in folds[training]], dtype=np.int64)
        records.append(
            {
                "outer_fold": int(outer),
                "train_indices": training,
                "heldout_indices": heldout,
                "inner_fold": inner,
                "train_features": features[training],
                "heldout_features": features[heldout],
            }
        )
    return records


def test_nested_stack_predictions_are_outer_group_heldout_and_calibrator_uses_them():
    rng, labels, groups, folds = _grouped_sample(group_count=40)
    signed = 2 * labels - 1
    features = np.column_stack(
        [2 * signed + rng.normal(0, 0.4, len(labels)), rng.normal(size=len(labels))]
    )
    records = _outer_records(features, labels, groups, folds)
    result = select_nested_stack_regularization(
        features, records, labels, groups, folds, [0.1], seed=9,
        calibration_method="temperature",
    )
    assert result["metric_scope"] == "strict nested outer-group-held-out"
    assert result["metrics"]["group_equal"]["auroc"] > 0.98
    assert result["calibrator"]["fit_source"] == "strict nested outer-heldout stack scores"
    broken = [dict(value) for value in records]
    broken[0]["train_indices"] = np.r_[broken[0]["train_indices"], broken[0]["heldout_indices"][0]]
    broken[0]["train_features"] = np.vstack(
        [broken[0]["train_features"], broken[0]["heldout_features"][0]]
    )
    broken[0]["inner_fold"] = np.r_[broken[0]["inner_fold"], 0]
    with pytest.raises(AssertionError, match="registered outer fold"):
        select_nested_stack_regularization(
            features, broken, labels, groups, folds, [0.1], seed=9,
            calibration_method="temperature",
        )


def test_greedy_layer_selection_is_train_oof_and_keeps_anchor():
    rng, labels, groups, folds = _grouped_sample(group_count=40)
    signed = 2 * labels - 1
    layers = [1, 2, 3, 32]
    logits = np.column_stack(
        [
            rng.normal(size=len(labels)),
            3.0 * signed + rng.normal(0, 0.25, len(labels)),
            rng.normal(size=len(labels)),
            0.2 * signed + rng.normal(0, 1.0, len(labels)),
        ]
    )
    selected = greedy_learned_stack_layers(
        logits, layers, labels, groups, folds, [0.1], count=2, anchor=32, seed=3
    )
    assert selected["layers"] == [2, 32]
    assert selected["greedy_order"][0] == 32
    assert selected["trace"][0]["added_layer"] == 2


def test_four_layer_selection_uses_exactly_one_layer_per_registered_slot():
    rng, labels, groups, folds = _grouped_sample(group_count=40)
    signed = 2 * labels - 1
    layers = [1, 2, 9, 10, 17, 18, 31, 32]
    logits = np.column_stack(
        [
            rng.normal(size=len(labels)),
            2.0 * signed + rng.normal(0, 0.4, len(labels)),
            rng.normal(size=len(labels)),
            1.8 * signed + rng.normal(0, 0.4, len(labels)),
            rng.normal(size=len(labels)),
            1.6 * signed + rng.normal(0, 0.4, len(labels)),
            rng.normal(size=len(labels)),
            0.3 * signed + rng.normal(0, 0.8, len(labels)),
        ]
    )
    selected = greedy_learned_stack_layers(
        logits, layers, labels, groups, folds, [0.1], count=4, anchor=32, seed=4
    )
    assert {layer_slot(layer) for layer in selected["layers"]} == {0, 1, 2, 3}


def test_validation_gate_has_registered_pass_and_fallback_behavior():
    control = {"group_equal": {"bce": 0.20}, "row": {"accuracy": 0.90, "auroc": 0.95}}
    rules = {"minimum_bce_gain": 0.001, "accuracy_tolerance": 0.002, "auroc_tolerance": 0.001}
    passing = {"group_equal": {"bce": 0.19}, "row": {"accuracy": 0.899, "auroc": 0.950}}
    failing = {"group_equal": {"bce": 0.19}, "row": {"accuracy": 0.89, "auroc": 0.950}}
    assert validation_gate(passing, control, rules)["passed"] is True
    assert validation_gate(failing, control, rules)["passed"] is False


def test_each_paired_comparison_uses_its_own_candidate_probabilities():
    labels = np.asarray([0, 0, 1, 1])
    groups = np.asarray(["a", "b", "c", "d"])
    probabilities = {
        "repeat32": np.asarray([0.1, 0.9, 0.1, 0.9]),
        "fixed4": np.asarray([0.1, 0.2, 0.8, 0.9]),
        "selected4": np.asarray([0.9, 0.8, 0.2, 0.1]),
    }
    result = paired_arm_comparisons(labels, groups, probabilities, repetitions=32, seed=3)
    assert result["fixed4"]["accuracy_delta"] == pytest.approx(0.5)
    assert result["selected4"]["accuracy_delta"] == pytest.approx(-0.5)
    assert result["fixed4"]["bootstrap"]["accuracy"]["delta"] == pytest.approx(0.5)
    assert result["selected4"]["bootstrap"]["accuracy"]["delta"] == pytest.approx(-0.5)


def test_test_barrier_detects_any_locked_artifact_change(tmp_path):
    protocol = tmp_path / "protocol.json"
    artifact = tmp_path / "selection.json"
    test_input = tmp_path / "test.npy"
    protocol.write_text('{"registered": true}')
    artifact.write_text('{"selected_arm": "repeat32"}')
    test_input.write_bytes(b"opaque-test-bytes")
    barrier = {
        "locked": True,
        "protocol_sha256": _sha(protocol),
        "artifacts": {str(artifact): _sha(artifact)},
        "test_inputs": {str(test_input): _sha(test_input)},
        "selected_arm": "repeat32",
        "threshold": 0.5,
    }
    (tmp_path / "test_barrier.json").write_text(json.dumps(barrier))
    assert verify_test_barrier(tmp_path)["locked"] is True
    artifact.write_text('{"selected_arm": "selected4"}')
    with pytest.raises(ValueError, match="changed"):
        verify_test_barrier(tmp_path)
