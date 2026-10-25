"""CPU contracts for the input-conditioned, capacity-matched layer gate."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import dynamic_gate_diagnostics as dynamic
from scripts.hanguard.dynamic_gate_diagnostics import (
    ARMS,
    PRIMARY_THRESHOLD,
    DynamicLayerGateHead,
    aggregate_static_comparisons,
    arm_layers,
    build_matched_heads,
    build_parser,
    discover_static_comparison_seeds,
    gate_diagnostics,
    select_arm_features,
    validation_thresholds,
)
from scripts.hanguard.frozen_lora_layer_diagnostics import state_digest


SELECTED = (17, 18, 28, 32)
UNION = (8, 16, 17, 18, 24, 28, 32)


def test_dynamic_gate_outputs_probability_weights_that_change_by_input():
    torch.manual_seed(7)
    head = DynamicLayerGateHead(hidden_size=9, width=6, gate_width=4, dropout=0.)
    features = torch.randn(11, 4, 9)
    output = head(features)

    assert output['logits'].shape == (11,)
    assert output['representation'].shape == (11, 6)
    assert output['gate_logits'].shape == (11, 4)
    assert output['gate_weights'].shape == (11, 4)
    torch.testing.assert_close(output['gate_weights'].sum(1), torch.ones(11))
    assert torch.isfinite(output['gate_weights']).all()
    assert float(output['gate_weights'].std(0).max()) > 0

    with pytest.raises(ValueError, match='Expected'):
        head(torch.randn(11, 3, 9))


def test_repeat_control_has_identical_capacity_and_small_nonzero_residual_initialization():
    heads, audit = build_matched_heads(
        hidden_size=9, width=6, gate_width=4, seed=31, device='cpu',
    )
    assert audit['arms'] == list(ARMS)
    assert len({sum(parameter.numel() for parameter in head.parameters())
                for head in heads.values()}) == 1
    assert len({state_digest(head) for head in heads.values()}) == 1
    expected = torch.eye(6) * 1e-3
    for head in heads.values():
        torch.testing.assert_close(head.fusion_output.weight, expected)
        assert float(head.fusion_output.weight.abs().sum()) > 0


def test_repeat_control_runs_the_exact_gate_when_all_layer_inputs_are_equal():
    torch.manual_seed(11)
    reference = DynamicLayerGateHead(7, width=5, gate_width=3, dropout=0.)
    candidate = copy.deepcopy(reference)
    final = torch.randn(6, 7)
    repeated = final[:, None, :].expand(-1, 4, -1)

    a = reference(repeated)
    b = candidate(repeated.clone())
    for key in ('logits', 'representation', 'gate_logits', 'gate_weights', 'layer_values'):
        torch.testing.assert_close(a[key], b[key])

    union = torch.randn(6, len(UNION), 7)
    control = select_arm_features(union, UNION, 'dynamic_repeat_last', SELECTED)
    expected = union[:, -1:, :].expand(-1, 4, -1)
    torch.testing.assert_close(control, expected)
    assert arm_layers('dynamic_repeat_last', SELECTED) == (32, 32, 32, 32)


def test_dynamic_gate_gradients_reach_conditioner_and_every_value_branch():
    torch.manual_seed(97)
    head = DynamicLayerGateHead(7, width=5, gate_width=3, dropout=0.)
    features = torch.randn(19, 4, 7, requires_grad=True)
    result = head(features)
    loss = result['logits'].square().mean()
    loss.backward()

    for parameter in (
        head.gate_query.weight,
        head.gate_key.weight,
        head.gate_score.weight,
        head.layer_bias,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0
    for branch in head.layer_values:
        assert float(branch[0].weight.grad.abs().sum()) > 0
    assert all(float(features.grad[:, slot].abs().sum()) > 0 for slot in range(4))


def test_primary_threshold_is_fixed_and_validation_tuning_is_sensitivity_only():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([.4, .6, .7, .8])
    result = validation_thresholds(labels, probabilities)
    assert result['primary'] == PRIMARY_THRESHOLD == .5
    assert result['validation_accuracy_sensitivity'] != result['primary']
    assert build_parser().parse_args(['init']).seeds == [42, 43, 44, 45, 46]


def test_gate_diagnostics_distinguish_uniform_and_input_conditioned_weights():
    uniform = np.full((5, 4), .25)
    labels = np.array([0, 0, 1, 1, 1])
    result = gate_diagnostics(uniform, SELECTED, labels)
    assert result['normalized_mean_entropy'] == pytest.approx(1.)
    assert result['mean_effective_layers'] == pytest.approx(4.)
    assert result['mean_l1_from_global_mean'] == pytest.approx(0.)
    assert result['maximum_slot_std'] == pytest.approx(0.)
    assert set(result['mean_by_label']) == {'0', '1'}

    conditional = np.array([
        [.7, .1, .1, .1],
        [.1, .7, .1, .1],
        [.1, .1, .7, .1],
        [.1, .1, .1, .7],
    ])
    dynamic = gate_diagnostics(conditional, SELECTED)
    assert dynamic['mean_l1_from_global_mean'] > 0
    assert dynamic['maximum_slot_std'] > 0
    assert dynamic['normalized_mean_entropy'] < 1

    with pytest.raises(ValueError, match='probabilities'):
        gate_diagnostics(np.ones((3, 4)), SELECTED)


def test_layer_contract_rejects_unanchored_selection_or_incomplete_union():
    with pytest.raises(ValueError, match='anchored'):
        arm_layers('dynamic_selected', (1, 2, 3, 4))
    with pytest.raises(ValueError, match='lacks'):
        select_arm_features(
            torch.randn(2, 2, 5), (8, 32), 'dynamic_fixed', SELECTED,
        )


def test_static_comparison_uses_only_real_seed_intersection(tmp_path):
    source = tmp_path / "source"
    for seed in (42, 43, 44, 99):
        directory = source / "heads" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "test_predictions.parquet").write_bytes(f"pred-{seed}".encode())
        (directory / "selection.json").write_text(f'{{"seed": {seed}}}')
    incomplete = source / "heads/seed_45"
    incomplete.mkdir(parents=True)
    (incomplete / "selection.json").write_text('{"seed": 45}')

    result = discover_static_comparison_seeds(
        {"source": str(source)}, [42, 43, 44, 45, 46]
    )

    assert result["dynamic_available_seeds"] == [42, 43, 44, 45, 46]
    assert result["source_static_available_seeds"] == [42, 43, 44, 99]
    assert result["common_seeds"] == [42, 43, 44]
    assert result["n_common_seeds"] == 3
    assert result["excluded_dynamic_seeds"] == [45, 46]
    assert result["missing_source_artifacts"] == {
        "45": ["test_predictions.parquet"],
        "46": ["test_predictions.parquet", "selection.json"],
    }
    assert set(result["source_artifact_sha256"]) == {"42", "43", "44"}


def _static_rows(seeds=(42, 43, 44)):
    rows = []
    for candidate, baseline in dynamic.STATIC_ARM_PAIRS:
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "baseline": baseline,
                    "group_bootstrap": {
                        "accuracy": {"delta": (seed - 43) / 1000},
                        "f1": {"delta": (seed - 42) / 1000},
                    },
                }
            )
    return rows


def test_static_aggregates_and_report_disclose_n_and_seed_list(tmp_path, monkeypatch):
    aggregates = aggregate_static_comparisons(_static_rows())
    assert all(row["n_seeds"] == 3 for row in aggregates)
    assert all(row["seeds"] == [42, 43, 44] for row in aggregates)

    scalar = {"mean": 0.9, "std": 0.01, "minimum": 0.8, "maximum": 1.0, "seeds": 5}
    seed_aggregates = [
        {
            "arm": arm,
            "test_metrics": {key: scalar for key in ("accuracy", "f1", "auroc", "bce")},
            "gate": {
                key: scalar
                for key in ("normalized_mean_entropy", "mean_effective_layers")
            },
        }
        for arm in ARMS
    ]
    matched = [
        {
            "candidate": arm,
            "accuracy_delta": scalar,
            "f1_delta": scalar,
            "positive_accuracy_seeds": 3,
            "seeds": [42, 43, 44, 45, 46],
        }
        for arm in ("dynamic_fixed", "dynamic_selected")
    ]
    protocol = {
        "training": {"seeds": [42, 43, 44, 45, 46]},
        "selected_layers": list(SELECTED),
        "fixed_layers": list(dynamic.FIXED_LAYERS),
        "limitations": [],
    }
    result = {
        "evaluation_amendment_sha256": "a" * 64,
        "primary_reproducibility_audit_sha256": "b" * 64,
        "post_barrier_evaluation_amendment": {
            "applied": True,
            "locked_evaluation_code_sha256": "c" * 64,
            "intersection_patch_code_sha256": "d" * 64,
            "amended_evaluation_code_sha256": "e" * 64,
            "secondary_static_seed_scope": [42, 43, 44],
        },
        "seed_aggregates": seed_aggregates,
        "matched_seed_aggregates": matched,
        "secondary_static_seed_aggregates": aggregates,
        "secondary_static_seed_scope": {
            "common_seeds": [42, 43, 44],
            "n_common_seeds": 3,
            "excluded_dynamic_seeds": [45, 46],
        },
    }
    monkeypatch.setattr(dynamic, "load_protocol", lambda output: protocol)
    report = dynamic.write_report(tmp_path, result).read_text()
    assert "seeds=[42, 43, 44]（n=3）" in report
    assert "dynamic-only seeds=[45, 46]" in report
    assert "主等容量比较仍为完整 5 seeds" in report
    assert "| 3 | [42, 43, 44] |" in report
    assert "Post-barrier evaluator amendment" in report
    assert "原 test barrier 未修改或重锁" in report
    assert "seed 42–45 精确重算审计" in report
    assert "a" * 64 in report
    assert "b" * 64 in report


def _prediction_frame(rows=3):
    frame = pd.DataFrame(
        {
            "base_id": [f"id-{index}" for index in range(rows)],
            "group_id": [f"group-{index // 2}" for index in range(rows)],
            "label": np.arange(rows, dtype=np.int8) % 2,
        }
    )
    for arm_index, arm in enumerate(ARMS):
        probability = np.linspace(.1, .9, rows, dtype=np.float64) + arm_index * .001
        frame[f"probability_{arm}"] = probability
        frame[f"prediction_{arm}"] = (probability >= .5).astype(np.int8)
        for slot in range(1, 5):
            frame[f"gate_{arm}_slot_{slot}"] = np.full(rows, .25, dtype=np.float32)
    return frame


def test_primary_recomputation_evidence_requires_bit_exact_frame():
    existing = _prediction_frame()
    evidence = dynamic._prediction_frame_evidence(existing, existing.copy())
    assert evidence["all_exact"] is True
    assert evidence["ordered_base_id_sha256"]
    assert evidence["ordered_group_id_sha256"]
    assert all(
        row["maximum_absolute_difference"] == 0
        for row in evidence["numeric_columns"].values()
    )

    changed = existing.copy()
    changed.loc[1, "probability_dynamic_fixed"] = np.nextafter(
        changed.loc[1, "probability_dynamic_fixed"], np.inf
    )
    with pytest.raises(ValueError, match="not exactly identical"):
        dynamic._prediction_frame_evidence(existing, changed)

    reordered = existing.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="not exactly identical"):
        dynamic._prediction_frame_evidence(existing, reordered)


def test_changed_evaluator_requires_amendment_before_any_recovery(tmp_path, monkeypatch):
    protocol = {"code_sha256": {"dynamic_gate_diagnostics.py": "0" * 64}}
    monkeypatch.setattr(dynamic, "_barrier_paths", lambda output, value: [])
    monkeypatch.setattr(dynamic, "_barrier_identity", lambda output, value: {})
    monkeypatch.setattr(dynamic.frozen, "verify_test_barrier", lambda *args, **kwargs: {})

    with pytest.raises(FileNotFoundError, match="evaluation_amendment.json is required"):
        dynamic.verify_evaluation_amendment(
            tmp_path, protocol, require_reproducibility=False
        )


def test_evaluate_checks_amendment_before_returning_cached_results(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text('{"stale": true}')
    monkeypatch.setattr(dynamic, "load_protocol", lambda output: {"protocol": "locked"})

    def reject(*args, **kwargs):
        raise RuntimeError("amendment gate reached")

    monkeypatch.setattr(dynamic, "verify_evaluation_amendment", reject)
    with pytest.raises(RuntimeError, match="amendment gate reached"):
        dynamic.evaluate(SimpleNamespace(output=tmp_path, device="cpu"))


def test_amendment_artifact_record_detects_post_lock_mutation(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked")
    record = dynamic._artifact_record(artifact)
    assert dynamic._verify_artifact_record("fixture", record) == artifact.resolve()

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        dynamic._verify_artifact_record("fixture", record)


def test_run_resumes_locked_evaluation_without_init_or_training(tmp_path, monkeypatch):
    (tmp_path / "test_barrier.json").write_text('{"state": "locked"}')
    calls = []
    monkeypatch.setattr(dynamic, "initialize", lambda args: pytest.fail("must not initialize"))
    monkeypatch.setattr(dynamic, "train", lambda args: pytest.fail("must not train"))
    monkeypatch.setattr(dynamic, "lock", lambda args: pytest.fail("must not relock"))
    monkeypatch.setattr(
        dynamic, "evaluate", lambda args: calls.append("evaluate") or {"status": "ok"}
    )
    monkeypatch.setattr(
        dynamic, "write_report", lambda output, result: calls.append((output, result))
    )

    result = dynamic.run_all(SimpleNamespace(output=tmp_path))

    assert result == {"status": "ok"}
    assert calls == ["evaluate", (tmp_path.resolve(), result)]
