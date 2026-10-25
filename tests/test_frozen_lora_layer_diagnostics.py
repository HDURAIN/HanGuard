"""CPU-only checks for the frozen-LoRA layer diagnostic protocol.

The tests concentrate on the parts that protect the interpretation of the
experiment: validation-only layer selection, paired error accounting,
group-aware inference, matched head capacity/initialization, and the test
selection barrier.  They deliberately do not load a tokenizer or backbone.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.hanguard.frozen_lora_layer_diagnostics import (
    LastTokenBlockCapture,
    assert_capacity_matched_initialization,
    create_test_barrier,
    error_complementarity,
    group_cluster_bootstrap,
    group_permutation_test,
    last_true_indices,
    ordered_base_id_sha256,
    select_complementary_layers,
    validate_feature_cache,
    validate_lora_load,
    verify_test_barrier,
)
from scripts.hanguard.repaired_heads import ExperimentHead


def _probabilities(labels, flipped=()):
    prediction = np.asarray(labels, dtype=int).copy()
    prediction[np.asarray(tuple(flipped), dtype=int)] ^= 1
    return np.where(prediction == 1, .9, .1)


def test_last_token_capture_uses_final_true_position_for_each_mask_row():
    mask = torch.tensor([
        [0, 0, 1, 1, 1],  # left padding
        [1, 1, 1, 0, 0],  # right padding
        [0, 1, 0, 1, 0],  # deliberately non-contiguous validity
    ], dtype=torch.bool)
    indices = last_true_indices(mask)
    torch.testing.assert_close(indices, torch.tensor([4, 2, 3]))

    backbone = SimpleNamespace(
        layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)]),
    )
    capture = LastTokenBlockCapture(backbone)
    hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2).float()
    capture.last_indices = indices
    capture.active = True
    backbone.layers[0](hidden)
    capture.active = False
    expected = hidden[torch.arange(3), indices].to(torch.float16)
    torch.testing.assert_close(capture.values[1], expected)
    capture.close()

    with pytest.raises(ValueError, match='true token'):
        last_true_indices(torch.zeros(2, 3, dtype=torch.bool))


def test_feature_cache_validator_rejects_extent_nonfinite_and_id_drift(tmp_path):
    shape = (3, 2, 4)
    ids = ['a', 'b', 'c']
    expected_hash = ordered_base_id_sha256(ids)

    valid = tmp_path / 'valid.npy'
    np.save(valid, np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
    result = validate_feature_cache(valid, shape, np.float16, expected_hash, ids, chunk_rows=1)
    assert result['finite_rows_checked'] == 3
    assert result['ordered_base_id_sha256'] == expected_hash

    wrong_extent = tmp_path / 'wrong_extent.npy'
    wrong_extent.write_bytes(valid.read_bytes() + b'extra')
    with pytest.raises(ValueError, match='byte size'):
        validate_feature_cache(wrong_extent, shape, np.float16, expected_hash, ids)

    nonfinite = tmp_path / 'nonfinite.npy'
    values = np.ones(shape, dtype=np.float16)
    values[1, 0, 0] = np.nan
    np.save(nonfinite, values)
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        validate_feature_cache(nonfinite, shape, np.float16, expected_hash, ids)

    with pytest.raises(ValueError, match='base_id'):
        validate_feature_cache(valid, shape, np.float16, expected_hash, ids[::-1])


def test_frozen_adapter_validation_rejects_missing_lora_parameters():
    adapter = {'base.layers.0.q_proj.lora_A.default.weight': torch.ones(2, 2)}
    loaded = SimpleNamespace(missing_keys=[], unexpected_keys=[])
    assert validate_lora_load(adapter, loaded)['missing_lora_keys'] == 0

    missing = SimpleNamespace(
        missing_keys=['base.layers.0.q_proj.lora_B.default.weight'],
        unexpected_keys=[],
    )
    with pytest.raises(ValueError, match='missing LoRA'):
        validate_lora_load(adapter, missing)
    with pytest.raises(ValueError, match='empty'):
        validate_lora_load({}, loaded)


def test_layer_selection_is_validation_only_anchored_and_uses_error_rescue():
    labels = np.array([0] * 6 + [1] * 6)
    probabilities = {
        # Layer 32 misses rows 0 and 6.  Layer 8 is the only candidate that
        # rescues both; the others rescue at most one of those errors.
        8: _probabilities(labels, (1,)),
        12: _probabilities(labels, (0, 6)),
        16: _probabilities(labels, (0, 2)),
        24: _probabilities(labels, (3, 6)),
        32: _probabilities(labels, (0, 6)),
    }
    selected = select_complementary_layers(
        labels, probabilities, split='validation', count=4, anchor=32,
        auc_tolerance=1.0,
    )

    assert len(selected['layers']) == 4
    assert 32 in selected['layers']
    assert len(set(selected['layers'])) == 4
    assert selected['trace'][0]['layer'] == 32
    assert selected['trace'][1]['layer'] == 8
    assert set(selected['eligibility']) == set(probabilities)

    # The split assertion makes accidental test-set selection an executable
    # protocol violation rather than a convention in the report text.
    with pytest.raises(ValueError, match='validation'):
        select_complementary_layers(
            labels, probabilities, split='test', count=4, anchor=32,
            auc_tolerance=1.0,
        )


def test_error_complementarity_has_exact_four_cells_and_derived_rates():
    labels = np.array([0, 0, 1, 1, 0, 1, 0])
    probability_a = np.array([.1, .9, .9, .1, .1, .9, .9])
    probability_b = np.array([.1, .1, .1, .9, .9, .9, .9])

    result = error_complementarity(
        labels, probability_a, probability_b, threshold_a=.5,
        threshold_b=.5,
    )

    assert result['both_correct'] == 2
    assert result['a_only_correct'] == 2
    assert result['b_only_correct'] == 2
    assert result['both_wrong'] == 1
    assert sum(result[key] for key in (
        'both_correct', 'a_only_correct', 'b_only_correct', 'both_wrong')) == len(labels)
    assert result['a_errors'] == 3
    assert result['b_errors'] == 3
    assert result['shared_errors'] == 1
    assert result['net_correct_a_minus_b'] == 0
    assert result['decision_disagreement_rate'] == pytest.approx(4 / 7)
    assert result['error_jaccard'] == pytest.approx(1 / 5)
    assert result['oracle_accuracy'] == pytest.approx(6 / 7)


def test_group_bootstrap_and_permutation_are_deterministic_and_find_gain():
    # Every complete group has one control-only error.  Sampling and swapping
    # whole groups should therefore find a stable +25 percentage-point gain.
    groups = np.repeat(np.arange(40), 4)
    labels = np.tile(np.array([0, 1, 0, 1]), 40)
    candidate = _probabilities(labels)
    control = _probabilities(labels, np.arange(0, len(labels), 4))

    arguments = dict(
        labels=labels, probability_a=candidate, probability_b=control,
        groups=groups, threshold_a=.5, threshold_b=.5,
        n_resamples=4096, seed=917,
    )
    bootstrap_a = group_cluster_bootstrap(**arguments)
    bootstrap_b = group_cluster_bootstrap(**arguments)
    permutation_a = group_permutation_test(**arguments)
    permutation_b = group_permutation_test(**arguments)

    assert bootstrap_a == bootstrap_b
    assert permutation_a == permutation_b
    assert bootstrap_a['observed'] == pytest.approx(.25)
    assert bootstrap_a['ci_low'] > 0
    assert bootstrap_a['ci_high'] >= bootstrap_a['ci_low']
    assert permutation_a['observed'] == pytest.approx(.25)
    assert 0 <= permutation_a['p_value'] < .01


def test_group_inference_resamples_and_swaps_clusters_not_individual_rows():
    labels = np.tile(np.array([0, 1]), 50)
    candidate = _probabilities(labels)
    control = _probabilities(labels, np.arange(0, len(labels), 2))
    # With one cluster, a cluster bootstrap can only reproduce the observed
    # data.  A two-sided cluster permutation has only +delta and -delta, so it
    # cannot claim significance regardless of the number of rows inside it.
    arguments = dict(
        labels=labels, probability_a=candidate, probability_b=control,
        groups=np.repeat('single_group', len(labels)), threshold_a=.5,
        threshold_b=.5, n_resamples=1024, seed=81,
    )
    bootstrap = group_cluster_bootstrap(**arguments)
    permutation = group_permutation_test(**arguments)

    assert bootstrap['observed'] == pytest.approx(.5)
    assert bootstrap['ci_low'] == pytest.approx(.5)
    assert bootstrap['ci_high'] == pytest.approx(.5)
    assert permutation['p_value'] >= .45


def _matched_heads(seed=19):
    specifications = {
        'repeat_last': ('repeat_last', (8, 16, 24, 32)),
        'fusion_fixed': ('fusion', (8, 16, 24, 32)),
        'fusion_selected': ('fusion', (4, 13, 27, 32)),
    }
    result = {}
    for name, (mode, layers) in specifications.items():
        torch.manual_seed(seed)
        result[name] = ExperimentHead(
            hidden_size=12, mode=mode, width=8, layers=layers, dropout=.1,
        )
    return result


def test_capacity_match_requires_equal_parameter_count_and_initial_state():
    heads = _matched_heads()
    assert len({sum(parameter.numel() for parameter in head.parameters())
                for head in heads.values()}) == 1
    assert_capacity_matched_initialization(heads)

    with torch.no_grad():
        heads['fusion_selected'].classifier.bias.add_(1.)
    with pytest.raises(ValueError, match='initial'):
        assert_capacity_matched_initialization(heads)

    heads = _matched_heads()
    heads['small_last'] = ExperimentHead(
        hidden_size=12, mode='last', width=8, layers=(32,), dropout=.1,
    )
    with pytest.raises(ValueError, match='capacity'):
        assert_capacity_matched_initialization(heads)


def _write(path: Path, value: str):
    path.write_text(value, encoding='utf-8')
    return path


def test_test_barrier_hashes_protocol_probe_and_every_head_selection(tmp_path):
    protocol = _write(tmp_path / 'protocol.json', '{"version": 1}')
    test_data = _write(tmp_path / 'test.parquet', 'fixed test bytes')
    probes = {
        'all_layers': _write(tmp_path / 'probe_selection.json', '{"epoch": 3}'),
    }
    heads = {
        name: _write(tmp_path / f'{name}.json', '{"epoch": 2}')
        for name in ('repeat_s42', 'fixed_s42', 'selected_s42')
    }

    barrier = create_test_barrier(
        protocol_path=protocol, probe_selection_paths=probes,
        head_selection_paths=heads, test_dataset_path=test_data,
    )
    assert barrier['locked'] is True
    assert set(barrier['probe_selection_sha256']) == set(probes)
    assert set(barrier['head_selection_sha256']) == set(heads)
    assert barrier['protocol_sha256']
    assert barrier['test_dataset_sha256']
    assert verify_test_barrier(
        barrier, protocol_path=protocol, probe_selection_paths=probes,
        head_selection_paths=heads, test_dataset_path=test_data,
    )

    with pytest.raises(ValueError, match='locked'):
        verify_test_barrier(
            dict(barrier, locked=False), protocol_path=protocol,
            probe_selection_paths=probes, head_selection_paths=heads,
            test_dataset_path=test_data,
        )

    for path, changed in (
        (protocol, '{"version": 2}'),
        (probes['all_layers'], '{"epoch": 99}'),
        (test_data, 'different test bytes'),
    ):
        original = path.read_text(encoding='utf-8')
        path.write_text(changed, encoding='utf-8')
        with pytest.raises(ValueError, match='protocol|probe|test|hash|changed'):
            verify_test_barrier(
                barrier, protocol_path=protocol, probe_selection_paths=probes,
                head_selection_paths=heads, test_dataset_path=test_data,
            )
        path.write_text(original, encoding='utf-8')

    # A changed artifact and an omitted selected head must both keep the test
    # set closed.  Checking only the protocol hash would miss either leak.
    heads['fixed_s42'].write_text('{"epoch": 99}', encoding='utf-8')
    with pytest.raises(ValueError, match='selection|hash|changed'):
        verify_test_barrier(
            barrier, protocol_path=protocol, probe_selection_paths=probes,
            head_selection_paths=heads, test_dataset_path=test_data,
        )
    heads['fixed_s42'].write_text('{"epoch": 2}', encoding='utf-8')
    with pytest.raises(ValueError, match='selection|missing'):
        verify_test_barrier(
            barrier, protocol_path=protocol, probe_selection_paths=probes,
            head_selection_paths={key: value for key, value in heads.items()
                                  if key != 'selected_s42'},
            test_dataset_path=test_data,
        )
