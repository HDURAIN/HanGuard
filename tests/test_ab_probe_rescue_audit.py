"""The oracle metric may favor weak rules; it is not deployed fusion accuracy."""
import numpy as np
import pytest
from scripts.hanguard import ab_probe_rescue_audit as audit
from scripts.hanguard import ab_frozen_readout_diagnostics as readouts


def fixture():
    y = np.array([0, 0, 0, 1, 1, 1])
    anchor = np.array([.1, .1, .9, .1, .9, .9])
    probabilities = np.repeat(anchor[:, None], 32, axis=1)
    probabilities[:, 0] = .75
    probabilities[:, 1] = .25
    thresholds = np.full(32, .5)
    selection = readouts.choose_layers(y, probabilities, thresholds)
    selection['thresholds'] = thresholds.tolist()
    return y, probabilities, selection


def test_fixed_constants_have_perfect_oracle_without_instance_information():
    y = np.array([0, 0, 0, 1])
    result = audit.constant_control(y, np.array([0, 0, 1, 0]), first_greedy_rescue=0)
    assert result['constant_pair_correctness_union_accuracy'] == 1
    assert result['constant_zero']['accuracy'] == .75
    assert result['constant_one']['accuracy'] == .25
    assert not result['instance_features_used']
    assert result['no_attainable_model_gain_claimed']
    assert result['constant_score_AUROC'] == .5


def test_majority_constant_can_win_first_rescue_slot_despite_many_new_errors():
    y = np.array([0, 0, 1, 1, 1])
    anchor = np.array([0, 0, 0, 1, 1])
    result = audit.constant_control(y, anchor, first_greedy_rescue=0)
    assert result['majority_rescue_strictly_exceeds_registered_first_slot']
    assert result['hypothetical_majority_first_slot_rescue'] == 1
    assert sum(row['harmed'] for row in result['majority_transitions_by_label']) == 2
    assert result['hypothetical_candidates_were_not_added']


def test_registered_greedy_can_choose_weaker_fixed_single_class_candidates():
    y, probabilities, selection = fixture()
    assert selection['best_single_layer'] == 32
    assert {step['layer'] for step in selection['rescue_trace'][:2]} == {1, 2}
    assert [step['newly_rescued'] for step in selection['rescue_trace'][:2]] == [1, 1]
    assert selection['dev_probe_oracle_accuracy'] == 1
    assert selection['inner_dev_layer_metrics']['1']['accuracy'] < selection['inner_dev_layer_metrics']['32']['accuracy']
    assert audit.validate_saved_selection(selection)['accuracy'] == 4 / 6


def test_saved_counts_trace_and_oracle_must_be_consistent():
    y, probabilities, selection = fixture()
    selection['rescue_trace'][0]['newly_rescued'] += 1
    with pytest.raises(ValueError, match='oracle'):
        audit.validate_saved_selection(selection)
    y, probabilities, selection = fixture()
    selection['inner_dev_layer_metrics']['3']['tn'] += 1
    with pytest.raises(ValueError, match='counts|accuracy'):
        audit.validate_saved_selection(selection)


def test_rescue_harm_by_label_uses_anchor_predictions_and_net_counts():
    y = np.array([0, 0, 1, 1])
    anchor = np.array([0, 1, 1, 0])
    candidate = np.array([1, 0, 0, 1])
    result = audit.transitions_by_label(y, candidate, anchor)
    assert [(r['rescued'], r['harmed'], r['net_correct']) for r in result] == [(1, 1, 0), (1, 1, 0)]
    assert sum(row['label_rows'] for row in result) == len(y)


def test_scope_trace_is_oracle_and_actual_candidate_harms_are_separate():
    y, probabilities, selection = fixture()
    decisions = (probabilities >= .5).astype(int)
    metrics, transitions, trace, summary = audit.analyze_scope(0, 'inner_dev_replay', y, decisions,
                                                            selection, selection['inner_dev_layer_metrics'])
    assert len(metrics) == 32 and len(transitions) == 64 and len(trace) == 8
    assert summary['selected_probe_oracle_accuracy'] == 1
    assert summary['anchor_accuracy'] == 4 / 6
    harmful = [row for row in transitions if row['layer'] == 1]
    assert sum(row['harmed'] for row in harmful) == 2
    assert sum(row['rescued'] for row in harmful) == 1
    assert all(row['oracle_not_deployable'] for row in trace)


def test_hard_metric_counts_positive_rate_and_single_class_controls_are_explicit():
    result = audit.hard_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert result['positive_prediction_rate'] == .75
    assert result['positive_label_rate'] == .5
    assert (result['tp'], result['tn'], result['fp'], result['fn']) == (2, 1, 1, 0)
    with pytest.raises(ValueError, match='both binary classes'):
        audit.constant_control(np.array([1, 1]), np.array([1, 1]), 0)


def test_explanation_refuses_existing_destination_without_opening_model_or_holdouts(tmp_path):
    with pytest.raises(FileExistsError):
        audit.build_audit(tmp_path / 'does-not-exist', tmp_path)


def test_frechet_bounds_contain_true_label_transitions_and_exact_net_counts():
    y = np.array([0, 0, 1, 1])
    anchor, candidate = np.array([0, 1, 1, 0]), np.array([1, 0, 0, 1])
    exact = audit.transitions_by_label(y, candidate, anchor)
    bounds = audit.marginal_transition_bounds(audit.hard_metrics(y, anchor), audit.hard_metrics(y, candidate), first_rescue=2)
    for value, bound in zip(exact, bounds):
        assert bound['rescued_lower'] <= value['rescued'] <= bound['rescued_upper']
        assert bound['harmed_lower'] <= value['harmed'] <= bound['harmed_upper']
        assert bound['net_correct'] == value['net_correct']
    assert all(not bound['exact_joint_decisions_available'] for bound in bounds)


def test_saved_dev_uses_exact_metrics_but_labels_rescue_harm_are_bounds():
    _, _, selection = fixture()
    metrics, transitions, trace, summary = audit.analyze_saved_dev(0, selection)
    assert len(metrics) == 32 and len(transitions) == 64 and len(trace) == 6
    assert metrics[0]['positive_prediction_rate'] == 1
    assert metrics[1]['positive_prediction_rate'] == 0
    assert summary['selected_probe_oracle_accuracy'] == 1
    assert all('rescued_lower' in row and 'rescued' not in row for row in transitions)
    assert all('newly_rescued_lower' in row for row in trace)
