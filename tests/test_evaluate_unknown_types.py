"""CPU-only evaluator acceptance of independent binary labels and unknown types."""
import json

import pytest

import evaluate


ABSENT = object()


def row(category, harmful, *, source='jailbench', mask=ABSENT, predicted=None, raw=2):
    predicted = harmful if predicted is None else predicted
    result = dict(prompt='完整评估样本', category_id=category,
                  prompt_harm_label=harmful, harmful_pred=predicted,
                  category_pred=raw if predicted == 'harmful' else 0)
    result.update({f'category_p_{i}': .6 if i == raw else .1 for i in range(1, 6)})
    if source is not ABSENT:
        result['source'] = source
    if mask is not ABSENT:
        result['type_label_mask'] = mask
    return result


def mixed_rows():
    return [
        row(2, 'harmful', source='chinese_curated', mask=True),
        row(0, 'unharmful', source='chinese_curated', mask=False),
        row(-1, 'harmful', mask=False, predicted='unharmful'),
        row(-1, 'unharmful', mask=False, predicted='harmful', raw=4),
        row('untrusted legacy mapping', 'harmful', source='wildguard_zh', raw=1),
        row(-1, 'unharmful', source='wildguard_zh', mask=False),
    ]


def test_unknown_binary_targets_and_errors_are_retained_while_type_metrics_exclude_them():
    records = mixed_rows()
    before = json.loads(json.dumps(records))
    report = evaluate.evaluate_records(records)
    assert records == before
    assert report['binary']['rows'] == 6
    assert report['binary']['confusion_matrix'] == [[2, 1], [1, 2]]
    assert report['binary']['accuracy'] == pytest.approx(2 / 3)
    assert report['primary_category']['rows'] == 1
    assert report['primary_category']['accuracy'] == 1.
    assert report['end_to_end']['rows'] == 2
    assert report['end_to_end']['accuracy'] == 1.
    scope = report['category_scope']
    assert scope['excluded_category_rows'] == 4
    assert scope['excluded_harmful_rows'] == 2
    assert scope['unknown_type_rows_excluded'] == 2
    assert scope['unknown_type_harmful_rows_excluded'] == 1
    assert scope['unknown_type_source_counts'] == {'jailbench': 2}
    assert scope['out_of_scope_source_rows'] == 2
    assert scope['out_of_scope_harmful_rows'] == 1
    assert scope['excluded_source_counts'] == {'wildguard_zh': 2}
    jailbench = report['by_source']['jailbench']
    assert jailbench['binary']['rows'] == 2 and jailbench['binary']['accuracy'] == 0.
    assert jailbench['end_to_end']['rows'] == 0
    assert jailbench['category_scope']['out_of_scope_source_rows'] == 0
    assert jailbench['category_scope']['unknown_type_rows_excluded'] == 2


def test_generic_unknown_types_keep_safe_known_rows_in_six_way_metrics():
    records = [row(0, 'unharmful', source=ABSENT, mask=False),
               row(-1, 'harmful', source=ABSENT, mask=False, predicted='unharmful'),
               row(-1, 'unharmful', source=ABSENT, mask=False)]
    report = evaluate.evaluate_records(records)
    assert report['binary']['rows'] == 3 and report['binary']['accuracy'] == pytest.approx(2 / 3)
    assert report['end_to_end']['rows'] == 1 and report['end_to_end']['accuracy'] == 1.
    assert report['primary_category']['rows'] == 0
    assert report['category_scope']['unknown_type_rows_excluded'] == 2
    assert report['category_scope']['out_of_scope_source_rows'] == 0
    assert report['category_scope']['mode'] == 'user_provided_labels_without_source'


@pytest.mark.parametrize('source', ['jailbench', 'chinese_curated', ABSENT])
@pytest.mark.parametrize('mask', [ABSENT, True])
def test_unknown_type_requires_explicit_false_boolean_mask(source, mask):
    with pytest.raises(ValueError, match='type_label_mask'):
        evaluate.evaluate_records([row(-1, 'harmful', source=source, mask=mask)])


@pytest.mark.parametrize('source', ['jailbench', 'wildguard_zh', ABSENT])
@pytest.mark.parametrize('mask', [None, 'False', 'True', 0, 1, 0., 1., [], {}])
def test_malformed_type_masks_are_rejected_without_truthiness_coercion(source, mask):
    with pytest.raises(ValueError, match='type_label_mask'):
        evaluate.evaluate_records([row(-1, 'harmful', source=source, mask=mask)])


@pytest.mark.parametrize('category,harmful,mask', [(0, 'unharmful', True), (1, 'harmful', False),
                                                 (5, 'harmful', False)])
def test_known_types_cannot_be_silently_removed_or_safe_types_supervised(category, harmful, mask):
    with pytest.raises(ValueError, match='type_label_mask'):
        evaluate.evaluate_records([row(category, harmful, mask=mask)])


@pytest.mark.parametrize('category,harmful', [(0, 'harmful'), (1, 'unharmful'), (5, 'unharmful')])
def test_known_type_cannot_override_an_inconsistent_source_binary_label(category, harmful):
    with pytest.raises(ValueError, match='有害性与类别不一致'):
        evaluate.evaluate_records([row(category, harmful, mask=category > 0)])


@pytest.mark.parametrize('category', [-2, 6, 2.5, None, True, 'bad category'])
def test_in_scope_invalid_categories_are_rejected_even_with_false_mask(category):
    with pytest.raises(ValueError, match='Category'):
        evaluate.evaluate_records([row(category, 'harmful', mask=False)])


@pytest.mark.parametrize('category', [-1, '-1', '-1.0'])
def test_explicit_wildguard_unknown_true_mask_is_rejected(category):
    with pytest.raises(ValueError, match='type_label_mask'):
        evaluate.evaluate_records([row(category, 'harmful', source='wildguard_zh', mask=True)])


@pytest.mark.parametrize('category', ['invalid legacy value', None, 99, 5])
def test_wildguard_other_categories_do_not_derive_or_gate_binary_truth(category):
    report = evaluate.evaluate_records([row(category, 'unharmful', source='wildguard_zh', mask=True)])
    assert report['binary']['confusion_matrix'] == [[1, 0], [0, 0]]
    assert report['end_to_end']['rows'] == 0
    assert report['category_scope']['unknown_type_rows_excluded'] == 0
    assert report['category_scope']['excluded_source_counts'] == {'wildguard_zh': 1}


def test_known_legacy_inputs_can_still_omit_type_mask():
    report = evaluate.evaluate_records([row(0, 'unharmful'), row(2, 'harmful')])
    assert report['binary']['accuracy'] == 1.
    assert report['primary_category']['rows'] == 1
    assert report['end_to_end']['rows'] == 2


def test_prediction_contract_stays_strict_for_unknown_truth_types():
    record = row(-1, 'harmful', mask=False)
    record['category_pred'] = -1
    with pytest.raises(ValueError, match='Category'):
        evaluate.evaluate_records([record])
    record = row(-1, 'harmful', mask=False)
    record['category_pred'] = 3
    with pytest.raises(ValueError, match='argmax'):
        evaluate.evaluate_records([record])


def test_preds_cli_reports_unknown_native_types_separately_from_outside_sources(tmp_path, capsys):
    input_path = tmp_path / 'predictions.json'
    input_path.write_text(json.dumps(mixed_rows(), ensure_ascii=False))
    output = tmp_path / 'report.json'
    assert evaluate.main(['--preds', str(input_path), '--output', str(output)]) == 0
    text = capsys.readouterr().out
    assert '2 条范围外来源记录' in text
    assert '2 条显式未知类型记录' in text
    report = json.loads(output.read_text())
    assert report['binary']['rows'] == 6
    assert report['by_source']['jailbench']['category_scope']['out_of_scope_source_rows'] == 0
