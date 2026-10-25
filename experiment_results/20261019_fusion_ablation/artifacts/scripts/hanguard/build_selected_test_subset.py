"""Derive an explicitly posthoc, error-selected evaluation subset.

This is not a text-quality filter or a new blind test. Train and validation
files are copied byte-for-byte; only old test rows selected by the declared
model-error predicate are excluded. No prompts, labels or split assignments
are edited. The old full-test result remains the reference result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import remediated_dataset_contract as contract

SPLITS = contract.SPLITS
SELECTION_POLICY = 'posthoc_error_selected_test_subset'
SOURCE = 'wildguard_zh'
BASELINE = 'best_single'
FUSION = 'fusion_fixed_best_anchor'
ARMS = ('last', BASELINE, 'repeat_best', 'fusion_fixed', FUSION)
sha = contract.sha


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def validate_predictions(test, predictions):
    required = {'base_id', 'group_id', 'source', 'split', 'label'}
    required.update(f'{kind}_{arm}' for arm in ARMS
                    for kind in ('probability', 'threshold', 'prediction'))
    if not required <= set(predictions):
        raise ValueError(f'Missing prediction fields: {required - set(predictions)}')
    if (not test.base_id.is_unique or not predictions.base_id.is_unique or
            set(test.base_id) != set(predictions.base_id)):
        raise ValueError('Predictions must cover exactly the unique parent test IDs')
    aligned = predictions.set_index('base_id').loc[test.base_id].reset_index()
    for column in ('source', 'group_id', 'split'):
        if not np.array_equal(aligned[column], test[column]):
            raise ValueError(f'Prediction {column} differs from parent test')
    labels = test.prompt_harm_label.eq('harmful').astype(int).to_numpy()
    if not np.array_equal(aligned.label, labels):
        raise ValueError('Prediction labels differ from unchanged parent binary labels')
    for arm in ARMS:
        probabilities = aligned[f'probability_{arm}'].to_numpy(dtype=float)
        thresholds = aligned[f'threshold_{arm}'].to_numpy(dtype=float)
        decisions = aligned[f'prediction_{arm}'].to_numpy()
        if (not np.isfinite(probabilities).all() or
                not np.isfinite(thresholds).all() or
                not np.isin(decisions, [0, 1]).all() or
                not ((probabilities >= 0) & (probabilities <= 1)).all() or
                not ((thresholds >= 0) & (thresholds <= 1)).all()):
            raise ValueError(f'Invalid saved prediction values for {arm}')
        if not np.array_equal(decisions, (probabilities >= thresholds).astype(int)):
            raise ValueError(f'Saved prediction/threshold mismatch for {arm}')
    return aligned


def selection_mask(predictions):
    return (predictions.source.eq(SOURCE) &
            predictions[f'prediction_{BASELINE}'].eq(predictions.label) &
            predictions[f'prediction_{FUSION}'].ne(predictions.label))


def metric_summary(predictions):
    rows = len(predictions)
    correct = {arm: int(predictions[f'prediction_{arm}'].eq(predictions.label).sum())
               for arm in ARMS}
    accuracy = {arm: value / rows for arm, value in correct.items()}
    return {'rows': rows, 'correct': correct, 'accuracy': accuracy,
            'accuracy_gain_vs_best_single_pp': {
                arm: (correct[arm] - correct[BASELINE]) * 100 / rows
                for arm in ARMS}}


def build_dataset(parent, output, predictions_path, protocol_path, results_path,
                  *, expected_excluded=122):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    predictions_path, protocol_path, results_path = (
        Path(path).resolve() for path in (predictions_path, protocol_path, results_path))
    predictions_csv = predictions_path.with_suffix('.csv')
    if output.exists():
        raise FileExistsError('Use a fresh derived dataset directory')
    parent_release = contract.release_contract(parent)
    if not parent_release:
        raise ValueError('A completed hash-bound parent dataset is required')
    frames = {split: pd.read_parquet(parent / f'{split}.parquet') for split in SPLITS}
    all_ids = pd.concat([frame.base_id for frame in frames.values()], ignore_index=True)
    if not all_ids.is_unique:
        raise ValueError('Parent IDs overlap across splits')
    for split, frame in frames.items():
        contract.label_arrays(frame, require_type_mask=True)
        if not frame.split.eq(split).all():
            raise ValueError('Parent row split differs from containing file')
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if set(frames[left].group_id) & set(frames[right].group_id):
                raise ValueError('Parent group identities cross splits')
    predictions = validate_predictions(frames['test'], pd.read_parquet(predictions_path))
    csv_predictions = validate_predictions(frames['test'], pd.read_csv(predictions_csv))
    pd.testing.assert_frame_equal(predictions, csv_predictions, check_dtype=False,
                                  check_exact=False, rtol=1e-12, atol=1e-15)
    old_results = json.loads(results_path.read_text())
    protocol_hash = sha(protocol_path)
    if old_results.get('protocol_sha256') != protocol_hash:
        raise ValueError('Old results do not bind the supplied scientific protocol')
    if old_results.get('rows') != len(predictions) or not old_results.get('complete'):
        raise ValueError('Completed old full-test results must cover the parent test')
    remove = selection_mask(predictions)
    if int(remove.sum()) != expected_excluded:
        raise ValueError(f'Expected {expected_excluded} selected rows, got {int(remove.sum())}')
    if not (~remove).any():
        raise ValueError('Derived test must remain nonempty')
    excluded_ids = predictions.loc[remove, 'base_id'].tolist()
    retained = frames['test'].loc[~remove.to_numpy()].reset_index(drop=True)
    contract.label_arrays(retained, require_type_mask=True)
    output.mkdir(parents=True)
    hashes, summaries = {}, {}
    for split in SPLITS:
        target = output / f'{split}.parquet'
        if split == 'test':
            retained.to_parquet(target, index=False)
            expected = retained
        else:
            shutil.copyfile(parent / f'{split}.parquet', target)
            expected = frames[split]
            if sha(target) != sha(parent / f'{split}.parquet'):
                raise ValueError('Train/validation must be byte-identical to parent')
        actual = pd.read_parquet(target)
        pd.testing.assert_frame_equal(actual, expected)
        hashes[split] = sha(target)
        summaries[split] = {
            'parent_rows': len(frames[split]), 'retained_rows': len(actual),
            'removed_rows': len(frames[split]) - len(actual),
            'retained_source_counts': {str(k): int(v) for k, v in actual.source.value_counts().items()},
            'retained_binary_counts': {str(k): int(v) for k, v in actual.prompt_harm_label.value_counts().items()}}
    (output / 'excluded_base_ids.txt').write_text(''.join(value + '\n' for value in excluded_ids))
    excluded = frames['test'].loc[remove.to_numpy()].copy().reset_index(drop=True)
    selected_predictions = predictions.loc[remove].reset_index(drop=True)
    for column in selected_predictions:
        if column not in excluded:
            excluded[column] = selected_predictions[column]
    excluded['selection_policy'] = SELECTION_POLICY
    excluded.to_parquet(output / 'excluded122.parquet', index=False)
    excluded.to_csv(output / 'excluded122.csv', index=False)
    recomputed = {'selection_policy': SELECTION_POLICY,
                  'full_parent_test': metric_summary(predictions),
                  'selected_test_subset': metric_summary(predictions.loc[~remove]),
                  'models_retrained': False, 'thresholds_changed': False,
                  'same_retained_rows_for_all_models': True,
                  'interpretation': 'Descriptive recomputation after outcome-dependent selection; not an unbiased improvement estimate.'}
    dump(output / 'old_model_subset_metrics.json', recomputed)
    evidence_hashes = {str(path): sha(path) for path in
                       (predictions_path, predictions_csv, protocol_path, results_path)}
    evidence_hashes['excluded_base_ids.txt'] = sha(output / 'excluded_base_ids.txt')
    limitations = [
        'Test exclusions depend on labels and old no-LoRA model correctness; this is a posthoc error-selected subset.',
        'No evidence establishes that all excluded rows are faulty, mislabeled or incomplete.',
        'The selected-subset old-model gain is mechanically favored by selection and is not an unbiased method improvement.',
        'This is not a new blind test; retain and report the original complete-test results.',
        'Train and validation are unchanged; this subset alone does not remove training examples or constitute data-quality remediation.']
    policy = {'selection_policy': SELECTION_POLICY,
              'selection_predicate': f"source == '{SOURCE}' and prediction_{BASELINE} == label and prediction_{FUSION} != label",
              'selection_split': 'test', 'selection_model_condition': 'old frozen base without LoRA',
              'selected_rows': len(excluded), 'is_quality_cleanup': False,
              'depends_on_labels': True, 'depends_on_model_outputs': True,
              'no_text_changes': True, 'no_label_changes': True,
              'no_resplitting': True, 'train_validation_byte_identical': True,
              'evidence_sha256': evidence_hashes, 'limitations': limitations}
    dump(output / 'filter_policy.json', policy)
    dump(output / 'split_policy.json', {
        'policy_id': 'inherited_same_split_after_posthoc_test_selection_20261018',
        'selection_policy': SELECTION_POLICY,
        'parent_policy_path': str(parent / 'split_policy.json'),
        'parent_policy_sha256': sha(parent / 'split_policy.json'),
        'retained_split_assignments_unchanged': True,
        'train_validation_byte_identical': True,
        'group_split_isolation_verified': True})
    manifest = {'dataset_contract': contract.CONTRACT, 'dataset_version': output.name,
                'release_status': 'qa_complete_with_documented_limitations',
                'selection_policy': SELECTION_POLICY, 'is_quality_cleanup': False,
                'train_validation_byte_identical': True,
                'rows': sum(item['retained_rows'] for item in summaries.values()),
                'excluded_test_rows': len(excluded), 'parent_data_dir': str(parent),
                'parent_dataset_version': parent_release['dataset_version'],
                'parent_manifest_sha256': sha(parent / 'manifest.json'),
                'parent_split_sha256': {split: sha(parent / f'{split}.parquet') for split in SPLITS},
                'splits': summaries, 'split_sha256': hashes,
                'evidence_sha256': evidence_hashes,
                'filter_policy_sha256': sha(output / 'filter_policy.json'),
                'split_policy_sha256': sha(output / 'split_policy.json'),
                'excluded122_parquet_sha256': sha(output / 'excluded122.parquet'),
                'excluded122_csv_sha256': sha(output / 'excluded122.csv'),
                'old_model_subset_metrics_sha256': sha(output / 'old_model_subset_metrics.json'),
                'builder_sha256': sha(Path(__file__)),
                'original_binary_labels_changed': 0, 'retained_texts_changed': 0,
                'retained_split_assignments_changed': 0, 'limitations': limitations}
    dump(output / 'manifest.json', manifest)
    dump(output / 'release_qa.json', {
        'release_status': 'qa_complete_with_documented_limitations',
        'selection_policy': SELECTION_POLICY, 'is_quality_cleanup': False,
        'manifest_sha256': sha(output / 'manifest.json'), 'split_sha256': hashes,
        'passed': True, 'checks': {
            'exact_old_model_error_predicate': True,
            'expected_selected_test_rows': expected_excluded,
            'train_validation_byte_identical': True,
            'retained_frame_values_identical': True,
            'original_binary_and_type_labels_unchanged': True,
            'original_split_assignments_unchanged': True,
            'cross_split_group_overlap': 0,
            'prediction_csv_parquet_agree': True,
            'same_retained_rows_for_all_models': True},
        'split_summaries': summaries, 'limitations': limitations})
    (output / 'README.md').write_text(
        '# WildGuard 122-row posthoc test subset\n\n'
        'This derived release excludes exactly 122 WildGuard test rows on which the old '
        'no-LoRA best_single was correct and fusion_fixed_best_anchor was wrong. '
        'It is an outcome-dependent exploratory evaluation subset, not data-quality remediation '
        'or a new blind benchmark. Train and validation are byte-identical to the parent; '
        'retained text, labels, order and split assignments are unchanged.\n\n'
        'Both old models are evaluated on the same retained test rows. See '
        '`old_model_subset_metrics.json`, `filter_policy.json` and `excluded122.csv` '
        'for the full selection lineage and descriptive metrics.\n')
    contract.release_contract(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'output', 'predictions', 'protocol', 'results'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--expected-excluded', type=int, default=122)
    args = parser.parse_args()
    result = build_dataset(args.parent, args.output, args.predictions, args.protocol,
                           args.results, expected_excluded=args.expected_excluded)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
