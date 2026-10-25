import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.hanguard import build_selected_test_subset as builder


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_inputs(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    for split in builder.SPLITS:
        frame = pd.DataFrame({
            'base_id': [f'{split}-{i}' for i in range(4)],
            'group_id': [f'group-{split}-{i}' for i in range(4)],
            'split': [split] * 4,
            'source': ['wildguard_zh', 'wildguard_zh', 'chinese_curated', 'wildguard_zh'],
            'prompt': ['A complete request.', 'Another full request.', 'A valid request.', 'Hello.'],
            'prompt_harm_label': ['unharmful', 'harmful', 'unharmful', 'harmful'],
            'category_id': [0, 1, 0, 1], 'type_label_mask': [False, True, False, True]})
        frame.to_parquet(parent / f'{split}.parquet', index=False)
    write_json(parent / 'split_policy.json', {'same_split': True})
    write_json(parent / 'manifest.json', {
        'dataset_contract': builder.contract.CONTRACT, 'dataset_version': 'parent'})
    write_json(parent / 'release_qa.json', {
        'release_status': 'qa_complete_with_documented_limitations',
        'manifest_sha256': builder.sha(parent / 'manifest.json'),
        'split_sha256': {split: builder.sha(parent / f'{split}.parquet') for split in builder.SPLITS}})
    frame = pd.read_parquet(parent / 'test.parquet')
    predictions = frame[['base_id', 'group_id', 'source', 'split']].copy()
    predictions['label'] = [0, 1, 0, 1]
    for arm in builder.ARMS:
        decisions = [0, 1, 0, 0] if arm != builder.FUSION else [1, 1, 1, 1]
        predictions[f'prediction_{arm}'] = decisions
        predictions[f'probability_{arm}'] = [.9 if value else .1 for value in decisions]
        predictions[f'threshold_{arm}'] = .5
    prediction_path = tmp_path / 'predictions.parquet'
    predictions.to_parquet(prediction_path, index=False)
    predictions.to_csv(prediction_path.with_suffix('.csv'), index=False)
    protocol = tmp_path / 'protocol.json'
    write_json(protocol, {'old_protocol': True})
    results = tmp_path / 'results.json'
    write_json(results, {'complete': True, 'rows': 4,
                         'protocol_sha256': builder.sha(protocol)})
    return parent, prediction_path, protocol, results


def check_subset(inputs, tmp_path):
    parent, predictions, protocol, results = inputs
    parent_hashes = {path.name: builder.sha(path) for path in parent.iterdir()}
    output = tmp_path / 'subset'
    manifest = builder.build_dataset(parent, output, predictions, protocol, results,
                                     expected_excluded=1)
    assert manifest['selection_policy'] == 'posthoc_error_selected_test_subset'
    assert manifest['is_quality_cleanup'] is False
    assert manifest['excluded_test_rows'] == 1
    assert {path.name: builder.sha(path) for path in parent.iterdir()} == parent_hashes
    for split in ('train', 'validation'):
        assert builder.sha(output / f'{split}.parquet') == builder.sha(parent / f'{split}.parquet')
    original = pd.read_parquet(parent / 'test.parquet')
    expected = original.loc[original.base_id.ne('test-0')].reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(output / 'test.parquet'), expected)
    assert pd.read_parquet(output / 'excluded122.parquet').base_id.tolist() == ['test-0']
    assert builder.contract.release_contract(output)['dataset_version'] == 'subset'
    metrics = json.loads((output / 'old_model_subset_metrics.json').read_text())
    assert metrics['full_parent_test']['correct']['best_single'] == 3
    assert metrics['selected_test_subset']['correct']['best_single'] == 2
    assert metrics['selected_test_subset']['correct'][builder.FUSION] == 2


def corrupt_predictions(inputs, change):
    parent, path, protocol, results = inputs
    frame = pd.read_parquet(path)
    if change == 'label':
        frame.loc[0, 'label'] = 1
    elif change == 'duplicate':
        frame.loc[0, 'base_id'] = frame.loc[1, 'base_id']
    elif change == 'source':
        frame.loc[0, 'source'] = 'chinese_curated'
    else:
        frame.loc[0, 'prediction_best_single'] = 1
    frame.to_parquet(path, index=False)


class SelectedTestSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = make_inputs(self.root)

    def test_subset_preserves_every_retained_value_and_split_bytes(self):
        check_subset(self.inputs, self.root)

    def test_wrong_expected_count_rejected_before_writing(self):
        parent, predictions, protocol, results = self.inputs
        output = self.root / 'subset'
        with self.assertRaisesRegex(ValueError, 'Expected 122 selected rows, got 1'):
            builder.build_dataset(parent, output, predictions, protocol, results)
        self.assertFalse(output.exists())

    def test_invalid_or_mismatched_predictions_rejected(self):
        for change in ('label', 'duplicate', 'source', 'prediction'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as path:
                root = Path(path)
                inputs = make_inputs(root)
                corrupt_predictions(inputs, change)
                parent, predictions, protocol, results = inputs
                with self.assertRaises(ValueError):
                    builder.build_dataset(parent, root / 'subset', predictions,
                                          protocol, results, expected_excluded=1)
                self.assertFalse((root / 'subset').exists())


if __name__ == '__main__':
    unittest.main()
