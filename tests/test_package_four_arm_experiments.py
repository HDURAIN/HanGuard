"""Small synthetic archives verify scope separation, pending states and privacy."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.hanguard.package_four_arm_experiments import (
    ARMS, DEFAULT_ARCHIVE, DEFAULT_STUDY, SEEDS, export_predictions,
    metric_rows, read_json, refresh, tables, verify, write_json,
)


def task_result(task, seed, accuracy=.8):
    result = dict(task=task, seed=seed, predictions_sha256='1' * 64,
                  rootselection_lock_sha256='2' * 64, arms={})
    for i, arm in enumerate(ARMS):
        if task == 'binary':
            result['arms'][arm] = {
                'full_original_test': {'validation_threshold': dict(accuracy=accuracy + i * .01, f1=.8, n=8350)},
                'fixed_selected122_subset': {'validation_threshold': dict(accuracy=accuracy + i * .02, f1=.8, n=8228)},
            }
        else:
            result['arms'][arm] = dict(native_known_harmful=dict(accuracy=accuracy + i * .01, macro_f1=.7, rows=2180))
    return result


class PackageFourArmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / DEFAULT_STUDY
        self.archive = self.root / DEFAULT_ARCHIVE

    def tearDown(self):
        self.temp.cleanup()

    def test_pending_snapshot_has_no_old_score_or_fabricated_mean(self):
        index = refresh(self.root)
        self.assertEqual(index['status'], 'pending')
        self.assertFalse(index['registered'])
        self.assertEqual(len(index['coverage']), 12)
        self.assertEqual(verify(self.archive)['status'], 'pending')
        with (self.archive / 'results_summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(row['status'] == 'pending' and row['accuracy_percent'] == '' for row in rows))
        with (self.archive / 'seed_summary.csv').open() as stream:
            means = list(csv.DictReader(stream))
        self.assertEqual(len(means), 20)
        self.assertTrue(all(row['mean_accuracy_percent'] == '' for row in means))

    def test_actual_evaluator_schema_scopes_and_completed_three_seed_mean(self):
        write_json(self.study / 'protocol.json', {'tasks': ['binary', 'category']})
        systems = {}
        for seed in SEEDS:
            for task in ('binary', 'category'):
                write_json(self.study / 'heads' / task / f's{seed}' / 'test_results.json', task_result(task, seed))
            arms = {arm: dict(metrics=dict(accuracy=.8 + i * .01, macro_f1=.7, rows=3704))
                    for i, arm in enumerate(ARMS)}
            systems[str(seed)] = dict(systems={name: dict(arms=arms) for name in (
                'classic_gate_type_factorial', 'fusion_in_both_stages_factorial')})
        write_json(self.study / 'analysis' / 'results.json', dict(complete=True, per_seed=systems))
        index = refresh(self.root)
        self.assertTrue(index['complete'])
        self.assertTrue(verify(self.archive)['complete'])
        rows, means, coverage = tables(self.study, self.root)
        self.assertEqual({row['rows'] for row in rows}, {8350, 8228, 2180, 3704})
        self.assertTrue(all(row['status'] == 'complete' for row in means))
        self.assertTrue(all(row['sample_std_accuracy_pp'] == 0 for row in means))
        # No seed-42 exception: missing seed 44 means no three-seed average.
        (self.study / 'heads' / 'binary' / 's44' / 'test_results.json').unlink()
        rows, means, coverage = tables(self.study, self.root)
        pending = next(row for row in means if row['task'] == 'binary')
        self.assertEqual(pending['status'], 'pending')
        self.assertNotIn('mean_accuracy_percent', pending)

    def test_only_test_receipts_and_exact_denominators_are_accepted(self):
        data = task_result('binary', 42)
        self.assertEqual(len(metric_rows(data, 'binary', 42, 'source')), 8)
        for change in ({'complete': False}, {'seed': 43}, {'predictions_sha256': None}):
            changed = dict(data, **change)
            self.assertEqual(metric_rows(changed, 'binary', 42, 'source'), [])
        data['arms']['baseline']['full_original_test']['validation_threshold']['n'] = 8228
        with self.assertRaises(ValueError):
            metric_rows(data, 'binary', 42, 'source')

    def test_corpus_fields_are_removed_and_prediction_numbers_preserved(self):
        write_json(self.study / 'protocol.json', dict(prompt='PRIVATE', nested={'text': 'PRIVATE', 'seed': 42}))
        folder = self.study / 'heads' / 'binary' / 's42'
        folder.mkdir(parents=True)
        source = folder / 'validation_predictions.csv'
        source.write_text('base_id,binary_label,known_native_category,baseline_p0,baseline_p1,fusion_only_argmax,prompt,unseen_text\n'
                          'a,0,True,0.8,0.2,0,PRIVATE,PRIVATE\n', encoding='utf-8')
        refresh(self.root)
        copied = self.archive / 'artifacts' / source.relative_to(self.root)
        self.assertNotIn('PRIVATE', copied.read_text())
        self.assertIn('baseline_p1', copied.read_text())
        protocol = self.archive / 'artifacts' / (self.study / 'protocol.json').relative_to(self.root)
        self.assertEqual(read_json(protocol), {'nested': {'seed': 42}})
        copied.write_text(copied.read_text() + 'tampered\n')
        with self.assertRaises(ValueError):
            verify(self.archive)

    def test_prediction_export_rejects_nonfinite_and_duplicate_ids(self):
        source, target = self.root / 'source.csv', self.root / 'target.csv'
        for body in ('a,NaN\n', 'a,0.2\na,0.3\n'):
            source.write_text('base_id,baseline_p1\n' + body)
            with self.subTest(body=body), self.assertRaises(ValueError):
                export_predictions(source, target)

    def test_parquet_predictions_convert_to_text_free_numeric_csv(self):
        import pandas as pd
        source, target = self.root / 'predictions.parquet', self.root / 'predictions.csv'
        pd.DataFrame(dict(base_id=['a'], binary_label=[1], known_native_category=[True],
                          fusion_category_p0=[.2], fusion_category_p1=[.8],
                          prompt=['PRIVATE'])).to_parquet(source, index=False)
        receipt = export_predictions(source, target)
        self.assertEqual(receipt['rows'], 1)
        self.assertNotIn('PRIVATE', target.read_text())
        self.assertIn('fusion_category_p1', receipt['columns'])

    def test_refresh_removes_only_previously_owned_stale_artifacts(self):
        path = self.study / 'protocol.json'
        write_json(path, {'seed': 42})
        refresh(self.root)
        archived = self.archive / 'artifacts' / path.relative_to(self.root)
        self.assertTrue(archived.is_file())
        path.unlink()
        refresh(self.root)
        self.assertFalse(archived.exists())
        (self.archive / 'unexpected.txt').write_text('untracked')
        with self.assertRaises(ValueError):
            verify(self.archive)


if __name__ == '__main__':
    unittest.main()
