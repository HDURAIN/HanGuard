"""Synthetic fresh-run evidence checks; these tests never open GPUs or old runs."""
import csv
import tempfile
from pathlib import Path
import unittest

from scripts.hanguard.package_clean_ablation_results import (
    ARMS, DEFAULT_ARCHIVE, DEFAULT_STUDY, MAIN_ARMS, SEEDS, SYSTEMS,
    export_predictions, metric_rows, read_json, refresh, sha256, tables, verify, write_json,
)


def result(task, seed, accuracy=.8):
    arms = {}
    for index, arm in enumerate(ARMS):
        values = dict(accuracy=accuracy + index * .01, macro_f1=.7, n=8350 if task == 'binary' else 2180)
        arms[arm] = ({'full_original_test': {'validation_threshold': values}}
                     if task == 'binary' else {'native_known_harmful': values})
    return dict(task=task, seed=seed, arms=arms, predictions_sha256='1' * 64,
                rootselection_lock_sha256='2' * 64)


class CleanPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / DEFAULT_STUDY
        self.archive = self.root / DEFAULT_ARCHIVE

    def tearDown(self):
        self.temp.cleanup()

    def write_completed(self, task, seed):
        import pandas as pd
        folder = self.study / 'heads' / task / f's{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'test_predictions.parquet'
        pd.DataFrame(dict(base_id=['a'], baseline_p1=[.8])).to_parquet(path, index=False)
        value = result(task, seed)
        value.update(predictions_sha256=sha256(path), rootselection_lock_sha256=sha256(self.study / 'selection_lock.json'))
        result_path = folder / 'test_results.json'
        write_json(result_path, value)
        write_json(folder / 'test_complete.json', dict(complete=True, task=task, seed=seed,
                   protocol_sha256=sha256(self.study / 'protocol.json'),
                   rootselection_lock_sha256=sha256(self.study / 'selection_lock.json'),
                   artifacts={str(path): sha256(path), str(result_path): sha256(result_path)}))

    def test_pending_counts_and_no_historical_read(self):
        old = self.root / 'outputs/hanguard_four_arm_experiments_20261019'
        write_json(old / 'results.json', {'accuracy': .99999, 'text': 'OLD_PRIVATE'})
        write_json(old / 'protocol.json', {'sources': []})
        index = refresh(self.root)
        self.assertEqual(index['status'], 'pending')
        self.assertFalse(index['historical_experiment_inputs_used'])
        self.assertEqual(index['posthoc_test_exclusions'], [])
        self.assertEqual(index['formal_heads'], 54)
        self.assertEqual(index['linear_probes'], 192)
        rows, means, coverage = tables(self.study, self.root)
        self.assertEqual((len(rows), len(means), len(coverage)), (90, 30, 15))
        self.assertTrue(all(row['status'] == 'pending' and 'accuracy_percent' not in row for row in rows))
        self.assertTrue(all('mean_accuracy_percent' not in row for row in means))
        self.assertEqual({row['rows'] for row in rows}, {8350, 2180, 3704})
        manifest = read_json(self.archive / 'manifest.json')
        self.assertFalse(any('20261019' in (entry['source'] or '') for entry in manifest['files']))
        self.assertTrue(verify(self.archive)['valid'])

    def test_completed_schema_three_seed_mean_and_missing_seed(self):
        write_json(self.study / 'protocol.json', {})
        write_json(self.study / 'selection_lock.json', {'complete': True})
        systems = {}
        for seed in SEEDS:
            for task in ('binary', 'category'):
                self.write_completed(task, seed)
            entries = {arm: dict(metrics=dict(accuracy=.8 + i * .01, macro_f1=.7, rows=3704)) for i, arm in enumerate(MAIN_ARMS)}
            systems[str(seed)] = {'systems': {name: {'arms': entries} for name in SYSTEMS}}
        write_json(self.study / 'analysis/results.json', dict(complete=True, per_seed=systems))
        index = refresh(self.root)
        self.assertTrue(index['complete'])
        rows, means, coverage = tables(self.study, self.root)
        self.assertEqual(len(rows), 90)
        self.assertTrue(all(row['sample_std_accuracy_pp'] == 0 for row in means))
        self.assertTrue(all(row['status'] == 'complete' for row in coverage))
        (self.study / 'heads/binary/s44/test_results.json').unlink()
        rows, means, _ = tables(self.study, self.root)
        for row in (item for item in means if item['task'] == 'binary'):
            self.assertEqual(row['status'], 'pending')
            self.assertNotIn('mean_accuracy_percent', row)

    def test_only_nine_arm_test_receipts_and_full_denominators(self):
        value = result('binary', 42)
        self.assertEqual(len(metric_rows(value, 'binary', 42, 'new')), 9)
        for change in ({'complete': False}, {'seed': 43}, {'predictions_sha256': None}):
            with self.subTest(change=change):
                self.assertEqual(metric_rows(dict(value, **change), 'binary', 42, 'new'), [])
        value['arms']['baseline']['full_original_test']['validation_threshold']['n'] = 8228
        with self.assertRaises(ValueError):
            metric_rows(value, 'binary', 42, 'new')

    def test_incomplete_or_tampered_receipt_never_promotes_scores(self):
        write_json(self.study / 'protocol.json', {})
        write_json(self.study / 'selection_lock.json', {'complete': True})
        write_json(self.study / 'heads/binary/s42/test_results.json', result('binary', 42))
        rows, _, _ = tables(self.study, self.root)
        self.assertTrue(all(row['status'] == 'pending' for row in rows))
        self.write_completed('binary', 42)
        rows, _, _ = tables(self.study, self.root)
        self.assertEqual(sum(row['status'] == 'complete' for row in rows), 9)
        (self.study / 'selection_lock.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'selection lock'):
            tables(self.study, self.root)

    def test_frozen_sources_detect_modified_source_and_do_not_read_old_results(self):
        source = self.root / 'scripts/hanguard/clean_ablation_study.py'
        source.parent.mkdir(parents=True)
        source.write_text('NEW_SOURCE = True\n')
        write_json(self.study / 'protocol.json', {'sources': [{'path': str(source.relative_to(self.root)), 'sha256': sha256(source)}]})
        index = refresh(self.root)
        self.assertEqual(index['frozen_sources']['status'], 'passed')
        self.assertTrue(index['frozen_sources']['passed'])
        source.write_text('NEW_SOURCE = False\n')
        with self.assertRaisesRegex(ValueError, 'Registered scientific sources changed'):
            refresh(self.root)

    def test_absolute_source_registration_is_independently_verified_in_archive(self):
        source = self.root / 'scripts/hanguard/clean_ablation_study.py'
        source.parent.mkdir(parents=True)
        source.write_text('NEW_SOURCE = True\n')
        write_json(self.study / 'protocol.json', {'code_sha256': {str(source): sha256(source)}})
        index = refresh(self.root)
        self.assertTrue(index['frozen_sources']['passed'])
        archived = self.archive / 'artifacts' / source.relative_to(self.root)
        archived.write_text('NEW_SOURCE = False\n')
        manifest_path = self.archive / 'manifest.json'
        manifest = read_json(manifest_path)
        for entry in manifest['files']:
            if entry['path'] == str(archived.relative_to(self.archive)):
                entry['sha256'] = sha256(archived)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'Archived registered source differs'):
            verify(self.archive)

    def test_prediction_export_removes_text_in_csv_and_nested_json(self):
        write_json(self.study / 'protocol.json', {'prompt': 'PRIVATE', 'nested': {'original_text': 'PRIVATE', 'seed': 42}})
        folder = self.study / 'heads/binary/s42'
        folder.mkdir(parents=True)
        source = folder / 'test_predictions.csv'
        source.write_text('base_id,source,binary_label,probe_selected_single_p1,category_fusion_repeat_last_argmax,prompt,unseen_text\na,native,1,.7,1,PRIVATE,PRIVATE\n')
        refresh(self.root)
        target = self.archive / 'artifacts' / source.relative_to(self.root)
        self.assertNotIn('PRIVATE', target.read_text())
        self.assertIn('probe_selected_single_p1', target.read_text())
        copied = self.archive / 'artifacts' / (self.study / 'protocol.json').relative_to(self.root)
        self.assertEqual(read_json(copied), {'nested': {'seed': 42}})

    def test_parquet_predictions_strip_text_and_keep_all_new_arms(self):
        import pandas as pd
        source = self.root / 'predictions.parquet'
        target = self.root / 'predictions.csv'
        pd.DataFrame(dict(base_id=['a'], L32_p4=[.3], fusion_category_shared_p0=[.2],
                          fusion_category_shared_p1=[.8], translated_prompt=['PRIVATE'])).to_parquet(source, index=False)
        metadata = export_predictions(source, target)
        self.assertEqual(metadata['rows'], 1)
        self.assertIn('fusion_category_shared_p1', metadata['columns'])
        self.assertIn('L32_p4', metadata['columns'])
        self.assertNotIn('PRIVATE', target.read_text())

    def test_prediction_nonfinite_duplicate_and_no_identity_rejected(self):
        source, target = self.root / 'prediction.csv', self.root / 'safe.csv'
        for contents in ('base_id,baseline_p1\na,NaN\n', 'base_id,baseline_p1\na,.2\na,.3\n', 'baseline_p1\n.2\n'):
            source.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                export_predictions(source, target)

    def test_archive_tampering_unlisted_files_and_old_inputs_rejected(self):
        refresh(self.root)
        (self.archive / 'coverage.csv').write_text('TAMPERED')
        with self.assertRaises(ValueError):
            verify(self.archive)
        refresh(self.root)
        (self.archive / 'unexpected.pt').write_text('UNLISTED')
        with self.assertRaises(ValueError):
            verify(self.archive)
        with self.assertRaises(ValueError):
            refresh(self.root, self.root / 'outputs/hanguard_four_arm_experiments_20261019')

    def test_refresh_removes_only_unchanged_previously_owned_files(self):
        path = self.study / 'probe_selection.json'
        write_json(path, {'selected_layers': {'binary': 19}})
        refresh(self.root)
        archived = self.archive / 'artifacts' / path.relative_to(self.root)
        self.assertTrue(archived.is_file())
        path.unlink()
        refresh(self.root)
        self.assertFalse(archived.exists())


if __name__ == '__main__':
    unittest.main()
