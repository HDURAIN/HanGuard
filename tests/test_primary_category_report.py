"""CPU audits for primary-label metrics, paired tests and immutable inputs."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np
import pandas as pd

from scripts.hanguard import primary_category_report as report


class MetricTests(unittest.TestCase):
    def test_seed_statistics_use_sample_std_not_pooled_example_uncertainty(self):
        value = report.seed_statistics([.6, .7, .8])
        self.assertAlmostEqual(value['mean'], .7)
        self.assertAlmostEqual(value['std'], .1)
        self.assertEqual(value['std_ddof'], 1)
        self.assertIsNone(report.seed_statistics([.7])['std'])
    def test_fixed_class_macro_and_confusion_orientation(self):
        value = report.classification_metrics([1, 1, 2], [1, 2, 2], report.TYPE_CLASSES)
        self.assertEqual(value['accuracy'], 2/3)
        self.assertAlmostEqual(value['macro_f1'], (2/3 + 2/3) / 5)
        self.assertEqual(value['confusion'][0][:2], [1, 1])
        self.assertEqual(value['confusion'][1][:2], [0, 1])
        self.assertEqual(value['per_class'][4]['support_flags'], ['no_true_examples'])

    def test_identical_predictions_zero_interval_and_mcnemar_one(self):
        y = np.tile(np.arange(1, 6), 3)
        result = report.paired_bootstrap(y, y, y, report.TYPE_CLASSES)
        for metric in ('accuracy', 'macro_f1'):
            self.assertEqual(result[metric]['ci95'], [0., 0.])
            self.assertEqual(result[metric]['delta'], 0.)
        self.assertEqual(result['mcnemar_exact']['two_sided_p'], 1.)

    def test_improvement_interval_and_exact_discordance(self):
        truth = np.tile(np.arange(1, 6), 4)
        wrong = truth % 5 + 1
        result = report.paired_bootstrap(truth, truth, wrong, report.TYPE_CLASSES)
        self.assertEqual(result['accuracy']['delta'], 1.)
        self.assertEqual(result['accuracy']['ci95'], [1., 1.])
        self.assertGreater(result['macro_f1']['ci95'][0], 0.)
        self.assertEqual(result['mcnemar_exact']['candidate_only_correct'], 20)
        self.assertEqual(result['mcnemar_exact']['baseline_only_correct'], 0)
        self.assertEqual(result['mcnemar_exact']['two_sided_p'], 2 / 2**20)

    def test_vectorized_bootstrap_statistics_match_direct_metrics(self):
        rng = np.random.default_rng(22)
        truth = rng.integers(0, 6, size=(7, 25))
        prediction = rng.integers(0, 6, size=(7, 25))
        accuracy, macro = report._batch_scores(truth, prediction, report.ALL_CLASSES)
        for index in range(7):
            direct = report.classification_metrics(truth[index], prediction[index], report.ALL_CLASSES)
            self.assertAlmostEqual(accuracy[index], direct['accuracy'])
            self.assertAlmostEqual(macro[index], direct['macro_f1'])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'inputs'
        self.data.mkdir()
        frozen = {'input_sha256': {}}
        for split in ('train', 'validation', 'test'):
            truth = np.tile(np.arange(6), 2)
            frame = pd.DataFrame(dict(base_id=[f'{split}_{i:02}' for i in range(12)],
                source=['alpha', 'beta'] * 6, prompt=[f'{split}中文正文{i}' for i in range(12)],
                category_id=truth.astype(str), prompt_harm_label=np.where(truth > 0, 'harmful', 'unharmful')))
            frame['prompt_sha256'] = frame.prompt.map(report.text_sha)
            frame.to_parquet(self.data / f'{split}.parquet', index=False)
            frozen['input_sha256'][f'{split}.parquet'] = report.sha(self.data / f'{split}.parquet')
        (self.root / 'sampling_manifest.json').write_text(json.dumps(frozen))
        self.study = self.root / 'study'
        self.study.mkdir()
        self.protocol = dict(data_dir=str(self.data), all_arms=list(report.ARMS), seeds=[42],
                             data_sha256={split: frozen['input_sha256'][f'{split}.parquet'] for split in ('train', 'validation', 'test')})
        report.dump(self.study / 'protocol.json', self.protocol)
        barrier = dict(locked=True, protocol_sha256=report.sha(self.study / 'protocol.json'), selection_sha256={})
        original = pd.read_parquet(self.data / 'test.parquet')
        truth = original.category_id.to_numpy(dtype=int)
        binary = np.where(truth > 0, .8, .2)
        binary[0] = .8  # Shared binary false positive, independent of type head.
        binary[1] = .2  # Shared binary false negative.
        for arm in report.ARMS:
            directory = self.study / 'runs' / f'{arm}_s42'
            directory.mkdir(parents=True)
            head_path = directory / 'head.pt'
            head_path.write_bytes(f'test-only checkpoint {arm}'.encode())
            predicted = np.where(truth > 0, truth, 1)
            if arm != 'description_queries':
                predicted[2] = 5
            if arm == 'last_mlp':
                predicted[3] = 2
            gated = np.where(binary >= .5, predicted, 0)
            type_metrics = report.classification_metrics(truth[truth > 0], predicted[truth > 0], report.TYPE_CLASSES)
            gated_metrics = report.classification_metrics(truth, gated, report.ALL_CLASSES)
            selection = dict(protocol_sha256=report.sha(self.study / 'protocol.json'),
                checkpoint_sha256=report.sha(head_path), best_epoch=3,
                head_parameters=328453 if arm == 'last_mlp' else 706177,
                parent_binary_threshold=.5,
                train_dataset_sha256=self.protocol['data_sha256']['train'],
                validation_dataset_sha256=self.protocol['data_sha256']['validation'])
            report.dump(directory / 'selection.json', selection)
            result = dict(arm=arm, seed=42, test_dataset_sha256=self.protocol['data_sha256']['test'],
                          checkpoint_sha256=report.sha(head_path), parent_binary_threshold=.5,
                          type_metrics=type_metrics, gated_metrics=gated_metrics, test_ce=.5)
            report.dump(directory / 'test_results.json', result)
            csv = original.drop(columns=['prompt', 'prompt_harm_label']).copy()
            csv['binary_probability'] = binary
            csv['predicted_category'] = predicted
            csv['gated_category'] = gated
            for category in report.TYPE_CLASSES:
                csv[f'p_{category}'] = np.where(predicted == category, .8, .05)
            csv.to_csv(directory / 'test.csv', index=False)
            path = directory / 'selection.json'
            barrier['selection_sha256'][str(path.relative_to(self.study))] = report.sha(path)
        report.dump(self.study / 'test_barrier.json', barrier)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_report_audits_original_labels_and_keeps_gate_separate(self):
        result = report.report(self.study)
        self.assertEqual(len(result['entries']), 3)
        self.assertEqual(len(result['paired_comparisons']), 2)
        self.assertTrue(result['random_description_parameters_equal'])
        self.assertFalse(result['new_multilabel_annotations_used'])
        self.assertEqual(result['training_safe_rows_excluded'], 2)
        self.assertAlmostEqual(result['common_binary_metrics']['accuracy'], 10/12)
        described = next(entry for entry in result['entries'] if entry['arm'] == 'description_queries')
        self.assertEqual(described['type_metrics']['accuracy'], 1.)
        self.assertAlmostEqual(described['gated_metrics']['accuracy'], 10/12)
        self.assertEqual(set(described['by_source']), {'alpha', 'beta'})
        text = (self.study / 'report.md').read_text()
        self.assertIn('1,734', text)
        self.assertIn('706,177', text)
        self.assertIn('McNemar', text)
        self.assertIn('二分类准确率', text)
        self.assertIn('不称全新盲测', text)

    def test_report_rejects_modified_checkpoint(self):
        (self.study / 'runs/last_mlp_s42/head.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            report.report(self.study)

    def test_csv_primary_label_or_gate_changes_rejected(self):
        path = self.study / 'runs/last_mlp_s42/test.csv'
        original = pd.read_csv(path)
        for column, value, pattern in [('category_id', 5, 'single-primary labels'),
                                       ('gated_category', 0, 'frozen parent binary gate')]:
            with self.subTest(column=column):
                changed = original.copy()
                changed.loc[0, column] = value
                changed.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, pattern):
                    report.report(self.study)

    def test_softmax_normalization_required(self):
        path = self.study / 'runs/last_mlp_s42/test.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'p_1'] = .9
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'five-way softmax'):
            report.report(self.study)

    def test_barrier_must_be_explicitly_locked(self):
        path = self.study / 'test_barrier.json'
        barrier = json.loads(path.read_text())
        barrier['locked'] = 'true'
        report.dump(path, barrier)
        with self.assertRaisesRegex(ValueError, 'locked test barrier'):
            report.report(self.study)

    def test_nine_run_two_source_report_summarizes_matched_seeds(self):
        self.protocol.update(seeds=[42, 43, 44], experiment_scope='two_source_full_primary_category',
                             epochs=40, early_stopping_patience=8, early_stopping_min_delta=1e-4,
                             head_width_by_arm={'last_mlp': 275, 'learned_queries': 128, 'description_queries': 128})
        for split in ('train', 'validation', 'test'):
            path = self.data / f'{split}.parquet'
            frame = pd.read_parquet(path)
            frame['source'] = frame.source.map({'alpha': 'chinese_curated', 'beta': 'jailbench'})
            frame.to_parquet(path, index=False)
            self.protocol['data_sha256'][split] = report.sha(path)
        report.dump(self.root / 'sampling_manifest.json',
                    dict(input_sha256={f'{split}.parquet': value for split, value in self.protocol['data_sha256'].items()}))
        report.dump(self.study / 'protocol.json', self.protocol)
        protocol_hash = report.sha(self.study / 'protocol.json')
        barrier = dict(locked=True, protocol_sha256=protocol_hash, selection_sha256={})
        for arm in report.ARMS:
            original = self.study / 'runs' / f'{arm}_s42'
            for seed in [43, 44]:
                shutil.copytree(original, self.study / 'runs' / f'{arm}_s{seed}')
            for seed in [42, 43, 44]:
                directory = self.study / 'runs' / f'{arm}_s{seed}'
                selection = json.loads((directory / 'selection.json').read_text())
                selection.update(protocol_sha256=protocol_hash, head_parameters=705655 if arm == 'last_mlp' else 706177,
                    train_dataset_sha256=self.protocol['data_sha256']['train'],
                    validation_dataset_sha256=self.protocol['data_sha256']['validation'])
                report.dump(directory / 'selection.json', selection)
                result = json.loads((directory / 'test_results.json').read_text())
                result.update(seed=seed, test_dataset_sha256=self.protocol['data_sha256']['test'])
                report.dump(directory / 'test_results.json', result)
                csv = pd.read_csv(directory / 'test.csv')
                csv['source'] = csv.source.map({'alpha': 'chinese_curated', 'beta': 'jailbench'})
                csv.to_csv(directory / 'test.csv', index=False)
                barrier['selection_sha256'][str((directory / 'selection.json').relative_to(self.study))] = report.sha(directory / 'selection.json')
        report.dump(self.study / 'test_barrier.json', barrier)
        result = report.report(self.study, repetitions=10)
        self.assertEqual(len(result['entries']), 9)
        self.assertEqual(len(result['paired_comparisons']), 9)
        self.assertEqual(len(result['paired_seed_summary']), 3)
        self.assertEqual(len(result['seed_aggregates']), 3)
        for row in result['seed_aggregates']:
            self.assertEqual(row['seeds'], [42, 43, 44])
            self.assertAlmostEqual(row['type_metrics']['accuracy']['std'], 0.)
            self.assertEqual(set(row['by_source']), {'chinese_curated', 'jailbench'})
        self.assertIn('均值 ± 样本标准差', (self.study / 'report.md').read_text())
        self.assertIn('705,655', (self.study / 'report.md').read_text())


if __name__ == '__main__':
    unittest.main()
