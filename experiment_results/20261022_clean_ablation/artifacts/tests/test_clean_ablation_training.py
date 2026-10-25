"""CPU integration of freshly fitted probes, nine heads and test barrier."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from scripts.hanguard import clean_ablation_training as training


class SyntheticTokens:
    def __init__(self, values): self.values = values
    def get_row(self, row): return self.values[row]


def synthetic_split(split, count=12, hidden=6):
    rng = np.random.default_rng({'train': 1, 'validation': 2, 'test': 3}[split])
    lengths = np.asarray([1 + i % 4 for i in range(count)], dtype=np.int64)
    binary = np.asarray([int(i % 3 != 0) for i in range(count)])
    categories = np.where(binary, np.arange(count) % 5 + 1, 0)
    sources = np.asarray(['wildguard_zh' if i % 4 == 0 else 'chinese_curated' if i % 2 else 'jailbench' for i in range(count)])
    frame = pd.DataFrame(dict(base_id=[f'{split}_{i}' for i in range(count)],
                              group_id=[f'{split}_g{i // 2}' for i in range(count)], source=sources,
                              category_id=categories, prompt_harm_label=np.where(binary, 'harmful', 'unharmful'),
                              type_label_mask=binary.astype(bool)))
    meta = frame[['base_id', 'group_id', 'source']].copy()
    meta['label'], meta['prompt_tokens'] = binary, lengths
    return dict(last=rng.normal(size=(count, 32, hidden)).astype(np.float16),
                tokens=SyntheticTokens([rng.normal(size=(int(n), hidden)).astype(np.float16) for n in lengths]),
                meta=meta, frame=frame, lengths=lengths, labels=training.label_arrays(frame), hidden_size=hidden,
                identity=dict(dataset_sha256=f'synthetic_{split}', joint_manifest_sha256=f'joint_{split}', rows=count))


class CleanTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.study = Path(temporary.name)
        self.data = {split: synthetic_split(split) for split in ('train', 'validation', 'test')}
        self.protocol = dict(tasks=['binary', 'category'], seeds=[42], data_dir=str(self.study / 'data'),
                             data_sha256={s: f'synthetic_{s}' for s in self.data}, probe_epochs=2,
                             training=dict(width=4, epochs=2, max_epochs=2, dropout=0., lr=.001,
                                           weight_decay=.001, token_budget=20, max_micro=4,
                                           effective_batch=5, grad_clip=5., bootstrap_draws=12))
        training.dump(self.study / 'protocol.json', self.protocol)

    def loader(self, study, split): return self.data[split]

    def probes(self):
        for task in training.TASKS: training.fit_probes(self.study, task, 42)
        return training.lock_probe_layers(self.study)

    def lock(self):
        value = training.selection_artifacts(self.study)
        training.dump(self.study / 'selection_lock.json', dict(
            complete=True, protocol_sha256=training.sha(self.study / 'protocol.json'), **value))

    def test_complete_fresh_study_synthetic(self):
        with mock.patch.object(training, '_load_split', side_effect=self.loader) as loader:
            probe_lock = self.probes()
            self.assertEqual(set(probe_lock['selected_layers']), set(training.TASKS))
            for task in training.TASKS:
                selected = training.fit_task(self.study, task, 42)
                self.assertEqual(len(selected['arms']), 9)
                self.assertEqual(selected['selected_probe_layer'], probe_lock['selected_layers'][task])
            self.assertNotIn('test', [call.args[1] for call in loader.call_args_list])
            self.lock()
            training.verify_selection_lock(self.study)
            for task in training.TASKS:
                result = training.evaluate_task(self.study, task, 42)
                predictions = pd.read_parquet(result['predictions_path'])
                self.assertEqual(len(predictions), len(self.data['test']['meta']))
                self.assertEqual(len(result['arms']), 9)
                self.assertIn('interaction_accuracy', result['diagnostics'])
                self.assertFalse({'prompt', 'original_english'} & set(predictions))
                if task == 'binary':
                    self.assertEqual(set(result['arms']['baseline']), {'full_original_test'})
            result = training.summarize(self.study)
            self.assertTrue(result['complete'])
            self.assertFalse(result['outcome_selected_subset'])
            self.assertEqual(len(result['per_seed']['42']['systems']), 3)

    def test_microbatch_gradient_is_sample_weighted(self):
        cfg = training.settings(self.protocol)
        cfg['token_budget'], cfg['max_micro'] = 100, 100
        a = training.build_heads(6, 'binary', 42, cfg, 'cpu')
        b = training.build_heads(6, 'binary', 42, cfg, 'cpu')
        oa = {k: torch.optim.SGD(v.parameters(), lr=.01) for k, v in a.items()}
        ob = {k: torch.optim.SGD(v.parameters(), lr=.01) for k, v in b.items()}
        x, _ = training.train_macro(a, oa, self.data['train'], np.arange(5), 'binary', cfg, 'cpu', seed=42, epoch=0, step=0)
        cfg['max_micro'] = 2
        y, _ = training.train_macro(b, ob, self.data['train'], np.arange(5), 'binary', cfg, 'cpu', seed=42, epoch=0, step=0)
        for arm in training.ARMS:
            self.assertAlmostEqual(x[arm], y[arm], places=6)
            for key, value in a[arm].state_dict().items():
                torch.testing.assert_close(value, b[arm].state_dict()[key], atol=1e-6, rtol=1e-5)

    def test_all_nine_common_extension(self):
        history = [dict(epoch=i + 1, validation_ce={a: .5 - i * .01 for a in training.ARMS}) for i in range(12)]
        best = {a: 1 for a in training.ARMS}
        best['category_shared'] = 12
        choice = training.extension_rule(history, best, 12, 40)
        self.assertEqual(choice['new_budget'], 24)
        self.assertTrue(choice['all_nine_arms_extended'])
        self.assertEqual(set(choice['triggers']), {'category_shared'})

    def test_fits_require_fresh_probes_and_test_requires_global_lock(self):
        with mock.patch.object(training, '_load_split') as loader:
            with self.assertRaises(FileNotFoundError): training.fit_task(self.study, 'binary', 42)
            with self.assertRaises(FileNotFoundError): training.evaluate_task(self.study, 'binary', 42)
            loader.assert_not_called()

    def test_no_historical_artifact_protocol_keys(self):
        for key in ('excluded_test_ids_path', 'last_token_cache_dir'):
            protocol = dict(self.protocol, **{key: '/historical/forbidden'})
            training.dump(self.study / 'protocol.json', protocol)
            with self.assertRaisesRegex(ValueError, 'old caches'):
                training.read_protocol(self.study)

    def test_probe_lock_rejects_tampering(self):
        with mock.patch.object(training, '_load_split', side_effect=self.loader): self.probes()
        path = self.study / 'probe_selection.json'
        value = training.read_json(path)
        value['selected_layers']['binary'] = 32 if value['selected_layers']['binary'] != 32 else 1
        training.dump(path, value)
        with self.assertRaisesRegex(ValueError, 'criterion'): training.verify_probe_selection(self.study)

    def test_exact_epoch_resume_matches_uninterrupted(self):
        self.protocol['training']['dropout'] = .2
        training.dump(self.study / 'protocol.json', self.protocol)
        with mock.patch.object(training, '_load_split', side_effect=self.loader): self.probes()
        original = training.train_macro
        interrupted = False
        def fail(*args, **kwargs):
            nonlocal interrupted
            if kwargs['epoch'] == 1 and not interrupted:
                interrupted = True
                raise RuntimeError('synthetic interruption')
            return original(*args, **kwargs)
        with mock.patch.object(training, '_load_split', side_effect=self.loader), mock.patch.object(training, 'train_macro', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                training.fit_task(self.study, 'binary', 42)
        with mock.patch.object(training, '_load_split', side_effect=self.loader):
            resumed = training.fit_task(self.study, 'binary', 42)
            clean = training.fit_task(self.study, 'category', 42)
        checkpoint = torch.load(resumed['checkpoint_path'], weights_only=True)
        # Refit same binary job from scratch while preserving its fresh probe lock.
        import shutil
        shutil.rmtree(training.task_folder(self.study, 'binary', 42))
        with mock.patch.object(training, '_load_split', side_effect=self.loader):
            fresh = training.fit_task(self.study, 'binary', 42)
        other = torch.load(fresh['checkpoint_path'], weights_only=True)
        for arm in training.ARMS:
            for key in checkpoint['states'][arm]:
                torch.testing.assert_close(checkpoint['states'][arm][key], other['states'][arm][key], atol=0., rtol=0.)

    def test_probe_exact_epoch_resume_matches_uninterrupted(self):
        original = training._probe_predict
        interrupted = False
        calls = 0
        def fail(*args, **kwargs):
            nonlocal interrupted, calls
            calls += 1
            if calls == 2 and not interrupted:
                interrupted = True
                raise RuntimeError('probe interruption')
            return original(*args, **kwargs)
        with mock.patch.object(training, '_load_split', side_effect=self.loader), mock.patch.object(training, '_probe_predict', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'probe interruption'):
                training.fit_probes(self.study, 'binary', 42)
        with mock.patch.object(training, '_load_split', side_effect=self.loader):
            resumed = training.fit_probes(self.study, 'binary', 42)
        checkpoint = torch.load(training.probe_folder(self.study, 'binary', 42) / 'selected_probes.pt', weights_only=True)
        import shutil
        shutil.rmtree(training.probe_folder(self.study, 'binary', 42))
        with mock.patch.object(training, '_load_split', side_effect=self.loader):
            fresh = training.fit_probes(self.study, 'binary', 42)
        other = torch.load(training.probe_folder(self.study, 'binary', 42) / 'selected_probes.pt', weights_only=True)
        self.assertEqual(resumed['layers'], fresh['layers'])
        for key in checkpoint['states']:
            torch.testing.assert_close(checkpoint['states'][key], other['states'][key], atol=0., rtol=0.)

    def test_native_type_mask_and_unchanged_binary_rows(self):
        labels = self.data['train']['labels']
        self.assertEqual(len(labels['binary']), 12)
        self.assertFalse(labels['typed'][self.data['train']['frame'].source.eq('wildguard_zh')].any())
        self.assertTrue((labels['categories'][labels['typed']] > 0).all())

    def test_vectorized_probes_are_independent(self):
        probe = training.LayerProbes(6, 2, 42)
        logits = probe(torch.tensor(self.data['train']['last'][:3], dtype=torch.float32))
        torch.nn.functional.cross_entropy(logits[:, 4], torch.tensor([0, 1, 1])).backward()
        self.assertTrue((probe.weight.grad[4] != 0).any())
        self.assertEqual(torch.count_nonzero(probe.weight.grad[:4]), 0)
        self.assertEqual(torch.count_nonzero(probe.weight.grad[5:]), 0)


if __name__ == '__main__': unittest.main()
