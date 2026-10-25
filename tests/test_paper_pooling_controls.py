"""CPU checks for complete, resumable generic full-token controls."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from scripts.hanguard import paper_pooling_controls as pooling


class Tokens:
    def __init__(self, values): self.values = values
    def get_row(self, row): return self.values[row]


def synthetic(split, count=20, hidden=6):
    rng = np.random.default_rng(dict(train=10, validation=11, test=12)[split])
    lengths = np.asarray([1 + i % 4 for i in range(count)])
    harm = np.asarray([i % 3 != 0 for i in range(count)])
    categories = np.where(harm, np.arange(count) % 5 + 1, 0)
    sources = ['wildguard_zh' if i % 4 == 0 else 'chinese_curated' if i % 2 else 'jailbench' for i in range(count)]
    frame = pd.DataFrame(dict(base_id=[f'{split}_{i:03d}' for i in range(count)],
        group_id=[f'{split}_group_{i // 2}' for i in range(count)], source=sources,
        category_id=categories, prompt_harm_label=np.where(harm, 'harmful', 'unharmful'),
        type_label_mask=harm))
    meta = frame[['base_id', 'group_id', 'source']].copy()
    meta['label'], meta['prompt_tokens'] = harm.astype(int), lengths
    return dict(meta=meta, frame=frame, lengths=lengths, hidden_size=hidden,
        tokens=Tokens([rng.normal(size=(length, hidden)).astype(np.float16) for length in lengths]),
        labels=pooling.original.label_arrays(frame),
        identity=dict(dataset_sha256=f'data_{split}', manifest_sha256=f'joint_{split}', rows=count))


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_padding_never_changes_valid_pooling_even_with_nan_padding(self):
        for arm in pooling.ARMS:
            with self.subTest(arm=arm):
                head = pooling.PoolingHead(6, width=4, num_classes=5, arm=arm, dropout=0.)
                head.eval()
                x = torch.randn(2, 3, 6)
                mask = torch.ones(2, 3, dtype=torch.bool)
                padded = torch.cat((x, torch.full((2, 7, 6), float('nan'))), dim=1)
                padded_mask = torch.cat((mask, torch.zeros(2, 7, dtype=torch.bool)), dim=1)
                torch.testing.assert_close(head(x, mask), head(padded, padded_mask), atol=1e-6, rtol=1e-6)
                self.assertTrue((head.attention_weights(padded, padded_mask)[:, 3:] == 0).all())
                with self.assertRaises(ValueError): head(x, torch.zeros_like(mask))

    def test_one_shared_attention_distribution_has_no_class_dimension(self):
        head = pooling.PoolingHead(6, 4, 5, 'shared_attention_mlp', dropout=0.)
        x = torch.randn(2, 3, 6); mask = torch.ones(2, 3, dtype=torch.bool)
        attention = head.attention_weights(x, mask)
        self.assertEqual(attention.shape, (2, 3))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2))
        self.assertEqual(head.global_query.shape, (4,))
        self.assertFalse(any('category' in name for name, _ in head.named_parameters()))
        other = pooling.PoolingHead(6, 4, 2, 'shared_attention_mlp', dropout=0.)
        self.assertEqual(head.metadata()['parameters'] - other.metadata()['parameters'], 3 * (4 + 1))

    def test_projection_and_classifier_initialized_identically_between_controls(self):
        mean = pooling.PoolingHead(6, 4, 5, pooling.ARMS[0], seed=43)
        attention = pooling.PoolingHead(6, 4, 5, pooling.ARMS[1], seed=43)
        for name, tensor in mean.state_dict().items():
            torch.testing.assert_close(tensor, attention.state_dict()[name], atol=0., rtol=0.)

    def test_parameters_are_functional_no_dummy_count_padding(self):
        for arm in pooling.ARMS:
            head = pooling.PoolingHead(6, 8, 5, arm, dropout=0., seed=42)
            x = torch.randn(6, 7, 6); mask = torch.ones(6, 7, dtype=torch.bool)
            torch.nn.functional.cross_entropy(head(x, mask), torch.tensor([0, 1, 2, 3, 4, 1])).backward()
            for name, parameter in head.named_parameters():
                with self.subTest(arm=arm, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            self.assertEqual(head.metadata()['parameters'], sum(p.numel() for p in head.parameters()))


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'pooling'; self.output.mkdir()
        self.parent = self.root / 'parent'; self.parent.mkdir()
        pooling.write(self.parent / 'results.json', {'aggregate': []})
        cfg = dict(pooling.DEFAULT_SETTINGS, width=4, epochs=2, max_epochs=2, dropout=.1,
                   max_micro=4, token_budget=20, effective_batch=5, bootstrap_draws=12)
        self.protocol = dict(schema=pooling.SCHEMA, output=str(self.output), parent=str(self.parent),
            tasks=list(pooling.TASKS), arms=list(pooling.ARMS), seeds=[42], settings=cfg,
            sources={str(Path(pooling.__file__).resolve()): pooling.sha(pooling.__file__)},
            inputs={}, cpu_threads=1)
        self.register()
        self.data = {split: synthetic(split) for split in pooling.SPLITS}

    def register(self):
        pooling.write(self.output / 'protocol.json', self.protocol)
        pooling.write(self.output / 'protocol_registration.json',
                      {'protocol_sha256': pooling.sha(self.output / 'protocol.json')})

    def loader(self, output, split): return self.data[split]

    def test_complete_fit_selection_barrier_test_prediction_and_summary(self):
        with mock.patch.object(pooling, '_load_split', side_effect=self.loader) as loader:
            a = pooling.fit(self.output, 'binary', 42)
            self.assertFalse(a['formal_test_opened'])
            self.assertEqual(a['training_rows'], 20)
            self.assertFalse((self.output / 'test_selection_lock.json').exists())
            self.assertNotIn('test', [call.args[1] for call in loader.call_args_list])
            state = pooling.progress(self.output)
            self.assertEqual(state['selected_jobs'], 1)
            self.assertEqual(state['pending_jobs'], [{'task': 'category', 'seed': 42, 'phase': 'fit'}])
            b = pooling.fit(self.output, 'category', 42)
            self.assertTrue(b['complete'])
            self.assertTrue((self.output / 'test_selection_lock.json').exists())
            state = pooling.progress(self.output)
            self.assertEqual(state['pending_jobs'], [{'task': 'binary', 'seed': 42, 'phase': 'evaluate'}])
            a = pooling.fit(self.output, 'binary', 42)
            self.assertEqual(a['rows'], 20)
            result = pooling.summarize(self.output)
            self.assertTrue(result['complete'])
            self.assertTrue(pooling.is_complete(self.output))
            self.assertEqual(len(result['aggregate']), 4)
            state = pooling.progress(self.output)
            self.assertEqual(state['completed_jobs'], state['total_jobs'])
            for task in pooling.TASKS:
                f = pooling.job_folder(self.output, task, 42) / 'test_predictions.parquet'
                frame = pd.read_parquet(f)
                self.assertEqual(len(frame), 20)
                self.assertFalse({'prompt', 'original_english'} & set(frame))
                self.assertTrue(frame.base_id.is_unique)

    def test_real_registration_and_mmap_loader_on_tiny_parent_features(self):
        parent_protocol = dict(data_dir=str(self.parent / 'data'), data_sha256={}, code_sha256={})
        for split, data in self.data.items():
            (self.parent / 'data').mkdir(exist_ok=True)
            data['frame'].to_parquet(self.parent / 'data' / f'{split}.parquet', index=False)
            parent_protocol['data_sha256'][split] = pooling.sha(self.parent / 'data' / f'{split}.parquet')
        pooling.write(self.parent / 'protocol.json', parent_protocol)
        parent_sha = pooling.sha(self.parent / 'protocol.json')
        pooling.write(self.parent / 'protocol_registration.json', {'protocol_sha256': parent_sha})
        pooling.write(self.parent / 'selection_lock.json', {'complete': True})
        pooling.write(self.parent / 'results.json', dict(complete=True, protocol_sha256=parent_sha,
            rootselection_lock_sha256=pooling.sha(self.parent / 'selection_lock.json'), aggregate=[]))
        for split, data in self.data.items():
            folder = self.parent / 'features' / split; folder.mkdir(parents=True)
            data['meta'].to_parquet(folder / 'metadata.parquet', index=False)
            np.save(folder / 'lengths.npy', data['lengths'])
            chunk = folder / 'chunks/000000/attempt_001'; chunk.mkdir(parents=True)
            tokens = np.concatenate(data['tokens'].values)
            offsets = np.concatenate(([0], np.cumsum(data['lengths'])))
            last = np.zeros((len(data['meta']), 32, data['hidden_size']), np.float16)
            last[:, -1] = tokens[offsets[1:] - 1]
            arrays = dict(tokens=tokens, offsets=offsets, rows=np.arange(len(data['meta'])),
                          lengths=data['lengths'], last=last)
            artifacts = {}
            for name, value in arrays.items():
                path = chunk / f'{name}.npy'; np.save(path, value)
                artifacts[path.name] = dict(path=str(path), sha256=pooling.sha(path))
            pooling.write(chunk / 'row_identity.json', {'base_ids': data['meta'].base_id.tolist()})
            artifacts['row_identity.json'] = dict(path=str(chunk / 'row_identity.json'),
                                                 sha256=pooling.sha(chunk / 'row_identity.json'))
            pooling.write(folder / 'plan.json', dict(rows=len(data['meta']), num_layers=32,
                hidden_size=data['hidden_size'], artifacts={name: pooling.sha(folder / name)
                    for name in ('metadata.parquet', 'lengths.npy')}))
            pooling.write(chunk.parent / 'receipt.json', {'artifacts': artifacts})
            pooling.write(folder / 'manifest.json', dict(complete=True, joint_same_forward=True,
                protocol_sha256=parent_sha, plan_sha256=pooling.sha(folder / 'plan.json'),
                chunks=[dict(artifacts=artifacts, receipt_path=str(chunk.parent / 'receipt.json'),
                             receipt_sha256=pooling.sha(chunk.parent / 'receipt.json'))]))
        registered = self.root / 'registered_pooling'
        with mock.patch.object(pooling.original, 'read_protocol', return_value=parent_protocol), \
             mock.patch.object(pooling.original, 'verify_selection_lock', return_value={'complete': True}):
            status = pooling.initialize(registered, self.parent)
        self.assertEqual(status['total_jobs'], 6)
        loaded = pooling._load_split(registered, 'train')
        self.assertEqual(loaded['meta'].base_id.tolist(), self.data['train']['meta'].base_id.tolist())
        for row in range(len(loaded['meta'])):
            np.testing.assert_array_equal(loaded['tokens'].get_row(row), self.data['train']['tokens'].get_row(row))

    def test_exact_epoch_resume_matches_clean_training(self):
        self.protocol['settings']['dropout'] = .2; self.register()
        original = pooling.train_macro
        failed = False
        def interrupt(*args, **kwargs):
            nonlocal failed
            if kwargs['epoch'] == 1 and not failed:
                failed = True
                raise RuntimeError('synthetic interrupted epoch')
            return original(*args, **kwargs)
        with mock.patch.object(pooling, '_load_split', side_effect=self.loader), \
             mock.patch.object(pooling, 'train_macro', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interrupted'):
                pooling.fit(self.output, 'binary', 42)
        with mock.patch.object(pooling, '_load_split', side_effect=self.loader):
            pooling.fit(self.output, 'binary', 42)
        folder = pooling.job_folder(self.output, 'binary', 42)
        resumed = torch.load(folder / 'selected_heads.pt', weights_only=True)
        shutil.rmtree(folder)
        with mock.patch.object(pooling, '_load_split', side_effect=self.loader):
            pooling.fit(self.output, 'binary', 42)
        clean = torch.load(folder / 'selected_heads.pt', weights_only=True)
        for arm in pooling.ARMS:
            for name in clean['states'][arm]:
                torch.testing.assert_close(resumed['states'][arm][name], clean['states'][arm][name], atol=0., rtol=0.)

    def test_macro_loss_is_sample_weighted_across_microbatches(self):
        data, cfg = self.data['train'], dict(self.protocol['settings'], dropout=0., token_budget=100, max_micro=100)
        a = pooling.build_heads(6, 'binary', 42, cfg, 'cpu')
        b = pooling.build_heads(6, 'binary', 42, cfg, 'cpu')
        oa = {k: torch.optim.SGD(v.parameters(), lr=.01) for k, v in a.items()}
        ob = {k: torch.optim.SGD(v.parameters(), lr=.01) for k, v in b.items()}
        pooling.train_macro(a, oa, data, np.arange(5), 'binary', cfg, 'cpu',
                           seed=42, epoch=0, step=0, oom_events=[])
        cfg['max_micro'] = 2
        pooling.train_macro(b, ob, data, np.arange(5), 'binary', cfg, 'cpu',
                           seed=42, epoch=0, step=0, oom_events=[])
        for arm in pooling.ARMS:
            for name in a[arm].state_dict():
                torch.testing.assert_close(a[arm].state_dict()[name], b[arm].state_dict()[name], atol=1e-6, rtol=1e-5)

    def test_global_test_lock_prevents_early_test_loading_and_detects_tampering(self):
        with self.assertRaisesRegex(ValueError, 'must be locked'):
            pooling.verify_test_lock(self.output)
        with mock.patch.object(pooling, '_load_split', side_effect=self.loader):
            for task in pooling.TASKS: pooling.fit(self.output, task, 42)
        folder = pooling.job_folder(self.output, 'binary', 42)
        frame = pd.read_parquet(folder / 'validation_predictions.parquet')
        frame.loc[0, 'binary_label'] ^= 1
        frame.to_parquet(folder / 'validation_predictions.parquet', index=False)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            pooling.verify_test_lock(self.output)

    def test_strong_stat_binding_rejects_parent_payload_changed_with_old_mtime(self):
        payload = self.root / 'tokens.npy'
        payload.write_bytes(b'abcdefgh')
        self.protocol['inputs'] = {str(payload): dict(sha256=pooling.sha(payload),
            stat=pooling.stat_identity(payload), large_payload=True)}
        self.register(); pooling.verify(self.output)
        info = payload.stat(); payload.write_bytes(b'ABCDEFGH')
        os.utime(payload, ns=(info.st_atime_ns, info.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, '(strong identity|digest) changed'):
            pooling.verify(self.output)

    def test_gpu_requires_exact_root_admission_before_cuda_allocation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'root runner admission'):
                pooling._admit(self.output, 'cuda')
        p = Path('/proc') / str(os.getpid())
        stat = (p / 'stat').read_text(); fields = stat[stat.rfind(')') + 2:].split()
        record = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=int(fields[19]),
            command_sha256=hashlib.sha256((p / 'cmdline').read_bytes()).hexdigest(),
            gpu_uuid='GPU-fixture', physical_gpu=4, allocator_cap_gib=4,
            followup_protocol_sha256=pooling.sha(self.output / 'protocol.json'))
        path = self.root / 'admission.json'; pooling.write(path, record)
        env = dict(CUDA_VISIBLE_DEVICES='GPU-fixture',
            HANGUARD_FOLLOWUP_ADMITTED=record['followup_protocol_sha256'],
            HANGUARD_FOLLOWUP_ADMISSION_RECORD=str(path))
        with mock.patch.dict(os.environ, env): pooling._admit(self.output, 'cuda')
        record['start_time_ticks'] += 1; pooling.write(path, record)
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(RuntimeError, 'exact worker'):
                pooling._admit(self.output, 'cuda')

    def test_extension_is_common_and_uses_validation_only(self):
        history = [dict(epoch=i + 1, validation_ce={arm: .5 - .01 * i for arm in pooling.ARMS}) for i in range(12)]
        best = {pooling.ARMS[0]: 1, pooling.ARMS[1]: 12}
        choice = pooling.extension_rule(history, best, 12, 40)
        self.assertEqual(choice['new_budget'], 24)
        self.assertEqual(set(choice['triggers']), {pooling.ARMS[1]})
        self.assertTrue(choice['all_followup_arms_extended'])


if __name__ == '__main__': unittest.main()
