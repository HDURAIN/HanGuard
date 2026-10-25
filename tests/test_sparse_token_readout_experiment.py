"""CPU checks for logical-batch fairness and explicit cache identity binding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.hanguard import sparse_token_readout_experiment as experiment
from scripts.hanguard.sparse_token_readout_heads import ARMS, parameter_count


class LogicalBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(233)
        self.settings = dict(width=128, seed=42, dropout=0., projection_chunk_size=13)
        self.heads, self.audit = experiment.make_heads(self.settings, 'cpu')
        self.lengths = np.array([2, 5, 1, 8, 3, 6, 4])
        self.raw = torch.randn(7, 8, 2560)
        self.mask = torch.arange(8)[None] < torch.as_tensor(self.lengths)[:, None]
        self.labels = torch.tensor([0., 1., 0., 1., 1., 0., 1.])

    def test_logical_batch_gradients_match_uneven_microbatches_with_active_local_routes(self):
        parts = list(experiment.split_microbatches(np.arange(7), self.lengths, budget=15, max_rows=3))
        self.assertEqual([len(rows) for rows in parts], [3, 1, 2, 1])
        self.assertEqual(np.concatenate(parts).tolist(), list(range(7)))
        for arm, template in self.heads.items():
            with self.subTest(arm=arm):
                if hasattr(template, 'pool_output'):
                    # The zero initial residual would hide the query/token
                    # gradients. Activate it to make this check meaningful.
                    with torch.no_grad():
                        template.pool_output.weight.copy_(torch.eye(128) * .1)
                dense, micro = copy.deepcopy(template), copy.deepcopy(template)
                logits = experiment.forward(dense, arm, self.raw, self.mask)['logits']
                F.binary_cross_entropy_with_logits(logits, self.labels).backward()
                for rows in parts:
                    indices = torch.as_tensor(rows)
                    maximum = int(self.lengths[rows].max())
                    logits = experiment.forward(micro, arm, self.raw[indices, :maximum], self.mask[indices, :maximum])['logits']
                    (F.binary_cross_entropy_with_logits(logits, self.labels[indices], reduction='sum') / len(self.labels)).backward()
                for (name, left), (_, right) in zip(dense.named_parameters(), micro.named_parameters()):
                    self.assertIsNotNone(left.grad, name)
                    self.assertIsNotNone(right.grad, name)
                    torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=2e-5,
                                               msg=lambda message: f'{arm}.{name}: {message}')
                if arm == 'global_query_entmax15':
                    self.assertGreater(float(dense.query_global.weight.grad.abs().sum()), 1e-5)
                    self.assertGreater(float(dense.token_project[0].weight.grad.abs().sum()), 1e-3)

    def test_global_initialization_nominal_capacity_and_initial_logits_match(self):
        self.assertEqual(tuple(self.heads), experiment.ALL_ARMS)
        single = self.heads['last_mlp']
        self.assertEqual(parameter_count(single), 327937)
        self.assertEqual(self.audit['parameters'], 705665)
        self.assertEqual(self.audit['single_parameters'], 327937)
        expected = experiment.forward(single, 'last_mlp', self.raw, self.mask)['logits']
        for arm in ARMS:
            head = self.heads[arm]
            self.assertEqual(parameter_count(head), 705665)
            for module in ('project', 'classifier'):
                old, new = getattr(single, module).state_dict(), getattr(head, module).state_dict()
                for name in old:
                    torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
            torch.testing.assert_close(experiment.forward(head, arm, self.raw, self.mask)['logits'], expected)

    def test_microbatch_full_row_coverage_and_no_truncation(self):
        order = np.array([6, 1, 4, 0, 5, 2, 3])
        parts = list(experiment.split_microbatches(order, self.lengths, budget=16, max_rows=2))
        self.assertEqual(np.concatenate(parts).tolist(), order.tolist())
        for rows in parts:
            self.assertLessEqual(len(rows), 2)
            self.assertLessEqual(len(rows) * int(self.lengths[rows].max()), 16)
        for lengths in (np.array([0]), np.array([17])):
            with self.assertRaises(ValueError):
                list(experiment.split_microbatches([0], lengths, budget=16, max_rows=2))


class TokenCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hanguard_sparse_cache_contract_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.cache = self.root / 'token_cache'
        self.cache.mkdir()
        reference = self.source / 'lora_study/features/adapted/B.metadata.parquet'
        reference.parent.mkdir(parents=True)
        self.metadata = pd.DataFrame(dict(base_id=['a', 'b'], group_id=['ga', 'gb'], source=['WG', 'WG'],
                                          label=[0, 1], outer_fold=[0, 1], prompt_tokens=[2, 3]))
        self.metadata.to_parquet(reference, index=False)
        self.metadata.to_parquet(self.cache / 'B.metadata.parquet', index=False)
        np.save(self.cache / 'offsets.npy', np.array([0, 2, 5], dtype=np.int64), allow_pickle=False)
        raw = np.arange(5 * 2 * 2560, dtype=np.float32).reshape(5, 2, 2560).astype(np.float16)
        raw.tofile(self.cache / 'tokens.bin')
        self.raw = raw
        self.protocol = dict(source_study=str(self.source), adapter_sha256='a' * 64,
            training_protocol_sha256='t' * 64, partition_manifest_sha256='p' * 64,
            B_sha256='b' * 64, cache_layers=[21, 29], rows=2,
            fold_layers={'0': 21, '1': 29, '2': 21, '3': 29, '4': 21})
        self.manifest = dict(complete=True, source_study=str(self.source / 'lora_study'),
            readout_study=str(self.source / 'readouts/adapted_last'), adapter_sha256='a' * 64,
            training_protocol_sha256='t' * 64, ab_protocol_sha256='p' * 64,
            B_sha256='b' * 64, layers=[21, 29], fold_best_layers=self.protocol['fold_layers'],
            hidden_size=2560, shape=[5, 2, 2560], dtype='float16', rows=2, tokens=5,
            path=str(self.cache / 'tokens.bin'), raw_block_outputs=True,
            formal_test_opened=False, truncation=False, chat_template=False, special_tokens=False)

    def store(self, overrides=None):
        manifest = copy.deepcopy(self.manifest)
        manifest.update(overrides or {})
        (self.cache / 'manifest.json').write_text(json.dumps(manifest))
        # Treat the cache as internally valid; this isolates the experiment's
        # obligation to bind it to its own registered adapter/layer identities.
        validated = dict(passed=True, rows=2, tokens=5, layers=manifest['layers'], shape=manifest['shape'],
                         adapter_sha256=manifest['adapter_sha256'],
                         manifest_sha256=experiment.sha256(self.cache / 'manifest.json'))
        with patch('scripts.hanguard.sparse_token_feature_cache.validate', return_value=validated):
            return experiment.TokenStore(self.cache, self.protocol)

    def test_matching_manifest_and_ragged_layer_batch_are_accepted(self):
        store = self.store()
        values, mask = store.batch(np.array([1, 0]), 29, 'cpu')
        torch.testing.assert_close(values[0], torch.as_tensor(self.raw[2:5, 1]))
        torch.testing.assert_close(values[1, :2], torch.as_tensor(self.raw[:2, 1]))
        torch.testing.assert_close(values[1, 2], torch.zeros(2560, dtype=torch.float16))
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False]])

    def test_internally_valid_cache_with_wrong_adapter_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store({'adapter_sha256': 'z' * 64})

    def test_internally_valid_cache_with_wrong_or_reordered_layers_is_rejected(self):
        for layers in ([20, 29], [29, 21]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                self.store({'layers': layers})

    def test_internally_valid_cache_with_wrong_training_partition_or_B_is_rejected(self):
        for field in ('training_protocol_sha256', 'ab_protocol_sha256', 'B_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store({field: 'z' * 64})


if __name__ == '__main__':
    unittest.main()
