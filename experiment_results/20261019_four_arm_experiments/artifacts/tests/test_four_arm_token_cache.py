"""Synthetic CPU checks for chunk integrity, test barrier and ragged readout."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.hanguard import four_arm_token_cache as cache


class _Padded(dict):
    def to(self, device):
        return _Padded({k: v.to(device) for k, v in self.items()})


class _Tokenizer:
    def pad(self, rows, **kwargs):
        multiple = kwargs['pad_to_multiple_of']
        width = max(len(r['input_ids']) for r in rows)
        width = ((width + multiple - 1) // multiple) * multiple
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for n, row in enumerate(rows):
            ids[n, :len(row['input_ids'])] = torch.tensor(row['input_ids'])
            mask[n, :len(row['input_ids'])] = 1
        return _Padded(input_ids=ids, attention_mask=mask)


def _capture(batch):
    ids = batch['input_ids'][batch['attention_mask'].bool()].to(torch.float16)
    return torch.stack((ids, ids * 2), dim=-1)


class TokenCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.study = self.root / 'study'
        self.parent = self.root / 'parent'
        self.data = self.root / 'data'
        self.study.mkdir()
        self.parent.mkdir()
        self.data.mkdir()
        self.lengths = np.array([2, 1, 3], dtype=np.int32)
        self.offsets = np.array([0, 2, 3, 6], dtype=np.int64)
        self.ids = np.array([2, 4, 6, 8, 10, 12], dtype=np.int32)
        self.extraction = dict(token_budget=8, max_micro=2, pad_multiple=1, max_tokens=20)
        self.groups = [x.tolist() for x in cache.base.legacy.microbatches(
            np.argsort(self.lengths, kind='stable'), self.lengths, 8, 2, 1)]
        self.chunks = [dict(chunk_id=n, group_start=n, group_stop=n + 1, rows=len(g)) for n, g in enumerate(self.groups)]
        pp = dict(use_lora=False, adapter_path=None, data_dir=str(self.data), model_path='synthetic',
            model_type='test', num_layers=2, hidden_size=2, extraction=self.extraction,
            inputs={str(self.data / f'{split}.parquet'): 'a' * 64 for split in ('train', 'validation', 'test')})
        cache.dump(self.parent / 'protocol.json', pp)
        cache.dump(self.parent / 'protocol_registration.json', dict(protocol_sha256=cache.sha(self.parent / 'protocol.json')))
        p = dict(use_lora=False, adapter_path=None, parent_study=str(self.parent), data_dir=str(self.data),
            token_cache_dir=str(self.study / 'token_cache'), model_path='synthetic', model_type='test',
            num_layers=2, hidden_size=2, code_sha256={}, token_cache=dict(atol=0, rtol=0))
        cache.dump(self.study / 'protocol.json', p)
        cache.dump(self.study / 'protocol_registration.json', dict(protocol_sha256=cache.sha(self.study / 'protocol.json')))
        for split in ('train', 'validation', 'test'):
            folder = self.parent / 'parallel_features' / split
            folder.mkdir(parents=True)
            np.save(folder / 'token_ids.npy', self.ids)
            np.save(folder / 'lengths.npy', self.lengths)
            np.save(folder / 'offsets.npy', self.offsets)
            pd.DataFrame(dict(base_id=['a', 'b', 'c'], label=[0, 1, 1], category_id=[0, 1, 2])).to_parquet(folder / 'metadata.parquet')
            cache.dump(folder / 'batch_plan.json', dict(rows=3, shape=[3, 2, 2], groups=self.groups, chunks=self.chunks))
            cache.dump(folder / 'prepared_manifest.json', dict(complete=True,
                protocol_sha256=cache.sha(self.parent / 'protocol.json'), input_sha256='a' * 64,
                artifacts={name: cache.sha(folder / name) for name in cache.PREPARED_FILES}))

    def _lock_test(self):
        artifact = self.study / 'selected.pt'
        artifact.write_bytes(b'model weights selected on validation')
        cache.dump(self.study / 'selection_lock.json', dict(complete=True, test_opened=False,
            protocol_sha256=cache.sha(self.study / 'protocol.json'), artifacts={str(artifact): cache.sha(artifact)}))

    def _finish(self, split='train', worker_id='cpu_worker', mutate_calibration=False):
        cache.prepare(self.study, split)
        plan, data = cache._prepared(self.study, split)
        protocol = cache._protocol(self.study)
        previous = np.zeros((3, 2, 2), dtype=np.float16)
        last_ids = self.ids[self.offsets[1:] - 1]
        previous[:, -1, 0] = last_ids
        previous[:, -1, 1] = last_ids * 2
        cp = cache._calibration(self.study, split, worker_id, _capture, data, _Tokenizer(), plan,
                                protocol, previous, 'cpu')
        if mutate_calibration:
            cal = cache.read(cp)
            cal['records'][0]['tokens_sha256'] = '0' * 64
            cache.dump(cp, cal)
        receipts = []
        while True:
            claim = cache.claim_chunk(self.study, split, worker_id, plan)
            if claim is None:
                break
            receipts.append(cache.extract_chunk(self.study, split, worker_id, claim, plan, data,
                _Tokenizer(), _capture, cp, protocol, previous, 'cpu'))
        return receipts

    def test_round_trip_has_all_tokens_and_correct_row_order(self):
        self._finish()
        manifest = cache.completion_manifest_finalize(self.study, 'train')
        self.assertEqual(manifest['shape'], [6, 2])
        self.assertEqual(cache.completion_manifest_finalize(self.study, 'train'), manifest)
        loaded = cache.load(self.study, 'train')
        self.assertEqual(loaded.meta.base_id.tolist(), ['a', 'b', 'c'])
        self.assertEqual(loaded.get_row(0).tolist(), [[2, 4], [4, 8]])
        self.assertEqual(loaded.get_row(2).tolist(), [[8, 16], [10, 20], [12, 24]])
        batch, mask = loaded.padded_batch([2, 1])
        self.assertEqual(batch.shape, (2, 3, 2))
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, False]])
        self.assertTrue(cache.all_complete(self.study, 'train'))

    def test_test_barrier_before_any_parent_payload_access(self):
        for fn in (cache.prepare, cache.load, cache.progress, cache.completion_manifest_finalize):
            with self.subTest(fn=fn.__name__), patch.object(cache, '_parent_identity', side_effect=AssertionError('test payload touched')):
                with self.assertRaisesRegex(ValueError, 'locked before reading test'):
                    fn(self.study, 'test')

    def test_test_selection_lock_and_selected_artifacts_remain_bound(self):
        self._lock_test()
        self._finish('test')
        cache.completion_manifest_finalize(self.study, 'test')
        self.assertEqual(len(cache.load(self.study, 'test')), 3)
        (self.study / 'selected.pt').write_bytes(b'changed after test')
        with self.assertRaisesRegex(ValueError, 'Selection artifact changed'):
            cache.load(self.study, 'test')

    def test_changed_parent_tokenization_rejected(self):
        cache.prepare(self.study, 'train')
        path = self.parent / 'parallel_features' / 'train' / 'token_ids.npy'
        ids = self.ids.copy()
        ids[0] += 1
        np.save(path, ids)
        with self.assertRaisesRegex(ValueError, 'Parent token artifact changed'):
            cache._prepared(self.study, 'train')

    def test_corrupt_committed_feature_rejected(self):
        receipt = self._finish()[0]
        path = Path(receipt['artifacts']['tokens.npy']['path'])
        array = np.load(path, mmap_mode='r+')
        array[0, 0] += 1
        array.flush()
        del array
        with self.assertRaisesRegex(ValueError, 'Committed token artifact bytes changed'):
            cache.completion_manifest_finalize(self.study, 'train')

    def test_duplicate_row_coverage_is_rejected(self):
        receipts = self._finish()
        receipt = receipts[0]
        path = Path(receipt['artifacts']['rows.npy']['path'])
        rows = np.load(path)
        rows[0] = 2
        np.save(path, rows)
        receipt['artifacts']['rows.npy']['sha256'] = cache.sha(path)
        cache.dump(cache._chunk(self.study, 'train', receipt['chunk']['chunk_id']) / 'receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'coverage differs'):
            cache.completion_manifest_finalize(self.study, 'train')

    def test_last_token_parent_comparison_rejects_wrong_backbone(self):
        actual = np.ones((3, 2), np.float16)
        with self.assertRaisesRegex(RuntimeError, 'Raw L32 last tokens differ'):
            cache._compare_last(actual, np.array([0, 1]), self.lengths, np.zeros((3, 2, 2)), {})

    def test_dead_claim_requires_explicit_release_without_deleting_partial(self):
        plan = cache.prepare(self.study, 'train')
        claim = cache.claim_chunk(self.study, 'train', 'worker_a', plan)
        (Path(claim['directory']) / 'tokens.partial.npy').write_bytes(b'uncommitted')
        path = cache._chunk(self.study, 'train', claim['chunk']['chunk_id']) / 'claim.json'
        claim['identity']['start_time_ticks'] = -1
        cache.dump(path, claim)
        self.assertEqual(cache.release_dead_claims(self.study, 'train'), [claim['chunk']['chunk_id']])
        retry = cache.claim_chunk(self.study, 'train', 'worker_b', plan)
        self.assertEqual(retry['attempt'], 2)
        self.assertTrue((Path(claim['directory']) / 'tokens.partial.npy').exists())

    def test_live_claim_is_never_released(self):
        plan = cache.prepare(self.study, 'train')
        cache.claim_chunk(self.study, 'train', 'worker_a', plan)
        self.assertEqual(cache.release_dead_claims(self.study, 'train'), [])

    def test_capture_reads_final_raw_block_and_valid_tokens_only(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x + 1
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(), Block()])
            def forward(self, input_ids, **kwargs):
                x = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
                for layer in self.layers:
                    x = layer(x)
                return x * 100  # A later normalization must not enter the raw cache.
        capture = cache.RawLastTokenCapture(Backbone())
        try:
            actual = capture(_Padded(input_ids=torch.tensor([[1, 2], [3, 0]]), attention_mask=torch.tensor([[1, 1], [1, 0]])))
            self.assertEqual(actual.tolist(), [[3, 3], [4, 4], [5, 5]])
            self.assertEqual(actual.dtype, torch.float16)
        finally:
            capture.close()


if __name__ == '__main__':
    unittest.main()
