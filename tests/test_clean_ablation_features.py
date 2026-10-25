"""CPU-only checks for fresh same-forward features and immutable chunk lineage."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.hanguard import clean_ablation_features as cache


class Padded(dict):
    def to(self, device):
        return Padded({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return dict(input_ids=[[int(v) for v in text.split()] for text in texts])

    def pad(self, rows, **kwargs):
        multiple = kwargs['pad_to_multiple_of']
        width = max(len(row['input_ids']) for row in rows)
        width = ((width + multiple - 1) // multiple) * multiple
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for number, row in enumerate(rows):
            ids[number, :len(row['input_ids'])] = torch.tensor(row['input_ids'])
            mask[number, :len(row['input_ids'])] = 1
        return Padded(input_ids=ids, attention_mask=mask)


class Layer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(), Layer()])

    def forward(self, input_ids, **kwargs):
        value = torch.stack((input_ids, input_ids * 2), dim=-1).float()
        for layer in self.layers:
            value = layer(value)
        return value


class CleanFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / 'study'
        self.study.mkdir()
        self.data = self.study / 'data'
        self.data.mkdir()
        self.model = self.root / 'model'
        self.model.mkdir()
        frame = pd.DataFrame(dict(prompt=['8 10 12', '2 4', '6'], base_id=['c', 'a', 'b'],
            group_id=['gc', 'ga', 'gb'], source=['jailbench'] * 3,
            prompt_harm_label=['harmful', 'unharmful', 'harmful'], category_id=[2, 0, 1],
            type_label_mask=[True, False, True], prompt_tokens=[999] * 3, split=['train'] * 3))
        for split in ('train', 'validation', 'test'):
            frame.assign(split=split).to_parquet(self.data / f'{split}.parquet', index=False)
        self.protocol = dict(model_path=str(self.model), model_type='test', num_layers=2, hidden_size=2,
            data_dir=str(self.data), use_lora=False, adapter_path=None, code_sha256={},
            inputs={str(self.data / f'{split}.parquet'): cache.sha(self.data / f'{split}.parquet')
                    for split in ('train', 'validation', 'test')},
            features_dir=str(self.study / 'features'), rows={split: 3 for split in ('train', 'validation', 'test')},
            extraction=dict(token_budget=8, max_micro=2, pad_multiple=1, max_tokens=20, groups_per_chunk=1))
        self._write_protocol()
        self.patch_tokenizer = patch.object(cache, '_tokenizer', return_value=Tokenizer())
        self.patch_tokenizer.start()
        self.addCleanup(self.patch_tokenizer.stop)

    def _write_protocol(self):
        cache.dump(self.study / 'protocol.json', self.protocol)
        cache.dump(self.study / 'protocol_registration.json', dict(protocol_sha256=cache.sha(self.study / 'protocol.json')))

    def _finish(self, worker='cpu_1'):
        plan = cache.prepare(self.study, 'train')
        _, data = cache._prepared(self.study, 'train')
        capture = cache.JointCapture(Backbone())
        self.addCleanup(capture.close)
        calibration = cache._calibration(self.study, worker, capture, Tokenizer(), 'cpu')
        while True:
            claim = cache.claim_chunk(self.study, 'train', worker, plan)
            if claim is None:
                break
            cache.extract_chunk(self.study, 'train', worker, claim, plan, data,
                                Tokenizer(), capture, calibration, device='cpu')
        return plan, data

    def test_fresh_tokenization_sorted_rows_and_recomputed_lengths(self):
        plan = cache.prepare(self.study, 'train')
        _, data = cache._prepared(self.study, 'train')
        self.assertEqual(data['meta'].base_id.tolist(), ['a', 'b', 'c'])
        self.assertEqual(data['lengths'].tolist(), [2, 1, 3])
        self.assertEqual(data['meta'].prompt_tokens.tolist(), [2, 1, 3])
        self.assertEqual(data['encoded'][2], [8, 10, 12])
        self.assertEqual(plan['total_tokens'], 6)
        self.assertFalse((self.study / 'features' / 'test').exists())

    def test_joint_capture_same_forward_padding_and_all_layers(self):
        capture = cache.JointCapture(Backbone())
        self.addCleanup(capture.close)
        batch = Tokenizer().pad([dict(input_ids=[2, 4]), dict(input_ids=[6])], pad_to_multiple_of=4)
        last, tokens = capture(batch)
        self.assertEqual(last.shape, (2, 2, 2))
        self.assertEqual(tokens.tolist(), [[4, 6], [6, 10], [8, 14]])
        self.assertEqual(last[:, -1].tolist(), [[6, 10], [8, 14]])
        self.assertEqual(last[:, 0].tolist(), [[5, 9], [7, 13]])

    def test_capture_rejects_internal_holes(self):
        capture = cache.JointCapture(Backbone())
        self.addCleanup(capture.close)
        with self.assertRaisesRegex(ValueError, 'right padding'):
            capture(Padded(input_ids=torch.tensor([[1, 0, 2]]), attention_mask=torch.tensor([[1, 0, 1]])))

    def test_over_budget_refuses_truncation(self):
        self.protocol['extraction']['max_tokens'] = 2
        self._write_protocol()
        with self.assertRaisesRegex(ValueError, 'truncation'):
            cache.prepare(self.study, 'train')

    def test_historical_paths_are_rejected(self):
        self.protocol['parent_study'] = '/some/old/outputs'
        self._write_protocol()
        with self.assertRaisesRegex(ValueError, 'Historical'):
            cache.prepare(self.study, 'train')

    def test_external_input_is_rejected(self):
        self.protocol['inputs']['/old/outputs/test.npy'] = '0' * 64
        self._write_protocol()
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache.prepare(self.study, 'train')

    def test_roundtrip_ragged_last_advanced_index_and_manifest(self):
        self._finish()
        manifest = cache.finalize(self.study, 'train')
        loaded = cache.load(self.study, 'train')
        self.assertEqual(loaded.meta.base_id.tolist(), ['a', 'b', 'c'])
        self.assertEqual(loaded.get_row(2).tolist(), [[10, 18], [12, 22], [14, 26]])
        self.assertEqual(loaded.last[[2, 0], 1].tolist(), [[14, 26], [6, 10]])
        self.assertEqual(loaded.last[1, 0].tolist(), [7, 13])
        self.assertEqual(loaded.last[[2, 0]].shape, (2, 2, 2))
        self.assertEqual(manifest, cache.finalize(self.study, 'train'))
        values, mask = loaded.padded_batch([2, 1])
        self.assertEqual(values.shape, (2, 3, 2))
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, False]])
        self.assertTrue(cache.all_complete(self.study, 'train'))

    def test_payload_tamper_is_detected(self):
        self._finish()
        cache.finalize(self.study, 'train')
        path = next((self.study / 'features' / 'train' / 'chunks').glob('*/attempt_*/tokens.npy'))
        values = np.load(path, mmap_mode='r+')
        values[0, 0] += 1
        values.flush()
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'):
            cache.load(self.study, 'train')

    def test_prepared_token_tamper_is_detected(self):
        cache.prepare(self.study, 'train')
        path = self.study / 'features' / 'train' / 'token_ids.npy'
        values = np.load(path, mmap_mode='r+')
        values[0] += 1
        values.flush()
        with self.assertRaisesRegex(ValueError, 'prepared artifact changed'):
            cache._prepared(self.study, 'train')

    def test_bad_repeated_calibration_fails(self):
        cache.prepare(self.study, 'train')
        calls = [0]
        capture = cache.JointCapture(Backbone())
        self.addCleanup(capture.close)

        def nondeterministic(batch):
            last, tokens = capture(batch)
            calls[0] += 1
            if calls[0] % 2 == 0:
                last += 1
                tokens += 1
            return last, tokens

        with self.assertRaisesRegex(RuntimeError, 'Repeated fresh'):
            cache._calibration(self.study, 'bad', nondeterministic, Tokenizer(), 'cpu')

    def test_cross_process_signature_mismatch_fails(self):
        cache.prepare(self.study, 'train')
        capture = cache.JointCapture(Backbone())
        self.addCleanup(capture.close)
        cache._calibration(self.study, 'first', capture, Tokenizer(), 'cpu')

        def shifted(batch):
            last, tokens = capture(batch)
            return last + 1, tokens + 1

        with self.assertRaisesRegex(RuntimeError, 'calibration baseline'):
            cache._calibration(self.study, 'second', shifted, Tokenizer(), 'cpu')
        self.assertTrue((self.study / 'calibration' / 'second.failed.json').exists())

    def test_live_claim_not_released_and_partial_attempt_not_reused(self):
        plan = cache.prepare(self.study, 'train')
        claim = cache.claim_chunk(self.study, 'train', 'worker', plan)
        self.assertEqual(cache.release_dead_claims(self.study, 'train'), [])
        (Path(claim['directory']) / 'partial.bin').write_bytes(b'partial')
        _, data = cache._prepared(self.study, 'train')
        with self.assertRaisesRegex(ValueError, 'Partial joint'):
            cache.extract_chunk(self.study, 'train', 'worker', claim, plan, data, None, None, None, 'cpu')
        claim['identity']['start_time_ticks'] -= 1
        cache.dump(cache._chunk(self.study, 'train', claim['chunk']['chunk_id']) / 'claim.json', claim)
        self.assertEqual(cache.release_dead_claims(self.study, 'train'), [claim['chunk']['chunk_id']])
        again = cache.claim_chunk(self.study, 'train', 'worker2', plan)
        self.assertEqual(again['attempt'], 2)
        self.assertEqual(list(Path(again['directory']).iterdir()), [])

    def test_test_barrier_runs_before_reading_any_raw_test(self):
        with patch.object(cache, '_barrier', side_effect=RuntimeError('test selection lock missing')):
            with self.assertRaisesRegex(RuntimeError, 'selection lock'):
                cache.prepare(self.study, 'test')
        self.assertFalse((self.study / 'features' / 'test').exists())


if __name__ == '__main__':
    unittest.main()
