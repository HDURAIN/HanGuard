"""CPU checks for real Guard template alignment and raw-block cache semantics."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from scripts.hanguard import qwen3guard_supervised_features_v2 as features


ROOT = Path(__file__).resolve().parents[1]


class GuardFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3Guard-Gen-4B', local_files_only=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'study'
        self.output.mkdir()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.prompts = ['中文测试', 'A 😊 𠮷 B', ' \n\t ', '<|im_end|>警告', 'longer 中文 text abc']
        self.protocol = dict(model_path=str(ROOT / 'models/Qwen3Guard-Gen-4B'),
            parent_data_dir=str(self.data), hidden_size=16, num_hidden_layers=2,
            tasks=['binary', 'category'], seeds=[42],
            arms=['last_mlp', 'mean_pool_mlp', 'shared_attention_mlp'],
            shard_rows=2, data_sha256={}, threads=1,
            extraction=dict(max_tokens=32768, max_micro=1, token_budget=4096,
                            canonical_singleton=True))
        for split in features.SPLITS:
            frame = pd.DataFrame(dict(prompt=self.prompts,
                base_id=[f'{split}_{i}' for i in range(len(self.prompts))],
                group_id=[f'{split}_g{i}' for i in range(len(self.prompts))],
                source=['chinese_curated'] * len(self.prompts),
                prompt_harm_label=['harmful'] * len(self.prompts),
                category_id=[1, 2, 3, 4, 5], type_label_mask=[True] * len(self.prompts),
                split=[split] * len(self.prompts)))
            frame.to_parquet(self.data / f'{split}.parquet', index=False)
            self.protocol['data_sha256'][split] = features.sha(self.data / f'{split}.parquet')
        features.write(self.output / 'protocol.json', self.protocol)

    def tiny_model(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=152064, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8, max_position_embeddings=32768,
                            attention_dropout=0., pad_token_id=151643)
        config._attn_implementation = 'sdpa'
        return Qwen3Model(config).eval().requires_grad_(False)

    def make_test_lock(self):
        digest = features.sha(self.output / 'protocol.json')
        artifacts, selections = {}, {}
        for task in self.protocol['tasks']:
            folder = self.output / 'jobs' / task / 's42'
            folder.mkdir(parents=True)
            selected_artifacts = {}
            for name in ('selected_heads.pt', 'validation_predictions.parquet'):
                path = folder / name
                path.write_bytes(b'locked test fixture')
                selected_artifacts[str(path)] = features.sha(path)
            features.write(folder / 'selection.json', dict(protocol_sha256=digest,
                task=task, seed=42, arms={arm: {} for arm in self.protocol['arms']},
                artifacts=selected_artifacts))
            selection_sha = features.sha(folder / 'selection.json')
            selections[f'{task}/s42'] = selection_sha
            artifacts[str(folder / 'selection.json')] = selection_sha
            artifacts.update(selected_artifacts)
        features.write(self.output / 'selection_lock.json', dict(complete=True,
            protocol_sha256=digest, selections=selections, artifacts=artifacts))

    def test_real_template_offsets_cover_unicode_whitespace_and_literal_special_tokens(self):
        prefix, suffix = features.template_parts(self.tokenizer)
        self.assertEqual(len(prefix), 919)
        for text in self.prompts + ['a' * 1000, '“引用”\r\nUSER: x', '最后一行\n']:
            with self.subTest(text=text):
                ids, start, end = features.encode_text(self.tokenizer, text, prefix, suffix)
                encoded = self.tokenizer(prefix + text + suffix, add_special_tokens=False,
                                         return_offsets_mapping=True)
                expected = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                            if b > len(prefix) and a < len(prefix) + len(text) and b > a]
                self.assertEqual(list(range(start, end)), expected)
                self.assertEqual(ids, encoded['input_ids'])
        with self.assertRaises(ValueError):
            features.encode_text(self.tokenizer, '', prefix, suffix)
        with self.assertRaises(ValueError):
            features.encode_text(self.tokenizer, 'hello', prefix, suffix, max_tokens=2)

    def test_no_test_tokenization_before_matching_selection_lock(self):
        with self.assertRaises(FileNotFoundError):
            features.prepare(self.output, 'test')
        self.assertFalse((self.output / 'features' / 'test').exists())
        features.write(self.output / 'selection_lock.json', {'protocol_sha256': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'selection lock'):
            features.prepare(self.output, 'test')
        features.write(self.output / 'selection_lock.json',
                       {'protocol_sha256': features.sha(self.output / 'protocol.json')})
        with self.assertRaisesRegex(ValueError, 'selections are incomplete'):
            features.prepare(self.output, 'test')
        self.make_test_lock()
        self.assertEqual(features.prepare(self.output, 'test')['rows'], 5)
        (self.output / 'jobs/category/s42/selected_heads.pt').write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            features.prepare(self.output, 'test')

    def test_prepare_is_row_aligned_immutable_and_raw_data_bound(self):
        registration = features.prepare(self.output, 'train')
        self.assertEqual([(s['row_start'], s['row_end']) for s in registration['shards']],
                         [(0, 2), (2, 4), (4, 5)])
        folder = self.output / 'features' / 'train'
        meta = pd.read_parquet(folder / 'prepared/metadata.parquet')
        self.assertEqual(meta.base_id.tolist(), [f'train_{i}' for i in range(5)])
        self.assertNotIn('prompt', meta)
        self.assertEqual(features.prepare(self.output, 'train'), registration)
        with (folder / 'prepared/lengths.npy').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Prepared feature input changed'):
            features.prepare(self.output, 'train')
        with (self.data / 'validation.parquet').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'raw data changed'):
            features.prepare(self.output, 'validation')

    def test_exact_raw_block_capture_and_padding_invariance(self):
        model = self.tiny_model()
        sequences = [[4, 5, 6, 7], [8, 9]]
        starts, ends = [1, 0], [3, 1]
        content, last = features.forward_rows(model, sequences, starts, ends, 'cpu', 0)
        captured = []
        handle = model.layers[-1].register_forward_hook(lambda _m, _a, out: captured.append(out.detach().clone()))
        try:
            with torch.inference_mode():
                normalized = model(input_ids=torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]), use_cache=False).last_hidden_state
        finally:
            handle.remove()
        for i, seq in enumerate(sequences):
            expected = captured[0][i, starts[i]:ends[i]].numpy().astype(np.float16)
            np.testing.assert_array_equal(content[i], expected)
            np.testing.assert_array_equal(last[i], captured[0][i, len(seq) - 1].numpy().astype(np.float16))
            self.assertFalse(np.allclose(last[i], normalized[i, len(seq) - 1].numpy()))
        checks = features._calibrate(model, sequences[:1], starts[:1], ends[:1], 'cpu', 0)
        self.assertTrue(all(item['repeated_padded_batch_bitwise_equal'] for item in checks))
        self.assertEqual(checks[0]['relative_rmse'], 0.)
        self.assertEqual(checks[0]['cosine'], 1.)
        self.assertTrue(checks[0]['repeated_canonical_input_bitwise_equal'])
        with self.assertRaisesRegex(ValueError, 'canonical singleton'):
            features._calibrate(model, sequences, starts, ends, 'cpu', 0)
        with self.assertRaisesRegex(ValueError, 'token span'):
            features.forward_rows(model, sequences, [0, 0], [99, 1], 'cpu')

    def test_cpu_end_to_end_shards_mmap_completion_and_corruption_detection(self):
        registration = features.prepare(self.output, 'validation')
        model = self.tiny_model()
        for shard in range(3):
            receipt = features.extract(self.output, 'validation', shard, 'cpu', model=model)
            folder = self.output / 'features/validation' / f'shard_{shard:05d}'
            descriptors = registration['shards'][shard]
            tokens = np.load(folder / 'token_features.npy', mmap_mode='r')
            last = np.load(folder / 'last.npy', mmap_mode='r')
            offsets = np.load(folder / 'offsets.npy')
            self.assertEqual(tokens.shape, (descriptors['content_tokens'], 16))
            self.assertEqual(last.shape, (descriptors['row_end'] - descriptors['row_start'], 16))
            self.assertEqual(offsets[-1], len(tokens))
            self.assertTrue(np.isfinite(tokens).all())
            metadata = features.read(folder / 'metadata.json')
            self.assertTrue(all(len(batch) == 1 for batch in metadata['input_batches']))
            self.assertTrue(metadata['calibration'][0]['canonical_singleton'])
            self.assertEqual(receipt, features.extract(self.output, 'validation', shard, 'cpu', model=model))
        self.assertEqual(features.complete_jobs(self.output, 'validation', verify=True), [0, 1, 2])
        self.assertEqual(features.pending_jobs(self.output, 'validation'), [])
        target = self.output / 'features/validation/shard_00000/last.npy'
        with target.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'stat changed'):
            features.complete_jobs(self.output, 'validation')

    def test_auto_model_loading_full_causal_lm_checkpoint_uses_real_transformers_api(self):
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8)
        model_path = self.root / 'tiny_saved_model'
        Qwen3ForCausalLM(config).save_pretrained(model_path)
        self.protocol['model_path'] = str(model_path)
        features.write(self.output / 'protocol.json', self.protocol)
        loaded = features.load_model(self.output, 'cpu')
        self.assertIsInstance(loaded, Qwen3Model)
        content, last = features.forward_rows(loaded, [[1, 2, 3], [4, 5]], [0, 0], [2, 1], 'cpu', 0)
        self.assertEqual(content[0].shape, (2, 16))
        self.assertEqual(last.shape, (2, 16))

    def test_singleton_oom_fails_closed_without_partial_cache_reuse(self):
        features.prepare(self.output, 'train')
        model = self.tiny_model()
        with mock.patch.object(features, 'forward_rows',
                side_effect=torch.cuda.OutOfMemoryError('Simulated singleton allocator exhaustion')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                features.extract(self.output, 'train', 0, 'cpu', model=model)
        self.assertEqual(features.complete_jobs(self.output, 'train'), [])
        self.assertTrue(list((self.output / 'features/train').glob('.shard_00000.*/failed.json')))
        receipt = features.extract(self.output, 'train', 0, 'cpu', model=model)
        self.assertEqual(receipt['row_end'] - receipt['row_start'], 2)
        metadata = features.read(self.output / 'features/train/shard_00000/metadata.json')
        self.assertEqual(metadata['oom_splits'], [])
        self.assertEqual(features.complete_jobs(self.output, 'train', verify=True), [0])

    def test_batch_plan_fixed_by_lengths_and_registered_budget(self):
        lengths = np.asarray([9, 2, 2, 5, 20])
        batches = list(features.length_batches(range(5), lengths, max_micro=3, token_budget=12))
        self.assertEqual(batches, [[0], [1], [2], [3], [4]])
        self.assertEqual(sorted(sum(batches, [])), list(range(5)))
        self.assertTrue(all(len(batch) == 1 for batch in batches))
        with self.assertRaises(ValueError):
            list(features.length_batches([0], [0]))


if __name__ == '__main__':
    unittest.main()
