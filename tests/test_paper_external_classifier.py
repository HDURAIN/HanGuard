"""CPU-only checks for full-document external classification and resume."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.hanguard import paper_external_classifier as external


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.dropout = nn.Dropout(.2)
        self.linear = nn.Linear(8, 5)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask):
        values = self.embedding(input_ids)
        mask = attention_mask[..., None].float()
        pooled = (values * mask).sum(1) / mask.sum(1).clamp_min(1)
        return SimpleNamespace(logits=self.linear(self.dropout(pooled)))


def tiny_data(count=10):
    categories = np.arange(count) % 5 + 1
    meta = pd.DataFrame(dict(base_id=[f'id_{i:03}' for i in range(count)],
        group_id=[f'g_{i // 2}' for i in range(count)], source=['chinese_curated'] * count,
        category_id=categories, label=categories - 1,
        raw_token_count=[3] * count, window_count=[1 + (i % 2) for i in range(count)]))
    docs = [dict(input_ids=[[1, int(c) + 2, 2]] * (1 + i % 2)) for i, c in enumerate(categories)]
    return dict(meta=meta, docs=docs, manifest={'documents': count, 'truncation': False})


class WindowTests(unittest.TestCase):
    def test_bert_special_token_wrapper_accepts_empty_and_rejects_invalid_ids(self):
        class Tokenizer:
            cls_token_id, sep_token_id = 1, 2
            def __len__(self): return 32
        tokenizer = Tokenizer()
        self.assertEqual(external.bert_window(tokenizer, []), [1, 2])
        self.assertEqual(external.bert_window(tokenizer, [3, 4]), [1, 3, 4, 2])
        for cls_id, sep_id in [(None, 2), (True, 2), (-1, 2), (32, 2), (2, 2)]:
            with self.subTest(cls_id=cls_id, sep_id=sep_id):
                tokenizer.cls_token_id, tokenizer.sep_token_id = cls_id, sep_id
                with self.assertRaises(ValueError):
                    external.bert_window(tokenizer, [])
        tokenizer.cls_token_id, tokenizer.sep_token_id = 1, 2
        for values in [[-1], [32], [None], [True]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    external.bert_window(tokenizer, values)

    def test_full_coverage_long_text_and_overlap(self):
        for length in (0, 1, 509, 510, 511, 512, 1024, 3199, 20000):
            with self.subTest(length=length):
                spans = external.windows(list(range(length)))
                coverage = {i for start, end in spans for i in range(start, end)}
                self.assertEqual(coverage, set(range(length)))
                self.assertTrue(all(end - start <= 510 for start, end in spans))
                self.assertEqual(spans[-1][1], length)
                self.assertEqual(len(spans), 1 if length <= 510 else len(spans))
                for previous, current in zip(spans, spans[1:]):
                    self.assertEqual(previous[1] - current[0], 128)
        with self.assertRaises(ValueError):
            external.windows([1], overlap=510)

    def test_native_mask_preserves_only_native_known_harmful_types(self):
        rows = []
        for index, (source, harm, category, known) in enumerate([
            ('chinese_curated', 'harmful', 1, True), ('jailbench', 'harmful', 5, True),
            ('wildguard_zh', 'harmful', 2, True), ('jailbench', 'harmful', -1, False),
            ('chinese_curated', 'unharmful', 0, False)]):
            rows.append(dict(base_id=f'id{index}', group_id=f'g{index}', source=source,
                prompt='example', prompt_harm_label=harm, category_id=category, type_label_mask=known))
        result = external._native(pd.DataFrame(rows))
        self.assertEqual(result.category_id.tolist(), [1, 5])

    def test_document_logits_are_mean_windows_before_document_ce(self):
        external._seed(3)
        model = TinyEncoder().eval()
        docs = [dict(input_ids=[[1, 3, 2], [1, 4, 2], [1, 5, 2]]), dict(input_ids=[[1, 6, 2]])]
        logits = external.document_logits(model, docs, 'cpu', window_micro=2)
        windows = [external.document_logits(model, [dict(input_ids=[window])], 'cpu')[0]
                   for doc in docs for window in doc['input_ids']]
        manual = torch.stack([torch.stack(windows[:3]).mean(0), windows[3]])
        self.assertTrue(torch.allclose(logits, manual, atol=1e-7, rtol=1e-7))
        target = torch.tensor([1, 4])
        loss = F.cross_entropy(logits, target)
        window_loss = F.cross_entropy(torch.stack(windows), torch.tensor([1, 1, 1, 4]))
        self.assertFalse(torch.allclose(loss, window_loss))
        first = torch.autograd.grad(loss, model.linear.weight, retain_graph=True)[0]
        second = torch.autograd.grad(F.cross_entropy(manual, target), model.linear.weight)[0]
        self.assertTrue(torch.allclose(first, second, atol=1e-7, rtol=1e-7))

    def test_window_micro_batches_are_invariant_at_eval_and_keep_long_singleton(self):
        external._seed(7)
        model = TinyEncoder().eval()
        docs = [dict(input_ids=[[1, 3, 2]] * 13), dict(input_ids=[[1, 4, 2]])]
        a = external.document_logits(model, docs, 'cpu', window_micro=1)
        b = external.document_logits(model, docs, 'cpu', window_micro=4)
        self.assertTrue(torch.allclose(a, b, atol=1e-7))
        loss = F.cross_entropy(a, torch.tensor([2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        data = tiny_data(2)
        data['docs'][0] = docs[0]
        self.assertEqual(list(external.document_groups([0, 1], data, 4)), [[0], [1]])

    def test_macro_oom_replays_partial_gradients_but_steps_optimizer_once(self):
        model = TinyEncoder()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        calls = []
        def attempt(model, optimizer, data, indices, device, **kwargs):
            calls.append(kwargs['window_limit'])
            optimizer.zero_grad(set_to_none=True)
            if len(calls) == 1:
                for parameter in model.parameters():
                    parameter.grad = torch.full_like(parameter, 999.)
                raise torch.OutOfMemoryError('synthetic forward failure')
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            return 1.25
        events = []
        with patch.object(external, '_train_attempt', side_effect=attempt), patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            value, limit = external.train_macro(model, optimizer, tiny_data(2), [0, 1], 'cpu',
                seed=42, epoch=0, step=0, window_limit=4, events=events)
        self.assertEqual((value, limit), (1.25, 2))
        self.assertEqual(calls, [4, 2])
        self.assertEqual(len(events), 1)
        step.assert_called_once()

    def test_optimizer_oom_is_not_silently_replayed_after_partial_update(self):
        model = TinyEncoder()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with patch.object(external, '_train_attempt', return_value=1.), patch.object(optimizer, 'step', side_effect=torch.OutOfMemoryError):
            with self.assertRaises(torch.OutOfMemoryError):
                external.train_macro(model, optimizer, tiny_data(2), [0, 1], 'cpu',
                    seed=42, epoch=0, step=0, window_limit=4, events=[])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'external'
        self.parent = self.root / 'parent'
        self.output.mkdir()
        self.addCleanup(self.temp.cleanup)
        self.settings = dict(external.SETTINGS, epochs=3, patience=10, effective_batch=4,
                             micro_documents=2, micro_windows=2, bootstrap_draws=20)
        self.setting_patch = patch.object(external, 'SETTINGS', self.settings)
        self.setting_patch.start()
        self.addCleanup(self.setting_patch.stop)
        self.data = {'train': tiny_data(10), 'validation': tiny_data(5), 'test': tiny_data(5)}
        self.protocol = dict(schema=external.SCHEMA, model='fake', parent=str(self.parent),
            seeds=list(external.SEEDS), settings=self.settings,
            model_origin={'repo_id': 'fake', 'revision': 'a' * 40},
            comparison_note='same labels, different backbone and training budget')
        external.write(self.output / 'protocol.json', self.protocol)
        external.write(self.output / 'protocol_registration.json', {'protocol_sha256': external.sha(self.output / 'protocol.json')})
        for split in ('train', 'validation'):
            external.write(self.output / 'data' / split / 'manifest.json', {'fixture': split})
        (self.output / 'source_snapshot').mkdir()
        (self.output / 'source_snapshot/paper_external_classifier.py').write_text('fixture source')
        parent_results = dict(per_seed={})
        for seed in external.SEEDS:
            frame = self.data['test']['meta'][['base_id', 'category_id']].copy()
            for arm in ('category_only', 'fusion_category'):
                for k in range(5):
                    frame[f'{arm}_p{k}'] = (frame.category_id == k + 1).astype(float)
            path = self.parent / 'heads/category' / f's{seed}' / 'test_predictions.parquet'
            external.save_frame(frame, path)
            parent_results['per_seed'][str(seed)] = dict(category=dict(predictions_sha256=external.sha(path)))
        external.write(self.parent / 'results.json', parent_results)
        self.verify_patch = patch.object(external, 'verify', return_value=self.protocol)
        self.verify_patch.start()
        self.addCleanup(self.verify_patch.stop)
        self.model_patch = patch.object(external, '_load_model', side_effect=self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.data_patch = patch.object(external, 'load_data', side_effect=self.load)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)

    def model(self, path, seed, device):
        external._seed(seed)
        return TinyEncoder().to(device)

    def load(self, output, split):
        if split == 'test':
            external.verify_lock(output)
        return self.data[split]

    def test_full_three_seed_train_selection_barrier_test_summary_and_archive(self):
        self.assertFalse((self.output / 'selection_lock.json').exists())
        for seed in external.SEEDS:
            selected = external.fit(self.output, seed, 'cpu')
            self.assertEqual(selected['selection_scope'], 'validation_only')
            self.assertFalse(selected['formal_test_opened'])
            self.assertEqual(len(external.read(self.output / 'seeds' / f's{seed}' / 'history.json')), 3)
            self.assertFalse((self.output / 'data/test').exists())
        state = external.progress(self.output)
        self.assertEqual(state['completed_jobs'], 3)
        self.assertEqual([item['phase'] for item in state['pending_jobs']], ['evaluate'] * 3)
        self.assertTrue((self.output / 'selection_lock.json').exists())
        for seed in external.SEEDS:
            result = external.fit(self.output, seed, 'cpu')
            self.assertTrue(result['complete'])
            self.assertEqual(result['rows'], 5)
        self.assertTrue(external.is_complete(self.output))
        result = external.read(self.output / 'results.json')
        self.assertEqual(len(result['per_seed']), 3)
        self.assertIn('macro_f1', result['aggregate'])
        archive = self.output / 'archive'
        manifest = external.read(archive / 'manifest.json')
        self.assertTrue(all(external.sha(archive / name) == digest for name, digest in manifest['files'].items()))
        saved = pd.read_parquet(archive / 's42/test_predictions.parquet')
        self.assertFalse({'prompt', 'text', 'input_ids'} & set(saved))

    def test_test_cannot_open_before_every_seed_selected(self):
        external.fit(self.output, 42, 'cpu')
        result = external.fit(self.output, 42, 'cpu')
        self.assertEqual(result['state'], 'waiting_for_other_selections')
        self.assertFalse((self.output / 'selection_lock.json').exists())
        with self.assertRaises(FileNotFoundError):
            external.evaluate(self.output, 42, 'cpu')
        self.assertFalse((self.output / 'seeds/s42/test_predictions.parquet').exists())

    def test_changed_selected_checkpoint_is_rejected_before_test(self):
        for seed in external.SEEDS:
            external.fit(self.output, seed, 'cpu')
        path = self.output / 'seeds/s42/selected_model.pt'
        with path.open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'changed'):
            external.fit(self.output, 42, 'cpu')

    def test_epoch_resume_matches_uninterrupted_dropout_training_bitwise(self):
        uninterrupted = external.fit(self.output, 42, 'cpu')
        final = torch.load(self.output / 'seeds/s42/selected_model.pt', weights_only=True)['state']
        folder = self.output / 'seeds/s42'
        import shutil
        shutil.rmtree(folder)
        original = external.train_macro
        def interrupted(*args, **kwargs):
            if kwargs['epoch'] == 1:
                raise RuntimeError('interrupt after one atomic epoch')
            return original(*args, **kwargs)
        with patch.object(external, 'train_macro', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                external.fit(self.output, 42, 'cpu')
        self.assertTrue((folder / 'epoch_resume.pt').exists())
        self.assertFalse((folder / 'selection.json').exists())
        resumed = external.fit(self.output, 42, 'cpu')
        final_resumed = torch.load(folder / 'selected_model.pt', weights_only=True)['state']
        self.assertEqual(uninterrupted['validation_ce'], resumed['validation_ce'])
        self.assertTrue(all(torch.equal(final[name], final_resumed[name]) for name in final))

    def test_resume_with_different_backend_is_rejected_before_loading_tensors(self):
        original = external.train_macro
        def interrupted(*args, **kwargs):
            if kwargs['epoch'] == 1:
                raise RuntimeError('pause')
            return original(*args, **kwargs)
        with patch.object(external, 'train_macro', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                external.fit(self.output, 42, 'cpu')
        saved = torch.load(self.output / 'seeds/s42/epoch_resume.pt', weights_only=True)
        saved['identity']['device_type'] = 'cuda'
        external.save_torch(self.output / 'seeds/s42/epoch_resume.pt', saved)
        with self.assertRaisesRegex(ValueError, 'device differs'):
            external.fit(self.output, 42, 'cpu')


class AdmissionTests(unittest.TestCase):
    def test_gpu_requires_exact_own_process_record_protocol_uuid_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            external.write(output / 'protocol.json', {'fixture': True})
            proc = Path('/proc') / str(os.getpid())
            stat = (proc / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            record = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=int(fields[19]),
                command_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest(),
                followup_protocol_sha256=external.sha(output / 'protocol.json'),
                gpu_uuid='GPU-test', allocator_cap_gib=8.)
            path = output / 'admission.json'
            external.write(path, record)
            environment = dict(CUDA_VISIBLE_DEVICES='GPU-test',
                HANGUARD_FOLLOWUP_ADMITTED=external.sha(output / 'protocol.json'),
                HANGUARD_FOLLOWUP_ADMISSION_RECORD=str(path))
            with patch.dict(os.environ, environment):
                external._admission(output, 'cuda')
                for key, value in [('pid', os.getpid() + 1), ('start_time_ticks', 1),
                                   ('gpu_uuid', 'GPU-other'), ('allocator_cap_gib', 14.)]:
                    with self.subTest(key=key):
                        external.write(path, dict(record, **{key: value}))
                        with self.assertRaises(RuntimeError):
                            external._admission(output, 'cuda')
            external._admission(output, 'cpu')


class InitializationTests(unittest.TestCase):
    def test_real_local_transformers_tokenizer_prepares_synthetic_full_documents(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(external.DEFAULT_MODEL, local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, output = root / 'synthetic_parent', root / 'prepared'
            frame = pd.DataFrame(dict(base_id=['a', 'b', 'c'], group_id=['ga', 'gb', 'gc'],
                source=['chinese_curated'] * 3, prompt=['合成文本' * 400, '合成短文本', '\u200b'],
                category_id=[1, 2, 5], prompt_harm_label=['harmful'] * 3, type_label_mask=[True] * 3))
            external.save_frame(frame, parent / 'data/train.parquet')
            with patch.object(external, 'EXPECTED_ROWS', dict(train=3, validation=3, test=3)):
                manifest = external._prepare(output, parent, 'train', tokenizer)
            docs = [json.loads(line) for line in (output / 'data/train/windows.jsonl').read_text().splitlines()]
            self.assertEqual(manifest['documents'], 3)
            self.assertEqual(manifest['long_documents'], 1)
            self.assertEqual(manifest['empty_token_documents'], 1)
            self.assertFalse(manifest['truncation'])
            for text, doc in zip(frame.prompt, docs):
                raw = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                self.assertEqual(doc['raw_token_count'], len(raw))
                covered = {i for start, end in doc['spans'] for i in range(start, end)}
                self.assertEqual(covered, set(range(len(raw))))
                for (start, end), ids in zip(doc['spans'], doc['input_ids']):
                    self.assertEqual(ids, [tokenizer.cls_token_id, *raw[start:end], tokenizer.sep_token_id])
                    self.assertLessEqual(len(ids), 512)
            self.assertEqual(docs[2]['input_ids'], [[tokenizer.cls_token_id, tokenizer.sep_token_id]])
            self.assertFalse(torch.cuda.is_initialized())

    def test_real_preparation_registration_no_test_text_and_tamper_detection(self):
        class Tokenizer:
            cls_token_id, sep_token_id = 1, 2
            def __len__(self): return 32
            def encode(self, text, **kwargs):
                self.assert_no_truncation = kwargs['truncation'] is False
                return [ord(char) % 20 + 3 for char in text]
            def num_special_tokens_to_add(self, pair=False):
                return 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, model, output = root / 'parent', root / 'model', root / 'external'
            model.mkdir()
            external.write(model / 'config.json', dict(model_type='bert', max_position_embeddings=512))
            (model / 'pytorch_model.bin').write_bytes(b'fake official checkpoint for CPU init fixture')
            external.write(model / 'MODEL_ORIGIN.json', dict(repo_id='hfl/chinese-macbert-base',
                revision='a' * 40, files={'pytorch_model.bin': external.sha(model / 'pytorch_model.bin')}))
            hashes = {}
            for split in ('train', 'validation', 'test'):
                frame = pd.DataFrame(dict(base_id=[f'{split}_{i}' for i in range(4)],
                    group_id=[f'{split}_g{i}' for i in range(4)], source=['chinese_curated'] * 4,
                    prompt=['长' * 1100, '短', '文本', '安全审核'], category_id=[1, 2, 4, 5],
                    prompt_harm_label=['harmful'] * 4, type_label_mask=[True] * 4))
                external.save_frame(frame, parent / 'data' / f'{split}.parquet')
                hashes[split] = external.sha(parent / 'data' / f'{split}.parquet')
            external.write(parent / 'protocol.json', dict(data_sha256=hashes))
            parent_sha = external.sha(parent / 'protocol.json')
            external.write(parent / 'protocol_registration.json', dict(protocol_sha256=parent_sha))
            external.write(parent / 'selection_lock.json', dict(complete=True, protocol_sha256=parent_sha))
            external.write(parent / 'results.json', dict(complete=True, protocol_sha256=parent_sha,
                rootselection_lock_sha256=external.sha(parent / 'selection_lock.json')))
            tokenizer = Tokenizer()
            with patch.object(external, 'EXPECTED_ROWS', dict(train=4, validation=4, test=4)), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
                state = external.initialize(output, parent, model)
            self.assertEqual(state['total_jobs'], 6)
            self.assertFalse((output / 'data/test').exists())
            self.assertTrue(tokenizer.assert_no_truncation)
            protocol = external.verify(output)
            self.assertEqual(protocol['model_origin']['revision'], 'a' * 40)
            data = external.load_data(output, 'train')
            self.assertEqual(len(data['meta']), 4)
            self.assertEqual(data['manifest']['long_documents'], 1)
            self.assertEqual(data['manifest']['max_windows'], 3)
            self.assertFalse(data['manifest']['truncation'])
            self.assertEqual(data['docs'][0]['spans'][-1][1], 1100)
            with (output / 'data/train/windows.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'tokenization or metadata changed'):
                external.load_data(output, 'train')
            (model / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                external.verify(output)


if __name__ == '__main__':
    unittest.main()
