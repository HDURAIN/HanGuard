"""CPU checks for numerical audits and genuinely online efficiency calls."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
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

from scripts.hanguard import paper_error_efficiency as audit


class NumericalTests(unittest.TestCase):
    def test_binary_and_multiclass_metrics_match_confusion_counts(self):
        m = audit.metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(m['accuracy'], .75)
        self.assertAlmostEqual(m['harmful_f1'], .8)
        self.assertAlmostEqual(m['macro_f1'], ((2 / 3) + .8) / 2)
        self.assertEqual(m['fpr'], .5)
        self.assertEqual(m['fnr'], 0)
        with self.assertRaises(ValueError):
            audit.metrics([1], [7], [1, 2])

    def test_paired_rescue_harm_and_cluster_interval(self):
        m = audit.paired([0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0], ['a', 'a', 'b', 'c'], draws=100)
        self.assertEqual((m['rescued'], m['harmed'], m['net_correct']), (2, 1, 1))
        self.assertEqual(m['delta_accuracy'], .25)
        self.assertEqual(m['bootstrap']['groups'], 3)
        self.assertEqual(m, audit.paired([0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0], ['a', 'a', 'b', 'c'], draws=100))

    def test_benchmark_selection_is_label_free_unique_and_includes_longest(self):
        frame = pd.DataFrame(dict(base_id=[f'id_{i}' for i in range(100)], prompt_tokens=np.arange(1, 101) * 20))
        selected = audit.select_benchmark_rows(frame)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({x['base_id'] for x in selected}), 32)
        self.assertEqual([x['raw_tokens'] for x in selected if x['length_target'] == 'longest'],
                         sorted([x['raw_tokens'] for x in selected if x['length_target'] == 'longest'], reverse=True))
        self.assertEqual(selected, audit.select_benchmark_rows(frame))

    def test_full_cpu_analysis_recomputes_all_scopes_and_emits_no_text(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, output = Path(temp) / 'parent', Path(temp) / 'out'
            output.mkdir()
            (parent / 'data').mkdir(parents=True)
            native_safe, native_harm = 1524, 2180
            n = 8350
            ids = [f'id_{i:05d}' for i in range(n)]
            category = np.zeros(n, dtype=int)
            category[native_safe:native_safe + native_harm] = np.arange(native_harm) % 5 + 1
            category[native_safe + native_harm:] = -1
            harmful = category > 0
            truth = pd.DataFrame(dict(base_id=ids, group_id=[f'g_{i // 2}' for i in range(n)],
                source=['chinese_curated'] * (native_safe + native_harm) + ['wildguard_zh'] * (n - native_safe - native_harm),
                category_id=category, prompt_harm_label=np.where(harmful, 'harmful', 'unharmful'),
                type_label_mask=harmful, prompt_tokens=np.arange(n) % 300 + 1))
            truth.to_parquet(parent / 'data/test.parquet', index=False)
            for split in ('train', 'test'):
                (parent / f'features/{split}').mkdir(parents=True)
                truth[['base_id', 'prompt_tokens']].to_parquet(parent / f'features/{split}/metadata.parquet', index=False)
            fake_binary = truth[['base_id', 'group_id', 'source', 'category_id', 'prompt_tokens']].copy()
            fake_category = fake_binary.copy()
            for arm in audit.ARMS:
                fake_binary[f'{arm}_p1'] = np.where(harmful, .9, .1)
                fake_category[f'{arm}_argmax'] = np.maximum(category - 1, 0)
            bm = audit.metrics(harmful.astype(int), harmful.astype(int), [0, 1])
            cm = audit.metrics(category[harmful], category[harmful], range(1, 6))
            sm = audit.metrics(category[:native_safe + native_harm], category[:native_safe + native_harm], range(6))
            root = {'per_seed': {}}
            for seed in audit.SEEDS:
                root['per_seed'][str(seed)] = dict(binary=dict(arms={a: dict(full_original_test=dict(
                    validation_threshold=dict(accuracy=bm['accuracy'], f1=bm['harmful_f1']))) for a in audit.ARMS}),
                    category=dict(arms={a: dict(native_known_harmful=cm) for a in audit.ARMS}),
                    systems=dict(corresponding_both_stages=dict(arms={a: dict(metrics=sm) for a in audit.MAIN})))
            audit.write(parent / 'results.json', root)
            audit.write(output / 'protocol.json', {'fixture': True})
            protocol = dict(parent=str(parent), draws=10, scopes=dict(binary=8350, category=2180, corresponding_system=3704))
            selection = dict(arms={a: dict(threshold=.5, parameters=128) for a in audit.ARMS})
            def aligned(parent, task, seed, frame):
                return (fake_binary if task == 'binary' else fake_category), selection
            with patch.object(audit, 'initialize', return_value=protocol), \
                 patch.object(audit, '_aligned_predictions', side_effect=aligned), \
                 patch.object(audit, 'GUARD', Path(temp) / 'no_guard'):
                result = audit.analyze(output, parent)
            self.assertTrue(result['parent_metrics_recomputed_and_matched'])
            self.assertEqual(len(result['per_seed']), 3 * 9 * 3)
            self.assertTrue(all(x['accuracy'] == 1 for x in result['per_seed']))
            self.assertTrue(all(x['delta_accuracy'] == 0 for x in result['paired_per_seed']))
            self.assertEqual(len(pd.read_csv(output / 'parameters.csv')), 54)
            for file in output.glob('*.csv'):
                self.assertNotIn('prompt', pd.read_csv(file).columns)


class AdmissionTests(unittest.TestCase):
    def test_efficiency_requires_exact_exclusive_scheduler_record(self):
        from scripts.hanguard.qwen3guard_clean_queue import identity
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            audit.write(output / 'protocol.json', {'fixture': True})
            owner = identity(os.getpid())
            record = dict(owner, followup_protocol_sha256=audit.sha(output / 'protocol.json'),
                          physical_gpu=2, gpu_uuid='GPU-fixture', exclusive=True)
            audit.write(output / 'admission.json', record)
            env = dict(HANGUARD_FOLLOWUP_ADMITTED=record['followup_protocol_sha256'],
                       HANGUARD_FOLLOWUP_ADMISSION_RECORD=str(output / 'admission.json'),
                       CUDA_VISIBLE_DEVICES='GPU-fixture')
            with patch.dict(os.environ, env), patch.object(audit, '_exclusive_gpu', return_value={'index': 2}):
                self.assertEqual(audit._admitted(output, 2), {'index': 2})
                record['exclusive'] = False
                audit.write(output / 'admission.json', record)
                with self.assertRaises(RuntimeError):
                    audit._admitted(output, 2)

    def test_any_foreign_process_or_unknown_gpu_invalidates_timing(self):
        from scripts.hanguard import qwen3guard_clean_queue as queue
        card = dict(index=2, uuid='GPU-fixture', pids=[os.getpid()], free_gib=40)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
             patch.object(queue, 'gpu_inventory', return_value=[card]):
            self.assertEqual(audit._exclusive_gpu(2), card)
            card['pids'].append(98765)
            with self.assertRaises(audit.BenchmarkContended):
                audit._exclusive_gpu(2)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
        self.calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        hidden = input_ids.float()[..., None].repeat(1, 1, 4)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden + index + 1)
        return hidden


class OnlineTests(unittest.TestCase):
    def test_capture_calls_real_backbone_on_each_invocation_and_raw_tokens_match(self):
        backbone = TinyBackbone()
        capture = audit.OnlineCapture(backbone, (8, 16, 24, 32), True, torch, nullcontext)
        batch = dict(input_ids=torch.tensor([[1, 2], [3, 0]]), attention_mask=torch.tensor([[1, 1], [1, 0]]))
        try:
            first, tokens, mask = capture(batch)
            second, _, _ = capture(batch)
            self.assertEqual(backbone.calls, 2)
            self.assertEqual(capture.forward_calls, 2)
            self.assertEqual(tuple(first.shape), (2, 32, 4))
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first[0, 31], tokens[0, 1]))
            self.assertTrue(torch.equal(first[1, 31], tokens[1, 0]))
            self.assertEqual(int(tokens[1, 1].abs().sum()), 0)
        finally:
            capture.close()
        self.assertTrue(all(not layer._forward_hooks for layer in backbone.layers))

    def test_online_system_tokenizes_and_forwards_once_for_both_heads(self):
        class Batch(dict):
            def to(self, device): return self
        class Tokenizer:
            calls = 0
            def __call__(self, texts, **kwargs):
                self.calls += 1
                return Batch(input_ids=torch.tensor([[1, 2], [3, 0]]), attention_mask=torch.tensor([[1, 1], [1, 0]]))
        def binary(last, tokens, mask):
            return torch.tensor([[0., 2.], [2., 0.]])
        def category(last, tokens, mask):
            return torch.tensor([[0., 1., 2., 3., 4.], [4., 3., 2., 1., 0.]])
        backbone, tokenizer = TinyBackbone(), Tokenizer()
        capture = audit.OnlineCapture(backbone, (32,), True, torch, nullcontext)
        runner = audit.OnlineReadout(tokenizer, capture, {'binary': {'category_only': binary},
                                    'category': {'category_only': category}}, 'category_only', 'system', .5, torch)
        try:
            first, second = runner(['a', 'b']), runner(['a', 'b'])
        finally:
            capture.close()
        self.assertEqual((tokenizer.calls, backbone.calls), (2, 2))
        self.assertEqual(first['rows'], 2)
        self.assertEqual(second['rows'], 2)

    def test_measurement_calls_every_warmup_and_measurement_without_cache(self):
        calls = []
        function = lambda batch: calls.append(list(batch)) or dict(rows=len(batch), tokenization_seconds=.001)
        fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 1024,
            max_memory_reserved=lambda: 2048))
        monitor = SimpleNamespace(check=lambda: None)
        measured = audit.measure_callable(function, [['a'], ['b']], fake, monitor, warmup=3, repeats=10)
        self.assertEqual(len(calls), 26)
        self.assertEqual(measured['invocations'], 20)
        self.assertEqual(measured['examples'], 20)
        self.assertGreater(measured['examples_per_second'], 0)

    def test_complete_benchmark_fake_backbones_produce_registered_results_and_receipt(self):
        from scripts.hanguard import clean_ablation_runtime as runtime
        from scripts.hanguard import frozen_base_readout_study as base
        class Batch(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self, device): return self
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, texts, **kwargs):
                return Batch(input_ids=torch.ones((len(texts), 2), dtype=torch.long),
                             attention_mask=torch.ones((len(texts), 2), dtype=torch.long))
            def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
            def batch_decode(self, tokens, **kwargs): return ['Safety: Safe\nCategories: None'] * len(tokens)
        class Guard(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(max_position_embeddings=32768)
                self.generation_config = SimpleNamespace(pad_token_id=0)
            def generate(self, input_ids, **kwargs):
                return torch.cat((input_ids, torch.ones((len(input_ids), 2), dtype=torch.long)), dim=1)
        class Monitor:
            observations = []
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def check(self): pass
        def bh(last, tokens, mask): return torch.ones((len(last), 2))
        def ch(last, tokens, mask): return torch.ones((len(last), 5))
        heads = {task: {arm: bh if task == 'binary' else ch for arm in audit.MAIN}
                 for task in ('binary', 'category')}
        selections = {task: dict(arms={arm: dict(parameters=128, threshold=.5) for arm in audit.MAIN})
                      for task in ('binary', 'category')}
        backbone = TinyBackbone()
        foundation = torch.nn.Linear(1, 1)
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as temp:
            output, parent = Path(temp) / 'out', Path(temp) / 'parent'
            output.mkdir()
            (parent / 'data').mkdir(parents=True)
            audit.write(output / 'protocol.json', {'fixture': True})
            audit.write(parent / 'protocol.json', dict(hidden_size=4, model_type='qwen3_5'))
            pd.DataFrame(dict(base_id=['a', 'b', 'c', 'd'], prompt=['alpha', 'beta', 'gamma', 'delta'])).to_parquet(parent / 'data/validation.parquet')
            protocol = dict(parent=str(parent), model_inputs={}, guard_model_inputs={}, guard_model='fake-guard',
                            benchmark_sample=[dict(base_id=x, raw_tokens=2) for x in ['a', 'b', 'c', 'd']])
            settings = dict(audit.BENCHMARK, warmup_passes=0, measured_passes=1)
            with ExitStack() as stack:
                stack.enter_context(patch.object(audit, 'verify', return_value=protocol))
                stack.enter_context(patch.object(audit, '_admitted', return_value={'index': 2}))
                stack.enter_context(patch.object(audit, '_exclusive_gpu', return_value={'index': 2}))
                stack.enter_context(patch.object(audit, 'IsolationMonitor', return_value=Monitor()))
                stack.enter_context(patch.object(audit, '_load_heads', return_value=(heads, selections)))
                stack.enter_context(patch.object(audit, 'BENCHMARK', settings))
                stack.enter_context(patch.object(base, 'load_frozen_backbone', return_value=(foundation, backbone, tokenizer)))
                stack.enter_context(patch.object(runtime, 'DeterministicKernelPath', return_value=SimpleNamespace(snapshot=lambda: {'fake': True})))
                stack.enter_context(patch.object(runtime, 'math_sdpa', nullcontext))
                stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=Guard()))
                stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()))
                for method in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
                    stack.enter_context(patch.object(torch.cuda, method, return_value=None))
                for method in ('max_memory_allocated', 'max_memory_reserved'):
                    stack.enter_context(patch.object(torch.cuda, method, return_value=1024))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='fake-exclusive-gpu'))
                self.assertEqual(audit.benchmark(output, parent, 2), 0)
            result = audit.read(output / 'efficiency_results.json')
            receipt = audit.read(output / 'efficiency_complete.json')
            self.assertEqual(len(result['cases']), 26)
            self.assertEqual(result['gpu_name'], 'fake-exclusive-gpu')
            self.assertEqual(backbone.calls, 12 * 5)
            self.assertEqual(receipt['results_sha256'], audit.sha(output / 'efficiency_results.json'))
            self.assertEqual(receipt['csv_sha256'], audit.sha(output / 'efficiency.csv'))

    def test_monitor_invalidates_if_foreign_activity_appears(self):
        with patch.object(audit, '_exclusive_gpu', side_effect=[{'index': 2}, audit.BenchmarkContended('other user')]):
            with audit.IsolationMonitor(2, .001) as monitor:
                self.assertTrue(monitor.invalid.wait(.5))
                with self.assertRaises(audit.BenchmarkContended):
                    monitor.check()


if __name__ == '__main__':
    unittest.main()
