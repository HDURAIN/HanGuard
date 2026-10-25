"""CPU-only scientific and scheduling checks for fresh Qwen3Guard evaluation."""
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd

from scripts.hanguard import qwen3guard_clean_eval as guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'fresh_guard'
        self.output.mkdir()
        for name in ('claims', 'chunks', 'workers', 'source_snapshot'):
            (self.output / name).mkdir()
        guard.write(self.output / 'protocol.json', {'fixture': True})

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, labels=('unharmful', 'harmful', 'harmful', 'unharmful')):
        count = len(labels)
        metadata = pd.DataFrame(dict(row_index=range(count), base_id=[f'id_{i}' for i in range(count)],
            source=['a', 'a', 'b', 'b'][:count], label=[int(label == 'harmful') for label in labels],
            input_tokens=[301] * count))
        guard._parquet(metadata, self.output / 'metadata.parquet')
        plan = dict(total_rows=count, total_chunks=2,
                    chunks=[dict(id=0, batches=[list(range(2))], rows=2),
                            dict(id=1, batches=[list(range(2, count))], rows=count - 2)])
        guard.write(self.output / 'plan.json', plan)
        return metadata, plan

    def complete_chunk(self, metadata, chunk, outputs):
        indices = [index for batch in chunk['batches'] for index in batch]
        rows = []
        for index, text in zip(indices, outputs):
            parsed = guard.parse_output(text)
            rows.append(dict(metadata.iloc[index], raw_output=text,
                generated_tokens=10, stopped_on_eos=True, hit_generation_limit=False,
                **dict(parsed, categories=json.dumps(parsed['categories']))))
        _, path, receipt = guard._chunk_paths(self.output, chunk['id'])
        guard._parquet(pd.DataFrame(rows), path)
        guard.write(receipt, dict(protocol_sha256=guard.sha(self.output / 'protocol.json'),
            chunk_id=chunk['id'], rows=chunk['rows'], predictions_sha256=guard.sha(path)))

    def test_parser_requires_unique_safety_but_category_error_is_separate(self):
        text = 'Safety: Unsafe\nCategories: Violent, PII\n'
        result = guard.parse_output(text)
        self.assertEqual(result['safety_label'], 'unsafe')
        self.assertEqual(result['categories'], ['Violent', 'PII'])
        self.assertTrue(result['categories_valid'])
        for text in ('Safety: Safe\nSafety: Unsafe', 'Safety: Safe\nSafety: unknown',
                     'The answer is Safety: Safe', 'Safety: Safe maybe'):
            with self.subTest(text=text):
                self.assertEqual(guard.parse_output(text)['safety_label'], 'invalid')
        result = guard.parse_output('Safety: Controversial\nCategories: Unknown')
        self.assertEqual(result['safety_label'], 'controversial')
        self.assertFalse(result['categories_valid'])
        self.assertTrue(guard.parse_output('Safety: Safe\nCategories: None')['categories_valid'])
        self.assertFalse(guard.parse_output('Safety: Unsafe\nCategories: None, Violent')['categories_valid'])

    def test_length_plan_is_exhaustive_deterministic_and_within_budget(self):
        lengths = [513, 10, 4000, 15, 70] * 100
        plan = guard.chunk_plan(lengths)
        self.assertEqual(plan, guard.chunk_plan(lengths))
        indices = [i for chunk in plan['chunks'] for batch in chunk['batches'] for i in batch]
        self.assertEqual(sorted(indices), list(range(len(lengths))))
        self.assertEqual(len(set(indices)), len(indices))
        for chunk in plan['chunks']:
            self.assertLessEqual(chunk['rows'], 256)
            for batch in chunk['batches']:
                self.assertLessEqual(len(batch), 32)
                self.assertLessEqual(len(batch) * max(lengths[i] for i in batch), 8192)

    def test_invalid_safety_is_not_safe_or_dropped(self):
        frame = pd.DataFrame(dict(label=[0, 1, 0, 1],
                                  safety_label=['safe', 'unsafe', 'invalid', 'invalid']))
        metrics = guard.binary_metrics(frame, ['unsafe'])
        self.assertEqual(metrics['n'], 4)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['coverage'], .5)
        self.assertEqual(metrics['tn'], 1)
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['invalid_positive'], 1)
        self.assertEqual(metrics['invalid_negative'], 1)
        self.assertEqual(metrics['harmful_recall'], .5)
        self.assertAlmostEqual(metrics['macro_f1'], 2 / 3)

    def test_two_mappings_fixed_and_different_on_controversial(self):
        frame = pd.DataFrame(dict(label=[0, 1], safety_label=['safe', 'controversial']))
        self.assertEqual(guard.binary_metrics(frame, guard.MAPPINGS['unsafe_only'])['accuracy'], .5)
        self.assertEqual(guard.binary_metrics(frame, guard.MAPPINGS['unsafe_or_controversial'])['accuracy'], 1.)

    def test_dead_claim_reclaimed_live_claim_skipped_and_receipt_avoids_repeat(self):
        metadata, plan = self.fixture()
        current = guard.identity(os.getpid())
        stale = dict(current, start_time_ticks=current['start_time_ticks'] + 1)
        guard.write(self.output / 'claims/00000.json', dict(identity=stale))
        chunk = guard.claim_next(self.output, current)
        self.assertEqual(chunk['id'], 0)
        self.assertEqual(len(list((self.output / 'retired_claims').glob('*.json'))), 1)
        self.assertEqual(guard.claim_next(self.output, current)['id'], 1)
        self.assertIsNone(guard.claim_next(self.output, current))
        self.complete_chunk(metadata, plan['chunks'][0], ['Safety: Safe\nCategories: None', 'Safety: Unsafe\nCategories: Violent'])
        state = guard.progress(self.output)
        self.assertEqual(state['complete_chunks'], 1)
        self.assertEqual(state['claimed_chunks'], 1)
        self.assertEqual(state['completed_rows'], 2)
        self.assertEqual(state['pending_chunks'], 0)

    def test_unknown_process_identity_never_steals_claim(self):
        self.fixture()
        current = guard.identity(os.getpid())
        guard.write(self.output / 'claims/00000.json', dict(identity=current))
        guard.write(self.output / 'claims/00001.json', dict(identity=current))
        for failure in (PermissionError('denied'), ValueError('unknown command'), OSError('unknown state')):
            with self.subTest(failure=type(failure).__name__), patch.object(guard, 'identity', side_effect=failure):
                self.assertTrue(guard._alive(current))
                self.assertIsNone(guard.claim_next(self.output, current))
        self.assertFalse((self.output / 'retired_claims').exists())

    def test_confirmed_zombie_process_is_dead_for_claim_recovery(self):
        for state in ('Z', 'X', 'x'):
            stat = '123 (old worker) ' + state + ' ' + ' '.join(['0'] * 30)
            with self.subTest(state=state), patch.object(Path, 'read_text', return_value=stat):
                with self.assertRaises(ProcessLookupError):
                    guard.identity(123)
                self.assertFalse(guard._alive({'pid': 123}))

    def test_gpu_admission_requires_uuid_and_no_foreign_even_same_uid(self):
        card = dict(index=2, uuid='GPU-test', free_gib=40., pids=[])
        with patch.object(guard, '_gpu_inventory', return_value=[card]), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-test'):
            self.assertTrue(guard.admission(2)[0])
            card['pids'] = [os.getpid()]
            self.assertTrue(guard.admission(2)[0])
            card['pids'] = [os.getpid(), 123456]
            self.assertFalse(guard.admission(2)[0])
            card['pids'] = []
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'):
                self.assertFalse(guard.admission(2)[0])
        with patch.object(guard, '_gpu_inventory', side_effect=ValueError('unknown')):
            self.assertFalse(guard.admission(2)[0])

    def test_double_telemetry_union_and_unknown_xml_rejection(self):
        csv = '2, GPU-test, 40000\n'
        xml = '<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes><process_info><pid>44</pid></process_info></processes></gpu></nvidia_smi_log>'
        with patch.object(guard, '_run_smi', side_effect=[csv, 'GPU-test, 33\n', xml]):
            self.assertEqual(guard._gpu_inventory()[0]['pids'], [33, 44])
        for content in ('<processes>N/A</processes>', '<processes><process_info><pid>0</pid></process_info></processes>', ''):
            xml = f'<nvidia_smi_log><gpu><uuid>GPU-test</uuid>{content}</gpu></nvidia_smi_log>'
            with self.subTest(content=content), patch.object(guard, '_run_smi', side_effect=[csv, '', xml]):
                with self.assertRaises((ValueError, TypeError)):
                    guard._gpu_inventory()

    def system_display_fixture(self):
        command = b'/usr/lib/xorg/Xorg\0-fixture\0'
        expected = dict(guard.SYSTEM_DISPLAY, command_sha256=hashlib.sha256(command).hexdigest())
        fields = ['S'] + ['0'] * 30
        fields[19] = str(expected['start_time_ticks'])
        stat = '9539 (Xorg) ' + ' '.join(fields)
        process = '<process_info><pid>9539</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name></process_info>'
        xml = f'<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes>{process}</processes></gpu></nvidia_smi_log>'
        return expected, command, stat, xml, ET.fromstring(process)

    def test_exact_audited_graphics_service_is_ignored(self):
        expected, command, stat, xml, process = self.system_display_fixture()
        with patch.object(guard, 'SYSTEM_DISPLAY', expected), patch.object(Path, 'read_text', return_value=stat), patch.object(Path, 'read_bytes', return_value=command), patch.object(Path, 'stat', return_value=SimpleNamespace(st_uid=126)):
            self.assertTrue(guard._verified_system_display(process, 9539))
            with patch.object(guard, '_run_smi', side_effect=['2, GPU-test, 40000\n', '', xml]):
                self.assertEqual(guard._gpu_inventory()[0]['pids'], [])

    def test_display_spoof_or_unverifiable_identity_always_blocks(self):
        expected, command, stat, xml, process = self.system_display_fixture()
        for changed in (dict(expected, uid=1003), dict(expected, start_time_ticks=10028),
                        dict(expected, command_sha256='0' * 64), dict(expected, executable='/tmp/Xorg')):
            with self.subTest(changed=changed), patch.object(guard, 'SYSTEM_DISPLAY', changed), patch.object(Path, 'read_text', return_value=stat), patch.object(Path, 'read_bytes', return_value=command), patch.object(Path, 'stat', return_value=SimpleNamespace(st_uid=126)):
                self.assertFalse(guard._verified_system_display(process, 9539))
        with patch.object(Path, 'read_text', side_effect=PermissionError('unknown')):
            self.assertFalse(guard._verified_system_display(process, 9539))
        for kind in ('C', 'C+G', 'Unknown'):
            masked = ET.fromstring(ET.tostring(process))
            masked.find('type').text = kind
            self.assertFalse(guard._verified_system_display(masked, 9539))

    def test_compute_record_takes_precedence_over_audited_display(self):
        _, _, _, xml, _ = self.system_display_fixture()
        with patch.object(guard, '_verified_system_display', return_value=True) as verify_display:
            with patch.object(guard, '_run_smi', side_effect=['2, GPU-test, 40000\n', 'GPU-test, 9539\n', xml]):
                self.assertEqual(guard._gpu_inventory()[0]['pids'], [9539])
            verify_display.assert_not_called()

    def test_rejected_worker_returns_75_before_importing_torch(self):
        protocol = dict(settings=guard.SETTINGS)
        import builtins
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('torch imported before rejected GPU admission')
            return original_import(name, *args, **kwargs)
        with patch.object(guard, 'verify', return_value=protocol), patch.object(guard, 'admission', return_value=(False, {'reason': 'occupied'})), patch('builtins.__import__', side_effect=checked_import):
            self.assertEqual(guard.work(self.output, 2), 75)

    def test_oom_recursive_split_covers_every_row_without_reordering(self):
        class OOM(Exception):
            pass
        empty = []
        torch = SimpleNamespace(OutOfMemoryError=OOM, cuda=SimpleNamespace(empty_cache=lambda: empty.append(True)))
        def generate(model, tokenizer, token_ids, torch, max_new_tokens):
            if len(token_ids) > 1:
                raise OOM('synthetic')
            return [{'value': token_ids[0][0]}]
        events = []
        with patch.object(guard, '_generate_batch', side_effect=generate):
            rows = guard.generate_with_oom_retry(None, None, [[10], [20], [30]], [2, 0, 1], torch, events)
        self.assertEqual(rows, [(2, {'value': 30}), (0, {'value': 10}), (1, {'value': 20})])
        self.assertEqual(len(events), 2)
        self.assertEqual(len(empty), 2)

    def test_generation_slices_full_padded_input_width_and_tracks_eos(self):
        class Tensor:
            def __init__(self, rows):
                self.rows, self.shape = rows, (len(rows), len(rows[0]))
            def __getitem__(self, value):
                _, columns = value
                return Tensor([row[columns] for row in self.rows])
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.rows
        class Batch(dict):
            def __init__(self):
                super().__init__(input_ids=Tensor([[0, 11, 12], [21, 22, 23]]))
                self.input_ids = self['input_ids']
            def to(self, device): return self
        class Tokenizer:
            pad_token_id = 0
            def pad(self, *args, **kwargs): return Batch()
            def decode(self, tokens, **kwargs):
                self.decoded.append(tokens)
                return 'Safety: Safe\nCategories: None'
            decoded = []
        tokenizer = Tokenizer()
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99, 0]),
            generate=lambda **kwargs: Tensor([[0, 11, 12, 5, 99, 0], [21, 22, 23, 6, 7, 99]]))
        torch = SimpleNamespace(inference_mode=nullcontext)
        rows = guard._generate_batch(model, tokenizer, [[11, 12], [21, 22, 23]], torch, 128)
        self.assertEqual(tokenizer.decoded, [[5], [6, 7]])
        self.assertEqual([row['generated_tokens'] for row in rows], [1, 2])
        self.assertTrue(all(row['stopped_on_eos'] for row in rows))

    def test_summary_requires_all_chunks_then_archives_without_text(self):
        metadata, plan = self.fixture()
        protocol = dict(rows=4, data_sha256='dataset-sha', study=str(self.root / 'main'), comparison_note='zero-shot versus supervised')
        with patch.object(guard, 'verify', return_value=protocol):
            with self.assertRaises(RuntimeError):
                guard.summarize(self.output)
        for chunk in plan['chunks']:
            texts = ['Safety: Safe\nCategories: None', 'Safety: Unsafe\nCategories: Violent'] if chunk['id'] == 0 else ['Safety: Controversial\nCategories: PII', 'Safety: Safe\nCategories: None']
            self.complete_chunk(metadata, chunk, texts)
        main = dict(arms=['baseline'], seeds=[42, 43], per_seed={})
        for seed in main['seeds']:
            main['per_seed'][str(seed)] = dict(binary=dict(arms=dict(baseline=dict(full_original_test={
                regime: dict(accuracy=.9) for regime in ('validation_threshold', 'fixed_threshold_05')}))))
        guard.write(self.root / 'main/results.json', main)
        guard.write(self.output / 'protocol_registration.json', {'fixture': True})
        (self.output / 'source_snapshot/qwen3guard_clean_eval.py').write_text('source fixture')
        with patch.object(guard, 'verify', return_value=protocol):
            result = guard.summarize(self.output)
        self.assertTrue(result['complete'])
        self.assertEqual(result['mappings']['unsafe_only']['accuracy'], .75)
        self.assertEqual(result['mappings']['unsafe_or_controversial']['accuracy'], 1.)
        self.assertTrue(guard.is_complete(self.output))
        archive = self.output / 'archive'
        exported = pd.read_parquet(archive / 'predictions.parquet')
        self.assertNotIn('raw_output', exported)
        self.assertNotIn('prompt', exported)
        manifest = guard.read(archive / 'manifest.json')
        for name, digest in manifest['files'].items():
            self.assertEqual(guard.sha(archive / name), digest)
        self.assertEqual(result['five_class'], 'N/A')

    def clean_inputs(self):
        study, model = self.root / 'clean_study', self.root / 'model'
        (study / 'data').mkdir(parents=True)
        model.mkdir()
        dataset = study / 'data/test.parquet'
        frame = pd.DataFrame(dict(base_id=[f'id_{i:05d}' for i in range(8350)],
            source=['fixture'] * 8350, prompt=['fresh raw text'] * 8350,
            prompt_harm_label=['harmful' if i % 2 else 'unharmful' for i in range(8350)]))
        frame.to_parquet(dataset, index=False)
        guard.write(study / 'protocol.json', dict(inputs={str(dataset.resolve()): guard.sha(dataset)}))
        protocol_sha = guard.sha(study / 'protocol.json')
        guard.write(study / 'protocol_registration.json', dict(protocol_sha256=protocol_sha))
        guard.write(study / 'selection_lock.json', dict(complete=True, protocol_sha256=protocol_sha))
        guard.write(study / 'results.json', dict(complete=True, protocol_sha256=protocol_sha,
            rootselection_lock_sha256=guard.sha(study / 'selection_lock.json')))
        guard.write(model / 'config.json', dict(model_type='qwen3', architectures=['Qwen3ForCausalLM'], max_position_embeddings=32768))
        for name in ('generation_config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json'):
            guard.write(model / name, {'fixture': True})
        (model / 'merges.txt').write_text('fixture')
        (model / 'model-00001-of-00001.safetensors').write_bytes(b'fresh model bytes')
        guard.write(model / 'model.safetensors.index.json', dict(weight_map={'w': 'model-00001-of-00001.safetensors'}))
        return study, model

    def test_initialize_fresh_tokenization_parent_data_binding_and_tamper_detection(self):
        study, model = self.clean_inputs()
        output = self.root / 'initialized'
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return 'official ' + messages[0]['content']
            def __call__(self, prompts, **kwargs):
                self.assertions = kwargs
                return {'input_ids': [[1, 2, 3] for _ in prompts]}
        tokenizer = Tokenizer()
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer))
        with patch.dict('sys.modules', {'transformers': fake_transformers}):
            state = guard.initialize(output, study, model)
        self.assertEqual(state['total_rows'], 8350)
        self.assertEqual(state['completed_rows'], 0)
        self.assertEqual(tokenizer.padding_side, 'left')
        self.assertEqual(tokenizer.assertions, dict(add_special_tokens=False, truncation=False))
        self.assertEqual(calls[0], ([dict(role='user', content='fresh raw text')], dict(tokenize=False)))
        protocol = guard.verify(output)
        self.assertEqual(protocol['data_sha256'], guard.sha(study / 'data/test.parquet'))
        self.assertEqual(protocol['exclusions'], [])
        self.assertFalse(protocol['training'])
        self.assertFalse(protocol['historical_predictions_used'])
        with patch.dict('sys.modules', {'transformers': fake_transformers}):
            with self.assertRaises(FileExistsError):
                guard.initialize(output, study, model)
        # A same-size modification of prepared tokens must also be rejected.
        tokens = output / 'input_ids.jsonl'
        tokens.write_text(tokens.read_text().replace('[1,2,3]', '[1,2,4]', 1))
        with self.assertRaisesRegex(ValueError, 'prepared'):
            guard.verify(output)

    def test_parent_test_modified_after_main_results_is_rejected(self):
        study, model = self.clean_inputs()
        dataset = study / 'data/test.parquet'
        frame = pd.read_parquet(dataset)
        frame.loc[0, 'prompt'] = 'changed after the main study'
        frame.to_parquet(dataset, index=False)
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: self.fail('must reject before tokenization')))
        with patch.dict('sys.modules', {'transformers': fake_transformers}):
            with self.assertRaisesRegex(ValueError, 'registered completed clean study'):
                guard.initialize(self.root / 'bad_parent', study, model)


if __name__ == '__main__':
    unittest.main()
