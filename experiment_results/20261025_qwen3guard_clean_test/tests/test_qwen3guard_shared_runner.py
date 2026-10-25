"""CPU-only checks for the explicitly authorized Guard admission amendment."""
from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts.hanguard import qwen3guard_shared_runner as shared


def card(free=24., pids=(1234, 5678)):
    return dict(index=4, uuid='GPU-shared-fixture', free_gib=free, pids=list(pids))


class AdmissionTests(unittest.TestCase):
    def test_foreign_processes_allowed_only_with_registered_headroom(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-shared-fixture'):
            for free, accepted in [(29., True), (25., True), (22., True), (21.99, False), (0., False)]:
                with self.subTest(free=free), patch.object(shared.evaluator, '_gpu_inventory', return_value=[card(free)]):
                    allowed, observed = shared.admission(4)
                    self.assertEqual(allowed, accepted)
                    if allowed:
                        self.assertEqual(observed['pids'], [1234, 5678])
                        self.assertEqual(observed['admission_policy'], shared.SCHEMA)
            with patch.object(shared.evaluator, '_gpu_inventory', return_value=[card(24)]):
                self.assertFalse(shared.admission(4, minimum_free_gib=26)[0])

    def test_wrong_uuid_missing_gpu_and_unknown_inventory_rejected(self):
        with patch.object(shared.evaluator, '_gpu_inventory', return_value=[card()]), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='4'):
            self.assertFalse(shared.admission(4)[0])
        with patch.object(shared.evaluator, '_gpu_inventory', return_value=[]):
            self.assertFalse(shared.admission(4)[0])
        for error in (ValueError('unknown inventory'), PermissionError('denied')):
            with self.subTest(error=error), patch.object(shared.evaluator, '_gpu_inventory', side_effect=error):
                self.assertFalse(shared.admission(4)[0])

    def test_rejected_work_returns_before_importing_torch(self):
        original = builtins.__import__
        def reject_torch(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('Rejected admission must not import torch')
            return original(name, *args, **kwargs)
        with patch.object(shared, 'verify_amendment'), patch.object(shared.evaluator, 'verify'), \
             patch.object(shared, 'admission', return_value=(False, {'reason': 'busy'})), \
             patch.object(shared.evaluator, 'work') as work, \
             patch('builtins.__import__', side_effect=reject_torch):
            self.assertEqual(shared.work(Path('/tmp/shared-test-fixture'), 4), 75)
        work.assert_not_called()


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'fresh_guard'
        self.output.mkdir()
        self.original = {}
        for name in ['protocol.json', 'queue_registration.json', 'plan.json']:
            shared.queue.write(self.output / name, {'name': name, 'old': True})
            self.original[name] = (self.output / name).read_bytes()
        self.paths = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temp.cleanup)
        for module, label in [(shared, 'wrapper'), (shared.evaluator, 'evaluator'), (shared.queue, 'queue')]:
            path = self.root / f'{label}.py'
            path.write_text(f'{label.upper()} = True\n')
            self.paths[label] = path
            self.stack.enter_context(patch.object(module, '__file__', str(path)))
        self.args = argparse.Namespace(gpus=list(range(8)), poll_seconds=1., minimum_free_gib=22.)

    def test_registration_binds_old_inputs_sources_and_preserves_old_registration(self):
        record = shared.registration(self.output, self.args)
        self.assertEqual(record, shared.registration(self.output, self.args))
        self.assertEqual(record['allocator_cap_gib'], 14.)
        self.assertEqual(record['admission_headroom_gib'], 8.)
        self.assertEqual(record['minimum_free_gib'], 22.)
        self.assertEqual(record['protocol_sha256'], shared.queue.sha(self.output / 'protocol.json'))
        self.assertEqual(record['plan_sha256'], shared.queue.sha(self.output / 'plan.json'))
        self.assertEqual(record['original_queue_registration_sha256'], shared.queue.sha(self.output / 'queue_registration.json'))
        self.assertEqual(set(record['sources']), {str(x.resolve()) for x in self.paths.values()})
        for filename, content in self.original.items():
            self.assertEqual((self.output / filename).read_bytes(), content)
        for filename, digest in record['sources'].items():
            self.assertEqual(shared.queue.sha(self.output / 'shared_source_snapshot' / Path(filename).name), digest)
        self.assertEqual(shared.verify_amendment(self.output), record)

    def test_changed_old_protocol_plan_or_queue_registration_is_rejected(self):
        shared.registration(self.output, self.args)
        for name in self.original:
            with self.subTest(name=name):
                (self.output / name).write_text('changed input')
                with self.assertRaises(ValueError):
                    shared.verify_amendment(self.output)
                (self.output / name).write_bytes(self.original[name])

    def test_source_change_and_changed_amendment_settings_are_rejected(self):
        shared.registration(self.output, self.args)
        original = self.paths['wrapper'].read_bytes()
        self.paths['wrapper'].write_text('CHANGED = True\n')
        with self.assertRaises(ValueError):
            shared.verify_amendment(self.output)
        with self.assertRaises(ValueError):
            shared.registration(self.output, self.args)
        self.paths['wrapper'].write_bytes(original)
        self.args.poll_seconds = 2.
        with self.assertRaises(ValueError):
            shared.registration(self.output, self.args)
        record = shared.queue.read(shared.amendment_path(self.output))
        record['allocator_cap_gib'] = 16.
        shared.queue.write(shared.amendment_path(self.output), record)
        with self.assertRaises(ValueError):
            shared.verify_amendment(self.output)


class WorkerTests(unittest.TestCase):
    def fixture(self, total_gib=48.):
        cuda = types.SimpleNamespace(
            get_device_properties=MagicMock(return_value=types.SimpleNamespace(total_memory=int(total_gib * 1024**3))),
            set_per_process_memory_fraction=MagicMock(),
            max_memory_allocated=MagicMock(return_value=10 * 1024**3),
            max_memory_reserved=MagicMock(return_value=12 * 1024**3))
        torch = types.ModuleType('torch')
        torch.cuda = cuda
        return torch

    def test_allocator_cap_precedes_original_work_and_telemetry_keeps_foreign_pids(self):
        torch = self.fixture()
        writes = []
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'torch': torch}))
            stack.enter_context(patch.object(shared, 'verify_amendment'))
            stack.enter_context(patch.object(shared.evaluator, 'verify'))
            stack.enter_context(patch.object(shared, 'admission', return_value=(True, card())))
            stack.enter_context(patch.object(shared.evaluator, 'admission'))
            stack.enter_context(patch.object(shared.evaluator, 'identity', return_value=dict(pid=123, start_time_ticks=456)))
            stack.enter_context(patch.object(shared.queue, 'sha', return_value='amendment-sha'))
            stack.enter_context(patch.object(shared.evaluator, 'write', side_effect=lambda path, data: writes.append((path, dict(data)))))
            def work(output, gpu):
                torch.cuda.set_per_process_memory_fraction.assert_called_once_with(14 / 48, device=0)
                self.assertIs(shared.evaluator.admission, shared.admission)
                return 0
            stack.enter_context(patch.object(shared.evaluator, 'work', side_effect=work))
            self.assertEqual(shared.work(Path('/tmp/shared-test-fixture'), 4), 0)
        self.assertEqual(writes[-1][1]['allocator_cap_gib'], 14.)
        self.assertEqual(writes[-1][1]['entry_inventory']['pids'], [1234, 5678])
        self.assertEqual(writes[-1][1]['exit_code'], 0)
        self.assertEqual(writes[-1][1]['peak_allocated_gib'], 10.)
        self.assertEqual(writes[-1][1]['peak_reserved_gib'], 12.)

    def test_original_work_failure_keeps_error_and_peak_telemetry(self):
        torch = self.fixture()
        writes = []
        with patch.dict(sys.modules, {'torch': torch}), patch.object(shared, 'verify_amendment'), \
             patch.object(shared.evaluator, 'verify'), patch.object(shared, 'admission', return_value=(True, card())), \
             patch.object(shared.evaluator, 'admission'), patch.object(shared.evaluator, 'identity', return_value=dict(pid=123, start_time_ticks=456)), \
             patch.object(shared.queue, 'sha', return_value='amendment-sha'), \
             patch.object(shared.evaluator, 'write', side_effect=lambda path, data: writes.append(dict(data))), \
             patch.object(shared.evaluator, 'work', side_effect=RuntimeError('synthetic worker failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
                shared.work(Path('/tmp/shared-test-fixture'), 4)
        self.assertIn('synthetic worker failure', writes[-1]['error'])
        self.assertEqual(writes[-1]['peak_reserved_gib'], 12.)

    def test_impossibly_small_device_rejected_before_original_work(self):
        torch = self.fixture(total_gib=10.)
        with patch.dict(sys.modules, {'torch': torch}), patch.object(shared, 'verify_amendment'), \
             patch.object(shared.evaluator, 'verify'), patch.object(shared, 'admission', return_value=(True, card())), \
             patch.object(shared.evaluator, 'work') as work:
            with self.assertRaisesRegex(ValueError, 'smaller'):
                shared.work(Path('/tmp/shared-test-fixture'), 4)
        torch.cuda.set_per_process_memory_fraction.assert_not_called()
        work.assert_not_called()


class StateTests(unittest.TestCase):
    def test_states_and_events_explicitly_describe_shared_capacity(self):
        event = dict(event='worker_launch', gpu=4, empty_inventory=card())
        with patch.object(shared, '_ORIGINAL_STATE') as state, patch.object(shared.queue, 'sha', return_value='amendment-sha'):
            shared.state(Path('/tmp/shared-test-fixture'), 'waiting_for_empty_gpu', recent_events=[event])
        args, kwargs = state.call_args
        self.assertEqual(args[1], 'waiting_for_shared_capacity')
        self.assertTrue(kwargs['shared_gpus_allowed'])
        self.assertEqual(kwargs['admission_amendment_sha256'], 'amendment-sha')
        changed = kwargs['recent_events'][0]
        self.assertEqual(changed['event'], 'shared_worker_launch')
        self.assertIn('admission_inventory', changed)
        self.assertNotIn('empty_inventory', changed)
        self.assertEqual(event['event'], 'worker_launch')


if __name__ == '__main__':
    unittest.main()
