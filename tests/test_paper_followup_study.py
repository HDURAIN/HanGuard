"""CPU-only admission, source binding, and dispatch checks for paper followups."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts.hanguard import paper_followup_study as study


def card(free=48, pids=(), uuid='GPU-fixture'):
    return dict(index=2, uuid=uuid, free_gib=free, pids=list(pids), utilization=100)


class DispatchTests(unittest.TestCase):
    def test_shared_training_uses_memory_budget_not_foreign_owner_or_utilization(self):
        self.assertTrue(study.eligible(card(8, [999]), {'kind': 'pooling'}))
        self.assertFalse(study.eligible(card(7.99, []), {'kind': 'pooling'}))
        self.assertTrue(study.eligible(card(14, [999]), {'kind': 'external'}))
        self.assertFalse(study.eligible(card(13.99, []), {'kind': 'external'}))

    def test_efficiency_requires_completely_empty_compute_inventory(self):
        self.assertTrue(study.eligible(card(40, []), {'kind': 'analysis'}))
        self.assertFalse(study.eligible(card(48, [os.getpid()]), {'kind': 'analysis'}))
        self.assertFalse(study.eligible(card(39.99, []), {'kind': 'analysis'}))

    def test_fit_and_test_for_same_model_group_share_exclusion_key(self):
        first = dict(kind='pooling', task='category', seed=42, phase='fit')
        self.assertEqual(study.job_key(first), study.job_key(dict(first, phase='evaluate')))
        self.assertNotEqual(study.job_key(first), study.job_key(dict(first, seed=43)))

    def test_jobs_prioritize_empty_card_measurements_and_native_category_heads(self):
        status = dict(analysis=dict(cpu_analysis_complete=True, benchmark_complete=False),
                      external=dict(pending_jobs=[dict(seed=42, phase='fit')]),
                      pooling=dict(pending_jobs=[dict(task='binary', seed=42),
                                                 dict(task='category', seed=44),
                                                 dict(task='category', seed=42)]))
        jobs = study.jobs(status)
        self.assertEqual([job['kind'] for job in jobs], ['analysis', 'external', 'pooling', 'pooling', 'pooling'])
        self.assertEqual([(j['task'], j['seed']) for j in jobs[2:]],
                         [('category', 42), ('category', 44), ('binary', 42)])
        status['analysis']['cpu_analysis_complete'] = False
        self.assertFalse(any(job['kind'] == 'analysis' for job in study.jobs(status)))

    def test_numeric_policy_disables_tf32_and_enforces_pooling_benchmark_determinism(self):
        for kind in study.MODULES:
            cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
            cudnn = SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)
            torch = SimpleNamespace(backends=SimpleNamespace(cuda=cuda, cudnn=cudnn),
                                    use_deterministic_algorithms=MagicMock())
            with self.subTest(kind=kind):
                study.configure_numeric(torch, kind)
                self.assertFalse(cuda.matmul.allow_tf32)
                self.assertFalse(cudnn.allow_tf32)
                self.assertFalse(cudnn.benchmark)
                if kind in ('pooling', 'analysis'):
                    self.assertTrue(cudnn.deterministic)
                    torch.use_deterministic_algorithms.assert_called_once_with(True)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output, self.parent = self.root / 'output', self.root / 'parent'
        self.output.mkdir(); self.parent.mkdir()
        study.common.write(self.parent / 'protocol.json', {'parent': True})
        study.common.write(self.parent / 'results.json', {'complete': True})
        for kind in study.MODULES:
            (self.output / kind).mkdir()
            study.common.write(self.output / kind / 'protocol.json', {'kind': kind})
        self.args = argparse.Namespace(output=self.output, parent=self.parent, model=self.root / 'model',
                                       kind='pooling', task='category', seed=42, gpu=2)
        numeric = patch.object(study, 'configure_numeric')
        self.numeric = numeric.start(); self.addCleanup(numeric.stop)

    def test_root_registration_binds_children_parent_and_actual_sources(self):
        scripts = {}
        for kind in ['root', 'common', *study.MODULES]:
            scripts[kind] = self.root / (kind + '.py')
            scripts[kind].write_text(f'NAME={kind!r}\n')
        with patch.object(study, '__file__', str(scripts['root'])), \
             patch.object(study.common, '__file__', str(scripts['common'])), \
             patch.object(study, 'script', side_effect=lambda kind: scripts[kind]), \
             patch.object(study, 'module', return_value=SimpleNamespace(progress=lambda _: {'cpu_analysis_complete': True})), \
             patch.object(study, 'refresh_archive'):
            protocol = study.initialize(self.args)
            self.assertEqual(study.verify(self.output), protocol)
            self.assertEqual(study.initialize(self.args), protocol)
            scripts['pooling'].write_text('CHANGED=True\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                study.verify(self.output)
            scripts['pooling'].write_text("NAME='pooling'\n")
            study.common.write(self.output / 'external/protocol.json', {'kind': 'changed'})
            with self.assertRaisesRegex(ValueError, 'child protocol changed'):
                study.verify(self.output)

    def test_missing_cpu_analysis_prevents_root_registration(self):
        with patch.object(study, 'module', return_value=SimpleNamespace(progress=lambda _: {'cpu_analysis_complete': False})):
            with self.assertRaisesRegex(ValueError, 'CPU error analysis'):
                study.initialize(self.args)
        self.assertFalse((self.output / 'protocol.json').exists())

    def test_rejected_admission_never_imports_torch_or_dispatches_model(self):
        import builtins
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('CUDA library reached before admission')
            return original(name, *args, **kwargs)
        for observed in (card(7), card(48, uuid='GPU-other')):
            with self.subTest(card=observed), patch.object(study, 'verify'), \
                 patch.object(study.common, 'gpu_inventory', return_value=[observed]), \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
                 patch('builtins.__import__', side_effect=guarded_import), \
                 patch.object(study, 'module') as target:
                self.assertEqual(study.worker(self.args), 75)
                target.assert_not_called()

    def test_allocator_and_identity_record_precede_classification_head(self):
        study.common.write(self.output / 'protocol.json', {'root': True})
        calls = []
        cuda = SimpleNamespace(
            get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3),
            set_per_process_memory_fraction=lambda fraction, device: calls.append(('cap', fraction, device)),
            max_memory_allocated=lambda _: 1024**3,
            max_memory_reserved=lambda _: 2 * 1024**3)
        def fit(output, task, seed, device):
            self.assertEqual(calls[0], ('cap', 4 / 48, 0))
            record = json.loads(Path(os.environ['HANGUARD_FOLLOWUP_ADMISSION_RECORD']).read_text())
            self.assertEqual(record['pid'], os.getpid())
            self.assertEqual(record['gpu_uuid'], 'GPU-fixture')
            self.assertFalse(record['exclusive'])
            self.assertEqual(record['followup_protocol_sha256'], os.environ['HANGUARD_FOLLOWUP_ADMITTED'])
            self.assertEqual((task, seed, device), ('category', 42, 'cuda'))
            calls.append(('fit',))
        with patch.object(study, 'verify'), \
             patch.object(study.common, 'gpu_inventory', return_value=[card(12, [999])]), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
             patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
             patch.object(study, 'module', return_value=SimpleNamespace(fit=fit)):
            self.assertEqual(study.worker(self.args), 0)
        record = json.loads(next((self.output / 'admissions').glob('*.json')).read_text())
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['peak_allocated_gib'], 1)
        self.assertEqual(record['peak_reserved_gib'], 2)

    def test_exclusive_foreign_arrival_after_cuda_context_forces_retry(self):
        self.args.kind = 'analysis'
        calls = []
        cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3),
            set_per_process_memory_fraction=lambda *a, **k: calls.append('cap'))
        with patch.object(study, 'verify'), \
             patch.object(study.common, 'gpu_inventory', side_effect=[[card(48)], [card(47, [999])]]), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
             patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
             patch.object(study, 'module') as target:
            self.assertEqual(study.worker(self.args), 75)
            target.assert_not_called()
        self.assertEqual(calls, ['cap'])

    def test_complete_study_finishes_without_gpu_inventory_or_child_launch(self):
        statuses = dict(analysis=dict(cpu_analysis_complete=True, benchmark_complete=True),
                        pooling=dict(pending_jobs=[]), external=dict(pending_jobs=[]))
        with patch.object(study, 'verify', return_value={'poll_seconds': .001}), \
             patch.object(study, 'progress', return_value=statuses), \
             patch.object(study, 'refresh_archive'), \
             patch.object(study, 'module', return_value=SimpleNamespace(is_complete=lambda _: True)), \
             patch.object(study.common, 'gpu_inventory') as inventory, \
             patch.object(study.subprocess, 'Popen') as popen:
            study.run(self.args)
        inventory.assert_not_called(); popen.assert_not_called()
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'complete')

    def test_worker_record_is_accepted_by_actual_pooling_admission_validator(self):
        from scripts.hanguard import paper_pooling_controls as pooling
        study.common.write(self.output / 'protocol.json', {'root': True})
        cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3),
            set_per_process_memory_fraction=lambda *a, **k: None,
            max_memory_allocated=lambda _: 0, max_memory_reserved=lambda _: 0)
        checked = []
        def fit(output, task, seed, device):
            pooling._admit(output, device)
            checked.append(True)
        with patch.object(study, 'verify'), \
             patch.object(study.common, 'gpu_inventory', return_value=[card(12, [12345])]), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
             patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
             patch.object(study, 'module', return_value=SimpleNamespace(fit=fit)):
            self.assertEqual(study.worker(self.args), 0)
        self.assertEqual(checked, [True])

    def test_unknown_worker_telemetry_retries_without_importing_torch(self):
        import builtins
        original = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'torch':
                raise AssertionError('torch imported after unknown admission')
            return original(name, *args, **kwargs)
        with patch.object(study, 'verify'), \
             patch.object(study.common, 'gpu_inventory', side_effect=PermissionError('unknown telemetry')), \
             patch('builtins.__import__', side_effect=guarded_import):
            self.assertEqual(study.worker(self.args), 75)

    def test_changed_uuid_at_second_worker_check_retries(self):
        cuda = SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=48 * 1024**3),
            set_per_process_memory_fraction=lambda *a, **k: None)
        with patch.object(study, 'verify'), \
             patch.object(study.common, 'gpu_inventory', side_effect=[[card(12)], [card(12, uuid='GPU-replaced')]]), \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), \
             patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}), \
             patch.object(study, 'module') as target:
            self.assertEqual(study.worker(self.args), 75)
            target.assert_not_called()


def statuses(*, pooling=(), external=(), benchmark_complete=True):
    return dict(analysis=dict(cpu_analysis_complete=True, benchmark_complete=benchmark_complete),
                pooling=dict(pending_jobs=list(pooling)), external=dict(pending_jobs=list(external)))


def gpu(index, free=48., pids=()):
    return dict(index=index, uuid=f'GPU-fixture-{index}', free_gib=free,
                pids=list(pids), utilization=100)


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.output = self.root / 'study'; self.output.mkdir()
        (self.root / 'outputs').mkdir()
        self.args = argparse.Namespace(output=self.output)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(study, 'ROOT', self.root))
        self.stack.enter_context(patch.object(study, 'verify', return_value={'poll_seconds': .001}))
        self.archive = self.stack.enter_context(patch.object(study, 'refresh_archive'))
        self.target = SimpleNamespace(is_complete=MagicMock(return_value=True),
                                      summarize=MagicMock(return_value={'complete': True}))
        self.stack.enter_context(patch.object(study, 'module', return_value=self.target))
        self.popen = self.stack.enter_context(patch.object(study.subprocess, 'Popen'))
        self.real_identity = study.common.identity
        self.identities = {}
        self.stack.enter_context(patch.object(study.common, 'identity', side_effect=self.identity))

    def identity(self, pid):
        return self.identities.get(pid) or self.real_identity(pid)

    def child(self, pid, polls=(0,)):
        child = MagicMock(pid=pid)
        child.poll.side_effect = list(polls)
        self.identities[pid] = dict(pid=pid, uid=os.getuid(), start_time_ticks=pid + 100,
                                    command_sha256='owned-fixture')
        return child

    def test_multiple_shared_cards_launch_only_remaining_distinct_jobs(self):
        initial = statuses(pooling=[dict(task='category', seed=42, phase='fit')],
                           external=[dict(seed=42, phase='fit')])
        self.popen.side_effect = [self.child(400001), self.child(400002)]
        cards = [gpu(0, 16., [555]), gpu(1, 12., [666]), gpu(2)]
        with patch.object(study, 'progress', side_effect=[initial, statuses()]), \
             patch.object(study.common, 'gpu_inventory', return_value=cards), \
             patch.object(study.time, 'sleep'), patch.object(study.common.os, 'kill') as kill:
            study.run(self.args)
        self.assertEqual(self.popen.call_count, 2)
        calls = self.popen.call_args_list
        self.assertEqual([c.kwargs['env']['CUDA_VISIBLE_DEVICES'] for c in calls],
                         ['GPU-fixture-0', 'GPU-fixture-1'])
        self.assertIn('external', calls[0].args[0]); self.assertIn('pooling', calls[1].args[0])
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'complete')
        kill.assert_not_called()

    def test_exclusive_benchmark_waits_for_empty_card_while_pooling_can_share(self):
        initial = statuses(pooling=[dict(task='category', seed=42)], benchmark_complete=False)
        self.popen.side_effect = [self.child(400001), self.child(400002)]
        cards = [gpu(0, 48., [777]), gpu(1, 48.)]
        with patch.object(study, 'progress', side_effect=[initial, statuses()]), \
             patch.object(study.common, 'gpu_inventory', return_value=cards), \
             patch.object(study.time, 'sleep'):
            study.run(self.args)
        first, second = self.popen.call_args_list
        self.assertIn('pooling', first.args[0]); self.assertIn('analysis', second.args[0])
        self.assertEqual(first.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture-0')
        self.assertEqual(second.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture-1')

    def test_fit_becoming_evaluate_does_not_launch_same_group_twice(self):
        job = dict(task='category', seed=42, phase='fit')
        initial, evaluating = statuses(pooling=[job]), statuses(pooling=[dict(job, phase='evaluate')])
        process = self.child(400001, polls=(None, 0))
        self.popen.return_value = process
        with patch.object(study, 'progress', side_effect=[initial, evaluating, statuses()]), \
             patch.object(study.common, 'gpu_inventory', return_value=[gpu(0), gpu(1)]), \
             patch.object(study.time, 'sleep'):
            study.run(self.args)
        self.popen.assert_called_once()
        process.terminate.assert_not_called()

    def test_failed_job_is_recorded_once_and_not_polled_or_restarted_forever(self):
        initial = statuses(pooling=[dict(task='category', seed=42)])
        self.popen.return_value = self.child(400001, polls=(1,))
        with patch.object(study, 'progress', return_value=initial), \
             patch.object(study.common, 'gpu_inventory', return_value=[gpu(0)]), \
             patch.object(study.time, 'sleep'):
            study.run(self.args)
        self.popen.assert_called_once()
        result = study.common.read(self.output / 'queue_state.json')
        self.assertEqual(result['state'], 'failed')
        self.assertFalse(result['automatic_run'])
        self.assertEqual(result['failures']['pooling:category:42']['exit_code'], 1)

    def test_admission_retry_75_is_rescheduled_without_marking_job_failed(self):
        initial = statuses(pooling=[dict(task='category', seed=42)])
        self.popen.side_effect = [self.child(400001, polls=(75,)), self.child(400002, polls=(0,))]
        with patch.object(study, 'progress', side_effect=[initial, initial, statuses()]), \
             patch.object(study.common, 'gpu_inventory', return_value=[gpu(0)]), \
             patch.object(study.time, 'sleep'):
            study.run(self.args)
        self.assertEqual(self.popen.call_count, 2)
        result = study.common.read(self.output / 'queue_state.json')
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['failures'], {})

    def test_guard_cooperative_gpu_lock_prevents_conflicting_followup_worker(self):
        folder = self.root / 'outputs/.qwen3guard_gpu_admission'; folder.mkdir(parents=True)
        pending = statuses(pooling=[dict(task='category', seed=42)])
        def stop(_seconds): (self.output / 'STOP').touch()
        with (folder / 'GPU-fixture-0.lock').open('a+') as guard_lock:
            import fcntl
            fcntl.flock(guard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(study, 'progress', return_value=pending), \
                 patch.object(study.common, 'gpu_inventory', return_value=[gpu(0)]), \
                 patch.object(study.time, 'sleep', side_effect=stop):
                study.run(self.args)
        self.popen.assert_not_called()
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'stopped')

    def test_second_inventory_capacity_race_does_not_launch(self):
        pending = statuses(pooling=[dict(task='category', seed=42)])
        def stop(_seconds): (self.output / 'STOP').touch()
        with patch.object(study, 'progress', return_value=pending), \
             patch.object(study.common, 'gpu_inventory', side_effect=[[gpu(0, 12)], [gpu(0, 7)]]), \
             patch.object(study.time, 'sleep', side_effect=stop):
            study.run(self.args)
        self.popen.assert_not_called()
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'stopped')

    def test_single_coordinator_lock_and_live_prior_worker_block_duplicates(self):
        import fcntl
        with (self.output / '.coordinator.lock').open('a+') as first:
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): study.run(self.args)
        self.popen.assert_not_called()
        old = dict(pid=400001, uid=os.getuid(), start_time_ticks=400101, command_sha256='prior')
        self.identities[400001] = old
        study.common.write(self.output / 'queue_state.json', {'active_workers': [{'identity': old}]})
        with self.assertRaisesRegex(RuntimeError, 'prior exact'):
            study.run(self.args)
        self.popen.assert_not_called()

    def test_empty_pending_children_are_aggregated_before_overall_completion(self):
        self.target.is_complete.return_value = False
        with patch.object(study, 'progress', return_value=statuses()), \
             patch.object(study.common, 'gpu_inventory') as inventory:
            study.run(self.args)
        self.assertEqual(self.target.summarize.call_args_list,
                         [unittest.mock.call(self.output / 'pooling'),
                          unittest.mock.call(self.output / 'external')])
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'complete')
        inventory.assert_not_called(); self.popen.assert_not_called()

    def test_active_kind_is_not_aggregated_while_its_worker_finishes(self):
        self.target.is_complete.return_value = False
        status = statuses(external=[dict(seed=42)])
        active = {0: {'job': {'kind': 'pooling', 'task': 'category', 'seed': 42}}}
        study.finalize_children(self.output, status, active)
        self.target.is_complete.assert_not_called()
        self.target.summarize.assert_not_called()

    def test_failed_aggregation_is_a_failed_study_not_a_false_complete(self):
        self.target.is_complete.return_value = False
        self.target.summarize.side_effect = RuntimeError('synthetic damaged test artifact')
        with patch.object(study, 'progress', return_value=statuses()), \
             patch.object(study.common, 'gpu_inventory') as inventory:
            with self.assertRaisesRegex(RuntimeError, 'damaged test artifact'):
                study.run(self.args)
        self.assertEqual(study.common.read(self.output / 'queue_state.json')['state'], 'failed')
        inventory.assert_not_called(); self.popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
