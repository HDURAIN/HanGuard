"""CPU-only serial saved-checkpoint recovery and provenance checks."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts.hanguard import paper_followup_recovery as recovery


def card(free=48, pids=(999,), uuid='GPU-fixture', index=2):
    return dict(index=index, uuid=uuid, free_gib=free, pids=list(pids), utilization=100)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'outputs' / 'followup'
        (self.output / 'source_snapshot').mkdir(parents=True)
        self.source = self.root / 'recovery.py'
        self.source.write_text('RECOVERY_FIXTURE = True\n')
        common = recovery.study.common
        for name in ('protocol.json', 'external/protocol.json',
                     'external/selection_lock.json', 'external/data/test/manifest.json'):
            (self.output / name).parent.mkdir(parents=True, exist_ok=True)
            common.write(self.output / name, {'fixture': name})
        self.original_failure = dict(state='failed', automatic_run=False, active_workers=[],
            failures={'external::43': {'exit_code': 1, 'reason': 'lock contention'},
                      'external::44': {'exit_code': 1, 'reason': 'lock contention'}})
        common.write(self.output / 'queue_state.json', self.original_failure)
        self.status = dict(pooling={'pending_jobs': []},
            analysis=dict(cpu_analysis_complete=True, benchmark_complete=True),
            external=dict(selected_seeds=[42, 43, 44], evaluated_seeds=[42],
                          pending_jobs=[dict(seed=43, phase='evaluate'), dict(seed=44, phase='evaluate')]))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temp.cleanup)
        self.stack.enter_context(patch.object(recovery, 'ROOT', self.root))
        self.stack.enter_context(patch.object(recovery, '__file__', str(self.source)))
        self.parent_verify = self.stack.enter_context(patch.object(recovery.study, 'verify'))
        self.stack.enter_context(patch.object(recovery.study, 'module',
            return_value=SimpleNamespace(progress=lambda _: self.status['external'])))
        self.stack.enter_context(patch.object(recovery.study, 'progress', side_effect=lambda _: self.status))
        self.finalize = self.stack.enter_context(patch.object(recovery.study, 'finalize_children'))
        self.archive = self.stack.enter_context(patch.object(recovery.study, 'refresh_archive'))
        self.inventory = self.stack.enter_context(patch.object(recovery.study.common, 'gpu_inventory', return_value=[card()]))
        self.popen = self.stack.enter_context(patch.object(recovery.subprocess, 'Popen'))

    def test_registration_rejects_fit_work_or_incomplete_selections(self):
        cases = [dict(pending_jobs=[dict(seed=43, phase='fit')], selected_seeds=[42, 43, 44]),
                 dict(pending_jobs=[dict(seed=43)], selected_seeds=[42, 43, 44]),
                 dict(pending_jobs=[dict(seed=43, phase='evaluate')], selected_seeds=[42, 43]),
                 dict(pending_jobs=[], selected_seeds=[42, 43, 44, 45])]
        for status in cases:
            with self.subTest(status=status):
                self.status['external'] = status
                with self.assertRaises(ValueError):
                    recovery.register(self.output)
        self.assertFalse((self.output / 'serial_test_recovery_registration.json').exists())
        self.popen.assert_not_called()

    def test_registration_requires_existing_frozen_test_barrier(self):
        (self.output / 'external/selection_lock.json').unlink()
        with self.assertRaisesRegex(ValueError, 'test barrier'):
            recovery.register(self.output)
        self.popen.assert_not_called()

    def test_registration_binds_source_inputs_and_preserves_failure_snapshot(self):
        common = recovery.study.common
        registration = recovery.register(self.output)
        self.assertEqual(common.read(self.output / 'queue_before_serial_recovery.json'), self.original_failure)
        self.assertEqual(common.sha(self.output / 'source_snapshot/recovery.py'), common.sha(self.source))
        self.assertEqual(registration['mode'], 'Sequential original external workers; saved checkpoints only; no retraining')
        recovery.verify(self.output, registration)
        for filename in ('protocol.json', 'external/protocol.json',
                         'external/selection_lock.json', 'external/data/test/manifest.json'):
            with self.subTest(filename=filename):
                path = self.output / filename
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'Frozen recovery'):
                    recovery.verify(self.output, registration)
                path.write_bytes(original)
        self.source.write_text('RECOVERY_FIXTURE = False\n')
        with self.assertRaisesRegex(ValueError, 'Frozen recovery'):
            recovery.verify(self.output, registration)
        with self.assertRaisesRegex(ValueError, 'registration changed'):
            recovery.register(self.output)
        self.source.write_text('RECOVERY_FIXTURE = True\n')
        common.write(self.output / 'queue_state.json', {'state': 'complete', 'active_workers': []})
        self.assertEqual(recovery.register(self.output), registration)
        self.assertEqual(common.read(self.output / 'queue_before_serial_recovery.json'), self.original_failure)

    def test_second_coordinator_is_rejected_before_registration(self):
        with (self.output / '.coordinator.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                recovery.run(self.output)
        self.popen.assert_not_called()
        self.assertFalse((self.output / 'serial_test_recovery_registration.json').exists())

    def test_live_exact_prior_worker_blocks_recovery(self):
        recovery.study.common.write(self.output / 'queue_state.json', dict(self.original_failure,
            active_workers=[{'identity': {'pid': 12345, 'uid': os.getuid()}}]))
        with patch.object(recovery.study.common, 'same_process', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'prior worker'):
                recovery.run(self.output)
        self.popen.assert_not_called()
        self.assertFalse((self.output / 'serial_test_recovery_registration.json').exists())

    def _children(self, codes):
        """Return Popen children whose completion updates synthetic disk status."""
        outstanding = set()
        maximum = []
        launched = []
        children = []
        original_identity = recovery.study.common.identity
        def create(command, **kwargs):
            self.assertFalse(outstanding, 'A second test was started while another remained active')
            index = len(children)
            code = codes[index]
            seed = int(command[command.index('--seed') + 1])
            self.assertEqual(command[1], str(Path(recovery.study.__file__).resolve()))
            self.assertEqual(command[2], 'worker')
            self.assertEqual(command[command.index('--kind') + 1], 'external')
            self.assertNotIn('fit', command)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture')
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '4')
            launched.append(seed)
            class Child:
                pid = 20000000 + index
                returncode = None
                polled = 0
                def poll(child):
                    child.polled += 1
                    if child.polled == 1:
                        return None
                    if child.returncode is None:
                        child.returncode = code
                        outstanding.remove(child.pid)
                        if code == 0:
                            self.assertEqual(self.status['external']['pending_jobs'][0]['seed'], seed)
                            self.status['external']['pending_jobs'].pop(0)
                            self.status['external']['evaluated_seeds'].append(seed)
                    return child.returncode
                def terminate(child):
                    raise AssertionError('Successful mock children must never be signalled')
            child = Child()
            children.append(child)
            outstanding.add(child.pid)
            maximum.append(len(outstanding))
            return child
        def identity(pid):
            if any(child.pid == pid for child in children):
                return dict(pid=pid, uid=os.getuid(), start_time_ticks=pid + 1, command_sha256='mock-child')
            return original_identity(pid)
        self.popen.side_effect = create
        return launched, children, maximum, identity

    def test_only_one_original_worker_runs_and_order_is_43_then_44(self):
        launched, children, maximum, identity = self._children([0, 0])
        with patch.object(recovery.study.common, 'identity', side_effect=identity), \
             patch.object(recovery.time, 'sleep'):
            recovery.run(self.output)
        self.assertEqual(launched, [43, 44])
        self.assertEqual(max(maximum), 1)
        self.assertTrue(all(child.returncode == 0 for child in children))
        self.finalize.assert_called_once_with(self.output, self.status, {})
        self.archive.assert_called_once_with(self.output)
        state = recovery.study.common.read(self.output / 'queue_state.json')
        self.assertEqual(state['state'], 'complete')
        self.assertEqual(state['recovered_failures'], self.original_failure['failures'])
        self.assertEqual(state['failures'], {})
        self.assertEqual(recovery.study.common.read(self.output / 'queue_before_serial_recovery.json'), self.original_failure)

    def test_exit75_requeues_same_test_before_next_seed(self):
        launched, children, maximum, identity = self._children([75, 0, 0])
        with patch.object(recovery.study.common, 'identity', side_effect=identity), \
             patch.object(recovery.time, 'sleep'):
            recovery.run(self.output)
        self.assertEqual(launched, [43, 43, 44])
        self.assertEqual(max(maximum), 1)
        state = recovery.study.common.read(self.output / 'queue_state.json')
        self.assertEqual([event['exit_code'] for event in state['recent_events']], [75, 0, 0])
        self.assertEqual(state['state'], 'complete')

    def test_real_worker_failure_stops_without_starting_another_seed(self):
        launched, children, maximum, identity = self._children([1])
        with patch.object(recovery.study.common, 'identity', side_effect=identity), \
             patch.object(recovery.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'saved-model test failed'):
                recovery.run(self.output)
        self.assertEqual(launched, [43])
        self.assertEqual(max(maximum), 1)
        self.finalize.assert_not_called()
        self.archive.assert_not_called()
        self.assertEqual(recovery.study.common.read(self.output / 'queue_state.json')['state'], 'failed')

    def test_complete_input_only_finalizes_and_archives_without_gpu_query(self):
        self.status['external']['pending_jobs'] = []
        recovery.run(self.output)
        self.finalize.assert_called_once_with(self.output, self.status, {})
        self.archive.assert_called_once_with(self.output)
        self.inventory.assert_not_called()
        self.popen.assert_not_called()

    def test_telemetry_failure_waits_without_launch_and_honors_stop(self):
        self.inventory.side_effect = PermissionError('unavailable inventory')
        observed = []
        original_state = recovery.study.state
        def state(output, phase, **details):
            observed.append((phase, details))
            return original_state(output, phase, **details)
        def stop(_):
            (self.output / 'STOP').write_text('stop fixture')
        with patch.object(recovery.study, 'state', side_effect=state), \
             patch.object(recovery.time, 'sleep', side_effect=stop):
            recovery.run(self.output)
        self.assertEqual(observed[0][0], 'waiting_for_recovery_capacity')
        self.assertIn('unavailable inventory', observed[0][1]['inventory_error'])
        self.assertEqual(observed[-1][0], 'stopped')
        self.popen.assert_not_called()

    def test_second_inventory_capacity_race_prevents_launch(self):
        self.inventory.side_effect = [[card()], [card(13)]]
        def stop(_):
            (self.output / 'STOP').touch()
        with patch.object(recovery.time, 'sleep', side_effect=stop):
            recovery.run(self.output)
        self.popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
