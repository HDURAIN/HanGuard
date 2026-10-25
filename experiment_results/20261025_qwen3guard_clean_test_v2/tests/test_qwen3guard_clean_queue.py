"""CPU-only admission, provenance and process-ownership checks for fresh Guard evaluation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scripts import hanguard
from scripts.hanguard import qwen3guard_clean_queue as queue


def card(index=0, *, pids=(), free=48.0, util=0):
    return dict(index=index, uuid=f'GPU-fixture-{index}', free_gib=free,
                utilization=util, pids=list(pids))


def xml_inventory(rows):
    parts = ['<nvidia_smi_log>']
    for uuid, processes in rows:
        if processes is None:
            body = ''
        elif isinstance(processes, str):
            body = f'<processes>{processes}</processes>'
        else:
            body = '<processes>' + ''.join(
                f'<process_info><pid>{pid}</pid><type>G</type></process_info>'
                for pid in processes) + '</processes>'
        parts.append(f'<gpu><uuid>{uuid}</uuid>{body}</gpu>')
    return ''.join(parts) + '</nvidia_smi_log>'


class InventoryTests(unittest.TestCase):
    def inventory(self, gpu=None, xml=None, compute=''):
        gpu = '0, GPU-fixture-0, 49152, 0\n' if gpu is None else gpu
        xml = xml_inventory([('GPU-fixture-0', [])]) if xml is None else xml
        responses = [types.SimpleNamespace(stdout=text) for text in (gpu, xml, compute)]
        with patch.object(queue.subprocess, 'run', side_effect=responses) as call:
            result = queue.gpu_inventory()
        self.assertEqual(call.call_count, 3)
        for args, kwargs in call.call_args_list:
            self.assertEqual(args[0][0], 'nvidia-smi')
            self.assertTrue(kwargs['check'])
            self.assertEqual(kwargs['timeout'], 15)
        return result

    def test_union_graphics_and_compute_inventory_per_physical_uuid(self):
        gpu = '1, GPU-fixture-1, 40960, 0\n0, GPU-fixture-0, 49152, 0\n'
        xml = xml_inventory([('GPU-fixture-1', []), ('GPU-fixture-0', [101, 101])])
        result = self.inventory(gpu, xml,
                                'GPU-fixture-0, 202\nGPU-fixture-1, 303\n')
        self.assertEqual(result, [card(0, pids=[101, 202]), card(1, pids=[303], free=40.0)])

    def test_empty_device_is_eligible_without_reserving_another_gpu(self):
        result = self.inventory()[0]
        self.assertTrue(queue.eligible(result, 16.0))
        self.assertFalse(queue.eligible(result, 49.0))

    def test_zero_utilization_and_large_free_memory_do_not_override_any_pid(self):
        for pid in (os.getpid(), 1, 987654):
            with self.subTest(pid=pid):
                self.assertFalse(queue.eligible(card(pids=[pid], free=48, util=0), 16))

    def test_missing_duplicate_or_unknown_full_inventory_is_rejected(self):
        cases = [
            ('missing gpu', '<nvidia_smi_log/>'),
            ('missing processes', xml_inventory([('GPU-fixture-0', None)])),
            ('unknown uuid', xml_inventory([('GPU-unregistered', [])])),
            ('duplicate gpu', xml_inventory([('GPU-fixture-0', []), ('GPU-fixture-0', [])])),
            ('unknown pid', xml_inventory([('GPU-fixture-0', [-1])])),
        ]
        for name, xml in cases:
            with self.subTest(name=name):
                with patch.object(queue.subprocess, 'run', side_effect=[
                    types.SimpleNamespace(stdout='0, GPU-fixture-0, 49152, 0\n'),
                    types.SimpleNamespace(stdout=xml),
                ]):
                    with self.assertRaises(ValueError):
                        queue.gpu_inventory()

    def test_unknown_scalar_or_compute_telemetry_is_rejected(self):
        cases = [
            ('unknown gpu util', '0, GPU-fixture-0, 49152, N/A\n', ''),
            ('unknown memory', '0, GPU-fixture-0, N/A, 0\n', ''),
            ('duplicate index', '0, GPU-fixture-0, 49152, 0\n0, GPU-fixture-1, 49152, 0\n', ''),
            ('duplicate uuid', '0, GPU-fixture-0, 49152, 0\n1, GPU-fixture-0, 49152, 0\n', ''),
            ('unregistered compute uuid', '0, GPU-fixture-0, 49152, 0\n', 'GPU-other, 44\n'),
            ('unknown compute pid', '0, GPU-fixture-0, 49152, 0\n', 'GPU-fixture-0, N/A\n'),
        ]
        for name, gpu, compute in cases:
            with self.subTest(name=name):
                responses = [types.SimpleNamespace(stdout=x) for x in (
                    gpu, xml_inventory([('GPU-fixture-0', [])]), compute)]
                with patch.object(queue.subprocess, 'run', side_effect=responses):
                    with self.assertRaises(ValueError):
                        queue.gpu_inventory()

    def test_unavailable_full_inventory_is_not_treated_as_empty(self):
        for text in ('N/A', 'Unknown'):
            with self.subTest(text=text):
                responses = [types.SimpleNamespace(stdout=x) for x in (
                    '0, GPU-fixture-0, 49152, 0\n',
                    xml_inventory([('GPU-fixture-0', text)]), '')]
                with patch.object(queue.subprocess, 'run', side_effect=responses):
                    with self.assertRaises(ValueError):
                        queue.gpu_inventory()

    def test_command_failure_or_malformed_xml_propagates_as_unknown(self):
        for error in (subprocess.TimeoutExpired('nvidia-smi', 15),
                      subprocess.CalledProcessError(1, ['nvidia-smi']),
                      PermissionError('no inventory access')):
            with self.subTest(error=error):
                with patch.object(queue.subprocess, 'run', side_effect=error):
                    with self.assertRaises(type(error)):
                        queue.gpu_inventory()
        with patch.object(queue.subprocess, 'run', side_effect=[
            types.SimpleNamespace(stdout='0, GPU-fixture-0, 49152, 0\n'),
            types.SimpleNamespace(stdout='<broken>'),
        ]):
            with self.assertRaises(queue.ET.ParseError):
                queue.gpu_inventory()

    def test_exact_system_display_is_ignored_only_when_not_a_compute_process(self):
        xml = ('<nvidia_smi_log><gpu><uuid>GPU-fixture-0</uuid><processes>'
               '<process_info><pid>9539</pid><type>G</type>'
               '<process_name>/usr/lib/xorg/Xorg</process_name></process_info>'
               '</processes></gpu></nvidia_smi_log>')
        with patch.object(queue, 'identity', return_value=queue.SYSTEM_DISPLAY):
            result = self.inventory(xml=xml)[0]
            self.assertTrue(queue.eligible(result, 20))
            self.assertEqual(result['ignored_system_display_pids'], [9539])
            result = self.inventory(xml=xml, compute='GPU-fixture-0, 9539\n')[0]
            self.assertFalse(queue.eligible(result, 20))
            self.assertEqual(result['pids'], [9539])

    def test_other_graphics_birth_or_unverified_system_identity_still_blocks(self):
        original = ('<nvidia_smi_log><gpu><uuid>GPU-fixture-0</uuid><processes>'
                    '<process_info><pid>9539</pid><type>G</type>'
                    '<process_name>/usr/lib/xorg/Xorg</process_name></process_info>'
                    '</processes></gpu></nvidia_smi_log>')
        for xml, observed in [
            (original, dict(queue.SYSTEM_DISPLAY, start_time_ticks=99999)),
            (original, dict(queue.SYSTEM_DISPLAY, uid=1000)),
            (original.replace('<type>G</type>', '<type>C+G</type>'), queue.SYSTEM_DISPLAY),
            (original.replace('/usr/lib/xorg/Xorg', '/home/other/Xorg'), queue.SYSTEM_DISPLAY),
            (original.replace('<pid>9539</pid>', '<pid>9540</pid>'), queue.SYSTEM_DISPLAY),
        ]:
            with self.subTest(xml=xml, observed=observed):
                with patch.object(queue, 'identity', return_value=observed):
                    self.assertFalse(queue.eligible(self.inventory(xml=xml)[0], 20))
        with patch.object(queue, 'identity', side_effect=PermissionError):
            self.assertFalse(queue.eligible(self.inventory(xml=original)[0], 20))


class IdentityTests(unittest.TestCase):
    def test_own_identity_and_disappeared_process(self):
        current = queue.identity(os.getpid())
        self.assertEqual(current['pid'], os.getpid())
        self.assertEqual(current['uid'], os.getuid())
        self.assertGreater(current['start_time_ticks'], 0)
        self.assertEqual(len(current['command_sha256']), 64)
        self.assertTrue(queue.same_process(current))
        with patch.object(queue, 'identity', return_value=None):
            self.assertFalse(queue.same_process(current))

    def test_pid_reuse_or_changed_command_does_not_match_old_identity(self):
        expected = dict(pid=100, uid=500, start_time_ticks=123, command_sha256='original')
        for actual in (dict(expected, start_time_ticks=124),
                       dict(expected, command_sha256='replacement'),
                       dict(expected, uid=501)):
            with self.subTest(actual=actual):
                with patch.object(queue, 'identity', return_value=actual):
                    self.assertFalse(queue.same_process(expected))

    def test_permission_or_unparseable_identity_is_conservatively_alive(self):
        expected = dict(pid=100, uid=500, start_time_ticks=123, command_sha256='original')
        for error in (PermissionError('hidden proc'), OSError('unknown'),
                      ValueError('bad stat'), IndexError('incomplete stat')):
            with self.subTest(error=error):
                with patch.object(queue, 'identity', side_effect=error):
                    self.assertTrue(queue.same_process(expected))

    def test_stat_parsing_handles_spaces_and_parentheses_in_comm_and_zombies(self):
        # Linux fields after the final ')' begin with state; starttime is index 19.
        fields = ['S'] + ['0'] * 18 + ['987654']
        stat = '123 (worker (name) with spaces) ' + ' '.join(fields)
        with patch.object(Path, 'read_bytes', return_value=b'python\x00worker\x00'), \
             patch.object(Path, 'read_text', return_value=stat), \
             patch.object(Path, 'stat', return_value=types.SimpleNamespace(st_uid=50)):
            actual = queue.identity(123)
        self.assertEqual(actual['start_time_ticks'], 987654)
        self.assertEqual(actual['command_sha256'], hashlib.sha256(b'python\x00worker\x00').hexdigest())
        for state in ('Z', 'X', 'x'):
            with self.subTest(state=state):
                with patch.object(Path, 'read_bytes', return_value=b''), \
                     patch.object(Path, 'read_text', return_value=stat.replace(') S ', f') {state} ')):
                    self.assertIsNone(queue.identity(123))


class QueueFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'outputs' / 'fresh_guard_fixture'
        self.output.mkdir(parents=True)
        queue.write(self.output / 'protocol.json', {'fresh': True})
        self.source = self.root / 'queue.py'
        self.source.write_text('QUEUE_FIXTURE = True\n')
        self.evaluator_source = self.root / 'evaluator.py'
        self.evaluator_source.write_text('EVALUATOR_FIXTURE = True\n')
        self.args = argparse.Namespace(output=self.output, gpus=[0, 1, 2], poll_seconds=.01,
                                       minimum_free_gib=16.0)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temp.cleanup)
        for name, value in (('ROOT', self.root), ('EVALUATOR', self.evaluator_source),
                            ('__file__', str(self.source))):
            self.stack.enter_context(patch.object(queue, name, value))
        self.evaluator = types.ModuleType('scripts.hanguard.qwen3guard_clean_eval')
        self.evaluator.progress = MagicMock(return_value=dict(total_chunks=3, complete_chunks=0))
        self.evaluator.is_complete = MagicMock(return_value=False)
        self.evaluator.summarize = MagicMock(return_value={'complete': True})
        self.stack.enter_context(patch.dict(sys.modules,
            {'scripts.hanguard.qwen3guard_clean_eval': self.evaluator}))
        self.stack.enter_context(patch.object(hanguard, 'qwen3guard_clean_eval',
                                             self.evaluator, create=True))
        self.popen = self.stack.enter_context(patch.object(queue.subprocess, 'Popen'))

    def stop_on_sleep(self, _seconds):
        (self.output / 'STOP').touch()

    def run_until_stopped(self, inventory):
        with patch.object(queue, 'gpu_inventory', side_effect=inventory), \
             patch.object(queue.time, 'sleep', side_effect=self.stop_on_sleep):
            queue.run(self.args)
        return queue.read(self.output / 'queue_state.json')

    def test_frozen_registration_snapshots_sources_and_rejects_changes(self):
        registration = queue.register_queue(self.output, self.args)
        self.assertEqual(registration, queue.register_queue(self.output, self.args))
        for filename, digest in registration['sources'].items():
            snapshot = self.output / 'queue_source_snapshot' / Path(filename).name
            self.assertEqual(queue.sha(snapshot), digest)
        self.evaluator_source.write_text('EVALUATOR_FIXTURE = False\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            queue.verify_sources(registration)
        with self.assertRaisesRegex(ValueError, 'protocol or source changed'):
            queue.register_queue(self.output, self.args)

    def test_changed_protocol_or_gpu_admission_configuration_is_rejected(self):
        queue.register_queue(self.output, self.args)
        original = (self.output / 'protocol.json').read_bytes()
        queue.write(self.output / 'protocol.json', {'fresh': False})
        with self.assertRaises(ValueError):
            queue.register_queue(self.output, self.args)
        (self.output / 'protocol.json').write_bytes(original)
        self.args.gpus = [2]
        with self.assertRaises(ValueError):
            queue.register_queue(self.output, self.args)

    def test_second_coordinator_cannot_enter_existing_lock(self):
        with (self.output / '.coordinator.lock').open('a+') as first:
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another coordinator'):
                queue.run(self.args)
        self.popen.assert_not_called()
        self.evaluator.progress.assert_not_called()

    def test_live_prior_worker_prevents_duplicate_run_but_reused_pid_does_not(self):
        old = dict(pid=100, uid=1000, start_time_ticks=10, command_sha256='old')
        queue.write(self.output / 'queue_state.json', {'active_workers': [{'identity': old}]})
        with patch.object(queue, 'identity', return_value=old):
            with self.assertRaisesRegex(RuntimeError, 'prior exact worker'):
                queue.run(self.args)
        self.popen.assert_not_called()
        with patch.object(queue, 'identity', return_value=dict(old, start_time_ticks=11)):
            state = self.run_until_stopped([[card(pids=[111])]])
        self.assertEqual(state['state'], 'stopped')
        self.popen.assert_not_called()

    def test_unreadable_prior_worker_identity_blocks_new_workers(self):
        queue.write(self.output / 'queue_state.json', {'active_workers': [
            {'identity': dict(pid=100, uid=1000, start_time_ticks=10, command_sha256='old')}]})
        with patch.object(queue, 'identity', side_effect=PermissionError('hidden proc')):
            with self.assertRaisesRegex(RuntimeError, 'prior exact worker'):
                queue.run(self.args)
        self.popen.assert_not_called()

    def test_unknown_inventory_waits_and_does_not_launch_or_signal(self):
        observed = []
        original = queue.coordinator_state
        def record(output, state, **details):
            observed.append((state, details))
            original(output, state, **details)
        with patch.object(queue, 'coordinator_state', side_effect=record), \
             patch.object(queue.os, 'kill') as kill:
            state = self.run_until_stopped([PermissionError('telemetry blocked')])
        self.assertEqual(observed[0][0], 'waiting_for_empty_gpu')
        self.assertIn('telemetry blocked', observed[0][1]['inventory_error'])
        self.assertEqual(state['state'], 'stopped')
        self.popen.assert_not_called()
        kill.assert_not_called()

    def test_foreign_idle_pid_blocks_admission(self):
        state = self.run_until_stopped([[card(0, pids=[os.getpid()]), card(1, free=8)]])
        self.assertEqual(state['state'], 'stopped')
        self.popen.assert_not_called()

    def test_second_inventory_race_blocks_launch(self):
        state = self.run_until_stopped([[card()], [card(pids=[777])]])
        self.assertEqual(state['state'], 'stopped')
        self.popen.assert_not_called()

    def test_completion_summarizes_before_final_state_without_using_gpu(self):
        self.evaluator.progress.return_value = dict(total_chunks=3, complete_chunks=3)
        self.evaluator.is_complete.side_effect = [False, True]
        with patch.object(queue, 'gpu_inventory') as inventory:
            queue.run(self.args)
        self.evaluator.summarize.assert_called_once_with(self.output)
        self.assertEqual(queue.read(self.output / 'queue_state.json')['state'], 'completed')
        inventory.assert_not_called()
        self.popen.assert_not_called()

    def test_invalid_safety_outputs_stop_for_review_instead_of_repeated_summary(self):
        self.evaluator.progress.return_value = dict(total_chunks=3, complete_chunks=3)
        self.evaluator.summarize.return_value = {'complete': False, 'invalid_safety_rows': 1}
        with patch.object(queue, 'gpu_inventory') as inventory:
            with self.assertRaisesRegex(RuntimeError, 'invalid Safety outputs'):
                queue.run(self.args)
        self.evaluator.summarize.assert_called_once_with(self.output)
        self.assertEqual(queue.read(self.output / 'queue_state.json')['state'], 'failed')
        inventory.assert_not_called()
        self.popen.assert_not_called()

    def test_admits_multiple_empty_gpus_only_up_to_pending_chunks(self):
        children = [MagicMock(pid=424240 + i) for i in range(2)]
        for child in children:
            child.poll.return_value = 0
        self.popen.side_effect = children
        identities = {child.pid: dict(pid=child.pid, uid=os.getuid(),
                      start_time_ticks=900 + i, command_sha256='owned')
                      for i, child in enumerate(children)}
        original_identity = queue.identity
        self.evaluator.progress.return_value = dict(total_chunks=2, complete_chunks=0)
        self.evaluator.is_complete.side_effect = [False, True]
        def identity(pid):
            return identities.get(pid) or original_identity(pid)
        with patch.object(queue, 'identity', side_effect=identity), \
             patch.object(queue, 'gpu_inventory', side_effect=[
                 [card(0), card(1), card(2)], [card(0)], [card(1)]]), \
             patch.object(queue.time, 'sleep'):
            queue.run(self.args)
        self.assertEqual(self.popen.call_count, 2)
        admitted = [call.kwargs['env']['CUDA_VISIBLE_DEVICES']
                    for call in self.popen.call_args_list]
        self.assertEqual(admitted, ['GPU-fixture-0', 'GPU-fixture-1'])

    def test_existing_cooperative_admission_lock_blocks_a_second_worker(self):
        folder = self.root / 'outputs/.qwen3guard_gpu_admission'
        folder.mkdir()
        with (folder / 'GPU-fixture-0.lock').open('a+') as first:
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self.run_until_stopped([[card()]])
        self.assertEqual(state['state'], 'stopped')
        self.popen.assert_not_called()

    def test_identity_permission_failure_stops_only_just_spawned_unreaped_child(self):
        child = MagicMock(pid=424242)
        child.poll.return_value = None
        self.popen.return_value = child
        original_identity = queue.identity
        def identity(pid):
            if pid == child.pid:
                raise PermissionError('child identity unavailable')
            return original_identity(pid)
        with patch.object(queue, 'identity', side_effect=identity), \
             patch.object(queue, 'gpu_inventory', side_effect=[[card()], [card()]]), \
             patch.object(queue.os, 'kill') as kill:
            with self.assertRaises(PermissionError):
                queue.run(self.args)
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=30)
        child.kill.assert_not_called()
        kill.assert_not_called()
        state = queue.read(self.output / 'queue_state.json')
        self.assertEqual(state['state'], 'failed')
        self.assertEqual(state['active_workers'], [])

    def test_unidentified_still_running_child_is_not_orphaned_or_relaunched(self):
        child = MagicMock(pid=424242)
        child.poll.return_value = None
        self.popen.return_value = child
        original_identity = queue.identity
        def identity(pid):
            return None if pid == child.pid else original_identity(pid)
        with patch.object(queue, 'identity', side_effect=identity), \
             patch.object(queue, 'gpu_inventory', side_effect=[[card()], [card()]]), \
             patch.object(queue.time, 'sleep', side_effect=self.stop_on_sleep), \
             patch.object(queue.os, 'kill') as kill:
            queue.run(self.args)
        self.assertEqual(self.popen.call_count, 1)
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=30)
        kill.assert_not_called()

    def test_owned_worker_survives_later_foreign_arrival_and_queue_finishes(self):
        process = MagicMock(pid=424242)
        process.poll.side_effect = [None, 0]
        self.popen.return_value = process
        worker_identity = dict(pid=process.pid, uid=os.getuid(), start_time_ticks=900,
                               command_sha256='owned')
        actual_identity = queue.identity
        def identities(pid):
            return worker_identity if pid == process.pid else actual_identity(pid)
        self.evaluator.is_complete.side_effect = [False, False, True]
        self.evaluator.progress.side_effect = [dict(total_chunks=1, complete_chunks=0),
                                               dict(total_chunks=1, complete_chunks=0),
                                               dict(total_chunks=1, complete_chunks=1)]
        inventories = [[card()], [card()], [card(pids=[process.pid, 777])]]
        with patch.object(queue, 'identity', side_effect=identities), \
             patch.object(queue, 'gpu_inventory', side_effect=inventories), \
             patch.object(queue.time, 'sleep'), patch.object(queue.os, 'kill') as kill:
            queue.run(self.args)
        state = queue.read(self.output / 'queue_state.json')
        self.assertEqual(state['state'], 'completed')
        self.assertEqual(self.popen.call_count, 1)
        launch = self.popen.call_args
        self.assertEqual(launch.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-fixture-0')
        self.assertEqual(launch.args[0][-2:], ['--gpu', '0'])
        arrivals = [x for x in state['recent_events'] if x['event'] == 'post_admission_foreign_pids']
        self.assertEqual(arrivals[0]['pids'], [777])
        self.assertEqual(arrivals[0]['action'], 'continue_owned_worker')
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
