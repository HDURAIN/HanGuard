"""CPU-only scheduler checks for public-GPU admission and selection barriers."""
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.hanguard import four_arm_experiment_study as study
from scripts.hanguard import four_arm_token_cache as cache
from scripts.hanguard import four_arm_experiment_training as training


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'study'
        self.output.mkdir()
        self.protocol = dict(schema=study.SCHEMA, use_lora=False, adapter_path=None,
            fusion_weights=[.25] * 4, tasks=['binary', 'category'], seeds=[42, 43, 44],
            inputs={}, input_stats={}, code_sha256={}, execution=dict(study.POLICY))
        self._register()
        study.write(self.output / 'state.json', dict(state='registered'))

    def _register(self):
        study.write(self.output / 'protocol.json', self.protocol)
        study.write(self.output / 'protocol_registration.json',
                    dict(protocol_sha256=study.sha(self.output / 'protocol.json')))

    def _worker(self, phase='fit'):
        args = SimpleNamespace(output=self.output, worker_id='worker_1', phase=phase,
                               task=None if phase == 'extract' else 'binary', split='train' if phase == 'extract' else None,
                               seed=None if phase == 'extract' else 42, device='cuda', gpu=3)
        exact = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=12345,
                     command_sha256='command', argv=['python', 'study.py'], cwd=str(study.ROOT))
        record = dict(identity=exact, phase=args.phase, task=args.task, split=args.split,
                      seed=args.seed, device=args.device, gpu=args.gpu)
        path = self.output / 'workers' / 'worker_1.json'
        study.write(path, record)
        return args, exact, path

    def test_every_compute_pid_blocks_empty_card_admission(self):
        cards = [dict(index=0, pids=[], free_gib=40),
                 dict(index=1, pids=[os.getpid()], free_gib=47),
                 dict(index=2, pids=[999999999], free_gib=47),
                 dict(index=3, pids=[], free_gib=19)]
        self.assertEqual(study.empty_cards(cards, list(range(8)), 20), [0])

    def test_worker_blocks_foreign_unknown_or_low_memory_before_cuda(self):
        args, exact, ledger = self._worker()
        for card in (dict(index=3, pids=[999999999], free_gib=47),
                     dict(index=3, pids=[os.getpid()], free_gib=47),
                     dict(index=3, pids=[], free_gib=19)):
            with self.subTest(card=card), patch.object(study, 'identity', return_value=exact), \
                    patch.object(study, 'inventory', return_value=[card]), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), \
                    patch.object(torch.cuda, 'init') as initialize:
                self.assertEqual(study.worker(args), 75)
                initialize.assert_not_called()
                self.assertFalse(ledger.with_suffix('.admitted.json').exists())

    def test_missing_inventory_or_card_blocks_cuda_and_retries(self):
        args, exact, ledger = self._worker()
        for result, error in (([], None), (None, RuntimeError('nvidia-smi unavailable'))):
            with self.subTest(error=error), patch.object(study, 'identity', return_value=exact), \
                    patch.object(study, 'inventory', return_value=result, side_effect=error), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), \
                    patch.object(torch.cuda, 'init') as initialize:
                self.assertEqual(study.worker(args), 75)
                initialize.assert_not_called()
                self.assertFalse(ledger.with_suffix('.admitted.json').exists())

    def test_worker_identity_change_is_rejected_before_inventory_or_cuda(self):
        args, exact, _ = self._worker()
        changed = dict(exact, start_time_ticks=exact['start_time_ticks'] + 1)
        with patch.object(study, 'identity', return_value=changed), \
                patch.object(study, 'inventory') as inventory, patch.object(torch.cuda, 'init') as initialize:
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                study.worker(args)
            inventory.assert_not_called()
            initialize.assert_not_called()

    def test_admitted_extraction_uses_exact_stamp_and_single_admission_check(self):
        args, exact, ledger = self._worker('extract')
        with patch.object(study, 'identity', return_value=exact), \
                patch.object(study, 'inventory', return_value=[dict(index=3, pids=[], free_gib=40)]) as inventory, \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), \
                patch.object(torch.cuda, 'init'), patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch, 'zeros', return_value=object()), patch.object(torch, 'set_num_threads'), \
                patch.object(cache, 'extract_worker') as extract:
            self.assertEqual(study.worker(args), 0)
            extract.assert_called_once_with(self.output.resolve(), 'train', 'worker_1',
                                            device='cuda', admission_already_checked=True)
            inventory.assert_called_once()
        admitted = study.read(ledger.with_suffix('.admitted.json'))
        self.assertEqual(admitted['identity'], exact)
        self.assertEqual(admitted['physical_gpu'], 3)
        self.assertEqual(admitted['protocol_sha256'], study.sha(self.output / 'protocol.json'))

    def test_active_foreign_arrival_is_recorded_without_stopping_worker(self):
        scheduler = study.Scheduler(self.output)
        child = Mock()
        scheduler.active['owned'] = dict(worker_id='owned', gpu=3, identity={'pid': 123}, child=child)
        cards = [dict(index=3, pids=[123, 987], free_gib=1)]
        with patch.object(study, 'inventory', return_value=cards):
            self.assertEqual(scheduler.cards(), cards)
            scheduler.cards()
        arrivals = list((self.output / 'foreign_arrivals').glob('*.json'))
        self.assertEqual(len(arrivals), 1)
        self.assertEqual(study.read(arrivals[0])['action'], 'record_only_continue_owned_worker')
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_inventory_failure_does_not_make_any_card_available(self):
        scheduler = study.Scheduler(self.output)
        with patch.object(study, 'inventory', side_effect=RuntimeError('unreadable')):
            self.assertEqual(scheduler.cards(), [])
        self.assertEqual(study.read(self.output / 'state.json')['state'], 'gpu_inventory_unavailable')

    def test_all_six_selections_bind_checkpoints_and_validation_predictions(self):
        for task in self.protocol['tasks']:
            for seed in self.protocol['seeds']:
                folder = study.selection_path(self.output, task, seed).parent
                folder.mkdir(parents=True)
                checkpoint, predictions = folder / 'selected_heads.pt', folder / 'validation_predictions.npz'
                checkpoint.write_bytes(f'{task}/{seed}/weights'.encode())
                predictions.write_bytes(f'{task}/{seed}/validation-only predictions'.encode())
                study.write(folder / 'selection.json', dict(task=task, seed=seed,
                    protocol_sha256=study.sha(self.output / 'protocol.json'),
                    checkpoint_path=str(checkpoint), checkpoint_sha256=study.sha(checkpoint),
                    validation_predictions_path=str(predictions), validation_predictions_sha256=study.sha(predictions)))
        lock = study.lock_selections(self.output)
        self.assertEqual(set(lock['selections']), {f'{task}/s{seed}' for task in self.protocol['tasks'] for seed in self.protocol['seeds']})
        self.assertEqual(len(lock['artifacts']), 18)
        self.assertEqual(study.lock_selections(self.output), lock)
        (self.output / 'heads' / 'category' / 's44' / 'selected_heads.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Selection artifact changed'):
            study.lock_selections(self.output)

    def test_missing_task_selection_cannot_publish_test_lock(self):
        with self.assertRaises(FileNotFoundError):
            study.lock_selections(self.output)
        self.assertFalse((self.output / 'selection_lock.json').exists())

    def test_registered_source_change_blocks_execution(self):
        source = self.root / 'scientific_source.py'
        source.write_text('constant = 1\n')
        self.protocol['code_sha256'] = {str(source): study.sha(source)}
        self._register()
        study.verify(self.output)
        source.write_text('constant = 2\n')
        with self.assertRaisesRegex(ValueError, 'Registered implementation changed'):
            study.verify(self.output)

    def test_extraction_does_not_launch_more_models_than_remaining_chunks(self):
        scheduler = study.Scheduler(self.output)
        calls = []
        def launch(phase, **kwargs):
            calls.append((phase, kwargs))
            scheduler.active[str(len(calls))] = dict(gpu=kwargs['gpu'])
        iteration = 0
        def poll():
            nonlocal iteration
            iteration += 1
            if iteration == 2:
                scheduler.active.clear()
        progress = [dict(total_chunks=2, complete_chunks=0, pending_chunks=2, claimed_chunks=0),
                    dict(total_chunks=2, complete_chunks=2, pending_chunks=0, claimed_chunks=0)]
        cards = [dict(index=n, pids=[], free_gib=40) for n in range(8)]
        with patch.object(cache, 'prepare'), patch.object(cache, 'progress', side_effect=progress), \
                patch.object(cache, 'all_complete', side_effect=[False, True]), \
                patch.object(cache, 'completion_manifest_finalize') as finalize, \
                patch.object(scheduler, 'launch', side_effect=launch), \
                patch.object(scheduler, 'poll', side_effect=poll), \
                patch.object(scheduler, 'cards', return_value=cards), patch.object(scheduler, 'state'), \
                patch.object(study.time, 'sleep'):
            scheduler.extract('validation')
        self.assertEqual(len(calls), 2)
        self.assertEqual([r[1]['gpu'] for r in calls], [0, 1])
        finalize.assert_called_once_with(self.output, 'validation')


if __name__ == '__main__':
    unittest.main()
