"""CPU checks for fresh inputs, test barriers and process-bound GPU scheduling."""
import argparse
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from scripts.hanguard import clean_ablation_study as study
from scripts.hanguard import frozen_base_readout_study as base


class CleanStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'outputs/hanguard_clean_ablation_fixture'
        self.output.mkdir(parents=True)
        self.data = self.output / 'data'
        self.model = self.root / 'models/base'
        self.data.mkdir()
        self.model.mkdir(parents=True)
        self.source = self.root / 'source.py'
        self.source.write_text('SOURCE = True\n')
        self.dependency = self.root / 'dependency.py'
        self.dependency.write_text('DEPENDENCY = True\n')
        self.input = self.data / 'train.parquet'
        self.input.write_bytes(b'raw data fixture')
        info = self.input.stat()
        self.protocol = dict(schema=study.SCHEMA, study=str(self.output),
            historical_experiment_inputs=[], selected_test_exclusions=[],
            posthoc_subset_enabled=False, use_lora=False, joint_same_forward=True,
            fusion_weights=[.25] * 4, runtime={'synthetic': True},
            data_dir=str(self.data), features_dir=str(self.output / 'features'), model_path=str(self.model),
            code_sha256={str(self.source): study.sha(self.source)},
            dependency_sha256={str(self.dependency): study.sha(self.dependency)},
            inputs={str(self.input): study.sha(self.input)},
            input_stats={str(self.input): dict(size=info.st_size, mtime_ns=info.st_mtime_ns)},
            execution=copy.deepcopy(study.EXECUTION),
            calibration={'required_distinct_processes': 2}, tasks=['binary', 'category'],
            seeds=[42, 43, 44], numeric_path={'cublas_workspace_config': ':4096:8',
                'triton_cache_dir': str(self.output / 'runtime/triton_cache'),
                'torchinductor_cache_dir': str(self.output / 'runtime/torchinductor_cache'),
                'cuda_cache_path': str(self.output / 'runtime/cuda_cache')})
        self.save_protocol()
        study.write(self.output / 'state.json', {'state': 'registered'})
        self.fake_identity = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=100,
            command_sha256='a' * 64, argv=['synthetic'], cwd=str(study.ROOT))

    def tearDown(self):
        study._SELECTION_VERIFICATION_CACHE.pop(str(self.output), None)
        self.temp.cleanup()

    def save_protocol(self):
        study.write(self.output / 'protocol.json', self.protocol)
        study.write(self.output / 'protocol_registration.json', {'protocol_sha256': study.sha(self.output / 'protocol.json')})

    def scheduler(self):
        with patch.object(study, 'verify', return_value=self.protocol):
            return study.Scheduler(self.output)

    def worker_fixture(self):
        args = argparse.Namespace(output=self.output, worker_id='fixture_worker', phase='calibrate',
            task=None, seed=None, split=None, device='cuda', gpu=2)
        folder = self.output / 'workers'
        folder.mkdir(exist_ok=True)
        record = dict(identity=self.fake_identity, phase=args.phase, task=None, seed=None,
                      split=None, device='cuda', gpu=2)
        study.write(folder / f'{args.worker_id}.json', record)
        return args

    def preflight_fixture(self):
        from scripts.hanguard.clean_ablation_features import SCHEMA
        folder = self.output / 'calibration'
        folder.mkdir()
        plan = self.output / 'features/train/plan.json'
        study.write(plan, {'groups': [[0], [1]]})
        report = dict(schema=SCHEMA, passed=True, protocol_sha256=study.sha(self.output / 'protocol.json'),
                      train_plan_sha256=study.sha(plan),
                      records=[{'group': 0, 'repeated_exact': True, 'same_forward_last_exact': True}])
        study.write(folder / 'baseline.json', report)
        records = []
        for number in range(2):
            path = folder / f'worker{number}.json'
            process = dict(self.fake_identity, pid=100 + number, start_time_ticks=50 + number)
            actual = dict(report, worker_id=f'worker{number}', identity=process, physical_gpu=str(number))
            study.write(path, actual)
            study.write(self.output / 'workers' / f'worker{number}.json',
                dict(worker_id=f'worker{number}', identity=process, phase='calibrate', device='cuda',
                     gpu=number, protocol_sha256=report['protocol_sha256']))
            records.append(dict(worker_id=f'worker{number}', path=str(path), sha256=study.sha(path),
                                identity=process, gpu=number))
        value = dict(complete=True, protocol_sha256=study.sha(self.output / 'protocol.json'),
            baseline_sha256=study.sha(folder / 'baseline.json'), calibrations=records)
        study.write(self.output / 'feature_preflight_lock.json', value)
        return value

    def test_raw_registered_inputs_source_and_dependency_immutability(self):
        with patch.object(base, 'runtime_identity', return_value={'synthetic': True}):
            self.assertEqual(study.verify(self.output, full_hash=True)['historical_experiment_inputs'], [])
            for source in (self.source, self.dependency):
                before = source.read_bytes()
                source.write_bytes(before + b'# changed')
                with self.subTest(path=source), self.assertRaisesRegex(ValueError, 'implementation changed'):
                    study.verify(self.output)
                source.write_bytes(before)
            before = self.input.read_bytes()
            info = self.input.stat()
            self.input.write_bytes(b'RAW data fixture')
            os.utime(self.input, ns=(info.st_atime_ns, info.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'Input bytes changed'):
                study.verify(self.output, full_hash=True)
            self.input.write_bytes(before)

    def test_historical_inputs_subsets_and_changed_protocol_rejected(self):
        with patch.object(base, 'runtime_identity', return_value={'synthetic': True}):
            for key, bad in (('historical_experiment_inputs', ['old.npy']),
                             ('selected_test_exclusions', ['sample']), ('posthoc_subset_enabled', True)):
                before = self.protocol[key]
                self.protocol[key] = bad
                self.save_protocol()
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract changed'):
                    study.verify(self.output)
                self.protocol[key] = before
            outside = self.root / 'outputs/old/cache.npy'
            outside.parent.mkdir(parents=True)
            outside.write_bytes(b'OLD')
            info = outside.stat()
            self.protocol['inputs'] = {str(outside): study.sha(outside)}
            self.protocol['input_stats'] = {str(outside): dict(size=info.st_size, mtime_ns=info.st_mtime_ns)}
            self.save_protocol()
            with self.assertRaisesRegex(ValueError, 'historical experiment artifact'):
                study.verify(self.output)
            study.write(self.output / 'protocol.json', {})
            with self.assertRaisesRegex(ValueError, 'registration changed'):
                study.verify(self.output)

    def test_initialization_refuses_old_output_before_reading_old_protocol(self):
        with patch.object(study, 'ROOT', self.root), patch.object(study, 'read') as read:
            with self.assertRaisesRegex(ValueError, 'raw labeled data'):
                study.initialize(self.root / 'new', self.root / 'outputs/old', self.model)
            read.assert_not_called()

    def test_preflight_requires_two_distinct_births_and_intact_reports(self):
        value = self.preflight_fixture()
        with patch.object(study, 'verify', return_value=self.protocol):
            self.assertTrue(study.verify_preflight(self.output)['complete'])
            value['calibrations'][1]['identity'] = value['calibrations'][0]['identity']
            second = Path(value['calibrations'][1]['path'])
            report = study.read(second)
            report['identity'] = value['calibrations'][0]['identity']
            study.write(second, report)
            value['calibrations'][1]['sha256'] = study.sha(second)
            ledger_path = self.output / 'workers/worker1.json'
            ledger = study.read(ledger_path)
            ledger['identity'] = report['identity']
            study.write(ledger_path, ledger)
            study.write(self.output / 'feature_preflight_lock.json', value)
            with self.assertRaisesRegex(ValueError, 'Independent process'):
                study.verify_preflight(self.output)
            identity = dict(self.fake_identity, pid=102, start_time_ticks=52)
            value['calibrations'][1]['identity'] = identity
            report['identity'], ledger['identity'] = identity, identity
            study.write(second, report)
            study.write(ledger_path, ledger)
            value['calibrations'][1]['sha256'] = study.sha(second)
            study.write(self.output / 'feature_preflight_lock.json', value)
            Path(value['calibrations'][0]['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'calibration evidence'):
                study.verify_preflight(self.output)

    def test_preflight_binds_protocol_train_plan_outputs_and_worker_identity(self):
        value = self.preflight_fixture()
        path = Path(value['calibrations'][1]['path'])
        original = study.read(path)
        changes = ({'protocol_sha256': 'f' * 64}, {'train_plan_sha256': 'e' * 64},
                   {'records': [{'repeated_exact': False}]}, {'physical_gpu': '7'},
                   {'identity': dict(original['identity'], uid=9999)})
        with patch.object(study, 'verify', return_value=self.protocol):
            for change in changes:
                study.write(path, dict(original, **change))
                value['calibrations'][1]['sha256'] = study.sha(path)
                study.write(self.output / 'feature_preflight_lock.json', value)
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'exact worker identity changed'):
                    study.verify_preflight(self.output)
            study.write(path, original)
            value['calibrations'][1]['sha256'] = study.sha(path)
            baseline_path = self.output / 'calibration/baseline.json'
            baseline = study.read(baseline_path)
            baseline['records'][0]['same_forward_last_exact'] = False
            study.write(baseline_path, baseline)
            value['baseline_sha256'] = study.sha(baseline_path)
            study.write(self.output / 'feature_preflight_lock.json', value)
            with self.assertRaisesRegex(ValueError, 'baseline does not bind exact'):
                study.verify_preflight(self.output)

    def test_all_probe_and_head_artifacts_are_required_at_test_barrier(self):
        from scripts.hanguard import clean_ablation_training as training
        artifacts = {}
        for index in range(38):
            path = self.output / 'selected' / f'{index}.json'
            study.write(path, {'index': index})
            artifacts[str(path)] = study.sha(path)
        selections = {f'{task}/s{seed}': 'a' * 64 for task in study.TASKS for seed in study.SEEDS}
        expected = dict(artifacts=artifacts, selections=selections)
        lock = dict(complete=True, protocol_sha256=study.sha(self.output / 'protocol.json'), **expected)
        study.write(self.output / 'selection_lock.json', lock)
        with patch.object(study, 'verify'), patch.object(study, 'verify_preflight'), patch.object(training, 'selection_artifacts', return_value=expected):
            self.assertTrue(study.verify_selection_lock(self.output)['complete'])
            altered = copy.deepcopy(lock)
            altered['artifacts'].pop(next(iter(artifacts)))
            study.write(self.output / 'selection_lock.json', altered)
            with self.assertRaisesRegex(ValueError, 'selection artifacts changed'):
                study.verify_selection_lock(self.output)
            study.write(self.output / 'selection_lock.json', lock)
            Path(next(iter(artifacts))).write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artifact escaped.*or changed'):
                study.verify_selection_lock(self.output)

    def test_cached_test_barrier_rehashes_on_same_size_restored_mtime_change(self):
        from scripts.hanguard import clean_ablation_training as training
        path = self.output / 'selected.pt'
        path.write_bytes(b'1234')
        expected = dict(selections={'binary/s42': 'a' * 64}, artifacts={str(path): study.sha(path)})
        study.write(self.output / 'selection_lock.json', dict(complete=True,
            protocol_sha256=study.sha(self.output / 'protocol.json'), **expected))
        with patch.object(study, 'verify'), patch.object(study, 'verify_preflight'), \
             patch.object(training, 'selection_artifacts', return_value=expected) as artifacts:
            study.verify_selection_lock(self.output)
            time.sleep(.12)
            study.verify_selection_lock(self.output)
            self.assertEqual(artifacts.call_count, 1)
            info = path.stat()
            path.write_bytes(b'4321')
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'artifact escaped.*or changed'):
                study.verify_selection_lock(self.output)
            self.assertEqual(artifacts.call_count, 2)

    def test_occupied_unknown_missing_and_low_memory_cards_never_admit(self):
        args = self.worker_fixture()
        cases = ([dict(index=2, pids=[987654], free_gib=45.)],
                 [dict(index=2, pids=[], free_gib=19.)], [], RuntimeError('unknown telemetry'),
                 [dict(index=2, free_gib=45.)], [dict(index=2, pids=[])],
                 [dict(index=2, pids=[], free_gib=float('nan'))])
        for cards in cases:
            with self.subTest(cards=cards), patch.object(study, 'verify', return_value=self.protocol), \
                 patch.object(study, 'identity', return_value=self.fake_identity), \
                 patch.object(study, 'inventory', **({'side_effect': cards} if isinstance(cards, Exception) else {'return_value': cards})), \
                 patch('torch.cuda.init') as cuda_init, \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'):
                self.assertEqual(study.worker(args), 75)
                cuda_init.assert_not_called()
                self.assertFalse((self.output / 'workers/fixture_worker.admitted.json').exists())

    def test_wrong_worker_birth_or_physical_device_binding_blocks_cuda(self):
        args = self.worker_fixture()
        with patch.object(study, 'verify', return_value=self.protocol), patch.object(study, 'identity', return_value=dict(self.fake_identity, start_time_ticks=101)):
            with self.assertRaisesRegex(ValueError, 'exact phase/process identity'):
                study.worker(args)
        with patch.object(study, 'verify', return_value=self.protocol), patch.object(study, 'identity', return_value=self.fake_identity), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='4'):
            with self.assertRaisesRegex(ValueError, 'Physical CUDA binding'):
                study.worker(args)

    def test_foreign_arrival_is_recorded_and_does_not_terminate_any_worker(self):
        scheduler = self.scheduler()
        child = MagicMock()
        scheduler.active['owned'] = dict(worker_id='owned', gpu=2,
            identity=dict(self.fake_identity, pid=100), child=child)
        with patch.object(study, 'inventory', return_value=[dict(index=2, pids=[100, 987654], free_gib=3.)]):
            scheduler.cards()
            scheduler.cards()
        files = list((self.output / 'foreign_arrivals').glob('*.json'))
        self.assertEqual(len(files), 1)
        self.assertEqual(study.read(files[0])['action'], 'record_only_continue_owned_worker')
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_launch_uses_only_fresh_compiler_caches_and_binds_owned_popen(self):
        scheduler = self.scheduler()
        child = MagicMock()
        child.pid = 7777
        child.poll.return_value = None
        with patch.object(study.subprocess, 'Popen', return_value=child) as popen:
            def exact(pid):
                return dict(self.fake_identity, pid=pid, argv=popen.call_args.args[0])
            with patch.object(study, 'identity', side_effect=exact):
                name = scheduler.launch('calibrate', device='cuda', gpu=2)
            environment = popen.call_args.kwargs['env']
        self.assertIs(scheduler.active[name]['child'], child)
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '2')
        for variable in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH'):
            self.assertTrue(Path(environment[variable]).is_relative_to(self.output / 'runtime'))
            self.assertTrue(Path(environment[variable]).is_dir())
        self.assertEqual(study.read(scheduler.workers / f'{name}.json')['identity']['pid'], 7777)

    def test_resume_device_dispatch_keeps_cpu_and_cuda_checkpoints_separate(self):
        import torch
        scheduler = self.scheduler()
        scheduler.p = dict(scheduler.p, tasks=['binary'], seeds=[42, 43])
        for seed, kind in ((42, 'cpu'), (43, 'cuda')):
            folder = self.output / 'heads/binary' / f's{seed}'
            folder.mkdir(parents=True)
            torch.save({'identity': {'device_type': kind}}, folder / 'epoch_resume.pt')
        launched = []
        def launch(phase, **values):
            launched.append(values)
            scheduler.active[str(values['seed'])] = dict(values)
        def poll():
            if launched:
                for values in launched:
                    study.write(self.output / 'heads/binary' / f's{values["seed"]}/selection.json', {})
                scheduler.active.clear()
        with patch.object(scheduler, 'cards', return_value=[dict(index=2, pids=[], free_gib=45.)]), \
             patch.object(scheduler, 'launch', side_effect=launch), patch.object(scheduler, 'poll', side_effect=poll), \
             patch.object(scheduler, 'state'), patch.object(study.time, 'sleep'):
            scheduler.tasks('fit')
        self.assertEqual({(values['seed'], values['device']) for values in launched}, {(42, 'cpu'), (43, 'cuda')})
        with patch('torch.load', side_effect=AssertionError('unchanged resume should use cached device')):
            self.assertEqual(scheduler.resume_device('fit', 'binary', 42), 'cpu')
        self.assertIsNone(scheduler.resume_device('evaluate', 'binary', 43))

    def test_cuda_resume_waits_instead_of_falling_back_to_cpu(self):
        import torch
        scheduler = self.scheduler()
        scheduler.p = dict(scheduler.p, tasks=['binary'], seeds=[42])
        path = self.output / 'heads/binary/s42/epoch_resume.pt'
        path.parent.mkdir(parents=True)
        torch.save({'identity': {'device_type': 'cuda'}}, path)
        with patch.object(scheduler, 'cards', return_value=[dict(index=2, pids=[987654], free_gib=45.)]), \
             patch.object(scheduler, 'launch') as launch, patch.object(scheduler, 'state'), \
             patch.object(study.time, 'sleep', side_effect=RuntimeError('one scheduler cycle')):
            with self.assertRaisesRegex(RuntimeError, 'one scheduler cycle'):
                scheduler.tasks('fit')
        launch.assert_not_called()

    def test_exit_75_retry_allowed_only_before_admission(self):
        scheduler = self.scheduler()
        child = MagicMock()
        child.poll.return_value = 75
        scheduler.active['retry'] = dict(child=child)
        scheduler.poll()
        self.assertFalse(scheduler.active)
        study.write(scheduler.workers / 'admitted.admitted.json', {'admitted': True})
        scheduler.active['admitted'] = dict(child=child)
        with self.assertRaisesRegex(RuntimeError, 'Owned worker failed'):
            scheduler.poll()

    def test_stop_only_targets_live_unreaped_owned_popen_children(self):
        scheduler = self.scheduler()
        live, exited = MagicMock(), MagicMock()
        live.poll.return_value, exited.poll.return_value = None, 0
        scheduler.active = {'live': {'child': live}, 'exited': {'child': exited}}
        scheduler.stop_owned()
        live.terminate.assert_called_once()
        exited.terminate.assert_not_called()
        self.assertFalse(scheduler.active)
        (self.output / 'STOP').touch()
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            scheduler.poll()

    def test_extraction_worker_launch_is_capped_by_remaining_chunks(self):
        from scripts.hanguard import clean_ablation_features as features
        scheduler = self.scheduler()
        cards = [dict(index=index, pids=[], free_gib=45.) for index in range(8)]
        launched = []
        def launch(phase, **values):
            launched.append((phase, values))
            scheduler.active[str(values['gpu'])] = {'gpu': values['gpu']}
        def poll():
            if launched:
                scheduler.active.clear()
        with patch.object(features, 'prepare'), patch.object(features, 'progress', return_value={'total_chunks': 37, 'complete_chunks': 35}), \
             patch.object(features, 'all_complete', side_effect=[False, True]), patch.object(features, 'finalize') as finalize, \
             patch.object(scheduler, 'cards', return_value=cards), patch.object(scheduler, 'launch', side_effect=launch), \
             patch.object(scheduler, 'poll', side_effect=poll), patch.object(scheduler, 'state'), patch.object(study.time, 'sleep'):
            scheduler.extract('train')
        self.assertEqual(len(launched), 2)
        self.assertEqual({item[1]['gpu'] for item in launched}, {0, 1})
        finalize.assert_called_once_with(self.output, 'train')


if __name__ == '__main__':
    unittest.main()
