"""Admission, phase barriers, and concurrent shard ownership invariants."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.hanguard import qwen3guard_supervised_study_v2 as study


class StudyTests(unittest.TestCase):
    def test_launch_identity_waits_for_exec_and_stability(self):
        empty = dict(pid=123, command_sha256=hashlib.sha256(b'').hexdigest())
        real = dict(pid=123, command_sha256='populated')
        with patch.object(study.common, 'identity', side_effect=[empty, real, real]), \
                patch.object(study.time, 'sleep'):
            self.assertEqual(study.launched_identity(123), real)

    def test_launch_identity_reports_early_exit(self):
        with patch.object(study.common, 'identity', return_value=None):
            self.assertIsNone(study.launched_identity(123))

    def test_shared_gpu_requires_budget_plus_headroom(self):
        for phase, minimum in [('extract', 22.), ('fit', 8.), ('evaluate', 8.)]:
            with self.subTest(phase=phase):
                self.assertFalse(study.eligible({'free_gib': minimum - .01, 'pids': []}, phase))
                self.assertTrue(study.eligible({'free_gib': minimum, 'pids': [123]}, phase))

    def test_shifted_bundles_never_duplicate_live_shards(self):
        todo = [dict(phase='extract', split='train', shards=[2, 3, 4, 5]),
                dict(phase='extract', split='train', shards=[6, 7, 8, 9])]
        active = [dict(phase='extract', split='train', shards=[0, 1, 2, 3]),
                  dict(phase='extract', split='train', shards=[4, 5, 6, 7])]
        result = study.available_jobs(todo, active, {})
        self.assertEqual(result, [dict(phase='extract', split='train', shards=[8, 9])])
        self.assertEqual(todo[0]['shards'], [2, 3, 4, 5])

    def test_failed_shards_remain_blocked_after_bundle_shift(self):
        todo = [dict(phase='extract', split='validation', shards=[3, 4, 5, 6])]
        failures = {'extract:validation:4': {}, 'extract:validation:5': {}}
        result = study.available_jobs(todo, [], failures)
        self.assertEqual(result[0]['shards'], [3, 6])

    def test_live_fit_and_failed_fit_not_relaunched(self):
        jobs = [dict(phase='fit', task='binary', seed=s) for s in (42, 43, 44)]
        self.assertEqual(study.available_jobs(jobs, [jobs[0]], {'fit:binary:43': {}}), jobs[2:])

    def test_fit_blocks_test_preparation(self):
        features, training = Mock(), Mock()
        current = dict(features={s: {'complete': True} for s in ('train', 'validation')},
            training={'pending_jobs': [dict(phase='fit', task='category', seed=42)]})
        with patch.object(study, 'module', side_effect=lambda kind: features if kind == 'features' else training):
            self.assertEqual(study.pending(Path('/unused'), current), current['training']['pending_jobs'])
        features.prepare.assert_not_called()
        training.lock_selections.assert_not_called()

    def test_all_selection_lock_precedes_test_preparation(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            events = []
            features, training = Mock(), Mock()
            training.lock_selections.side_effect = lambda _: events.append('lock')
            features.prepare.side_effect = lambda _, split: events.append('prepare_' + split)
            current = dict(features={s: {'complete': True} for s in ('train', 'validation')},
                           training={'pending_jobs': []})
            with patch.object(study, 'module', side_effect=lambda kind: features if kind == 'features' else training), \
                    patch.object(study, 'extraction_jobs', return_value=[]):
                study.pending(output, current)
            self.assertEqual(events, ['lock', 'prepare_test'])

    def test_evaluation_serialized(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / 'selection_lock.json').write_text('{}')
            jobs = [dict(phase='evaluate', task='category', seed=s) for s in (42, 43, 44)]
            current = dict(features={s: {'complete': True} for s in ('train', 'validation', 'test')},
                           training={'pending_jobs': jobs})
            self.assertEqual(study.pending(output, current), jobs[:1])

    def test_no_training_before_both_feature_splits_complete(self):
        current = dict(features={'train': {'complete': True}, 'validation': {'complete': False}},
                       training={'pending_jobs': [dict(phase='fit', task='binary', seed=42)]})
        sentinel = [dict(phase='extract', split='validation', shards=[1])]
        with patch.object(study, 'extraction_jobs', return_value=sentinel) as mocked:
            self.assertEqual(study.pending(Path('/unused'), current), sentinel)
        mocked.assert_called_once_with(Path('/unused'), 'validation')

    def test_verify_rejects_changed_data_even_with_cached_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            parent, data = output / 'parent', output / 'data'
            parent.mkdir()
            data.mkdir()
            (parent / 'protocol.json').write_text('{}')
            target = data / 'train.parquet'
            target.write_bytes(b'original')
            protocol = dict(schema=study.SCHEMA, output=str(output), budgets=study.BUDGETS,
                sources={}, parent=str(parent), parent_protocol_sha256=study.common.sha(parent / 'protocol.json'),
                parent_data_dir=str(data), data_sha256={'train': study.common.sha(target)}, model_files={})
            study.write(output / 'protocol.json', protocol)
            study.write(output / 'protocol_registration.json',
                        {'protocol_sha256': study.common.sha(output / 'protocol.json')})
            study.verify(output)
            # Some shared filesystems coalesce timestamps on adjacent tiny
            # writes. Change size too, so this checks portable stat invalidation.
            target.write_bytes(b'changed input payload')
            with self.assertRaisesRegex(ValueError, 'Registered input changed'):
                study.verify(output)


if __name__ == '__main__':
    unittest.main()
