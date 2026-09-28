"""CPU-only orchestration checks; workers, checkpoints and inputs are all mocks."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts.hanguard import primary_category_queue as queue_module


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    code = root / 'scripts/hanguard'
    code.mkdir(parents=True)
    for name in ['primary_category_queue.py', 'primary_category_study.py',
                 'primary_category_report.py', 'multilabel_heads.py', 'multilabel_study.py',
                 'primary_category_contract.py']:
        (code / name).write_text('# frozen mock implementation\n')
    monkeypatch.setattr(queue_module, 'ROOT', root)
    data = tmp_path / 'data'
    data.mkdir()
    for split in ['train', 'validation', 'test']:
        (data / f'{split}.parquet').write_bytes(f'unchanged {split} inputs'.encode())
    description = code / 'category_descriptions.json'
    description.write_text('{"classes": []}')
    study = tmp_path / 'study'
    study.mkdir()
    protocol = dict(all_arms=list(queue_module.ARMS), seeds=[42], epochs=12,
                    gpu_ids=[6, 7], data_dir=str(data),
                    data_sha256={s: queue_module.sha(data / f'{s}.parquet')
                                 for s in ['train', 'validation', 'test']},
                    description_file=str(description),
                    description_sha256=queue_module.sha(description),
                    new_multilabel_annotations_used=False)
    (study / 'protocol.json').write_text(json.dumps(protocol))
    (study / 'data_audit.json').write_text('{"passed": true}')
    return SimpleNamespace(root=root, code=code, data=data, study=study,
                           protocol=protocol, description=description)


class MockQueue(queue_module.Queue):
    def __init__(self, study, *, fail=None, omit=None, after_first_preflight=None):
        super().__init__(study)
        self.events = []
        self.seed_events = []
        self.fail = fail
        self.omit = omit
        self.after_first_preflight = after_first_preflight

    def run(self, action, arm=None, gpu=None, seed=42):
        self.check()
        self.events.append((action, arm))
        self.seed_events.append((action, arm, seed, gpu))
        if (action, arm) == self.fail:
            raise RuntimeError('mock worker failed')
        if action == 'train' and arm != self.omit and (arm, seed) != self.omit:
            directory = self.study / 'runs' / f'{arm}_s{seed}'
            directory.mkdir(parents=True)
            (directory / 'head.pt').write_bytes(arm.encode())
            (directory / 'selection.json').write_text(json.dumps(dict(
                protocol_sha256=self.protocol_sha,
                checkpoint_sha256=queue_module.sha(directory / 'head.pt'))))
        elif action == 'evaluate':
            barrier = json.loads((self.study / 'test_barrier.json').read_text())
            assert barrier['locked'] is True
            assert barrier['protocol_sha256'] == self.protocol_sha
            assert barrier['test_dataset_sha256'] == self.protocol['data_sha256']['test']
            assert len(barrier['selection_sha256']) == 3 * len(self.protocol['seeds'])
            for selected_seed in self.protocol['seeds']:
                for selected_arm in queue_module.ARMS:
                    relative = f'runs/{selected_arm}_s{selected_seed}/selection.json'
                    assert barrier['selection_sha256'][relative] == queue_module.sha(self.study / relative)
                    assert any(event[:3] == ('train', selected_arm, selected_seed) for event in self.seed_events)
        elif action == 'report':
            (self.study / 'report.md').write_text('Completed mock report')
        if action == 'preflight' and arm == 'last_mlp' and self.after_first_preflight:
            self.after_first_preflight()


def test_all_three_selections_precede_any_evaluation(environment):
    q = MockQueue(environment.study)
    q.execute()
    first_test = min(q.events.index(('evaluate', arm)) for arm in queue_module.ARMS)
    assert all(q.events.index(('train', arm)) < first_test for arm in queue_module.ARMS)
    assert len([event for event in q.events if event[0] == 'train']) == 3
    assert all(event[0] not in {'annotate', 'annotate_full', 'export_labels'} for event in q.events)
    state = json.loads((environment.study / 'queue_status.json').read_text())
    assert state['stage'] == 'complete'
    assert (environment.study / 'report.md').exists()


def test_successful_worker_without_selection_cannot_unlock_test(environment):
    q = MockQueue(environment.study, omit='description_queries')
    with pytest.raises(FileNotFoundError):
        q.execute()
    assert not (environment.study / 'test_barrier.json').exists()
    assert not any(action == 'evaluate' for action, _ in q.events)


@pytest.mark.parametrize('failure', [('preflight', 'last_mlp'), ('train', 'learned_queries'),
                                     ('evaluate', 'last_mlp'), ('report', None)])
def test_main_records_worker_failure_and_never_reports_complete(environment, monkeypatch, failure):
    q = MockQueue(environment.study, fail=failure)
    monkeypatch.setattr(queue_module, 'Queue', lambda study: q)
    monkeypatch.setattr(sys, 'argv', ['primary_category_queue', '--output', str(environment.study)])
    with pytest.raises(RuntimeError, match='mock worker failed'):
        queue_module.main()
    state = json.loads((environment.study / 'queue_status.json').read_text())
    assert state['stage'] == 'failed' and state['error_type'] == 'RuntimeError'
    if failure[0] in {'preflight', 'train'}:
        assert not (environment.study / 'test_barrier.json').exists()
        assert not any(action == 'evaluate' for action, _ in q.events)


@pytest.mark.parametrize('target', ['protocol', 'data', 'description', 'code'])
def test_metadata_or_code_drift_blocks_work_before_training(environment, target):
    paths = dict(protocol=environment.study / 'protocol.json',
                 data=environment.data / 'train.parquet',
                 description=environment.description,
                 code=environment.code / 'multilabel_heads.py')
    def change():
        paths[target].write_text('changed after first preflight')
    q = MockQueue(environment.study, after_first_preflight=change)
    with pytest.raises(ValueError, match='changed'):
        q.execute()
    assert q.events == [('preflight', 'last_mlp')]
    assert not (environment.study / 'test_barrier.json').exists()


def test_failed_data_audit_prevents_all_workers(environment):
    (environment.study / 'data_audit.json').write_text('{"passed": false}')
    q = MockQueue(environment.study)
    with pytest.raises(ValueError, match='audit must pass'):
        q.execute()
    assert q.events == []


def test_audit_or_registered_code_changes_cannot_be_resumed_as_same_execution(environment):
    q = MockQueue(environment.study, fail=('preflight', 'last_mlp'))
    with pytest.raises(RuntimeError):
        q.execute()
    (environment.study / 'data_audit.json').write_text('{"passed": true, "changed": true}')
    resumed = MockQueue(environment.study)
    with pytest.raises(ValueError, match='different execution'):
        resumed.execute()
    assert resumed.events == []


@pytest.mark.parametrize('omit', [None, ('learned_queries', 43)])
def test_all_nine_seeded_runs_precede_test_and_use_three_gpus(environment, omit):
    protocol = dict(environment.protocol, seeds=[42, 43, 44], gpu_ids=[5, 6, 7], epochs=40,
                    early_stopping_patience=8, early_stopping_min_delta=1e-4,
                    experiment_scope='two_source_full_primary_category')
    (environment.study / 'protocol.json').write_text(json.dumps(protocol))
    q = MockQueue(environment.study, omit=omit)
    if omit is not None:
        with pytest.raises(FileNotFoundError):
            q.execute()
        assert not (environment.study / 'test_barrier.json').exists()
        assert not any(event[0] == 'evaluate' for event in q.seed_events)
        return
    q.execute()
    trained = [event for event in q.seed_events if event[0] == 'train']
    assert len(trained) == 9
    assert {(event[1], event[2]) for event in trained} == {(arm, seed) for arm in queue_module.ARMS for seed in [42, 43, 44]}
    assert {event[3] for event in trained} == {5, 6, 7}
    first_test = next(index for index, event in enumerate(q.seed_events) if event[0] == 'evaluate')
    assert all(index < first_test for index, event in enumerate(q.seed_events) if event[0] == 'train')
    assert len([event for event in q.seed_events if event[0] == 'evaluate']) == 9
