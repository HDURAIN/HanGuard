"""CPU-only registration and mocked orchestration for the maintained entrypoint."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.hanguard import current_training as training


@pytest.fixture
def environment(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    for name in set(training.code_files('binary') + training.code_files('category')):
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# code snapshot fixture: {name}\n')
    monkeypatch.setattr(training, 'ROOT', repo)
    data = tmp_path / 'data'; data.mkdir()
    for split in training.SPLITS:
        pd.DataFrame(dict(base_id=[f'{split}-{i}' for i in range(6)],
            group_id=[f'group-{split}-{i}' for i in range(6)], split=[split] * 6,
            source=['chinese_curated', 'chinese_curated', 'jailbench', 'chinese_curated', 'jailbench', 'jailbench'],
            prompt=[f'完整中文 {split} 样本{i}' for i in range(6)], prompt_tokens=[10] * 6,
            category_id=[str(i) for i in range(6)],
            prompt_harm_label=['unharmful'] + ['harmful'] * 5)).to_parquet(data / f'{split}.parquet', index=False)
    training.dump(data / 'descriptions.json', dict(classes=[dict(id=i, name=f'类{i}', description=f'定义{i}') for i in range(1, 6)]))
    model = tmp_path / 'model'; model.mkdir()
    training.dump(model / 'config.json', dict(text_config=dict(hidden_size=2560)))
    training.dump(model / 'tokenizer.json', dict(fixture=True))
    parent = tmp_path / 'parent'; run = parent / 'runs/E04_s42'; run.mkdir(parents=True)
    training.dump(parent / 'protocol.json', dict(model_path=str(model),
        model_config_sha256=training.sha(model / 'config.json'), tokenizer_sha256=training.sha(model / 'tokenizer.json')))
    training.dump(parent / 'arms.json', dict(E04=training.BINARY_ARMS['E04']))
    for name in ['adapter.pt', 'head.pt']:
        (run / name).write_bytes(name.encode())
    training.dump(run / 'selection.json', dict(arm='E04', seed=42, threshold=.5,
        protocol_sha256=training.sha(parent / 'protocol.json'), arms_sha256=training.sha(parent / 'arms.json'),
        checkpoint_hashes={name: training.sha(run / name) for name in ['head.pt', 'adapter.pt']}))
    def args(stage='binary', output=None, action='init', extra=()):
        return training.parser().parse_args([stage, action, '--output', str(output or tmp_path / 'study'),
            '--data', str(data), '--model', str(model), '--parent-run', str(run), '--gpus', '5', '6', '7', *extra])
    return SimpleNamespace(root=tmp_path, repo=repo, data=data, model=model, parent=run, args=args)


def test_dry_run_creates_nothing_and_cannot_spawn_workers(environment, monkeypatch, capsys):
    monkeypatch.setattr(training.subprocess, 'Popen', lambda *a, **k: pytest.fail('dry-run spawned a worker'))
    args = environment.args()
    training.main(['binary', 'dry-run', '--output', str(args.output), '--data', str(environment.data),
                   '--model', str(environment.model), '--gpus', '5'])
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] and not result['output_created'] and not result['gpu_workers_started']
    assert result['runs'] == 6 and not args.output.exists()


@pytest.mark.parametrize('stage,arms', [('binary', ['E03', 'E04']), ('category', training.CATEGORY_ARMS)])
def test_new_draft_registration_is_immutable_and_has_actual_dependencies(environment, stage, arms):
    args = environment.args(stage)
    protocol = training.init(args)
    assert protocol['state'] == 'draft' and protocol['all_arms'] == arms
    registered = training.register(args.output)
    assert registered['state'] == 'registered'
    before = (args.output / 'protocol.json').read_bytes()
    assert training.register(args.output) == registered
    assert (args.output / 'protocol.json').read_bytes() == before
    evidence = json.loads((args.output / 'registration.json').read_text())
    assert set(evidence['code_sha256']) == set(training.code_files(stage))
    assert not any('feature_fusion' in name or 'lora_fusion' in name for name in evidence['code_sha256'])
    for name, digest in evidence['code_sha256'].items():
        assert training.sha(args.output / 'registered_code' / name) == digest
    if stage == 'category':
        assert registered['head_width_by_arm'] == dict(last_mlp=275, learned_queries=128, description_queries=128)
        assert registered['early_stopping_patience'] == 8
    with pytest.raises(FileExistsError):
        training.init(args)


@pytest.mark.parametrize('target', ['data', 'protocol', 'audit', 'code', 'snapshot', 'arms'])
def test_registered_drift_is_rejected_before_workers(environment, target):
    args = environment.args(); training.init(args); training.register(args.output)
    paths = dict(data=environment.data / 'train.parquet', protocol=args.output / 'protocol.json',
        audit=args.output / 'data_audit.json', code=environment.repo / 'scripts/hanguard/repaired_study.py',
        snapshot=args.output / 'registered_code/scripts/hanguard/repaired_study.py', arms=args.output / 'arms.json')
    if target == 'protocol':
        value = json.loads(paths[target].read_text()); value['epochs'] += 1
        training.dump(paths[target], value)
    else:
        paths[target].write_text('changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        training.verify(args.output)


def test_category_parent_checkpoint_and_descriptions_are_bound(environment):
    args = environment.args('category'); training.init(args); training.register(args.output)
    original = (environment.parent / 'head.pt').read_bytes()
    (environment.parent / 'head.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        training.verify(args.output)
    (environment.parent / 'head.pt').write_bytes(original)
    (environment.data / 'descriptions.json').write_text('{}')
    with pytest.raises(ValueError, match='descriptions changed'):
        training.verify(args.output)


@pytest.mark.parametrize('problem', ['wildguard', 'contradictory_label', 'cross_split_group', 'cross_split_text', 'too_long'])
def test_data_scope_and_split_quality_are_checked_without_relabeling(environment, problem):
    path = environment.data / 'train.parquet'
    frame = pd.read_parquet(path)
    if problem == 'wildguard': frame.loc[1, 'source'] = 'wildguard_zh'
    elif problem == 'contradictory_label': frame.loc[1, 'category_id'] = '0'
    elif problem == 'cross_split_group': frame.loc[1, 'group_id'] = 'group-test-1'
    elif problem == 'cross_split_text': frame.loc[1, 'prompt'] = '完整中文 test 样本1'
    else: frame.loc[1, 'prompt_tokens'] = 9999
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        training.build_protocol(environment.args('category'))
    assert path.read_bytes() == before


def test_historical_protocol_cannot_be_registered_or_overwritten(environment):
    output = environment.root / 'historical'; output.mkdir()
    training.dump(output / 'protocol.json', dict(state='registered', data_dir=str(environment.data)))
    before = (output / 'protocol.json').read_bytes()
    with pytest.raises(ValueError, match='historical'):
        training.register(output)
    assert (output / 'protocol.json').read_bytes() == before


def install_binary_mock(environment, monkeypatch, missing=None):
    args = environment.args(); training.init(args); protocol = training.register(args.output)
    events = []
    def worker(study, label, arguments, gpu=None):
        training.verify(study)
        if arguments[0].endswith('report_repaired_study.py'):
            events.append(('report', None, None))
            training.dump(study / 'summary.json', dict(complete=True, completed_runs=6))
            return
        phase = arguments[1]
        arm, seed = arguments[arguments.index('--arm') + 1], int(arguments[arguments.index('--seed') + 1])
        events.append((phase, arm, seed))
        if phase == 'preflight':
            training.dump(study / 'preflight' / f'{arm}_s{seed}.json', dict(passed=True))
        if phase == 'train' and (arm, seed) != missing:
            directory = study / 'runs' / f'{arm}_s{seed}'; directory.mkdir(parents=True)
            for name in ['head.pt', 'adapter.pt']:
                (directory / name).write_bytes(f'{arm}-{seed}-{name}'.encode())
            training.dump(directory / 'selection.json', dict(arm=arm, seed=seed,
                protocol_sha256=training.sha(study / 'protocol.json'), arms_sha256=training.sha(study / 'arms.json'),
                checkpoint_hashes={name: training.sha(directory / name) for name in ['head.pt', 'adapter.pt']}))
        if phase == 'evaluate':
            barrier = json.loads((study / 'test_barrier.json').read_text())
            assert barrier['locked'] and len(barrier['selection_sha256']) == 6
            assert all(('train', arm, seed) in events for arm in protocol['all_arms'] for seed in protocol['seeds'])
    monkeypatch.setattr(training, 'run_worker', worker)
    return args.output, events


def test_binary_complete_queue_locks_every_selection_before_test(environment, monkeypatch):
    output, events = install_binary_mock(environment, monkeypatch)
    training.run_registered(output)
    assert sum(phase == 'train' for phase, _, _ in events) == 6
    assert sum(phase == 'evaluate' for phase, _, _ in events) == 6
    assert events[-1][0] == 'report'


def test_binary_missing_one_selection_prevents_all_test_evaluation(environment, monkeypatch):
    output, events = install_binary_mock(environment, monkeypatch, missing=('E04', 44))
    with pytest.raises(FileNotFoundError):
        training.run_registered(output)
    assert not (output / 'test_barrier.json').exists()
    assert not any(phase == 'evaluate' for phase, _, _ in events)


def test_category_launch_uses_registered_cache_and_queue_without_manual_gate(environment, monkeypatch):
    args = environment.args('category'); training.init(args); training.register(args.output)
    events = []
    def worker(study, label, arguments, gpu=None):
        training.verify(study)
        events.append((label, arguments, gpu))
        if label == 'primary_queue': training.dump(study / 'queue_status.json', dict(stage='complete'))
    monkeypatch.setattr(training, 'run_worker', worker)
    training.run_registered(args.output)
    assert [event[0] for event in events] == ['extract', 'primary_queue']
    assert events[0][1][1] == 'extract' and events[0][2] == 5
    assert not (args.output / 'training_code_ready.json').exists()


def test_resume_rejects_silent_setting_changes_and_stage_mismatch(environment):
    args = environment.args(); training.init(args); training.register(args.output)
    before = (args.output / 'protocol.json').read_bytes()
    with pytest.raises(ValueError, match='immutable registered settings'):
        training.main(['binary', 'run', '--output', str(args.output), '--resume', '--epochs', '10'])
    with pytest.raises(ValueError, match='stage differs'):
        training.main(['category', 'register', '--output', str(args.output)])
    assert (args.output / 'protocol.json').read_bytes() == before


def test_current_binary_rows_are_registered_but_legacy_counts_are_preserved(environment):
    from scripts.hanguard import repaired_study
    args = environment.args(extra=['--max-tokens', '64'])
    training.init(args); training.register(args.output)
    protocol, _ = repaired_study.json_protocol(args.output)
    class Tokenizer:
        def __call__(self, texts, **kwargs): return dict(input_ids=[[1, 2] for _ in texts])
    loaded = repaired_study.load_split('train', Tokenizer(), protocol)
    assert len(loaded['encoded']) == 6
    protocol.pop('training_entrypoint')
    with pytest.raises(ValueError, match='expected 62155'):
        repaired_study.load_split('train', Tokenizer(), protocol)
