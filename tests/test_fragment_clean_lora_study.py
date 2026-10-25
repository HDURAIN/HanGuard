"""CPU checks for admission, adapter provenance, barriers and head resumption."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import fragment_clean_lora_study as study


@pytest.fixture(autouse=True)
def small_cpu_pool():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_foreign_compute_blocks_idle_card():
    cards = [dict(index=0, pids=[123], free_gib=47., utilization=0),
             dict(index=1, pids=[], free_gib=46., utilization=0),
             dict(index=2, pids=[], free_gib=20., utilization=0)]
    assert study.empty_cards(cards, [0, 1, 2], 36.) == [1]
    assert study.empty_cards(cards, [0], 1.) == []


def scheduler_fixture(tmp_path, monkeypatch, inventory_reader):
    monkeypatch.setattr(study, 'verify', lambda *_args, **_kwargs: dict(
        gpu_ids=list(range(8)), minimum_free_gib=36., poll_seconds=.001))
    return study.Scheduler(tmp_path, inventory_reader=inventory_reader,
        identity_reader=lambda pid: dict(pid=pid, uid=777), sleeper=lambda _seconds: None)


def test_foreign_arrival_is_record_only(tmp_path, monkeypatch):
    child = SimpleNamespace(terminate=lambda: pytest.fail('Foreign arrival stopped our worker'))
    scheduler = scheduler_fixture(tmp_path, monkeypatch,
        lambda: [dict(index=2, pids=[41, 99], free_gib=3.)])
    scheduler.active['owned'] = dict(worker_id='owned', gpu=2, identity=dict(pid=41), child=child)
    assert scheduler.cards()[0]['pids'] == [41, 99]
    scheduler.cards()
    receipts = list((tmp_path / 'foreign_arrivals').glob('*.json'))
    assert len(receipts) == 1
    assert study.read(receipts[0])['action'] == 'record_only_continue_owned_worker'
    assert scheduler.active['owned']['child'] is child


def test_inventory_failure_does_not_retire_active_worker(tmp_path, monkeypatch):
    def failed_inventory():
        raise RuntimeError('NVIDIA telemetry timeout')
    scheduler = scheduler_fixture(tmp_path, monkeypatch, failed_inventory)
    scheduler.active['owned'] = dict(identity=dict(pid=1), gpu=0)
    assert scheduler.cards() == []
    assert len(scheduler.active) == 1
    assert study.read(tmp_path / 'state.json')['state'] == 'gpu_inventory_unavailable'


def test_prelaunch_foreign_card_never_spawns(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path, monkeypatch,
        lambda: [dict(index=2, pids=[900], free_gib=47.)])
    scheduler.popen = lambda *_a, **_k: pytest.fail('Launched on occupied card')
    assert scheduler.launch(2, 'lora') is False


def test_worker_waits_for_ledger_registration_then_rechecks_foreign_pid(tmp_path, monkeypatch):
    worker_id = 'test'
    exact = dict(pid=55, uid=7, start_time_ticks=8, command_sha256='command', argv=['worker'], cwd='cwd')
    path = study.worker_ledger(tmp_path, worker_id)
    study.write(tmp_path / 'protocol.json', {'test': True})
    study.write(path, dict(identity=exact, protocol_sha256=study.sha(tmp_path / 'protocol.json'), gpu=2))
    monkeypatch.setattr(study, 'verify', lambda *_a, **_k: dict(gpu_ids=[2]))
    monkeypatch.setattr(study, 'identity', lambda _pid: exact)
    monkeypatch.setattr(study, 'inventory', lambda: [dict(index=2, pids=[999], free_gib=47.)])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    polls = []
    def publish_registration(_seconds):
        polls.append(True)
        study.write(path.with_suffix('.registration.json'), dict(ledger_sha256=study.sha(path)))
    monkeypatch.setattr(study.time, 'sleep', publish_registration)
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('CUDA started despite foreign PID'))
    assert study.admit_worker(tmp_path, worker_id) is False
    assert polls == [True]


def test_test_barrier_precedes_data_read(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, 'read_parquet', lambda *_a, **_k: pytest.fail('Read test payload before lock'))
    with pytest.raises(RuntimeError, match='Test barrier'):
        study.load_feature_data(tmp_path, 'test', {})
    with pytest.raises(RuntimeError, match='Test barrier'):
        study.merge_features(tmp_path, 'test')


def test_warmup_selection_stops_without_changing_selection(tmp_path):
    adapter = dict(actual_selected_phase=dict(phase_name='head_warmup', epoch=0),
                   adapter_delta=dict(nonzero_lora_b_tensors=0))
    with pytest.raises(study.NeedsReview, match='joint LoRA adaptation'):
        study.require_adapted_adapter(tmp_path, adapter)
    state = study.read(tmp_path / 'state.json')
    assert state['state'] == 'needs_review'
    assert state['actual_selected_phase']['phase_name'] == 'head_warmup'


def cache_fixture(tmp_path, monkeypatch):
    folder = tmp_path / 'readouts'
    (folder / 'caches').mkdir(parents=True)
    snapshot = folder / 'adapter_readout_runtime.py.snapshot'
    snapshot.write_bytes(Path(study.__file__).read_bytes())
    protocol = dict(schema=study.READOUT_SCHEMA, data_dir=str(tmp_path / 'data'),
        model_path='synthetic_cpu_model', dataset_release=dict(dataset_version='synthetic'),
        num_layers=32, hidden_size=6, adapter_path=str(tmp_path / 'adapter.pt'),
        adapter_sha256='adapter_sha', use_lora=True, rows={}, inputs={},
        settings=dict(seed=42, width=5, dropout=.2, batch_size=8, lr=.001,
            weight_decay=.007, grad_clip=.5, epochs=2, max_epochs=2,
            layers=[8, 16, 24, 32], bootstrap_draws=8),
        limitations=['Synthetic CPU fixture'])
    (tmp_path / 'data').mkdir()
    for split, count in (('train', 24), ('validation', 12), ('test', 12)):
        source = tmp_path / 'data' / f'{split}.parquet'
        source.write_text(f'{split} input identity')
        protocol['inputs'][str(source)] = study.sha(source)
        protocol['rows'][split] = count
    study.write(folder / 'protocol.json', protocol)
    monkeypatch.setattr(study, 'verify_readouts', lambda *_a, **_k: protocol)
    from scripts.hanguard.frozen_lora_layer_diagnostics import ordered_base_id_sha256
    for split, count in protocol['rows'].items():
        values = np.random.default_rng(count + len(split)).normal(size=(count, 32, 6)).astype(np.float16)
        metadata = pd.DataFrame(dict(base_id=[f'{split}_{k}' for k in range(count)],
            group_id=[f'{split}_group_{k // 2}' for k in range(count)],
            source=np.where(np.arange(count) % 3, 'source_a', 'source_b'),
            label=np.arange(count) % 2, prompt_tokens=np.arange(count) + 1))
        path, meta, audit = [folder / 'caches' / f'{split}.{ext}' for ext in ('npy', 'parquet', 'replay.json')]
        np.save(path, values)
        metadata.to_parquet(meta, index=False)
        study.write(audit, dict(passed=True))
        study.write(folder / 'caches' / f'{split}.manifest.json', dict(
            complete=True, split=split, cohort=split, representation='frozen_lora',
            shape=list(values.shape), path=str(path), metadata_path=str(meta), sha256=study.sha(path),
            metadata_sha256=study.sha(meta), protocol_sha256=study.sha(folder / 'protocol.json'),
            adapter_sha256=protocol['adapter_sha256'], adapter_path=protocol['adapter_path'],
            input_sha256=protocol['inputs'][str(tmp_path / 'data' / f'{split}.parquet')],
            pooling='last_valid_token', raw_block_outputs=True, truncation=False,
            backbone_trainable_parameters=0, replay_audit_path=str(audit), replay_audit_sha256=study.sha(audit),
            ordered_base_id_sha256=ordered_base_id_sha256(metadata.base_id)))
    return folder


def test_adapter_cache_rejects_base_and_other_adapter(tmp_path, monkeypatch):
    folder = cache_fixture(tmp_path, monkeypatch)
    path = folder / 'caches/train.manifest.json'
    valid = study.read(path)
    assert study.load_cache(folder, 'train')[0].shape == (24, 32, 6)
    for change in (dict(representation='base', adapter_sha256=None), dict(adapter_sha256='other')):
        study.write(path, dict(valid, **change))
        with pytest.raises(ValueError, match='cache identity mismatch'):
            study.load_cache(folder, 'train')


def test_five_arm_cpu_fit_evaluate_and_epoch_resume_keep_lora_identity(tmp_path, monkeypatch):
    folder = cache_fixture(tmp_path, monkeypatch)
    from scripts.hanguard import frozen_base_resumable_heads as resumable
    from scripts.hanguard import frozen_base_readout_heads as original
    prior_load = original.load_cache
    real_save = resumable._save_resume
    saved = []
    def interrupt_after_epoch(path, payload):
        real_save(path, payload)
        saved.append(Path(path))
        raise RuntimeError('Synthetic interruption after committed epoch')
    monkeypatch.setattr(resumable, '_save_resume', interrupt_after_epoch)
    with study.adapted_heads(folder) as (_heads, runner):
        with pytest.raises(RuntimeError, match='Synthetic interruption'):
            runner.fit(folder, device='cpu', extra_source_paths=[folder / 'adapter_readout_runtime.py.snapshot'])
    assert original.load_cache is prior_load
    assert saved and saved[0].name == 'layers_resume.pt'
    committed = torch.load(saved[0], map_location='cpu', weights_only=True)
    assert committed['completed_epochs'] == 1
    monkeypatch.setattr(resumable, '_save_resume', real_save)
    with study.adapted_heads(folder) as (heads, runner):
        runner.fit(folder, device='cpu', extra_source_paths=[folder / 'adapter_readout_runtime.py.snapshot'])
        heads.evaluate(folder, device='cpu')
    selection = study.read(folder / 'heads/selection.json')
    result = study.read(folder / 'analysis/results.json')
    assert selection['schema'] == study.READOUT_SCHEMA
    assert set(result['metrics']) == {'last', 'best_single', 'repeat_best', 'fusion_fixed', 'fusion_fixed_best_anchor'}
    for artifact in (selection, result):
        assert artifact['no_lora'] is False
        assert artifact['use_lora'] is True
        assert artifact['adapter_sha256'] == 'adapter_sha'
    assert study.read(folder / 'caches/train.manifest.json')['representation'] == 'frozen_lora'
    assert study.read(folder / 'analysis/complete.json')['complete'] is True
    assert (folder / 'heads/budget_2/fusion_resume.pt').exists()
    assert original.load_cache is prior_load


def test_chunk_plan_covers_exact_original_groups(tmp_path, monkeypatch):
    folder = tmp_path / 'readouts'
    area = folder / 'features/train'
    study.write(folder / 'protocol.json', {'fake': True})
    meta = area / 'metadata.parquet'
    meta.parent.mkdir(parents=True)
    meta.write_text('identity')
    p = dict(rows=dict(train=5), num_layers=32, hidden_size=6)
    monkeypatch.setattr(study, 'verify_readouts', lambda *_a, **_k: p)
    plan = dict(protocol_sha256=study.sha(folder / 'protocol.json'), split='train',
        shape=[5, 32, 6], rows=5, metadata_path=str(meta), metadata_sha256=study.sha(meta),
        groups=[[4, 3], [1, 2], [0]], chunks=[[0, 1], [2]])
    study.write(area / 'plan.json', plan)
    study.write(area / 'plan.registration.json', dict(plan_sha256=study.sha(area / 'plan.json')))
    assert study.verify_plan(folder, 'train') == plan
    assert study.chunk_rows(plan, 0) == [4, 3, 1, 2]
    plan['groups'][1] = [1, 3]
    study.write(area / 'plan.json', plan)
    study.write(area / 'plan.registration.json', dict(plan_sha256=study.sha(area / 'plan.json')))
    with pytest.raises(ValueError, match='cover every row/group exactly once'):
        study.verify_plan(folder, 'train')


def test_cpu_registration_binds_new_release_and_never_initializes_cuda(tmp_path, monkeypatch):
    from scripts.hanguard import remediated_dataset_contract as contract
    data, model = tmp_path / 'fragment_release', tmp_path / 'model'
    data.mkdir()
    model.mkdir()
    study.write(model / 'config.json', dict(model_type='qwen3_5',
        text_config=dict(num_hidden_layers=32, hidden_size=6)))
    study.write(model / 'tokenizer.json', {})
    study.write(model / 'model.safetensors.index.json', dict(weight_map=dict(fake='model.safetensors')))
    (model / 'model.safetensors').write_bytes(b'Only hash-bound CPU test bytes, never model-loaded')
    for split in study.SPLITS:
        harmful = np.array([False, True, False, True])
        pd.DataFrame(dict(base_id=[f'{split}_{i}' for i in range(4)],
            group_id=[f'{split}_g{i}' for i in range(4)], split=split, source='wildguard_zh',
            prompt=[f'{split} unique prompt {i}' for i in range(4)], prompt_tokens=5,
            category_id=harmful.astype(int), type_label_mask=harmful,
            prompt_harm_label=np.where(harmful, 'harmful', 'unharmful'))).to_parquet(data / f'{split}.parquet', index=False)
    study.write(data / 'manifest.json', dict(dataset_contract=contract.CONTRACT, dataset_version=data.name))
    study.write(data / 'split_policy.json', dict(no_resplitting=True))
    study.write(data / 'release_qa.json', dict(release_status='qa_complete_with_documented_limitations',
        manifest_sha256=study.sha(data / 'manifest.json'),
        split_sha256={split: study.sha(data / f'{split}.parquet') for split in study.SPLITS}))
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('CPU registration initialized CUDA'))
    args = study.parser().parse_args(['init', '--output', str(tmp_path / 'study'), '--data', str(data), '--model', str(model)])
    registered = study.initialize(args)
    assert registered['policy']['active_foreign'] == 'record_only'
    assert study.verify(args.output, full_hash=True)['schema'] == study.SCHEMA
    assert study.read(args.output / 'state.json')['training_started'] is False
    assert study.read(args.output / 'lora/protocol.json')['all_arms'] == ['E03']
    assert study.read(args.output / 'lora/protocol.json')['seeds'] == [42]
    source = data / 'train.parquet'
    source.write_bytes(source.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='input metadata changed'):
        study.verify(args.output)


def test_chunk_merge_preserves_row_order_and_lora_identity(tmp_path, monkeypatch):
    folder = cache_fixture(tmp_path, monkeypatch)
    protocol = study.verify_readouts(folder)
    values, metadata, manifest = study.load_cache(folder, 'train')
    expected = np.array(values)
    del values
    (folder / 'caches/train.manifest.json').unlink()
    plan = dict(split='train', protocol_sha256=study.sha(folder / 'protocol.json'),
        input_sha256=manifest['input_sha256'], rows=24, shape=[24, 32, 6],
        metadata_path=manifest['metadata_path'], metadata_sha256=manifest['metadata_sha256'],
        ordered_base_id_sha256=manifest['ordered_base_id_sha256'],
        groups=[list(range(12, 24)), list(range(12))], chunks=[[0], [1]])
    area = folder / 'features/train'
    study.write(area / 'plan.json', plan)
    study.write(area / 'plan.registration.json', dict(plan_sha256=study.sha(area / 'plan.json')))
    for number in range(2):
        ids = study.chunk_rows(plan, number)
        path = area / f'chunk{number}.npy'
        np.save(path, expected[ids])
        study.write(study.chunk_receipt(folder, 'train', number), dict(complete=True,
            split='train', chunk=number, worker_id=f'worker_{number}',
            protocol_sha256=study.sha(folder / 'protocol.json'), plan_sha256=study.sha(area / 'plan.json'),
            adapter_sha256=protocol['adapter_sha256'], row_indices=ids,
            path=str(path), sha256=study.sha(path), calibration_sha256='same_batch',
            original_batch_replay_passed=True))
    merged = study.merge_features(folder, 'train')
    assert merged['representation'] == 'frozen_lora'
    assert merged['adapter_sha256'] == 'adapter_sha'
    np.testing.assert_array_equal(np.load(merged['path']), expected)
    assert study.read(merged['replay_audit_path'])['every_row_exactly_once'] is True
