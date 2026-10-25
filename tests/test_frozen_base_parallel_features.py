"""CPU-only independent checks for original-group sharding and GPU ownership."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import frozen_base_parallel_features as features
from scripts.hanguard import frozen_base_readout_study as base


class TinyTokenizer:
    pad_token_id = 0
    def __call__(self, prompts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return {'input_ids': [[100 + int(p.split(':')[0])] * int(p.split(':')[1]) for p in prompts]}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'study'
    root.mkdir()
    data = root / 'data'
    data.mkdir()
    rows = {'train': 12, 'validation': 4, 'test': 4}
    lengths = [5, 1, 5, 3, 8, 2, 3, 1, 7, 9, 4, 6]
    for split, count in rows.items():
        frame = pd.DataFrame(dict(base_id=[f'{split}_{i:02d}' for i in range(count)],
            group_id=[f'{split}_g{i}' for i in range(count)], source=['wildguard_zh'] * count,
            prompt=[f'{i}:{lengths[i]}' for i in range(count)],
            prompt_harm_label=['harmful' if i % 2 else 'unharmful' for i in range(count)],
            category_id=[-1 if i % 2 else 0 for i in range(count)],
            type_label_mask=[False] * count, split=[split] * count))
        frame.to_parquet(data / f'{split}.parquet', index=False)
    protocol = dict(rows=rows, num_layers=2, hidden_size=3, data_dir=str(data),
        inputs={str(data / f'{s}.parquet'): base.sha(data / f'{s}.parquet') for s in rows},
        extraction=dict(token_budget=16, max_micro=2, pad_multiple=1, max_tokens=4096),
        parallel_features=dict(group_chunk_size=2), code_sha256={str(Path(features.__file__).resolve()): base.sha(features.__file__)})
    features.dump(root / 'protocol.json', protocol)
    features.dump(root / 'state.json', dict(state='registered', formal_test_opened=False))
    monkeypatch.setattr(base, 'load_protocol', lambda study, **kwargs: protocol)
    features.prepare_split(root, 'train', tokenizer=TinyTokenizer())
    return root, protocol


def make_selection(root):
    checkpoint = root / 'selected_head.pt'
    checkpoint.write_bytes(b'independently selected validation head')
    features.dump(root / 'selection_lock.json', dict(complete=True, test_opened=False,
        protocol_sha256=base.sha(root / 'protocol.json'), artifacts={checkpoint.name: base.sha(checkpoint)}))


def make_calibration(root, worker_id, plan, data, capture, gpu):
    value = features.global_calibration(capture, data, None, {},
        [np.asarray(g, dtype=np.int64) for g in plan['groups']])
    value.update(worker_id=worker_id, runtime_gpu=dict(physical_gpu=gpu, device_name='CPU fixture'),
        protocol_sha256=base.sha(root / 'protocol.json'),
        plan_sha256=base.sha(features.split_directory(root, 'train') / 'batch_plan.json'))
    path = features.worker_directory(root) / f'{worker_id}.calibration.json'
    features.dump(path, value)
    return path


def complete_chunks(root, protocol, monkeypatch):
    prepared, plan, data = features.load_prepared(root, 'train', protocol=protocol)
    expected = torch.arange(plan['rows'] * 6, dtype=torch.float16).reshape(plan['rows'], 2, 3)
    groups_seen = []
    def make_batch(data, indices, tokenizer, extraction):
        groups_seen.append(np.asarray(indices).tolist())
        return indices
    monkeypatch.setattr(base.legacy, 'make_batch', make_batch)
    capture = lambda indices: expected[indices].clone()
    calibrations = {w: make_calibration(root, w, plan, data, capture, gpu)
                    for w, gpu in (('worker_a', 2), ('worker_b', 4))}
    receipts = []
    while features.has_work(root, 'train'):
        worker = ('worker_a', 'worker_b')[len(receipts) % 2]
        claim = features.claim_chunk(root, 'train', worker, prepared, plan)
        receipts.append(features.extract_chunk(root, 'train', worker, claim, prepared, plan,
            data, None, capture, calibrations[worker], protocol))
    return expected.numpy(), receipts, plan, groups_seen


def test_preparation_preserves_full_original_groups_and_token_order(prepared):
    root, protocol = prepared
    ready, plan, data = features.load_prepared(root, 'train', protocol=protocol)
    groups = base.legacy.microbatches(np.argsort(data['lengths'], kind='stable'), data['lengths'], 16, 2, 1)
    assert plan['groups'] == [g.tolist() for g in groups]
    assert np.array_equal(np.concatenate(groups), np.argsort(data['lengths'], kind='stable'))
    assert sorted(np.concatenate(groups).tolist()) == list(range(12))
    assert data['encoded'][4] == [104] * 8
    assert ready['complete'] and ready['chunks'] > 1
    for chunk in plan['chunks']:
        assert chunk['group_stop'] - chunk['group_start'] <= 2
    assert not (root / 'caches').exists()


@pytest.mark.parametrize('which', ['token_ids.npy', 'offsets.npy', 'metadata.parquet'])
def test_prepared_byte_tampering_rejected(prepared, which):
    root, protocol = prepared
    path = features.split_directory(root, 'train') / which
    path.write_bytes(path.read_bytes() + b'tampering')
    with pytest.raises(ValueError, match='bytes changed'):
        features.load_prepared(root, 'train', protocol=protocol)


@pytest.mark.parametrize('which', ['groups', 'chunks'])
def test_self_hashed_rebatch_or_group_omission_still_rejected(prepared, which):
    root, protocol = prepared
    folder = features.split_directory(root, 'train')
    plan = features.read(folder / 'batch_plan.json')
    if which == 'groups':
        plan['groups'][0] = list(reversed(plan['groups'][0]))
    else:
        plan['chunks'][0]['group_stop'] -= 1
    features.dump(folder / 'batch_plan.json', plan)
    manifest = features.read(folder / 'prepared_manifest.json')
    manifest['artifacts']['batch_plan.json'] = manifest['plan_sha256'] = base.sha(folder / 'batch_plan.json')
    features.dump(folder / 'prepared_manifest.json', manifest)
    with pytest.raises(ValueError, match='groups|Chunk plan'):
        features.load_prepared(root, 'train', protocol=protocol)


def test_test_barrier_precedes_every_prompt_or_prepared_payload_access(prepared, monkeypatch):
    root, protocol = prepared
    reads = []
    monkeypatch.setattr(base, 'load_frame', lambda *args: reads.append('forbidden prompt read'))
    for callback in (lambda: features.prepare_split(root, 'test', tokenizer=TinyTokenizer()),
                     lambda: features.load_prepared(root, 'test', protocol=protocol),
                     lambda: features.merge_split(root, 'test')):
        with pytest.raises(ValueError, match='locked'):
            callback()
    assert reads == [] and not features.split_directory(root, 'test').exists()


def test_locked_test_preparation_keeps_lock_hash_and_rejects_changed_checkpoint(prepared):
    root, protocol = prepared
    make_selection(root)
    features.prepare_split(root, 'test', tokenizer=TinyTokenizer())
    assert features.read(root / 'state.json')['formal_test_opened'] is True
    features.load_prepared(root, 'test', protocol=protocol)
    (root / 'selected_head.pt').write_bytes(b'changed after test prepared')
    with pytest.raises(ValueError, match='artifact changed'):
        features.load_prepared(root, 'test', protocol=protocol)


def test_disjoint_work_stealing_complete_receipts_and_canonical_merge(prepared, monkeypatch):
    root, protocol = prepared
    expected, receipts, plan, groups_seen = complete_chunks(root, protocol, monkeypatch)
    assert len(receipts) == len(plan['chunks'])
    assert len({r['worker_id'] for r in receipts}) == 2
    assert all(group in plan['groups'] for group in groups_seen)
    assert features.split_complete(root, 'train')
    manifest = features.merge_split(root, 'train')
    assert manifest['contributing_gpu_workers'] == 2
    assert base.cache_complete(root, protocol, 'train')
    assert np.array_equal(np.load(manifest['path']), expected)
    audit = features.read(manifest['replay_audit_path'])
    assert audit['every_row_exactly_once'] and not audit['incomplete_chunk_rows_reused']
    assert len(audit['cross_gpu_same_global_group_fp16_sha256']) == 3


def test_merge_rejects_incomplete_chunks_before_writing_cache(prepared):
    root, _ = prepared
    assert features.has_work(root, 'train')
    with pytest.raises(ValueError, match='must be complete'):
        features.merge_split(root, 'train')
    assert not (root / 'caches').exists()


@pytest.mark.parametrize('fault', ['bytes', 'rows', 'nan', 'gpu_calibration'])
def test_complete_receipts_cannot_hide_feature_errors_or_cross_gpu_math(prepared, monkeypatch, fault):
    root, protocol = prepared
    _, receipts, _, _ = complete_chunks(root, protocol, monkeypatch)
    receipt = receipts[1]
    receipt_path = features.chunk_directory(root, 'train', receipt['chunk']['chunk_id']) / 'receipt.json'
    if fault == 'bytes':
        path = Path(receipt['path'])
        path.write_bytes(path.read_bytes() + b'changed bytes')
    elif fault == 'rows':
        rows = np.load(receipt['row_indices_path'])
        np.save(receipt['row_indices_path'], rows[::-1])
        receipt['row_indices_sha256'] = base.sha(receipt['row_indices_path'])
        features.dump(receipt_path, receipt)
    elif fault == 'nan':
        values = np.load(receipt['path'])
        values[0, 0, 0] = float('nan')
        np.save(receipt['path'], values)
        receipt['sha256'] = base.sha(receipt['path'])
        features.dump(receipt_path, receipt)
    else:
        calibration = features.read(receipt['calibration_path'])
        calibration['records'][0]['feature_sha256'] = 'f' * 64
        features.dump(receipt['calibration_path'], calibration)
        # Update every worker_b receipt so this is a true consistency check,
        # independent of file hash tamper detection.
        for current in receipts:
            if current['worker_id'] == receipt['worker_id']:
                current['calibration_sha256'] = base.sha(current['calibration_path'])
                features.dump(features.chunk_directory(root, 'train', current['chunk']['chunk_id']) / 'receipt.json', current)
    with pytest.raises((ValueError, RuntimeError), match='hash changed|row coverage|differs across'):
        features.merge_split(root, 'train')
    assert not (root / 'caches' / 'train.manifest.json').exists()


def test_worker_stop_interrupts_without_publishing_a_partial_chunk(prepared, monkeypatch):
    root, protocol = prepared
    ready, plan, data = features.load_prepared(root, 'train', protocol=protocol)
    claim = features.claim_chunk(root, 'train', 'stopped', ready, plan)
    features.worker_directory(root).mkdir(parents=True, exist_ok=True)
    (features.worker_directory(root) / 'stopped.STOP').write_text('ownership changed')
    monkeypatch.setattr(base.legacy, 'make_batch', lambda *args: pytest.fail('GPU batch after STOP'))
    with pytest.raises(RuntimeError, match='ownership changed'):
        features.extract_chunk(root, 'train', 'stopped', claim, ready, plan, data, None,
            lambda batch: pytest.fail('CUDA forward after STOP'), root / 'unused.json', protocol)
    assert not (features.chunk_directory(root, 'train', claim['chunk']['chunk_id']) / 'receipt.json').exists()


def worker_ledger(root, worker_id, *, split='train', pid=None, ticks=None):
    actual = features.inventory.process_identity(os.getpid())
    identity = dict(actual, pid=pid if pid is not None else actual['pid'],
                    start_ticks=ticks if ticks is not None else actual['start_time_ticks'])
    if ticks is not None:
        identity['start_time_ticks'] = ticks
    ledger = dict(worker_id=worker_id, split=split, process_identity=identity,
        root_protocol_sha256=base.sha(root / 'protocol.json'))
    path = features.worker_directory(root) / f'{worker_id}.json'
    features.dump(path, ledger)
    features.dump(path.with_suffix('.registration.json'), dict(ledger_sha256=base.sha(path)))
    return ledger


def test_retirement_requires_exact_worker_exit_and_preserves_partial_audit(prepared, monkeypatch):
    root, protocol = prepared
    ready, plan, _ = features.load_prepared(root, 'train', protocol=protocol)
    ledger = worker_ledger(root, 'lost')
    claim = features.claim_chunk(root, 'train', 'lost', ready, plan)
    partial = Path(claim['directory']) / 'features.partial.npy'
    partial.write_bytes(b'incomplete rows must never be adopted')
    with pytest.raises(ValueError, match='still live'):
        features.abandon_worker(root, 'train', 'lost')
    assert partial.exists()
    monkeypatch.setattr(features, '_same_live_process', lambda identity: False)
    retirement = features.abandon_worker(root, 'train', 'lost')
    assert retirement['retired_chunk_ids'] == [claim['chunk']['chunk_id']]
    assert not partial.exists()
    retired = features.read(features.chunk_directory(root, 'train', claim['chunk']['chunk_id']) / 'claim.json')
    assert (Path(retired['directory']) / 'features.partial.npy').read_bytes().startswith(b'incomplete rows')
    new = features.claim_chunk(root, 'train', 'replacement', ready, plan)
    assert new['chunk'] == claim['chunk'] and new['attempt'] == 2
    assert not any(Path(new['directory']).iterdir())


def test_dead_claim_is_reissued_fresh_and_never_reads_partial_rows(prepared, monkeypatch):
    root, protocol = prepared
    ready, plan, _ = features.load_prepared(root, 'train', protocol=protocol)
    first = features.claim_chunk(root, 'train', 'first', ready, plan)
    (Path(first['directory']) / 'features.partial.npy').write_bytes(b'partial')
    monkeypatch.setattr(features, '_same_live_process', lambda identity: False)
    second = features.claim_chunk(root, 'train', 'second', ready, plan)
    assert first['chunk'] == second['chunk'] and second['attempt'] == 2
    assert (Path(first['directory']) / 'features.partial.npy').exists()
    assert not any(Path(second['directory']).iterdir())


def bind_fixture(root, protocol, monkeypatch):
    worker_id = 'bound'
    command = [str(Path(features.__file__).resolve()), 'feature-worker', '--study', str(root), '--worker-id', worker_id, '--split', 'train']
    raw = ('\0'.join(command) + '\0').encode()
    actual = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=123,
        command_sha256=hashlib.sha256(raw).hexdigest(), argv=command, cwd=str(features.ROOT))
    ledger = dict(worker_id=worker_id, physical_gpu=2, CUDA_VISIBLE_DEVICES='2', split='train',
        process_identity=dict(actual, start_ticks=123), command=command,
        root_protocol_sha256=base.sha(root / 'protocol.json'), script_sha256=base.sha(features.__file__))
    features.dump(root / 'parallel_resource_protocol.json', {'registered': 'CPU fixture'})
    resource_sha = base.sha(root / 'parallel_resource_protocol.json')
    features.dump(root / 'parallel_resource_registration.json', {'protocol_sha256': resource_sha})
    ledger['resource_protocol_sha256'] = resource_sha
    path = features.worker_directory(root) / f'{worker_id}.json'
    features.dump(path, ledger)
    features.dump(path.with_suffix('.registration.json'), dict(ledger_sha256=base.sha(path)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setattr(features.inventory, 'process_identity', lambda pid: actual)
    monkeypatch.setattr(features.inventory, 'gpu_inventory', lambda: [dict(index=2, pids=[])])
    return worker_id, actual, ledger, path


def test_worker_binding_accepts_only_exact_registered_card_and_birth(prepared, monkeypatch):
    root, protocol = prepared
    worker_id, _, ledger, _ = bind_fixture(root, protocol, monkeypatch)
    assert features.bind_worker(root, worker_id, 'train') == ledger


@pytest.mark.parametrize('fault', ['birth', 'uid', 'argv', 'cwd', 'raw_command', 'foreign', 'split', 'cvd', 'resource', 'registration'])
def test_worker_binding_fails_closed_before_cuda_on_identity_or_ownership_fault(prepared, monkeypatch, fault):
    root, protocol = prepared
    worker_id, actual, ledger, path = bind_fixture(root, protocol, monkeypatch)
    if fault == 'birth':
        actual['start_time_ticks'] += 1
    elif fault in ('uid', 'argv', 'cwd', 'raw_command'):
        key = 'command_sha256' if fault == 'raw_command' else fault
        actual[key] = {'uid': os.getuid() + 1, 'argv': ['other'], 'cwd': '/other', 'raw_command': 'changed'}[fault]
    elif fault == 'foreign':
        monkeypatch.setattr(features.inventory, 'gpu_inventory', lambda: [dict(index=2, pids=[314159])])
    elif fault == 'cvd':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    elif fault == 'split':
        ledger['split'] = 'test'
        features.dump(path, ledger)
        features.dump(path.with_suffix('.registration.json'), dict(ledger_sha256=base.sha(path)))
    elif fault == 'resource':
        features.dump(root / 'parallel_resource_protocol.json', {'tampered': True})
    else:
        features.dump(path.with_suffix('.registration.json'), dict(ledger_sha256='bad'))
    with pytest.raises(ValueError):
        features.bind_worker(root, worker_id, 'train')


def test_retirement_retains_only_verified_complete_chunks(prepared, monkeypatch):
    root, protocol = prepared
    _, receipts, _, _ = complete_chunks(root, protocol, monkeypatch)
    worker_ledger(root, 'worker_a')
    monkeypatch.setattr(features, '_same_live_process', lambda identity: False)
    result = features.abandon_worker(root, 'train', 'worker_a')
    assert result['retired_chunk_ids'] == []
    assert features.split_complete(root, 'train')
    receipt = next(r for r in receipts if r['worker_id'] == 'worker_a')
    Path(receipt['path']).write_bytes(Path(receipt['path']).read_bytes() + b'corrupted complete')
    with pytest.raises(ValueError, match='Committed chunk changed'):
        features.abandon_worker(root, 'train', 'worker_a')


def test_actual_cuda_readiness_follows_binding_without_loading_model_or_data(prepared, monkeypatch):
    root, _ = prepared
    events = []
    monkeypatch.setattr(features, 'bind_worker', lambda *args: events.append('bind') or {'physical_gpu': 2})
    monkeypatch.setattr(torch.cuda, 'init', lambda: events.append('cuda_init'))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('cuda_sync'))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'CPU fake card')
    monkeypatch.setattr(torch, 'zeros', lambda size, device: events.append(('allocate', device)) or torch.tensor([0.]))
    monkeypatch.setattr(features.subprocess, 'run', lambda *args, **kwargs: type('Result', (), {'stdout': 'GPUfixture,name,driver'})())
    result = features.cuda_startup(root, 'ready', 'train')
    assert events == ['bind', 'cuda_init', ('allocate', 'cuda'), 'cuda_sync']
    assert result['actual_cuda_allocation_verified']
    assert result['model_or_data_used_before_full_hash'] is False


def test_worker_rejects_full_content_hash_before_any_token_or_model_use(prepared, monkeypatch):
    root, _ = prepared
    events = []
    monkeypatch.setattr(features, 'cuda_startup', lambda *args: events.append('bound_real_cuda') or {})
    def full(study, *, full_hash=False):
        assert full_hash and events == ['bound_real_cuda']
        events.append('full_hash')
        raise ValueError('Model bytes changed')
    monkeypatch.setattr(base, 'load_protocol', full)
    monkeypatch.setattr(features, 'load_prepared', lambda *args, **kwargs: pytest.fail('Token data before successful hash'))
    monkeypatch.setattr(base, 'load_frozen_backbone', lambda *args: pytest.fail('Model before successful hash'))
    with pytest.raises(ValueError, match='Model bytes changed'):
        features.run_worker(root, 'hash_failure', 'train')
    assert events == ['bound_real_cuda', 'full_hash']
