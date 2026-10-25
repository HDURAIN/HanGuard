"""Independent CPU smoke tests for the registered sparse-readout execution path."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from scripts.hanguard import sparse_token_readout_experiment as experiment
from scripts.hanguard import sparse_token_readout_queue as guardian


SETTINGS = dict(seed=42, width=8, dropout=0., logical_batch_size=13,
    micro_token_budget=15, micro_max_rows=4, lr=.001, weight_decay=.001,
    grad_clip=5., epochs=2, max_epochs=2, bootstrap_draws=50,
    alpha=1.5, projection_chunk_size=512)


def test_real_heads_microbatch_accumulation_matches_whole_logical_batch():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    raw = torch.randn(7, 5, 2560)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    labels = torch.tensor([0., 1., 1., 0., 1., 0., 1.])
    heads, _ = experiment.make_heads(SETTINGS, 'cpu')
    for arm, original in heads.items():
        if arm != 'last_mlp':
            with torch.no_grad():
                original.pool_output.weight.copy_(torch.eye(SETTINGS['width']))
        dense, micro = copy.deepcopy(original), copy.deepcopy(original)
        dense.train(); micro.train()
        F.binary_cross_entropy_with_logits(experiment.forward(dense, arm, raw, mask)['logits'], labels).backward()
        for rows in experiment.split_microbatches(np.arange(7), mask.sum(1).numpy(), 10, 3):
            F.binary_cross_entropy_with_logits(experiment.forward(micro, arm, raw[rows], mask[rows])['logits'],
                labels[rows], reduction='sum').div(len(labels)).backward()
        for (name, expected), (_, observed) in zip(dense.named_parameters(), micro.named_parameters()):
            torch.testing.assert_close(expected.grad, observed.grad, atol=3e-7, rtol=3e-4, msg=f'{arm} {name}')


def fixture_study(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    output, source = tmp_path / 'study', tmp_path / 'source'
    output.mkdir(); (source / 'partition/folds').mkdir(parents=True)
    metadata = pd.DataFrame(dict(base_id=[f'row{i:03d}' for i in range(40)],
        group_id=[f'group{i:03d}' for i in range(40)], label=np.arange(40) % 2,
        source=['synthetic'] * 40, outer_fold=np.repeat(np.arange(5), 8),
        prompt_tokens=(np.arange(40) % 4) + 2))
    roles = ['outer_validation'] * 8 + ['inner_validation'] * 4 + ['inner_train'] * 28
    assignment = metadata[['base_id', 'group_id']].copy()
    assignment['role'], assignment['outer_fold'] = roles, 0
    assignment.to_parquet(source / 'partition/folds/fold_0_assignments.parquet', index=False)
    protocol = dict(settings=SETTINGS, source_study=str(source), token_cache=str(output / 'token_cache'),
        fold_layers={str(i): 21 for i in range(5)}, cache_layers=[21], stop_directories=[str(output)])
    experiment.dump(output / 'protocol.json', protocol)
    (output / 'token_cache').mkdir()
    experiment.dump(output / 'token_cache/manifest.json', dict(complete=True, synthetic=True))
    monkeypatch.setattr(experiment, 'load_protocol', lambda path: copy.deepcopy(protocol))
    class Store:
        def __init__(self, *args):
            self.metadata, self.lengths = metadata, metadata.prompt_tokens.to_numpy()
            self.manifest_sha256 = experiment.sha256(output / 'token_cache/manifest.json')
            rng = np.random.default_rng(17)
            self.raw = rng.normal(size=(40, 5, 2560)).astype(np.float32)
            self.raw[:, :, 0] += (metadata.label.to_numpy() * 2 - 1)[:, None] * 3

        def batch(self, rows, layer, device):
            assert layer == 21
            mask = np.arange(5)[None] < self.lengths[rows, None]
            return torch.from_numpy(self.raw[rows]).to(device), torch.from_numpy(mask).to(device)
    monkeypatch.setattr(experiment, 'TokenStore', Store)
    experiment.preflight(output, 'cpu')
    return output, protocol, metadata


def test_six_arm_fold_atomic_publication_and_guardian_validator(tmp_path, monkeypatch):
    output, protocol, metadata = fixture_study(tmp_path, monkeypatch)
    result = experiment.train_fold(output, 0, 'cpu')
    final = output / 'fold_0'
    assert result['complete'] and result['rows'] == 8
    assert result['row_identity']['fit']['rows'] == 28
    assert result['row_identity']['dev']['rows'] == 4
    assert result['row_identity']['held']['rows'] == 8
    assert not any((output / 'fold_attempts').iterdir())
    record = experiment.read(final / 'selection.json')
    assert set(record['thresholds']) == set(experiment.ALL_ARMS)
    assert set(record['best_epoch']) == set(experiment.ALL_ARMS)
    assert all(epoch in (1, 2) for epoch in record['best_epoch'].values())
    frame = pd.read_parquet(final / 'held_predictions.parquet')
    assert frame.base_id.tolist() == metadata.base_id.iloc[:8].tolist()
    assert set(frame.outer_fold) == {0}
    for arm in experiment.ALL_ARMS:
        np.testing.assert_array_equal(frame[f'prediction_{arm}'],
            frame[f'probability_{arm}'].to_numpy() >= frame[f'threshold_{arm}'].to_numpy())
    stage = guardian.RegisteredStage('fold_0', ('not-executed',),
        outputs=(str(final / 'complete.json'),), validator='complete_json', gpu_required=False)
    guardian.validate_stage(stage)
    # Completion resume is artifact-bound and does not fit anything anew.
    monkeypatch.setattr(experiment, 'TokenStore', lambda *args: pytest.fail('Completed fold retrained'))
    assert experiment.train_fold(output, 0, 'cpu') == result
    (final / 'heads.pt').write_bytes(b'corrupted weights')
    with pytest.raises(ValueError, match='artifact changed'):
        guardian.validate_stage(stage)
    with pytest.raises(ValueError, match='artifact changed'):
        experiment.train_fold(output, 0, 'cpu')


def test_STOP_during_real_fit_publishes_no_complete_fold(tmp_path, monkeypatch):
    output, _, _ = fixture_study(tmp_path, monkeypatch)
    original = experiment.forward
    calls = 0
    def stop_after_some_updates(*args):
        nonlocal calls
        calls += 1
        if calls == 7:
            (output / 'STOP').write_text('Independent cancellation')
        return original(*args)
    monkeypatch.setattr(experiment, 'forward', stop_after_some_updates)
    with pytest.raises(RuntimeError, match='Independent cancellation'):
        experiment.train_fold(output, 0, 'cpu')
    assert not (output / 'fold_0').exists()
    assert not any((output / 'fold_attempts').glob('*/complete.json'))


def test_uncommitted_final_directory_is_never_overwritten(tmp_path, monkeypatch):
    output, _, _ = fixture_study(tmp_path, monkeypatch)
    (output / 'fold_0').mkdir()
    with pytest.raises(ValueError, match='Uncommitted final'):
        experiment.train_fold(output, 0, 'cpu')


@pytest.mark.parametrize('field', ['passed', 'protocol_sha256', 'cache_manifest_sha256'])
def test_fit_requires_successful_same_protocol_and_same_cache_preflight(tmp_path, monkeypatch, field):
    output, _, _ = fixture_study(tmp_path, monkeypatch)
    gate = experiment.read(output / 'head_preflight.json')
    gate[field] = False if field == 'passed' else 'another-artifact'
    experiment.dump(output / 'head_preflight.json', gate)
    with pytest.raises(ValueError, match='same-protocol, same-cache'):
        experiment.train_fold(output, 0, 'cpu')
    assert not (output / 'fold_0').exists()
    assert not (output / 'fold_attempts').exists()


def test_full_20048_row_report_checks_paired_math_artifacts_and_evidence_identity(tmp_path, monkeypatch):
    output, source = tmp_path / 'study', tmp_path / 'source'
    output.mkdir(); (source / 'partition/folds').mkdir(parents=True)
    (source / 'lora_study/features/adapted').mkdir(parents=True)
    count = 20048
    metadata = pd.DataFrame(dict(base_id=[f'row{i:05d}' for i in range(count)],
        group_id=[f'group{i:05d}' for i in range(count)], label=np.arange(count) % 2,
        source=['synthetic'] * count, outer_fold=np.arange(count) % 5,
        prompt_tokens=np.full(count, 4)))
    metadata.to_parquet(source / 'lora_study/features/adapted/B.metadata.parquet', index=False)
    (output / 'token_cache').mkdir()
    experiment.dump(output / 'token_cache/manifest.json', dict(complete=True, synthetic=True))
    cache_digest = experiment.sha256(output / 'token_cache/manifest.json')
    protocol = dict(source_study=str(source), fold_layers={str(i): 21 for i in range(5)},
        token_cache=str(output / 'token_cache'),
        settings=dict(SETTINGS, bootstrap_draws=20), scope='Synthetic aggregation test only')
    experiment.dump(output / 'protocol.json', protocol)
    monkeypatch.setattr(experiment, 'load_protocol', lambda path: copy.deepcopy(protocol))
    expected_net = 0
    for fold in range(5):
        final = output / f'fold_{fold}'; final.mkdir()
        held_mask = metadata.outer_fold.to_numpy() == fold
        roles = np.where(held_mask, 'outer_validation', 'inner_train')
        roles[np.flatnonzero(~held_mask)[:4]] = 'inner_validation'
        assignments = metadata[['base_id', 'group_id']].copy()
        assignments['role'], assignments['outer_fold'] = roles, fold
        assignments.to_parquet(source / f'partition/folds/fold_{fold}_assignments.parquet', index=False)
        held = metadata.loc[held_mask].copy()
        for arm in experiment.ALL_ARMS:
            predicted = held.label.to_numpy().copy()
            errors = 10 if arm == 'repeat_last' else 2 if arm == 'global_query_entmax15' else 6
            predicted[:errors] = 1 - predicted[:errors]
            held[f'probability_{arm}'] = np.where(predicted, .9, .1)
            held[f'threshold_{arm}'] = .5
            held[f'prediction_{arm}'] = predicted
        expected_net += 8
        held.to_parquet(final / 'held_predictions.parquet', index=False)
        evidence = pd.DataFrame(dict(row=np.flatnonzero(held_mask), base_id=held.base_id.to_numpy()))
        for arm in experiment.ARMS:
            evidence[f'support_{arm}'] = 2 if arm == 'global_query_entmax15' else 4
            evidence[f'effective_tokens_{arm}'] = 2. if arm == 'global_query_entmax15' else 4.
        evidence.to_parquet(final / 'held_attention.parquet', index=False)
        for name in experiment.FOLD_ARTIFACTS:
            if not (final / name).exists():
                (final / name).write_bytes(b'Synthetic report fixture')
        record = dict(complete=True, protocol_sha256=experiment.sha256(output / 'protocol.json'),
            fold=fold, layer=21, rows=len(held), cache_manifest_sha256=cache_digest,
            row_identity=dict(held=dict(ordered_base_id_sha256=experiment.ordered_base_id_sha256(held.base_id.tolist()))),
            artifacts={name: experiment.sha256(final / name) for name in experiment.FOLD_ARTIFACTS})
        experiment.dump(final / 'complete.json', record)
    result = experiment.report(output)
    assert result['complete'] and result['rows'] == count and result['groups'] == count
    comparison = next(value for value in result['comparisons'] if
        value['candidate'] == 'global_query_entmax15' and value['baseline'] == 'repeat_last')
    assert comparison['net_correct'] == comparison['rescued'] == expected_net
    assert comparison['harmed'] == 0
    assert comparison['group_bootstrap']['accuracy']['delta'] == pytest.approx(expected_net / count)
    assert comparison['fixed_0_5_group_bootstrap']['accuracy']['delta'] == pytest.approx(expected_net / count)
    assert result['attention_summary']['global_query_entmax15']['median_support_fraction'] == .5
    guardian.validate_stage(guardian.RegisteredStage('report', ('not-executed',),
        outputs=(str(output / 'analysis/results.json'),), validator='complete_json', gpu_required=False))
    # Even a newly consistent artifact checksum cannot excuse wrong held IDs.
    path = output / 'fold_0/held_attention.parquet'
    evidence = pd.read_parquet(path); evidence.loc[0, 'base_id'] = 'different-held-id'
    evidence.to_parquet(path, index=False)
    record_path = path.with_name('complete.json')
    record = experiment.read(record_path)
    record['artifacts']['held_attention.parquet'] = experiment.sha256(path)
    experiment.dump(record_path, record)
    with pytest.raises(ValueError, match='registered held groups'):
        experiment.report(output)
