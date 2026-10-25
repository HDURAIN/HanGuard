"""Executable controls for the existing fusion head before A/B diagnosis.

The synthetic task makes the first layer informative and pairs identical last
layer features with opposite labels. A correctly connected fusion head can
solve held-out examples, while repeated last-layer inputs cannot identify the
label. This checks trainability rather than merely checking finite gradients.
"""
from contextlib import contextmanager
import argparse
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from scripts.hanguard.repaired_heads import ExperimentHead
from scripts.hanguard import ab_diagnostic_experiment as experiment
from scripts.hanguard import remediated_dataset_contract as contract


@contextmanager
def _small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _complementary_examples(seed, pairs=64, hidden=16):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0., 1.]).repeat(pairs)
    first = torch.randn(2 * pairs, hidden, generator=generator) * .15
    sign = labels.mul(2).sub(1)
    first[:, 0] += 3 * sign
    first[:, 1] -= 3 * sign
    middle = [torch.randn(2 * pairs, hidden, generator=generator) for _ in range(2)]
    last = torch.randn(pairs, hidden, generator=generator).repeat_interleave(2, dim=0)
    raw = tuple(value[:, None, :] for value in (first, *middle, last))
    return raw, labels


def _inputs(raw, mode):
    return {
        'layers': raw if mode == 'fusion' else (raw[-1],),
        'mask': torch.ones(raw[0].shape[:2], dtype=torch.bool),
    }


def test_actual_fusion_learns_heldout_complementarity_with_matched_repeat_control():
    with _small_cpu_pool():
        train, y_train = _complementary_examples(102)
        heldout, y_heldout = _complementary_examples(907)
        heads = {}
        for mode in ('fusion', 'repeat_last'):
            torch.manual_seed(55)
            heads[mode] = ExperimentHead(16, mode=mode, width=32,
                                         layers=(1, 2, 3, 4), dropout=0.)
        initial_states = [head.state_dict() for head in heads.values()]
        assert initial_states[0].keys() == initial_states[1].keys()
        for key in initial_states[0]:
            assert torch.equal(initial_states[0][key], initial_states[1][key]), key
        assert sum(p.numel() for p in heads['fusion'].parameters()) == sum(
            p.numel() for p in heads['repeat_last'].parameters())

        initial_branch = [branch[0].weight.detach().clone()
                          for branch in heads['fusion'].layer_values]
        optimizers = {mode: torch.optim.Adam(head.parameters(), lr=.01)
                      for mode, head in heads.items()}
        initial_loss = {}
        branch_gradient_after_residual_update = None
        for step in range(80):
            for mode, head in heads.items():
                head.train()
                optimizer = optimizers[mode]
                optimizer.zero_grad(set_to_none=True)
                logits = head(_inputs(train, mode))['logits']
                loss = F.binary_cross_entropy_with_logits(logits, y_train)
                assert torch.isfinite(loss)
                if step == 0:
                    initial_loss[mode] = float(loss.detach())
                loss.backward()
                # Zero residual initialization legitimately blocks upstream
                # fusion gradients on step one. Audit the second backward.
                if mode == 'fusion' and step == 1:
                    branch_gradient_after_residual_update = [
                        float(branch[0].weight.grad.abs().sum())
                        for branch in head.layer_values]
                optimizer.step()

        assert branch_gradient_after_residual_update is not None
        assert all(value > 0 for value in branch_gradient_after_residual_update)
        for before, branch in zip(initial_branch, heads['fusion'].layer_values):
            assert not torch.equal(before, branch[0].weight.detach())
        accuracy = {}
        with torch.no_grad():
            for mode, head in heads.items():
                head.eval()
                train_logits = head(_inputs(train, mode))['logits']
                train_loss = float(F.binary_cross_entropy_with_logits(train_logits, y_train))
                if mode == 'fusion':
                    assert train_loss < .1 * initial_loss[mode]
                heldout_logits = head(_inputs(heldout, mode))['logits']
                assert torch.isfinite(heldout_logits).all()
                accuracy[mode] = float(((heldout_logits >= 0) == y_heldout.bool()).float().mean())
        assert accuracy['fusion'] >= .98, accuracy
        # Opposite labels share exactly the same last-layer features, so every
        # deterministic repeated-layer classifier must get one of each pair.
        assert accuracy['repeat_last'] == .5, accuracy
        assert accuracy['fusion'] - accuracy['repeat_last'] >= .45, accuracy


def test_actual_fusion_matches_unpadded_predictions_on_both_padding_sides():
    with _small_cpu_pool(), torch.no_grad():
        torch.manual_seed(133)
        head = ExperimentHead(16, mode='fusion', width=16,
                              layers=(1, 2, 3, 4), dropout=0.).eval()
        # Exercise all four branches, rather than an initial zero residual.
        head.fusion_output.weight.copy_(torch.eye(16))
        sequences = [tuple(torch.randn(1, length, 16) for _ in range(4))
                     for length in (2, 5, 3)]
        expected = torch.cat([head({'layers': raw,
                                   'mask': torch.ones(raw[0].shape[:2], dtype=torch.bool)})['logits']
                              for raw in sequences])
        for side in ('left', 'right'):
            mask = torch.zeros(3, 8, dtype=torch.bool)
            raw = [torch.randn(3, 8, 16) * 1e4 for _ in range(4)]
            for row, sequence in enumerate(sequences):
                length = sequence[0].shape[1]
                begin = 8 - length if side == 'left' else 0
                mask[row, begin:begin + length] = True
                for layer in range(4):
                    raw[layer][row, begin:begin + length] = sequence[layer][0]
            actual = head({'layers': tuple(raw), 'mask': mask})['logits']
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class _FullInputTokenizer:
    def __call__(self, texts, *, add_special_tokens, truncation):
        assert add_special_tokens is False
        assert truncation is False
        return {'input_ids': [[ord(character) for character in text] for text in texts]}


def _a_frame(role='A_train', split='train'):
    return pd.DataFrame([
        dict(base_id=f'{role}_{i}', group_id=f'{role}_group_{i}', prompt=f'完整输入{i}',
             source='wildguard_zh', prompt_harm_label='harmful' if i else 'unharmful',
             category_id=-1 if i else 0, type_label_mask=False,
             diagnostic_partition=role, split=split)
        for i in range(2)])


def _a_protocol(directory, frame):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / 'train.parquet', index=False)
    return dict(data_dir=str(directory), data_sha256={'train': contract.sha(directory / 'train.parquet')},
                rows={'train': len(frame)}, max_tokens=4096)


@pytest.mark.parametrize('split', ['B', 'test', 'formal_validation'])
def test_lora_loader_rejects_B_and_formal_holdouts_before_touching_files(split):
    with pytest.raises(ValueError, match='cannot access B or formal test'):
        experiment.load_a_split(split, _FullInputTokenizer(), {})


@pytest.mark.parametrize('role,original_split', [('B', 'train'), ('A_train', 'test')])
def test_lora_loader_rejects_misrouted_rows_even_with_matching_file_hash(tmp_path, role, original_split):
    frame = _a_frame(role, original_split)
    protocol = _a_protocol(tmp_path / 'data', frame)
    with pytest.raises(ValueError, match='outside its registered A partition|original-training identities'):
        experiment.load_a_split('train', _FullInputTokenizer(), protocol)


def test_lora_loader_preserves_binary_supervision_for_unknown_type_and_rejects_changed_input(tmp_path):
    protocol = _a_protocol(tmp_path / 'data', _a_frame())
    data = experiment.load_a_split('train', _FullInputTokenizer(), protocol)
    assert data['y'].tolist() == [0., 1.]
    assert data['meta'].category_id.tolist() == [0, -1]
    assert data['lengths'].tolist() == [5, 5]
    changed = _a_frame()
    changed.loc[1, 'prompt'] += '非法修改'
    changed.to_parquet(tmp_path / 'data/train.parquet', index=False)
    with pytest.raises(ValueError, match='input hash changed'):
        experiment.load_a_split('train', _FullInputTokenizer(), protocol)


def _registered_study(tmp_path):
    partition_dir = tmp_path / 'partition'
    lora_dir = partition_dir / 'lora_data'
    lora_dir.mkdir(parents=True)
    paths = {}
    for relative, role in [('lora_data/train.parquet', 'A_train'),
                           ('lora_data/validation.parquet', 'A_validation'), ('B.parquet', 'B')]:
        path = partition_dir / relative
        _a_frame(role).to_parquet(path, index=False)
        paths[relative] = path
    manifest = dict(policy_id='hanguard_train_internal_ab_grouped_v1',
        invariants={'final_test_enabled': False}, lora_data_dir=str(lora_dir),
        a_train_path=str(paths['lora_data/train.parquet']),
        a_validation_path=str(paths['lora_data/validation.parquet']), b_path=str(paths['B.parquet']),
        partitions={role: {'rows': 2} for role in ('A_train', 'A_validation', 'B')},
        artifact_sha256={relative: contract.sha(path) for relative, path in paths.items()})
    manifest_path = partition_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    model = tmp_path / 'model'
    model.mkdir()
    for filename in ('config.json', 'tokenizer.json', 'model.safetensors.index.json'):
        (model / filename).write_text('{}')
    output = tmp_path / 'study'
    experiment.initialize(argparse.Namespace(output=output, partition_manifest=manifest_path,
        model=model, epochs=3, token_budget=8192, seed=42, gpu=4))
    return output, paths


def test_registered_protocol_rejects_changed_partition_artifact(tmp_path):
    output, paths = _registered_study(tmp_path)
    assert experiment.load_protocol(output)['adapter_training_cohort'] == 'A'
    changed = _a_frame('B')
    changed.loc[0, 'prompt'] += '非法修改'
    changed.to_parquet(paths['B.parquet'], index=False)
    with pytest.raises(ValueError, match='Partition artifact changed'):
        experiment.load_protocol(output)


@pytest.mark.parametrize('field,value', [('epochs', 4), ('adapter_lr', .1), ('formal_test_enabled', True)])
def test_registered_protocol_rejects_setting_drift_even_when_state_is_replaced(tmp_path, field, value):
    output, _ = _registered_study(tmp_path)
    (output / 'state.json').write_text(json.dumps({'state': 'preflight_passed'}))
    protocol_path = output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol[field] = value
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Registered protocol or arms changed'):
        experiment.load_protocol(output)


def test_registered_A_path_must_match_partition_even_if_registration_digest_is_updated(tmp_path):
    output, _ = _registered_study(tmp_path)
    protocol_path = output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['data_dir'] = str(tmp_path / 'misrouted_B_data')
    protocol_path.write_text(json.dumps(protocol))
    registration_path = output / 'registration.json'
    registration = json.loads(registration_path.read_text())
    registration['protocol_sha256'] = contract.sha(protocol_path)
    registration_path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match='data directory differs from registered A partition'):
        experiment.load_protocol(output)
