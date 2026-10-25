"""CPU protocol checks for the cross-LoRA-seed fixed-layer replication."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import multiseed_fixed_layer_replication as replication
from scripts.hanguard.frozen_lora_layer_diagnostics import dump, ordered_base_id_sha256, sha256


def test_feature_arms_are_fixed_and_repeat_last_is_an_exact_four_slot_control():
    values = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    fixed = replication.select_arm_features(values, "dynamic_fixed")
    repeat = replication.select_arm_features(values, "dynamic_repeat_last")

    assert replication.FIXED_LAYERS == (8, 16, 24, 32)
    assert torch.equal(fixed, values)
    assert repeat.shape == fixed.shape
    for slot in range(4):
        assert torch.equal(repeat[:, slot], values[:, 3])
    with pytest.raises(ValueError, match="four registered layers"):
        replication.select_arm_features(values[:, :3], "dynamic_fixed")


def test_matched_heads_have_identical_topology_initialization_and_live_first_step_gradients():
    heads, audit = replication.build_matched_replication_heads(
        hidden_size=7, width=6, gate_width=5, seed=43, device="cpu"
    )
    assert set(heads) == set(replication.ARMS)
    assert audit["parameter_counts"]["dynamic_fixed"] == audit["parameter_counts"]["dynamic_repeat_last"]
    assert audit["initial_state_sha256"]["dynamic_fixed"] == audit["initial_state_sha256"]["dynamic_repeat_last"]
    assert heads["dynamic_fixed"].classifier.weight.data_ptr() != heads["dynamic_repeat_last"].classifier.weight.data_ptr()

    raw = torch.randn(12, 4, 7)
    for arm, head in heads.items():
        head.zero_grad(set_to_none=True)
        loss = head(replication.select_arm_features(raw, arm))["logits"].square().mean()
        loss.backward()
        assert head.gate_query.weight.grad.abs().sum() > 0
        assert head.gate_key.weight.grad.abs().sum() > 0
        assert all(projector[0].weight.grad.abs().sum() > 0 for projector in head.layer_values)


def test_checkpoint_choice_uses_validation_bce_only_and_ties_choose_earlier_epoch():
    history = [
        {"split": "validation", "epoch": 1, "validation_bce": {
            "dynamic_repeat_last": .31, "dynamic_fixed": .30,
        }},
        {"split": "validation", "epoch": 2, "validation_bce": {
            "dynamic_repeat_last": .29, "dynamic_fixed": .30,
        }},
        {"split": "validation", "epoch": 3, "validation_bce": {
            "dynamic_repeat_last": .32, "dynamic_fixed": .28,
        }},
    ]
    result = replication.choose_validation_checkpoints(history)
    assert result["dynamic_repeat_last"] == {"epoch": 2, "validation_bce": .29}
    assert result["dynamic_fixed"] == {"epoch": 3, "validation_bce": .28}

    contaminated = history + [{
        "split": "test", "epoch": 4,
        "validation_bce": {"dynamic_repeat_last": 0, "dynamic_fixed": 0},
    }]
    with pytest.raises(ValueError, match="validation-only"):
        replication.choose_validation_checkpoints(contaminated)


def test_selected_capture_hooks_only_four_layers_and_uses_each_rows_last_true_token():
    class AddBlock(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, hidden):
            return (hidden + self.value,)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(AddBlock(index) for index in range(1, 33))

        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            hidden = input_ids.float()[..., None].repeat(1, 1, 2)
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return hidden

    backbone = Backbone()
    capture = replication.SelectedTokenBlockCapture(backbone)
    assert len(capture.handles) == 4
    batch = {
        'input_ids': torch.tensor([[2, 7, 5, 9], [8, 3, 4, 1]]),
        'attention_mask': torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]]),
    }
    values = capture(batch).float()
    expected_tokens = torch.tensor([5, 1], dtype=torch.float32)
    for slot, layer in enumerate(replication.FIXED_LAYERS):
        expected = expected_tokens + sum(range(1, layer + 1))
        assert torch.equal(values[:, slot, 0], expected)
    capture.close()
    assert capture.handles == []


def test_four_layer_cache_validation_rejects_wrong_layout_nonfinite_and_row_order(tmp_path):
    path = tmp_path / "features.npy"
    base_ids = ["a", "b", "c"]
    values = np.arange(3 * 4 * 6, dtype=np.float16).reshape(3, 4, 6)
    np.save(path, values)
    identity = ordered_base_id_sha256(base_ids)
    result = replication.validate_four_layer_cache(path, 3, 6, identity, base_ids)
    assert result["shape"] == [3, 4, 6]

    with pytest.raises(ValueError, match="shape/dtype"):
        replication.validate_four_layer_cache(path, 3, 7, identity, base_ids)
    with pytest.raises(ValueError, match="row identity"):
        replication.validate_four_layer_cache(path, 3, 6, identity, list(reversed(base_ids)))
    values[1, 2, 3] = np.nan
    np.save(path, values)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        replication.validate_four_layer_cache(path, 3, 6, identity, base_ids)


def test_head_checkpoint_contract_is_anchored_to_protocol_and_validation_selection(tmp_path):
    output = tmp_path / 'study'
    directory = output / 'heads/adapter_s42/seed_42'
    directory.mkdir(parents=True)
    protocol_path = output / 'protocol.json'
    protocol_path.write_text('{"registered":true}\n', encoding='utf-8')
    protocol = {
        'hidden_size': 7,
        'architecture': {'width': 6, 'gate_width': 5, 'dropout': .1, 'fusion_init_scale': 1e-3},
        'adapters': {'42': {'adapter_sha256': 'a' * 64}},
    }
    heads, audit = replication.build_matched_replication_heads(7, 6, 5, 42)
    records = {}
    for arm, head in heads.items():
        checkpoint = {
            'state_dict': head.state_dict(), 'adapter_seed': 42, 'head_seed': 42,
            'arm': arm, 'layers': list(replication.FIXED_LAYERS), 'hidden_size': 7,
            'width': 6, 'gate_width': 5, 'dropout': .1, 'fusion_init_scale': 1e-3,
            'adapter_sha256': 'a' * 64, 'best_epoch': 2, 'best_validation_bce': .2,
            'protocol_sha256': sha256(protocol_path),
        }
        path = directory / f'{arm}.pt'
        replication.frozen.atomic_torch_save(checkpoint, path)
        records[arm] = {
            'threshold': .5, 'validation_bce': .2, 'best_epoch': 2,
            'checkpoint_sha256': sha256(path),
        }
    predictions = directory / 'validation_predictions.parquet'
    pd.DataFrame({'base_id': ['x']}).to_parquet(predictions, index=False)
    selection = {
        'split': 'validation', 'adapter_seed': 42, 'head_seed': 42,
        'layers': list(replication.FIXED_LAYERS), 'arms': records,
        'matched_initialization': audit,
        'validation_predictions_sha256': sha256(predictions),
    }
    dump(directory / 'selection.json', selection)
    assert replication._validate_head_selection(output, protocol, 42, 42) == selection

    bad_path = directory / 'dynamic_fixed.pt'
    bad = torch.load(bad_path, map_location='cpu', weights_only=True)
    bad['width'] = 99
    replication.frozen.atomic_torch_save(bad, bad_path)
    selection['arms']['dynamic_fixed']['checkpoint_sha256'] = sha256(bad_path)
    dump(directory / 'selection.json', selection)
    with pytest.raises(ValueError, match='Checkpoint contract changed'):
        replication._validate_head_selection(output, protocol, 42, 42)


def test_test_barrier_is_required_and_hashes_adapter_code_data_and_every_head(tmp_path):
    output = tmp_path / "study"
    output.mkdir()
    protocol = output / "protocol.json"
    adapter = output / "adapter.pt"
    code = output / "implementation.py"
    checkpoint_a = output / "head_a.pt"
    checkpoint_b = output / "head_b.pt"
    test_data = output / "test.parquet"
    for path, content in (
        (protocol, b"protocol"), (adapter, b"adapter"), (code, b"code"),
        (checkpoint_a, b"head-a"), (checkpoint_b, b"head-b"), (test_data, b"test"),
    ):
        path.write_bytes(content)
    artifacts = {
        "protocol": protocol, "adapter/s42": adapter, "code": code,
        "head/s42/42/repeat": checkpoint_a, "head/s42/42/fixed": checkpoint_b,
    }
    identity = {"adapter_seeds": [42, 43, 44], "head_seeds": [42, 43, 44, 45, 46]}

    with pytest.raises(FileNotFoundError, match="barrier"):
        replication.verify_replication_test_barrier(output, artifacts, identity, test_data)
    barrier = replication.create_replication_test_barrier(
        output, artifacts, identity, test_data
    )
    assert barrier["state"] == "locked"
    assert set(barrier["artifacts"]) == set(artifacts)
    assert replication.verify_replication_test_barrier(
        output, artifacts, identity, test_data
    )["test_dataset_sha256"]

    checkpoint_b.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact.*changed"):
        replication.verify_replication_test_barrier(output, artifacts, identity, test_data)


def test_source_adapter_registry_requires_three_validation_selected_exact_checkpoints(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    source_protocol = {
        'data_sha256': {'train': 'train-hash', 'validation': 'validation-hash'},
    }
    dump(source / 'protocol.json', source_protocol)
    protocol_hash = sha256(source / 'protocol.json')
    for seed in replication.ADAPTER_SEEDS:
        directory = source / 'runs' / f'E03_s{seed}'
        directory.mkdir(parents=True)
        adapter = directory / 'adapter.pt'
        adapter.write_bytes(f'adapter-{seed}'.encode())
        dump(directory / 'selection.json', {
            'arm': 'E03', 'seed': seed, 'protocol_sha256': protocol_hash,
            'train_sha256': 'train-hash', 'validation_sha256': 'validation-hash',
            'validation_bce': .1 + seed / 1000, 'best_epoch': {'epoch': 3},
            'checkpoint_hashes': {'adapter.pt': sha256(adapter)},
        })
    records = replication._adapter_records(source, source_protocol)
    assert set(records) == {'42', '43', '44'}
    assert all(len(record['adapter_sha256']) == 64 for record in records.values())

    selection_path = source / 'runs/E03_s44/selection.json'
    selection = replication._json(selection_path)
    selection['validation_sha256'] = 'wrong'
    dump(selection_path, selection)
    with pytest.raises(ValueError, match='valid validation-selected adapter'):
        replication._adapter_records(source, source_protocol)


def test_cross_adapter_distribution_treats_adapter_as_replication_unit():
    gains = {
        42: {"accuracy": .01, "balanced_accuracy": .02, "bce": -.03},
        43: {"accuracy": .03, "balanced_accuracy": .04, "bce": -.01},
        44: {"accuracy": -.01, "balanced_accuracy": .00, "bce": .02},
    }
    result = replication.adapter_gain_distribution(gains)
    assert result["adapters"] == 3
    assert result["unit"] == "independent E03 LoRA seed"
    assert result["metrics"]["accuracy"]["mean"] == pytest.approx(.01)
    assert result["metrics"]["accuracy"]["positive_fraction"] == pytest.approx(2 / 3)
    assert result["metrics"]["bce"]["improved_fraction"] == pytest.approx(2 / 3)
    assert result["adapter_seeds"] == [42, 43, 44]
    head_result = replication.head_seed_gain_distribution(gains)
    assert head_result['unit'].startswith('head seed')
    assert 'adapter_seeds' not in head_result
