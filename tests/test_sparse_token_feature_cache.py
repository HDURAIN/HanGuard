"""Ragged full-input extraction and immutable commit controls, CPU only."""
import copy
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch import nn

from scripts.hanguard import sparse_token_feature_cache as cache


def test_offsets_preserve_every_token_and_reject_empty_inputs():
    lengths = np.asarray([2, 5, 1], dtype=np.int32)
    offsets = cache.row_offsets(lengths)
    assert offsets.dtype == np.int64
    np.testing.assert_array_equal(offsets, [0, 2, 7, 8])
    cache.validate_offsets(offsets, lengths, 8)
    for changed in (np.asarray([0, 2, 7, 9], dtype=np.int64), offsets.astype(np.int32)):
        with pytest.raises(ValueError, match='offsets'):
            cache.validate_offsets(changed, lengths, 8)
    for bad in ([], [0, 1], [-1, 2], [[1, 2]]):
        with pytest.raises(ValueError, match='valid token'):
            cache.row_offsets(bad)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_pack_omits_padding_on_both_sides_and_preserves_token_order(side):
    lengths = [2, 4, 1]
    hidden = torch.full((3, 6, 3), float('nan'))
    mask = torch.zeros((3, 6), dtype=torch.bool)
    wanted = []
    for row, length in enumerate(lengths):
        start = 6 - length if side == 'left' else 0
        values = torch.arange(length * 3).reshape(length, 3).float() + 100 * row
        hidden[row, start:start + length] = values
        mask[row, start:start + length] = True
        wanted.append(values)
    packed = cache.pack_valid(hidden, mask)
    assert packed.dtype == torch.float16 and packed.device.type == 'cpu'
    torch.testing.assert_close(packed, torch.cat(wanted).half())
    assert torch.isfinite(packed).all()


def test_pack_masks_noncontiguous_positions_and_rejects_empty_rows():
    hidden = torch.arange(24).reshape(2, 4, 3).float()
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    torch.testing.assert_close(cache.pack_valid(hidden, mask), hidden[mask].half())
    mask[1] = False
    with pytest.raises(ValueError, match='valid tokens'):
        cache.pack_valid(hidden, mask)
    with pytest.raises(ValueError, match='shapes'):
        cache.pack_valid(hidden, torch.ones(2, 3))


class _Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, value):
        return value + self.index


class _Backbone(nn.Module):
    def __init__(self, blocks=4):
        super().__init__()
        self.layers = nn.ModuleList([_Block(i) for i in range(1, blocks + 1)])
        self.calls = 0

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        assert use_cache is False
        self.calls += 1
        value = torch.stack((input_ids.float(), position_ids.float()), dim=-1)
        for layer in self.layers:
            value = layer(value)
        return value


def test_capture_returns_actual_raw_selected_blocks_and_removes_hooks():
    backbone = _Backbone()
    capture = cache.TokenBlockCapture(backbone, (1, 3))
    batch = dict(input_ids=torch.tensor([[91, 2, 3], [4, 92, 93]]),
                 attention_mask=torch.tensor([[0, 1, 1], [1, 0, 0]]))
    result = capture(batch)
    assert result.shape == (3, 2, 2)
    np.testing.assert_array_equal(result[:, 0], [[3, 1], [4, 2], [5, 1]])
    np.testing.assert_array_equal(result[:, 1], [[8, 6], [9, 7], [10, 6]])
    assert capture.mask is None and not capture.values
    capture.close()
    assert not any(layer._forward_hooks for layer in backbone.layers)
    with pytest.raises(ValueError, match='unique'):
        cache.TokenBlockCapture(backbone, (3, 1))


def _small_cache(tmp_path):
    offsets = cache.row_offsets([2, 1, 3])
    raw = cache.raw_memmap(tmp_path / 'raw.bin', (6, 2, 3), 'w+')
    raw[:] = 0
    groups = [[1, 0], [2]]
    first = np.arange(18).reshape(3, 2, 3).astype(np.float16)
    cache.write_group(raw, first, groups[0], offsets)
    raw.flush()
    identity = dict(adapter='fixed', ids='fixed')
    progress = dict(identity=identity, next_group=1, rows=2,
        commits=[dict(group=0, rows=2, tokens=3, sha256=cache.group_digest(raw, groups[0], offsets))])
    return raw, offsets, groups, identity, progress, first


def test_ragged_random_row_writes_and_restart_preserve_committed_prefix(tmp_path):
    raw, offsets, groups, identity, progress, first = _small_cache(tmp_path)
    np.testing.assert_array_equal(raw[offsets[1]:offsets[2]], first[:1])
    np.testing.assert_array_equal(raw[offsets[0]:offsets[1]], first[1:])
    prefix = cache.group_digest(raw, groups[0], offsets)
    assert cache.verify_journal(progress, identity, raw, groups, offsets) == 1
    # Interrupted next-group writes are allowed, since they were not committed.
    raw[offsets[2]:offsets[3]] = 73
    assert cache.verify_journal(progress, identity, raw, groups, offsets) == 1
    cache.write_group(raw, np.full((3, 2, 3), 12, np.float16), groups[1], offsets)
    assert prefix == cache.group_digest(raw, groups[0], offsets)
    reopened = cache.raw_memmap(tmp_path / 'raw.bin', raw.shape, 'r')
    assert cache.verify_journal(progress, identity, reopened, groups, offsets) == 1


def test_restart_rejects_changed_committed_bytes_and_scientific_identity(tmp_path):
    raw, offsets, groups, identity, progress, _ = _small_cache(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        cache.verify_journal(progress, dict(adapter='different'), raw, groups, offsets)
    raw[0, 0, 0] += 1
    with pytest.raises(ValueError, match='Committed feature bytes changed'):
        cache.verify_journal(progress, identity, raw, groups, offsets)


@pytest.mark.parametrize('mutation', ['next_group', 'group', 'rows', 'tokens'])
def test_restart_rejects_noncontiguous_or_misreported_journal(tmp_path, mutation):
    raw, offsets, groups, identity, progress, _ = _small_cache(tmp_path)
    changed = copy.deepcopy(progress)
    if mutation == 'next_group':
        changed['next_group'] = 2
    else:
        changed['commits'][0][mutation] += 1
    with pytest.raises(ValueError, match='journal|ordering/extent'):
        cache.verify_journal(changed, identity, raw, groups, offsets)


def test_raw_cache_rejects_npy_header_and_trailing_or_missing_bytes(tmp_path):
    array = np.arange(36).reshape(6, 2, 3).astype(np.float16)
    np.save(tmp_path / 'wrong.bin', array, allow_pickle=False)
    with pytest.raises(ValueError, match='NPY headers'):
        cache.raw_memmap(tmp_path / 'wrong.bin.npy', array.shape, 'r')
    path = tmp_path / 'raw.bin'
    path.write_bytes(array.tobytes() + b'X')
    with pytest.raises(ValueError, match='byte extent'):
        cache.raw_memmap(path, array.shape, 'r')
    path.write_bytes(array.tobytes()[:-1])
    with pytest.raises(ValueError, match='byte extent'):
        cache.raw_memmap(path, array.shape, 'r')
    # Reject a header even when an adversarial size matches the expected extent.
    path.write_bytes(b'\x93NUMPY' + b'\x00' * (array.nbytes - 6))
    with pytest.raises(ValueError, match='NPY header'):
        cache.raw_memmap(path, array.shape, 'r')


def test_write_group_rejects_wrong_layout_and_nonfinite_valid_tokens(tmp_path):
    raw, offsets, groups, identity, progress, _ = _small_cache(tmp_path)
    with pytest.raises(ValueError, match='layout'):
        cache.write_group(raw, np.ones((4, 2, 3), np.float16), groups[1], offsets)
    with pytest.raises(ValueError, match='layout'):
        cache.write_group(raw, np.ones((3, 2, 3), np.float32), groups[1], offsets)
    bad = np.ones((3, 2, 3), np.float16)
    bad[1, 0, 0] = np.inf
    with pytest.raises(FloatingPointError, match='nonfinite'):
        cache.write_group(raw, bad, groups[1], offsets)


def test_exclusive_gpu_checks_all_other_pids_even_same_owner(monkeypatch):
    class Result:
        stdout = f'{cache.os.getpid()}\n31415\n'
    monkeypatch.setattr(cache.subprocess, 'run', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='other compute PIDs'):
        cache.require_exclusive_gpu(4)
    Result.stdout = f'{cache.os.getpid()}\n'
    cache.require_exclusive_gpu(4)


def _probe_stage(tmp_path):
    selection = dict(best_single_layer=21,
        provenance=dict(outer_fold=0, stage='probes', protocol_sha256='registered'))
    cache.dump(tmp_path / 'selection.json', selection)
    complete = dict(complete=True, provenance=selection['provenance'],
        artifacts={'selection.json': cache.sha(tmp_path / 'selection.json')})
    cache.dump(tmp_path / 'complete.json', complete)
    return selection, complete


def test_inner_dev_layer_source_requires_matching_protocol_and_completed_sha(tmp_path):
    selection, complete = _probe_stage(tmp_path)
    layer, files = cache.selected_probe_identity(tmp_path, 0, 'registered')
    assert layer == 21
    assert str(tmp_path / 'selection.json') in files
    assert str(tmp_path / 'complete.json') in files
    with pytest.raises(ValueError, match='protocol'):
        cache.selected_probe_identity(tmp_path, 0, 'another protocol')
    with pytest.raises(ValueError, match='protocol'):
        cache.selected_probe_identity(tmp_path, 1, 'registered')
    selection['best_single_layer'] = 29
    cache.dump(tmp_path / 'selection.json', selection)
    with pytest.raises(ValueError, match='SHA'):
        cache.selected_probe_identity(tmp_path, 0, 'registered')


def test_probe_layer_source_rejects_incomplete_or_mismatched_provenance(tmp_path):
    selection, complete = _probe_stage(tmp_path)
    complete['complete'] = False
    cache.dump(tmp_path / 'complete.json', complete)
    with pytest.raises(ValueError, match='completion provenance'):
        cache.selected_probe_identity(tmp_path, 0, 'registered')
    complete['complete'] = True
    complete['provenance'] = dict(selection['provenance'], seed='changed')
    cache.dump(tmp_path / 'complete.json', complete)
    with pytest.raises(ValueError, match='completion provenance'):
        cache.selected_probe_identity(tmp_path, 0, 'registered')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_new_token_capture_matches_original_pool_capture_exact_on_same_batch(monkeypatch, side):
    # Exercise the actual old PoolCapture hook/last-position convention on CPU.
    # No CUDA device is created by this purely synthetic implementation check.
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    backbone = _Backbone(blocks=32)
    new = cache.TokenBlockCapture(backbone, cache.DEFAULT_LAYERS)
    original = cache.source.PoolCapture(backbone)
    lengths = np.asarray([1, 3, 5])
    ids = torch.full((3, 8), -100)
    mask = torch.zeros((3, 8), dtype=torch.long)
    for row, length in enumerate(lengths):
        start = 8 - length if side == 'left' else 0
        ids[row, start:start + length] = torch.arange(length) + row * 10
        mask[row, start:start + length] = 1
    batch = dict(input_ids=ids, attention_mask=mask)
    packed, last = new.capture_with_original(batch, original)
    assert backbone.calls == 1
    np.testing.assert_array_equal(packed[cache.row_offsets(lengths)[1:] - 1],
                                  last[:, np.asarray(cache.DEFAULT_LAYERS) - 1])
    new.close()
    original.close()
    assert not any(block._forward_hooks for block in backbone.layers)


def test_original_group_plan_keeps_registered_order_and_batch_shapes():
    lengths = np.asarray([3, 1, 2, 5, 1])
    protocol = dict(token_budget=64, max_micro=2, pad_multiple=32)
    planned = cache.original_group_plan(lengths, protocol)
    assert [group.tolist() for group in planned] == [[1, 4], [2, 0], [3]]


def test_legacy_replay_gate_preserves_strict_old_thresholds_and_exact_capture_check():
    record = dict(new_vs_original_capture_exact=True,
        comparisons={'legacy_last': dict(relative_rmse_max=.02499, cosine_min=.9996)},
        cross_batch_descriptive={'mixed': dict(relative_rmse_max=.03, cosine_min=.99955)})
    decisions = [dict(decisions_equal=True)]
    assert cache.legacy_replay_gate([record], decisions)
    changed = copy.deepcopy(record)
    changed['comparisons']['legacy_last']['relative_rmse_max'] = .025001
    assert not cache.legacy_replay_gate([changed], decisions)
    changed = copy.deepcopy(record)
    changed['comparisons']['legacy_last']['cosine_min'] = .9995
    assert not cache.legacy_replay_gate([changed], decisions)
    changed = copy.deepcopy(record)
    changed['new_vs_original_capture_exact'] = False
    assert not cache.legacy_replay_gate([changed], decisions)
    assert not cache.legacy_replay_gate([record], [dict(decisions_equal=False)])


def test_original_all32_scalar_rmse_is_auxiliary_distinct_from_layer_max():
    current = np.ones((32, 2), dtype=np.float16)
    saved = current.copy()
    saved[0] += .04
    assert cache.summary_similarity(saved, current)['relative_rmse_max'] > .025
    assert cache.scalar_relative_rmse(saved, current) < .025


def _internal_record():
    return dict(new_vs_original_capture_exact=True, same_gpu_repeated_group_all_tokens_exact=True,
        cached_group_all_tokens_exact=None,
        comparisons={'legacy_last': dict(relative_rmse_max=.03, cosine_min=.9996)})


def test_v3_gate_does_not_claim_failed_legacy_equivalence():
    record = _internal_record()
    assert cache.replay_gate([record])
    assert not cache.legacy_replay_gate([record], [dict(decisions_equal=True)])
    # A fresh cache may pass internal reproducibility, while the old numerical
    # gate stays false. Its threshold is never increased to produce agreement.
    assert not cache.replay_gate([record], require_cached=True)
    record['cached_group_all_tokens_exact'] = True
    assert cache.replay_gate([record], require_cached=True)
    assert not cache.legacy_replay_gate([record], [dict(decisions_equal=True)])


@pytest.mark.parametrize('field', ['new_vs_original_capture_exact', 'same_gpu_repeated_group_all_tokens_exact',
                                 'cached_group_all_tokens_exact'])
def test_internal_repeat_or_cached_feature_failure_blocks_publication(field):
    record = _internal_record()
    record['cached_group_all_tokens_exact'] = True
    record[field] = False
    assert not cache.replay_gate([record], require_cached=True)
    assert not cache.replay_gate([])


def test_repeated_group_check_catches_nonfinal_token_changes_even_with_exact_hooks():
    lengths = [2, 3]
    offsets = cache.row_offsets(lengths)
    packed = np.arange(20).reshape(5, 2, 2).astype(np.float16)
    repeated = packed.copy()
    original_last = np.zeros((2, 4, 2), dtype=np.float16)
    original_last[:, [1, 3]] = packed[offsets[1:] - 1]
    end_exact, repeat_exact = cache.exact_group_checks(packed, repeated, original_last,
        original_last.copy(), offsets, np.asarray([1, 3]))
    assert end_exact and repeat_exact
    repeated[0, 0, 0] += 1
    end_exact, repeat_exact = cache.exact_group_checks(packed, repeated, original_last,
        original_last.copy(), offsets, np.asarray([1, 3]))
    assert end_exact and not repeat_exact


def test_cached_group_replay_checks_all_group_rows_and_all_tokens(tmp_path):
    raw, offsets, groups, identity, progress, packed = _small_cache(tmp_path)
    assert cache.cached_group_exact(raw, packed, groups[0], offsets)
    # Corrupt an internal token in the other group row rather than the stratum
    # anchor or final token; full group replay must still fail.
    raw[0, 1, 0] += 1
    assert not cache.cached_group_exact(raw, packed, groups[0], offsets)
    with pytest.raises(ValueError, match='lost valid tokens'):
        cache.cached_group_exact(raw, packed[:-1], groups[0], offsets)


@pytest.mark.parametrize('field', ['physical_gpu_uuid', 'torch_runtime_version', 'nvidia_driver_version',
                                 'qwen_fast_path_available'])
def test_common_cache_rejects_runtime_gpu_or_implementation_drift(field):
    fingerprint = dict(physical_gpu_uuid='GPU-original', torch_runtime_version='2.6',
        nvidia_driver_version='stable', qwen_fast_path_available=True)
    cache.require_same_runtime(fingerprint.copy(), fingerprint)
    changed = fingerprint.copy()
    changed[field] = 'changed'
    with pytest.raises(RuntimeError, match='fingerprint changed'):
        cache.require_same_runtime(changed, fingerprint)
