"""Public-server admission checks without allocating CUDA memory."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hanguard import nolora_ablation_queue as queue


def registered_worker(tmp_path, monkeypatch, *, device="cuda", gpu=3):
    expected = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=19,
                    command_sha256="command", argv=["test-worker"], cwd=str(tmp_path))
    args = SimpleNamespace(output=tmp_path, task="components", device=device,
                           gpu=gpu, worker_id="test_worker")
    monkeypatch.setattr(queue, "verify_control", lambda _: {})
    monkeypatch.setattr(queue, "identity", lambda _: expected)
    queue.write(tmp_path / "queue_workers/test_worker.json", dict(
        task=args.task, device=device, gpu=gpu, identity=expected))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(gpu))
    return args


def test_foreign_compute_pid_blocks_even_with_free_memory(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "inventory", lambda: [dict(index=3, pids=[12345], free_gib=47)])
    assert queue.worker(args) == 75
    assert not (tmp_path / "queue_workers/test_worker.admitted.json").exists()


def test_low_memory_blocks_without_foreign_pid(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "inventory", lambda: [dict(index=3, pids=[], free_gib=1)])
    assert queue.worker(args) == 75


def test_unknown_inventory_never_admits(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    def unavailable():
        raise RuntimeError("nvidia-smi failed")
    monkeypatch.setattr(queue, "inventory", unavailable)
    with pytest.raises(RuntimeError, match="nvidia-smi failed"):
        queue.worker(args)
    assert not (tmp_path / "queue_workers/test_worker.admitted.json").exists()


def test_same_uid_unregistered_pid_is_foreign(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "inventory", lambda: [dict(index=3, pids=[os.getpid(), 98765], free_gib=45)])
    assert queue.worker(args) == 75


def test_worker_identity_change_rejected_before_cuda(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "identity", lambda _: dict(pid=os.getpid(), start_time_ticks=20))
    with pytest.raises(ValueError, match="identity mismatch"):
        queue.worker(args)


def test_wrong_visible_card_rejected_before_cuda(tmp_path, monkeypatch):
    args = registered_worker(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="visible GPU"):
        queue.worker(args)


def test_empty_card_policy_never_reserves_a_card():
    cards = [dict(index=0, pids=[111], free_gib=47), dict(index=1, pids=[], free_gib=46),
             dict(index=2, pids=[], free_gib=42)]
    assert queue.empty_cards(cards, list(range(8)), 4) == [1, 2]
    assert queue.POLICY["reserved_gpus"] == []
    assert queue.POLICY["active_foreign"] == "record_only"


def test_task_output_identity():
    assert queue.task_folder(Path("out"), "seed43") == Path("out/seed_43")
    assert queue.task_folder(Path("out"), "components") == Path("out/components")
