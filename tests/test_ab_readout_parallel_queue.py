"""Foreign ownership, rather than spare VRAM, controls GPU admission."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.hanguard import ab_readout_parallel_queue as parallel


def identity(pid=123, study=Path('/workspace/study'), script='ab_diagnostic_experiment.py'):
    return dict(pid=pid, uid=os.getuid(), start_time_ticks=100, command_sha256='exact-command',
        argv=['python', str(parallel.ROOT / 'scripts/hanguard' / script), 'train', '--output', str(study / 'lora_study')])


@pytest.mark.parametrize('own_uid', [True, False])
def test_occupied_low_utilization_high_memory_GPU_is_forbidden(own_uid):
    current = identity()
    if not own_uid:
        current['uid'] += 1
    gpu = dict(index=0, uuid='GPU-0', free_gib=44, pids=[123], utilization=0)
    decision = parallel.assess_gpu(gpu, {}, '/workspace/study', identity_reader=lambda pid: current)
    assert decision['eligible'] is False
    assert decision['foreign_processes'][0]['pid'] == 123


def test_empty_and_explicit_registered_same_study_are_allowed_but_pid_reuse_is_not():
    gpu = dict(index=4, uuid='GPU-4', free_gib=44, pids=[])
    assert parallel.assess_gpu(gpu, {}, '/workspace/study')['eligible']
    current = identity()
    gpu['pids'] = [123]
    assert parallel.assess_gpu(gpu, {123: current}, '/workspace/study', identity_reader=lambda pid: current)['eligible']
    changed = dict(current, start_time_ticks=101)
    assert not parallel.assess_gpu(gpu, {123: current}, '/workspace/study', identity_reader=lambda pid: changed)['eligible']
    elsewhere = identity(study=Path('/other/study'))
    assert not parallel.assess_gpu(gpu, {123: elsewhere}, '/workspace/study', identity_reader=lambda pid: elsewhere)['eligible']


def test_unknown_PID_owner_blocks_admission():
    def unknown(pid):
        raise PermissionError('proc unavailable')
    decision = parallel.assess_gpu(dict(index=1, uuid='GPU-1', free_gib=44, pids=[123]), {},
        '/workspace/study', identity_reader=unknown)
    assert not decision['eligible']
    assert 'Unknown' in decision['foreign_processes'][0]['reason']


def _inventory():
    return [dict(index=i, uuid=f'GPU-{i}', free_gib=8, pids=[]) for i in range(3)]


def _fast_confirmation_clock():
    current = [0]
    def next_time():
        current[0] += 61
        return current[0]
    return next_time


def test_CPU_fake_folds_have_isolated_outputs_and_max_three_workers(tmp_path):
    commands = {fold: (sys.executable, '-c',
        f'from pathlib import Path; import time; time.sleep(.1); '
        f'Path({str(tmp_path / f"result_{fold}")!r}).write_text("complete")') for fold in range(5)}
    queue = parallel.ParallelQueue(tmp_path, tmp_path / 'queue', tmp_path / 'readouts', commands, {},
        inventory=_inventory, poll=.02, watch_seconds=.1, clock=_fast_confirmation_clock())
    assert queue.run()['state'] == 'folds_completed'
    assert set(queue.completed) == set(range(5))
    assert all((tmp_path / f'result_{fold}').read_text() == 'complete' for fold in range(5))


def test_failed_CPU_worker_stops_only_scheduler_siblings(tmp_path):
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        commands = {0: (sys.executable, '-c', 'import time; time.sleep(.1); raise SystemExit(7)'),
                    1: (sys.executable, '-c', 'import time; time.sleep(20)')}
        queue = parallel.ParallelQueue(tmp_path, tmp_path / 'queue', tmp_path / 'readouts', commands, {},
            inventory=_inventory, poll=.02, clock=_fast_confirmation_clock())
        with pytest.raises(RuntimeError, match='exited 7'):
            queue.run()
        assert all(worker['process'].poll() is not None for worker in queue.children.values())
        assert foreign.poll() is None
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_root_STOP_cleans_own_CPU_workers(tmp_path):
    command = (sys.executable, '-c', f'from pathlib import Path; import time; time.sleep(.1); '
               f'Path({str(tmp_path / "STOP")!r}).write_text("user stop"); time.sleep(20)')
    queue = parallel.ParallelQueue(tmp_path, tmp_path / 'queue', tmp_path / 'readouts', {0: command}, {},
        inventory=_inventory, poll=.02, clock=_fast_confirmation_clock())
    with pytest.raises(parallel.StopRequested, match='user stop'):
        queue.run()
    assert all(worker['process'].poll() is not None for worker in queue.children.values())


def test_empty_public_card_needs60seconds_and_gap_resets_confirmation():
    confirmer = parallel.IdleConfirmation(60)
    empty = parallel.assess_gpu(dict(index=0, uuid='GPU-0', free_gib=44, pids=[]), {}, '/workspace/study')
    assert not confirmer.apply(empty, 0)['eligible']
    assert not confirmer.apply(empty, 59)['eligible']
    assert confirmer.apply(empty, 60)['eligible']
    occupied = parallel.assess_gpu(dict(index=0, uuid='GPU-0', free_gib=44, pids=[123]), {},
        '/workspace/study', identity_reader=lambda pid: identity())
    assert not confirmer.apply(occupied, 61)['eligible']
    assert not confirmer.apply(empty, 70)['eligible']
    assert not confirmer.apply(empty, 129)['eligible']
    assert confirmer.apply(empty, 130)['eligible']
    own = identity()
    owned_card = parallel.assess_gpu(dict(index=4, uuid='GPU-4', free_gib=44, pids=[123]), {123: own},
        '/workspace/study', identity_reader=lambda pid: own)
    assert confirmer.apply(owned_card, 0)['eligible']
    with pytest.raises(ValueError, match='at least60'):
        parallel.IdleConfirmation(59)


@pytest.mark.parametrize('classification', ['foreign', 'unknown'])
def test_exact_ownership_trigger_is_saved_before_cleanup_and_in_final_STOP(tmp_path, classification):
    command = (sys.executable, '-c', 'import time; time.sleep(20)')
    queue = None
    def inventory():
        return [dict(index=0, uuid='GPU-0', free_gib=44, pids=[999] if queue.children else [])]
    def reader(pid):
        if classification == 'unknown':
            raise PermissionError('Ownership unavailable')
        return dict(identity(pid=pid), uid=os.getuid() + 100)
    queue = parallel.ParallelQueue(tmp_path, tmp_path / 'queue', tmp_path / 'readouts', {0: command}, {},
        inventory=inventory, poll=.02, clock=_fast_confirmation_clock(), identity_reader=reader)
    cleanup = queue.terminate_ours
    before_cleanup = []
    def checked_cleanup():
        proof = parallel.read_json(queue.output / 'ownership_stop_trigger.json')
        stopping = parallel.read_json(queue.output / 'state.json')
        assert stopping['state'] == 'stopping_for_ownership_conflict'
        assert proof['termination_started'] is False
        assert proof['triggering_gpu_observations'][0]['blocking_processes'][0]['classification'] == classification
        before_cleanup.append(proof)
        cleanup()
    queue.terminate_ours = checked_cleanup
    with pytest.raises(parallel.StopRequested, match='process observed'):
        queue.run()
    assert before_cleanup
    state = parallel.read_json(queue.output / 'state.json')
    assert state['state'] == 'stopped'
    assert state['trigger_gpu_observations'] == before_cleanup[0]['trigger_gpu_observations']
    assert state['trigger_confirmed_foreign_pids'] == ([999] if classification == 'foreign' else [])
    assert state['trigger_unknown_ownership_pids'] == ([999] if classification == 'unknown' else [])
    assert state['termination_completed'] is True
    assert all(worker['process'].poll() is not None for worker in queue.children.values())


def test_fresh_attempt_adopts_verified_completed_fold_without_worker_or_GPU_query(tmp_path):
    readout = tmp_path / 'readouts'
    artifacts = {}
    for phase in ('probes', 'heads'):
        path = readout / 'fold_0' / phase / 'complete.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"complete": true, "unchanged": true}')
        artifacts[str(path)] = parallel.sha(path)
    def validate(fold):
        assert fold == 0
        assert {name: parallel.sha(name) for name in artifacts} == artifacts
        return artifacts
    def no_resource_query():
        raise AssertionError('Completed fold must not query or occupy GPUs')
    queue = parallel.ParallelQueue(tmp_path, tmp_path / 'new_attempt', readout,
        {0: (sys.executable, '-c', 'raise AssertionError("must not launch")')}, {},
        inventory=no_resource_query, validator=validate)
    assert queue.run()['state'] == 'folds_completed'
    receipt = parallel.read_json(queue.output / 'fold_0.receipt.json')
    assert receipt['adopted_verified_completed_fold'] is True
    assert receipt['command_not_executed_by_this_attempt'] is True
    assert {name: parallel.sha(name) for name in artifacts} == artifacts
