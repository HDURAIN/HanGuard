"""CPU tests for waiting, strict admission, own exit races, and original commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.hanguard import ab_diagnostic_dual_gpu_followup as dual
from scripts.hanguard import ab_diagnostic_followup as original
from scripts.hanguard import ab_diagnostic_queue as queue
from scripts.hanguard import ab_readout_parallel_queue as ownership


def test_init_and_report_change_only_phase_and_CPU_device():
    command = ('python', 'original.py', 'run', '--device', 'cuda', '--seed', '42',
               '--max-epochs', '40', '--joint-features')
    stage = queue.Stage('original', command, ('cache',), ('protocol', 'results', 'predictions'), 'heads')
    for phase in ('init', 'report'):
        new = dual.cpu_stage(stage, phase, f'{phase}_original')
        assert new.command == command[:2] + (phase, '--device', 'cpu') + command[5:]
        assert new.inputs == stage.inputs
        assert new.outputs == (stage.outputs[:1] if phase == 'init' else stage.outputs)


def _guard(root):
    attempt = root / 'attempt'
    output = attempt / 'extraction'
    output.mkdir(parents=True)
    stage = queue.Stage('extract_adapted', (), validator='cache')
    guard = dual.WaitingExtraction(root, [stage], {}, queue_state_dir=output)
    queue.dump(output / 'queue_protocol.json', dict(frozen=True))
    guard.protocol_sha = queue.sha(output / 'queue_protocol.json')
    guard.active_stage = stage.name
    return guard


def test_extraction_waits_past_original_deadline_without_GPU_child(tmp_path, monkeypatch):
    guard = _guard(tmp_path)
    calls = []
    def observation():
        calls.append(True)
        return dict(eligible=len(calls) == 201, blocking_processes=[dict(pid=999)] if len(calls) < 201 else [])
    guard.observation = observation
    monkeypatch.setattr(dual.time, 'sleep', lambda value: None)
    monkeypatch.setattr(dual.time, 'monotonic', lambda: len(calls))
    guard._guard_gpu_idle()
    assert len(calls) == 201 and guard.child is None
    assert queue.read_json(guard.output / 'queue_state.json')['state'] == 'strict_GPU4_admitted'


def test_CPU_wait_honors_attempt_STOP_without_own_or_foreign_signal(tmp_path, monkeypatch):
    guard = _guard(tmp_path)
    def observation():
        (guard.output.parent / 'STOP').write_text('test stop')
        return dict(eligible=False)
    guard.observation = observation
    monkeypatch.setattr(dual.time, 'sleep', lambda value: None)
    with pytest.raises(queue.StopRequested, match='test stop'):
        guard._guard_gpu_idle()
    assert guard.child is None


def _owned(study, pid=123):
    return dict(pid=pid, uid=os.getuid(), start_time_ticks=100,
        command_sha256='owned-command', cwd=str(queue.ROOT),
        argv=[sys.executable, str(queue.ROOT / 'scripts/hanguard/ab_frozen_readout_diagnostics.py'),
              'run', '--output', str(study / 'readouts/adapted_last'), '--fold', '0'])


class FakeChild:
    pid = 123
    def __init__(self, code):
        self.code = code
    def poll(self):
        return self.code


def _race(root, code=0, snapshot=None, validation_error=False):
    output = root / 'dispatcher'
    output.mkdir()
    def missing(pid):
        raise ValueError('Process command is unavailable')
    def validator(fold):
        if validation_error:
            raise ValueError('Completed fold artifact changed')
        return {'completed.predictions': 'verified-SHA'}
    dispatcher = dual.RaceSafeParallelQueue(root, output, root / 'readouts/adapted_last', {}, {},
        owned={123: _owned(root)}, candidates=(2, 4), max_workers=2,
        raw_identity_reader=missing, snapshot_reader=lambda pid: snapshot, validator=validator)
    dispatcher.children[0] = dict(process=FakeChild(code), gpu=4)
    return dispatcher


@pytest.mark.parametrize('snapshot', [None, dict(pid=123, uid=os.getuid(), start_time_ticks=100, state='Z', exit_code=0)])
def test_registered_successful_exit_is_reconciled_only_after_full_fold_verification(tmp_path, snapshot):
    dispatcher = _race(tmp_path, snapshot=snapshot)
    result = dispatcher.resolve_identity(123)
    assert result == _owned(tmp_path)
    proof = queue.read_json(dispatcher.output / 'verified_own_exit_fold_0.json')
    assert proof['complete_fold_artifact_sha256'] == {'completed.predictions': 'verified-SHA'}
    assert proof['exit_code'] == 0 and proof['ended_process_snapshot'] == snapshot


@pytest.mark.parametrize('code,snapshot', [
    (None, None), (1, None),
    (0, dict(pid=123, uid=os.getuid(), start_time_ticks=100, state='R', exit_code=0)),
    (0, dict(pid=123, uid=os.getuid(), start_time_ticks=101, state='Z', exit_code=0)),
    (0, dict(pid=123, uid=os.getuid() + 1, start_time_ticks=100, state='Z', exit_code=0)),
    (0, dict(pid=123, uid=os.getuid(), start_time_ticks=100, state='Z', exit_code=256))])
def test_live_unknown_foreign_birth_UID_or_failed_exit_is_not_excused(tmp_path, code, snapshot):
    dispatcher = _race(tmp_path, code=code, snapshot=snapshot)
    with pytest.raises(ValueError, match='command is unavailable'):
        dispatcher.resolve_identity(123)
    assert not list(dispatcher.output.glob('verified_own_exit*'))


def test_exit_race_requires_completed_payload_SHA_and_does_not_excuse_unregistered_PID(tmp_path):
    dispatcher = _race(tmp_path, validation_error=True)
    with pytest.raises(ValueError, match='artifact changed'):
        dispatcher.resolve_identity(123)
    with pytest.raises(ValueError, match='command is unavailable'):
        dispatcher.resolve_identity(999)


def test_exit_race_PermissionError_for_proc_snapshot_still_blocks(tmp_path):
    dispatcher = _race(tmp_path)
    def denied(pid):
        raise PermissionError('Unknown owner')
    dispatcher.snapshot_reader = denied
    with pytest.raises(PermissionError):
        dispatcher.resolve_identity(123)


def test_valid_live_foreign_identity_is_returned_and_original_assessor_blocks_it(tmp_path):
    dispatcher = _race(tmp_path)
    foreign = dict(_owned(tmp_path), command_sha256='foreign-command')
    dispatcher.raw_identity_reader = lambda pid: foreign
    result = ownership.assess_gpu(dict(index=4, uuid='GPU4', free_gib=47, pids=[123]),
        dispatcher.owned, tmp_path, identity_reader=dispatcher.resolve_identity)
    assert not result['eligible']
    assert result['confirmed_foreign_processes'][0]['pid'] == 123


def _toy_dispatcher(root, inventory=None):
    output, readout = root / 'dispatcher', root / 'readout'
    targets = [root / f'fold{fold}.done' for fold in range(5)]
    commands = {fold: (sys.executable, '-c',
        f'import time; from pathlib import Path; time.sleep(.04); Path({str(path)!r}).write_text("{fold}")')
        for fold, path in enumerate(targets)}
    counter = [0]
    def clock():
        counter[0] += 30
        return counter[0]
    def validator(fold):
        assert targets[fold].read_text() == str(fold)
        return {str(targets[fold]): queue.sha(targets[fold])}
    dispatcher = dual.RaceSafeParallelQueue(root, output, readout, commands, {},
        candidates=(2, 4), max_workers=2, watch_seconds=float('inf'), poll=.01,
        clock=clock, validator=validator)
    seen = []
    def empty():
        seen.append({worker['gpu'] for worker in dispatcher.children.values()})
        return [dict(index=i, uuid=f'GPU{i}', free_gib=47, pids=[]) for i in (2, 4)]
    dispatcher.inventory = inventory or empty
    return dispatcher, targets, seen


def test_five_CPU_toy_folds_use_two_distinct_cards_and_one_worker_per_card(tmp_path):
    dispatcher, targets, seen = _toy_dispatcher(tmp_path)
    assert dispatcher.run()['state'] == 'folds_completed'
    assert all(path.exists() for path in targets)
    assert any(gpus == {2, 4} for gpus in seen)
    assert all(len(gpus) <= 2 for gpus in seen)
    assert dispatcher.watch_seconds == float('inf')
    assert 'Infinity' not in (dispatcher.output / 'protocol.json').read_text()
    for fold in range(5):
        receipt = queue.read_json(dispatcher.output / f'fold_{fold}.receipt.json')
        assert receipt['gpu'] in (2, 4)


def test_foreign_appearance_stops_only_own_CPU_toy_siblings(tmp_path):
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    try:
        dispatcher, _, _ = _toy_dispatcher(tmp_path)
        def inventory():
            used = {worker['gpu'] for worker in dispatcher.children.values()}
            return [dict(index=i, uuid=f'GPU{i}', free_gib=47,
                         pids=[foreign.pid] if i in used else []) for i in (2, 4)]
        dispatcher.inventory = inventory
        with pytest.raises(queue.StopRequested, match='foreign'):
            dispatcher.run()
        assert foreign.poll() is None
        assert all(worker['process'].poll() is not None for worker in dispatcher.children.values())
        proof = queue.read_json(dispatcher.output / 'ownership_stop_trigger.json')
        assert foreign.pid in proof['trigger_confirmed_foreign_pids']
        assert queue.read_json(dispatcher.output / 'state.json')['state'] == 'stopped'
    finally:
        foreign.terminate()
        foreign.wait()


def test_original_readout_commands_keep_every_registered_scientific_setting(tmp_path):
    protocol = dict(cache_manifest_path=str(tmp_path / 'cache.json'),
        settings={key: value for key, value in zip(ownership.SETTINGS,
                  (42, 12, 12, 40, .001, .001, .001, 128, .1, 256, 2000))})
    protocol['settings']['joint_features'] = True
    commands = [ownership._arguments(protocol, tmp_path / 'readout', sys.executable, 'run', fold)
                for fold in range(5)]
    assert all(command[:-2] == commands[0][:-2] for command in commands)
    assert [command[-1] for command in commands] == [str(fold) for fold in range(5)]
    assert all('--joint-features' in command and command[command.index('--seed') + 1] == '42' for command in commands)


def test_resource_registration_rejects_modified_GPU_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(original, 'check_registration', lambda output: {'unchanged_science': True})
    scientific = tmp_path / 'science.json'
    scientific.write_text('{}')
    resource = dict(schema=dual.SCHEMA, scientific_protocol_path=str(scientific),
        scientific_protocol_sha256=queue.sha(scientific), bindings={}, candidates=[2, 4])
    queue.dump(tmp_path / 'resource_protocol.json', resource)
    queue.dump(tmp_path / 'resource_registration.json',
               dict(protocol_sha256=queue.sha(tmp_path / 'resource_protocol.json')))
    assert dual.check_registration(tmp_path)[0] == resource
    resource['candidates'] = [0, 1]
    queue.dump(tmp_path / 'resource_protocol.json', resource)
    with pytest.raises(ValueError, match='independent registration'):
        dual.check_registration(tmp_path)


def test_missing_registration_failure_is_recorded_without_retry(tmp_path):
    with pytest.raises(FileNotFoundError):
        dual.run(tmp_path)
    state = queue.read_json(tmp_path / 'resource_state.json')
    assert state['state'] == 'failed' and state['automatic_retry'] is False
    original_sha = queue.sha(tmp_path / 'resource_state.json')
    with pytest.raises(RuntimeError, match='cannot retry'):
        dual.run(tmp_path)
    assert queue.sha(tmp_path / 'resource_state.json') == original_sha
