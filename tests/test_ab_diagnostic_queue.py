"""CPU subprocess tests for diagnosis failure, interruption and provenance."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.hanguard.ab_diagnostic_queue import (
    SequentialSupervisor, Stage, StopRequested, read_json, sha, validate_outputs,
)


def _queue(root, stages):
    return SequentialSupervisor(root, stages, {}, poll_seconds=.05, terminate_seconds=.5)


def _write_stage(root, name, payload='complete'):
    artifact = root / f'{name}.json'
    count = root / f'{name}.launches'
    command = (sys.executable, '-c',
        f'from pathlib import Path; p=Path({str(count)!r}); '
        f'p.write_text(p.read_text()+"x" if p.exists() else "x"); '
        f'Path({str(artifact)!r}).write_text({json.dumps(payload)!r})')
    return Stage(name, command, outputs=(str(artifact),))


def test_completed_receipts_resume_without_relaunch_and_reject_output_drift(tmp_path):
    stages = [_write_stage(tmp_path, name) for name in ('first', 'second')]
    assert _queue(tmp_path, stages).run()['state'] == 'completed'
    assert _queue(tmp_path, stages).run()['state'] == 'completed'
    assert (tmp_path / 'first.launches').read_text() == 'x'
    assert (tmp_path / 'second.launches').read_text() == 'x'
    (tmp_path / 'first.json').write_text('changed')
    with pytest.raises(ValueError, match='provenance changed'):
        _queue(tmp_path, stages).run()
    assert read_json(tmp_path / 'queue_state.json')['state'] == 'failed'


def test_failed_stage_stops_followups_and_cannot_automatically_retry(tmp_path):
    failed = Stage('failed', (sys.executable, '-c', 'raise SystemExit(7)'))
    stages = [failed, _write_stage(tmp_path, 'must_not_run')]
    with pytest.raises(RuntimeError, match='code 7'):
        _queue(tmp_path, stages).run()
    state = read_json(tmp_path / 'queue_state.json')
    assert state['state'] == 'failed'
    assert state['scientific_conclusion'] is None
    assert not (tmp_path / 'must_not_run.launches').exists()
    assert read_json(tmp_path / 'diagnostic_summary.json')['scientific_conclusions'] == []
    with pytest.raises(RuntimeError, match='no automatic retry'):
        _queue(tmp_path, stages).run()


def test_STOP_terminates_owned_child_and_leaves_unrelated_process_alive(tmp_path):
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        command = (sys.executable, '-c',
            f'from pathlib import Path; import time; time.sleep(.1); '
            f'Path({str(tmp_path / "STOP")!r}).write_text("direction review"); time.sleep(20)')
        queue = _queue(tmp_path, [Stage('long', command), _write_stage(tmp_path, 'must_not_run')])
        began = time.monotonic()
        with pytest.raises(StopRequested, match='direction review'):
            queue.run()
        assert time.monotonic() - began < 3
        assert queue.child.poll() is not None
        assert foreign.poll() is None
        assert read_json(tmp_path / 'queue_state.json')['state'] == 'stopped'
        assert not (tmp_path / 'must_not_run.launches').exists()
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


def test_second_supervisor_cannot_acquire_live_study_lock(tmp_path):
    with (tmp_path / 'supervisor.pid').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='holds the study lock'):
            _queue(tmp_path, [_write_stage(tmp_path, 'never')]).run()
    assert not (tmp_path / 'never.launches').exists()


def test_command_registration_cannot_change_on_resume(tmp_path):
    first = _write_stage(tmp_path, 'stage', 'first')
    _queue(tmp_path, [first]).run()
    changed = _write_stage(tmp_path, 'stage', 'different settings')
    with pytest.raises(ValueError, match='registration differs'):
        _queue(tmp_path, [changed]).run()
    assert (tmp_path / 'stage.launches').read_text() == 'x'


def test_completed_cache_payload_and_metadata_must_still_match(tmp_path):
    files = {name: tmp_path / name for name in ('features.npy', 'metadata.parquet', 'protocol.json', 'partition.json')}
    for path in files.values():
        path.write_bytes(b'original artifact')
    files['partition.json'].write_text(json.dumps(dict(artifact_sha256={'B.parquet': 'registered-B'})))
    files['protocol.json'].write_text(json.dumps(dict(partition_manifest_path=str(files['partition.json']),
        partition_manifest_sha256=sha(files['partition.json']))))
    replay = tmp_path / 'replay_audit.json'
    replay.write_text('{"passed": true}')
    manifest = dict(complete=True, cohort='B', representation='base', adapter_training_cohort='none',
        adapter_sha256=None, B_sha256='registered-B', path=str(files['features.npy']),
        sha256=sha(files['features.npy']), metadata_path=str(files['metadata.parquet']),
        metadata_sha256=sha(files['metadata.parquet']), training_protocol_path=str(files['protocol.json']),
        training_protocol_sha256=sha(files['protocol.json']), ab_protocol_path=str(files['partition.json']),
        ab_protocol_sha256=sha(files['partition.json']), replay_audit_sha256=sha(replay))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    stage = Stage('extract_base', (), inputs=(str(files['protocol.json']),),
                  outputs=(str(manifest_path),), validator='cache')
    validate_outputs(stage)
    manifest['representation'] = 'adapted'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='current A/B protocol or representation'):
        validate_outputs(stage)
    manifest['representation'] = 'base'
    manifest_path.write_text(json.dumps(manifest))
    files['features.npy'].write_bytes(b'corrupted payload')
    with pytest.raises(ValueError, match='artifact changed: path'):
        validate_outputs(stage)
    files['features.npy'].write_bytes(b'original artifact')
    files['metadata.parquet'].write_bytes(b'corrupted metadata')
    with pytest.raises(ValueError, match='artifact changed: metadata_path'):
        validate_outputs(stage)


def test_foreign_GPU4_job_stops_before_child_launch(tmp_path, monkeypatch):
    def fake_query(command, **kwargs):
        answer = '0, GPU-zero, 10000, 20000\n4, GPU-four, 32000, 50000\n' if '--query-gpu' in command[1] else 'GPU-four, 12345\n'
        return subprocess.CompletedProcess(command, 0, stdout=answer, stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_query)
    queue = SequentialSupervisor(tmp_path, [_write_stage(tmp_path, 'never')], {},
        poll_seconds=.05, terminate_seconds=.5, environment={'CUDA_VISIBLE_DEVICES': '4'})
    with pytest.raises(RuntimeError, match='occupied.*12345'):
        queue.run()
    assert queue.child is None
    assert not (tmp_path / 'never.launches').exists()


@pytest.mark.parametrize('free_mib', [32768, 25000])
def test_minimum_free_memory_never_authorizes_foreign_GPU_sharing(tmp_path, monkeypatch, free_mib):
    def fake_query(command, **kwargs):
        answer = f'4, GPU-four, {free_mib}, 50000\n' if '--query-gpu' in command[1] else 'GPU-four, 12345\n'
        return subprocess.CompletedProcess(command, 0, stdout=answer, stderr='')
    monkeypatch.setattr(subprocess, 'run', fake_query)
    queue = SequentialSupervisor(tmp_path, [_write_stage(tmp_path, 'shared')], {},
        poll_seconds=.05, terminate_seconds=.5, environment={'CUDA_VISIBLE_DEVICES': '4'},
        minimum_free_gib=26, resource_evidence={'preflight_peak_gib': 18.2})
    with pytest.raises(RuntimeError, match='foreign/unknown.*12345.*free memory does not permit sharing'):
        queue.run()
    assert queue.child is None
    assert not (tmp_path / 'shared.launches').exists()
    registered = read_json(tmp_path / 'queue_protocol.json')
    assert registered['resource_policy']['mode'] == 'strict_ownership'
    assert registered['resource_policy']['minimum_free_gib'] == 26
    assert registered['initial_resource_observation']['foreign_compute_pids'] == [12345]


def test_separate_queue_attempt_preserves_failed_root_and_honors_root_STOP(tmp_path):
    old_files = {name: tmp_path / name for name in
                 ('queue_state.json', 'queue_protocol.json', 'diagnostic_summary.json')}
    for path in old_files.values():
        path.write_text('{"state": "failed", "unchanged": true}')
    before = {name: sha(path) for name, path in old_files.items()}
    recovery = tmp_path / 'queue_recovery_1'
    queue = SequentialSupervisor(tmp_path, [_write_stage(tmp_path, 'new')], {},
        poll_seconds=.05, terminate_seconds=.5, queue_state_dir=recovery)
    assert queue.run()['state'] == 'completed'
    assert before == {name: sha(path) for name, path in old_files.items()}
    assert (recovery / 'queue_receipts/new.json').is_file()
    assert not (tmp_path / 'queue_receipts').exists()
    (tmp_path / 'STOP').write_text('Review scientific direction')
    with pytest.raises(StopRequested, match='scientific direction'):
        queue.run()
    assert read_json(recovery / 'queue_state.json')['state'] == 'stopped'
    assert before == {name: sha(path) for name, path in old_files.items()}
