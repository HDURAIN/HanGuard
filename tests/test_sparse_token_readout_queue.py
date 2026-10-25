"""CPU-only guardian tests; no real CUDA process or foreign signal is launched."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.hanguard import sparse_token_readout_queue as queue


def write_stage(root, name='fit', *, gpu=False, complete=True):
    artifact = root / f'{name}.json'
    command = (sys.executable, '-c',
        'import json, os; from pathlib import Path; '
        f'Path({str(artifact)!r}).write_text(json.dumps(dict(complete={complete!r}, '
        'cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))))')
    return queue.RegisteredStage(name, command, outputs=(str(artifact),),
                                validator='complete_json', gpu_required=gpu)


def supervisor(root, stages, **kwargs):
    return queue.SingleCardSupervisor(root, stages, {}, queue_state_dir=root / 'execution',
                                     poll_seconds=.02, terminate_seconds=.5, **kwargs)


def gpu_row(pids=(), free=40, index=2):
    return dict(index=index, uuid=f'GPU-{index}', free_gib=free, pids=list(pids))


def test_CPU_stage_does_not_query_gpu_and_forces_empty_CUDA_mask(tmp_path):
    def prohibited():
        raise AssertionError('CPU stage queried the public GPU inventory')
    stage = write_stage(tmp_path)
    guardian = supervisor(tmp_path, [stage], inventory=prohibited)
    assert guardian.run()['state'] == 'completed'
    assert queue.legacy.read_json(stage.outputs[0])['cuda'] == ''
    registered = queue.legacy.read_json(tmp_path / 'execution/queue_protocol.json')
    assert registered['resource_policy']['physical_gpu'] == 2
    assert registered['resource_policy']['reserved_gpu'] == 4


def test_empty_GPU2_is_admitted_immediately_while_GPU4_foreign_is_unused(tmp_path):
    stage = write_stage(tmp_path, gpu=True)
    guardian = supervisor(tmp_path, [stage], inventory=lambda: [gpu_row(), gpu_row([99999], index=4)])
    assert guardian.run()['state'] == 'completed'
    assert queue.legacy.read_json(stage.outputs[0])['cuda'] == '2'
    receipt = queue.legacy.read_json(tmp_path / 'execution/queue_receipts/fit.json')
    assert receipt['state'] == 'completed'


def test_busy_card_waits_without_child_and_honors_STOP_even_with_free_memory(tmp_path):
    calls = 0
    def inventory():
        nonlocal calls
        calls += 1
        if calls == 3:
            (tmp_path / 'STOP').write_text('No busy public-card use')
        return [gpu_row([987654321], free=47)]
    stage = write_stage(tmp_path, gpu=True)
    guardian = supervisor(tmp_path, [stage], inventory=inventory)
    with pytest.raises(queue.legacy.StopRequested, match='No busy'):
        guardian.run()
    assert guardian.child is None
    assert not Path(stage.outputs[0]).exists()
    assert calls == 3


def test_foreign_join_stops_only_owned_child_and_preserves_trigger(tmp_path):
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    stage = queue.RegisteredStage('long', (sys.executable, '-c', 'import time; time.sleep(10)'))
    guardian = None
    def inventory():
        return [gpu_row([guardian.child.pid, foreign.pid] if guardian.child is not None else [])]
    guardian = supervisor(tmp_path, [stage], inventory=inventory)
    try:
        with pytest.raises(queue.legacy.StopRequested, match='foreign/unknown'):
            guardian.run()
        assert guardian.child.poll() is not None
        assert foreign.poll() is None
        trigger = queue.legacy.read_json(tmp_path / 'execution/ownership_stop_trigger.json')
        assert trigger['foreign_processes_signalled'] is False
        blocked = trigger['gpu_observation']['blocking_processes']
        assert [item['pid'] for item in blocked] == [foreign.pid]
        final = queue.legacy.read_json(tmp_path / 'execution/queue_state.json')
        assert final['trigger_evidence']['termination_completed'] is True
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)


@pytest.mark.parametrize('changed', ('uid', 'start_time_ticks', 'command_sha256', 'argv', 'cwd'))
def test_same_UID_or_PID_alone_does_not_authorize_ownership(tmp_path, changed):
    stage = write_stage(tmp_path, gpu=True)
    expected = dict(pid=2345, uid=os.getuid(), start_time_ticks=10, command_sha256='hash',
                    argv=list(stage.command), cwd=str(queue.ROOT))
    current = dict(expected)
    current[changed] = {'uid': os.getuid() + 1, 'start_time_ticks': 11,
        'command_sha256': 'different', 'argv': ['another'], 'cwd': '/tmp'}[changed]
    guardian = supervisor(tmp_path, [stage], inventory=lambda: [gpu_row([2345])],
                          identity_reader=lambda pid: current)
    guardian.active_stage = stage.name
    guardian.owned_processes[2345] = expected
    guardian.child = type('Child', (), {'pid': 2345})()
    observation = guardian.observation()
    assert observation['owned_pids'] == []
    assert observation['blocking_processes'][0]['classification'] == 'foreign'
    assert observation['eligible'] is False


def test_completion_failure_has_no_success_receipt_and_followup_never_runs(tmp_path):
    failed = write_stage(tmp_path, complete=False)
    followup = write_stage(tmp_path, 'next')
    guardian = supervisor(tmp_path, [failed, followup])
    with pytest.raises(ValueError, match='incomplete'):
        guardian.run()
    assert not (tmp_path / 'execution/queue_receipts/fit.json').exists()
    assert not Path(followup.outputs[0]).exists()
    assert queue.legacy.read_json(tmp_path / 'execution/queue_state.json')['state'] == 'failed'


def test_completion_validator_hashes_all_published_artifacts(tmp_path):
    artifact = tmp_path / 'weights.pt'
    artifact.write_bytes(b'weights')
    complete = tmp_path / 'complete.json'
    queue.legacy.dump(complete, dict(complete=True, artifacts={'weights.pt': queue.legacy.sha(artifact)}))
    stage = queue.RegisteredStage('fit', (), outputs=(str(complete),), validator='complete_json')
    queue.validate_stage(stage)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        queue.validate_stage(stage)


def test_resource_registration_tamper_and_foreign_output_path_are_rejected(tmp_path):
    study = tmp_path / 'new-study'
    study.mkdir()
    stage = write_stage(study)
    attempt = study / 'attempt'
    protocol = queue.initialize(study, attempt, [stage], {})
    assert queue.check_registration(attempt) == protocol
    protocol['resource_policy']['physical_gpu'] = 4
    queue.legacy.dump(attempt / 'resource_protocol.json', protocol)
    with pytest.raises(ValueError, match='independent SHA'):
        queue.check_registration(attempt)
    outside = queue.RegisteredStage('foreign', ('python', 'x.py'), outputs=(str(tmp_path / 'old.json'),))
    with pytest.raises(ValueError, match='within the new study'):
        queue.initialize(study, study / 'other-attempt', [outside], {})


def test_registered_CPU_run_cannot_retry_or_overwrite_existing_attempt(tmp_path):
    stage = write_stage(tmp_path)
    attempt = tmp_path / 'attempt'
    queue.initialize(tmp_path, attempt, [stage], {})
    assert queue.run(attempt)['state'] == 'completed'
    with pytest.raises(RuntimeError, match='already executed; no automatic retry'):
        queue.run(attempt)
    with pytest.raises(FileExistsError, match='immutable'):
        queue.initialize(tmp_path, attempt, [stage], {})


def test_failed_child_stops_queue_and_forbids_automatic_retry(tmp_path):
    stage = queue.RegisteredStage('fail', (sys.executable, '-c', 'raise SystemExit(7)'), gpu_required=False)
    guardian = supervisor(tmp_path, [stage])
    with pytest.raises(RuntimeError, match='code 7'):
        guardian.run()
    assert queue.legacy.read_json(tmp_path / 'execution/queue_state.json')['scientific_conclusion'] is None
    with pytest.raises(RuntimeError, match='no automatic retry'):
        supervisor(tmp_path, [stage]).run()


class ExitDuringInventory:
    pid = 2345

    def __init__(self, returncode=0):
        self.calls, self.returncode = 0, returncode

    def poll(self):
        self.calls += 1
        return None if self.calls == 1 else self.returncode


def own_exit_guardian(root, *, returncode=0, complete=True, snapshot=None):
    stage = write_stage(root, gpu=True, complete=complete)
    queue.legacy.dump(stage.outputs[0], dict(complete=complete))
    def unknown(pid):
        raise FileNotFoundError('Exited child no longer has a command line')
    guardian = supervisor(root, [stage], inventory=lambda: [gpu_row([2345])],
        identity_reader=unknown, snapshot_reader=lambda pid: snapshot)
    guardian.output.mkdir()
    queue.legacy.dump(guardian.output / 'queue_protocol.json', guardian.registration)
    guardian.protocol_sha = queue.legacy.sha(guardian.output / 'queue_protocol.json')
    guardian.active_stage = stage.name
    guardian.child = ExitDuringInventory(returncode)
    guardian.owned_processes[2345] = dict(pid=2345, uid=os.getuid(), start_time_ticks=10,
        command_sha256='birth', argv=list(stage.command), cwd=str(queue.ROOT))
    return guardian


@pytest.mark.parametrize('snapshot', [None, dict(pid=2345, uid=os.getuid(), start_time_ticks=10,
                                               state='Z', exit_code=0)])
def test_successful_own_exit_race_requires_ended_birth_identity_and_completed_outputs(tmp_path, snapshot):
    guardian = own_exit_guardian(tmp_path, snapshot=snapshot)
    guardian._check()
    proof = queue.legacy.read_json(tmp_path / 'execution/verified_own_exit_fit.json')
    assert proof['completed_artifacts_verified'] is True
    assert not (tmp_path / 'execution/ownership_stop_trigger.json').exists()


@pytest.mark.parametrize('snapshot,returncode', [
    (dict(pid=2345, uid=os.getuid(), start_time_ticks=11, state='Z', exit_code=0), 0),
    (dict(pid=2345, uid=os.getuid(), start_time_ticks=10, state='R', exit_code=None), 0),
    (None, 1),
])
def test_exit_race_never_excuses_reused_live_PID_or_failed_child(tmp_path, snapshot, returncode):
    guardian = own_exit_guardian(tmp_path, snapshot=snapshot, returncode=returncode)
    with pytest.raises(queue.legacy.StopRequested, match='foreign/unknown'):
        guardian._check()
    assert not (tmp_path / 'execution/verified_own_exit_fit.json').exists()


def test_own_success_without_completed_artifact_still_fails_closed(tmp_path):
    guardian = own_exit_guardian(tmp_path, complete=False)
    with pytest.raises(ValueError, match='incomplete'):
        guardian._check()
    assert not (tmp_path / 'execution/verified_own_exit_fit.json').exists()


@pytest.mark.parametrize('gpu,reserved', [(2, 2), (4, 4), (1, 4), (2, 3)])
def test_only_one_authorized_card_and_one_reserved_card(tmp_path, gpu, reserved):
    with pytest.raises(ValueError, match='one of GPU2/GPU4'):
        supervisor(tmp_path, [write_stage(tmp_path)], gpu=gpu, reserved_gpu=reserved)
