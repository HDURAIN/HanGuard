"""The continuation must wait, adopt frozen results, and never retrain A/base."""
import json
import os
from pathlib import Path

import pytest

from scripts.hanguard import ab_diagnostic_followup as followup
from scripts.hanguard import ab_diagnostic_queue as queue


def expected_process():
    return dict(pid=302664, uid=os.getuid(), start_time_ticks=123,
                command_sha256='original-command')


def test_live_registered_A_blocks_even_before_inspecting_any_checkpoint(tmp_path):
    current = dict(expected_process(), state='R', exit_code=None)
    result = followup.verify_A_completion(tmp_path, expected_process(), lambda pid: current)
    assert result['ready'] is False
    assert not list(tmp_path.iterdir())


def test_zombie_is_ended_without_signalling_paused_parent_and_PID_reuse_is_not_owned():
    current = dict(expected_process(), state='Z', command_sha256='empty-zombie-command', exit_code=0)
    assert followup.original_process_finished(expected_process(), lambda pid: current)['finished']
    reused = dict(current, state='R', start_time_ticks=124)
    result = followup.original_process_finished(expected_process(), lambda pid: reused)
    assert result['finished']
    assert 'no signal' in result['reason']
    with pytest.raises(followup.PrerequisiteFailure, match='unsuccessfully'):
        followup.original_process_finished(expected_process(), lambda pid: dict(current, exit_code=256))


def test_changed_live_A_command_is_rejected():
    changed = dict(expected_process(), state='R', command_sha256='other-command', exit_code=None)
    with pytest.raises(followup.PrerequisiteFailure, match='command differs'):
        followup.original_process_finished(expected_process(), lambda pid: changed)


def test_exact_original_downstream_plan_has_no_A_or_base_training(tmp_path):
    (tmp_path / 'lora_study').mkdir()
    (tmp_path / 'lora_study/protocol.json').write_text('{"seeds": [42]}')
    stages = followup.followup_stages(tmp_path, queue.DEFAULT_PYTHON)
    assert tuple(stage.name for stage in stages) == followup.FOLLOWUP_NAMES
    original = queue.build_plan(tmp_path, queue.DEFAULT_PYTHON)
    for stage in stages:
        assert stage == next(value for value in original if value.name == stage.name)
    assert all('train' not in stage.command for stage in stages)
    assert all('base_last_readouts' != stage.name for stage in stages)


def _A_artifacts(root):
    study = root / 'lora_study'
    run = study / 'runs/E03_s42'
    run.mkdir(parents=True)
    partition = root / 'partition.json'
    partition.write_text(json.dumps(dict(artifact_sha256={'B.parquet': 'B-fixed'})))
    protocol = dict(seeds=[42], partition_manifest_path=str(partition),
        partition_manifest_sha256=queue.sha(partition), data_sha256={'train': 'A-train-fixed', 'validation': 'A-val-fixed'})
    (study / 'protocol.json').write_text(json.dumps(protocol))
    (run / 'adapter.pt').write_bytes(b'fresh-A-adapter')
    (run / 'head.pt').write_bytes(b'A-only-head')
    hashes = {name: queue.sha(run / name) for name in ('adapter.pt', 'head.pt')}
    (run / 'selection.json').write_text(json.dumps(dict(checkpoint_hashes=hashes)))
    lineage = dict(adapter_training_cohort='A', formal_test_opened=False, adapter_sha256=hashes['adapter.pt'],
        protocol_sha256=queue.sha(study / 'protocol.json'), partition_manifest_sha256=queue.sha(partition),
        A_train_sha256='A-train-fixed', A_selection_sha256='A-val-fixed', B_sha256='B-fixed')
    (run / 'diagnostic_lineage.json').write_text(json.dumps(lineage))
    (study / 'state.json').write_text(json.dumps(dict(state='A_LoRA_frozen', adapter_sha256=hashes['adapter.pt'])))
    return run


def test_A_adoption_requires_complete_exact_lineage_and_detects_weight_tamper(tmp_path):
    run = _A_artifacts(tmp_path)
    result = followup.verify_A_completion(tmp_path, expected_process(), lambda pid: None)
    assert result['ready']
    assert result['no_A_retraining']
    lineage = queue.read_json(run / 'diagnostic_lineage.json')
    lineage['B_sha256'] = 'different-B'
    (run / 'diagnostic_lineage.json').write_text(json.dumps(lineage))
    with pytest.raises(followup.PrerequisiteFailure, match='lineage differs'):
        followup.verify_A_completion(tmp_path, expected_process(), lambda pid: None)
    lineage['B_sha256'] = 'B-fixed'
    (run / 'diagnostic_lineage.json').write_text(json.dumps(lineage))
    (run / 'adapter.pt').write_bytes(b'changed-weights')
    with pytest.raises(ValueError, match='checkpoint hash changed'):
        followup.verify_A_completion(tmp_path, expected_process(), lambda pid: None)


def test_failed_base_attempt_stops_followup_without_launching(tmp_path):
    attempt = tmp_path / 'base_attempt'
    attempt.mkdir()
    (attempt / 'state.json').write_text('{"state": "stopped"}')
    with pytest.raises(followup.PrerequisiteFailure, match='without completion'):
        followup.verify_base_completion(tmp_path, attempt)


def test_strict_followup_CPU_stage_does_not_query_GPU_or_signal_existing_processes(tmp_path):
    import sys
    artifact = tmp_path / 'CPU-complete'
    stage = queue.Stage('CPU-analysis', (sys.executable, '-c',
        f'from pathlib import Path; Path({str(artifact)!r}).write_text("done")'), outputs=(str(artifact),))
    def no_gpu_query():
        raise AssertionError('CPU analysis must not query GPU resources')
    supervisor = followup.StrictFollowup(tmp_path, [stage], {}, queue_state_dir=tmp_path / 'execution',
        inventory=no_gpu_query, environment={'CUDA_VISIBLE_DEVICES': '4'}, poll_seconds=.05)
    assert supervisor.run()['state'] == 'completed'
    assert artifact.read_text() == 'done'


def test_ownership_stop_proof_is_kept_in_final_state_without_duplicate_status_keys(tmp_path):
    stage = queue.Stage('extract_adapted', (), validator='cache')
    supervisor = followup.StrictFollowup(tmp_path, [stage], {}, queue_state_dir=tmp_path / 'execution')
    supervisor.output.mkdir()
    supervisor.protocol_sha = 'frozen-queue'
    supervisor.active_stage = 'extract_adapted'
    observation = dict(index=4, uuid='GPU4', blocking_processes=[dict(pid=999, classification='unknown')])
    with pytest.raises(queue.StopRequested, match='foreign/unknown'):
        supervisor.stop_for_ownership(observation)
    before = queue.read_json(supervisor.output / 'ownership_stop_trigger.json')
    assert before['termination_started'] is False
    final = supervisor._state('stopped', reason='test cleanup')
    assert final['trigger_evidence']['trigger_gpu_observation'] == before['trigger_gpu_observation']
    assert final['trigger_evidence']['termination_completed'] is True


def test_continuation_protocol_tamper_is_rejected_by_independent_registration(tmp_path):
    study = tmp_path / 'study'
    study.mkdir()
    protocol = dict(schema=followup.SCHEMA, study=str(study), bindings={},
                    original_A_process_identity=expected_process())
    queue.dump(tmp_path / 'continuation_protocol.json', protocol)
    queue.dump(tmp_path / 'registration.json',
               dict(protocol_sha256=queue.sha(tmp_path / 'continuation_protocol.json')))
    assert followup.check_registration(tmp_path) == protocol
    protocol['original_A_process_identity']['pid'] = 999
    queue.dump(tmp_path / 'continuation_protocol.json', protocol)
    with pytest.raises(ValueError, match='independent SHA registration'):
        followup.check_registration(tmp_path)


def _complete_stopped_base(root, monkeypatch):
    import hashlib
    attempt = root / 'base_attempt'
    readout = root / 'readouts/base_last'
    readout.mkdir(parents=True)
    queue.dump(readout / 'protocol.json', dict(frozen=True))
    protocol_sha = queue.sha(readout / 'protocol.json')
    for fold in range(5):
        for phase in ('probes', 'heads'):
            directory = readout / f'fold_{fold}' / phase
            directory.mkdir(parents=True)
            (directory / 'predictions.parquet').write_bytes(f'fold{fold}-{phase}'.encode())
            queue.dump(directory / 'complete.json', dict(complete=True,
                provenance=dict(outer_fold=fold, stage=phase, protocol_sha256=protocol_sha),
                artifacts={'predictions.parquet': queue.sha(directory / 'predictions.parquet')}))
    commands = {str(fold): ['python', str(queue.ROOT / 'scripts/hanguard/ab_frozen_readout_diagnostics.py'),
        'run', '--output', str(readout), '--fold', str(fold)] for fold in range(5)}
    queue.dump(attempt / 'protocol.json', dict(commands=commands))
    command = ['python', str(queue.ROOT / 'scripts/hanguard/ab_readout_parallel_queue.py'),
               'run', '--study', str(root), '--output', str(attempt)]
    identity = dict(pid=20, uid=os.getuid(), start_time_ticks=400, argv=command,
        command_sha256=hashlib.sha256(('\0'.join(command) + '\0').encode()).hexdigest(), cwd=str(queue.ROOT))
    queue.dump(attempt / 'root_launch.json', dict(pid=20, command=command, process_identity=identity))
    active = {'4': dict(pid=21, gpu=4)}
    queue.dump(attempt / 'state.json', dict(state='stopped', supervisor_pid=20, trigger_active_workers=active))
    queue.dump(attempt / 'ownership_stop_trigger.json', dict(trigger_unknown_ownership_pids=[21],
        trigger_confirmed_foreign_pids=[], trigger_active_workers=active,
        trigger_dispatcher_protocol_sha256=queue.sha(attempt / 'protocol.json')))
    stage = queue.Stage('base_last_readouts', (), outputs=(str(readout / 'analysis/results.json'),))
    monkeypatch.setattr(queue, 'build_plan', lambda *args: [stage])
    return attempt, readout, identity


def test_stopped_base_only_adopts_all_five_verified_folds_from_owned_exit(tmp_path, monkeypatch):
    attempt, readout, _ = _complete_stopped_base(tmp_path, monkeypatch)
    original_state_sha = queue.sha(attempt / 'state.json')
    result = followup.verify_base_completion(tmp_path, attempt, lambda pid: None)
    assert result['needs_CPU_report'] and not result['ready']
    assert result['adoption']['unknown_owned_child_pids'] == [21]
    assert result['adoption']['original_stopped_state_sha256'] == original_state_sha
    assert len(result['completed_fold_artifact_sha256']) == 20
    queue.dump(readout / 'analysis/results.json', dict(complete=True))
    result = followup.verify_base_completion(tmp_path, attempt, lambda pid: None)
    assert result['ready'] and result['no_base_retraining']
    assert queue.sha(attempt / 'state.json') == original_state_sha
    (readout / 'fold_3/heads/predictions.parquet').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        followup.verify_base_completion(tmp_path, attempt, lambda pid: None)


def test_stopped_base_requires_dispatcher_and_every_old_writer_to_have_ended(tmp_path, monkeypatch):
    attempt, _, identity = _complete_stopped_base(tmp_path, monkeypatch)
    with pytest.raises(followup.PrerequisiteFailure, match='dispatcher is still alive'):
        followup.verify_base_completion(tmp_path, attempt,
            lambda pid: dict(identity, state='R') if pid == 20 else None)
    trigger = queue.read_json(attempt / 'ownership_stop_trigger.json')
    trigger['trigger_active_workers']['3'] = dict(pid=22, gpu=0)
    queue.dump(attempt / 'ownership_stop_trigger.json', trigger)
    state = queue.read_json(attempt / 'state.json')
    state['trigger_active_workers'] = trigger['trigger_active_workers']
    queue.dump(attempt / 'state.json', state)
    with pytest.raises(followup.PrerequisiteFailure, match='live writer'):
        followup.verify_base_completion(tmp_path, attempt,
            lambda pid: dict(pid=22, state='R') if pid == 22 else None)


@pytest.mark.parametrize('change', ('foreign', 'unregistered_unknown', 'unsuccessful_exit'))
def test_stopped_base_never_adopts_foreign_unregistered_or_failed_exit(tmp_path, monkeypatch, change):
    attempt, _, _ = _complete_stopped_base(tmp_path, monkeypatch)
    reader = lambda pid: None
    trigger = queue.read_json(attempt / 'ownership_stop_trigger.json')
    if change == 'foreign':
        trigger['trigger_confirmed_foreign_pids'] = [999]
    elif change == 'unregistered_unknown':
        trigger['trigger_unknown_ownership_pids'] = [999]
    else:
        reader = lambda pid: dict(pid=21, state='Z', exit_code=256) if pid == 21 else None
    queue.dump(attempt / 'ownership_stop_trigger.json', trigger)
    with pytest.raises(followup.PrerequisiteFailure):
        followup.verify_base_completion(tmp_path, attempt, reader)


class _ExitDuringInventory:
    pid = 123
    def __init__(self, exit_code):
        self.calls, self.exit_code = 0, exit_code
    def poll(self):
        self.calls += 1
        return None if self.calls == 1 else self.exit_code


def _race_supervisor(root, classification, exit_code, artifact=True):
    output = root / 'execution'
    output.mkdir()
    target = root / 'completed'
    if artifact:
        target.write_text('done')
    stage = queue.Stage('adapted_last_readouts', (), outputs=(str(target),))
    supervisor = followup.StrictFollowup(root, [stage], {}, queue_state_dir=output)
    queue.dump(output / 'queue_protocol.json', dict(frozen=True))
    supervisor.protocol_sha = queue.sha(output / 'queue_protocol.json')
    supervisor.active_stage = stage.name
    supervisor.gpu_stages.add(stage.name)
    supervisor.child = _ExitDuringInventory(exit_code)
    supervisor.owned_processes[123] = dict(pid=123)
    supervisor.observation = lambda: dict(index=4, uuid='GPU4',
        blocking_processes=[dict(pid=123, classification=classification)])
    return supervisor


def test_exit_between_poll_and_inventory_is_only_ignored_for_verified_own_success(tmp_path):
    supervisor = _race_supervisor(tmp_path, 'unknown', 0)
    supervisor._check()
    proof = queue.read_json(supervisor.output / 'verified_own_exit_cleanup.json')
    assert proof['pid'] == 123 and proof['exit_code'] == 0
    assert proof['completed_artifacts_verified']
    assert not (supervisor.output / 'ownership_stop_trigger.json').exists()


@pytest.mark.parametrize('classification,exit_code', [('foreign', 0), ('unknown', None), ('unknown', 1)])
def test_exit_race_does_not_excuse_foreign_live_unknown_or_unsuccessful_child(tmp_path, classification, exit_code):
    supervisor = _race_supervisor(tmp_path, classification, exit_code)
    with pytest.raises(queue.StopRequested, match='foreign/unknown'):
        supervisor._check()
    assert not (supervisor.output / 'verified_own_exit_cleanup.json').exists()


def test_own_exit_race_still_requires_complete_validated_artifacts(tmp_path):
    supervisor = _race_supervisor(tmp_path, 'unknown', 0, artifact=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        supervisor._check()


def test_attempt_STOP_during_execution_terminates_only_the_new_owned_child(tmp_path):
    import sys
    attempt = tmp_path / 'continuation'
    attempt.mkdir()
    stop = attempt / 'STOP'
    stage = queue.Stage('CPU-long-child', (sys.executable, '-c',
        f'import time; from pathlib import Path; time.sleep(.1); Path({str(stop)!r}).write_text("test stop"); time.sleep(20)'))
    supervisor = followup.StrictFollowup(tmp_path, [stage], {}, queue_state_dir=attempt / 'execution',
        poll_seconds=.02, terminate_seconds=1)
    with pytest.raises(queue.StopRequested, match='test stop'):
        supervisor.run()
    assert supervisor.child.poll() is not None
    assert queue.read_json(attempt / 'execution/queue_state.json')['state'] == 'stopped'


def test_base_aggregation_uses_original_report_arguments_and_disables_GPU(tmp_path, monkeypatch):
    study = tmp_path / 'study'
    command = ('python', 'readout.py', 'run', '--device', 'cuda', '--head-epochs', '12', '--joint-features')
    original = queue.Stage('base_last_readouts', command, ('input',), ('output',), 'heads')
    monkeypatch.setattr(queue, 'build_plan', lambda *args: [original])
    recorded = {}
    class Capture:
        def __init__(self, study, stages, bindings, **kwargs):
            recorded.update(stages=stages, kwargs=kwargs)
            self.stop_directories = (Path(study), kwargs['queue_state_dir'])
        def run(self):
            recorded['ran'] = True
    monkeypatch.setattr(queue, 'SequentialSupervisor', Capture)
    followup.finalize_verified_base_CPU(dict(study=str(study), python='python', bindings={}), tmp_path / 'attempt')
    changed = list(command)
    changed[2], changed[4] = 'report', 'cpu'
    assert recorded['stages'][0].command == tuple(changed)
    assert recorded['stages'][0].inputs == original.inputs
    assert recorded['kwargs']['environment']['CUDA_VISIBLE_DEVICES'] == ''
    assert recorded['ran']
