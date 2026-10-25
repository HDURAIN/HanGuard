import os
from pathlib import Path

import pytest

from scripts.hanguard import frozen_base_parallel_queue_v4 as queue
from test_frozen_base_parallel_queue_v3 import bare_scheduler, recovery2
from test_frozen_base_parallel_queue_v2 import Child, identity, recovery

old = queue.original


@pytest.fixture
def recovery3(recovery2, tmp_path, monkeypatch):
    study, control2, worker, _ = recovery2
    coordinator = tmp_path / "coordinator3.py"
    coordinator.write_text("# immutable reviewed head restart fixture\n")
    monkeypatch.setattr(queue, "COORDINATOR_PATH", coordinator)
    folder = study / "recovery_003"
    folder.mkdir()
    old._write(study / "parallel_execution/queue_state.json", dict(
        requires_review=True, active_workers=0, phase="fit"))
    snapshot = folder / "queue_state.before.json"
    snapshot.write_bytes((study / "parallel_execution/queue_state.json").read_bytes())
    archive = study / "fit_attempts/001/heads"
    archive.mkdir(parents=True)
    (archive / "budget_12").mkdir()
    old._write(archive / "fit_started.json", dict(
        protocol_sha256=old._sha(study / "protocol.json"), test_opened=False,
        source_sha256={str(worker): old._sha(worker)}))
    archive_manifest = study / "fit_attempts/001/archive_manifest.json"
    old._write(archive_manifest, dict(files={"heads/fit_started.json": old._sha(archive / "fit_started.json")}))
    worker_id = "fit_0037_gpu4"
    ledger = study / "parallel_execution/workers" / f"{worker_id}.json"
    owned = identity(pid=123456789)
    old._write(ledger, dict(schema_version=1, worker_id=worker_id, phase="fit", physical_gpu=4,
        process_identity=owned, root_protocol_sha256=old._sha(study / "protocol.json"),
        resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json")))
    old._write(ledger.with_suffix(".registration.json"), dict(ledger_sha256=old._sha(ledger)))
    ledger.with_suffix(".log").write_text("")
    event_path = study / "recovery_002/failure_events/failure_event_fixture.json"
    old._write(event_path, dict(phase="fit", reason="Foreign process entered an active GPU",
        recovery_control_sha256=old._sha(study / "recovery_002/control_protocol.json"),
        active_workers=[dict(worker_id=worker_id, physical_gpu=4, process_identity=owned)],
        trigger_evidence=dict(index=4, blocking_processes=[dict(
            classification="foreign", pid=999, uid=os.getuid() + 1, start_time_ticks=789)])))
    termination = folder / "termination_evidence.before.json"
    old._write(termination, dict(workers=[dict(worker_id=worker_id, pid=owned["pid"], exit_code=-15,
        verified_SIGTERM_sent=True, verified_SIGKILL_sent=False, verified_original_child_exited=True)]))
    (study / "parallel_execution/termination_evidence.json").write_bytes(termination.read_bytes())
    bindings = dict(control2["bindings"])
    for path in (Path(queue.__file__).resolve(), coordinator, snapshot, archive_manifest,
            archive / "fit_started.json", event_path, ledger, ledger.with_suffix(".registration.json"),
            ledger.with_suffix(".log"), termination, study / "recovery_002/control_protocol.json",
            study / "recovery_002/control_registration.json"):
        bindings[str(path)] = old._sha(path)
    control3 = dict(schema=queue.SCHEMA, study=str(study),
        root_protocol_sha256=old._sha(study / "protocol.json"),
        resource_protocol_sha256=old._sha(study / "parallel_resource_protocol.json"),
        prior_control_sha256=old._sha(study / "recovery_002/control_protocol.json"),
        unchanged_scientific_settings=old._read(study / "protocol.json")["settings"],
        bindings=bindings, policy=dict(queue.POLICY), automatic_head_retry=False,
        actual_gpu_commands_use_original_registered_entrypoints=True,
        prior_review=dict(queue_state_snapshot=str(snapshot), queue_state_sha256=old._sha(snapshot),
            prior_workers_verified_exited=True, resume_authorized=True,
            reviewed_same_seed_restart_by_root=True, formal_test_opened=False,
            selection_lock_absent=True, partial_checkpoint_reuse=False, automatic_head_retry=False,
            restart_seed=42, archived_fit_attempt=1, new_fit_attempt=2,
            archive_manifest=str(archive_manifest), archive_manifest_sha256=old._sha(archive_manifest),
            reason="confirmed_foreign_gpu_interruption_during_fit"))
    rewrite_control(study, control3)
    return study, control3, worker, coordinator


def rewrite_control(study, control):
    old._write(study / "recovery_003/control_protocol.json", control)
    old._write(study / "recovery_003/control_registration.json", dict(
        protocol_sha256=old._sha(study / "recovery_003/control_protocol.json")))


def test_reviewed_control_preserves_sources_seed_and_registered_workers(recovery3):
    study, control, _, _ = recovery3
    assert queue.check_control3(study) == control
    assert queue.ParallelScheduler.run_workers is queue.prior.ParallelScheduler.run_workers
    assert queue.ParallelScheduler._observe is queue.prior.ParallelScheduler._observe
    assert queue.ParallelScheduler._poll_workers is queue.prior.ParallelScheduler._poll_workers
    assert queue.ParallelScheduler._launch is queue.prior.ParallelScheduler._launch


@pytest.mark.parametrize("field,value", [("restart_seed", 43), ("restart_seed", True),
    ("partial_checkpoint_reuse", True), ("automatic_head_retry", True),
    ("reviewed_same_seed_restart_by_root", False), ("formal_test_opened", True),
    ("archived_fit_attempt", 2), ("new_fit_attempt", 3)])
def test_same_seed_fresh_restart_policy_cannot_change(recovery3, field, value):
    study, control, _, _ = recovery3
    control["prior_review"][field] = value
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="same-seed archive"):
        queue.check_control3(study)


@pytest.mark.parametrize("field,value", [("automatic_head_retry", True),
    ("actual_gpu_commands_use_original_registered_entrypoints", False)])
def test_top_level_retry_and_worker_source_flags_are_enforced(recovery3, field, value):
    study, control, _, _ = recovery3
    control[field] = value
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="resource policy changed"):
        queue.check_control3(study)


def test_changed_scientific_settings_are_rejected(recovery3):
    study, control, _, _ = recovery3
    control["unchanged_scientific_settings"]["epochs"] = 99
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="scientific binding"):
        queue.check_control3(study)


def test_archive_cannot_include_history_or_checkpoint(recovery3):
    study, _, _, _ = recovery3
    (study / "fit_attempts/001/heads/budget_12/layer_history.json").write_text("[]\n")
    with pytest.raises(ValueError, match="zero completed epochs"):
        queue.check_control3(study)


def test_archive_symlink_is_rejected(recovery3):
    study, _, _, _ = recovery3
    (study / "fit_attempts/001/heads/budget_12/foreign").symlink_to(study / "protocol.json")
    with pytest.raises(ValueError, match="zero completed epochs"):
        queue.check_control3(study)


def test_old_source_cannot_be_unbound(recovery3):
    study, control, worker, _ = recovery3
    del control["bindings"][str(worker)]
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="preserve every prior source binding"):
        queue.check_control3(study)


def test_foreign_event_missing_from_bindings_is_rejected(recovery3):
    study, control, _, _ = recovery3
    del control["bindings"][str(study / "recovery_002/failure_events/failure_event_fixture.json")]
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="genuine foreign-GPU interruption"):
        queue.check_control3(study)


def test_old_worker_log_must_be_empty(recovery3):
    study, control, _, _ = recovery3
    path = study / "parallel_execution/workers/fit_0037_gpu4.log"
    path.write_text("epoch completed\n")
    control["bindings"][str(path)] = old._sha(path)
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="exact fit-worker provenance"):
        queue.check_control3(study)


def test_false_foreign_uid_is_not_accepted(recovery3):
    study, control, _, _ = recovery3
    path = study / "recovery_002/failure_events/failure_event_fixture.json"
    event = old._read(path)
    event["trigger_evidence"]["blocking_processes"][0]["uid"] = os.getuid()
    old._write(path, event)
    control["bindings"][str(path)] = old._sha(path)
    rewrite_control(study, control)
    with pytest.raises(ValueError, match="genuine foreign-GPU interruption"):
        queue.check_control3(study)


def test_entry_preserves_root_sticky_state_and_only_runs_same_coordinator(recovery3):
    study, _, _, _ = recovery3
    old._write(study / "state.json", dict(training_started=True, formal_test_opened=False))
    for _ in range(2):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass
    assert old._read(study / "state.json")["training_started"] is True
    state = old._read(study / "parallel_execution/queue_state.json")
    assert state["head_fit_attempt"] == 2
    assert state["current_fit_started"] is False
    assert not (study / "recovery_002/scheduler_recovery_entry.json").exists()


def test_canonical_partial_heads_cannot_be_adopted(recovery3):
    study, _, _, _ = recovery3
    (study / "heads").mkdir()
    with pytest.raises(queue.ResourceReviewRequired, match="cannot adopt canonical partial heads"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass


def test_later_interruption_never_automatically_retries_heads(recovery3):
    study, _, _, _ = recovery3
    with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
        pass
    old._write(study / "parallel_execution/queue_state.json", dict(requires_review=True, phase="fit"))
    with pytest.raises(queue.ResourceReviewRequired, match="automatic head retry remains disabled"):
        with queue.ParallelScheduler(study, inventory=lambda: [], birth_reader=lambda pid: None):
            pass


def test_failure_uses_control3_evidence_once_and_signals_only_recorded_child(tmp_path, monkeypatch):
    child = Child()
    scheduler = bare_scheduler(tmp_path, child, lambda pid: identity(pid=pid))
    scheduler.__class__ = queue.ParallelScheduler
    control_path = tmp_path / "recovery_003/control_protocol.json"
    old._write(control_path, dict(fixture="exact control3"))
    signals = []
    def kill(pid, sig):
        signals.append((pid, sig))
        child.code = -int(sig)
    monkeypatch.setattr(old.os, "kill", kill)
    with pytest.raises(queue.ResourceReviewRequired, match="real foreign"):
        scheduler._resource_failure("real foreign", dict(pid=999))
    events = list((tmp_path / "recovery_003/failure_events").glob("*.json"))
    assert len(events) == 1
    event = old._read(events[0])
    assert event["recovery_control_sha256"] == old._sha(control_path)
    assert event["active_workers"][0]["process_identity"]["pid"] == 12345
    assert {pid for pid, number in signals} == {12345}
    assert not (tmp_path / "recovery_002/failure_events").exists()
