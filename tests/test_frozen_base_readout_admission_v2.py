"""Focused CPU mocks for the exact system-display exception."""
import subprocess

import pytest

from scripts.hanguard import frozen_base_readout_admission_v2 as admission


def identity(**changed):
    return dict(admission.DISPLAY_EXPECTED, state="S", proc_exe=None, **changed)


def registered():
    return dict(expected=admission.DISPLAY_EXPECTED, identity=identity(), verified=True)


def entry(pid=9539, kind="G", name="/usr/lib/xorg/Xorg"):
    return dict(pid=pid, type=kind, process_name=name)


def test_only_verified_exact_registered_display_can_be_ignored():
    assert admission.verified_display(entry(), registered(), lambda pid: identity()) is True
    assert admission.verified_display(entry(), None, lambda pid: identity()) is False
    assert admission.verified_display(entry(), dict(registered(), verified=False), lambda pid: identity()) is False


@pytest.mark.parametrize("changed", ("pid", "uid", "start_time_ticks", "command_sha256", "executable"))
def test_any_display_identity_mismatch_blocks(changed):
    current = identity()
    current[changed] = current[changed] + 1 if isinstance(current[changed], int) else "different"
    assert admission.verified_display(entry(), registered(), lambda pid: current) is False


@pytest.mark.parametrize("kind", ("C", "C+G", "G+C", "", None))
def test_compute_mixed_or_unknown_nvidia_type_blocks(kind):
    assert admission.verified_display(entry(kind=kind), registered(), lambda pid: identity()) is False


@pytest.mark.parametrize("bad", (entry(pid=9540), entry(name="/home/other/Xorg"), entry(name=None)))
def test_other_graphics_process_or_name_blocks(bad):
    assert admission.verified_display(bad, registered(), lambda pid: identity()) is False


def test_unreadable_display_identity_and_zombie_never_exempt():
    def unreadable(pid):
        raise PermissionError("foreign proc")
    assert admission.verified_display(entry(), registered(), unreadable) is False
    assert admission.capture_display(unreadable)["verified"] is False
    assert admission.verified_display(entry(), registered(), lambda pid: dict(identity(), state="Z")) is False
    assert admission.verified_display(entry(), registered(), lambda pid: dict(identity(), proc_exe="/wrong/exe")) is False


def mocked_inventory(monkeypatch, compute_pids, entries):
    row = dict(index=2, uuid="GPU-2", free_gib=40, pids=list(compute_pids))
    monkeypatch.setattr(admission.queue.guardian.inventory_module, "gpu_inventory", lambda: [dict(row)])
    processes = "".join("<process_info><pid>{pid}</pid><type>{type}</type><process_name>{process_name}</process_name></process_info>".format(**value) for value in entries)
    xml = f"<nvidia_smi_log><gpu><uuid>GPU-2</uuid><processes>{processes}</processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(admission.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout=xml))


def test_raw_compute_pid_always_blocks_even_if_xml_says_exact_graphics(monkeypatch):
    mocked_inventory(monkeypatch, [9539], [entry()])
    row = admission.full_gpu_inventory(registered(), lambda pid: identity())[0]
    assert row["pids"] == [9539]
    assert row["ignored_verified_display_pids"] == []


def test_exact_display_ignored_but_second_graphics_and_compute_remain_blocking(monkeypatch):
    mocked_inventory(monkeypatch, [12345], [entry(), entry(pid=999, name="other"), entry(pid=12345, kind="C", name="python")])
    row = admission.full_gpu_inventory(registered(), lambda pid: identity())[0]
    assert row["pids"] == [999, 12345]
    assert row["all_nvidia_pids"] == [999, 9539, 12345]
    assert row["ignored_verified_display_pids"] == [9539]


def test_exact_pure_graphics_display_alone_allows_admission(monkeypatch):
    mocked_inventory(monkeypatch, [], [entry()])
    rows = admission.full_gpu_inventory(registered(), lambda pid: identity())
    candidates, _, error = admission.observations(lambda: rows, [], lambda pid: None)
    assert error is None
    assert candidates[0]["eligible"] is True
    assert candidates[0]["pids"] == []
    assert candidates[0]["all_nvidia_pids"] == [9539]


def test_gpu6_pipeline_gate_remains_even_with_display_exception(monkeypatch):
    record = admission.capture_blockers(lambda pid: dict(pid=pid, uid=1047, start_time_ticks=100, state="S"))[0]
    rows = [dict(index=6, uuid="GPU-6", free_gib=40, pids=[], all_nvidia_pids=[9539])]
    candidates, births, _ = admission.observations(lambda: rows, [record],
        lambda pid: dict(pid=pid, uid=1047, start_time_ticks=100, state="S"))
    assert candidates[-1]["eligible"] is False
    assert births[0]["blocked"] is True
