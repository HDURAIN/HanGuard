"""CPU integration checks for the selected-122 wrapper's scientific contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from scripts.hanguard import fragment_clean_lora_study as historical
from scripts.hanguard import selected_subset_lora_study as wrapper

study = wrapper._runtime
_spec = importlib.util.spec_from_file_location(
    '_selected122_shared_cpu_checks', wrapper.ROOT / 'tests/test_fragment_clean_lora_study.py')
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)


@pytest.fixture(autouse=True)
def small_cpu_pool(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(shared, 'study', study)
    yield
    torch.set_num_threads(before)


def test_private_runtime_and_worker_entrypoint_are_bound_to_wrapper(tmp_path, monkeypatch):
    assert historical.SCHEMA == 'hanguard_fragment_clean_fresh_lora_v1'
    assert historical.READOUT_SCHEMA == 'hanguard_fragment_clean_frozen_lora_readouts_v1'
    assert Path(historical.__file__).resolve() == wrapper.IMPLEMENTATION
    assert Path(study.__file__).resolve() == Path(wrapper.__file__).resolve()
    assert wrapper.parser().parse_args(['status', '--output', str(tmp_path)]).data == wrapper.DEFAULT_DATA
    assert wrapper.IMPLEMENTATION in wrapper.source_paths()
    assert Path(wrapper.__file__).resolve() in wrapper.source_paths()
    study.write(tmp_path / 'protocol.json', dict(schema=wrapper.SCHEMA))
    monkeypatch.setattr(study, 'verify', lambda *_a, **_k: dict(
        gpu_ids=[2], minimum_free_gib=36., poll_seconds=.001))
    commands = []
    child = SimpleNamespace(pid=123, poll=lambda: None)

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2'
        return child

    def identity(pid):
        return dict(pid=pid, uid=study.os.getuid(), argv=commands[-1], cwd=str(wrapper.ROOT))

    scheduler = study.Scheduler(tmp_path,
        inventory_reader=lambda: [dict(index=2, pids=[], free_gib=46.)],
        identity_reader=identity, popen=popen, sleeper=lambda _s: None)
    assert scheduler.launch(2, 'lora') is True
    assert commands[0][1] == str(Path(wrapper.__file__).resolve())
    ledger = study.read(next((tmp_path / 'workers').glob('*_gpu2.json')))
    assert ledger['command'][1] == str(Path(wrapper.__file__).resolve())


def test_cpu_registration_locks_both_sources_and_posthoc_release(tmp_path, monkeypatch):
    # Reuse the complete synthetic registration contract test. Add the explicit
    # subset metadata that the real release must contain before initialization.
    validate = wrapper.validate_selection_policy

    def add_synthetic_selection_policy(data):
        data = Path(data)
        manifest = study.read(data / 'manifest.json')
        manifest.update(selection_policy=wrapper.SELECTION_POLICY,
                        excluded_test_rows=122, train_validation_byte_identical=True,
                        is_quality_cleanup=False)
        study.write(data / 'manifest.json', manifest)
        study.write(data / 'filter_policy.json', dict(selection_policy=wrapper.SELECTION_POLICY,
            selected_rows=122, train_validation_byte_identical=True, is_quality_cleanup=False))
        (data / 'excluded_base_ids.txt').write_text('synthetic_excluded_id\n')
        qa = study.read(data / 'release_qa.json')
        qa['manifest_sha256'] = study.sha(data / 'manifest.json')
        study.write(data / 'release_qa.json', qa)
        validate(data)

    monkeypatch.setattr(wrapper, 'validate_selection_policy', add_synthetic_selection_policy)
    shared.test_cpu_registration_binds_new_release_and_never_initializes_cuda(tmp_path, monkeypatch)
    protocol = study.read(tmp_path / 'study/protocol.json')
    child = study.read(tmp_path / 'study/lora/protocol.json')
    assert protocol['schema'] == wrapper.SCHEMA
    assert protocol['scope'] == wrapper.SCOPE
    assert protocol['scientific_scope'] == wrapper.SELECTION_POLICY
    assert protocol['is_quality_cleanup'] is False
    assert protocol['blind_test'] is False
    assert protocol['limitations'] == wrapper.LIMITATIONS
    for source in (wrapper.IMPLEMENTATION, Path(wrapper.__file__).resolve()):
        assert protocol['code_sha256'][str(source)] == study.sha(source)
    for name in ('filter_policy.json', 'excluded_base_ids.txt'):
        path = tmp_path / 'fragment_release' / name
        assert protocol['inputs'][str(path)] == study.sha(path)
    assert child['scientific_scope'] == wrapper.SELECTION_POLICY
    assert child['limitations'] == wrapper.LIMITATIONS
    assert child['warm_head_epochs'] == 1 and child['epochs'] == 3
    registration = study.read(tmp_path / 'study/lora/registration.json')
    assert str(Path(wrapper.__file__).resolve().relative_to(wrapper.ROOT)) in registration['code_sha256']


def test_wrong_selection_policy_rejected_before_registration(tmp_path):
    data = tmp_path / 'data'
    study.write(data / 'manifest.json', dict(selection_policy='text_only_filter'))
    study.write(data / 'filter_policy.json', dict(selection_policy='text_only_filter'))
    output = tmp_path / 'must_not_exist'
    args = wrapper.parser().parse_args(['init', '--data', str(data), '--output', str(output)])
    with pytest.raises(ValueError, match='post hoc 122-row'):
        wrapper.initialize(args)
    assert not output.exists()


def test_readout_registration_uses_wrapper_snapshot_and_same_joint_adapter(tmp_path, monkeypatch):
    from scripts.hanguard import remediated_dataset_contract as contract
    data = tmp_path / 'data'
    data.mkdir()
    parent = dict(schema=wrapper.SCHEMA, data_dir=str(data), inputs={}, input_stats={},
        code_sha256={str(Path(wrapper.__file__).resolve()): study.sha(wrapper.__file__)})
    study.write(tmp_path / 'protocol.json', parent)
    adapter = dict(adapter_path=str(tmp_path / 'adapter.pt'), adapter_sha256='same_adapter',
        selection_path=str(tmp_path / 'selection.json'), selection_sha256='same_selection',
        lora_rank=8, lora_alpha=16, lora_dropout=.05,
        selected_lora_phase=dict(phase_name='joint', epoch=1),
        actual_selected_phase=dict(phase_name='joint', epoch=1),
        adapter_delta=dict(nonzero_lora_b_tensors=1), rows=dict(train=4, validation=4, test=4))
    monkeypatch.setattr(study, 'verify', lambda *_a, **_k: parent)
    monkeypatch.setattr(study, 'selected_adapter', lambda *_a, **_k: adapter)
    monkeypatch.setattr(contract, 'release_contract', lambda *_a, **_k: dict(dataset_version='synthetic'))
    registered = study.initialize_readouts(tmp_path)
    assert registered['schema'] == wrapper.READOUT_SCHEMA
    assert registered['scientific_scope'] == wrapper.SELECTION_POLICY
    assert registered['scope'] == wrapper.SCOPE
    assert registered['adapter_sha256'] == 'same_adapter'
    assert registered['use_lora'] is True
    assert (tmp_path / 'readouts/adapter_readout_runtime.py.snapshot').read_bytes() == Path(wrapper.__file__).read_bytes()
    assert study.verify_readouts(tmp_path / 'readouts')['adapter_sha256'] == 'same_adapter'


def test_all_five_arms_resume_and_results_keep_scope_and_adapter(tmp_path, monkeypatch):
    shared.test_five_arm_cpu_fit_evaluate_and_epoch_resume_keep_lora_identity(tmp_path, monkeypatch)
    folder = tmp_path / 'readouts'
    for path in (folder / 'heads/selection.json', folder / 'analysis/results.json'):
        artifact = study.read(path)
        assert artifact['schema'] == wrapper.READOUT_SCHEMA
        assert artifact['scientific_scope'] == wrapper.SELECTION_POLICY
        assert artifact['scope'] == wrapper.SCOPE
        assert artifact['limitations'] == wrapper.LIMITATIONS
        assert artifact['adapter_sha256'] == 'adapter_sha'
        assert artifact['no_lora'] is False and artifact['use_lora'] is True
        assert artifact['blind_test'] is False and artifact['is_quality_cleanup'] is False
    result = study.read(folder / 'analysis/results.json')
    assert len(result['metrics']) == 5
    assert '+2.10' in ' '.join(result['limitations'])


def test_test_barrier_still_precedes_read(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, 'read_parquet', lambda *_a, **_k: pytest.fail('Test payload read without lock'))
    shared.test_test_barrier_precedes_data_read(tmp_path, monkeypatch)


def test_warmup_selected_adapter_still_stops_for_review(tmp_path):
    shared.test_warmup_selection_stops_without_changing_selection(tmp_path)


def test_admission_and_foreign_arrival_policy_is_unchanged(tmp_path, monkeypatch):
    shared.test_foreign_compute_blocks_idle_card()
    shared.test_prelaunch_foreign_card_never_spawns(tmp_path, monkeypatch)
    shared.test_foreign_arrival_is_record_only(tmp_path, monkeypatch)
