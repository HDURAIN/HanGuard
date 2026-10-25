"""CPU checks for the one-shot external holdout protocol."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import external_holdout_evaluation as external
from scripts.hanguard.frozen_lora_layer_diagnostics import dump, sha256


def _build_registry(tmp_path: Path) -> Path:
    registry = tmp_path / 'registry'
    registry.mkdir()
    source = tmp_path / 'source.parquet'
    source_frame = pd.DataFrame({
        'prompt': ['Ａ B\u200b', 'harm one', 'safe two', 'harm one'],
        'prompt_harm_label': ['unharmful', 'harmful', 'unharmful', 'harmful'],
        'category_id': ['0', '1', '0', '1'],
    })
    source_frame.to_parquet(source, index=False)
    rows = []
    for source_row, row in source_frame.iterrows():
        prompt = row.prompt
        exact = external.sha256_text(prompt)
        normalized = external.normalized_prompt_sha256(prompt)
        source_name = 'toy_source'
        rows.append({
            'base_id': external.expected_base_id(source_name, source_row, exact),
            'group_id': external.expected_group_id(normalized),
            'source': source_name,
            'source_file': str(source),
            'source_row': source_row,
            'prompt_sha256': exact,
            'normalized_prompt_sha256': normalized,
            'prompt_harm_label': row.prompt_harm_label,
            'label': int(row.prompt_harm_label == 'harmful'),
            'category_id': row.category_id,
        })
    index = pd.DataFrame(rows).sort_values('base_id').reset_index(drop=True)
    index_path = registry / 'toy_index.parquet'
    index.to_parquet(index_path, index=False)
    builder = registry / 'build.py'
    builder.write_text('# frozen registry builder\n', encoding='utf-8')
    manifest = {
        'schema': external.REGISTRY_SCHEMA,
        'prediction_status': 'none',
        'toy': {
            'role': 'test fixture',
            'source_file': str(source),
            'source_sha256': sha256(source),
            'source_bytes': source.stat().st_size,
            'source_rows': len(source_frame),
            'selected_rows': len(index),
            'index_file': str(index_path),
            'index_sha256': sha256(index_path),
            'index_bytes': index_path.stat().st_size,
            'ordered_base_id_sha256': external.registry_ordered_base_id_sha256(index.base_id),
        },
    }
    dump(registry / 'manifest.json', manifest)
    files = {}
    for path in (builder, index_path, registry / 'manifest.json'):
        files[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    dump(registry / 'lock.json', {'state': 'registered_unconsumed', 'files': files})
    return registry


def _build_current_study_artifacts(
    tmp_path: Path, *, stale_frozen_code: bool = False,
) -> tuple[Path, Path, Path]:
    """Small byte-complete artifacts with the current dynamic/OOF JSON schemas."""
    frozen_source = tmp_path / 'frozen_source'
    model = tmp_path / 'base_model'
    model.mkdir()
    (model / 'config.json').write_text('{"hidden_size": 4}\n', encoding='utf-8')
    (model / 'tokenizer.json').write_text('{"version": "fixture"}\n', encoding='utf-8')
    shard = model / 'model-00001.safetensors'
    shard.write_bytes(b'fixture base weights')
    dump(model / 'model.safetensors.index.json', {
        'weight_map': {'fixture.weight': shard.name},
    })
    adapter = tmp_path / 'adapter.pt'
    adapter.write_bytes(b'fixture frozen lora')
    data_dir = tmp_path / 'internal_data'
    data_dir.mkdir()
    for split in ('train', 'validation', 'test'):
        pd.DataFrame({'base_id': [f'{split}-row']}).to_parquet(data_dir / f'{split}.parquet', index=False)
    frozen_source.mkdir()
    frozen_code_identity = external.frozen._code_identity()
    if stale_frozen_code:
        frozen_code_identity['frozen_lora_layer_diagnostics.py'] = '0' * 64
    source_protocol = {
        'version': 1,
        'data_dir': str(data_dir),
        'data_sha256': {
            split: sha256(data_dir / f'{split}.parquet')
            for split in ('train', 'validation', 'test')
        },
        'rows': {'train': 2, 'validation': 2, 'test': 2},
        'model_path': str(model),
        'model_artifacts': external.frozen._model_artifact_identity(model),
        'model_config_sha256': sha256(model / 'config.json'),
        'tokenizer_sha256': sha256(model / 'tokenizer.json'),
        'adapter_path': str(adapter),
        'adapter_sha256': sha256(adapter),
        'layers': list(external.frozen.ALL_LAYERS),
        'code_sha256': frozen_code_identity,
    }
    dump(frozen_source / 'protocol.json', source_protocol)
    feature_dir = frozen_source / 'features'
    feature_dir.mkdir()
    dump(feature_dir / 'train.manifest.json', {
        'split': 'train',
        'layers': 32,
        'hidden_size': 4,
        'adapter_sha256': source_protocol['adapter_sha256'],
        'protocol_sha256': sha256(frozen_source / 'protocol.json'),
        'complete': True,
        'representation': 'raw block output at final valid token, selected inside each forward hook',
    })
    for split in ('validation', 'test'):
        dump(feature_dir / f'{split}.manifest.json', {
            'split': split, 'complete': True, 'rows': 2, 'layers': 32, 'hidden_size': 4,
        })
    metadata = pd.DataFrame({
        'base_id': ['a', 'b'], 'group_id': ['ga', 'gb'], 'label': [0, 1],
        'source': ['fixture', 'fixture'], 'category_id': ['0', '1'],
        'text_form': ['direct', 'direct'],
    })
    metadata.to_parquet(feature_dir / 'train.metadata.parquet', index=False)
    metadata.assign(base_id=['c', 'd'], group_id=['gc', 'gd']).to_parquet(
        feature_dir / 'validation.metadata.parquet', index=False,
    )
    dump(frozen_source / 'selected_layers.json', {
        'split': 'validation', 'layers': [4, 12, 20, 32],
    })

    dynamic_study = tmp_path / 'dynamic'
    dynamic_study.mkdir()
    selected_layers = [4, 12, 20, 32]
    dynamic_protocol = {
        'source': str(frozen_source),
        'source_protocol_sha256': sha256(frozen_source / 'protocol.json'),
        'selected_layers_sha256': sha256(frozen_source / 'selected_layers.json'),
        'cache_manifest_sha256': {
            split: sha256(feature_dir / f'{split}.manifest.json')
            for split in ('train', 'validation', 'test')
        },
        'adapter_sha256': source_protocol['adapter_sha256'],
        'data_dir': str(data_dir),
        'data_sha256': source_protocol['data_sha256'],
        'rows': source_protocol['rows'],
        'selected_layers': selected_layers,
        'fixed_layers': [8, 16, 24, 32],
        'arms': list(external.dynamic.ARMS),
        'architecture': {
            'width': 8, 'gate_width': 4, 'layer_count': 4,
            'dropout': .1, 'fusion_init_scale': 1e-3,
        },
        'training': {'seeds': [42, 43, 44, 45, 46], 'batch_size': 8},
        'code_sha256': {
            'dynamic_gate_diagnostics.py': sha256(Path(external.dynamic.__file__)),
        },
    }
    dump(dynamic_study / 'protocol.json', dynamic_protocol)
    entries = []
    for seed in dynamic_protocol['training']['seeds']:
        directory = dynamic_study / 'heads' / f'seed_{seed}'
        directory.mkdir(parents=True)
        arms = {}
        for arm in external.dynamic.ARMS:
            checkpoint = directory / f'{arm}.pt'
            validation_bce = {
                'dynamic_repeat_last': .30,
                'dynamic_fixed': .20 + (seed - 42) * .001,
                'dynamic_selected': .24 + (seed - 42) * .001,
            }[arm]
            head = external.dynamic.DynamicLayerGateHead(
                hidden_size=4, width=8, gate_width=4, layer_count=4,
                dropout=.1, fusion_init_scale=1e-3,
            )
            checkpoint_value = {
                'state_dict': {key: value.detach().clone() for key, value in head.state_dict().items()},
                'arm': arm,
                'seed': seed,
                'selected_layers': selected_layers,
                'input_layers': list(external.dynamic.arm_layers(arm, selected_layers)),
                'hidden_size': 4,
                'width': 8,
                'gate_width': 4,
                'best_epoch': 2,
                'best_validation_bce': validation_bce,
                'protocol_sha256': sha256(dynamic_study / 'protocol.json'),
            }
            external.frozen.atomic_torch_save(checkpoint_value, checkpoint)
            arms[arm] = {
                'parameters': sum(parameter.numel() for parameter in head.parameters()),
                'best_epoch': 2,
                'threshold': .5,
                'validation_bce': validation_bce,
                'checkpoint_sha256': sha256(checkpoint),
            }
        predictions = directory / 'validation_predictions.parquet'
        pd.DataFrame({'base_id': ['validation'], 'label': [0]}).to_parquet(predictions, index=False)
        selection = {
            'split': 'validation',
            'seed': seed,
            'selected_layers': selected_layers,
            'arms': arms,
            'validation_predictions_sha256': sha256(predictions),
        }
        dump(directory / 'selection.json', selection)
        entries.append(selection)
    dump(dynamic_study / 'heads/selection.json', {
        'split': 'validation',
        'seeds': dynamic_protocol['training']['seeds'],
        'selected_layers': selected_layers,
        'matched_arms': list(external.dynamic.ARMS),
        'entries': entries,
    })
    external.frozen.create_test_barrier(
        dynamic_study,
        external.dynamic._barrier_paths(dynamic_study, dynamic_protocol),
        external.dynamic._barrier_identity(dynamic_study, dynamic_protocol),
    )

    oof_study = tmp_path / 'oof'
    (oof_study / 'models').mkdir(parents=True)
    oof_protocol = {
        'source': str(frozen_source),
        'source_protocol_sha256': sha256(frozen_source / 'protocol.json'),
        'source_adapter_sha256': source_protocol['adapter_sha256'],
        'hidden_size': 4,
        'candidate_layers': list(range(1, 33)),
        'fixed_layers': [8, 16, 24, 32],
        'probes': {'batch_size': 8, 'seed': 17},
        'train_metadata': external.oof._metadata_identity(feature_dir / 'train.metadata.parquet'),
        'validation_metadata': external.oof._metadata_identity(
            feature_dir / 'validation.metadata.parquet'
        ),
        'source_feature_artifacts': {
            key: {'path': str(path), 'sha256': sha256(path)}
            for key, path in {
                'train_manifest': feature_dir / 'train.manifest.json',
                'train_metadata': feature_dir / 'train.metadata.parquet',
                'validation_manifest': feature_dir / 'validation.manifest.json',
                'validation_metadata': feature_dir / 'validation.metadata.parquet',
            }.items()
        },
        'code_sha256': external.oof._code_hashes(),
        'external_oof': None,
    }
    dump(oof_study / 'protocol.json', oof_protocol)
    calibration_selection = {
        'selected': 'temperature',
        'tie_tolerance': 1e-6,
        'selection_source': external.OOF_CALIBRATION_SELECTION_SOURCE,
        'criterion': 'minimum group-equal BCE; within tolerance prefer fewer calibration parameters',
        'comparisons': {},
    }
    learned_selection = {'layers': selected_layers, 'trace': [], 'selection_scope': 'strict nested outer-group-held-out'}
    learned_file = {
        **learned_selection,
        'selection_split': 'train strict nested outer-group-OOF',
        'metric': 'nested outer-heldout calibrated group-equal BCE',
    }
    dump(oof_study / 'learned_layers.json', learned_file)
    dump(oof_study / 'calibration_selection.json', calibration_selection)
    arm_layers = {
        'single32': [32],
        'repeat32': [32, 32, 32, 32],
        'fixed4': [8, 16, 24, 32],
        'selected4': selected_layers,
    }
    arm_models = {}
    for arm in external.oof.ALL_ARMS:
        width = len(arm_layers[arm])
        arm_models[arm] = {
            'layers': arm_layers[arm],
            'C': .1,
            'metrics': {'group_equal': {'bce': .2}},
            'search': [],
            'stack': {
                'mean': [0.] * width, 'scale': [1.] * width,
                'coef': [1.] * width, 'intercept': 0., 'C': .1, 'iterations': 1,
            },
            'calibrator': {
                'method': 'temperature', 'slope': 1., 'intercept': 0.,
                'parameter_count': 1,
                'fit_source': external.OOF_CALIBRATOR_FIT_SOURCE,
            },
            'coefficients': {},
            'parameter_count': {'branch_probes': width * 5, 'stack': width + 1, 'calibrator': 1, 'total': width * 6 + 2},
            'metric_scope': 'strict nested outer-group-held-out',
        }
        dump(oof_study / 'models' / f'{arm}.json', arm_models[arm])
    checkpoint = oof_study / 'base_probes/full_probes.pt'
    checkpoint.parent.mkdir()
    pool = external.oof.BranchProbes(
        4, 32, oof_protocol['probes']['seed'],
        [external.oof.layer_slot(layer) for layer in oof_protocol['candidate_layers']],
    )
    repeat_pool = external.oof.BranchProbes(
        4, 4, oof_protocol['probes']['seed'], [0, 1, 2, 3],
    )
    checkpoint_identity = {
        'split': 'train',
        'rows': oof_protocol['train_metadata']['rows'],
        'protocol_sha256': sha256(oof_study / 'protocol.json'),
        'base_id_sha256': oof_protocol['train_metadata']['ordered_base_id_sha256'],
    }
    external.frozen.atomic_torch_save(
        external.oof._checkpoint_payload(
            pool, repeat_pool, oof_protocol['candidate_layers'], checkpoint_identity, [],
        ),
        checkpoint,
    )
    bundle = oof_study / 'base_probes/oof_logits.npz'
    with bundle.open('wb') as stream:
        np.savez(stream, fixture=np.array([1], dtype=np.int8))
    bundle_manifest = oof_study / 'base_probes/oof_logits.manifest.json'
    dump(bundle_manifest, {
        'complete': True,
        'bundle_sha256': sha256(bundle),
        'full_checkpoint': str(checkpoint),
        'full_checkpoint_sha256': sha256(checkpoint),
        'fold_artifacts': {},
        'nested_artifacts': {},
        'nested_pair_artifacts': {},
    })
    oof_predictions = oof_study / 'oof_predictions.parquet'
    validation_predictions = oof_study / 'validation_predictions.parquet'
    pd.DataFrame({'base_id': ['a'], 'label': [0]}).to_parquet(oof_predictions, index=False)
    pd.DataFrame({'base_id': ['c'], 'label': [0]}).to_parquet(validation_predictions, index=False)
    selection = {
        'protocol_sha256': sha256(oof_study / 'protocol.json'),
        'learned_layers': learned_selection,
        'train_candidate': 'fixed4',
        'validation_gate': {'passed': True},
        'calibration_selection': calibration_selection,
        'selected_arm': 'fixed4',
        'validation_metrics': {},
        'matched_parameter_counts': {},
        'arms': arm_models,
        'oof_bundle_sha256': sha256(bundle),
        'full_probe_checkpoint': str(checkpoint),
        'full_probe_checkpoint_sha256': sha256(checkpoint),
        'oof_predictions_sha256': sha256(oof_predictions),
        'validation_predictions_sha256': sha256(validation_predictions),
    }
    dump(oof_study / 'selection.json', selection)
    oof_artifacts = {
        str(path.resolve()): sha256(path)
        for path in external.oof._barrier_artifacts(oof_study, oof_protocol)
    }
    dump(oof_study / 'test_barrier.json', {
        'locked': True,
        'protocol_sha256': sha256(oof_study / 'protocol.json'),
        'artifacts': oof_artifacts,
        'test_inputs': {},
        'selected_arm': 'fixed4',
        'threshold': .5,
    })
    return frozen_source, dynamic_study, oof_study


def _write_completed_external_cache(output: Path, snapshot: dict, dataset: str = 'toy') -> None:
    index = pd.read_parquet(snapshot['datasets'][dataset]['index_path'])
    metadata = index.drop(columns=['category_id'], errors='ignore').copy()
    features = np.arange(len(metadata) * 32 * 4, dtype=np.float16).reshape(len(metadata), 32, 4)
    final, _, _, manifest_path, metadata_path = external._cache_paths(output, dataset)
    final.parent.mkdir(parents=True, exist_ok=True)
    np.save(final, features)
    metadata.to_parquet(metadata_path, index=False)
    ordered = external.frozen.ordered_base_id_sha256(metadata.base_id.astype(str).tolist())
    validation = external.frozen.validate_feature_cache(
        final, features.shape, np.float16, ordered, metadata.base_id.astype(str).tolist(),
    )
    dump(manifest_path, {
        'dataset': dataset,
        'shape': list(features.shape),
        'dtype': 'float16',
        'evaluation_lock_sha256': sha256(output / 'evaluation_lock.json'),
        'dataset_snapshot': snapshot['datasets'][dataset],
        'adapter_sha256': 'fixture',
        'representation': 'raw block output at final valid token',
        'prompt_mode': 'direct source-language prompt; no translation',
        'truncation': False,
        'complete': True,
        'rows': len(metadata),
        'layers': 32,
        'hidden_size': 4,
        'bytes': validation['bytes'],
        'sha256': sha256(final),
        'metadata_sha256': sha256(metadata_path),
        'ordered_base_id_sha256': ordered,
    })


def test_registry_recovers_source_rows_and_verifies_every_prompt_identity(tmp_path):
    registry = _build_registry(tmp_path)
    snapshot = external.validate_registry(registry)
    metadata, prompts, evidence = external.recover_prompts(registry, 'toy')

    assert snapshot['datasets']['toy']['rows'] == 4
    assert evidence['identity_checks'] == {
        'prompt_sha256': True,
        'normalized_prompt_sha256': True,
        'group_id': True,
        'base_id': True,
        'prompt_harm_label': True,
        'binary_label': True,
    }
    assert len(prompts) == len(metadata) == 4
    assert 'prompt' not in metadata
    # NFKC, casefold, Cf removal, and whitespace removal are all observable.
    assert external.normalize_prompt('Ａ B\u200b') == 'ab'

    source = Path(snapshot['datasets']['toy']['source_path'])
    changed = pd.read_parquet(source)
    changed.loc[0, 'prompt'] = 'changed after registration'
    changed.to_parquet(source, index=False)
    with pytest.raises(ValueError, match='source parquet changed'):
        external.validate_registry(registry)


def test_evaluation_lock_freezes_registry_and_model_artifacts_before_access(tmp_path):
    registry = _build_registry(tmp_path)
    snapshot = external.validate_registry(registry)
    checkpoint = tmp_path / 'checkpoint.pt'
    selection = tmp_path / 'selection.json'
    checkpoint.write_bytes(b'frozen checkpoint bytes')
    selection.write_text('{"threshold": 0.5}', encoding='utf-8')
    output = tmp_path / 'evaluation'
    lock = external.create_evaluation_lock(
        output,
        snapshot,
        {'model/checkpoint': checkpoint, 'model/selection': selection},
        {'datasets': ['toy'], 'threshold': .5, 'ensemble': 'fixed mean'},
    )

    assert lock['locked_before_prediction'] is True
    assert external.verify_evaluation_lock(output)['evaluation']['threshold'] == .5
    consumed = external.mark_consumed(output)
    assert consumed['datasets'] == ['toy']
    assert (registry / 'consumed.json').is_file()
    assert external.mark_consumed(output) == consumed

    checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact.*changed'):
        external.verify_evaluation_lock(output)


def test_access_marker_cannot_be_created_without_pre_prediction_lock(tmp_path):
    with pytest.raises(FileNotFoundError, match='locked before model access'):
        external.mark_consumed(tmp_path / 'unlocked')


def test_dynamic_primary_is_chosen_only_from_mean_validation_bce_with_fixed_tie_break():
    selections = {
        42: {
            'split': 'validation', 'seed': 42,
            'arms': {
                'dynamic_fixed': {'validation_bce': .20},
                'dynamic_selected': {'validation_bce': .18},
            },
        },
        43: {
            'split': 'validation', 'seed': 43,
            'arms': {
                'dynamic_fixed': {'validation_bce': .10},
                'dynamic_selected': {'validation_bce': .14},
            },
        },
    }
    result = external.select_dynamic_primary(selections)
    assert result['selected_arm'] == 'dynamic_fixed'
    assert result['split'] == 'validation'
    assert result['mean_validation_bce']['dynamic_fixed'] == pytest.approx(.15)
    assert result['mean_validation_bce']['dynamic_selected'] == pytest.approx(.16)

    tied = {
        seed: {**record, 'arms': {
            'dynamic_fixed': {'validation_bce': .1},
            'dynamic_selected': {'validation_bce': .1},
        }}
        for seed, record in selections.items()
    }
    assert external.select_dynamic_primary(tied)['selected_arm'] == 'dynamic_fixed'
    with pytest.raises(ValueError, match='validation-only'):
        external.select_dynamic_primary({42: {**selections[42], 'split': 'test'}})


def test_studies_must_share_frozen_representation_and_calibration_family():
    identity = {
        'source': '/frozen', 'source_protocol_sha256': 'a' * 64,
        'adapter_path': '/adapter.pt', 'adapter_sha256': 'b' * 64,
        'model_path': '/model', 'model_artifacts': {'index_sha256': 'c' * 64},
        'model_config_sha256': 'd' * 64, 'tokenizer_sha256': 'e' * 64,
        'layers': list(range(1, 33)), 'hidden_size': 8,
        'representation': 'raw block output at final valid token',
    }
    assert external.assert_same_frozen_representation(
        {'representation': identity}, {'representation': dict(identity)}
    ) == identity
    changed = dict(identity, adapter_sha256='f' * 64)
    with pytest.raises(ValueError, match='different frozen representations'):
        external.assert_same_frozen_representation(
            {'representation': identity}, {'representation': changed}
        )

    models = {
        arm: {'calibrator': {
            'method': 'temperature',
            'fit_source': external.OOF_CALIBRATOR_FIT_SOURCE,
        }}
        for arm in external.oof.ALL_ARMS
    }
    selection = {'calibration_selection': {'selected': 'temperature'}}
    assert external.selected_calibration_family(selection, models) == 'temperature'
    models['repeat32']['calibrator']['method'] = 'platt'
    with pytest.raises(ValueError, match='differ from selected family'):
        external.selected_calibration_family(selection, models)
    models['repeat32']['calibrator']['method'] = 'temperature'
    models['repeat32']['calibrator']['fit_source'] = 'rowwise nonnested scores'
    with pytest.raises(ValueError, match='strict nested outer-heldout'):
        external.selected_calibration_family(selection, models)


def test_current_artifact_schema_runs_lock_extract_evaluate_and_resume_without_gpu(
    tmp_path, monkeypatch,
):
    registry = _build_registry(tmp_path)
    frozen_source, dynamic_study, oof_study = _build_current_study_artifacts(tmp_path)
    del frozen_source
    dynamic_protocol = external._json(dynamic_study / 'protocol.json')
    locked_dynamic_code = dynamic_protocol['code_sha256']['dynamic_gate_diagnostics.py']
    amendment_path = dynamic_study / external.dynamic.EVALUATION_AMENDMENT_FILE
    audit_path = dynamic_study / external.dynamic.PRIMARY_REPRODUCIBILITY_FILE
    provenance_path = dynamic_study / 'provenance/primary_invariance.patch'
    provenance_path.parent.mkdir()
    amendment_path.write_text('{"schema":"fixture-amendment"}\n', encoding='utf-8')
    audit_path.write_text('{"schema":"fixture-reproducibility"}\n', encoding='utf-8')
    provenance_path.write_text('fixture exact primary invariance evidence\n', encoding='utf-8')
    verified_amendment = {
        'required': True,
        'locked_evaluation_code_sha256': locked_dynamic_code,
        'amendment_sha256': sha256(amendment_path),
        'artifacts': {
            'provenance/primary_invariance': {
                'path': str(provenance_path),
                'bytes': provenance_path.stat().st_size,
                'sha256': sha256(provenance_path),
            },
        },
        'primary_reproducibility': {
            'state': 'exact_primary_reproduction_verified',
            'all_exact': True,
            'audit_sha256': sha256(audit_path),
        },
    }

    monkeypatch.setattr(
        external.dynamic, 'verify_evaluation_amendment',
        lambda output, protocol, require_reproducibility: (
            verified_amendment
            if require_reproducibility
            else pytest.fail('external discovery must require the reproducibility audit')
        ),
    )
    output = tmp_path / 'evaluation'
    args = SimpleNamespace(
        registry=registry,
        dynamic_study=dynamic_study,
        oof_study=oof_study,
        dynamic_primary_arm='auto',
        output=output,
        device='cpu',
        token_budget=64,
        max_micro=4,
        pad_multiple=8,
        prediction_batch_size=8,
        repetitions=32,
        analysis_seed=101,
        dataset=None,
        status_every=1,
    )
    lock = external.lock_evaluation(args)
    assert lock['evaluation']['dynamic']['primary_arm'] == 'dynamic_fixed'
    provenance = lock['evaluation']['dynamic']['evaluation_provenance']
    assert provenance['amendment_required'] is True
    assert provenance['exact_primary_reproduction'] is True
    assert provenance['primary_reproducibility_sha256'] == sha256(audit_path)
    assert 'dynamic/evaluation_amendment' in lock['artifacts']
    assert 'dynamic/evaluation_primary_reproducibility' in lock['artifacts']
    assert 'dynamic/amendment_artifact/provenance/primary_invariance' in lock['artifacts']
    assert lock['evaluation']['dynamic']['primary_selection']['split'] == 'validation'
    assert set(lock['evaluation']['dynamic']['primary_selection']['evidence']['seeds']) == {
        '42', '43', '44', '45', '46',
    }
    assert lock['evaluation']['oof']['selected_arm'] == 'fixed4'
    assert lock['evaluation']['oof']['calibration'] == {
        'family': 'temperature',
        'family_selection_source': external.OOF_CALIBRATION_SELECTION_SOURCE,
        'fit_source': external.OOF_CALIBRATOR_FIT_SOURCE,
        'external_refit': False,
    }
    assert (
        lock['evaluation']['dynamic']['representation']
        == lock['evaluation']['oof']['representation']
        == lock['evaluation']['shared_frozen_representation']
    )

    snapshot = external.validate_registry(registry)
    _write_completed_external_cache(output, snapshot)
    assert external.extract(args)['state'] == 'already_complete'

    repeat = np.array([.2, .7, .4, .8])
    fixed = np.array([.1, .9, .2, .9])
    selected = np.array([.15, .85, .25, .85])
    monkeypatch.setattr(
        external, '_dynamic_predictions',
        lambda study, cache, batch_size, device: ({
            'dynamic_ensemble_dynamic_repeat_last': repeat,
            'dynamic_ensemble_dynamic_fixed': fixed,
            'dynamic_ensemble_dynamic_selected': selected,
        }, {'seeds': [42, 43, 44, 45, 46]}),
    )
    monkeypatch.setattr(
        external, '_oof_predictions',
        lambda study, cache: ({
            'oof_repeat32': repeat,
            'oof_fixed4': fixed,
            'oof_selected4': selected,
            'oof_single32': repeat,
        }, {'selected_arm': 'fixed4', 'learned_layers': [4, 12, 20, 32]}),
    )
    result = external.evaluate(args)
    assert result['schema'] == 'hanguard_external_holdout_results_v1'
    assert result['datasets']['toy']['primary_systems']['dynamic'] == 'dynamic_ensemble_dynamic_fixed'
    assert external.evaluate(args) == result
    assert external.verify_completed_evaluation(output) == result


def test_checkpoint_preflight_fails_before_one_shot_registry_is_consumed(tmp_path):
    registry = _build_registry(tmp_path)
    _, dynamic_study, oof_study = _build_current_study_artifacts(tmp_path)
    seed, arm = 42, 'dynamic_fixed'
    checkpoint_path = dynamic_study / f'heads/seed_{seed}/{arm}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint['best_epoch'] = 99
    external.frozen.atomic_torch_save(checkpoint, checkpoint_path)
    selection_path = dynamic_study / f'heads/seed_{seed}/selection.json'
    selection = external._json(selection_path)
    selection['arms'][arm]['checkpoint_sha256'] = sha256(checkpoint_path)
    dump(selection_path, selection)
    summary_path = dynamic_study / 'heads/selection.json'
    summary = external._json(summary_path)
    summary['entries'] = [selection if row['seed'] == seed else row for row in summary['entries']]
    dump(summary_path, summary)
    (dynamic_study / 'test_barrier.json').unlink()
    protocol = external.dynamic.load_protocol(dynamic_study)
    external.frozen.create_test_barrier(
        dynamic_study,
        external.dynamic._barrier_paths(dynamic_study, protocol),
        external.dynamic._barrier_identity(dynamic_study, protocol),
    )

    output = tmp_path / 'evaluation'
    args = SimpleNamespace(
        registry=registry, dynamic_study=dynamic_study, oof_study=oof_study,
        dynamic_primary_arm='auto', output=output, device='cpu', token_budget=64,
        max_micro=4, pad_multiple=8, prediction_batch_size=8, repetitions=8,
        analysis_seed=7, dataset=None, status_every=1,
    )
    external.lock_evaluation(args)
    with pytest.raises(ValueError, match='checkpoint differs from validation selection'):
        external.extract(args)
    assert not (registry / 'consumed.json').exists()
    assert not (output / 'access_started.json').exists()


def test_stale_frozen_extractor_fails_during_lock_without_consuming_registry(tmp_path):
    registry = _build_registry(tmp_path)
    _, dynamic_study, oof_study = _build_current_study_artifacts(
        tmp_path, stale_frozen_code=True,
    )
    output = tmp_path / 'evaluation'
    args = SimpleNamespace(
        registry=registry, dynamic_study=dynamic_study, oof_study=oof_study,
        dynamic_primary_arm='auto', output=output, device='cpu', token_budget=64,
        max_micro=4, pad_multiple=8, prediction_batch_size=8, repetitions=8,
        analysis_seed=7,
    )

    with pytest.raises(ValueError, match='implementation changed after protocol registration'):
        external.lock_evaluation(args)
    assert not (output / 'evaluation_lock.json').exists()
    assert not (registry / 'consumed.json').exists()


def test_oof_preflight_rejects_stack_schema_drift(tmp_path):
    frozen_source, _, oof_study = _build_current_study_artifacts(tmp_path)
    representation, _ = external._frozen_representation_identity(frozen_source)
    path = oof_study / 'models/selected4.json'
    model = external._json(path)
    model['stack']['scale'] = [1., 1., 1.]
    dump(path, model)

    with pytest.raises(ValueError, match='stack width changed'):
        external._preflight_oof_deployment(oof_study, representation)


def test_completed_evaluation_revalidates_results_report_and_prediction_hashes(tmp_path):
    registry = _build_registry(tmp_path)
    snapshot = external.validate_registry(registry)
    artifact = tmp_path / 'model.pt'
    artifact.write_bytes(b'model')
    output = tmp_path / 'evaluation'
    external.create_evaluation_lock(
        output, snapshot, {'model': artifact}, {'datasets': ['toy'], 'threshold': .5},
    )
    external.mark_consumed(output)
    prediction = output / 'predictions/toy.parquet'
    prediction.parent.mkdir(parents=True)
    pd.DataFrame({'base_id': ['x'], 'probability_model': [.5]}).to_parquet(prediction, index=False)
    report = output / 'report.md'
    report.write_text('frozen report\n', encoding='utf-8')
    preflight = output / 'deployment_preflight.json'
    preflight.write_text('{"state":"fixture"}\n', encoding='utf-8')
    lock_hash = sha256(output / 'evaluation_lock.json')
    result = {
        'schema': 'hanguard_external_holdout_results_v1',
        'threshold': .5,
        'evaluation_lock_sha256': lock_hash,
        'registry_manifest_sha256': snapshot['manifest_sha256'],
        'datasets': {
            'toy': {
                'predictions': 'predictions/toy.parquet',
                'predictions_sha256': sha256(prediction),
            },
        },
    }
    dump(output / 'results.json', result)
    dump(output / 'completion.json', {
        'schema': 'hanguard_external_holdout_completion_v1',
        'state': 'complete_no_retuning_authorized',
        'evaluation_lock_sha256': lock_hash,
        'results_sha256': sha256(output / 'results.json'),
        'report_sha256': sha256(report),
        'deployment_preflight_sha256': sha256(preflight),
        'prediction_sha256': {'toy': sha256(prediction)},
    })
    assert external.verify_completed_evaluation(output) == result

    prediction_bytes = prediction.read_bytes()
    prediction.write_bytes(prediction.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='predictions changed'):
        external.verify_completed_evaluation(output)
    prediction.write_bytes(prediction_bytes)
    preflight.write_text('{"state":"tampered"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='preflight changed'):
        external.verify_completed_evaluation(output)


def test_external_metrics_report_fixed_threshold_balanced_and_group_units():
    labels = np.array([0, 1, 0, 1, 1])
    probabilities = np.array([.1, .9, .6, .8, .4])
    groups = np.array(['a', 'b', 'c', 'd', 'd'])
    result = external.external_metrics(labels, probabilities, groups)

    assert result['row']['threshold'] == external.PRIMARY_THRESHOLD == .5
    assert result['row']['accuracy'] == pytest.approx(3 / 5)
    assert result['row']['balanced_accuracy'] == pytest.approx((.5 + 2 / 3) / 2)
    assert result['groups'] == 4
    assert result['duplicate_rows_beyond_first'] == 1
    assert result['group_collapsed']['rows'] == 4
    assert result['group_equal_rows']['accuracy'] != result['row']['accuracy']
    assert result['row']['auroc'] is not None
    assert result['row']['bce'] > 0


def test_group_paired_inference_is_deterministic_and_detects_improvement():
    group = np.repeat(np.arange(40), 2)
    labels = np.repeat(np.arange(40) % 2, 2)
    candidate = np.where(labels == 1, .9, .1)
    baseline = candidate.copy()
    bad_groups = set(range(0, 40, 4))
    mask = np.array([value in bad_groups for value in group])
    baseline[mask] = 1 - baseline[mask]
    arguments = dict(
        labels=labels, probability_a=candidate, probability_b=baseline,
        groups=group, repetitions=2048, seed=73,
    )
    first = external.paired_group_inference(**arguments)
    second = external.paired_group_inference(**arguments)

    assert first == second
    assert first['groups'] == 40
    assert first['observed']['accuracy'] == pytest.approx(.25)
    assert first['observed']['balanced_accuracy'] == pytest.approx(.25)
    assert first['observed']['bce'] < 0
    assert first['inference']['balanced_accuracy']['ci95'][0] > 0
    assert first['inference']['balanced_accuracy']['permutation_two_sided_p'] < .01
    assert sum(first['four_cells'].values()) == 40
