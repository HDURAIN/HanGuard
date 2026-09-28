"""Intent relabeling must not silently change scope, masks, or cached truths."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import primary_category_contract as contract
from scripts.hanguard import primary_category_report as report
from scripts.hanguard import primary_category_study as study


@pytest.fixture
def release(tmp_path):
    inputs = tmp_path / 'original'
    data = tmp_path / 'intent'
    inputs.mkdir(); data.mkdir()
    old = pd.DataFrame(dict(base_id=list('abcdef'), prompt=[f'完整中文文本{i}' for i in range(6)],
        source=['wildguard_zh', 'chinese_curated', 'wildguard_zh', 'wildguard_zh', 'wildguard_zh', 'jailbench'],
        category_id=[1, 2, 3, 4, 0, 0],
        prompt_harm_label=['harmful', 'harmful', 'harmful', 'harmful', 'unharmful', 'unharmful']))
    policy = tmp_path / 'annotation_protocol.json'
    policy.write_text('{"rule":"core intent, no priority"}')
    frame = old.copy()
    frame['original_category_id'] = old.category_id
    frame['original_prompt_harm_label'] = old.prompt_harm_label
    frame['category_id'] = [0, 2, -1, -1, 1, 0]
    frame['prompt_harm_label'] = ['unharmful', 'harmful', 'harmful', 'unknown', 'harmful', 'unharmful']
    frame['type_label_mask'] = frame.category_id.gt(0)
    frame['category_annotation_status'] = ['unharmful', 'retained', 'ambiguous', 'unknown', 'clear', 'retained']
    frame['annotation_status'] = ['valid', 'retained', 'valid', 'invalid', 'valid', 'retained']
    annotated = frame.source.eq('wildguard_zh')
    frame['label_origin'] = np.where(annotated, 'wildguard_intent_reannotation', 'original_source_retained')
    frame['annotation_protocol_sha256'] = np.where(annotated, contract.sha(policy), '')
    for split in contract.SPLITS:
        old.to_parquet(inputs / f'{split}.parquet', index=False)
        frame.to_parquet(data / f'{split}.parquet', index=False)
    protocol = dict(task=contract.TASK, data_dir=str(data), input_data_dir=str(inputs),
        data_sha256={s: contract.sha(data / f'{s}.parquet') for s in contract.SPLITS},
        input_data_sha256={s: contract.sha(inputs / f'{s}.parquet') for s in contract.SPLITS},
        annotation_protocol_file=str(policy), annotation_protocol_sha256=contract.sha(policy),
        annotation_source_scope=['wildguard_zh'], retained_source_scope=['chinese_curated', 'jailbench'],
        label_quality='machine_reannotated_not_human_gold', new_multilabel_annotations_used=False,
        all_arms=list(study.ARMS), seeds=[42])
    manifest = tmp_path / 'dataset_manifest.json'
    manifest.write_text(json.dumps(dict(split_sha256=protocol['data_sha256'],
        source_sha256=protocol['input_data_sha256'], annotation_protocol_sha256=contract.sha(policy))))
    protocol.update(annotation_manifest_file=str(manifest), annotation_manifest_sha256=contract.sha(manifest))
    return tmp_path, protocol, old, frame


def test_new_release_binds_machine_annotation_protocol_scope_and_all_splits(release):
    _, protocol, old, frame = release
    contract.validate_protocol(protocol)
    categories, indices, binary, known = contract.validate_frame(frame, protocol, original=old)
    assert categories.tolist() == [0, 2, -1, -1, 1, 0]
    assert indices.tolist() == [1, 4]
    assert binary.tolist() == [False, True, True, False, True, False]
    assert known.tolist() == [True, True, True, False, True, True]
    for field, value in [('annotation_source_scope', ['jailbench']),
                         ('label_quality', 'human_gold'),
                         ('annotation_manifest_sha256', 'wrong')]:
        changed = dict(protocol, **{field: value})
        with pytest.raises(ValueError):
            contract.validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed['data_sha256']['test'] = 'different'
    with pytest.raises(ValueError, match='three new label datasets'):
        contract.validate_protocol(changed)


@pytest.mark.parametrize('row,column,value,pattern', [
    (1, 'category_id', 4, 'outside WildGuard'),
    (2, 'type_label_mask', True, 'type_label_mask'),
    (4, 'annotation_status', 'invalid', 'Invalid annotations'),
    (4, 'category_annotation_status', 'ambiguous', 'clear primary'),
    (4, 'annotation_protocol_sha256', 'wrong', 'hash differs'),
    (3, 'original_category_id', 3, 'original input release'),
    (0, 'prompt', 'changed text', 'full texts'),
])
def test_bad_releases_fail_before_type_training(release, row, column, value, pattern):
    _, protocol, old, frame = release
    frame.loc[row, column] = value
    with pytest.raises(ValueError, match=pattern):
        contract.validate_frame(frame, protocol, original=old)


def test_legacy_contract_does_not_accept_new_unknown_labels(release):
    _, protocol, _, frame = release
    with pytest.raises(ValueError):
        study.category_arrays(frame)
    assert study.category_arrays(frame, protocol)[1].tolist() == [1, 4]


def test_cache_binary_truth_remains_original_despite_intent_binary_flips(release):
    root, protocol, old, frame = release
    cache = root / 'feature_cache'; cache.mkdir()
    meta = old[['base_id', 'source']].copy()
    meta['prompt_sha256'] = old.prompt.map(study.shared.text_sha)
    meta['prompt_tokens'] = 1
    meta['binary_probability'] = [.2, .9, .9, .9, .9, .2]
    meta['original_binary_label'] = old.category_id.gt(0).astype(int)
    meta.to_parquet(cache / 'train_index.parquet', index=False)
    np.save(cache / 'train_offsets.npy', np.arange(7))
    raw = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    np.save(cache / 'train_tokens.npy', raw.view(torch.uint16).numpy())
    identity = study.shared.text_identity(old)
    manifest = dict(text_identities={'train': identity}, hidden_size=4,
        splits={'train': dict(rows=6, tokens=6, maximum_tokens=1, text_identity=identity,
            index_sha256=contract.sha(cache / 'train_index.parquet'),
            offsets_sha256=contract.sha(cache / 'train_offsets.npy'),
            tokens_sha256=contract.sha(cache / 'train_tokens.npy'))})
    protocol.update(max_tokens=4096, cache_on_gpu=True)
    loaded = study.load_cached_split(root, protocol, manifest, 'train', device='cpu')
    assert loaded['harmful_indices'].tolist() == [1, 4]
    torch.testing.assert_close(loaded['tokens'], raw)
    assert loaded['meta'].original_binary_label.tolist() == [1, 1, 1, 1, 0, 0]
    assert loaded['meta'].category_id.tolist() == [0, 2, -1, -1, 1, 0]
    meta['original_binary_label'] = frame.category_id.gt(0).astype(int)
    meta.to_parquet(cache / 'train_index.parquet', index=False)
    manifest['splits']['train']['index_sha256'] = contract.sha(cache / 'train_index.parquet')
    with pytest.raises(ValueError, match='Cache binary labels'):
        study.load_cached_split(root, protocol, manifest, 'train', device='cpu')


def test_report_uses_separate_binary_type_and_end_to_end_masks(release):
    root, protocol, _, original = release
    output = root / 'study'; output.mkdir()
    report.dump(output / 'protocol.json', protocol)
    protocol_hash = report.sha(output / 'protocol.json')
    barrier = dict(locked=True, protocol_sha256=protocol_hash, selection_sha256={})
    truth = original.category_id.to_numpy()
    prediction = np.where(truth > 0, truth, 1)
    binary = np.array([.2, .9, .9, .9, .9, .2])
    gated = np.where(binary >= .5, prediction, 0)
    for arm in study.ARMS:
        directory = output / 'runs' / f'{arm}_s42'; directory.mkdir(parents=True)
        (directory / 'head.pt').write_bytes(arm.encode())
        selection = dict(protocol_sha256=protocol_hash, checkpoint_sha256=report.sha(directory / 'head.pt'),
            best_epoch=1, head_parameters=10, parent_binary_threshold=.5,
            train_dataset_sha256=protocol['data_sha256']['train'],
            validation_dataset_sha256=protocol['data_sha256']['validation'])
        report.dump(directory / 'selection.json', selection)
        result = dict(arm=arm, seed=42, **study.metric_pair(truth, prediction, gated),
            test_dataset_sha256=protocol['data_sha256']['test'], checkpoint_sha256=selection['checkpoint_sha256'])
        report.dump(directory / 'test_results.json', result)
        csv = original.drop(columns='prompt').copy()
        csv['prompt_sha256'] = original.prompt.map(report.text_sha)
        csv['binary_probability'] = binary
        csv['predicted_category'] = prediction
        csv['gated_category'] = gated
        for category in range(1, 6):
            csv[f'p_{category}'] = np.where(prediction == category, .8, .05)
        csv.to_csv(directory / 'test.csv', index=False)
        barrier['selection_sha256'][str((directory / 'selection.json').relative_to(output))] = report.sha(directory / 'selection.json')
    report.dump(output / 'test_barrier.json', barrier)
    result = report.report(output, repetitions=10)
    assert result['common_binary_metrics']['rows'] == 5
    assert result['common_binary_metrics']['accuracy'] == 1
    assert result['training_unknown_rows_excluded'] == 2
    assert result['coverage_by_split']['test']['end_to_end_fraction'] == 4/6
    for entry in result['entries']:
        assert entry['type_metrics']['rows'] == 2
        assert entry['gated_metrics']['rows'] == 4
        source = entry['by_source']['wildguard_zh']
        assert source['type_metrics']['rows'] == 1
        assert source['gated_metrics']['rows'] == 2
        assert source['binary_metrics']['rows'] == 3
        assert source['coverage']['end_to_end_fraction'] == .5
    assert '人工金标' in (output / 'report.md').read_text()
    csv_path = output / 'runs/last_mlp_s42/test.csv'
    csv = pd.read_csv(csv_path)
    csv.loc[2, 'prompt_harm_label'] = 'unknown'
    csv.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='provenance changed'):
        report.report(output, repetitions=10)
