"""Prevent type uncertainty from silently altering binary targets or releases."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.hanguard import current_training as training
from scripts.hanguard import remediated_dataset_contract as contract
from scripts.hanguard import remediated_binary_study as repaired_study


def frame(split='train'):
    return pd.DataFrame(dict(
        base_id=[f'{split}-{i}' for i in range(4)],
        group_id=[f'{split}-group-{i}' for i in range(4)],
        split=[split] * 4, source=['wildguard_zh'] * 4,
        prompt=[f'{split} 完整样本 {i}' for i in range(4)],
        prompt_tokens=[2] * 4, category_id=[0, 2, -1, -1],
        type_label_mask=[False, True, False, False],
        prompt_harm_label=['unharmful', 'harmful', 'harmful', 'unharmful']))


def write_qa(directory):
    training.dump(directory / 'release_qa.json', dict(
        release_status='qa_complete_with_documented_limitations',
        manifest_sha256=contract.sha(directory / 'manifest.json'),
        split_sha256={split: contract.sha(directory / f'{split}.parquet') for split in contract.SPLITS}))


@pytest.fixture
def release(tmp_path):
    for split in contract.SPLITS:
        frame(split).to_parquet(tmp_path / f'{split}.parquet', index=False)
    training.dump(tmp_path / 'manifest.json', dict(dataset_contract=contract.CONTRACT,
        dataset_version='three_source_remediated_20261012'))
    write_qa(tmp_path)
    return tmp_path


def test_unknown_type_preserves_both_source_binary_labels_without_type_targets():
    data = frame()
    before = data.copy(deep=True)
    categories, mask, binary = contract.label_arrays(data, require_type_mask=True)
    np.testing.assert_array_equal(categories, [0, 2, -1, -1])
    np.testing.assert_array_equal(mask, [False, True, False, False])
    np.testing.assert_array_equal(binary, [False, True, True, False])
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize('index,category', [(0, 1), (1, 0)])
def test_known_type_cannot_contradict_source_binary(index, category):
    data = frame(); data.loc[index, 'category_id'] = category
    data['type_label_mask'] = data.category_id.gt(0)
    with pytest.raises(ValueError, match='binary labels disagree'):
        contract.label_arrays(data)


@pytest.mark.parametrize('problem', ['missing', 'safe_true', 'unknown_true', 'harmful_false', 'null', 'string'])
def test_missing_or_wrong_type_mask_is_rejected(problem):
    data = frame()
    if problem == 'missing': data = data.drop(columns='type_label_mask')
    else:
        index, value = dict(safe_true=(0, True), unknown_true=(2, True),
            harmful_false=(1, False), null=(3, None), string=(1, 'True'))[problem]
        data['type_label_mask'] = data.type_label_mask.astype(object)
        data.loc[index, 'type_label_mask'] = value
    with pytest.raises(ValueError, match='type_label_mask'):
        contract.label_arrays(data)


def test_legacy_known_categories_may_omit_mask_but_new_contract_may_not():
    data = frame().iloc[:2].drop(columns='type_label_mask')
    _, mask, _ = contract.label_arrays(data)
    np.testing.assert_array_equal(mask, [False, True])
    with pytest.raises(ValueError, match='Explicit type_label_mask'):
        contract.label_arrays(data, require_type_mask=True)


def test_registration_and_binary_loader_keep_unknown_types(release):
    audit = training.audit_data(release, 'binary', 64)
    assert audit['dataset_contract'] == contract.CONTRACT
    assert audit['split_summaries']['train']['unknown_type_rows'] == 2
    assert audit['split_summaries']['train']['harmful_rows'] == 2
    assert audit['split_summaries']['train']['safe_rows'] == 2
    protocol = dict(training_entrypoint=training.ENTRYPOINT, data_dir=str(release),
        rows={split: 4 for split in contract.SPLITS}, data_sha256=audit['data_sha256'],
        max_tokens=64, dataset_contract=contract.CONTRACT,
        dataset_version='three_source_remediated_20261012')
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == dict(add_special_tokens=False, truncation=False)
            return dict(input_ids=[[1, 2] for _ in texts])
    loaded = repaired_study.load_split('train', Tokenizer(), protocol)
    assert loaded['y'].tolist() == [0., 1., 1., 0.]
    assert len(loaded['encoded']) == 4
    protocol['max_tokens'] = 1
    with pytest.raises(ValueError, match='refusing truncation'):
        repaired_study.load_split('train', Tokenizer(), protocol)


def test_worker_routing_is_explicit_and_keeps_old_registration():
    assert training.binary_script({}) == 'scripts/hanguard/repaired_study.py'
    assert training.binary_script({'dataset_contract': contract.CONTRACT}) == 'scripts/hanguard/remediated_binary_study.py'


@pytest.mark.parametrize('problem', ['missing_qa', 'pending_qa', 'manifest_drift', 'data_drift', 'missing_split_hash'])
def test_unaccepted_or_changed_release_cannot_register(release, problem):
    path = release / 'release_qa.json'
    if problem == 'missing_qa': path.unlink()
    elif problem == 'manifest_drift':
        manifest = json.loads((release / 'manifest.json').read_text())
        manifest['dataset_version'] = 'changed'
        training.dump(release / 'manifest.json', manifest)
    elif problem == 'data_drift':
        data = pd.read_parquet(release / 'train.parquet'); data.loc[0, 'prompt'] += ' 改动'
        data.to_parquet(release / 'train.parquet', index=False)
    else:
        qa = json.loads(path.read_text())
        if problem == 'pending_qa': qa['release_status'] = 'candidate_pending_root_qa'
        else: del qa['split_sha256']['test']
        training.dump(path, qa)
    with pytest.raises(ValueError, match='QA'):
        training.audit_data(release, 'binary', 64)


@pytest.mark.parametrize('problem', ['long', 'cross_split_group', 'binary_conflict_text'])
def test_length_and_group_rules_survive_type_decoupling(release, problem):
    data = pd.read_parquet(release / 'train.parquet')
    if problem == 'long': data.loc[0, 'prompt_tokens'] = 65
    elif problem == 'cross_split_group': data.loc[0, 'group_id'] = 'test-group-0'
    else: data.loc[1, 'prompt'] = data.loc[0, 'prompt']
    data.to_parquet(release / 'train.parquet', index=False)
    write_qa(release)
    with pytest.raises(ValueError):
        training.audit_data(release, 'binary', 64)


def test_a_template_family_group_may_have_safe_and_harmful_examples(release):
    data = pd.read_parquet(release / 'train.parquet')
    data['group_id'] = 'train-template-family'
    data.to_parquet(release / 'train.parquet', index=False)
    write_qa(release)
    assert training.audit_data(release, 'binary', 64)['passed']


def test_original_category_stage_does_not_treat_unknown_types_as_safe(release):
    data = pd.read_parquet(release / 'train.parquet')
    data['source'] = 'chinese_curated'
    data.to_parquet(release / 'train.parquet', index=False)
    write_qa(release)
    with pytest.raises(ValueError, match='requires known types'):
        training.audit_data(release, 'category', 64)


@pytest.fixture
def category_lineage(tmp_path):
    parent_dir = tmp_path / 'three_source_remediated_20261012'; parent_dir.mkdir()
    category_dir = tmp_path / 'hanguard_two_source_remediated_20261012'; category_dir.mkdir()
    for split in contract.SPLITS:
        source = frame(split)
        source.loc[:1, 'source'] = 'chinese_curated'
        source.to_parquet(parent_dir / f'{split}.parquet', index=False)
        source.iloc[:2].to_parquet(category_dir / f'{split}.parquet', index=False)
    training.dump(parent_dir / 'manifest.json', dict(dataset_contract=contract.CONTRACT,
        dataset_version=parent_dir.name))
    write_qa(parent_dir)
    training.dump(category_dir / 'manifest.json', dict(dataset_contract=contract.CONTRACT,
        dataset_version=category_dir.name, parent_dataset_version=parent_dir.name,
        parent_manifest_sha256=contract.sha(parent_dir / 'manifest.json')))
    write_qa(category_dir)
    parent = dict(dataset_contract=contract.CONTRACT, dataset_version=parent_dir.name,
        data_dir=str(parent_dir), data_sha256={s: contract.sha(parent_dir / f'{s}.parquet') for s in contract.SPLITS})
    return category_dir, parent_dir, parent


def test_category_subset_verifies_full_same_split_identity_and_binds_parent(category_lineage):
    directory, parent_dir, parent = category_lineage
    assert training.audit_data(directory, 'category', 64)['passed']
    lineage, hashes = contract.category_parent_lineage(directory, parent)
    assert lineage['category_same_split_parent_rows_verified'] == dict(train=2, validation=2, test=2)
    assert set(hashes) == {str(parent_dir / name) for name in
        ['manifest.json', 'release_qa.json', 'train.parquet', 'validation.parquet', 'test.parquet']}


@pytest.mark.parametrize('problem', ['old_parent', 'old_category', 'wrong_version', 'parent_hash', 'text', 'binary', 'inherited_type', 'train_example_in_test'])
def test_mixed_category_and_binary_versions_or_cross_split_examples_rejected(category_lineage, problem):
    directory, parent_dir, parent = category_lineage
    manifest = json.loads((directory / 'manifest.json').read_text())
    if problem == 'old_parent': parent = {}
    elif problem == 'old_category':
        manifest.pop('parent_dataset_version'); training.dump(directory / 'manifest.json', manifest)
    elif problem == 'wrong_version': parent['dataset_version'] = 'old_version'
    elif problem == 'parent_hash':
        manifest['parent_manifest_sha256'] = 'stale'; training.dump(directory / 'manifest.json', manifest)
    else:
        data = pd.read_parquet(directory / 'test.parquet')
        if problem == 'text': data.loc[0, 'prompt'] += ' 编辑'
        elif problem == 'binary':
            data.loc[1, 'prompt_harm_label'] = 'unharmful'
            data.loc[1, 'category_id'] = 0; data.loc[1, 'type_label_mask'] = False
        elif problem == 'inherited_type': data.loc[1, 'category_id'] = 3
        else:
            # The example is absent from category train, so child-only overlap
            # checks pass; the binary parent nevertheless trained on it.
            added = pd.read_parquet(parent_dir / 'train.parquet').iloc[[2]].copy()
            added['split'] = 'test'; added['source'] = 'chinese_curated'
            added['category_id'] = 2; added['type_label_mask'] = True
            added['group_id'] = 'new-test-group'
            data = pd.concat([data, added], ignore_index=True)
        data.to_parquet(directory / 'test.parquet', index=False)
    with pytest.raises(ValueError):
        contract.category_parent_lineage(directory, parent)


def test_legacy_category_and_legacy_parent_keep_the_historical_contract(tmp_path):
    assert contract.category_parent_lineage(tmp_path, {}) == ({}, {})
