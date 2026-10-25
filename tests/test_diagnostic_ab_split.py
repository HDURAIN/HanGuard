"""Train-internal diagnostics must not feed frozen-readout rows into LoRA."""
import json

import pandas as pd
import pytest

from scripts.hanguard import diagnostic_ab_split as diagnostic
from scripts.hanguard import remediated_dataset_contract as contract


def _write_release(directory, train):
    directory.mkdir()
    for split in contract.SPLITS:
        frame = train.copy() if split == 'train' else train.iloc[:2].copy()
        if split != 'train':
            for column in ('base_id', 'group_id', 'canonical_original_identity', 'prompt'):
                frame[column] = split + '_' + frame[column]
        frame['split'] = split
        frame.to_parquet(directory / f'{split}.parquet', index=False)
    manifest = dict(dataset_contract=contract.CONTRACT, dataset_version=directory.name)
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    _update_qa(directory)
    return directory


def _update_qa(directory):
    (directory / 'release_qa.json').write_text(json.dumps(dict(
        release_status='qa_complete_with_documented_limitations',
        manifest_sha256=contract.sha(directory / 'manifest.json'),
        split_sha256={split: contract.sha(directory / f'{split}.parquet') for split in contract.SPLITS})))


@pytest.fixture
def release(tmp_path):
    rows = []
    for i in range(160):
        label = 'harmful' if i % 2 else 'unharmful'
        # Linked variants, including a group spanning source/binary strata.
        for variant in range(2 if i % 7 == 0 else 1):
            row_label = ('unharmful' if variant and i % 14 == 7 else label)
            rows.append(dict(base_id=f'row{i:04d}_{variant}', group_id=f'group{i:04d}',
                canonical_original_identity=f'original{i:04d}_{variant}',
                prompt=f'完整输入样本{i:04d}变体{variant}', split='train',
                source=('chinese_curated' if i % 3 == 0 else 'wildguard_zh'),
                prompt_harm_label=row_label, category_id=2 if row_label == 'harmful' else 0,
                type_label_mask=row_label == 'harmful', prompt_tokens=14,
                original_english=f'original {i} {variant}', untouched_metadata='保留原值'))
    return _write_release(tmp_path / 'release', pd.DataFrame(rows))


def _load_outputs(manifest):
    return {name: pd.read_parquet(manifest[key]) for name, key in (
        ('A_train', 'a_train_path'), ('A_validation', 'a_validation_path'), ('B', 'b_path'))}


def test_group_isolation_conservation_and_original_fields(release, tmp_path):
    before = {path.name: contract.sha(path) for path in release.iterdir()}
    manifest = diagnostic.build_partition(release, tmp_path / 'diagnosis')
    parts = _load_outputs(manifest)
    original = pd.read_parquet(release / 'train.parquet').sort_values('base_id').reset_index(drop=True)
    combined = pd.concat([frame[original.columns] for frame in parts.values()], ignore_index=True)
    combined = combined.sort_values('base_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(original, combined, check_exact=True)
    assert before == {path.name: contract.sha(path) for path in release.iterdir()}
    for left, right in (('A_train', 'A_validation'), ('A_train', 'B'), ('A_validation', 'B')):
        assert not set(parts[left].group_id) & set(parts[right].group_id)
    for name, frame in parts.items():
        assert frame.split.eq('train').all()
        assert frame.diagnostic_partition.eq(name).all()
    assert not (tmp_path / 'diagnosis/lora_data/test.parquet').exists()
    assert manifest['invariants']['final_test_enabled'] is False
    assert manifest['invariants']['formal_holdout_labels_used'] is False
    for counts in manifest['derived_identity_intersections'].values():
        assert not any(counts.values())
    for relative, digest in manifest['artifact_sha256'].items():
        assert contract.sha(tmp_path / 'diagnosis' / relative) == digest


def test_outer_predictions_are_once_per_B_group_with_isolated_inner_roles(release, tmp_path):
    manifest = diagnostic.build_partition(release, tmp_path / 'diagnosis')
    b = pd.read_parquet(manifest['b_path'])
    assert b.groupby('group_id').outer_fold.nunique().max() == 1
    assert b.outer_fold.equals(b.readout_outer_fold)
    held_ids = []
    for fold in manifest['readout_folds']:
        roles = pd.read_parquet(fold['assignments_path'])
        assert set(roles.role) == {'inner_train', 'inner_validation', 'outer_validation'}
        assert set(roles.base_id) == set(b.base_id)
        assert roles.base_id.is_unique
        assert roles.groupby('group_id').role.nunique().max() == 1
        held = roles.loc[roles.role == 'outer_validation']
        assert set(held.base_id) == set(b.loc[b.outer_fold == fold['fold']].base_id)
        held_ids.extend(held.base_id)
    assert len(held_ids) == len(set(held_ids)) == len(b)


def test_deterministic_assignments_and_byte_identical_artifacts(release, tmp_path):
    first = diagnostic.build_partition(release, tmp_path / 'first', seed=123)
    second = diagnostic.build_partition(release, tmp_path / 'second', seed=123)
    assert first['artifact_sha256'] == second['artifact_sha256']
    assert first['allocations'] == second['allocations']
    train = pd.read_parquet(release / 'train.parquet').sample(frac=1, random_state=4)
    train.to_parquet(release / 'train.parquet', index=False)
    _update_qa(release)
    third = diagnostic.build_partition(release, tmp_path / 'third', seed=123)
    assert first['artifact_sha256'] == third['artifact_sha256']


@pytest.mark.parametrize('column', ['base_id', 'group_id', 'canonical_original_identity', 'prompt'])
def test_formal_holdout_identity_leakage_rejected(release, tmp_path, column):
    train = pd.read_parquet(release / 'train.parquet')
    held = pd.read_parquet(release / 'test.parquet')
    held.loc[0, column] = train.loc[0, column]
    held.to_parquet(release / 'test.parquet', index=False)
    _update_qa(release)
    with pytest.raises(ValueError, match='Identity leakage'):
        diagnostic.build_partition(release, tmp_path / 'diagnosis')
    assert not (tmp_path / 'diagnosis').exists()


def test_stale_source_QA_and_existing_output_rejected(release, tmp_path):
    output = tmp_path / 'diagnosis'
    diagnostic.build_partition(release, output)
    with pytest.raises(FileExistsError, match='overwrite'):
        diagnostic.build_partition(release, output)
    frame = pd.read_parquet(release / 'train.parquet')
    frame.loc[0, 'prompt'] += '被改动'
    frame.to_parquet(release / 'train.parquet', index=False)
    with pytest.raises(ValueError, match='hash differs'):
        diagnostic.build_partition(release, tmp_path / 'stale')


def test_output_cannot_be_inside_or_replace_original_release(release):
    with pytest.raises(ValueError, match='separate'):
        diagnostic.build_partition(release, release / 'diagnosis')


def test_metric_free_approximately_stratified_whole_large_groups():
    rows = [dict(group_id=f'parent{i}', source='jailbench', prompt_harm_label='harmful')
            for i in range(43) for _ in range(20)]
    rows.extend(dict(group_id=f'safe{i}', source='wildguard_zh', prompt_harm_label='unharmful')
                for i in range(300))
    frame = pd.DataFrame(rows)
    assignments, summary = diagnostic._allocate(frame, (.7, .3), 20261012, 'test')
    assert pd.DataFrame(dict(group=frame.group_id, part=assignments)).groupby('group').part.nunique().max() == 1
    assert summary['max_group_rows'] == 20
    assert summary['max_absolute_stratum_count_deviation'] <= 20
    assert abs(summary['rows'][0] - .7 * len(frame)) <= 20
