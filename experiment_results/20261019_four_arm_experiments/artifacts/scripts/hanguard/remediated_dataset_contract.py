"""Binary labels and optional type supervision for remediated data releases.

Source binary labels remain authoritative. A missing or ambiguous type is -1;
only known harmful types 1..5 provide type supervision. Legacy releases with
known categories 0..5 may omit the explicit type mask.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT = 'hanguard_binary_type_decoupled_v1'
SPLITS = ('train', 'validation', 'test')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def label_arrays(frame, *, require_type_mask=False):
    """Validate without deriving or changing a binary label from its category."""
    required = {'category_id', 'prompt_harm_label'}
    if not required <= set(frame):
        raise ValueError(f'Missing binary/type label fields: {required - set(frame)}')
    values = pd.to_numeric(frame.category_id, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, np.arange(-1, 6)).all():
        raise ValueError('Expected integral category_id in -1..5')
    categories = values.astype(np.int64)
    if not frame.prompt_harm_label.isin(['harmful', 'unharmful']).all():
        raise ValueError('Binary labels must remain known source harmful/unharmful labels')
    harmful = frame.prompt_harm_label.eq('harmful').to_numpy()
    if ((categories > 0) & ~harmful).any() or ((categories == 0) & harmful).any():
        raise ValueError('Known category and source binary labels disagree')
    expected_mask = categories > 0
    if 'type_label_mask' not in frame:
        if require_type_mask or (categories < 0).any():
            raise ValueError('Explicit type_label_mask required for the remediated/unknown-type data')
        mask = expected_mask
    else:
        values = frame.type_label_mask.to_numpy()
        if (frame.type_label_mask.isna().any() or
                not np.isin(values, [False, True, 0, 1]).all() or
                not np.array_equal(values.astype(bool), expected_mask)):
            raise ValueError('type_label_mask must exactly identify known harmful categories 1..5')
        mask = values.astype(bool)
    return categories, mask, harmful


def release_contract(directory):
    """Require completed, hash-bound QA for an explicitly remediated release.

    Legacy data keeps its existing QA rules in current_training. This function
    never creates an activation pointer or writes a release artifact.
    """
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get('dataset_contract')
    if declared is None:
        return {}
    if declared != CONTRACT:
        raise ValueError(f'Unsupported dataset_contract: {declared}')
    version = manifest.get('dataset_version')
    if not isinstance(version, str) or not version.strip():
        raise ValueError('Remediated dataset_version must be explicit')
    qa_path = directory / 'release_qa.json'
    if not qa_path.is_file():
        raise ValueError('Remediated dataset requires completed release QA before training registration')
    qa = json.loads(qa_path.read_text())
    if qa.get('release_status') != 'qa_complete_with_documented_limitations':
        raise ValueError('Remediated dataset release QA is not complete')
    manifest_hashes = [qa[key] for key in ('manifest_sha256', 'dataset_manifest_sha256') if key in qa]
    if not manifest_hashes or any(value != sha(manifest_path) for value in manifest_hashes):
        raise ValueError('Remediated release QA manifest hash differs')
    hashes = qa.get('split_sha256')
    if not isinstance(hashes, dict):
        raise ValueError('Remediated release QA must bind all three split hashes')
    if set(hashes) == {f'{split}.parquet' for split in SPLITS}:
        hashes = {split: hashes[f'{split}.parquet'] for split in SPLITS}
    if set(hashes) != set(SPLITS):
        raise ValueError('Remediated release QA must bind exactly the three splits')
    for split in SPLITS:
        if hashes[split] != sha(directory / f'{split}.parquet'):
            raise ValueError(f'Remediated release QA {split} hash differs')
    return dict(dataset_contract=declared, dataset_version=version)


def category_parent_lineage(directory, parent):
    """Verify a category subset inherits its binary parent's actual splits.

    Legacy category/legacy binary combinations keep their historical contract.
    A remediated binary parent requires an explicit versioned subset, so an old
    category test cannot silently reuse examples from the parent's new train.
    """
    directory = Path(directory)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    declared = manifest.get('parent_dataset_version')
    parent_dir = Path(parent['data_dir']) if parent.get('data_dir') else None
    parent_manifest_path = parent_dir / 'manifest.json' if parent_dir is not None else None
    parent_manifest = (json.loads(parent_manifest_path.read_text())
                       if parent_manifest_path is not None and parent_manifest_path.is_file() else {})
    remediated_parent = (parent.get('dataset_contract') == CONTRACT or
                        parent_manifest.get('dataset_contract') == CONTRACT)
    if declared is None and not remediated_parent:
        return {}, {}
    if (not isinstance(declared, str) or not declared.strip() or not remediated_parent or
            parent_dir is None or parent_dir.name != declared):
        raise ValueError('Category data requires its matching remediated binary parent dataset version; old/new data cannot be mixed')
    release = release_contract(parent_dir)
    if parent.get('dataset_version') != declared or release.get('dataset_version') != declared:
        raise ValueError('Category parent_dataset_version differs from the registered binary parent release')
    for key in ('parent_manifest_sha256', 'parent_dataset_manifest_sha256'):
        if key in manifest and manifest[key] != sha(parent_manifest_path):
            raise ValueError('Category subset parent manifest hash differs')
    expected = parent.get('data_sha256', parent.get('data_hashes', {}))
    if set(expected) == {f'{split}.parquet' for split in SPLITS}:
        expected = {split: expected[f'{split}.parquet'] for split in SPLITS}
    if set(expected) != set(SPLITS):
        raise ValueError('Remediated binary parent must register all three split hashes')
    bindings = {str(parent_manifest_path): sha(parent_manifest_path),
                str(parent_dir / 'release_qa.json'): sha(parent_dir / 'release_qa.json')}
    counts = {}
    for split in SPLITS:
        parent_path = parent_dir / f'{split}.parquet'
        if sha(parent_path) != expected[split]:
            raise ValueError(f'Registered binary parent {split} data changed')
        bindings[str(parent_path)] = expected[split]
        source = pd.read_parquet(parent_path).set_index('base_id')
        subset = pd.read_parquet(directory / f'{split}.parquet').set_index('base_id')
        if not source.index.is_unique or not subset.index.is_unique or not subset.index.isin(source.index).all():
            raise ValueError(f'Category {split} IDs are not a subset of the binary parent same split')
        matched = source.loc[subset.index]
        for column in ('prompt', 'source', 'prompt_harm_label', 'split'):
            if not subset[column].equals(matched[column]):
                raise ValueError(f'Category {split} {column} differs from its binary parent same split')
        categories, _, _ = label_arrays(subset)
        parent_categories, _, _ = label_arrays(matched, require_type_mask=True)
        if (categories < 0).any() or not np.array_equal(categories, parent_categories):
            raise ValueError(f'Category {split} changed inherited known type labels')
        counts[split] = len(subset)
    return dict(parent_dataset_version=declared, parent_data_dir=str(parent_dir),
                category_same_split_parent_rows_verified=counts), bindings
