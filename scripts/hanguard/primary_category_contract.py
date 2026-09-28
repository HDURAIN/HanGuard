"""Strict data contract for intent-based primary labels, separate from old labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK = 'intent_primary_category_classification'
SPLITS = ('train', 'validation', 'test')
ANNOTATED_SOURCES = ('wildguard_zh',)
RETAINED_SOURCES = ('chinese_curated', 'jailbench')
PROVENANCE_COLUMNS = ('prompt_harm_label', 'type_label_mask', 'category_annotation_status',
                      'annotation_status', 'label_origin', 'original_category_id',
                      'original_prompt_harm_label', 'annotation_protocol_sha256')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_intent(protocol):
    return protocol.get('task') == TASK


def _split_hashes(value):
    if not isinstance(value, dict):
        raise ValueError('Three registered dataset hashes are required')
    if set(value) == set(SPLITS):
        return value
    if set(value) == {f'{s}.parquet' for s in SPLITS}:
        return {s: value[f'{s}.parquet'] for s in SPLITS}
    raise ValueError('Dataset hash registration must cover exactly three splits')


def validate_protocol(protocol):
    """Validate explicit source scope and immutable annotation provenance."""
    if not is_intent(protocol):
        raise ValueError('An explicit intent-primary task is required')
    if protocol.get('new_multilabel_annotations_used') is not False:
        raise ValueError('Multilabel annotations are excluded from the primary task')
    if protocol.get('annotation_source_scope') != list(ANNOTATED_SOURCES):
        raise ValueError('Only wildguard_zh may be reannotated')
    if set(protocol.get('retained_source_scope', [])) != set(RETAINED_SOURCES):
        raise ValueError('Chinese curated and JailBench labels must be retained')
    if protocol.get('label_quality') != 'machine_reannotated_not_human_gold':
        raise ValueError('Machine label quality limitation must be registered')
    if Path(protocol['data_dir']).resolve() == Path(protocol['input_data_dir']).resolve():
        raise ValueError('New intent labels require a separate immutable data release')
    data_hashes = _split_hashes(protocol.get('data_sha256'))
    input_hashes = _split_hashes(protocol.get('input_data_sha256'))
    for name in ('annotation_protocol', 'annotation_manifest'):
        if sha(protocol[f'{name}_file']) != protocol.get(f'{name}_sha256'):
            raise ValueError(f'Registered {name} changed')
    manifest = json.loads(Path(protocol['annotation_manifest_file']).read_text())
    if manifest.get('annotation_protocol_sha256') != protocol['annotation_protocol_sha256']:
        raise ValueError('Annotation manifest belongs to another annotation protocol')
    if _split_hashes(manifest.get('split_sha256')) != data_hashes:
        raise ValueError('Annotation manifest must bind the three new label datasets')
    if _split_hashes(manifest.get('source_sha256')) != input_hashes:
        raise ValueError('Annotation manifest must bind the three original datasets')
    return manifest


def _categories(values, allowed, name):
    numeric = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, allowed).all():
        raise ValueError(f'Invalid {name}; expected integral IDs in {list(allowed)}')
    return numeric.astype(np.int64)


def label_arrays(frame):
    """Unknown type and unknown binary judgments have independent masks."""
    required = {'category_id', 'prompt_harm_label', 'type_label_mask'}
    if not required <= set(frame):
        raise ValueError(f'Missing intent label fields: {required - set(frame)}')
    categories = _categories(frame.category_id, range(-1, 6), 'intent category_id')
    if not frame.prompt_harm_label.isin(['harmful', 'unharmful', 'unknown']).all():
        raise ValueError('Invalid intent binary labels')
    mask = frame.type_label_mask.to_numpy()
    if not np.isin(mask, [False, True, 0, 1]).all() or not np.array_equal(mask.astype(bool), categories > 0):
        raise ValueError('type_label_mask must exactly identify known categories 1..5')
    harmful = frame.prompt_harm_label.eq('harmful').to_numpy()
    safe = frame.prompt_harm_label.eq('unharmful').to_numpy()
    if ((categories > 0) & ~harmful).any() or ((categories == 0) & ~safe).any():
        raise ValueError('Known primary categories contradict binary labels')
    return categories, np.flatnonzero(categories > 0), harmful, harmful | safe


def validate_frame(frame, protocol, *, original=None):
    required = {'base_id', 'prompt', 'source', 'category_id'} | set(PROVENANCE_COLUMNS)
    if not required <= set(frame):
        raise ValueError(f'Missing intent annotation provenance: {required - set(frame)}')
    categories, _, _, _ = label_arrays(frame)
    old = _categories(frame.original_category_id, range(6), 'original_category_id')
    if not frame.original_prompt_harm_label.isin(['harmful', 'unharmful']).all():
        raise ValueError('Invalid preserved original binary labels')
    if not np.array_equal(old > 0, frame.original_prompt_harm_label.eq('harmful').to_numpy()):
        raise ValueError('Preserved original category and binary labels contradict')
    if not frame.source.isin(ANNOTATED_SOURCES + RETAINED_SOURCES).all():
        raise ValueError('Unexpected source outside the registered three-source scope')
    annotated = frame.source.eq('wildguard_zh').to_numpy()
    retained = ~annotated
    if not frame.loc[annotated, 'label_origin'].eq('wildguard_intent_reannotation').all():
        raise ValueError('WildGuard rows require new intent annotation provenance')
    if not frame.loc[retained, 'label_origin'].eq('original_source_retained').all():
        raise ValueError('Retained source annotation provenance differs')
    if not frame.loc[annotated, 'annotation_protocol_sha256'].eq(protocol['annotation_protocol_sha256']).all():
        raise ValueError('WildGuard row annotation protocol hash differs')
    for name in ('annotation_status', 'category_annotation_status'):
        if frame[name].isna().any() or not frame[name].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'Explicit {name} is required')
    if not frame.loc[retained, 'category_annotation_status'].eq('retained').all():
        raise ValueError('Retained source decision status must remain retained')
    if not np.array_equal(categories[retained], old[retained]) or not np.array_equal(
            frame.loc[retained, 'prompt_harm_label'], frame.loc[retained, 'original_prompt_harm_label']):
        raise ValueError('A source outside WildGuard had its labels changed')
    if not frame.loc[annotated & (categories >= 0), 'annotation_status'].eq('valid').all():
        raise ValueError('Invalid annotations cannot provide category supervision')
    status = frame.category_annotation_status
    if not status[annotated].isin(['clear', 'unharmful', 'ambiguous', 'out_of_scope', 'unknown']).all():
        raise ValueError('Unexpected intent decision status')
    if not np.array_equal(status[annotated].eq('clear').to_numpy(), categories[annotated] > 0):
        raise ValueError('Only clear primary decisions may have a known harmful category')
    if not np.array_equal(status[annotated].eq('unharmful').to_numpy(), categories[annotated] == 0):
        raise ValueError('Only unharmful decisions may have category zero')
    if original is not None:
        if frame.base_id.tolist() != original.base_id.tolist() or not frame.source.equals(original.source):
            raise ValueError('Intent release changed original row identities or source provenance')
        if not frame.prompt.equals(original.prompt):
            raise ValueError('Intent release changed original full texts')
        if not np.array_equal(old, _categories(original.category_id, range(6), 'input category_id')):
            raise ValueError('Preserved categories differ from the original input release')
        if not frame.original_prompt_harm_label.equals(original.prompt_harm_label):
            raise ValueError('Preserved binary labels differ from the original input release')
    return label_arrays(frame)


def original_frame(protocol, split):
    path = Path(protocol['input_data_dir']) / f'{split}.parquet'
    if sha(path) != _split_hashes(protocol['input_data_sha256'])[split]:
        raise ValueError(f'Original input labels changed: {split}')
    frame = pd.read_parquet(path)
    if frame.base_id.isna().any() or frame.base_id.duplicated().any():
        raise ValueError('Original inputs require unique nonnull identities')
    return frame.sort_values('base_id').reset_index(drop=True)


def coverage(categories, binary_known=None):
    categories = np.asarray(categories)
    known = categories >= 0
    value = dict(total_rows=len(categories), type_rows=int((categories > 0).sum()),
                 safe_rows=int((categories == 0).sum()), unknown_type_rows=int((categories < 0).sum()),
                 end_to_end_rows=int(known.sum()),
                 end_to_end_fraction=float(known.mean()) if len(known) else None)
    if binary_known is not None:
        binary_known = np.asarray(binary_known, dtype=bool)
        value.update(binary_rows=int(binary_known.sum()), unknown_binary_rows=int((~binary_known).sum()),
                     binary_fraction=float(binary_known.mean()) if len(binary_known) else None)
    return value
