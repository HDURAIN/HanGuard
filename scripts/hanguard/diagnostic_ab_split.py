"""Freeze train-internal, group-isolated A/B partitions for layer diagnosis.

Only the accepted release's training rows enter any derived parquet. A is split
into LoRA training/selection data; B is reserved for frozen-representation
readouts, with nested group-held-out folds. Formal validation/test payloads are
never copied, and this deliberately does not create a legacy ``test.parquet``.
Original columns, including ``split='train'``, are preserved verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np
import pandas as pd

try:
    from scripts.hanguard import remediated_dataset_contract as contract
except ModuleNotFoundError:
    import remediated_dataset_contract as contract

POLICY = 'hanguard_train_internal_ab_grouped_v1'
DEFAULT_SEED = 20261012
IDENTITY_COLUMNS = ('base_id', 'group_id', 'canonical_original_identity')
ADDED_COLUMNS = ('diagnostic_partition', 'readout_outer_fold', 'outer_fold')


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', '', ''.join(c for c in text if unicodedata.category(c) != 'Cf'))


def _identity_sets(frame):
    result = {column: set(frame[column].dropna().astype(str)) - {''}
              for column in IDENTITY_COLUMNS if column in frame}
    normal = frame.prompt.map(_normalized)
    result['normalized_prompt'] = set(normal)
    result['punctuation_style_prompt'] = {
        ''.join(c for c in value if not unicodedata.category(c).startswith('P'))
        for value in normal}
    # Empty punctuation keys do not assert sample identity.
    result['punctuation_style_prompt'].discard('')
    return result


def _isolation(named_frames):
    identities = {name: _identity_sets(frame) for name, frame in named_frames.items()}
    result = {}
    names = list(named_frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps = {key: len(identities[left][key] & identities[right][key])
                        for key in identities[left].keys() & identities[right].keys()}
            if any(overlaps.values()):
                raise ValueError(f'Identity leakage between {left} and {right}: {overlaps}')
            result[f'{left}/{right}'] = overlaps
    return result


def _targets(total, fractions):
    exact = fractions * total
    result = np.floor(exact).astype(np.int64)
    for index in sorted(range(len(fractions)), key=lambda i: (-(exact[i] - result[i]), i))[:int(total - result.sum())]:
        result[index] += 1
    return result


def _allocate(frame, fractions, seed, namespace):
    """Assign whole families; metric-free refinement minimizes quota residuals."""
    fractions = np.asarray(fractions, dtype=np.float64)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError('Partition fractions must be positive and sum to one')
    group_ids = sorted(frame.group_id.unique())
    if len(group_ids) < len(fractions):
        raise ValueError(f'{namespace}: too few independent groups')
    strata = sorted(set(zip(frame.source, frame.prompt_harm_label)))
    group_index = {value: index for index, value in enumerate(group_ids)}
    stratum_index = {value: index for index, value in enumerate(strata)}
    vectors = np.zeros((len(group_ids), len(strata)), dtype=np.int64)
    for group, source, label in zip(frame.group_id, frame.source, frame.prompt_harm_label):
        vectors[group_index[group], stratum_index[(source, label)]] += 1
    totals = vectors.sum(axis=0)
    targets = np.stack([_targets(int(total), fractions) for total in totals], axis=1)
    row_targets = _targets(len(frame), fractions)
    counts = np.zeros_like(targets)
    row_counts = np.zeros(len(fractions), dtype=np.int64)
    sizes = vectors.sum(axis=1)
    order = sorted(range(len(group_ids)), key=lambda i: (
        -float(np.max(vectors[i] / np.maximum(totals, 1))), -int(sizes[i]),
        _digest(f'{seed}:{namespace}:order:{group_ids[i]}')))
    chosen = np.full(len(group_ids), -1, dtype=np.int64)
    denominators = np.maximum(targets, 1)
    row_denominators = np.maximum(row_targets, 1)

    def delta(part, vector, size):
        return float(np.sum(((counts[part] + vector - targets[part]) ** 2 -
                             (counts[part] - targets[part]) ** 2) / denominators[part]) +
                     .1 * (((row_counts[part] + size - row_targets[part]) ** 2 -
                            (row_counts[part] - row_targets[part]) ** 2) / row_denominators[part]))

    for i in order:
        vector, size = vectors[i], int(sizes[i])
        part = min(range(len(fractions)), key=lambda j: (
            delta(j, vector, size), _digest(f'{seed}:{namespace}:tie:{group_ids[i]}:{j}')))
        chosen[i] = part
        counts[part] += vector
        row_counts[part] += size
    moves = 0
    for _ in range(2):
        changed = 0
        for i in reversed(order):
            old = int(chosen[i])
            vector, size = vectors[i], int(sizes[i])
            removal = delta(old, -vector, -size)
            value, _, new = min((removal + delta(j, vector, size),
                _digest(f'{seed}:{namespace}:move:{group_ids[i]}:{j}'), j)
                for j in range(len(fractions)) if j != old)
            if value < -1e-12:
                counts[old] -= vector
                counts[new] += vector
                row_counts[old] -= size
                row_counts[new] += size
                chosen[i] = new
                changed += 1
        moves += changed
        if not changed:
            break
    if np.any(row_counts == 0):
        raise ValueError(f'{namespace}: a group-held-out partition is empty')
    assignments = pd.Series(chosen, index=group_ids)
    deviation = counts - targets
    summary = dict(namespace=namespace, fractions=fractions.tolist(),
        rows=row_counts.tolist(), target_rows=row_targets.tolist(),
        groups=[int((chosen == i).sum()) for i in range(len(fractions))],
        max_group_rows=int(sizes.max()), whole_group_refinement_moves=moves,
        max_absolute_stratum_count_deviation=int(np.abs(deviation).max()),
        source_binary_strata=[dict(source=source, binary_label=label,
            targets=targets[:, i].tolist(), actual=counts[:, i].tolist(),
            deviations=deviation[:, i].tolist()) for i, (source, label) in enumerate(strata)])
    return frame.group_id.map(assignments).to_numpy(dtype=np.int64), summary


def _summary(frame):
    return dict(rows=len(frame), groups=int(frame.group_id.nunique()),
        max_group_rows=int(frame.groupby('group_id').size().max()),
        source_binary=[dict(source=source, binary_label=label, rows=int(count))
            for (source, label), count in frame.groupby(['source', 'prompt_harm_label']).size().items()])


def _require_binary(frame, name):
    if set(frame.prompt_harm_label) != {'harmful', 'unharmful'}:
        raise ValueError(f'{name}: both source binary labels are required')


def build_partition(data_dir, output_dir, seed=DEFAULT_SEED):
    """Create one immutable diagnostic partition and return its SHA-bound manifest."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if (output_dir == data_dir or output_dir.is_relative_to(data_dir) or
            data_dir.is_relative_to(output_dir)):
        raise ValueError('Diagnostic output must be separate from the source release')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic partition: {output_dir}')
    release = contract.release_contract(data_dir)
    if release.get('dataset_contract') != contract.CONTRACT:
        raise ValueError('An accepted remediated source release is required')
    source_files = ('manifest.json', 'release_qa.json', 'train.parquet', 'validation.parquet', 'test.parquet')
    source_sha = {name: contract.sha(data_dir / name) for name in source_files}
    train = pd.read_parquet(data_dir / 'train.parquet')
    required = {'base_id', 'group_id', 'prompt', 'split', 'source', 'prompt_harm_label',
                'category_id', 'type_label_mask'}
    if not required <= set(train):
        raise ValueError(f'Missing inherited training columns: {required - set(train)}')
    if any(column in train for column in ADDED_COLUMNS):
        raise ValueError('Source release already contains diagnostic partition fields')
    if (train.empty or not train.split.eq('train').all() or not train.base_id.is_unique or
            train[['base_id', 'group_id', 'source']].isna().any().any() or
            not train.prompt.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError('Training identities, original split and full prompts must be valid')
    contract.label_arrays(train, require_type_mask=True)
    train = train.sort_values('base_id', kind='stable').reset_index(drop=True)
    # Formal holdout labels are not read or used by the allocation objective.
    identity_columns = ['prompt'] + [column for column in IDENTITY_COLUMNS if column in train]
    formal = {split: pd.read_parquet(data_dir / f'{split}.parquet', columns=identity_columns)
              for split in ('validation', 'test')}
    release_isolation = _isolation(dict(train=train, **formal))
    allocation, ab_summary = _allocate(train, (.7, .3), seed, 'A_B')
    a, b = train.loc[allocation == 0].copy(), train.loc[allocation == 1].copy()
    a_roles, a_summary = _allocate(a, (.9, .1), seed, 'A_train_validation')
    a_train, a_validation = a.loc[a_roles == 0].copy(), a.loc[a_roles == 1].copy()
    a_train['diagnostic_partition'] = 'A_train'
    a_validation['diagnostic_partition'] = 'A_validation'
    b['diagnostic_partition'] = 'B'
    outer, outer_summary = _allocate(b, (.2,) * 5, seed, 'B_outer_folds')
    b['readout_outer_fold'] = outer
    b['outer_fold'] = outer
    derived_isolation = _isolation(dict(A_train=a_train, A_validation=a_validation, B=b, **formal))
    for name, frame in dict(A_train=a_train, A_validation=a_validation, B=b).items():
        _require_binary(frame, name)
    fold_frames, fold_summaries = [], []
    for fold in range(5):
        held = b.loc[b.readout_outer_fold == fold]
        remaining = b.loc[b.readout_outer_fold != fold]
        inner, inner_summary = _allocate(remaining, (.9, .1), seed, f'B_fold_{fold}_inner')
        training, dev = remaining.loc[inner == 0], remaining.loc[inner == 1]
        _isolation(dict(inner_train=training, inner_validation=dev, outer_validation=held))
        for name, frame in dict(inner_train=training, inner_validation=dev, outer_validation=held).items():
            _require_binary(frame, f'B_fold_{fold}/{name}')
        roles = pd.concat([frame[['base_id', 'group_id']].assign(role=role)
                           for frame, role in ((training, 'inner_train'), (dev, 'inner_validation'),
                                               (held, 'outer_validation'))], ignore_index=True)
        roles = roles.sort_values('base_id', kind='stable').reset_index(drop=True)
        roles['outer_fold'] = fold
        fold_frames.append(roles)
        fold_summaries.append(dict(fold=fold, inner_allocation=inner_summary,
            roles={role: _summary(frame) for frame, role in
                   ((training, 'inner_train'), (dev, 'inner_validation'), (held, 'outer_validation'))}))
    combined = pd.concat([a_train[train.columns], a_validation[train.columns], b[train.columns]], ignore_index=True)
    combined = combined.sort_values('base_id', kind='stable').reset_index(drop=True)
    pd.testing.assert_frame_equal(train, combined, check_exact=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}-', dir=output_dir.parent) as temp:
        staging = Path(temp) / 'partition'
        (staging / 'lora_data').mkdir(parents=True)
        (staging / 'folds').mkdir()
        artifacts = {
            'lora_data/train.parquet': a_train,
            'lora_data/validation.parquet': a_validation,
            'B.parquet': b,
            **{f'folds/fold_{i}_assignments.parquet': frame for i, frame in enumerate(fold_frames)}}
        for name, frame in artifacts.items():
            frame.to_parquet(staging / name, index=False)
        manifest = dict(policy_id=POLICY, seed=int(seed),
            source_data_dir=str(data_dir), source_release=release, source_sha256=source_sha,
            builder_sha256=contract.sha(Path(__file__)),
            versions=dict(numpy=np.__version__, pandas=pd.__version__),
            source_rows=len(train), original_columns=train.columns.tolist(),
            lora_data_dir=str(output_dir / 'lora_data'),
            a_train_path=str(output_dir / 'lora_data/train.parquet'),
            a_validation_path=str(output_dir / 'lora_data/validation.parquet'),
            b_path=str(output_dir / 'B.parquet'),
            partitions={name: _summary(frame) for name, frame in
                        dict(A_train=a_train, A_validation=a_validation, B=b).items()},
            allocations=dict(A_B=ab_summary, A_train_validation=a_summary, B_outer_folds=outer_summary),
            readout_folds=[dict(**summary, assignments_path=str(output_dir / f'folds/fold_{i}_assignments.parquet'))
                          for i, summary in enumerate(fold_summaries)],
            folds=[dict(fold=i, assignments_path=str(output_dir / f'folds/fold_{i}_assignments.parquet'))
                   for i in range(5)],
            source_release_identity_intersections=release_isolation,
            derived_identity_intersections=derived_isolation,
            artifact_sha256={name: contract.sha(staging / name) for name in artifacts},
            invariants=dict(all_original_training_rows_conserved=True,
                all_original_fields_preserved=True, whole_groups_preserved=True,
                formal_holdout_labels_used=False, formal_holdout_payloads_copied=False,
                lora_can_access_B=False, final_test_enabled=False, classifier_scores_used=False),
            limitations=[
                'A/B is a temporary diagnostic allocation within the existing training split.',
                'A labels may fit and select LoRA; B labels may only fit/select frozen-representation readouts.',
                'B outer validation labels must not choose layers, head settings or thresholds for their own fold.',
                'The A-trained LoRA checkpoint must be frozen before any B analysis.',
                'Groups constrain exact row/stratum quotas; recorded deviations are expected.',
                'Formal release holdouts remain untouched and have no diagnostic training role.',
                'No test.parquet is created; the legacy three-split registration is intentionally unsupported.'])
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
        if source_sha != {name: contract.sha(data_dir / name) for name in source_files}:
            raise ValueError('Source release changed during diagnostic partition construction')
        staging.rename(output_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = build_partition(args.data_dir, args.output_dir, args.seed)
    print(json.dumps(dict(manifest_path=str(args.output_dir.resolve() / 'manifest.json'),
        partitions=manifest['partitions'], allocations=manifest['allocations']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
