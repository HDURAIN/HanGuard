"""Build an immutable same-split dataset from text-only fragment reviews.

Only explicitly approved remove_fragment decisions exclude rows. No text,
label, type supervision, source identity or split assignment is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import remediated_dataset_contract as contract

SPLITS = ('train', 'validation', 'test')
DECISIONS = {'keep', 'remove_fragment', 'uncertain'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validated_reviews(packet, review_paths):
    candidates = jsonl(packet)
    reviews = [row for path in review_paths for row in jsonl(path)]
    candidate_ids = [row['base_id'] for row in candidates]
    reviewed_ids = [row['base_id'] for row in reviews]
    if any(any(word in key.lower() for word in ('label','prediction','probability')) for row in candidates for key in row):
        raise ValueError('Review packets must not expose labels or classifier outputs')
    if (len(set(candidate_ids)) != len(candidate_ids) or
            len(set(reviewed_ids)) != len(reviewed_ids) or
            len({row['review_id'] for row in candidates}) != len(candidates) or
            len({row['review_id'] for row in reviews}) != len(reviews) or
            set(candidate_ids) != set(reviewed_ids)):
        raise ValueError('Exactly one review is required for every candidate, without extras or duplicates')
    lookup = {row['base_id']:row for row in candidates}
    for row in reviews:
        candidate = lookup[row['base_id']]
        if row['review_id'] != candidate['review_id']:
            raise ValueError('Review ID does not match its candidate identity')
        if row['decision'] not in DECISIONS:
            raise ValueError('Unsupported fragment-review decision')
        if not isinstance(row.get('reason_zh'), str) or not row['reason_zh'].strip():
            raise ValueError('Every reviewed row requires a recorded reason')
    return candidates, reviews


def build_dataset(parent, output, packet, review_paths, provenance_path=None):
    parent, output, packet = Path(parent).resolve(), Path(output).resolve(), Path(packet).resolve()
    review_paths = [Path(path).resolve() for path in review_paths]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh derived dataset directory')
    parent_release = contract.release_contract(parent)
    if not parent_release:
        raise ValueError('A completed hash-bound parent dataset is required')
    candidates, reviews = validated_reviews(packet, review_paths)
    review_by_id = {row['base_id']:row for row in reviews}
    removed_ids = {row['base_id'] for row in reviews if row['decision']=='remove_fragment'}
    parent_frames = {split:pd.read_parquet(parent/f'{split}.parquet') for split in SPLITS}
    all_ids = [sample_id for frame in parent_frames.values() for sample_id in frame.base_id]
    if len(set(all_ids)) != len(all_ids) or not set(review_by_id) <= set(all_ids):
        raise ValueError('Candidate identity is absent from the parent, or parent IDs overlap')
    for row in candidates:
        frame = parent_frames[row['split']]
        matched = frame.loc[frame.base_id.eq(row['base_id'])]
        if len(matched)!=1 or matched.iloc[0].prompt != row['prompt']:
            raise ValueError('Candidate prompt or split differs from the pinned parent dataset')
        if row.get('source') != matched.iloc[0].source:
            raise ValueError('Candidate source differs from parent')
    output.mkdir(parents=True, exist_ok=True)
    excluded, summaries, hashes, groups = [], {}, {}, {}
    for split, frame in parent_frames.items():
        contract.label_arrays(frame, require_type_mask=True)
        keep = ~frame.base_id.isin(removed_ids)
        retained = frame.loc[keep].reset_index(drop=True)
        dropped = frame.loc[~keep].copy()
        if not len(retained) or not retained.prompt_harm_label.eq('harmful').any() or not retained.prompt_harm_label.eq('unharmful').any():
            raise ValueError('Each derived split must remain nonempty and contain both binary classes')
        if not retained.split.eq(split).all() or not retained.base_id.is_unique:
            raise ValueError('Derived split membership or identities are invalid')
        contract.label_arrays(retained, require_type_mask=True)
        target = output/f'{split}.parquet'
        retained.to_parquet(target,index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(target), retained)
        pd.testing.assert_frame_equal(retained, frame.loc[keep].reset_index(drop=True))
        dropped['fragment_review_id'] = dropped.base_id.map(lambda key:review_by_id[key]['review_id'])
        dropped['fragment_filter_reason'] = dropped.base_id.map(lambda key:review_by_id[key]['reason_zh'])
        excluded.append(dropped)
        hashes[split] = sha(target)
        groups[split] = set(retained.group_id)
        summaries[split] = {'parent_rows':len(frame),'retained_rows':len(retained),'removed_rows':len(dropped),
            'retained_source_counts':{str(k):int(v) for k,v in retained.source.value_counts().items()},
            'removed_source_counts':{str(k):int(v) for k,v in dropped.source.value_counts().items()},
            'retained_binary_counts':{str(k):int(v) for k,v in retained.prompt_harm_label.value_counts().items()},
            'removed_binary_counts':{str(k):int(v) for k,v in dropped.prompt_harm_label.value_counts().items()}}
    for index,left in enumerate(SPLITS):
        for right in SPLITS[index+1:]:
            if groups[left] & groups[right]:
                raise ValueError('Derived group identities cross splits')
    excluded = pd.concat(excluded,ignore_index=True)
    if set(excluded.base_id) != removed_ids:
        raise ValueError('Excluded rows do not equal the explicitly approved removal set')
    excluded.to_parquet(output/'excluded_fragments.parquet',index=False)
    review_columns=[key for key in ['fragment_review_id','base_id','source','split','prompt_tokens','prompt','original_prompt','fragment_filter_reason'] if key in excluded]
    excluded[review_columns].to_csv(output/'excluded_fragments_review.csv',index=False)
    pd.DataFrame(reviews).sort_values('review_id').to_json(output/'fragment_review_decisions.jsonl',orient='records',lines=True,force_ascii=False)
    if provenance_path is not None:
        provenance_path=Path(provenance_path).resolve()
        dump(output/'review_provenance.json',json.loads(provenance_path.read_text()))
    policy = {
        'policy_id':'hanguard_missing_context_fragment_filter_20261018_v1',
        'parent_dataset_version':parent_release['dataset_version'],
        'screen_scope':'All parent train/validation/test texts; labels and model predictions absent from review packets',
        'remove':'An incomplete request or follow-up depends on missing referents, prior content or unfilled required fields, so the current task cannot be independently recovered',
        'retain':['Complete short requests, ordinary greetings and independently interpretable statements',
                  'Complete harmful requests and risk-topic benign requests alike',
                  'Every uncertain review decision'],
        'decision_source':'AI text-only semantic review, not new human gold annotation',
        'no_label_or_prediction_filtering':True,
        'no_text_changes':True,'no_label_changes':True,'no_resplitting':True,
        'limitations':['Candidate screening is conservative and cannot certify all unscreened text is complete.',
                       'Original dataset/test identities and earlier error analyses have been explored; this is not a new blind benchmark.',
                       'AI semantic decisions can be imperfect; every removed row and reason remains auditable.'],
        'candidate_packet_sha256':sha(packet),
        'review_sha256':{str(path):sha(path) for path in review_paths},
        'review_decision_counts':dict(Counter(row['decision'] for row in reviews)),
        'review_provenance_sha256':sha(output/'review_provenance.json') if provenance_path is not None else None,
    }
    dump(output/'filter_policy.json',policy)
    dump(output/'split_policy.json',{'policy_id':'inherited_same_split_after_fragment_filter_20261018',
        'parent_policy_path':str(parent/'split_policy.json'),'parent_policy_sha256':sha(parent/'split_policy.json'),
        'retained_split_assignments_unchanged':True,'group_split_isolation_verified':True})
    manifest = {'dataset_contract':contract.CONTRACT,'dataset_version':output.name,
        'release_status':'qa_complete_with_documented_limitations','rows':sum(s['retained_rows'] for s in summaries.values()),
        'parent_rows':sum(len(frame) for frame in parent_frames.values()),'removed_fragment_rows':len(excluded),
        'parent_data_dir':str(parent),'parent_dataset_version':parent_release['dataset_version'],
        'parent_manifest_sha256':sha(parent/'manifest.json'),
        'parent_split_sha256':{split:sha(parent/f'{split}.parquet') for split in SPLITS},
        'splits':summaries,'split_sha256':hashes,
        'filter_policy_sha256':sha(output/'filter_policy.json'),
        'split_policy_sha256':sha(output/'split_policy.json'),
        'excluded_fragments_sha256':sha(output/'excluded_fragments.parquet'),
        'excluded_fragments_review_csv_sha256':sha(output/'excluded_fragments_review.csv'),
        'review_decisions_sha256':sha(output/'fragment_review_decisions.jsonl'),
        'builder_sha256':sha(Path(__file__)),
        'original_binary_labels_changed':0,'retained_texts_changed':0,'retained_split_assignments_changed':0,
        'limitations':policy['limitations']}
    dump(output/'manifest.json',manifest)
    qa = {'release_status':'qa_complete_with_documented_limitations','manifest_sha256':sha(output/'manifest.json'),
        'split_sha256':hashes,'passed':True,
        'checks':{'complete_candidate_review_coverage':True,'removed_only_explicit_remove_fragment':True,
                  'uncertain_decisions_retained':True,'retained_frame_values_identical':True,
                  'original_binary_and_type_labels_unchanged':True,'original_split_assignments_unchanged':True,
                  'cross_split_group_overlap':0,'no_truncation_or_text_changes':True},
        'split_summaries':summaries,'reviewed_candidates':len(reviews),
        'removed_fragment_rows':len(excluded),'semantic_review_is_human_gold':False}
    dump(output/'release_qa.json',qa)
    contract.release_contract(output)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,nargs='+',required=True)
    parser.add_argument('--review-provenance',type=Path)
    args=parser.parse_args()
    print(json.dumps(build_dataset(args.parent,args.output,args.packet,args.reviews,args.review_provenance),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
