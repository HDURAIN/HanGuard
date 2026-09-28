"""Export unchanged Chinese-curated/JailBench splits for primary-type training.

This CPU-only preparation does not annotate, repartition, truncate, or train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'validation', 'test')
SOURCES = ('chinese_curated', 'jailbench')
EXPECTED = {
    'train': {'rows': 27618, 'harmful_rows': 15578},
    'validation': {'rows': 3451, 'harmful_rows': 1875},
    'test': {'rows': 3452, 'harmful_rows': 1828},
}
DESCRIPTIONS = {
    'schema_version': '1.0',
    'name': 'hanguard 两来源主要风险类别表示',
    'usage': '仅用于类型分类器的类别描述编码，不是自动标注政策，不改写来源标签。类型预测仅在已经判有害的输入上选择一个主要类别。',
    'scope': '采用中文整理语料和JailBench已有五个主类别，描述参考国标五组风险框架及JailBench实际主领域范围；不宣称全部继承标签已经逐项重新核验国标。',
    'priority_rule': None,
    'label_order': [1, 2, 3, 4, 5],
    'source_label_policy': '两来源原有害性与原类别完全保留；无WildGuard类型重标，不采用已停用的意图标注草案。',
    'description_development': '依据标准与JailBench公开分类体系，并只阅读训练集类3/4/5中每来源前两条，共12条辅助检查范围；没有用验证或测试文本、预测或分数调试描述。',
    'classes': [
        {
            'id': 1,
            'name': '违反社会主义核心价值观的内容',
            'description': '主要涉及危害国家安全和利益、破坏国家统一与社会稳定，恐怖主义、极端主义、民族仇恨、邪教宣传，暴力或淫秽色情，虚假有害信息及其他法律禁止内容。依据请求的核心内容识别相应风险领域，不只看包装、人设或孤立关键词。',
        },
        {
            'id': 2,
            'name': '歧视性内容',
            'description': '主要涉及因民族、种族、信仰、国别、地域、性别、性取向、年龄、职业、健康等群体身份而产生的偏见、贬损、侮辱、刻板印象或排斥。判断实际针对的群体与身份条件，不能把普通行为条件改写成身份歧视。',
        },
        {
            'id': 3,
            'name': '商业违法违规',
            'description': '主要涉及侵犯知识产权、违反商业道德、泄露商业秘密、垄断和不正当竞争，以及虚假宣传、欺诈营销、内幕交易或其他商业违法违规行为。以商业经营行为、商业信息和竞争秩序为主要领域，不局限于营销欺诈。',
        },
        {
            'id': 4,
            'name': '侵犯他人合法权益',
            'description': '主要涉及危害身心健康，侵犯肖像、名誉、荣誉、财产、隐私、个人信息及其他合法权益，并包含JailBench所列公共安全风险。可以涉及盗窃、非法监控、泄露隐私、冒用形象或破坏公共安全，不将范围缩限为特定个人的直接伤害。',
        },
        {
            'id': 5,
            'name': '无法满足特定服务类型的安全需求',
            'description': '主要涉及医疗、心理、法律、金融、科学、历史等特定领域的服务安全，以及内容不准确或不可靠、无法有效帮助使用者等风险。按两来源已有风险领域语义识别，不要求用户明说造假；本类型判断以输入已判有害为前提，不把所有正常专业问题视为有害。',
        },
    ],
    'references': [
        'https://github.com/STAIR-BUPT/JailBench#分类标准',
        'https://arxiv.org/html/2502.18935v1#S3.SS1',
        'https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=F67D3F376E0A0A0FF5317FB36B32A30A',
    ],
    'limitations': [
        'JailBench五类是主要风险领域，40个官方细类、本地36个有数据细类与国标31项风险不能逐项等同。',
        '继承标签存在来源口径差异；类别描述只是模型条件信息，不证明它们已统一为人工金标。',
        '既有split曾用于探索，不能宣称全新盲测；模板可以跨集，已知种子由group_id隔离。',
    ],
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    return re.sub(r'\s+', '', text)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def counts(series):
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def token_stats(frame):
    tokens = frame.prompt_tokens
    return dict(rows=len(frame), min_tokens=int(tokens.min()) if len(tokens) else None,
                max_tokens=int(tokens.max()) if len(tokens) else None,
                total_tokens=int(tokens.sum()), mean_tokens=float(tokens.mean()) if len(tokens) else None)


def summarize(frame):
    return dict(rows=len(frame), harmful_rows=int(frame.prompt_harm_label.eq('harmful').sum()),
                unharmful_rows=int(frame.prompt_harm_label.eq('unharmful').sum()),
                category_counts=counts(frame.category_id), source_counts=counts(frame.source),
                tokens_all=token_stats(frame),
                tokens_harmful=token_stats(frame[frame.prompt_harm_label.eq('harmful')]),
                by_source={source: dict(rows=len(part),
                    harmfulness_counts=counts(part.prompt_harm_label),
                    category_counts=counts(part.category_id), tokens=token_stats(part))
                    for source, part in frame.groupby('source', sort=True)})


def prepare(source, output, enforce_expected=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty new output directory; published data is immutable')
    original_frames, selected, input_hashes = {}, {}, {}
    for split in SPLITS:
        path = source / f'{split}.parquet'
        frame = pd.read_parquet(path)
        required = {'prompt', 'category_id', 'prompt_harm_label', 'source', 'base_id',
                    'group_id', 'split', 'prompt_tokens'}
        if not required <= set(frame):
            raise ValueError(f'Missing fields in {split}: {required - set(frame)}')
        if not frame.split.eq(split).all():
            raise ValueError(f'Input split identity mismatch: {split}')
        part = frame.loc[frame.source.isin(SOURCES)].copy().reset_index(drop=True)
        if not len(part) or not part.prompt.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Selected prompts must be nonempty strings')
        if not part.base_id.is_unique or part[['base_id', 'group_id']].isna().any().any():
            raise ValueError('Missing group/base identity or duplicate base ID')
        categories = pd.to_numeric(part.category_id, errors='raise')
        if not categories.isin(range(6)).all():
            raise ValueError('Existing source categories must be 0..5')
        if not part.prompt_harm_label.isin(['harmful', 'unharmful']).all():
            raise ValueError('Unknown inherited harmfulness label')
        if not categories.gt(0).eq(part.prompt_harm_label.eq('harmful')).all():
            raise ValueError('Inherited type and binary labels disagree; do not overwrite them')
        tokens = pd.to_numeric(part.prompt_tokens, errors='raise')
        if not tokens.ge(1).all() or not tokens.eq(tokens.astype('int64')).all():
            raise ValueError('Stored token counts must be positive integers')
        if enforce_expected:
            actual = dict(rows=len(part), harmful_rows=int(categories.gt(0).sum()))
            if actual != EXPECTED[split]:
                raise ValueError(f'Unexpected full-source selection for {split}: {actual}')
        original_frames[split], selected[split] = frame, part
        input_hashes[split] = file_sha(path)

    identities = {
        split: {'base_id': set(frame.base_id), 'group_id': set(frame.group_id),
                'full_text_sha256': set(frame.prompt.map(text_sha)),
                'normalized_text': set(frame.prompt.map(normalize))}
        for split, frame in selected.items()
    }
    intersections = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = {key: len(identities[left][key] & identities[right][key])
                       for key in identities[left]}
            intersections[f'{left}__{right}'] = overlap
            if any(overlap.values()):
                raise ValueError(f'Cross-split overlap detected: {left}/{right}: {overlap}')

    output.mkdir(parents=True, exist_ok=True)
    output_hashes, preservation = {}, {}
    for split, frame in selected.items():
        path = output / f'{split}.parquet'
        frame.to_parquet(path, index=False)
        restored = pd.read_parquet(path)
        pd.testing.assert_frame_equal(frame, restored, check_exact=True, check_dtype=True)
        original_selected = original_frames[split].loc[
            original_frames[split].source.isin(SOURCES)].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_selected, restored, check_exact=True, check_dtype=True)
        output_hashes[split] = file_sha(path)
        preservation[split] = dict(all_original_columns_equal=True,
            original_column_order_equal=True, original_row_order_equal=True,
            full_text_unchanged=True, category_id_unchanged=True,
            binary_label_unchanged=True, split_unchanged=True, rows_checked=len(restored))

    description_path = output / 'descriptions.json'
    dump(description_path, DESCRIPTIONS)
    audit = dict(passed=True, selected_sources=list(SOURCES),
        split_summaries={split: summarize(frame) for split, frame in selected.items()},
        totals=summarize(pd.concat(selected.values(), ignore_index=True)),
        cross_split_intersection_counts=intersections,
        normalization='NFKC then casefold, remove Unicode Cf characters and all whitespace; audit only',
        preservation=preservation,
        token_statistics_basis='Inherited prompt_tokens from translation-repaired input; not independently retokenized here. Model chat-template overhead is not included.',
        label_quality='Original source labels retained; no new annotation and not newly human-verified gold',
        scope_limits=['Exact and normalized-text overlaps plus known group IDs checked; not a semantic-duplicate audit.',
                      'Group isolation does not imply unseen jailbreak templates.',
                      'Existing source labels and explored split identities retain their known limitations.'])
    dump(output / 'audit.json', audit)
    manifest = dict(dataset_name='hanguard_two_source_primary',
        purpose='Full two-source inherited primary-category training; retain safe rows for binary/end-to-end evaluation',
        source_directory=str(source), output_directory=str(output),
        selected_sources=list(SOURCES), excluded_sources=['wildguard_zh'],
        selection_rule='source in {chinese_curated,jailbench}; no other row filter',
        input_sha256=input_hashes, output_sha256=output_hashes,
        input_rows={split: len(frame) for split, frame in original_frames.items()},
        split_counts={split: dict(rows=len(frame),
            harmful_rows=int(frame.prompt_harm_label.eq('harmful').sum()),
            unharmful_rows=int(frame.prompt_harm_label.eq('unharmful').sum()))
            for split, frame in selected.items()},
        audit_sha256=file_sha(output / 'audit.json'),
        descriptions_file=str(description_path), descriptions_sha256=file_sha(description_path),
        prepare_script=str(Path(__file__).resolve()), prepare_script_sha256=file_sha(__file__),
        source_manifest_sha256=file_sha(source / 'manifest.json') if (source / 'manifest.json').exists() else None,
        original_split_preserved=True, no_resplitting=True, no_augmentation=True,
        no_new_annotations=True, no_label_changes=True, no_text_changes=True,
        no_length_filter=True, no_truncation=True,
        label_origin='original_source_retained',
        type_training='Only existing harmful rows with category_id 1..5; five-way cross entropy; softmax argmax',
        safe_rows='Preserved with category0; excluded only from five-way type loss',
        descriptions_frozen_before_training=True, descriptions_used_test_text_or_predictions=False,
        training_started=False, completed=True)
    dump(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/three_source_translation_repaired')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/hanguard_two_source_primary_20260928')
    parser.add_argument('--allow-different-counts', action='store_true',
                        help='For reuse with a different registered source snapshot; all safety/identity audits still apply')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, not args.allow_different_counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
