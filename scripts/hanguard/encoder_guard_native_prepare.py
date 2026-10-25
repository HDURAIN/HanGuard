"""CPU-only input preparation for a pending native Encoder-Guard comparison.

This does not load a model, infer, train, choose a threshold, or invent missing
native preprocessing. The preparation plan is not a scientific run protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
DEFAULT_REVISION = 'a9c40db54baab11d4c51f9e244f20fd3c3806eef'
SCHEMA = 'encoder_guard_native_input_preparation_v1'
EXPECTED_ROWS = 8350
EXPECTED_CATEGORY_ROWS = 2180
EXPECTED_SOURCES = {'chinese_curated': 2692, 'jailbench': 1080, 'wildguard_zh': 4578}
EXPECTED_TYPE_SOURCES = {'chinese_curated': 1168, 'jailbench': 1012}
EXPECTED_SUPPORT = {1: 680, 2: 413, 3: 276, 4: 506, 5: 305}
NATIVE_SOURCES = ('chinese_curated', 'jailbench')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def build_inputs(frame):
    """Keep raw order/text and derive the existing type mask without inference."""
    required = {'base_id', 'group_id', 'source', 'prompt', 'prompt_harm_label',
                'category_id', 'type_label_mask'}
    if not required.issubset(frame):
        raise ValueError('Missing original data identity, text, or labels')
    frame = frame.reset_index(drop=True)
    for column in ('base_id', 'group_id', 'source', 'prompt'):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError('Nonempty strings required: ' + column)
    if not frame.base_id.is_unique:
        raise ValueError('Duplicate original base_id')
    if not frame.prompt_harm_label.isin(['harmful', 'unharmful']).all():
        raise ValueError('Unknown original binary label')
    category = pd.to_numeric(frame.category_id, errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(category).all() or not np.equal(category, np.floor(category)).all()
            or not np.isin(category, np.arange(-1, 6)).all()):
        raise ValueError('Original categories must be integral -1..5')
    mask = frame.type_label_mask.fillna(False)
    if not mask.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError('Original type mask must be boolean')
    binary = frame.prompt_harm_label.eq('harmful').to_numpy(dtype=np.int64)
    category = category.astype(np.int64)
    typed = (frame.source.isin(NATIVE_SOURCES).to_numpy() & mask.to_numpy(dtype=bool)
             & (binary == 1) & np.isin(category, np.arange(1, 6)))
    records = frame[['base_id', 'prompt']].to_dict('records')
    metadata = frame[['base_id', 'group_id', 'source', 'prompt_harm_label']].copy()
    metadata['binary'] = binary
    metadata['category_id'] = category
    metadata['type_label_mask'] = mask.to_numpy(dtype=bool)
    metadata['type_known'] = typed
    if 'split' in frame:
        if not frame['split'].eq('test').all():
            raise ValueError('Only the original test split is allowed')
        metadata['split'] = frame['split'].to_numpy()
    return records, metadata, np.flatnonzero(typed).astype(np.int64)


def summary(metadata, indices):
    typed = metadata.iloc[indices]
    return dict(rows=len(metadata), category_rows=len(indices),
        binary_counts={str(k): int(v) for k, v in metadata.binary.value_counts().sort_index().items()},
        source_counts={str(k): int(v) for k, v in metadata.source.value_counts().items()},
        category_source_counts={str(k): int(v) for k, v in typed.source.value_counts().items()},
        category_support={str(k): int(v) for k, v in typed.category_id.value_counts().sort_index().items()},
        groups=int(metadata.group_id.nunique()), category_groups=int(typed.group_id.nunique()))


def _assert_real_scope(metadata, indices):
    stats = summary(metadata, indices)
    if (stats['rows'] != EXPECTED_ROWS or stats['category_rows'] != EXPECTED_CATEGORY_ROWS
            or stats['source_counts'] != EXPECTED_SOURCES
            or stats['category_source_counts'] != EXPECTED_TYPE_SOURCES
            or stats['category_support'] != {str(k): v for k, v in EXPECTED_SUPPORT.items()}
            or stats['binary_counts'] != {'0': 3728, '1': 4622}):
        raise ValueError('Original parent test counts differ from the registered comparison scope')
    return stats


def _parent_inputs(parent):
    parent = Path(parent).resolve()
    protocol = read(parent / 'protocol.json')
    results = read(parent / 'results.json')
    raw = parent / 'data/test.parquet'
    if not results.get('complete') or results.get('protocol_sha256') != sha(parent / 'protocol.json'):
        raise ValueError('Parent results are incomplete or not bound to their protocol')
    if sha(raw) != protocol['data_sha256']['test'] or protocol['rows']['test'] != EXPECTED_ROWS:
        raise ValueError('Parent raw test hash or row count changed')
    comparison = [row for row in results['aggregate']
                  if row['arm'] == 'fusion_category' and row['task'] in ('binary', 'category')]
    if {row['task'] for row in comparison} != {'binary', 'category'} or len(comparison) != 2:
        raise ValueError('Missing unique fixed Hanguard combined references')
    return protocol, raw, comparison


def verify(output, parent=DEFAULT_PARENT, model_revision=DEFAULT_REVISION):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    plan = read(output / 'preparation_plan.json')
    if (plan.get('schema') != SCHEMA or plan.get('output') != str(output)
            or plan.get('parent') != str(parent) or plan.get('model', {}).get('revision') != model_revision
            or read(output / 'preparation_registration.json').get('plan_sha256')
            != sha(output / 'preparation_plan.json')):
        raise ValueError('Preparation registration identity changed')
    protocol, raw, _ = _parent_inputs(parent)
    expected = {'parent_protocol_sha256': sha(parent / 'protocol.json'),
                'parent_results_sha256': sha(parent / 'results.json'),
                'raw_test_sha256': protocol['data_sha256']['test']}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepared parent provenance changed')
    if plan.get('preparer_sha256') != sha(Path(__file__)):
        raise ValueError('Preparation source changed')
    for name, digest in plan['files'].items():
        path = output / name
        if not path.resolve().is_relative_to(output) or sha(path) != digest:
            raise ValueError('Prepared artifact changed: ' + name)
    return plan


def prepare(output, parent=DEFAULT_PARENT, model_revision=DEFAULT_REVISION):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if not re.fullmatch('[0-9a-f]{40}', model_revision):
        raise ValueError('Use an explicit immutable forty-character model revision')
    if (output / 'preparation_plan.json').exists():
        return verify(output, parent, model_revision)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing an unregistered nonempty preparation directory')
    protocol, raw, comparison = _parent_inputs(parent)
    frame = pd.read_parquet(raw)
    records, metadata, indices = build_inputs(frame)
    stats = _assert_real_scope(metadata, indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing.', dir=output.parent))
    try:
        (temporary / 'data').mkdir()
        shutil.copyfile(raw, temporary / 'data/test.parquet')
        if sha(temporary / 'data/test.parquet') != protocol['data_sha256']['test']:
            raise ValueError('Copied raw test hash mismatch')
        with (temporary / 'inputs.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        metadata.to_parquet(temporary / 'metadata.parquet', index=False)
        np.save(temporary / 'category_indices.npy', indices, allow_pickle=False)
        state = dict(state='waiting_for_model_access', inference_started=False,
            automatic_run=False, results_absent=True, model_loaded=False, created_unix=time.time())
        write(temporary / 'state.json', state)
        (temporary / 'README.md').write_text('\n'.join([
            '# Encoder-Guard v3 原生能力对照：输入准备', '',
            '状态：waiting_for_model_access。尚未加载模型、提取特征或推理，没有成绩、没有自动 GPU 队列。',
            '已准备同一测试集 8,350 条原文；五分类固定索引为 2,180 条。inputs.jsonl 仅含 base_id 和 prompt，不包含标签。',
            '本文件及 preparation_plan.json 仅记录输入准备，不是已确定原生实现的科学运行协议。',
            '必须取得作者发布的 v3 权重、分类头、标签码表、阈值配置和可运行推理实现后，才能登记并启动正式测试。',
            '仅直接使用作者的原生模型；禁止新训练分类头、LoRA、类别查询或融合模块，禁止用我方验证／测试标签调整其阈值。',
            '原生 preprocessing、F32/FP16 计算方式、512 token 计数与截断侧、安全和放行阈值规则均待核验，不猜测实现。',
            '原生全量测试保留全部分母；另外预登记共同短文与长文分层，不以额外滑窗替代原生截断。', '']))
        filenames = ('data/test.parquet', 'inputs.jsonl', 'metadata.parquet', 'category_indices.npy',
                     'state.json', 'README.md')
        plan = dict(schema=SCHEMA, kind='preparation_plan_not_inference_protocol', output=str(output),
            parent=str(parent), parent_protocol_sha256=sha(parent / 'protocol.json'),
            parent_results_sha256=sha(parent / 'results.json'), raw_test_sha256=protocol['data_sha256']['test'],
            model=dict(repo_id='louivis/encoder-guard', revision=model_revision,
                       advertised_tag='v3-2026.10.25', advertised_tag_verified=False,
                       access='awaiting_authorized_files', native_contract_verified=False),
            input_fields=['base_id', 'prompt'], input_labels_exposed=False, **stats,
            hanguard_combined_reference=comparison,
            native_constraints=dict(training=False, new_heads=False, lora=False, fusion=False,
                threshold_fitting=False, source_code_required=True, labels_order_required=True,
                native_safety_and_release_rules='unverified_pending_author_files',
                tokenizer_and_512_token_policy='unverified_pending_author_files',
                inference_dtype_and_temperature='unverified_pending_author_files',
                category_rule='author-native five-class rule, fixed 2180 gold-known harmful rows; no safety filtering',
                no_results_until_real_inference=True),
            files={name: sha(temporary / name) for name in filenames},
            preparer_sha256=sha(Path(__file__)), created_unix=time.time())
        write(temporary / 'preparation_plan.json', plan)
        write(temporary / 'preparation_registration.json',
              dict(plan_sha256=sha(temporary / 'preparation_plan.json')))
        os.replace(temporary, output)
        return verify(output, parent, model_revision)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--model-revision', default=DEFAULT_REVISION)
    args = parser.parse_args()
    result = globals()[args.action](args.output, args.parent, args.model_revision)
    print(json.dumps({key: result[key] for key in ('output', 'rows', 'category_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
