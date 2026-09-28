"""Export the binary backbone and a validation-selected category head for inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TARGET = r'model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn)\.(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return actual


def export(binary_run, category_study, output):
    binary_run, category_study, output = map(lambda path: Path(path).resolve(),
                                             (binary_run, category_study, output))
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    binary_study = binary_run.parents[1]
    binary_protocol = read(binary_study / 'protocol.json')
    selection = read(binary_run / 'selection.json')
    checked(binary_study / 'protocol.json', selection['protocol_sha256'])
    checked(binary_study / 'arms.json', selection['arms_sha256'])
    spec = read(binary_study / 'arms.json')[selection['arm']]
    if not spec.get('lora') or spec['readout'] != 'mlp' or spec['mode'] != 'fusion':
        raise ValueError('Inference export requires the LoRA + fusion MLP binary model')
    parent_hashes = {name: checked(binary_run / name, expected)
                     for name, expected in selection['checkpoint_hashes'].items()}
    if not {'head.pt', 'adapter.pt'} <= parent_hashes.keys():
        raise ValueError('Both binary head and LoRA checkpoints are required')
    protocol = read(category_study / 'protocol.json')
    if Path(protocol['parent_run']).resolve() != binary_run:
        raise ValueError('Category features were extracted from a different binary model')
    candidates = []
    for seed in protocol['seeds']:
        directory = category_study / 'runs' / f'learned_queries_s{seed}'
        item = read(directory / 'selection.json')
        if item['arm'] != 'learned_queries' or item['seed'] != seed:
            raise ValueError('Category run identity mismatch')
        checked(category_study / 'protocol.json', item['protocol_sha256'])
        checked(directory / 'head.pt', item['checkpoint_sha256'])
        parent = item['binary_parent']
        if (parent['checkpoint_hashes'] != parent_hashes or
                parent['selection_sha256'] != sha(binary_run / 'selection.json') or
                parent['protocol_sha256'] != sha(binary_study / 'protocol.json') or
                parent['binary_threshold'] != selection['threshold']):
            raise ValueError('Category head and binary checkpoint provenance disagree')
        if not math.isfinite(item['validation_ce']):
            raise ValueError('Nonfinite validation selection criterion')
        candidates.append((item['validation_ce'], seed, directory, item))
    if not candidates:
        raise ValueError('No registered category candidates')
    validation_ce, seed, category_run, category_selection = min(candidates, key=lambda item: item[:2])
    base = Path(binary_protocol['model_path']).resolve()
    config_sha = checked(base / 'config.json', binary_protocol['model_config_sha256'])
    tokenizer_sha = checked(base / 'tokenizer.json', binary_protocol['tokenizer_sha256'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.export-', dir=output.parent))
    try:
        for source, name in [(binary_run / 'adapter.pt', 'binary_adapter.pt'),
                             (binary_run / 'head.pt', 'binary_head.pt'),
                             (category_run / 'head.pt', 'category_head.pt')]:
            shutil.copyfile(source, temporary / name)
            checked(temporary / name, sha(source))
        artifact = lambda name: dict(path=name, sha256=sha(temporary / name))
        manifest = dict(schema='hanguard_runtime_1', base_model=os.path.relpath(base, output),
            model_config_sha256=config_sha, tokenizer_sha256=tokenizer_sha,
            max_tokens=min(binary_protocol['max_tokens'], protocol['max_tokens']),
            pad_multiple=protocol['pad_multiple'], token_budget=protocol['token_budget'],
            binary=dict(mode=spec['mode'], readout=spec['readout'],
                width=binary_protocol['head_width'], layers=binary_protocol['layers'],
                threshold=selection['threshold'],
                lora=dict(rank=binary_protocol['lora_rank'], alpha=binary_protocol['lora_alpha'],
                    dropout=binary_protocol['lora_dropout'], target_modules=TARGET),
                adapter=artifact('binary_adapter.pt'), head=artifact('binary_head.pt')),
            category=dict(mode='learned_queries', width=category_selection['head_width'],
                dropout=protocol['dropout'], seed=seed, validation_ce=validation_ce,
                selection_rule='minimum_validation_ce', head=artifact('category_head.pt')),
            provenance=dict(binary_run=os.path.relpath(binary_run, output),
                category_run=os.path.relpath(category_run, output),
                binary_selection_sha256=sha(binary_run / 'selection.json'),
                category_selection_sha256=sha(category_run / 'selection.json'),
                category_protocol_sha256=sha(category_study / 'protocol.json'),
                selection_candidates=[dict(seed=value[1], validation_ce=value[0]) for value in candidates],
                seed_selection_uses_test=False))
        (temporary / 'model.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-run', type=Path, default=ROOT / 'outputs/hanguard_repaired_core_20260928/runs/E04_s42')
    parser.add_argument('--category-study', type=Path, default=ROOT / 'outputs/hanguard_two_source_primary_20260928')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/hanguard')
    args = parser.parse_args()
    result = export(args.binary_run, args.category_study, args.output)
    print(json.dumps(dict(manifest=str(args.output / 'model.json'), category_seed=result['category']['seed'],
                          selection_rule=result['category']['selection_rule']), ensure_ascii=False))


if __name__ == '__main__':
    main()
