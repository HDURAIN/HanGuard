"""Register and run maintained binary and primary-category experiments.

This module imports no GPU framework and makes no annotation calls. Dry-run,
init and register perform CPU checks only. repaired_study remains the binary
LoRA trainer/loader; multilabel_study is used only for immutable feature
extraction, cache utilities and head construction, never multilabel training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'validation', 'test')
ENTRYPOINT = 'hanguard_current_training'
CATEGORY_ARMS = ['last_mlp', 'learned_queries', 'description_queries']
BINARY_ARMS = {
    'E03': dict(lora=True, mode='last', readout='mlp', objective='bce'),
    'E04': dict(lora=True, mode='fusion', readout='mlp', objective='bce'),
}
COMMON_CODE = ['hanguard_model.py', 'scripts/hanguard/train.py',
               'scripts/hanguard/current_training.py', 'scripts/hanguard/repaired_study.py',
               'scripts/hanguard/repaired_heads.py']
CATEGORY_CODE = ['multilabel_study.py', 'multilabel_heads.py', 'primary_category_study.py',
                 'primary_category_contract.py', 'primary_category_queue.py', 'primary_category_report.py']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def normalized(text):
    value = unicodedata.normalize('NFKC', text).casefold()
    return re.sub(r'\s+', '', ''.join(c for c in value if unicodedata.category(c) != 'Cf'))


def audit_data(directory, stage, max_tokens):
    """Inspect inherited labels and split isolation without transforming rows."""
    directory = Path(directory)
    summaries, hashes, identities = {}, {}, {}
    permitted = {'chinese_curated', 'jailbench'} | ({'wildguard_zh'} if stage == 'binary' else set())
    for split in SPLITS:
        path = directory / f'{split}.parquet'
        frame = pd.read_parquet(path)
        required = {'base_id', 'group_id', 'split', 'source', 'prompt', 'category_id', 'prompt_harm_label', 'prompt_tokens'}
        if not required <= set(frame) or not len(frame):
            raise ValueError(f'{split}: complete nonempty inherited split metadata required')
        if frame[['base_id', 'group_id', 'source']].isna().any().any() or not frame.base_id.is_unique:
            raise ValueError(f'{split}: null/duplicate source identities')
        if not frame.split.eq(split).all() or not frame.source.isin(permitted).all():
            raise ValueError(f'{split}: split or source outside the registered stage scope')
        if not frame.prompt.map(lambda text: isinstance(text, str) and bool(text.strip())).all():
            raise ValueError(f'{split}: full nonempty input text required')
        categories = pd.to_numeric(frame.category_id, errors='coerce')
        if not categories.isin(range(6)).all() or not frame.prompt_harm_label.isin(['harmful', 'unharmful']).all():
            raise ValueError(f'{split}: only inherited categories 0..5 and known binary labels are supported')
        if not categories.gt(0).eq(frame.prompt_harm_label.eq('harmful')).all():
            raise ValueError(f'{split}: inherited type and binary labels disagree')
        if not categories.eq(0).any() or not categories.gt(0).any():
            raise ValueError(f'{split}: both safe and harmful evaluation examples are required')
        lengths = pd.to_numeric(frame.prompt_tokens, errors='coerce')
        if lengths.isna().any() or not lengths.ge(1).all() or not lengths.eq(lengths.astype('int64')).all() or lengths.max() > max_tokens:
            raise ValueError(f'{split}: stored full input lengths exceed the budget; increase max_tokens, never truncate')
        identities[split] = {key: set(frame[key]) for key in ['base_id', 'group_id']}
        identities[split]['normalized_text'] = set(frame.prompt.map(normalized))
        hashes[split] = sha(path)
        summaries[split] = dict(rows=len(frame), type_rows=int(categories.gt(0).sum()),
            safe_rows=int(categories.eq(0).sum()), maximum_stored_tokens=int(lengths.max()),
            category_counts={str(key): int(value) for key, value in categories.value_counts().sort_index().items()},
            source_counts={str(key): int(value) for key, value in frame.source.value_counts().items()})
    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            counts = {key: len(identities[left][key] & identities[right][key]) for key in identities[left]}
            if any(counts.values()):
                raise ValueError(f'Cross-split identity/text overlap: {left}/{right}: {counts}')
            overlaps[f'{left}__{right}'] = counts
    qa = directory / 'release_qa.json'
    if stage == 'binary' and qa.exists():
        value = json.loads(qa.read_text())
        if value.get('release_status') != 'qa_complete_with_documented_limitations' or value.get('dataset_manifest_sha256') != sha(directory / 'manifest.json'):
            raise ValueError('Binary repaired dataset release QA is missing or inconsistent')
    return dict(passed=True, stage=stage, data_sha256=hashes, split_summaries=summaries,
        cross_split_intersection_counts=overlaps, no_reannotation=True, no_text_changes=True,
        no_resplitting=True, no_truncation=True,
        limitation='Inherited labels; no new human verification. Groups/normalized text audited, not semantic duplicates.',
        token_statistics_basis='Stored metadata checked here; workers retokenize complete input and reject overflow.')


def code_files(stage):
    extra = CATEGORY_CODE if stage == 'category' else ['report_repaired_study.py']
    return COMMON_CODE + [f'scripts/hanguard/{name}' for name in extra]


def parent_registration(run):
    run = Path(run).resolve()
    selection_path, protocol_path, arms_path = run / 'selection.json', run.parents[1] / 'protocol.json', run.parents[1] / 'arms.json'
    selection = json.loads(selection_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    arms = json.loads(arms_path.read_text())
    if selection.get('protocol_sha256') != sha(protocol_path) or selection.get('arms_sha256') != sha(arms_path):
        raise ValueError('Binary parent selection does not match its frozen protocol/arms')
    if selection['arm'] not in arms or not selection.get('checkpoint_hashes'):
        raise ValueError('Binary parent requires a registered arm and selected checkpoints')
    if not isinstance(selection.get('threshold'), (int, float)) or not 0 <= selection['threshold'] <= 1:
        raise ValueError('Binary parent requires its validation-selected probability threshold')
    frozen = {str(selection_path): sha(selection_path), str(protocol_path): sha(protocol_path), str(arms_path): sha(arms_path)}
    for name, value in selection['checkpoint_hashes'].items():
        if name not in {'head.pt', 'adapter.pt'} or sha(run / name) != value:
            raise ValueError('Binary parent checkpoint changed')
        frozen[str(run / name)] = value
    model = Path(protocol['model_path']).resolve()
    for name in ['config.json', 'tokenizer.json']:
        digest = sha(model / name)
        expected = protocol.get('model_config_sha256' if name == 'config.json' else 'tokenizer_sha256')
        if expected and expected != digest:
            raise ValueError('Binary parent base model configuration changed')
        frozen[str(model / name)] = digest
    return protocol, frozen


def build_protocol(args):
    """CPU-only planning; never create an output or spawn a worker here."""
    stage = args.stage
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory; existing experiments are immutable (or explicitly --resume)')
    data = (args.data or ROOT / ('data/three_source_translation_repaired' if stage == 'binary' else 'data/hanguard_two_source_primary_20260928')).resolve()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError('Unique nonnegative seeds required')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError('Unique nonnegative GPU IDs required')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('Epoch budget must be positive')
    if min(args.effective_batch, args.max_micro, args.pad_multiple, args.max_tokens, args.head_width) < 1:
        raise ValueError('Batch, padding, length and head widths must be positive')
    if args.patience < 0 or args.min_delta < 0 or args.cache_budget_gb <= 0 or args.warm_head_epochs < 0:
        raise ValueError('Invalid stopping, cache or warmup budget')
    if args.extraction_gpu is not None and args.extraction_gpu < 0:
        raise ValueError('Extraction GPU ID must be nonnegative')
    if any(not math.isfinite(value) for value in [args.min_delta, args.cache_budget_gb, args.adapter_lr] + ([] if args.head_lr is None else [args.head_lr])):
        raise ValueError('Floating-point training settings must be finite')
    audit = audit_data(data, stage, args.max_tokens)
    protocol = dict(training_entrypoint=ENTRYPOINT, training_stage=stage, state='draft', created_at=stamp(),
        data_dir=str(data), input_data_dir=str(data), data_sha256=audit['data_sha256'],
        rows={s: audit['split_summaries'][s]['rows'] for s in SPLITS}, seeds=args.seeds, gpu_ids=args.gpus,
        epochs=args.epochs or (3 if stage == 'binary' else 40), effective_batch=args.effective_batch,
        max_micro=args.max_micro, pad_multiple=args.pad_multiple, max_tokens=args.max_tokens,
        token_budget=args.token_budget if args.token_budget is not None else (24576 if stage == 'binary' else 16384),
        head_lr=args.head_lr if args.head_lr is not None else (1e-4 if stage == 'binary' else 1e-3), weight_decay=.01,
        head_width=args.head_width, dropout=.1, new_multilabel_annotations_used=False,
        metadata_sha256={str(data / name): sha(data / name) for name in
            ['manifest.json', 'audit.json', 'release_qa.json'] if (data / name).is_file()},
        limitations=['Existing split identities have been explored; not a new blind test.',
                    'Inherited source labels are not newly human-verified gold.',
                    'No augmentation, reannotation, sample filtering or text truncation is performed.'])
    if protocol['token_budget'] < 1 or protocol['head_lr'] <= 0 or args.adapter_lr <= 0:
        raise ValueError('Token budget and learning rates must be positive')
    if stage == 'binary':
        model = (args.model or ROOT / 'models/Qwen3.5-4B').resolve()
        for name in ['config.json', 'tokenizer.json']:
            if not (model / name).is_file():
                raise FileNotFoundError(f'Local Qwen3.5 model is required: {model / name}')
        selected = args.binary_arms
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(BINARY_ARMS):
            raise ValueError('Binary arms must be a unique subset of E03/E04')
        protocol.update(model_path=str(model), all_arms=selected, references=[],
            model_config_sha256=sha(model / 'config.json'), tokenizer_sha256=sha(model / 'tokenizer.json'),
            warm_head_epochs=args.warm_head_epochs, warm_head_lr=1e-3, adapter_lr=args.adapter_lr,
            lora_rank=8, lora_alpha=16, lora_dropout=.05, layers=[8, 16, 24, 32],
            checkpoint_selection='minimum_validation_bce', primary_threshold='maximum_validation_accuracy',
            raw_input=True, truncate=False, reuse_old_adapter=False, reuse_old_features=False)
    else:
        run = (args.parent_run or ROOT / 'outputs/hanguard_repaired_core_20260928/runs/E04_s42').resolve()
        parent, parent_hashes = parent_registration(run)
        descriptions = (args.descriptions or data / 'descriptions.json').resolve()
        content = json.loads(descriptions.read_text())
        classes = content.get('classes', [])
        if [str(row.get('id', row.get('category_id'))) for row in classes] != ['1', '2', '3', '4', '5'] or any(not row.get('description', '').strip() for row in classes):
            raise ValueError('Five ordered, nonempty category descriptions are required')
        config = json.loads((Path(parent['model_path']) / 'config.json').read_text())
        hidden = config['text_config']['hidden_size']; width = args.head_width
        query_count = 2 * hidden * width + 3 * width * width + 13 * width + 1
        mlp_width = max(1, round((query_count - 5) / (hidden + 6)))
        protocol.update(task='existing_primary_category_classification', experiment_scope='two_source_full_primary_category',
            all_arms=CATEGORY_ARMS, label_ids=[1, 2, 3, 4, 5], parent_run=str(run), parent_study=str(run.parents[1]),
            parent_artifact_sha256=parent_hashes, description_file=str(descriptions), description_sha256=sha(descriptions),
            feature_cache_dir=str(args.output.resolve() / 'feature_cache'), cache_budget_gb=args.cache_budget_gb, cache_on_gpu=True,
            head_width_by_arm=dict(last_mlp=mlp_width, learned_queries=width, description_queries=width),
            early_stopping_patience=args.patience, early_stopping_min_delta=args.min_delta,
            type_rows={s: audit['split_summaries'][s]['type_rows'] for s in SPLITS},
            type_sources=['chinese_curated', 'jailbench'], checkpoint_selection='minimum_validation_harmful_subset_cross_entropy',
            type_decision='five-way softmax argmax; frozen binary gate supplies safe category0',
            extraction_gpu=args.extraction_gpu if args.extraction_gpu is not None else args.gpus[0])
    return protocol, audit


def init(args):
    protocol, audit = build_protocol(args)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / 'protocol.json', protocol)
    dump(args.output / 'data_audit.json', audit)
    if args.stage == 'binary':
        dump(args.output / 'arms.json', {arm: BINARY_ARMS[arm] for arm in protocol['all_arms']})
    dump(args.output / 'draft_registration.json', dict(protocol_sha256=sha(args.output / 'protocol.json'),
        audit_sha256=sha(args.output / 'data_audit.json'),
        arms_sha256=sha(args.output / 'arms.json') if args.stage == 'binary' else None))
    return protocol


def verify(study, *, registered=True):
    study = Path(study)
    protocol = json.loads((study / 'protocol.json').read_text())
    if protocol.get('training_entrypoint') != ENTRYPOINT:
        raise ValueError('Existing historical experiments cannot be mutated by the new entrypoint')
    evidence = json.loads((study / ('registration.json' if registered else 'draft_registration.json')).read_text())
    if evidence['protocol_sha256'] != sha(study / 'protocol.json') or evidence['audit_sha256'] != sha(study / 'data_audit.json'):
        raise ValueError('Registered protocol or data audit changed')
    if protocol['training_stage'] == 'binary' and evidence['arms_sha256'] != sha(study / 'arms.json'):
        raise ValueError('Binary arms changed')
    for split, digest in protocol['data_sha256'].items():
        if sha(Path(protocol['data_dir']) / f'{split}.parquet') != digest:
            raise ValueError(f'Registered data changed: {split}')
    for name in ['metadata_sha256', 'parent_artifact_sha256']:
        for filename, digest in protocol.get(name, {}).items():
            if sha(filename) != digest:
                raise ValueError(f'Registered artifact changed: {filename}')
    if protocol['training_stage'] == 'category':
        if sha(protocol['description_file']) != protocol['description_sha256']:
            raise ValueError('Registered category descriptions changed')
    else:
        for filename, key in [('config.json', 'model_config_sha256'), ('tokenizer.json', 'tokenizer_sha256')]:
            if sha(Path(protocol['model_path']) / filename) != protocol[key]:
                raise ValueError('Registered base model configuration changed')
    if registered:
        if protocol['state'] != 'registered':
            raise ValueError('A registered protocol is required')
        for filename, digest in evidence['code_sha256'].items():
            if sha(ROOT / filename) != digest or sha(study / 'registered_code' / filename) != digest:
                raise ValueError(f'Registered training code changed: {filename}; use a new study')
    return protocol


def register(study):
    study = Path(study)
    if (study / 'registration.json').exists():
        return verify(study)
    protocol = verify(study, registered=False)
    if protocol['state'] != 'draft' or (study / 'runs').exists():
        raise ValueError('Only an unused draft from this entrypoint may be registered')
    if json.loads((study / 'data_audit.json').read_text()).get('passed') is not True:
        raise ValueError('Dataset audit must pass before registration')
    files = code_files(protocol['training_stage'])
    hashes = {filename: sha(ROOT / filename) for filename in files}
    for filename in files:
        target = study / 'registered_code' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    protocol.update(state='registered', registered_at=stamp())
    dump(study / 'protocol.json', protocol)
    dump(study / 'registration.json', dict(protocol_sha256=sha(study / 'protocol.json'),
        audit_sha256=sha(study / 'data_audit.json'), code_sha256=hashes,
        arms_sha256=sha(study / 'arms.json') if protocol['training_stage'] == 'binary' else None))
    return verify(study)


def run_worker(study, label, arguments, gpu=None):
    verify(study)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    logs = Path(study) / 'logs'; logs.mkdir(exist_ok=True)
    with (logs / f'{label}.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        dump(logs / f'{label}.process.json', dict(pid=process.pid, gpu=gpu, started=time.time()))
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} failed with exit {code}; inspect {logs / (label + ".log")}')
    verify(study)


def preflight(study):
    protocol = verify(study)
    if protocol['training_stage'] == 'category':
        run_worker(study, 'extract', ['scripts/hanguard/multilabel_study.py', 'extract', '--output', str(study)], protocol['extraction_gpu'])
        script = 'scripts/hanguard/primary_category_study.py'
    else:
        script = 'scripts/hanguard/repaired_study.py'
    for arm in protocol['all_arms']:
        seed = protocol['seeds'][0]
        arguments = [script, 'preflight', '--output', str(study), '--arm', arm, '--seed', str(seed)]
        if protocol['training_stage'] == 'binary':
            arguments += ['--include-longest', '--include-warmup']
        run_worker(study, f'preflight_{arm}_s{seed}', arguments, protocol['gpu_ids'][0])
        evidence = json.loads((Path(study) / 'preflight' / f'{arm}_s{seed}.json').read_text())
        if evidence.get('passed') is not True:
            raise ValueError(f'GPU preflight did not pass: {arm}')


def lock_binary_test(study, protocol):
    """Require every registered seed/arm selection before any test worker."""
    verify(study)
    selections = {}
    for seed in protocol['seeds']:
        for arm in protocol['all_arms']:
            path = Path(study) / 'runs' / f'{arm}_s{seed}' / 'selection.json'
            selected = json.loads(path.read_text())
            if selected['protocol_sha256'] != sha(Path(study) / 'protocol.json') or selected['arms_sha256'] != sha(Path(study) / 'arms.json'):
                raise ValueError('Binary selection belongs to a different protocol')
            if selected['arm'] != arm or selected['seed'] != seed:
                raise ValueError('Binary selection arm or seed differs')
            if set(selected.get('checkpoint_hashes', {})) != {'head.pt', 'adapter.pt'}:
                raise ValueError('Both selected binary head and LoRA checkpoints are required')
            for filename, digest in selected['checkpoint_hashes'].items():
                if sha(path.parent / filename) != digest:
                    raise ValueError('Selected binary checkpoint changed')
            selections[str(path.relative_to(study))] = sha(path)
    dump(Path(study) / 'test_barrier.json', dict(locked=True, locked_at=stamp(),
        protocol_sha256=sha(Path(study) / 'protocol.json'), selection_sha256=selections,
        test_dataset_sha256=protocol['data_sha256']['test']))


def run_registered(study):
    study = Path(study)
    protocol = verify(study)
    if protocol['training_stage'] == 'category':
        run_worker(study, 'extract', ['scripts/hanguard/multilabel_study.py', 'extract', '--output', str(study)], protocol['extraction_gpu'])
        run_worker(study, 'primary_queue', ['scripts/hanguard/primary_category_queue.py', '--output', str(study)])
        if json.loads((study / 'queue_status.json').read_text()).get('stage') != 'complete':
            raise ValueError('Category queue did not complete its report')
        return
    preflight(study)
    tasks = [(arm, seed) for seed in protocol['seeds'] for arm in protocol['all_arms']]
    def execute_phase(phase):
        def worker(gpu, assigned):
            for arm, seed in assigned:
                run_worker(study, f'{phase}_{arm}_s{seed}', ['scripts/hanguard/repaired_study.py', phase,
                    '--output', str(study), '--arm', arm, '--seed', str(seed)], gpu)
        gpus = protocol['gpu_ids']
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            jobs = [executor.submit(worker, gpu, tasks[index::len(gpus)]) for index, gpu in enumerate(gpus)]
            for job in jobs:
                job.result()
    execute_phase('train')
    lock_binary_test(study, protocol)
    execute_phase('evaluate')
    run_worker(study, 'report', ['scripts/hanguard/report_repaired_study.py', '--study', str(study), '--data', protocol['data_dir']])
    summary = json.loads((study / 'summary.json').read_text())
    if summary.get('complete') is not True or summary.get('completed_runs') != len(tasks):
        raise ValueError('Binary report is incomplete')


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=['binary', 'category'])
    value.add_argument('action', choices=['init', 'register', 'preflight', 'dry-run', 'run'])
    value.add_argument('--output', required=True, type=Path)
    value.add_argument('--resume', action='store_true', help='Continue only an unchanged study created by this entrypoint')
    value.add_argument('--data', type=Path)
    value.add_argument('--model', type=Path, help='Local base Qwen model for the binary stage')
    value.add_argument('--parent-run', type=Path, help='Selected binary run for category-stage features')
    value.add_argument('--descriptions', type=Path)
    value.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    value.add_argument('--gpus', type=int, nargs='+', default=[0])
    value.add_argument('--extraction-gpu', type=int)
    value.add_argument('--binary-arms', choices=list(BINARY_ARMS), nargs='+', default=list(BINARY_ARMS))
    value.add_argument('--epochs', type=int)
    value.add_argument('--warm-head-epochs', type=int, default=1)
    value.add_argument('--effective-batch', type=int, default=128)
    value.add_argument('--max-micro', type=int, default=64)
    value.add_argument('--token-budget', type=int)
    value.add_argument('--pad-multiple', type=int, default=32)
    value.add_argument('--max-tokens', type=int, default=4096)
    value.add_argument('--head-width', type=int, default=128)
    value.add_argument('--head-lr', type=float)
    value.add_argument('--adapter-lr', type=float, default=2e-5)
    value.add_argument('--patience', type=int, default=8)
    value.add_argument('--min-delta', type=float, default=1e-4)
    value.add_argument('--cache-budget-gb', type=float, default=8.)
    return value


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    args.output = args.output.resolve()
    if args.resume:
        if args.action not in {'run', 'preflight'}:
            raise ValueError('--resume is only supported for run/preflight')
        if any(token.startswith('--') and token.split('=')[0] not in {'--output', '--resume'} for token in arguments):
            raise ValueError('Resume uses immutable registered settings; supply only --output and --resume')
    if args.action == 'register' or (args.resume and (args.output / 'protocol.json').exists()):
        existing = json.loads((args.output / 'protocol.json').read_text())
        if existing.get('training_stage') != args.stage:
            raise ValueError('Requested stage differs from the registered stage')
    if args.action == 'dry-run':
        protocol, audit = build_protocol(args)
        print(json.dumps(dict(dry_run=True, output_created=False, gpu_workers_started=False,
            runs=len(protocol['all_arms']) * len(protocol['seeds']), protocol=protocol, audit=audit), ensure_ascii=False, indent=2))
        return
    if args.action == 'init':
        protocol = init(args)
    elif args.action == 'register':
        protocol = register(args.output)
    else:
        if args.output.exists() and any(args.output.iterdir()):
            if not args.resume:
                raise FileExistsError('Existing study requires explicit --resume; historical outputs are never rewritten')
            protocol = register(args.output)
        else:
            init(args)
            protocol = register(args.output)
        if protocol['training_stage'] != args.stage:
            raise ValueError('Requested stage differs from the registered stage')
        with (args.output / 'current_training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in (args.output / 'logs').glob('*.process.json'):
                record = json.loads(path.read_text())
                try:
                    command = (Path('/proc') / str(record['pid']) / 'cmdline').read_bytes()
                except FileNotFoundError:
                    continue
                if str(args.output).encode() in command:
                    raise RuntimeError(f'Prior worker {record["pid"]} is still alive; refusing duplicate execution')
            dump(args.output / 'training_status.json', dict(stage=args.action, started_at=stamp(), pid=os.getpid()))
            try:
                (preflight if args.action == 'preflight' else run_registered)(args.output)
                dump(args.output / 'training_status.json', dict(stage='preflight_complete' if args.action == 'preflight' else 'complete', finished_at=stamp()))
            except Exception as error:
                dump(args.output / 'training_status.json', dict(stage='failed', error_type=type(error).__name__, error=str(error)))
                raise
    print(json.dumps(dict(output=str(args.output), action=args.action, stage=protocol['training_stage'],
        state=protocol['state'], runs=len(protocol['all_arms']) * len(protocol['seeds'])), ensure_ascii=False))
