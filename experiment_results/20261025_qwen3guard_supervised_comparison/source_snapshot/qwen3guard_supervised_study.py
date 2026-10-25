"""Fresh matched-data frozen Qwen3Guard classifiers, binary and five-class.

Native generated Guard labels remain a separate zero-shot reference. These
classifiers learn this project's labels from train, select on validation, and
open test only after all eighteen heads have been hash locked.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import qwen3guard_clean_queue as common

DEFAULT_OUTPUT = ROOT / 'outputs/qwen3guard_supervised_comparison_20261025'
DEFAULT_PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
DEFAULT_MODEL = ROOT / 'models/Qwen3Guard-Gen-4B'
ARCHIVE = ROOT / 'experiment_results/20261025_qwen3guard_supervised_comparison'
SCHEMA = 'qwen3guard_supervised_matched_data_v1'
BUDGETS = {'extract': {'cap_gib': 14., 'headroom_gib': 8.},
           'fit': {'cap_gib': 4., 'headroom_gib': 4.},
           'evaluate': {'cap_gib': 4., 'headroom_gib': 4.}}
_VERIFIED = {}


def module(kind):
    suffix = 'features' if kind == 'features' else 'training'
    return importlib.import_module('scripts.hanguard.qwen3guard_supervised_' + suffix)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.write(path, value)


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, inode=value.st_ino, device=value.st_dev)


def initialize(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT, model=DEFAULT_MODEL):
    output, parent, model = map(lambda p: Path(p).resolve(), (output, parent, model))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        return verify(output)
    if any(output.iterdir()):
        raise ValueError('Fresh study initialization requires an empty output directory')
    prior = common.read(parent / 'protocol.json')
    source_names = ['qwen3guard_supervised_study', 'qwen3guard_supervised_features',
        'qwen3guard_supervised_training', 'qwen3guard_clean_queue', 'paper_pooling_controls',
        'clean_ablation_training', 'clean_ablation_heads', 'multilabel_heads',
        'frozen_lora_layer_diagnostics']
    paths = [ROOT / 'scripts/hanguard' / (name + '.py') for name in source_names]
    paths += [ROOT / 'tests' / ('test_' + name + '.py') for name in source_names[:3]]
    model_paths = [model / name for name in ('config.json', 'generation_config.json',
        'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
        'model.safetensors.index.json')]
    model_paths += sorted(model.glob('model-*.safetensors'))
    if len(list(model.glob('model-*.safetensors'))) != 3:
        raise ValueError('Expected all three local Guard weight shards')
    provenance = {str(path): dict(sha256=common.sha(path), stat=stat(path)) for path in model_paths}
    data_dir = parent / 'data'
    for split, digest in prior['data_sha256'].items():
        if common.sha(data_dir / (split + '.parquet')) != digest:
            raise ValueError('Parent raw data SHA mismatch')
    protocol = dict(schema=SCHEMA, output=str(output), parent=str(parent),
        parent_data_dir=str(data_dir), parent_protocol_sha256=common.sha(parent / 'protocol.json'),
        data_sha256=prior['data_sha256'], rows=prior['rows'], model_path=str(model),
        task_rows={'binary': {'train': 66802, 'validation': 8350, 'test': 8350},
                   'category': {'train': 17440, 'validation': 2180, 'test': 2180}},
        model_files=provenance, model_type='qwen3', hidden_size=2560, num_layers=36, num_hidden_layers=36,
        use_lora=False, backbone_trainable_parameters=0,
        tasks=['binary', 'category'], seeds=[42, 43, 44],
        arms=['last_mlp', 'mean_pool_mlp', 'shared_attention_mlp'], formal_heads=18,
        training=dict(width=128, dropout=.1, lr=.001, learning_rate=.001,
            weight_decay=.001, grad_clip=5., effective_batch=128, effective_batch_size=128,
            max_micro=32, max_micro_batch_size=32, token_budget=8192, pad_multiple=1,
            epochs=12, max_epochs=40, epoch_budgets=[12, 24, 40], bootstrap_draws=2000,
            selection_metric='validation_cross_entropy'),
        shard_rows=512, extraction=dict(max_tokens=32768, max_micro=8, token_budget=4096),
        input='Official one-user Guard chat template, full input, no truncation',
        capture='Raw final transformer block output before final RMSNorm, frozen BF16 backbone',
        pooling='Mean and independent shared attention read original-text token spans only',
        last_readout='Last valid token of full official template, including native assistant prefix',
        test_barrier='All 18 heads locked before any test tokenization or extraction',
        posthoc_test_context='Follow-up on an already inspected project test set; no unseen-test claim',
        reporting='Report all three heads and all seeds, never select the largest test accuracy',
        native_reference=str(ROOT / 'outputs/qwen3guard_clean_test_20261025_v2'),
        historical_features_used=False, historical_predictions_used=False,
        gpu_policy='User-authorized sharing; measured free memory, allocator cap, no foreign signals',
        budgets=BUDGETS, gpu_ids=list(range(8)), poll_seconds=3., bundle_shards=4,
        sources={str(path): common.sha(path) for path in paths}, created_unix=time.time())
    write(output / 'protocol.json', protocol)
    write(output / 'protocol_registration.json', dict(protocol_sha256=common.sha(output / 'protocol.json')))
    snapshots = output / 'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for path, digest in protocol['sources'].items():
        target = snapshots / Path(path).name
        shutil.copyfile(path, target)
        if common.sha(target) != digest:
            raise ValueError('Source changed during registration')
    refresh_archive(output)
    return protocol


def verify(output):
    output = Path(output).resolve()
    protocol = common.read(output / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('output') != str(output)
            or protocol.get('budgets') != BUDGETS
            or common.read(output / 'protocol_registration.json').get('protocol_sha256')
            != common.sha(output / 'protocol.json')):
        raise ValueError('Study registration changed')
    expected = dict(protocol['sources'])
    expected[str(Path(protocol['parent']) / 'protocol.json')] = protocol['parent_protocol_sha256']
    expected.update({str(Path(protocol['parent_data_dir']) / (split + '.parquet')): digest
                     for split, digest in protocol['data_sha256'].items()})
    for path, digest in expected.items():
        identity = stat(path)
        if _VERIFIED.get(path) != (identity, digest):
            if common.sha(path) != digest:
                raise ValueError('Registered input changed: ' + path)
            _VERIFIED[path] = (identity, digest)
    for path, record in protocol['model_files'].items():
        if stat(path) != record['stat']:
            raise ValueError('Model artifact changed: ' + path)
    return protocol


def eligible(card, phase):
    budget = BUDGETS[phase]
    return card['free_gib'] >= budget['cap_gib'] + budget['headroom_gib']


def job_key(job):
    if job['phase'] == 'extract':
        return 'extract:' + job['split'] + ':' + ','.join(map(str, job['shards']))
    return ':'.join(str(job[k]) for k in ('phase', 'task', 'seed'))


def job_units(job):
    if job['phase'] == 'extract':
        return {f"extract:{job['split']}:{s}" for s in job['shards']}
    return {job_key(job)}


def available_jobs(todo, active_jobs, failures):
    """A committed shard can change bundle boundaries while a worker runs."""
    occupied = set().union(*(job_units(j) for j in active_jobs)) if active_jobs else set()
    blocked = set(failures)
    result = []
    for original in todo:
        job = dict(original)
        if job['phase'] == 'extract':
            job['shards'] = [s for s in job['shards']
                if f"extract:{job['split']}:{s}" not in occupied | blocked]
            if not job['shards']:
                continue
        elif job_units(job) & (occupied | blocked):
            continue
        result.append(job)
    return result


def status(output):
    output = Path(output)
    result = dict(features={})
    for split in ('train', 'validation', 'test'):
        path = output / 'features' / split / 'manifest.json'
        if path.exists():
            plan = common.read(path)
            done = module('features').complete_jobs(output, split)
            result['features'][split] = dict(total_shards=len(plan['shards']),
                complete_shards=len(done), complete=len(done) == len(plan['shards']))
    result['training'] = module('training').progress(output)
    return result


def extraction_jobs(output, split, bundle=4):
    output = Path(output)
    plan_path = output / 'features' / split / 'manifest.json'
    if not plan_path.exists():
        return []
    plan = common.read(plan_path)
    done = set(module('features').complete_jobs(output, split))
    todo = [entry['shard_index'] for entry in plan['shards'] if entry['shard_index'] not in done]
    return [dict(phase='extract', split=split, shards=todo[i:i + bundle])
            for i in range(0, len(todo), bundle)]


def pending(output, current):
    for split in ('train', 'validation'):
        if not current['features'].get(split, {}).get('complete'):
            # Prepare both manifests before entering the scheduling loop.
            return extraction_jobs(output, split)
    fit = [job for job in current['training'].get('pending_jobs', []) if job['phase'] == 'fit']
    if fit:
        return fit
    if not (Path(output) / 'selection_lock.json').exists():
        module('training').lock_selections(output)
    if 'test' not in current['features']:
        module('features').prepare(output, 'test')
        return extraction_jobs(output, 'test')
    if not current['features']['test']['complete']:
        return extraction_jobs(output, 'test')
    # Serial evaluation avoids shared metadata/file-lock races.
    return [job for job in current['training'].get('pending_jobs', [])
            if job['phase'] == 'evaluate'][:1]


def state(output, phase, **details):
    write(Path(output) / 'queue_state.json', dict(schema=SCHEMA, state=phase,
        automatic_run=phase not in ('complete', 'failed', 'stopped'),
        coordinator_identity=common.identity(os.getpid()), updated_unix=time.time(), **details))


def worker(args):
    output = args.output.resolve()
    verify(output)
    budget = BUDGETS[args.phase]
    try:
        card = next(row for row in common.gpu_inventory() if row['index'] == args.gpu)
    except Exception:
        return 75
    if os.environ.get('CUDA_VISIBLE_DEVICES') != card['uuid'] or not eligible(card, args.phase):
        return 75
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(budget['cap_gib'] * 1024**3 / total, device=0)
    try:
        fresh = next(row for row in common.gpu_inventory() if row['index'] == args.gpu)
    except Exception:
        return 75
    if fresh['uuid'] != card['uuid'] or fresh['free_gib'] < sum(budget.values()) - 1.:
        return 75
    owner = common.identity(os.getpid())
    record = dict(owner, gpu_uuid=card['uuid'], physical_gpu=args.gpu,
        allocator_cap_gib=budget['cap_gib'], entry_inventory=card,
        protocol_sha256=common.sha(output / 'protocol.json'), phase=args.phase,
        task=args.task, seed=args.seed, split=args.split, shards=args.shards, started_unix=time.time())
    record_path = output / 'admissions' / f"{owner['pid']}_{owner['start_time_ticks']}.json"
    write(record_path, record)
    os.environ['HANGUARD_GUARD_SUPERVISED_ADMITTED'] = record['protocol_sha256']
    os.environ['HANGUARD_GUARD_SUPERVISED_ADMISSION_RECORD'] = str(record_path)
    try:
        if args.phase == 'extract':
            module('features').extract_many(output, args.split, args.shards, device='cuda')
        else:
            getattr(module('training'), args.phase)(output, args.task, args.seed, device='cuda')
        record['exit_code'] = 0
        return 0
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record.update(finished_unix=time.time(),
            peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(0) / 1024**3)
        write(record_path, record)


def refresh_archive(output):
    output = Path(output).resolve()
    archive = ARCHIVE if output == DEFAULT_OUTPUT.resolve() else output / 'archive'
    archive.mkdir(parents=True, exist_ok=True)
    names = ['protocol.json', 'protocol_registration.json', 'queue_state.json',
             'selection_lock.json', 'results.json', 'coordinator_launch.json', 'README.md']
    for name in names:
        source = output / name
        if source.exists():
            shutil.copyfile(source, archive / name)
    for pattern in ('source_snapshot/*.py', 'jobs/*/s*/selection.json', 'jobs/*/s*/test_results.json',
                    'jobs/*/s*/test_complete.json', 'jobs/*/s*/validation_predictions.parquet',
                    'jobs/*/s*/test_predictions.parquet', 'features/*/manifest.json'):
        for source in output.glob(pattern):
            target = archive / source.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (archive / 'PROTOCOL_NOTES.md').write_text('\n'.join([
        '# Qwen3Guard 同数据监督分类对照（20261025）', '',
        '冻结 Qwen3Guard-Gen-4B，不使用 LoRA；二分类和原生两来源五分类各三个 seed。',
        '末位置 MLP、均值池化 MLP、独立共享注意力 MLP 全部报告。',
        '官方完整模板；池化仅原始文本 token，末位置读出含官方后缀。',
        '与主实验相同 train/validation/test；全部验证集选模锁定后才提取测试特征。',
        '本项目测试集此前已查看，本轮属于后续对照。',
        '原生生成二分类在 ../20261025_qwen3guard_clean_test_v2 单列，不能混称监督成绩。',
        '', '实时状态：`outputs/qwen3guard_supervised_comparison_20261025/queue_state.json`。', '']))
    files = {str(p.relative_to(archive)): common.sha(p) for p in sorted(archive.rglob('*'))
             if p.is_file() and p != archive / 'manifest.json'}
    write(archive / 'manifest.json', dict(schema=SCHEMA, files=files, source=str(output)))


def run(output):
    output = Path(output).resolve()
    protocol = verify(output)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    locks = ROOT / 'outputs/.qwen3guard_gpu_admission'
    locks.mkdir(exist_ok=True)
    with (output / '.coordinator.lock').open('a+') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = common.read(output / 'queue_state.json') if (output / 'queue_state.json').exists() else {}
        if any(common.same_process(w['identity']) for w in prior.get('active_workers', [])):
            raise RuntimeError('An exact prior worker remains alive')
        active, failures, events, archive_at = {}, {}, [], 0.
        try:
            for split in ('train', 'validation'):
                state(output, 'preparing_' + split, active_workers=[])
                module('features').prepare(output, split)
            while True:
                verify(output)
                for gpu, entry in list(active.items()):
                    code = entry['process'].poll()
                    if code is None:
                        continue
                    entry['log'].close()
                    entry['gpu_lock'].close()
                    events.append(dict(event='exit', job=entry['job'], gpu=gpu, exit_code=code, time=time.time()))
                    if code not in (0, 75):
                        for unit in job_units(entry['job']):
                            failures[unit] = dict(exit_code=code, log=entry['log_path'])
                    del active[gpu]
                current = status(output)
                todo = pending(output, current)
                actionable = available_jobs(todo, [w['job'] for w in active.values()], failures)
                if not todo and not active:
                    if failures:
                        raise RuntimeError('Workers failed; inspect preserved failures before recovery')
                    module('training').summarize(output)
                    state(output, 'complete', progress=status(output), active_workers=[], failures=failures)
                    refresh_archive(output)
                    return
                if not actionable and not active and failures:
                    state(output, 'failed', progress=current, active_workers=[], failures=failures)
                    refresh_archive(output)
                    return
                if (output / 'STOP').exists() and not active:
                    state(output, 'stopped', progress=current, active_workers=[], failures=failures)
                    refresh_archive(output)
                    return
                try:
                    cards, inventory_error = common.gpu_inventory(), None
                except Exception as exc:
                    cards, inventory_error = [], repr(exc)
                for card in cards:
                    gpu = card['index']
                    if gpu in active or not actionable or (output / 'STOP').exists():
                        continue
                    job = next((j for j in actionable if eligible(card, j['phase'])), None)
                    if job is None:
                        continue
                    lock = (locks / (card['uuid'] + '.lock')).open('a+')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    try:
                        fresh = next(g for g in common.gpu_inventory() if g['uuid'] == card['uuid'])
                    except Exception:
                        lock.close()
                        continue
                    if not eligible(fresh, job['phase']):
                        lock.close()
                        continue
                    command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                               '--output', str(output), '--gpu', str(gpu), '--phase', job['phase']]
                    if job['phase'] == 'extract':
                        command += ['--split', job['split'], '--shards', *map(str, job['shards'])]
                    else:
                        command += ['--task', job['task'], '--seed', str(job['seed'])]
                    logfile = logs / (job_key(job).replace(':', '_').replace(',', '-') + f'_{time.time_ns()}.log')
                    stream = logfile.open('a')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                    # The child retains the admission lock if this coordinator
                    # exits, so another queue cannot overlap its live worker.
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    owner = common.identity(child.pid)
                    if owner is None:
                        stream.close()
                        lock.close()
                        raise RuntimeError('Launched worker exited before identity capture')
                    active[gpu] = dict(process=child, log=stream, gpu_lock=lock, identity=owner,
                        job=job, log_path=str(logfile), launched_unix=time.time(), admitted_inventory=fresh)
                    actionable.remove(job)
                    events.append(dict(event='launch', gpu=gpu, job=job, identity=owner, time=time.time()))
                workers = [{k: v for k, v in entry.items() if k not in ('process', 'log', 'gpu_lock')}
                           for entry in active.values()]
                state(output, 'running' if active else 'waiting_for_capacity', progress=current,
                      active_workers=workers, failures=failures, gpu_inventory=cards,
                      inventory_error=inventory_error, recent_events=events[-40:])
                if time.time() - archive_at >= 300:
                    refresh_archive(output)
                    archive_at = time.time()
                time.sleep(protocol['poll_seconds'])
        except BaseException as exc:
            workers = [{k: v for k, v in entry.items() if k not in ('process', 'log', 'gpu_lock')}
                       for entry in active.values()]
            state(output, 'failed', error=repr(exc), active_workers=workers, failures=failures)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'worker', 'status', 'archive'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--phase', choices=tuple(BUDGETS))
    parser.add_argument('--split', choices=('train', 'validation', 'test'))
    parser.add_argument('--shards', type=int, nargs='+')
    parser.add_argument('--task', choices=('binary', 'category'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.action == 'init':
        initialize(args.output, args.parent, args.model)
    elif args.action == 'run':
        run(args.output)
    elif args.action == 'worker':
        if args.gpu is None or args.phase is None:
            parser.error('Worker needs --gpu and --phase')
        if args.phase == 'extract' and (args.split is None or not args.shards):
            parser.error('Extraction needs --split and --shards')
        if args.phase != 'extract' and (args.task is None or args.seed is None):
            parser.error('Training/evaluation needs --task and --seed')
        return worker(args)
    elif args.action == 'archive':
        refresh_archive(args.output)
    else:
        print(common.read(args.output / 'queue_state.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
