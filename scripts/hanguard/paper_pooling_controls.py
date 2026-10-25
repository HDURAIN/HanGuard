"""Generic full-token pooling controls for the completed 20261022 experiment.

The mean control averages *projected, normalized valid token values*. The
attention control uses one learned, class-independent query and the same value
map. Both use a conventional MLP classifier. Output-layer class weights are
ordinary classification weights, not explicit class queries or class identity
embeddings. Parameter counts are reported, not inflated with unused parameters.

Only the completed clean parent's data and newly extracted features are reused.
Its files are read-only. This is a registered follow-up after earlier test
inspection, not a claim that the entire project has a never-inspected test set.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import clean_ablation_training as original

DEFAULT_PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
DEFAULT_OUTPUT = ROOT / 'outputs/hanguard_paper_followups_20261025/pooling'
SCHEMA = 'hanguard_generic_pooling_followup_v1'
ARMS = ('mean_pool_mlp', 'shared_attention_mlp')
TASKS = ('binary', 'category')
SPLITS = ('train', 'validation', 'test')
DEFAULT_SETTINGS = dict(width=128, dropout=.1, lr=.001, weight_decay=.001,
    grad_clip=5., effective_batch=128, max_micro=32, token_budget=8192,
    pad_multiple=1, epochs=12, max_epochs=40, bootstrap_draws=2000)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stat_identity(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                inode=value.st_ino, device=value.st_dev)


@contextmanager
def locked(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


class PoolingHead(nn.Module):
    def __init__(self, hidden_size, width=128, num_classes=5,
                 arm='mean_pool_mlp', dropout=.1, seed=42):
        super().__init__()
        if arm not in ARMS or hidden_size < 1 or width < 1 or num_classes not in (2, 5):
            raise ValueError('Invalid generic pooling head')
        self.hidden_size, self.width, self.num_classes, self.arm = hidden_size, width, num_classes, arm
        self.dropout_rate = dropout
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            # Shared initialization of the common value/scoring maps in both controls.
            self.value_project = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
            self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(width, num_classes))
            if arm == 'shared_attention_mlp':
                self.key_project = nn.Linear(hidden_size, width)
                self.key_norm = nn.LayerNorm(width)
                self.global_query = nn.Parameter(torch.randn(width))

    def metadata(self):
        return dict(arm=self.arm, hidden_size=self.hidden_size, width=self.width,
            num_classes=self.num_classes, dropout=self.dropout_rate,
            readout='L32_all_valid_tokens', explicit_class_embeddings=False,
            mean_definition='mean of LN(Linear/GELU(token-wise LN(raw))) projected values',
            attention_query='one independent learned width-vector' if hasattr(self, 'global_query') else None,
            classifier='ordinary width->width->classes MLP',
            parameters=sum(parameter.numel() for parameter in self.parameters()),
            parameter_count_matching='Actual functional parameter count; no dummy parameters')

    def _normalized(self, tokens, mask):
        if (tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size
                or mask.shape != tokens.shape[:2] or not mask.bool().any(1).all()):
            raise ValueError('Nonempty aligned full-token values and mask required')
        valid = mask.bool()
        raw = tokens.masked_fill(~valid[..., None], 0.).float()
        return F.layer_norm(raw, (self.hidden_size,)), valid

    def attention_weights(self, tokens, mask):
        normalized, valid = self._normalized(tokens, mask)
        if self.arm == 'mean_pool_mlp':
            return valid.float() / valid.sum(-1, keepdim=True)
        keys = self.key_norm(self.key_project(normalized))
        query = self.key_norm(self.global_query)
        scores = torch.einsum('d,btd->bt', query, keys) / math.sqrt(self.width)
        return scores.masked_fill(~valid, -torch.inf).softmax(-1)

    def forward(self, tokens, mask):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            normalized, valid = self._normalized(tokens, mask)
            values = self.value_project(normalized)
            if self.arm == 'mean_pool_mlp':
                weights = valid.float() / valid.sum(-1, keepdim=True)
            else:
                keys = self.key_norm(self.key_project(normalized))
                query = self.key_norm(self.global_query)
                scores = torch.einsum('d,btd->bt', query, keys) / math.sqrt(self.width)
                weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
            pooled = torch.einsum('bt,btd->bd', weights, values)
            return self.classifier(pooled)


def initialize(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh pooling follow-up directory')
    p = original.read_protocol(parent)
    main = read(parent / 'results.json')
    if (not main.get('complete') or main.get('protocol_sha256') != sha(parent / 'protocol.json')
            or main.get('rootselection_lock_sha256') != sha(parent / 'selection_lock.json')):
        raise ValueError('Completed parent experiment is required')
    original.verify_selection_lock(parent)
    output.mkdir(parents=True)
    inputs = {}
    def bind(path, digest=None, large=False):
        path = Path(path).resolve()
        if not path.is_relative_to(parent):
            raise ValueError('A follow-up input escaped the completed clean parent')
        before = stat_identity(path)
        actual = sha(path)
        if digest is not None and digest != actual:
            raise ValueError(f'Parent cache digest differs: {path}')
        if before != stat_identity(path):
            raise ValueError('Parent input changed during full digest verification')
        inputs[str(path)] = dict(sha256=actual, stat=before, large_payload=large)
        write(output / 'initialization_progress.json', dict(state='hashing_parent_inputs',
            verified_files=len(inputs), verified_bytes=sum(v['stat']['size'] for v in inputs.values()),
            last_verified=str(path), updated_unix=time.time()))
    for name in ('protocol.json', 'protocol_registration.json', 'selection_lock.json', 'results.json'):
        bind(parent / name)
    for split in SPLITS:
        bind(Path(p['data_dir']) / f'{split}.parquet', p['data_sha256'][split])
        folder = parent / 'features' / split
        bind(folder / 'manifest.json')
        bind(folder / 'plan.json')
        manifest, plan = read(folder / 'manifest.json'), read(folder / 'plan.json')
        if (not manifest.get('complete') or not manifest.get('joint_same_forward')
                or manifest.get('plan_sha256') != sha(folder / 'plan.json')
                or manifest.get('protocol_sha256') != sha(parent / 'protocol.json')):
            raise ValueError('Incomplete or inconsistent fresh parent feature manifest')
        for name in ('metadata.parquet', 'lengths.npy'):
            bind(folder / name, plan['artifacts'][name])
        for chunk in manifest['chunks']:
            bind(chunk['receipt_path'], chunk['receipt_sha256'])
            for name, item in chunk['artifacts'].items():
                bind(item['path'], item['sha256'], large=name in ('last.npy', 'tokens.npy'))
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    sources.update(p.get('code_sha256', {}))
    protocol = dict(schema=SCHEMA, output=str(output), parent=str(parent),
        parent_protocol_sha256=sha(parent / 'protocol.json'),
        parent_selection_lock_sha256=sha(parent / 'selection_lock.json'),
        tasks=list(TASKS), arms=list(ARMS), seeds=[42, 43, 44], settings=DEFAULT_SETTINGS,
        inputs=inputs, sources=sources, data_sha256=p['data_sha256'],
        feature_manifests={split: sha(parent / 'features' / split / 'manifest.json') for split in SPLITS},
        reused_features='Only the newly extracted, completed 20261022 clean parent cache',
        extra_training='Two new generic pooling heads per task/seed, from scratch',
        test_policy='All six task/seed selections and binary thresholds locked before follow-up test loading',
        provenance_limit='Follow-up controls were proposed after inspecting the completed parent test results',
        cuda_allocator_budget_gib=4, cpu_threads=8, created_unix=time.time())
    (output / 'source_snapshot').mkdir()
    snapshot = output / 'source_snapshot/paper_pooling_controls.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    if sha(snapshot) != sources[str(Path(__file__).resolve())]:
        raise ValueError('Follow-up code changed during registration')
    write(output / 'protocol.json', protocol)
    write(output / 'protocol_registration.json', dict(protocol_sha256=sha(output / 'protocol.json')))
    write(output / 'state.json', dict(state='registered', updated_unix=time.time()))
    return progress(output)


def verify(output):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('output') != str(output)
            or read(output / 'protocol_registration.json').get('protocol_sha256') != sha(output / 'protocol.json')
            or protocol.get('arms') != list(ARMS) or protocol.get('tasks') != list(TASKS)):
        raise ValueError('Pooling follow-up protocol changed')
    for name, digest in protocol['sources'].items():
        if sha(name) != digest:
            raise ValueError(f'Registered implementation changed: {name}')
    for name, item in protocol['inputs'].items():
        current = stat_identity(name)
        if current != item['stat']:
            raise ValueError(f'Parent input strong identity changed: {name}')
        # This filesystem can give same-size rewrite+mtime restoration the same
        # coarse ctime tick. Fresh timestamps need a digest until they are old
        # enough to use the immutable-file stat shortcut safely.
        recent = max(current['mtime_ns'], current['ctime_ns']) >= time.time_ns() - 100_000_000
        if (not item.get('large_payload') or recent) and sha(name) != item['sha256']:
            raise ValueError(f'Parent input digest changed: {name}')
    if (not protocol['seeds'] or len(set(protocol['seeds'])) != len(protocol['seeds'])
            or any(type(seed) is not int or seed < 0 for seed in protocol['seeds'])):
        raise ValueError('Invalid follow-up seeds')
    original.settings(dict(settings=protocol['settings']))
    return protocol


def _admit(output, device):
    if torch.device(device).type != 'cuda':
        return
    protocol_sha = sha(Path(output) / 'protocol.json')
    path = os.environ.get('HANGUARD_FOLLOWUP_ADMISSION_RECORD', '')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (os.environ.get('HANGUARD_FOLLOWUP_ADMITTED') != protocol_sha
            or not Path(path).is_absolute() or not visible.startswith('GPU-') or ',' in visible):
        raise RuntimeError('CUDA worker requires the root runner admission record')
    record = read(path)
    folder = Path('/proc') / str(os.getpid())
    stat = (folder / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    actual = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=int(fields[19]),
                  command_sha256=hashlib.sha256((folder / 'cmdline').read_bytes()).hexdigest())
    identity = record.get('identity', record)
    if (any(identity.get(key) != value for key, value in actual.items())
            or record.get('gpu_uuid') != visible
            or record.get('followup_protocol_sha256') != protocol_sha
            or record.get('allocator_cap_gib') != 4):
        raise RuntimeError('CUDA admission record does not belong to this exact worker')


def _stop(output):
    if (Path(output) / 'STOP').exists():
        raise RuntimeError('Pooling follow-up STOP requested')


def _load_split(output, split):
    protocol = verify(output)
    if split not in SPLITS:
        raise ValueError('Unknown split')
    if split == 'test':
        verify_test_lock(output)
    parent = Path(protocol['parent'])
    folder = parent / 'features' / split
    plan, manifest = read(folder / 'plan.json'), read(folder / 'manifest.json')
    frame = pd.read_parquet(parent / 'data' / f'{split}.parquet', columns=[
        'base_id', 'group_id', 'source', 'prompt_harm_label', 'category_id', 'type_label_mask'])
    frame = frame.sort_values('base_id', kind='stable').reset_index(drop=True)
    meta = pd.read_parquet(folder / 'metadata.parquet')
    lengths = np.load(folder / 'lengths.npy', mmap_mode='r', allow_pickle=False)
    labels = original.label_arrays(frame)
    if (frame.base_id.tolist() != meta.base_id.tolist()
            or frame.group_id.tolist() != meta.group_id.tolist()
            or frame.source.tolist() != meta.source.tolist()
            or not np.array_equal(labels['binary'], meta.label.to_numpy())
            or not np.array_equal(lengths, meta.prompt_tokens.to_numpy())):
        raise ValueError('Raw source and newly reused feature row identities differ')
    from scripts.hanguard.clean_ablation_features import JointCache
    cache = JointCache(plan, dict(meta=meta, lengths=lengths), manifest, folder / 'manifest.json')
    return dict(meta=meta, frame=frame, lengths=np.asarray(lengths), tokens=cache,
                labels=labels, hidden_size=plan['hidden_size'], identity=dict(
                    dataset_sha256=protocol['data_sha256'][split],
                    manifest_sha256=protocol['feature_manifests'][split], rows=len(meta)))


def _rows(data, task):
    return np.arange(len(data['meta'])) if task == 'binary' else np.flatnonzero(data['labels']['typed'])


def _labels(data, task):
    return data['labels']['binary'] if task == 'binary' else data['labels']['categories'] - 1


def token_batch(data, rows, device, pad_multiple=1):
    maximum = int(data['lengths'][rows].max())
    maximum = (maximum + pad_multiple - 1) // pad_multiple * pad_multiple
    values = np.zeros((len(rows), maximum, data['hidden_size']), dtype=np.float16)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for index, row in enumerate(rows):
        tokens = np.asarray(data['tokens'].get_row(int(row)))
        length = int(data['lengths'][row])
        if tokens.shape != (length, data['hidden_size']) or not np.isfinite(tokens).all():
            raise ValueError('Invalid complete parent token row')
        values[index, :length] = tokens
        mask[index, :length] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)


def build_heads(hidden, task, seed, cfg, device):
    return {arm: PoolingHead(hidden, cfg['width'], 2 if task == 'binary' else 5,
        arm, cfg['dropout'], seed).to(device) for arm in ARMS}


def train_macro(heads, optimizers, data, rows, task, cfg, device, *, seed, epoch, step, oom_events):
    budget, max_micro = cfg['token_budget'], cfg['max_micro']
    while True:
        for arm in ARMS:
            heads[arm].train()
            optimizers[arm].zero_grad(set_to_none=True)
        totals = dict.fromkeys(ARMS, 0.)
        try:
            for micro, indices in enumerate(original.micro_batches(rows, data['lengths'], budget,
                                                                    max_micro, cfg['pad_multiple'])):
                tokens, mask = token_batch(data, indices, device, cfg['pad_multiple'])
                labels = torch.as_tensor(_labels(data, task)[indices], dtype=torch.long, device=device)
                for arm in ARMS:
                    original._set_seed(original._step_seed(seed, epoch, step, micro), device)
                    loss = F.cross_entropy(heads[arm](tokens, mask), labels, reduction='sum') / len(rows)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite pooling control loss')
                    loss.backward()
                    totals[arm] += float(loss.detach())
            break
        except torch.OutOfMemoryError:
            pass
        tokens = mask = labels = loss = None
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        longest = int(max(data['lengths'][rows]))
        longest = (longest + cfg['pad_multiple'] - 1) // cfg['pad_multiple'] * cfg['pad_multiple']
        if max_micro == 1 and budget <= longest:
            raise RuntimeError('One full token row does not fit; refusing truncation')
        max_micro, budget = max(1, max_micro // 2), max(longest, budget // 2)
        oom_events.append(dict(epoch=epoch, macro_step=step, max_micro=max_micro, token_budget=budget))
    for arm in ARMS:
        torch.nn.utils.clip_grad_norm_(heads[arm].parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        optimizers[arm].step()
    return totals


@torch.no_grad()
def predict(heads, data, rows, cfg, device, output, *, oom_events=None, phase='evaluation'):
    oom_events = [] if oom_events is None else oom_events
    rows = np.asarray(rows, np.int64)
    predictions = {arm: np.empty((len(rows), heads[arm].num_classes), np.float32) for arm in ARMS}
    position = {int(row): index for index, row in enumerate(rows)}
    order = rows[np.argsort(data['lengths'][rows], kind='stable')]
    for head in heads.values():
        head.eval()
    def forward(indices):
        _stop(output)
        try:
            tokens, mask = token_batch(data, indices, device, cfg['pad_multiple'])
            result = {arm: heads[arm](tokens, mask).float().cpu().numpy() for arm in ARMS}
        except torch.OutOfMemoryError:
            result = None
        if result is None:
            tokens = mask = None
            torch.cuda.empty_cache()
            if len(indices) == 1:
                raise RuntimeError('One full evaluation row does not fit; refusing truncation')
            oom_events.append(dict(event='evaluation_oom_split', phase=phase,
                                   rows=len(indices), row_indices=indices.tolist()))
            middle = len(indices) // 2
            forward(indices[:middle]); forward(indices[middle:])
            return
        locations = [position[int(row)] for row in indices]
        for arm in ARMS:
            if not np.isfinite(result[arm]).all():
                raise FloatingPointError('Nonfinite generic pooling predictions')
            predictions[arm][locations] = result[arm]
    for indices in original.micro_batches(order, data['lengths'], cfg['token_budget'], cfg['max_micro'], cfg['pad_multiple']):
        forward(indices)
    return predictions


def _frame(data, rows, logits):
    frame = data['meta'].iloc[rows][['base_id', 'group_id', 'source', 'prompt_tokens']].reset_index(drop=True).copy()
    frame['binary_label'] = data['labels']['binary'][rows]
    frame['category_id'] = data['labels']['categories'][rows]
    frame['known_native_category'] = data['labels']['known_native'][rows]
    for arm in ARMS:
        probabilities = original._probabilities(logits[arm])
        for category in range(probabilities.shape[1]):
            frame[f'{arm}_p{category}'] = probabilities[:, category]
        frame[f'{arm}_argmax'] = probabilities.argmax(-1)
    return frame


def job_folder(output, task, seed):
    return Path(output) / 'jobs' / task / f's{seed}'


def extension_rule(history, best_epochs, budget, maximum):
    if budget >= maximum or len(history) < 3:
        return None
    triggers = {}
    for arm in ARMS:
        first, last = history[-3]['validation_ce'][arm], history[-1]['validation_ce'][arm]
        improvement = (first - last) / max(abs(first), 1e-12)
        if best_epochs[arm] == history[-1]['epoch'] and improvement >= .002:
            triggers[arm] = dict(best_epoch=best_epochs[arm], relative_recent_ce_improvement=improvement)
    if not triggers:
        return None
    target = min(next((value for value in (24, 40) if value > budget), maximum), maximum)
    return dict(previous_budget=budget, new_budget=target, triggers=triggers,
                all_followup_arms_extended=True, criterion='same as parent: best in last epoch and recent relative CE improvement >= .002')


def _selection(output, task, seed):
    folder = job_folder(output, task, seed)
    selected = read(folder / 'selection.json')
    if (selected.get('protocol_sha256') != sha(Path(output) / 'protocol.json')
            or selected.get('task') != task or selected.get('seed') != seed
            or set(selected.get('arms', {})) != set(ARMS)):
        raise ValueError('Pooling control selection identity changed')
    for name in ('selected_heads.pt', 'validation_predictions.parquet'):
        if sha(folder / name) != selected['artifacts'][str((folder / name).resolve())]:
            raise ValueError('Pooling validation selection artifact changed')
    return selected


def lock_selections(output):
    output = Path(output)
    protocol = verify(output)
    with locked(output / '.selection_lock'):
        artifacts, selections = {}, {}
        for task in TASKS:
            for seed in protocol['seeds']:
                folder = job_folder(output, task, seed)
                if not (folder / 'selection.json').exists():
                    return None
                selected = _selection(output, task, seed)
                selections[f'{task}/s{seed}'] = sha(folder / 'selection.json')
                artifacts[str((folder / 'selection.json').resolve())] = sha(folder / 'selection.json')
                artifacts.update(selected['artifacts'])
        expected = dict(complete=True, protocol_sha256=sha(output / 'protocol.json'),
                        selections=selections, artifacts=artifacts, scope='all tasks and seeds before any new test evaluation')
        path = output / 'test_selection_lock.json'
        if path.exists() and read(path) != expected:
            raise ValueError('Pooling global test barrier changed')
        if not path.exists():
            write(path, expected)
        return expected


def verify_test_lock(output):
    output = Path(output)
    if not (output / 'test_selection_lock.json').exists():
        raise ValueError('All pooling task/seed selections must be locked before test loading')
    lock = lock_selections(output)
    if lock is None:
        raise ValueError('Global pooling test selections are incomplete')
    return lock


def fit(output, task, seed, device='cpu'):
    output = Path(output).resolve()
    protocol = verify(output)
    if task not in TASKS or seed not in protocol['seeds']:
        raise ValueError('Unregistered pooling task or seed')
    _admit(output, device)
    _stop(output)
    torch.set_num_threads(protocol.get('cpu_threads', 8))
    folder = job_folder(output, task, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with locked(folder / '.job.lock', nonblocking=True):
        if not (folder / 'selection.json').exists():
            _fit_locked(output, task, seed, device, protocol, folder)
        selected = _selection(output, task, seed)
        lock = lock_selections(output)
        if lock is not None:
            return _evaluate_locked(output, task, seed, device, protocol, folder)
        return selected


def _fit_locked(output, task, seed, device, protocol, folder):
    if (output / 'test_selection_lock.json').exists():
        raise ValueError('Cannot fit after this follow-up test barrier was opened')
    cfg = protocol['settings']
    train, val = _load_split(output, 'train'), _load_split(output, 'validation')
    for key in ('base_id', 'group_id'):
        if set(train['meta'][key]) & set(val['meta'][key]):
            raise ValueError('Parent split grouping leaked between train and validation')
    tr_rows, val_rows = _rows(train, task), _rows(val, task)
    if not len(tr_rows) or not len(val_rows):
        raise ValueError('Task requires nonempty training and validation rows')
    identity = dict(protocol_sha256=sha(output / 'protocol.json'), task=task, seed=seed,
                    train=train['identity'], validation=val['identity'], device_type=torch.device(device).type)
    heads = build_heads(train['hidden_size'], task, seed, cfg, device)
    optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                  for arm, head in heads.items()}
    completed, budget, history, extensions, oom_events = 0, cfg['epochs'], [], [], []
    best_ce, best_epochs = dict.fromkeys(ARMS, float('inf')), dict.fromkeys(ARMS, 0)
    best_states = {arm: original._cpu(head.state_dict()) for arm, head in heads.items()}
    resume = folder / 'epoch_resume.pt'
    prior_seconds, started = 0., time.monotonic()
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved.get('identity') != identity:
            raise ValueError('Pooling resume data, source protocol, task or device identity changed')
        completed, budget = saved['completed_epochs'], saved['budget']
        history, extensions, oom_events = saved['history'], saved['extensions'], saved['oom_events']
        if len(history) != completed or not 0 < completed <= budget <= cfg['max_epochs']:
            raise ValueError('Invalid pooling epoch history')
        best_ce, best_epochs, best_states = saved['best_ce'], saved['best_epochs'], saved['best_states']
        for arm in ARMS:
            heads[arm].load_state_dict(saved['current_states'][arm], strict=True)
            optimizers[arm].load_state_dict(saved['optimizer_states'][arm])
        original._restore_rng(saved['rng'], device)
        prior_seconds = saved['wall_seconds']
    def commit():
        original.atomic_save(resume, original._cpu(dict(schema=SCHEMA, identity=identity,
            completed_epochs=completed, budget=budget, history=history, extensions=extensions,
            oom_events=oom_events, best_ce=best_ce, best_epochs=best_epochs, best_states=best_states,
            current_states={arm: head.state_dict() for arm, head in heads.items()},
            optimizer_states={arm: opt.state_dict() for arm, opt in optimizers.items()},
            rng=original._rng_state(device), wall_seconds=prior_seconds + time.monotonic() - started)))
    while True:
        while completed < budget:
            _stop(output)
            totals = dict.fromkeys(ARMS, 0.)
            batches = original.macro_batches(tr_rows, seed, completed, cfg['effective_batch'])
            for step, rows in enumerate(batches):
                _stop(output)
                values = train_macro(heads, optimizers, train, rows, task, cfg, device,
                    seed=seed, epoch=completed, step=step, oom_events=oom_events)
                for arm in ARMS:
                    totals[arm] += values[arm] * len(rows)
                if step == 0 or (step + 1) % 25 == 0:
                    write(folder / 'progress.json', dict(state='training', task=task, seed=seed,
                        device_type=torch.device(device).type, completed_epochs=completed,
                        current_epoch=completed + 1, budget=budget, macro_step=step + 1,
                        macro_steps=len(batches), updated_unix=time.time()))
            logits = predict(heads, val, val_rows, cfg, device, output,
                             oom_events=oom_events, phase='validation')
            labels = torch.as_tensor(_labels(val, task)[val_rows], dtype=torch.long)
            validation_ce = {arm: float(F.cross_entropy(torch.from_numpy(logits[arm]).double(), labels)) for arm in ARMS}
            completed += 1
            for arm in ARMS:
                if validation_ce[arm] < best_ce[arm]:
                    best_ce[arm], best_epochs[arm] = validation_ce[arm], completed
                    best_states[arm] = original._cpu(heads[arm].state_dict())
            history.append(dict(epoch=completed, validation_ce=validation_ce,
                                train_ce={arm: totals[arm] / len(tr_rows) for arm in ARMS}))
            commit()
            write(folder / 'history.json', history)
            print(json.dumps(dict(task=task, seed=seed, epoch=completed, budget=budget,
                                  validation_ce=validation_ce)), flush=True)
        extension = extension_rule(history, best_epochs, budget, cfg['max_epochs'])
        if extension is None:
            break
        extensions.append(extension)
        budget = extension['new_budget']
        commit()
    for arm in ARMS:
        heads[arm].load_state_dict(best_states[arm])
    logits = predict(heads, val, val_rows, cfg, device, output,
                     oom_events=oom_events, phase='validation_selected')
    original._save_frame(folder / 'validation_predictions.parquet', _frame(val, val_rows, logits))
    original.atomic_save(folder / 'selected_heads.pt', dict(schema=SCHEMA, identity=identity,
        states=best_states, metadata={arm: head.metadata() for arm, head in heads.items()}))
    arms = {}
    for arm, head in heads.items():
        item = dict(best_epoch=best_epochs[arm], validation_ce=best_ce[arm], architecture=head.metadata())
        if task == 'binary':
            p = original._probabilities(logits[arm])[:, 1]
            item['threshold'] = original.accuracy_threshold(_labels(val, task)[val_rows], p)
            item['validation_metrics'] = original.binary_metrics(_labels(val, task)[val_rows], p, item['threshold'])
        else:
            item['validation_metrics'] = original.multiclass_metrics(_labels(val, task)[val_rows] + 1,
                                                                    logits[arm].argmax(-1) + 1, range(1, 6))
        arms[arm] = item
    artifacts = {str((folder / name).resolve()): sha(folder / name)
                 for name in ('selected_heads.pt', 'validation_predictions.parquet')}
    write(folder / 'selection.json', dict(schema=SCHEMA, task=task, seed=seed,
        protocol_sha256=identity['protocol_sha256'], arms=arms, artifacts=artifacts,
        training_rows=len(tr_rows), validation_rows=len(val_rows), final_budget=budget,
        extensions=extensions, oom_events=oom_events, settings=cfg, device_type=torch.device(device).type,
        selection_scope='validation_only', formal_test_opened=False,
        train_length_edges=np.quantile(train['lengths'][tr_rows], [.25, .5, .75]).tolist(),
        wall_seconds=prior_seconds + time.monotonic() - started))
    write(folder / 'progress.json', dict(state='selected', task=task, seed=seed,
        device_type=torch.device(device).type, completed_epochs=completed, budget=budget, updated_unix=time.time()))


def _evaluate_locked(output, task, seed, device, protocol, folder):
    lock = verify_test_lock(output)
    selected = _selection(output, task, seed)
    receipt = folder / 'test_complete.json'
    if receipt.exists():
        saved = read(receipt)
        if (saved.get('protocol_sha256') != sha(output / 'protocol.json')
                or saved.get('selection_lock_sha256') != sha(output / 'test_selection_lock.json')):
            raise ValueError('Pooling completed test identity changed')
        for name, digest in saved['artifacts'].items():
            if sha(name) != digest:
                raise ValueError('Pooling completed test artifact changed')
        return read(folder / 'test_results.json')
    data = _load_split(output, 'test')
    checkpoint = torch.load(folder / 'selected_heads.pt', map_location='cpu', weights_only=True)
    if checkpoint['identity']['protocol_sha256'] != sha(output / 'protocol.json'):
        raise ValueError('Selected pooling checkpoint belongs to another protocol')
    heads = build_heads(data['hidden_size'], task, seed, protocol['settings'], device)
    for arm in ARMS:
        heads[arm].load_state_dict(checkpoint['states'][arm], strict=True)
    rows = np.arange(len(data['meta']))
    inference_oom_events = []
    logits = predict(heads, data, rows, protocol['settings'], device, output,
                     oom_events=inference_oom_events, phase='test')
    original._save_frame(folder / 'test_predictions.parquet', _frame(data, rows, logits))
    mask = np.ones(len(rows), bool) if task == 'binary' else data['labels']['typed']
    target = _labels(data, task)[mask]
    decisions, arms = {}, {}
    for arm in ARMS:
        if task == 'binary':
            p = original._probabilities(logits[arm])[:, 1]
            threshold = selected['arms'][arm]['threshold']
            decisions[arm] = (p[mask] >= threshold).astype(int)
            arms[arm] = dict(validation_threshold=original.binary_metrics(target, p[mask], threshold),
                fixed_threshold_05=original.binary_metrics(target, p[mask], .5),
                by_source={str(source): original.binary_metrics(target[data['meta'].source.to_numpy()[mask] == source],
                    p[mask][data['meta'].source.to_numpy()[mask] == source], threshold)
                    for source in sorted(set(data['meta'].source.to_numpy()[mask]))})
        else:
            decisions[arm] = logits[arm][mask].argmax(-1)
            arms[arm] = dict(native_known_harmful=original.multiclass_metrics(target + 1, decisions[arm] + 1, range(1, 6)),
                by_source={str(source): original.multiclass_metrics(target[data['meta'].source.to_numpy()[mask] == source] + 1,
                    decisions[arm][data['meta'].source.to_numpy()[mask] == source] + 1, range(1, 6))
                    for source in sorted(set(data['meta'].source.to_numpy()[mask]))})
    paired = original.paired_changes(target, decisions[ARMS[0]], decisions[ARMS[1]])
    result = dict(schema=SCHEMA, task=task, seed=seed, complete=True,
        protocol_sha256=sha(output / 'protocol.json'), selection_lock_sha256=sha(output / 'test_selection_lock.json'),
        rows=int(mask.sum()), arms=arms, inference_oom_events=inference_oom_events,
        paired_shared_minus_mean=dict(changes=paired,
            bootstrap=original.multiclass_bootstrap(target, decisions[ARMS[1]], decisions[ARMS[0]],
                data['meta'].group_id.to_numpy()[mask], protocol['settings']['bootstrap_draws'], seed)))
    write(folder / 'test_results.json', result)
    artifacts = {str((folder / name).resolve()): sha(folder / name)
                 for name in ('test_predictions.parquet', 'test_results.json')}
    write(receipt, dict(complete=True, protocol_sha256=sha(output / 'protocol.json'),
        selection_lock_sha256=sha(output / 'test_selection_lock.json'), artifacts=artifacts))
    write(folder / 'progress.json', dict(state='test_complete', task=task, seed=seed,
        device_type=torch.device(device).type, updated_unix=time.time()))
    return result


def progress(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    selected, tested, pending = [], [], []
    for task in TASKS:
        for seed in protocol['seeds']:
            folder = job_folder(output, task, seed)
            info = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
            job = dict(task=task, seed=seed)
            if info.get('device_type') and (folder / 'epoch_resume.pt').exists():
                job['resume_device_type'] = info['device_type']
            if (folder / 'selection.json').exists():
                selected.append(dict(task=task, seed=seed))
                if (folder / 'test_complete.json').exists():
                    tested.append(dict(task=task, seed=seed))
            else:
                pending.append(dict(job, phase='fit'))
    total = len(TASKS) * len(protocol['seeds'])
    if len(selected) == total:
        pending = [dict(job, phase='evaluate') for job in selected if job not in tested]
    return dict(total_jobs=total, selected_jobs=len(selected), fit_complete_jobs=len(selected),
        completed_jobs=len(tested), test_complete_jobs=len(tested), total_heads=total * len(ARMS),
        selected_heads=len(selected) * len(ARMS), test_complete_heads=len(tested) * len(ARMS),
        pending_jobs=pending, test_barrier_locked=(output / 'test_selection_lock.json').exists())


def is_complete(output=DEFAULT_OUTPUT):
    path = Path(output) / 'results.json'
    if not path.exists():
        return False
    result = read(path)
    status = progress(output)
    return (result.get('complete') is True
            and result.get('protocol_sha256') == sha(Path(output) / 'protocol.json')
            and status['completed_jobs'] == status['total_jobs'])


def summarize(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    protocol = verify(output)
    status = progress(output)
    if status['completed_jobs'] != status['total_jobs']:
        raise RuntimeError('All pooling fits and tests must complete before aggregation')
    verify_test_lock(output)
    aggregate, per_seed = [], {}
    for task in TASKS:
        per_seed[task] = {}
        for seed in protocol['seeds']:
            folder = job_folder(output, task, seed)
            receipt = read(folder / 'test_complete.json')
            if receipt.get('protocol_sha256') != sha(output / 'protocol.json'):
                raise ValueError('Pooling test belongs to another registered run')
            for name, digest in receipt['artifacts'].items():
                if sha(name) != digest:
                    raise ValueError('Pooling completed test was changed')
            per_seed[task][str(seed)] = read(folder / 'test_results.json')
        for arm in ARMS:
            entries = []
            for seed in protocol['seeds']:
                item = per_seed[task][str(seed)]['arms'][arm]
                entries.append(item['validation_threshold' if task == 'binary' else 'native_known_harmful'])
            keys = ('accuracy', 'f1', 'recall', 'fpr') if task == 'binary' else ('accuracy', 'macro_f1')
            metrics = {key: dict(mean=float(np.mean([item[key] for item in entries])),
                sample_std=float(np.std([item[key] for item in entries], ddof=1)) if len(entries) > 1 else None,
                values=[item[key] for item in entries]) for key in keys}
            aggregate.append(dict(task=task, arm=arm, seeds=len(entries), metrics=metrics))
    result = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(output / 'protocol.json'),
        parent=protocol['parent'], seeds=protocol['seeds'], aggregate=aggregate, per_seed=per_seed,
        parent_reference=read(Path(protocol['parent']) / 'results.json').get('aggregate', []),
        notes=['All control heads are newly trained on the same frozen clean features.',
               'Parameter counts differ functionally; no dummy padding is used.',
               'Follow-up proposed after parent test inspection; no new never-seen test claim.'])
    write(output / 'results.json', result)
    lines = ['# 通用全文池化补充对照', '', '| 任务 | 方法 | Accuracy 均值±SD / % |', '|---|---|---:|']
    for row in aggregate:
        m = row['metrics']['accuracy']
        sd = 'N/A' if m['sample_std'] is None else f"{100 * m['sample_std']:.2f}"
        lines.append(f"| {row['task']} | {row['arm']} | {100 * m['mean']:.2f} ± {sd} |")
    lines += ['', '两个读出均不使用显式类别查询或类别身份向量，使用普通 MLP 输出类别得分。',
        '均值池化明确为：逐 token 无参数 LN → Linear/GELU/LN 值投影 → 有效 token 均值。',
        '共享注意力为单个独立可学习 query；同一文本仅有一份共享摘要。',
        '参数量按每组 selection.json 实际功能参数报告，训练延长和 OOM 拆批亦单独记录。',
        '这是查看父实验结果后登记的补充对照；不声称整个研究的测试集从未被查看。', '']
    (output / 'README.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'worker', 'status', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    if args.action == 'init':
        result = initialize(args.output, args.parent)
    elif args.action == 'worker':
        if args.task is None or args.seed is None:
            parser.error('worker needs --task and --seed')
        try:
            result = fit(args.output, args.task, args.seed, args.device)
        except BlockingIOError:
            print(json.dumps(dict(state='job_already_running', task=args.task, seed=args.seed)), flush=True)
            return 75
    elif args.action == 'status':
        result = progress(args.output)
    else:
        result = summarize(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
