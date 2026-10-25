"""Supervised ordinary readouts on freshly extracted frozen Qwen3Guard features.

Both tasks use the clean parent's unchanged source rows and validation protocol.
No learned queries specific to classes, fusion, LoRA or historical cache is used.
CUDA admission belongs to the persistent root coordinator, checked before work.
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
from scripts.hanguard.paper_pooling_controls import PoolingHead
DEFAULT_OUTPUT = ROOT / 'outputs/qwen3guard_supervised_comparison_20261025'
SCHEMA = 'qwen3guard_supervised_readouts_v1'
ARMS = ('last_mlp', 'mean_pool_mlp', 'shared_attention_mlp')
TASKS = ('binary', 'category')
SPLITS = ('train', 'validation', 'test')


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

def verify(output):
    output = Path(output).resolve()
    p = read(output / 'protocol.json')
    if (p.get('output') != str(output)
            or read(output / 'protocol_registration.json').get('protocol_sha256') != sha(output / 'protocol.json')
            or p.get('tasks') != list(TASKS) or p.get('arms') != list(ARMS)):
        raise ValueError('Guard supervised registration changed')
    for path, value in p.get('sources', {}).items():
        expected = value.get('sha256') if isinstance(value, dict) else value
        if sha(path) != expected:
            raise ValueError(f'Registered source changed: {path}')
    raw = p.get('training', p.get('settings', {}))
    budgets = raw.get('epoch_budgets', [raw.get('epochs',12),24,raw.get('max_epochs',40)])
    cfg = dict(raw, lr=raw.get('lr',raw.get('learning_rate',.001)),
        effective_batch=raw.get('effective_batch',raw.get('effective_batch_size',128)),
        max_micro=raw.get('max_micro',raw.get('max_micro_batch_size',32)),
        epochs=budgets[0], max_epochs=budgets[-1])
    p['settings'] = original.settings({'settings':cfg})
    if not p.get('seeds') or any(type(s) is not int or s < 0 for s in p['seeds']) or len(set(p['seeds'])) != len(p['seeds']):
        raise ValueError('Invalid seeds')
    return p


def _admit(output, device):
    if torch.device(device).type != 'cuda':
        return
    digest = sha(Path(output) / 'protocol.json')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','')
    path = os.environ.get('HANGUARD_GUARD_SUPERVISED_ADMISSION_RECORD','')
    if (os.environ.get('HANGUARD_GUARD_SUPERVISED_ADMITTED') != digest
            or not Path(path).is_absolute() or not visible.startswith('GPU-') or ',' in visible):
        raise RuntimeError('Exact root CUDA admission is required')
    r = read(path)
    folder = Path('/proc') / str(os.getpid())
    stat = (folder / 'stat').read_text()
    actual = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        command_sha256=hashlib.sha256((folder / 'cmdline').read_bytes()).hexdigest())
    identity = r.get('identity',r)
    if (any(identity.get(k) != v for k,v in actual.items()) or r.get('gpu_uuid') != visible
            or r.get('protocol_sha256',r.get('study_protocol_sha256')) != digest
            or r.get('allocator_cap_gib') != 4):
        raise RuntimeError('CUDA admission identity mismatch')


def _stop(output):
    if (Path(output) / 'STOP').exists():
        raise RuntimeError('Guard supervised STOP requested')


class OrdinaryHead(nn.Module):
    def __init__(self, hidden_size, width, num_classes, arm, dropout=.1, seed=42):
        super().__init__()
        if arm not in ARMS or hidden_size < 1 or width < 1 or num_classes not in (2,5):
            raise ValueError('Invalid ordinary readout')
        self.hidden_size,self.width,self.num_classes,self.arm = hidden_size,width,num_classes,arm
        self.dropout_rate = dropout
        if arm == 'last_mlp':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.project=nn.Linear(hidden_size,width)
                self.dropout=nn.Dropout(dropout)
                self.classifier=nn.Linear(width,num_classes)
        else:
            self.pooling=PoolingHead(hidden_size,width,num_classes,arm,dropout,seed)

    def metadata(self):
        pooling = self.pooling.metadata() if hasattr(self,'pooling') else None
        if pooling is not None: pooling['readout']='L36 user-content tokens'
        return dict(arm=self.arm, hidden_size=self.hidden_size, width=self.width,
            num_classes=self.num_classes, dropout=self.dropout_rate,
            readout='L36 last valid full-template token' if self.arm=='last_mlp' else 'L36 user-content tokens',
            explicit_class_embeddings=False, frozen_backbone=True, lora=False,
            parameters=sum(p.numel() for p in self.parameters()),
            pooling=pooling)

    def forward(self,last,tokens,mask):
        if self.arm != 'last_mlp':
            return self.pooling(tokens,mask)
        if last.ndim != 2 or last.shape[-1] != self.hidden_size or not torch.isfinite(last).all():
            raise ValueError('Finite aligned last valid vectors required')
        with torch.autocast(device_type=last.device.type, enabled=False):
            return self.classifier(self.dropout(F.gelu(self.project(F.layer_norm(last.float(),(self.hidden_size,))))))


class ShardCache:
    def __init__(self, folder, manifest, hidden_size, protocol_sha, base_ids):
        self.folder,self.manifest,self.hidden_size,self.protocol_sha = Path(folder),manifest,hidden_size,protocol_sha
        self.shards=manifest['shards']; self.opened={}; self.base_ids=list(base_ids)
        self.ends=np.asarray([s['row_end'] for s in self.shards],np.int64)
        starts=[s['row_start'] for s in self.shards]
        if not len(starts) or starts != [0]+self.ends[:-1].tolist() or int(self.ends[-1])!=manifest['rows']:
            raise ValueError('Shard rows must partition the complete raw input')

    def _open(self, row):
        if row < 0 or row >= self.manifest['rows']:
            raise IndexError(row)
        index=int(np.searchsorted(self.ends,row,side='right'))
        s=self.shards[index]; key=s['shard_index']
        if key not in self.opened:
            folder=self.folder / f'shard_{key:05d}'
            receipt=read(folder/'receipt.json')
            if (receipt.get('protocol_sha256') != self.protocol_sha
                    or receipt.get('manifest_sha256') != sha(self.folder/'manifest.json')
                    or receipt.get('row_start') != s['row_start'] or receipt.get('row_end') != s['row_end']):
                raise ValueError('Feature receipt identity mismatch')
            files=receipt['files']
            for name in ('offsets.npy','token_features.npy','last.npy','metadata.json'):
                expected=files[name].get('sha256') if isinstance(files[name],dict) else files[name]
                if sha(folder/name) != expected:
                    raise ValueError(f'Fresh cache payload changed: {folder/name}')
            metadata=read(folder/'metadata.json')
            if (metadata.get('row_start')!=s['row_start'] or metadata.get('row_end')!=s['row_end']
                    or metadata.get('manifest_sha256')!=sha(self.folder/'manifest.json')
                    or metadata.get('protocol_sha256')!=self.protocol_sha
                    or metadata.get('base_ids')!=self.base_ids[s['row_start']:s['row_end']]):
                raise ValueError('Feature shard metadata identity mismatch')
            offsets=np.load(folder/'offsets.npy',mmap_mode='r',allow_pickle=False)
            tokens=np.load(folder/'token_features.npy',mmap_mode='r',allow_pickle=False)
            last=np.load(folder/'last.npy',mmap_mode='r',allow_pickle=False)
            n=s['row_end']-s['row_start']
            if (offsets.dtype != np.int64 or offsets.shape != (n+1,) or offsets[0]!=0
                    or (np.diff(offsets)<1).any() or tokens.shape!=(int(offsets[-1]),self.hidden_size)
                    or tokens.dtype!=np.float16 or last.shape!=(n,self.hidden_size) or last.dtype!=np.float16):
                raise ValueError('Malformed complete feature shard')
            self.opened[key]=(offsets,tokens,last)
        return s,self.opened[key]

    def get_row(self,row):
        s,(offsets,tokens,last)=self._open(row); local=row-s['row_start']
        return tokens[int(offsets[local]):int(offsets[local+1])]

    def get_last(self,row):
        s,(_,_,last)=self._open(row)
        return last[row-s['row_start']]


def _load_split(output, split):
    p=verify(output); output=Path(output)
    if split not in SPLITS: raise ValueError('Unknown split')
    if split=='test': verify_test_lock(output)
    path=Path(p['parent_data_dir'])/f'{split}.parquet'
    expected=p.get('data_sha256',{}).get(split)
    digest=sha(path)
    if expected is not None and digest!=expected: raise ValueError('Raw dataset digest changed')
    columns=['base_id','group_id','source','prompt_harm_label','category_id','type_label_mask']
    frame=pd.read_parquet(path,columns=columns).reset_index(drop=True)
    labels=original.label_arrays(frame)
    for task,count in (('binary',len(frame)),('category',int(labels['typed'].sum()))):
        expected_rows=p.get('task_rows',{}).get(task,{}).get(split)
        if expected_rows is not None and count!=expected_rows:
            raise ValueError(f'Registered {task}/{split} row count differs')
    folder=output/'features'/split; manifest=read(folder/'manifest.json')
    protocol_sha=sha(output/'protocol.json')
    if (manifest['rows']!=len(frame) or manifest.get('protocol_sha256')!=protocol_sha
            or manifest.get('raw_data_sha256')!=digest):
        raise ValueError('Fresh feature manifest differs from source input')
    # Check prepared identities and lengths before using newly extracted rows.
    for name in ('prepared/content_lengths.npy','prepared/metadata.parquet'):
        expected=manifest['files'][name]
        if isinstance(expected,dict): expected=expected['sha256']
        if sha(folder/name)!=expected: raise ValueError('Fresh preparation payload changed')
    prepared=pd.read_parquet(folder/'prepared/metadata.parquet')
    for column in columns:
        if not prepared[column].reset_index(drop=True).equals(frame[column]):
            raise ValueError(f'Prepared source row identities differ: {column}')
    lengths=np.asarray(np.load(folder/'prepared/content_lengths.npy',allow_pickle=False),np.int64)
    if lengths.shape!=(len(frame),) or (lengths<1).any(): raise ValueError('Invalid token lengths')
    meta=frame[['base_id','group_id','source']].copy(); meta['prompt_tokens']=lengths
    cache=ShardCache(folder,manifest,int(manifest['hidden_size']),protocol_sha,frame.base_id.tolist())
    return dict(meta=meta,frame=frame,lengths=lengths,tokens=cache,labels=labels,
        hidden_size=int(manifest['hidden_size']),identity=dict(dataset_sha256=digest,
        manifest_sha256=sha(folder/'manifest.json'),rows=len(frame)))


def _rows(data, task):
    return np.arange(len(data['meta'])) if task == 'binary' else np.flatnonzero(data['labels']['typed'])

def _labels(data, task):
    return data['labels']['binary'] if task == 'binary' else data['labels']['categories'] - 1

def token_batch(data, rows, device, pad_multiple=1):
    maximum=int(data['lengths'][rows].max())
    maximum=(maximum+pad_multiple-1)//pad_multiple*pad_multiple
    values=np.zeros((len(rows),maximum,data['hidden_size']),np.float16)
    last=np.empty((len(rows),data['hidden_size']),np.float16)
    mask=np.zeros((len(rows),maximum),bool)
    for i,row in enumerate(rows):
        raw=np.asarray(data['tokens'].get_row(int(row)))
        if raw.shape!=(int(data['lengths'][row]),data['hidden_size']) or not np.isfinite(raw).all():
            raise ValueError('Invalid full user-content feature row')
        values[i,:len(raw)]=raw; mask[i,:len(raw)]=True
        last[i]=data['tokens'].get_last(int(row))
    if not np.isfinite(last).all(): raise ValueError('Invalid last feature row')
    return torch.from_numpy(last).to(device),torch.from_numpy(values).to(device),torch.from_numpy(mask).to(device)


def build_heads(hidden,task,seed,cfg,device):
    return {a:OrdinaryHead(hidden,cfg['width'],2 if task=='binary' else 5,a,cfg['dropout'],seed).to(device) for a in ARMS}


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
                last, tokens, mask = token_batch(data, indices, device, cfg['pad_multiple'])
                labels = torch.as_tensor(_labels(data, task)[indices], dtype=torch.long, device=device)
                for arm in ARMS:
                    original._set_seed(original._step_seed(seed, epoch, step, micro), device)
                    loss = F.cross_entropy(heads[arm](last, tokens, mask), labels, reduction='sum') / len(rows)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite pooling control loss')
                    loss.backward()
                    totals[arm] += float(loss.detach())
            break
        except torch.OutOfMemoryError:
            pass
        last = tokens = mask = labels = loss = None
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
            last, tokens, mask = token_batch(data, indices, device, cfg['pad_multiple'])
            result = {arm: heads[arm](last, tokens, mask).float().cpu().numpy() for arm in ARMS}
        except torch.OutOfMemoryError:
            result = None
        if result is None:
            last = tokens = mask = None
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
        path = output / 'selection_lock.json'
        if path.exists() and read(path) != expected:
            raise ValueError('Pooling global test barrier changed')
        if not path.exists():
            write(path, expected)
        return expected

def verify_test_lock(output):
    output = Path(output)
    if not (output / 'selection_lock.json').exists():
        raise ValueError('All pooling task/seed selections must be locked before test loading')
    lock = lock_selections(output)
    if lock is None:
        raise ValueError('Global pooling test selections are incomplete')
    return lock

def selections(output):
    p=verify(output)
    return [dict(task=task,seed=seed,selection=_selection(output,task,seed))
        for task in TASKS for seed in p['seeds'] if (job_folder(output,task,seed)/'selection.json').exists()]


def fit(output, task, seed, device='cpu'):
    output=Path(output).resolve(); p=verify(output)
    if task not in TASKS or seed not in p['seeds']: raise ValueError('Unregistered task or seed')
    _admit(output,device); _stop(output); torch.set_num_threads(p.get('cpu_threads',4))
    folder=job_folder(output,task,seed); folder.mkdir(parents=True,exist_ok=True)
    with locked(folder/'.job.lock',nonblocking=True):
        if not (folder/'selection.json').exists(): _fit_locked(output,task,seed,device,p,folder)
        return _selection(output,task,seed)


def _fit_locked(output, task, seed, device, protocol, folder):
    if (output / 'selection_lock.json').exists():
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


def evaluate(output, task, seed, device='cpu'):
    output=Path(output).resolve(); p=verify(output)
    if task not in TASKS or seed not in p['seeds']: raise ValueError('Unregistered task or seed')
    _admit(output,device); _stop(output); torch.set_num_threads(p.get('cpu_threads',4))
    verify_test_lock(output)
    folder=job_folder(output,task,seed)
    with locked(folder/'.job.lock',nonblocking=True):
        return _evaluate_locked(output,task,seed,device,p,folder)


def _evaluate_locked(output,task,seed,device,protocol,folder):
    verify_test_lock(output)
    selected=_selection(output,task,seed)
    receipt=folder/'test_complete.json'
    lock_sha=sha(output/'selection_lock.json')
    if receipt.exists():
        saved=read(receipt)
        if (saved.get('protocol_sha256')!=sha(output/'protocol.json') or saved.get('selection_lock_sha256')!=lock_sha):
            raise ValueError('Completed test identity changed')
        for name,digest in saved['artifacts'].items():
            if sha(name)!=digest: raise ValueError('Completed test artifact changed')
        return read(folder/'test_results.json')
    data=_load_split(output,'test')
    checkpoint=torch.load(folder/'selected_heads.pt',map_location='cpu',weights_only=True)
    if checkpoint['identity']['protocol_sha256']!=sha(output/'protocol.json'):
        raise ValueError('Selected checkpoint identity changed')
    heads=build_heads(data['hidden_size'],task,seed,protocol['settings'],device)
    for arm in ARMS: heads[arm].load_state_dict(checkpoint['states'][arm],strict=True)
    rows=_rows(data,task)
    if not len(rows): raise ValueError('No test examples for this task')
    oom=[]
    logits=predict(heads,data,rows,protocol['settings'],device,output,oom_events=oom,phase='test')
    original._save_frame(folder/'test_predictions.parquet',_frame(data,rows,logits))
    target=_labels(data,task)[rows]
    sources=data['meta'].source.to_numpy()[rows]
    decisions,arms={},{}
    for arm in ARMS:
        if task=='binary':
            prob=original._probabilities(logits[arm])[:,1]
            threshold=selected['arms'][arm]['threshold']
            decisions[arm]=(prob>=threshold).astype(int)
            arms[arm]=dict(validation_threshold=original.binary_metrics(target,prob,threshold),
                fixed_threshold_05=original.binary_metrics(target,prob,.5),
                by_source={str(s):original.binary_metrics(target[sources==s],prob[sources==s],threshold) for s in sorted(set(sources))})
        else:
            decisions[arm]=logits[arm].argmax(-1)
            arms[arm]=dict(native_known_harmful=original.multiclass_metrics(target+1,decisions[arm]+1,range(1,6)),
                by_source={str(s):original.multiclass_metrics(target[sources==s]+1,decisions[arm][sources==s]+1,range(1,6)) for s in sorted(set(sources))})
    pairs={}
    groups=data['meta'].group_id.to_numpy()[rows]
    for candidate,baseline in (('mean_pool_mlp','last_mlp'),('shared_attention_mlp','last_mlp'),('shared_attention_mlp','mean_pool_mlp')):
        pairs[f'{candidate}_minus_{baseline}']=dict(changes=original.paired_changes(target,decisions[baseline],decisions[candidate]),
            bootstrap=original.multiclass_bootstrap(target,decisions[candidate],decisions[baseline],groups,protocol['settings']['bootstrap_draws'],seed))
    result=dict(schema=SCHEMA,task=task,seed=seed,complete=True,protocol_sha256=sha(output/'protocol.json'),
        selection_lock_sha256=lock_sha,selection_sha256=sha(folder/'selection.json'),
        feature_manifest_sha256=data['identity']['manifest_sha256'],rows=len(rows),arms=arms,
        paired_comparisons=pairs,inference_oom_events=oom)
    write(folder/'test_results.json',result)
    artifacts={str((folder/name).resolve()):sha(folder/name) for name in ('test_predictions.parquet','test_results.json')}
    write(receipt,dict(complete=True,protocol_sha256=sha(output/'protocol.json'),selection_lock_sha256=lock_sha,artifacts=artifacts))
    write(folder/'progress.json',dict(state='test_complete',task=task,seed=seed,updated_unix=time.time()))
    return result


def progress(output=DEFAULT_OUTPUT):
    output=Path(output).resolve(); p=read(output/'protocol.json')
    selected,tested,pending=[],[],[]
    for task in TASKS:
        for seed in p['seeds']:
            folder=job_folder(output,task,seed)
            job=dict(task=task,seed=seed)
            if (folder/'selection.json').exists():
                selected.append(job)
                if (folder/'test_complete.json').exists(): tested.append(job)
            else: pending.append(dict(job,phase='fit'))
    total=len(TASKS)*len(p['seeds'])
    if len(selected)==total:
        pending=[dict(job,phase='evaluate') for job in selected if job not in tested]
    return dict(total_jobs=total,selected_jobs=len(selected),completed_jobs=len(tested),
        total_heads=total*len(ARMS),selected_heads=len(selected)*len(ARMS),test_complete_heads=len(tested)*len(ARMS),
        pending_jobs=pending,test_barrier_locked=(output/'selection_lock.json').exists())


def summarize(output=DEFAULT_OUTPUT):
    output=Path(output).resolve(); p=verify(output); status=progress(output)
    if status['completed_jobs']!=status['total_jobs']: raise RuntimeError('All tests must complete before aggregation')
    verify_test_lock(output)
    aggregate,per_seed=[],{}
    for task in TASKS:
        per_seed[task]={}
        for seed in p['seeds']:
            folder=job_folder(output,task,seed); receipt=read(folder/'test_complete.json')
            if (receipt.get('protocol_sha256')!=sha(output/'protocol.json')
                    or receipt.get('selection_lock_sha256')!=sha(output/'selection_lock.json')):
                raise ValueError('Test receipt identity changed')
            for path,digest in receipt['artifacts'].items():
                if sha(path)!=digest: raise ValueError('Test payload changed')
            per_seed[task][str(seed)]=read(folder/'test_results.json')
        for arm in ARMS:
            entries=[per_seed[task][str(seed)]['arms'][arm]['validation_threshold' if task=='binary' else 'native_known_harmful'] for seed in p['seeds']]
            keys=('accuracy','f1','recall','fpr') if task=='binary' else ('accuracy','macro_f1')
            metrics={key:dict(mean=float(np.mean([e[key] for e in entries])),
                sample_std=float(np.std([e[key] for e in entries],ddof=1)) if len(entries)>1 else None,
                values=[e[key] for e in entries]) for key in keys}
            aggregate.append(dict(task=task,arm=arm,seeds=len(entries),metrics=metrics))
    result=dict(schema=SCHEMA,complete=True,protocol_sha256=sha(output/'protocol.json'),
        model_path=p['model_path'],seeds=p['seeds'],aggregate=aggregate,per_seed=per_seed,
        notes=['Frozen Qwen3Guard backbone and newly fitted ordinary heads; no LoRA or class-specific queries.',
            'Full official single-user template; token pooling uses original user-content positions only.',
            'Five-class labels learned directly from the same native training labels; no native taxonomy remapping.',
            'All checkpoint and threshold selections locked before fresh test feature extraction.',
            'Follow-up proposed after earlier test inspection; the test set is not claimed to be never seen.'])
    write(output/'results.json',result)
    lines=['# Qwen3Guard 冻结骨干监督分类对照','','| 任务 | 普通分类头 | Accuracy 均值±SD / % |','|---|---|---:|']
    for item in aggregate:
        m=item['metrics']['accuracy']; sd='N/A' if m['sample_std'] is None else f"{100*m['sample_std']:.2f}"
        lines.append(f"| {item['task']} | {item['arm']} | {100*m['mean']:.2f} ± {sd} |")
    lines+=['','二分类阈值只在验证集确定，同时保留固定 0.5 阈值成绩。五分类只使用同一原生已标注有害测试子集。',
        '原生生成式二分类与本次监督分类头对照分开报告。共享注意力为独立全局 query，未加入类别查询或多层融合。','']
    (output/'README.md').write_text('\n'.join(lines))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('fit','evaluate','status','lock','summarize'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--task',choices=TASKS); parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    args=parser.parse_args(argv)
    if args.action in ('fit','evaluate'):
        if args.task is None or args.seed is None: parser.error('task and seed required')
        result=(fit if args.action=='fit' else evaluate)(args.output,args.task,args.seed,args.device)
    elif args.action=='status': result=progress(args.output)
    elif args.action=='lock': result=lock_selections(args.output)
    else: result=summarize(args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
