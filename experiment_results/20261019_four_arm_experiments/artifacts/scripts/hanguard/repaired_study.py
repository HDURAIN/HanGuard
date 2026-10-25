"""Fresh, online hanguard study on the translation-repaired Chinese splits.

No feature cache, old adapter, or old head is read. ``train`` never opens the test
split. A separate ``evaluate`` command requires the study's locked test barrier.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TARGET = r'model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn)\.(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'
EXPECTED_ROWS = {'train': 62155, 'validation': 7753, 'test': 7778}
DEFAULTS = dict(data_dir=str(ROOT/'data/three_source_translation_repaired'),
                model_path=str(ROOT/'models/Qwen3.5-4B'), warm_head_epochs=1,
                epochs=3, effective_batch=128, token_budget=8192, max_micro=64,
                pad_multiple=32, max_tokens=4096, head_lr=1e-4,
                warm_head_lr=1e-3, adapter_lr=2e-5, weight_decay=.01,
                warmup_fraction=.05, position_weight=.1, scl_weight=.01,
                scl_temperature=.1, prototype_diversity_weight=.01,
                prototype_init_per_class=1024, shield_warmup_samples=16000,
                lora_rank=8, lora_alpha=16, lora_dropout=.05,
                head_width=128, layers=[8,16,24,32], seeds=[42,43,44])


def dump(path: Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(8*1024*1024), b''): h.update(piece)
    return h.hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def cpu_state(module: nn.Module) -> dict:
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def json_protocol(study: Path) -> tuple[dict, dict]:
    registered = json.loads((study/'protocol.json').read_text())
    protocol = dict(DEFAULTS, **registered)
    protocol['warmup_fraction'] = registered.get('lr_warmup_ratio', protocol['warmup_fraction'])
    protocol['shield_warmup_samples'] = registered.get('prototype_warmup_samples', protocol['shield_warmup_samples'])
    arms = json.loads((study/'arms.json').read_text())
    assert protocol['effective_batch'] > 0 and protocol['epochs'] > 0
    assert protocol['warm_head_epochs'] >= 0
    if protocol.get('training_entrypoint') == 'hanguard_current_training':
        if (set(protocol.get('rows', {})) != set(EXPECTED_ROWS) or
                set(protocol.get('data_sha256', {})) != set(EXPECTED_ROWS) or
                any(type(value) is not int or value < 1 for value in protocol['rows'].values())):
            raise ValueError('Current training requires registered three-split rows and hashes')
        assert protocol['max_tokens'] > 0
    else:
        assert protocol['max_tokens'] >= 3202, 'The repaired release contains a 3202-token text'
    return protocol, arms


def accuracy_threshold(y, p) -> float:
    """Exactly maximize accuracy; among ties take threshold nearest to 0.5."""
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('Nonempty, aligned, finite validation predictions required')
    order = np.argsort(p, kind='stable'); ys = y[order]; ps = p[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1])+1]
    cumulative = np.r_[0, np.cumsum(1-2*ys)]
    correct = np.r_[ys.sum()+cumulative[starts], (1-ys).sum()]
    thresholds = np.r_[ps[starts], np.nextafter(ps[-1], np.inf)]
    best = np.flatnonzero(correct == correct.max())
    return float(thresholds[best[np.argmin(abs(thresholds[best]-.5))]])



def secondary_thresholds(y, p) -> tuple[float, float]:
    """Validation-only F1 optimum and a conservative <=5% empirical FPR."""
    from sklearn.metrics import precision_recall_curve
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.float64)
    precision,recall,thresholds=precision_recall_curve(y,p)
    values=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    tied=np.flatnonzero(values==values.max())
    f1=float(thresholds[tied[np.argmin(abs(thresholds[tied]-.5))]])
    negatives=np.sort(p[y==0])[::-1]
    if len(negatives)==0: raise ValueError('FPR selection needs validation negatives')
    allowed=int(math.floor(.05*len(negatives)))
    fpr=float(np.nextafter(negatives[allowed],np.inf))
    return f1,fpr


def metrics(y, p, threshold=.5) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold
    tp = int(((y == 1)&pred).sum()); tn = int(((y == 0)&~pred).sum())
    fp = int(((y == 0)&pred).sum()); fn = int(((y == 1)&~pred).sum())
    return dict(n=len(y), accuracy=(tp+tn)/max(1,len(y)),
                f1=2*tp/max(1,2*tp+fp+fn), recall=tp/max(1,tp+fn),
                precision=tp/max(1,tp+fp), fpr=fp/max(1,fp+tn),
                auroc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
                tp=tp,tn=tn,fp=fp,fn=fn,threshold=float(threshold))


def batch_order(lengths, seed: int, bucket_size=2048) -> np.ndarray:
    """Paired order across arms, with local sorting solely to reduce padding."""
    lengths = np.asarray(lengths)
    order = np.random.default_rng(seed).permutation(len(lengths))
    return np.concatenate([block[np.argsort(lengths[block],kind='stable')]
                           for block in np.array_split(order, math.ceil(len(order)/bucket_size))])


def microbatches(indices, lengths, token_budget: int, max_micro: int,
                 pad_multiple: int=32) -> list[np.ndarray]:
    """Partition without dropping/reordering text; limit padded token workload."""
    indices = np.asarray(indices, dtype=np.int64); lengths = np.asarray(lengths)
    if token_budget < 1 or max_micro < 1 or pad_multiple < 1:
        raise ValueError('Positive batching limits required')
    result = []; start = 0
    while start < len(indices):
        size = 1
        while size*2 <= min(max_micro, len(indices)-start):
            candidate = int(lengths[indices[start:start+size*2]].max())
            padded = math.ceil(candidate/pad_multiple)*pad_multiple
            if padded*(size*2) > token_budget: break
            size *= 2
        # Every microbatch uses a power-of-two size. Even an input longer than
        # the workload budget is retained whole as a singleton.
        result.append(indices[start:start+size]); start += size
    return result


def load_split(split: str, tokenizer, protocol: dict) -> dict:
    if split not in EXPECTED_ROWS: raise ValueError(split)
    path = Path(protocol['data_dir'])/f'{split}.parquet'
    expected_hashes = protocol.get('data_sha256', protocol.get('data_hashes', {}))
    expected_hash = expected_hashes.get(split, expected_hashes.get(f'{split}.parquet'))
    actual_hash = sha(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f'Dataset hash changed: {path}')
    frame = pd.read_parquet(path).sort_values('base_id').reset_index(drop=True)
    expected_rows = (protocol['rows'][split] if protocol.get('training_entrypoint') == 'hanguard_current_training'
                     else EXPECTED_ROWS[split])
    if len(frame) != expected_rows:
        raise ValueError(f'{split} contains {len(frame)} rows; expected {expected_rows}')
    if frame.base_id.duplicated().any(): raise ValueError('Duplicate source identities')
    if not frame.prompt_harm_label.isin(['harmful','unharmful']).all():
        raise ValueError(f'Unexpected labels: {frame.prompt_harm_label.unique()}')
    encoded = tokenizer(frame.prompt.tolist(),add_special_tokens=False,truncation=False)['input_ids']
    lengths = np.asarray([len(ids) for ids in encoded],dtype=np.int32)
    if lengths.min() < 1 or lengths.max() > protocol['max_tokens']:
        raise ValueError(f'Input length {lengths.min()}..{lengths.max()} exceeds registered budget; refusing truncation')
    meta = frame[['base_id','source']].copy()
    meta['label'] = frame.prompt_harm_label.eq('harmful').astype(int)
    meta['prompt_tokens'] = lengths
    return dict(encoded=encoded,lengths=lengths,meta=meta,
                y=torch.tensor(meta.label.to_numpy(),dtype=torch.float32),sha256=actual_hash)


def make_batch(data: dict, indices, tokenizer, protocol: dict, device='cuda'):
    batch = tokenizer.pad([{'input_ids':data['encoded'][int(i)],
                            'attention_mask':[1]*len(data['encoded'][int(i)])} for i in indices],
                          padding=True,pad_to_multiple_of=protocol['pad_multiple'],return_tensors='pt')
    expected = torch.tensor(data['lengths'][indices],dtype=torch.long)
    assert torch.equal(batch['attention_mask'].sum(1),expected), 'Tokenizer changed full input lengths'
    return batch.to(device)


def setup(spec: dict, seed: int, protocol: dict):
    from transformers import AutoTokenizer
    from transformers.utils.logging import disable_progress_bar
    from hanguard_model import load_base_model
    from scripts.hanguard.repaired_heads import ExperimentHead, TapOnline
    disable_progress_bar(); torch.set_num_threads(4)
    for filename,key in [('config.json','model_config_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if protocol.get(key) and sha(Path(protocol['model_path'])/filename)!=protocol[key]:
            raise ValueError(f'Base model artifact changed: {filename}')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_path'],local_files_only=True)
    tokenizer.pad_token = '<|im_end|>'; tokenizer.padding_side = 'right'
    model = load_base_model(protocol['model_path'],local_files_only=True,
                            device_map={'':0},attn_implementation='sdpa')
    model.requires_grad_(False)
    model.config.use_cache = False; model.config.text_config.use_cache = False
    if spec['lora']:
        from peft import LoraConfig, get_peft_model
        # Reset before constructing adapters so all arm pairs share fresh initialization.
        torch.manual_seed(seed)
        model = get_peft_model(model,LoraConfig(r=protocol['lora_rank'],
            lora_alpha=protocol['lora_alpha'],lora_dropout=protocol['lora_dropout'],
            target_modules=TARGET,bias='none'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.eval()
    unexpected = [name for name,p in model.named_parameters() if p.requires_grad and 'lora_' not in name]
    if unexpected: raise AssertionError(f'Non-LoRA backbone weights trainable: {unexpected[:5]}')
    hidden = model.config.text_config.hidden_size
    torch.manual_seed(seed)
    head = ExperimentHead(hidden,mode=spec['mode'],readout=spec.get('readout','mlp'),
                          width=protocol['head_width'],layers=tuple(protocol['layers'])).cuda()
    tap = TapOnline(model,head)
    return model,head,tap,tokenizer


def adapter_parameters(model) -> list:
    return [p for name,p in model.named_parameters() if 'lora_' in name]


def set_mode(model, head, spec: dict, training: bool, joint: bool=True) -> None:
    update_backbone = bool(spec['lora'] and joint)
    for parameter in adapter_parameters(model): parameter.requires_grad_(update_backbone)
    model.train(training and update_backbone); head.train(training)


def classify(tap, batch, return_tokens=False):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result = tap(batch,return_tokens=return_tokens)
    assert torch.isfinite(result['logits']).all(), 'Nonfinite classifier output'
    return result


def seed_micro(seed, step, offset):
    # Same batch/step seed regardless of optional branches; gradient-cache replay
    # uses the exact same seed for both its detached and differentiable passes.
    torch.manual_seed((seed*1000003+step*1009+offset) % (2**63-1))


@torch.no_grad()
def predict(model,head,tap,data,tokenizer,protocol,directory,phase,spec) -> np.ndarray:
    set_mode(model,head,spec,False)
    order = np.argsort(data['lengths'],kind='stable')
    groups = microbatches(order,data['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])
    logits = np.empty(len(order),dtype=np.float32); start = time.monotonic(); done = 0
    for number,ids in enumerate(groups):
        batch = make_batch(data,ids,tokenizer,protocol)
        result = classify(tap,batch)
        logits[ids] = result['logits'].float().cpu().numpy(); done += len(ids)
        if number % 25 == 0 or done == len(order):
            dump(directory/'status.json',dict(state=phase,rows=done,total=len(order),
                  seconds=time.monotonic()-start,pid=os.getpid(),updated=time.time()))
        del result,batch
    assert np.isfinite(logits).all()
    return logits


def sigmoid(logits):
    return torch.as_tensor(logits,dtype=torch.float64).sigmoid().numpy()


def predictions_csv(data,logits,path):
    frame = data['meta'].copy(); frame['probability'] = sigmoid(logits)
    frame.to_csv(path,index=False)


def checkpoint_state(model,head,spec) -> dict:
    result = dict(head=cpu_state(head))
    if spec['lora']:
        from peft import get_peft_model_state_dict
        result['adapter'] = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    return result


def load_weights(model,head,spec,state):
    head.load_state_dict(state['head'])
    if spec['lora']:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model,state['adapter'])


def save_best(model,head,spec,directory):
    state = checkpoint_state(model,head,spec)
    atomic_torch_save(state['head'],directory/'head.pt')
    if spec['lora']: atomic_torch_save(state['adapter'],directory/'adapter.pt')


def load_best(model,head,spec,directory):
    state = {'head':torch.load(directory/'head.pt',map_location='cpu',weights_only=True)}
    if spec['lora']: state['adapter'] = torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True)
    load_weights(model,head,spec,state)


def build_optimizer(model,head,spec,protocol,joint):
    set_mode(model,head,spec,True,joint)
    groups = [dict(params=[p for p in head.parameters() if p.requires_grad],
                   lr=protocol['head_lr'] if joint else protocol['warm_head_lr'],
                   initial_lr=protocol['head_lr'] if joint else protocol['warm_head_lr'])]
    if spec['lora'] and joint:
        groups.append(dict(params=adapter_parameters(model),lr=protocol['adapter_lr'],initial_lr=protocol['adapter_lr']))
    return torch.optim.AdamW(groups,weight_decay=protocol['weight_decay'])


def gradient_cache(z: torch.Tensor,y: torch.Tensor,temperature=.1):
    """Full effective-batch SCL loss and d(loss)/dz for deterministic replay."""
    from scripts.hanguard.repaired_heads import supervised_contrastive_loss
    leaf = z.detach().float().requires_grad_(True)
    loss = supervised_contrastive_loss(leaf,y,temperature=temperature)
    gradient, = torch.autograd.grad(loss,leaf)
    return loss.detach(),gradient.detach()


@torch.no_grad()
def initialize_prototypes(model,head,tap,train,tokenizer,protocol,spec,seed,directory):
    labels = train['meta'].label.to_numpy(); rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(labels==c),
                           min(protocol['prototype_init_per_class'],int((labels==c).sum())),replace=False) for c in [0,1]])
    chosen = chosen[np.argsort(train['lengths'][chosen],kind='stable')]
    groups = microbatches(chosen,train['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])
    set_mode(model,head,spec,False,False); representations = []
    for ids in groups:
        output = classify(tap,make_batch(train,ids,tokenizer,protocol))
        representations.append(output['representation'].float().cpu())
    head.initialize_prototypes(torch.cat(representations),train['y'][chosen],seed=seed)
    info = dict(rows=len(chosen),split='train',samples_per_class=protocol['prototype_init_per_class'],
                ids_sha256=hashlib.sha256('\n'.join(train['meta'].base_id.iloc[chosen]).encode()).hexdigest())
    dump(directory/'prototype_initialization.json',info)


def make_shield(spec,head,train,protocol,total_steps):
    if spec.get('objective','bce') not in ['fixed','dynamic']: return None
    from scripts.hanguard.repaired_heads import ShieldState
    warmup = math.ceil(protocol['shield_warmup_samples']/protocol['effective_batch'])
    return ShieldState(sample_ids=train['meta'].base_id.tolist(),lengths=train['lengths'],
                       labels=train['y'].numpy(),hidden_size=head.hidden_size,
                       layer_count=len(head.layers_for_capture),mode=spec['objective'],device='cuda',
                       warmup_steps=warmup,ramp_steps=max(1,total_steps-warmup))



def position_objective(shield,output,data,indices,step):
    """Translate loader row positions to persistent source IDs at the boundary."""
    sample_ids=data['meta'].base_id.iloc[indices].tolist()
    return shield.loss(output['token_logits'],output['prototype_features'],sample_ids,
                       output['mask'],step,confidence=output.get('teacher_probabilities'))


def objective_gradient_ratio(main_loss,auxiliary_loss,head,scale):
    params=[p for p in head.parameters() if p.requires_grad]
    primary=torch.autograd.grad(main_loss,params,retain_graph=True,allow_unused=True)
    auxiliary=torch.autograd.grad(auxiliary_loss,params,retain_graph=True,allow_unused=True)
    def norm(values):
        pieces=[g.float().square().sum() for g in values if g is not None]
        return torch.stack(pieces).sum().sqrt() if pieces else main_loss.new_zeros(())
    main_norm=norm(primary);aux_norm=norm(auxiliary)
    return dict(sentence_head_grad_norm=float(main_norm),position_head_grad_norm=float(aux_norm),
                weighted_position_to_sentence_grad_ratio=float(scale*aux_norm/main_norm.clamp_min(1e-12)))


def macro_step(model,head,tap,data,tokenizer,protocol,spec,ids,optimizer,
               seed,step,joint,shield=None,audit_grad=False) -> dict:
    set_mode(model,head,spec,True,joint); optimizer.zero_grad(set_to_none=True)
    groups = microbatches(ids,data['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])
    objective = spec.get('objective','bce') if joint else 'bce'
    cached_gradient = None; contrastive = 0.; pair_counts = None
    if objective == 'scl':
        pieces = []
        with torch.no_grad():
            for offset,small in enumerate(groups):
                seed_micro(seed,step,offset)
                output = classify(tap,make_batch(data,small,tokenizer,protocol))
                pieces.append(output['representation'].float().detach())
        all_z = torch.cat(pieces); all_y = data['y'][ids].cuda()
        contrastive_value,cached_gradient = gradient_cache(all_z,all_y,protocol['scl_temperature'])
        contrastive = float(contrastive_value)
        positives = ((all_y[:,None]==all_y[None,:]).sum(1)-1)
        pair_counts = dict(batch=len(ids),positive_pairs=int(positives.sum()),
                           negative_pairs=int((all_y[:,None]!=all_y[None,:]).sum()))
        del pieces,all_z,all_y,output
    totals = dict(bce=0.,position_loss=0.,prototype_regularizer=0.,scl_loss=contrastive)
    auxiliary_stats = []; offset_rows = 0; actual_tokens = 0; padded_tokens = 0
    for offset,small in enumerate(groups):
        seed_micro(seed,step,offset)
        batch = make_batch(data,small,tokenizer,protocol)
        output = classify(tap,batch,return_tokens=objective in ['fixed','dynamic'])
        y = data['y'][small].cuda(); weight = len(small)/len(ids)
        ce = F.binary_cross_entropy_with_logits(output['logits'].float(),y)
        loss = ce*weight; totals['bce'] += float(ce.detach())*weight
        if objective in ['fixed','dynamic']:
            assert shield is not None
            position_loss,stats = position_objective(shield,output,data,small,step)
            if audit_grad and offset==0:
                totals['objective_gradient_audit']=objective_gradient_ratio(ce,position_loss,head,protocol['position_weight'])
            loss = loss+protocol['position_weight']*position_loss*weight
            totals['position_loss'] += float(position_loss.detach())*weight
            auxiliary_stats.append(stats)
        if spec.get('readout','mlp') == 'prototype':
            regularizer = head.prototype_regularization()
            loss = loss+protocol['prototype_diversity_weight']*regularizer*weight
            totals['prototype_regularizer'] += float(regularizer.detach())*weight
        if cached_gradient is not None:
            # The cache gradient already includes 1 / effective-batch normalization.
            grad = cached_gradient[offset_rows:offset_rows+len(small)]
            loss = loss+protocol['scl_weight']*(output['representation'].float()*grad).sum()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
        loss.backward(); offset_rows += len(small); actual_tokens += int(batch['attention_mask'].sum()); padded_tokens += batch['input_ids'].numel()
        del output,batch,loss,ce
    if shield is not None and objective in ['fixed','dynamic']:
        totals.update(shield.end_macro(step))
    params = [p for g in optimizer.param_groups for p in g['params']]
    grad_norm = float(nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    active_names = []
    if audit_grad:
        active_names = [name for name,p in model.named_parameters()
                        if p.grad is not None and bool(p.grad.detach().abs().sum()>0)]
        if active_names and not all('lora_' in name for name in active_names):
            raise AssertionError('A non-LoRA backbone weight received a gradient')
    if audit_grad:
        head_grad_parts=[parameter.grad.detach().float().square().sum() for parameter in head.parameters() if parameter.grad is not None]
        totals['head_grad_norm']=float(torch.stack(head_grad_parts).sum().sqrt()) if head_grad_parts else 0.
        totals['backbone_trainable_non_lora_tensors']=sum(parameter.requires_grad for name,parameter in model.named_parameters() if 'lora_' not in name)
    optimizer.step()
    totals.update(rows=len(ids),tokens=actual_tokens,padded_tokens=padded_tokens,microbatches=len(groups),
                  micro_sizes=[len(group) for group in groups],grad_norm=grad_norm,
                  adapter_gradient_audited=audit_grad,adapter_active_tensors=len(active_names) if audit_grad else None,
                  adapter_attention_grad=any('self_attn' in n for n in active_names),
                  adapter_linear_attention_grad=any('linear_attn' in n for n in active_names))
    if pair_counts is not None: totals['scl_pairs'] = pair_counts
    if auxiliary_stats:
        positions=sum(item['positions'] for item in auxiliary_stats)
        updated=sum(item['updated_layer_positions'] for item in auxiliary_stats)
        totals['position_summary']=dict(positions=positions,updated_layer_positions=updated,
            layer_target_variance=sum(item['layer_target_variance']*item['positions'] for item in auxiliary_stats)/max(1,positions),
            update_flip_rate=sum(item['update_flip_rate']*item['updated_layer_positions'] for item in auxiliary_stats)/max(1,updated),
            active=any(item['active'] for item in auxiliary_stats),gamma=auxiliary_stats[0]['gamma'],sigma=auxiliary_stats[0]['sigma'])
        totals['position_stats'] = auxiliary_stats
    return totals


def write_resume(path,model,head,spec,optimizer,shield,metadata):
    state = checkpoint_state(model,head,spec)
    state.update(optimizer=optimizer.state_dict(),metadata=metadata,
                 shield=shield.state_dict() if shield is not None else None)
    atomic_torch_save(state,path)


def train(args):
    study = Path(args.output); protocol,arms = json_protocol(study); spec = arms[args.arm]
    directory = study/'runs'/f'{args.arm}_s{args.seed}'; directory.mkdir(parents=True,exist_ok=True)
    if (directory/'selection.json').exists():
        prior=json.loads((directory/'selection.json').read_text())
        if prior.get('protocol_sha256')!=sha(study/'protocol.json') or prior.get('arms_sha256')!=sha(study/'arms.json'):
            raise ValueError('Completed run belongs to a different locked study protocol')
        return
    start = time.monotonic(); dump(directory/'status.json',dict(state='loading',pid=os.getpid(),updated=time.time()))
    model,head,tap,tokenizer = setup(spec,args.seed,protocol)
    train_data = load_split('train',tokenizer,protocol); val = load_split('validation',tokenizer,protocol)
    batch_size = protocol['effective_batch']; steps_epoch = math.ceil(len(train_data['encoded'])/batch_size)
    total_main_steps = steps_epoch*protocol['epochs']; warm_epochs = protocol['warm_head_epochs']
    total_phases = warm_epochs+protocol['epochs']; total_steps = total_phases*steps_epoch
    shield = make_shield(spec,head,train_data,protocol,total_main_steps)
    common = dict(arm=args.arm,seed=args.seed,protocol_sha256=sha(study/'protocol.json'),
                  arms_sha256=sha(study/'arms.json'),train_sha256=train_data['sha256'],validation_sha256=val['sha256'])
    dump(directory/'provenance.json',dict(**common,code_sha256=sha(Path(__file__)),
          heads_sha256=sha(ROOT/'scripts/hanguard/repaired_heads.py'),
          fresh_base=True,old_checkpoint_reuse=False,
          head_parameters=sum(p.numel() for p in head.parameters()),
          adapter_parameters=sum(p.numel() for p in adapter_parameters(model))))
    history=[]; best_loss=float('inf'); best_epoch=None; best_logits=None
    resume_phase=0; resume_cursor=0; resume_optimizer=None; accumulated_seconds=0.; accumulated_train_seconds=0.
    resume_path=directory/'resume.pt'
    if resume_path.exists():
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        meta=state['metadata']
        for key,value in common.items():
            if meta['identity'][key]!=value: raise ValueError(f'Resume identity changed: {key}')
        load_weights(model,head,spec,state)
        if shield is not None: shield.load_state_dict(state['shield'])
        resume_phase=meta['phase']; resume_cursor=meta['cursor']; resume_optimizer=state['optimizer']
        best_loss=meta['best_loss']; best_epoch=meta['best_epoch']; best_logits=meta['best_logits']
        history=meta['history']; accumulated_seconds=meta.get('seconds',0.)
        accumulated_train_seconds=meta.get('training_seconds',0.)
        del state
    elif spec.get('readout','mlp')=='prototype':
        initialize_prototypes(model,head,tap,train_data,tokenizer,protocol,spec,args.seed,directory)
    optimizer=None; previous_joint=None; train_elapsed=accumulated_train_seconds; completed_rows=resume_phase*len(train_data['encoded'])+min(len(train_data['encoded']),resume_cursor*batch_size)
    last_written=0.; order_hashes=[]
    for phase in range(resume_phase,total_phases):
        joint=phase>=warm_epochs; main_epoch=phase-warm_epochs if joint else None
        if optimizer is None or joint!=previous_joint:
            optimizer=build_optimizer(model,head,spec,protocol,joint)
            if resume_optimizer is not None:
                optimizer.load_state_dict(resume_optimizer);resume_optimizer=None
        previous_joint=joint
        order=batch_order(train_data['lengths'],args.seed*1009+phase)
        order_hashes.append(dict(phase=phase,sha256=hashlib.sha256(order.tobytes()).hexdigest()))
        first=resume_cursor if phase==resume_phase else 0
        for cursor in range(first,steps_epoch):
            macro=order[cursor*batch_size:(cursor+1)*batch_size]
            main_step=(main_epoch*steps_epoch+cursor) if joint else cursor
            overall_step=phase*steps_epoch+cursor
            if joint:
                ramp=max(1,math.ceil(protocol['warmup_fraction']*total_main_steps))
                factor=min((main_step+1)/ramp,max(0.,(total_main_steps-main_step)/max(1,total_main_steps-ramp)))
                for group in optimizer.param_groups: group['lr']=group['initial_lr']*factor
            began_step=time.monotonic()
            info=macro_step(model,head,tap,train_data,tokenizer,protocol,spec,macro,optimizer,
                            args.seed,main_step if joint else overall_step+total_main_steps,joint,shield,
                            audit_grad=(cursor==0 or cursor%100==0))
            duration=time.monotonic()-began_step;train_elapsed+=duration;completed_rows+=len(macro)
            info.update(rows_per_second=len(macro)/max(duration,1e-9),
                        valid_tokens_per_second=info['tokens']/max(duration,1e-9),
                        padded_tokens_per_second=info['padded_tokens']/max(duration,1e-9))
            # Keep step histories compact; position diagnostics are aggregated separately.
            step_record=dict(phase=phase,phase_name='joint' if joint else 'head_warmup',
                epoch=main_epoch+1 if joint else phase+1,step=cursor+1,main_step=main_step+1 if joint else 0,
                seconds=duration,**{k:v for k,v in info.items() if k!='position_stats'})
            with (directory/'steps.jsonl').open('a') as stream: stream.write(json.dumps(step_record,ensure_ascii=False)+'\n')
            if time.monotonic()-last_written>10 or cursor+1==steps_epoch:
                state=dict(state='training' if joint else 'head_warmup',**common,
                           phase=phase,epoch=main_epoch+1 if joint else phase+1,
                           step=overall_step+1,total_steps=total_steps,phase_step=cursor+1,
                           phase_steps=steps_epoch,rows=completed_rows,total_rows=total_phases*len(train_data['encoded']),
                           elapsed_seconds=accumulated_seconds+time.monotonic()-start,
                           training_seconds=train_elapsed,rows_per_second=completed_rows/max(1e-9,train_elapsed),
                           last_step=step_record,pid=os.getpid(),updated=time.time())
                dump(directory/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True);last_written=time.monotonic()
            validation_point=(cursor+1==steps_epoch) or (joint and cursor+1==math.ceil(steps_epoch/2))
            if validation_point:
                validation_start=time.monotonic()
                logits=predict(model,head,tap,val,tokenizer,protocol,directory,'validation',spec)
                loss=float(F.binary_cross_entropy_with_logits(torch.tensor(logits),val['y']))
                coordinate=(main_epoch+(cursor+1)/steps_epoch) if joint else -(warm_epochs-phase-1)
                record=dict(phase=phase,phase_name='joint' if joint else 'head_warmup',
                            epoch=coordinate,validation_bce=loss,seconds=time.monotonic()-validation_start)
                history.append(record)
                if loss<best_loss:
                    best_loss=loss;best_epoch=record;best_logits=logits.copy()
                    save_best(model,head,spec,directory)
                    predictions_csv(val,best_logits,directory/'validation.csv')
                dump(directory/'validation_history.json',history)
                if shield is not None: dump(directory/'shield_diagnostics.json',shield.diagnostics())
                # A cursor at the end of an epoch resumes through the empty loop,
                # then advances without accidentally resetting the joint optimizer.
                metadata=dict(identity=common,phase=phase,cursor=cursor+1,best_loss=best_loss,
                    best_epoch=best_epoch,best_logits=best_logits,history=history,
                    seconds=accumulated_seconds+time.monotonic()-start,training_seconds=train_elapsed)
                write_resume(resume_path,model,head,spec,optimizer,shield,metadata)
        resume_cursor=0
    if best_logits is None: raise RuntimeError('No validation checkpoint was selected')
    probabilities=sigmoid(best_logits); labels=val['meta'].label.to_numpy()
    threshold=accuracy_threshold(labels,probabilities)
    f1_threshold,fpr_threshold=secondary_thresholds(labels,probabilities)
    weight_hashes={name:sha(directory/name) for name in ['head.pt','adapter.pt'] if (directory/name).exists()}
    selected=dict(**common,validation_bce=best_loss,best_epoch=best_epoch,threshold=threshold,
                  f1_threshold=f1_threshold,fpr_threshold=fpr_threshold,
                  validation_f1=metrics(labels,probabilities,f1_threshold),
                  validation_fpr_5pct=metrics(labels,probabilities,fpr_threshold),
                  threshold_rule='maximum validation accuracy; ties nearest 0.5',
                  metrics=metrics(labels,probabilities,threshold),fixed_05=metrics(labels,probabilities,.5),
                  checkpoint_hashes=weight_hashes,history=history,train_order_hashes=order_hashes,
                  completed_training_rows=total_phases*len(train_data['encoded']),
                  seconds=accumulated_seconds+time.monotonic()-start,training_seconds=train_elapsed,
                  fresh_base=True,head_warmup_epochs=warm_epochs,epochs=protocol['epochs'])
    if shield is not None: selected['shield']=shield.diagnostics()
    dump(directory/'selection.json',selected)
    dump(directory/'status.json',dict(state='selected',arm=args.arm,seed=args.seed,
          seconds=selected['seconds'],training_seconds=train_elapsed,pid=os.getpid(),updated=time.time()))
    tap.close()
    # Full optimizer/soft-label state is no longer needed after successful selection.
    resume_path.unlink(missing_ok=True)


def require_test_barrier(study: Path) -> dict:
    path=study/'test_barrier.json'
    if not path.exists(): raise RuntimeError('Test barrier missing; complete and freeze all validation selections first')
    barrier=json.loads(path.read_text())
    if barrier.get('locked') is not True: raise RuntimeError('Test barrier is not locked')
    return barrier


def evaluate(args):
    study=Path(args.output);barrier=require_test_barrier(study);protocol,arms=json_protocol(study);spec=arms[args.arm]
    directory=study/'runs'/f'{args.arm}_s{args.seed}'
    relative=str((directory/'selection.json').relative_to(study))
    if barrier.get('protocol_sha256')!=sha(study/'protocol.json'):
        raise ValueError('Protocol does not match locked test barrier')
    if barrier.get('selection_sha256',{}).get(relative)!=sha(directory/'selection.json'):
        raise ValueError(f'Selection does not match locked test barrier: {relative}')
    if (directory/'test_results.json').exists(): return
    selection=json.loads((directory/'selection.json').read_text())
    if selection['protocol_sha256']!=sha(study/'protocol.json'): raise ValueError('Protocol changed after selection')
    for name,digest in selection['checkpoint_hashes'].items():
        if sha(directory/name)!=digest: raise ValueError(f'Selected checkpoint changed: {name}')
    model,head,tap,tokenizer=setup(spec,args.seed,protocol);load_best(model,head,spec,directory)
    data=load_split('test',tokenizer,protocol)
    logits=predict(model,head,tap,data,tokenizer,protocol,directory,'test',spec)
    predictions_csv(data,logits,directory/'test.csv')
    y=data['meta'].label.to_numpy();p=sigmoid(logits);threshold=selection['threshold'];sources=data['meta'].source.to_numpy()
    by_source={str(source):metrics(y[sources==source],p[sources==source],threshold) for source in sorted(set(sources))}
    groups={}
    for name,mask in [('at_most_370',data['lengths']<=370),('over_370',data['lengths']>370),
                      ('over_1024',data['lengths']>1024)]:
        groups[name]=metrics(y[mask],p[mask],threshold)
    dump(directory/'test_results.json',dict(arm=args.arm,seed=args.seed,metrics=metrics(y,p,threshold),
          fixed_05=metrics(y,p,.5),validation_f1=metrics(y,p,selection['f1_threshold']),
          validation_fpr_5pct=metrics(y,p,selection['fpr_threshold']),
          by_source=by_source,length_groups=groups,test_sha256=data['sha256'],
          threshold=threshold,checkpoint_hashes=selection['checkpoint_hashes']))
    dump(directory/'status.json',dict(state='complete',arm=args.arm,seed=args.seed,pid=os.getpid(),updated=time.time()))
    tap.close()


def preflight_batches(lengths,seed,protocol,phase_index,steps=2,include_longest=True):
    """Stress actual epoch macros, including the largest padded allocations.

    A long singleton does not test warmup activation memory. Every probe here
    retains an actual, whole effective batch from the registered training order.
    """
    if steps<1: raise ValueError('At least one ordinary preflight step is required')
    lengths=np.asarray(lengths);batch_size=protocol['effective_batch']
    order=batch_order(lengths,seed*1009+phase_index)
    macros=[order[start:start+batch_size] for start in range(0,len(order),batch_size)]
    workloads=[]
    for ids in macros:
        groups=microbatches(ids,lengths,protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])
        padded=[len(group)*math.ceil(int(lengths[group].max())/protocol['pad_multiple'])*protocol['pad_multiple'] for group in groups]
        workloads.append((max(padded),sum(padded),int(lengths[ids].max())))
    selected={}
    def add(index,reason):
        selected.setdefault(int(index),[]).append(reason)
    for index in range(min(steps,len(macros))):add(index,'ordinary_consecutive_macro')
    heaviest_peak=max(range(len(macros)),key=lambda index:workloads[index])
    heaviest_total=max(range(len(macros)),key=lambda index:(workloads[index][1],workloads[index][0],workloads[index][2]))
    add(heaviest_peak,'maximum_microbatch_padded_tokens')
    add(heaviest_total,'maximum_macro_padded_tokens')
    if include_longest:
        longest=int(lengths.argmax());position=int(np.flatnonzero(order==longest)[0])
        add(position//batch_size,'contains_longest_training_text')
    return [dict(indices=macros[index],macro_index=index,cases=cases,
                 maximum_microbatch_padded_tokens=workloads[index][0],
                 total_padded_tokens=workloads[index][1],max_tokens=workloads[index][2])
            for index,cases in selected.items()]


def preflight(args):
    study=Path(args.output);protocol,arms=json_protocol(study);spec=arms[args.arm]
    directory=study/'preflight'/f'{args.arm}_s{args.seed}';directory.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();model,head,tap,tokenizer=setup(spec,args.seed,protocol)
    data=load_split('train',tokenizer,protocol)
    steps_epoch=math.ceil(len(data['encoded'])/protocol['effective_batch'])
    shield=make_shield(spec,head,data,protocol,steps_epoch*protocol['epochs'])
    if spec.get('readout','mlp')=='prototype':
        probe_protocol=dict(protocol,prototype_init_per_class=16)
        initialize_prototypes(model,head,tap,data,tokenizer,probe_protocol,spec,args.seed,directory)
    include_warmup=getattr(args,'include_warmup',False)
    cycles=getattr(args,'preflight_cycles',1)
    if cycles<1:raise ValueError('preflight_cycles must be positive')
    phase_types=[False,True] if include_warmup else [True]
    records=[];phase_runs=[];joint_counter=0
    for cycle in range(cycles):
        for joint in phase_types:
            phase='train' if joint else 'head_warmup'
            # Do not mistake gradients left by the preceding joint probe for
            # newly generated warmup gradients when testing phase transitions.
            model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
            optimizer=build_optimizer(model,head,spec,protocol,joint)
            phase_index=protocol['warm_head_epochs'] if joint else 0
            probes=preflight_batches(data['lengths'],args.seed,protocol,phase_index,args.steps,
                                     include_longest=args.include_longest or include_warmup)
            shield_visits_before=shield.total_visits if shield is not None else 0
            torch.cuda.reset_peak_memory_stats();phase_began=time.monotonic()
            phase_start=len(records)
            for number,probe in enumerate(probes):
                ids=probe['indices'];start=time.monotonic()
                step=joint_counter if joint else number+cycle*len(probes)
                if joint and spec.get('objective')=='dynamic' and joint_counter>0:
                    step=math.ceil(protocol['shield_warmup_samples']/protocol['effective_batch'])+joint_counter
                info=macro_step(model,head,tap,data,tokenizer,protocol,spec,ids,optimizer,
                                args.seed,step,joint,shield,audit_grad=True)
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                if joint:joint_counter+=1
                assert info['head_grad_norm']>0, 'The classifier head did not receive a gradient'
                assert info['backbone_trainable_non_lora_tensors']==0, 'Backbone base weights were unfrozen'
                if joint and spec['lora']:
                    assert info['adapter_attention_grad'] and info['adapter_linear_attention_grad'], 'LoRA gradient missing in attention family'
                else:
                    assert info['adapter_active_tensors']==0, 'Frozen warmup/backbone received a gradient'
                diagnostics=dict(getattr(tap,'last_forward_diagnostics',{}))
                if diagnostics:
                    flags=diagnostics.get('captured_requires_grad',[])
                    if joint and spec['lora']:assert any(flags), 'Joint features were detached from LoRA'
                    else:assert not any(flags), 'Frozen backbone retained an input-gradient activation graph'
                records.append(dict(step=len(records),cycle=cycle,phase=phase,
                    phase_step=number,first_step_compilation=len(records)==0,seconds=elapsed,
                    macro_index=probe['macro_index'],cases=probe['cases'],
                    maximum_microbatch_padded_tokens=probe['maximum_microbatch_padded_tokens'],
                    planned_padded_tokens=probe['total_padded_tokens'],
                    rows_per_second=len(ids)/max(elapsed,1e-9),valid_tokens_per_second=info['tokens']/max(elapsed,1e-9),
                    padded_tokens_per_second=info['padded_tokens']/max(elapsed,1e-9),max_tokens=probe['max_tokens'],
                    peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,
                    tap_diagnostics=diagnostics,**info))
                assert info['tokens']==int(data['lengths'][ids].sum()), 'Preflight input was truncated'
                assert info['padded_tokens']==probe['total_padded_tokens'], 'Preflight batching drifted from registered policy'
                dump(directory/'status.json',dict(state='preflight',arm=args.arm,phase=phase,
                     cycle=cycle,records=records,updated=time.time()))
            if not joint and shield is not None:
                assert shield.total_visits==shield_visits_before, 'Sentence-only head warmup updated pseudo labels'
            phase_records=records[phase_start:]
            phase_runs.append(dict(cycle=cycle,phase=phase,steps=len(phase_records),
                 rows=sum(record['rows'] for record in phase_records),
                 seconds=time.monotonic()-phase_began,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9))
            del optimizer
    ordinary=[r for r in records if r['phase']=='train' and r['rows']>=protocol['effective_batch']//2]
    throughput=sum(r['rows'] for r in ordinary)/max(1e-9,sum(r['seconds'] for r in ordinary))
    result=dict(passed=True,arm=args.arm,seed=args.seed,records=records,phase_runs=phase_runs,
                include_warmup=include_warmup,preflight_cycles=cycles,
                longest_tokens=max(record['max_tokens'] for record in records),
                peak_gpu_gb=max(record['peak_gpu_gb'] for record in records),rows_per_second=throughput,
                seconds=time.monotonic()-began,
                note='Actual full training macros, including maximal padding and the longest text; timings include compilation and stress batches, not an epoch average')
    if shield is not None:result['shield']=shield.diagnostics()
    dump(directory/'result.json',result);dump(study/'preflight'/f'{args.arm}_s{args.seed}.json',result)
    tap.close();print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','evaluate','preflight']);parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=2);parser.add_argument('--include-longest',action='store_true')
    parser.add_argument('--include-warmup',action='store_true',help='Test frozen head warmup before joint training, including longest and maximal-padding full macros')
    parser.add_argument('--preflight-cycles',type=int,default=1,help='Repeat warmup/joint transitions to detect retained graphs or stale gradients')
    args=parser.parse_args()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
        {'train':train,'evaluate':evaluate,'preflight':preflight}[args.action](args)
    except Exception as error:
        area='preflight' if args.action=='preflight' else 'runs'
        directory=Path(args.output)/area/f'{args.arm}_s{args.seed}'
        dump(directory/'status.json',dict(state='failed',action=args.action,arm=args.arm,seed=args.seed,
              error=repr(error),pid=os.getpid(),updated=time.time()))
        raise


if __name__=='__main__': main()
