"""Frozen-backbone, small-head multilabel study with text-bound BF16 caches.

Extraction can run before annotation finishes. Feature caches bind source IDs and
full prompt hashes, not provisional labels. Training binds the finalized labels;
test evaluation requires a locked barrier for every prespecified arm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hanguard.repaired_study import (dump,sha,atomic_torch_save,cpu_state,
    batch_order,microbatches,accuracy_threshold)

ARMS=('last_mlp','learned_queries','description_queries')
LABEL_IDS=('1','2','3','4','5')
DEFAULTS=dict(epochs=12,effective_batch=128,max_micro=64,token_budget=16384,
    pad_multiple=32,max_tokens=4096,head_lr=1e-3,weight_decay=.01,dropout=.1,
    head_width=128,cache_budget_gb=8.,cache_on_gpu=True,seeds=[42],all_arms=list(ARMS),
    parent_run=str(ROOT/'outputs/hanguard_repaired_core_20260928/runs/E04_s42'),
    description_file=str(ROOT/'scripts/hanguard/category_descriptions.json'))


def text_sha(text):return hashlib.sha256(str(text).encode('utf-8')).hexdigest()

def json_sha(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read_protocol(study):
    registered=json.loads((study/'protocol.json').read_text());p=dict(DEFAULTS,**registered)
    if [str(value) for value in p.get('label_ids',list(LABEL_IDS))]!=list(LABEL_IDS):raise ValueError('Label order must be 1,2,3,4,5')
    if any(arm not in ARMS for arm in p['all_arms']):raise ValueError('Unknown registered head arm')
    if p['epochs']<1 or p['effective_batch']<1:raise ValueError('Invalid training budget')
    p.setdefault('input_data_dir',p['data_dir'])
    return p


def read_descriptions(protocol):
    if 'label_descriptions' in protocol:
        value=protocol['label_descriptions'];source_hash=json_sha(value)
    else:
        path=Path(protocol['description_file']);value=json.loads(path.read_text());source_hash=sha(path)
    if isinstance(value,dict):
        value=value.get('classes',value.get('descriptions',value.get('labels',value)))
        if isinstance(value,dict):value=[value[key] for key in LABEL_IDS]
    if isinstance(value,list) and value and isinstance(value[0],dict):
        mapping={str(item.get('category_id',item.get('id'))):item for item in value}
        value=[(str(mapping[key]['name'])+'：' if 'name' in mapping[key] else '')+str(mapping[key].get('description',mapping[key].get('text',''))) for key in LABEL_IDS]
    if not isinstance(value,list) or len(value)!=5 or any(not isinstance(x,str) or not x.strip() for x in value):
        raise ValueError('Exactly five nonempty category descriptions are required')
    return value,source_hash


def text_frame(path):
    frame=pd.read_parquet(path)
    needed={'base_id','prompt','source'}
    if not needed<=set(frame):raise ValueError(f'Missing input columns {needed-set(frame)}')
    if frame.base_id.isna().any() or frame.base_id.duplicated().any():raise ValueError('Unique nonnull source IDs required')
    if not frame.prompt.map(lambda value:isinstance(value,str) and bool(value.strip())).all():raise ValueError('Full nonempty text required')
    return frame.sort_values('base_id').reset_index(drop=True)


def text_identity(frame):
    return json_sha(list(zip(frame.base_id.astype(str),frame.prompt.map(text_sha))))


def label_arrays(frame):
    """An unreported category remains unknown; it never becomes a negative."""
    if 'labels' not in frame:raise ValueError('Finalized labels[5] are required for fitting/evaluation')
    labels=np.asarray(frame.labels.tolist(),dtype=np.float32)
    if labels.shape!=(len(frame),5) or not np.isin(labels,[-1.,0.,1.]).all():
        raise ValueError('Expected exactly five labels in {-1,0,1} per sample')
    known=labels!=-1
    if 'label_mask' in frame:
        supplied=np.asarray(frame.label_mask.tolist())
        if supplied.shape!=known.shape or not np.isin(supplied,[False,True,0,1]).all():raise ValueError('Invalid label_mask shape or values')
        if not np.array_equal(supplied.astype(bool),known):raise ValueError('label_mask must exactly equal labels != -1')
    if not known.any():raise ValueError('This split contains no observed labels')
    return labels,known


def multilabel_metrics(labels,probabilities,thresholds,known=None,binary_gate=None):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=np.float64)
    known=(y!=-1) if known is None else np.asarray(known,dtype=bool)
    thresholds=np.broadcast_to(np.asarray(thresholds,dtype=float),(5,))
    if y.shape!=p.shape or known.shape!=y.shape or y.ndim!=2 or y.shape[1]!=5:raise ValueError('Aligned [N,5] arrays required')
    if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
    pred=p>=thresholds[None]
    if binary_gate is not None:
        gate=np.asarray(binary_gate,dtype=bool)
        if gate.shape!=(len(y),):raise ValueError('Binary gate must align with rows')
        pred &= gate[:,None]
    per=[]
    for col,key in enumerate(LABEL_IDS):
        mask=known[:,col];target=y[:,col]==1;output=pred[:,col]
        tp=int((mask&target&output).sum());tn=int((mask&~target&~output).sum())
        fp=int((mask&~target&output).sum());fn=int((mask&target&~output).sum())
        per.append(dict(label=key,known=int(mask.sum()),unknown=int((~mask).sum()),positives=tp+fn,
             negatives=tn+fp,tp=tp,tn=tn,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
             f1=2*tp/max(1,2*tp+fp+fn),threshold=float(thresholds[col]),
             support_flags=[flag for condition,flag in [(mask.sum()==0,'no_known_labels'),
                          (tp+fn==0,'no_positive_labels'),(tn+fp==0,'no_negative_labels')] if condition]))
    tp=sum(row['tp'] for row in per);fp=sum(row['fp'] for row in per);fn=sum(row['fn'] for row in per)
    fully=known.all(1);has_known=known.any(0)
    return dict(rows=len(y),known_elements=int(known.sum()),fully_known_rows=int(fully.sum()),
         macro_f1=float(np.mean([row['f1'] for row,ok in zip(per,has_known) if ok])) if has_known.any() else None,
         macro_f1_known_classes=float(np.mean([row['f1'] for row,ok in zip(per,has_known) if ok])) if has_known.any() else None,
         macro_f1_all5_zero_undefined=float(np.mean([row['f1'] for row in per])),macro_known_classes=int(has_known.sum()),
         micro_f1=2*tp/max(1,2*tp+fp+fn),micro_precision=tp/max(1,tp+fp),micro_recall=tp/max(1,tp+fn),
         exact_match_fully_known=float((pred[fully]==(y[fully]==1)).all(1).mean()) if fully.any() else None,
         hamming_accuracy_known=float((pred[known]==(y[known]==1)).mean()) if known.any() else None,
         per_class=per,binary_gate_applied=binary_gate is not None)


def select_thresholds(labels,probabilities,known=None):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=np.float64);known=y!=-1 if known is None else np.asarray(known,bool)
    result=[];flags=[]
    for col in range(5):
        observed=y[known[:,col],col].astype(int);values=p[known[:,col],col]
        if len(observed)==0 or len(np.unique(observed))<2:
            result.append(.5);flags.append('no_known_labels' if len(observed)==0 else 'no_positive_labels' if observed.sum()==0 else 'no_negative_labels');continue
        order=np.argsort(values,kind='stable');ys=observed[order];ps=values[order]
        starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1]
        pref=np.r_[0,np.cumsum(ys)]
        tp=ys.sum()-pref[starts];fp=len(ys)-starts-tp;fn=ys.sum()-tp
        f1=2*tp/np.maximum(1,2*tp+fp+fn)
        candidates=np.r_[ps[starts],np.nextafter(ps[-1],np.inf)];f1=np.r_[f1,0.]
        tied=np.flatnonzero(f1==f1.max());best=tied[np.argmin(abs(candidates[tied]-.5))]
        result.append(float(candidates[best]));flags.append(None)
    return result,flags


def parent_identity(protocol):
    run=Path(protocol['parent_run']);study=run.parents[1]
    selection=json.loads((run/'selection.json').read_text())
    for name,digest in selection['checkpoint_hashes'].items():
        if sha(run/name)!=digest:raise ValueError(f'Binary parent checkpoint changed: {name}')
    return dict(run=str(run.resolve()),selection_sha256=sha(run/'selection.json'),
        protocol_sha256=sha(study/'protocol.json'),checkpoint_hashes=selection['checkpoint_hashes'],
        binary_threshold=float(selection['threshold']))


def make_text_batch(tokenizer,encoded,indices,protocol):
    batch=tokenizer.pad([dict(input_ids=encoded[int(i)],attention_mask=[1]*len(encoded[int(i)])) for i in indices],
                        padding=True,pad_to_multiple_of=protocol['pad_multiple'],return_tensors='pt').to('cuda')
    expected=torch.tensor([len(encoded[int(i)]) for i in indices],device='cuda')
    if not torch.equal(batch.attention_mask.sum(1),expected):raise ValueError('Input tokens changed during padding')
    return batch


@torch.no_grad()
def extract(args):
    from scripts.hanguard import repaired_study as binary
    study=Path(args.output);protocol=read_protocol(study);cache=study/'feature_cache';cache.mkdir(parents=True,exist_ok=True)
    descriptions,description_hash=read_descriptions(protocol);parent=parent_identity(protocol)
    frames={split:text_frame(Path(protocol['input_data_dir'])/f'{split}.parquet') for split in ['train','validation','test']}
    common=dict(parent=parent,text_identities={split:text_identity(frame) for split,frame in frames.items()},
                description_sha256=description_hash,label_order=list(LABEL_IDS),layer=32,dtype='bfloat16_uint16_bits',
                input='full raw Chinese text; no chat template, no truncation',pad_multiple=protocol['pad_multiple'])
    fingerprint=json_sha(common)
    manifest_path=cache/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous.get('fingerprint')!=fingerprint:raise ValueError('Existing cache belongs to different texts/model/descriptions')
        if previous.get('complete'):return
    dump(manifest_path,dict(**common,fingerprint=fingerprint,complete=False))
    run=Path(protocol['parent_run']);parent_study=run.parents[1]
    parent_protocol,parent_arms=binary.json_protocol(parent_study)
    parent_selection=json.loads((run/'selection.json').read_text());spec=parent_arms[parent_selection['arm']]
    model,head,tap,tokenizer=binary.setup(spec,parent_selection['seed'],parent_protocol)
    binary.load_best(model,head,spec,run);model.eval().requires_grad_(False);head.eval().requires_grad_(False)
    hidden_size=model.config.text_config.hidden_size
    encoded={split:tokenizer(frame.prompt.tolist(),add_special_tokens=False,truncation=False)['input_ids'] for split,frame in frames.items()}
    lengths={split:np.asarray([len(x) for x in values],dtype=np.int64) for split,values in encoded.items()}
    if any(v.min()<1 or v.max()>protocol['max_tokens'] for v in lengths.values()):raise ValueError('Full input exceeds budget; refusing truncation')
    required=sum(int(value.sum())*hidden_size*2 for value in lengths.values())
    if required>protocol['cache_budget_gb']*1024**3:raise ValueError(f'BF16 feature cache needs {required/1024**3:.2f} GiB, above registered budget')
    existing=sum(path.stat().st_size for path in cache.glob('*_tokens.npy'))
    if shutil.disk_usage(cache).free+existing<required*1.05:raise ValueError('Insufficient space for bounded feature cache')
    captured={}
    def capture(module,inputs,output):captured['last']=(output[0] if isinstance(output,tuple) else output).detach()
    handle=tap.backbone.layers[31].register_forward_hook(capture)
    began=time.monotonic();split_info={}
    try:
        for split,frame in frames.items():
            offsets=np.r_[0,np.cumsum(lengths[split])]
            np.save(cache/f'{split}_offsets.npy',offsets)
            values=np.lib.format.open_memmap(cache/f'{split}_tokens.npy',mode='w+',dtype=np.uint16,shape=(int(offsets[-1]),hidden_size))
            probabilities=np.empty(len(frame),dtype=np.float64)
            order=np.argsort(lengths[split],kind='stable')
            groups=microbatches(order,lengths[split],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])
            done=0
            for number,indices in enumerate(groups):
                batch=make_text_batch(tokenizer,encoded[split],indices,protocol)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=tap(batch)
                raw=captured.pop('last');mask=batch.attention_mask.bool()
                if raw.requires_grad or output['logits'].requires_grad:raise AssertionError('Frozen extraction retained a gradient graph')
                probabilities[indices]=output['logits'].double().sigmoid().cpu().numpy()
                for row,index in enumerate(indices):
                    bits=raw[row,mask[row]].contiguous().to(torch.bfloat16).view(torch.uint16).cpu().numpy()
                    values[offsets[index]:offsets[index+1]]=bits
                done+=len(indices)
                if number%10==0 or done==len(frame):
                    dump(study/'extract_status.json',dict(state='extracting',split=split,rows=done,total=len(frame),
                         cache_bytes=required,seconds=time.monotonic()-began,pid=os.getpid(),updated=time.time()))
                del output,raw,batch
            values.flush();del values
            index=frame[['base_id','source']].copy();index['prompt_sha256']=frame.prompt.map(text_sha)
            index['prompt_tokens']=lengths[split];index['binary_probability']=probabilities
            if 'prompt_harm_label' in frame:index['original_binary_label']=frame.prompt_harm_label.eq('harmful').astype(int)
            index.to_parquet(cache/f'{split}_index.parquet',index=False)
            split_info[split]=dict(rows=len(frame),tokens=int(offsets[-1]),maximum_tokens=int(lengths[split].max()),
                text_identity=common['text_identities'][split],index_sha256=sha(cache/f'{split}_index.parquet'),
                offsets_sha256=sha(cache/f'{split}_offsets.npy'),tokens_sha256=sha(cache/f'{split}_tokens.npy'))
        ids=tokenizer(descriptions,add_special_tokens=False,truncation=False)['input_ids']
        if min(map(len,ids))<1 or max(map(len,ids))>protocol['max_tokens']:raise ValueError('Description exceeds full-input budget')
        batch=make_text_batch(tokenizer,ids,np.arange(5),protocol)
        with torch.autocast('cuda',dtype=torch.bfloat16):tap(batch)
        raw=captured.pop('last');last=batch.attention_mask.sum(1)-1
        vectors=raw[torch.arange(5,device='cuda'),last].float().cpu()
        if not torch.isfinite(vectors).all():raise ValueError('Nonfinite description embeddings')
        atomic_torch_save(vectors,cache/'descriptions.pt')
        dump(cache/'descriptions.json',dict(label_ids=list(LABEL_IDS),descriptions=descriptions,
             description_sha256=description_hash,embedding_sha256=sha(cache/'descriptions.pt')))
    finally:
        handle.remove();tap.close()
    if parent_identity(protocol)!=parent:raise ValueError('Binary checkpoint changed during extraction')
    dump(manifest_path,dict(**common,fingerprint=fingerprint,complete=True,splits=split_info,
         hidden_size=hidden_size,cache_bytes=required,description_embedding_sha256=sha(cache/'descriptions.pt'),
         extraction_code_sha256=sha(Path(__file__)),seconds=time.monotonic()-began))
    dump(study/'extract_status.json',dict(state='complete',seconds=time.monotonic()-began,cache_bytes=required,updated=time.time()))


def cache_manifest(study,protocol):
    cache=study/'feature_cache';manifest=json.loads((cache/'manifest.json').read_text())
    if manifest.get('complete') is not True:raise ValueError('Feature extraction has not completed')
    if parent_identity(protocol)!=manifest['parent']:raise ValueError('Binary parent changed after extraction')
    _,description_hash=read_descriptions(protocol)
    if description_hash!=manifest['description_sha256']:raise ValueError('Descriptions changed after extraction')
    if sha(cache/'descriptions.pt')!=manifest['description_embedding_sha256']:raise ValueError('Description embeddings changed')
    return manifest



def validated_label_manifest(protocol,split):
    path=Path(protocol['data_dir'])/'label_manifest.json'
    if not path.exists():raise ValueError('A reviewed finalized label manifest is required before head fitting')
    manifest=json.loads(path.read_text())
    if manifest.get('training_ready') is not True:raise ValueError('Label export did not pass readiness checks')
    _,expected_descriptions=read_descriptions(protocol)
    if manifest.get('descriptions_sha256')!=expected_descriptions:raise ValueError('Label descriptions differ from finalized annotation manifest')
    if protocol.get('annotation_policy_sha256') and manifest.get('policy_sha256')!=protocol['annotation_policy_sha256']:
        raise ValueError('Annotation policy differs from registered protocol')
    data_path=Path(protocol['data_dir'])/f'{split}.parquet'
    if manifest.get('splits',{}).get(split,{}).get('sha256')!=sha(data_path):
        raise ValueError(f'Finalized {split} labels changed since readiness audit')
    return manifest,sha(path)


def load_cached_split(study,protocol,manifest,split):
    labels_manifest,labels_manifest_sha256=validated_label_manifest(protocol,split)
    cache=study/'feature_cache';path=Path(protocol['data_dir'])/f'{split}.parquet';frame=text_frame(path)
    if labels_manifest['splits'][split]['rows']!=len(frame):raise ValueError('Final label manifest row count mismatch')
    if text_identity(frame)!=manifest['text_identities'][split]:raise ValueError(f'Final {split} source IDs or full prompts differ from extracted inputs')
    labels,known=label_arrays(frame)
    info=manifest['splits'][split]
    for name,key in [(f'{split}_index.parquet','index_sha256'),(f'{split}_offsets.npy','offsets_sha256'),(f'{split}_tokens.npy','tokens_sha256')]:
        if sha(cache/name)!=info[key]:raise ValueError(f'Feature cache changed: {name}')
    meta=pd.read_parquet(cache/f'{split}_index.parquet')
    if meta.base_id.tolist()!=frame.base_id.tolist():raise ValueError('Feature cache ordering differs from final labels')
    if meta.source.tolist()!=frame.source.tolist():raise ValueError('Feature cache source provenance differs from final labels')
    offsets=np.load(cache/f'{split}_offsets.npy',allow_pickle=False);bits=np.load(cache/f'{split}_tokens.npy',mmap_mode='r',allow_pickle=False)
    if bits.dtype!=np.uint16 or bits.shape!=(int(offsets[-1]),manifest['hidden_size']):raise ValueError('Invalid BF16 cache shape')
    if not np.array_equal(np.diff(offsets),meta.prompt_tokens.to_numpy()):raise ValueError('Cache token lengths mismatch')
    device='cuda'
    if protocol['cache_on_gpu']:
        needed=bits.size*2
        free,_=torch.cuda.mem_get_info()
        if needed>free*.8:raise ValueError('GPU lacks cache capacity; register CPU streaming explicitly rather than silently changing protocol')
        tokens=torch.empty(bits.shape,dtype=torch.bfloat16,device=device)
        for start in range(0,len(bits),8192):
            block=torch.from_numpy(np.array(bits[start:start+8192],copy=True)).view(torch.bfloat16)
            tokens[start:start+len(block)].copy_(block)
    else:tokens=bits
    return dict(tokens=tokens,offsets=offsets,device_offsets=torch.tensor(offsets,device=device),
         lengths=np.diff(offsets),meta=meta,labels=labels,known=known,
         y=torch.tensor(labels,device=device),mask=torch.tensor(known,device=device),
         dataset_sha256=sha(path),label_manifest_sha256=labels_manifest_sha256,text_identity=manifest['text_identities'][split])


def feature_batch(data,indices,protocol):
    indices=np.asarray(indices,dtype=np.int64)
    padded=math.ceil(int(data['lengths'][indices].max())/protocol['pad_multiple'])*protocol['pad_multiple']
    lengths=torch.tensor(data['lengths'][indices],device='cuda')
    mask=torch.arange(padded,device='cuda')[None,:]<lengths[:,None]
    if isinstance(data['tokens'],torch.Tensor):
        starts=data['device_offsets'][torch.tensor(indices,device='cuda')]
        positions=starts[:,None]+torch.arange(padded,device='cuda')[None,:]
        positions=positions.masked_fill(~mask,0)
        hidden=data['tokens'][positions]
    else:
        hidden=torch.zeros((len(indices),padded,data['tokens'].shape[-1]),dtype=torch.bfloat16,device='cuda')
        for row,index in enumerate(indices):
            begin,end=data['offsets'][index:index+2]
            value=torch.from_numpy(np.array(data['tokens'][begin:end],copy=True)).view(torch.bfloat16)
            hidden[row,:len(value)].copy_(value)
    if hidden.requires_grad:raise AssertionError('Cached backbone features must remain frozen')
    return hidden,mask


def setup_head(study,protocol,manifest,arm,seed):
    from scripts.hanguard.multilabel_heads import MultiLabelHead
    torch.manual_seed(seed);torch.set_num_threads(4)
    query_vectors=torch.load(study/'feature_cache/descriptions.pt',weights_only=True,map_location='cpu') if arm=='description_queries' else None
    return MultiLabelHead(manifest['hidden_size'],width=protocol['head_width'],num_labels=5,
          mode=arm,query_vectors=query_vectors,dropout=protocol['dropout'],query_seed=seed).cuda()


@torch.no_grad()
def prediction(head,data,protocol):
    head.eval();order=np.argsort(data['lengths'],kind='stable');logits=np.empty((len(order),5),dtype=np.float32)
    for indices in microbatches(order,data['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple']):
        hidden,mask=feature_batch(data,indices,protocol);output=head(hidden,mask)
        values=output['logits'].float()
        if not torch.isfinite(values).all():raise ValueError('Nonfinite type logits')
        logits[indices]=values.cpu().numpy()
    from scripts.hanguard.multilabel_heads import masked_binary_cross_entropy
    bce=float(masked_binary_cross_entropy(torch.tensor(logits),torch.tensor(data['labels']),known_mask=torch.tensor(data['known'])))
    return logits,bce


def save_predictions(data,logits,path):
    frame=data['meta'].copy();probability=torch.tensor(logits,dtype=torch.float64).sigmoid().numpy()
    for col,key in enumerate(LABEL_IDS):
        frame[f'y_{key}']=data['labels'][:,col].astype(int);frame[f'known_{key}']=data['known'][:,col]
        frame[f'p_{key}']=probability[:,col]
    frame.to_csv(path,index=False)
    return probability


def train(args):
    from scripts.hanguard.multilabel_heads import masked_binary_cross_entropy
    study=Path(args.output);protocol=read_protocol(study);manifest=cache_manifest(study,protocol)
    directory=study/'runs'/f'{args.arm}_s{args.seed}';directory.mkdir(parents=True,exist_ok=True)
    identity=dict(arm=args.arm,seed=args.seed,protocol_sha256=sha(study/'protocol.json'),cache_fingerprint=manifest['fingerprint'])
    if (directory/'selection.json').exists():
        old=json.loads((directory/'selection.json').read_text())
        if any(old.get(key)!=value for key,value in identity.items()):raise ValueError('Completed run has another protocol')
        return
    start=time.monotonic();dump(directory/'status.json',dict(state='loading_features',**identity,pid=os.getpid(),updated=time.time()))
    train_data=load_cached_split(study,protocol,manifest,'train');val=load_cached_split(study,protocol,manifest,'validation')
    identity.update(train_dataset_sha256=train_data['dataset_sha256'],validation_dataset_sha256=val['dataset_sha256'],
                    label_manifest_sha256=train_data['label_manifest_sha256'])
    if train_data['label_manifest_sha256']!=val['label_manifest_sha256']:raise ValueError('Label manifest changed while loading splits')
    head=setup_head(study,protocol,manifest,args.arm,args.seed)
    optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['head_lr'],weight_decay=protocol['weight_decay'])
    best=float('inf');history=[];best_epoch=None;best_logits=None;start_epoch=0
    resume=directory/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume labels/cache/protocol changed')
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];best_logits=state['best_logits'];history=state['history']
    steps_epoch=math.ceil(len(train_data['labels'])/protocol['effective_batch']);rows_seen=start_epoch*len(train_data['labels'])
    for epoch in range(start_epoch,protocol['epochs']):
        head.train();order=batch_order(train_data['lengths'],args.seed*1009+epoch);loss_sum=0.;observed_sum=0;epoch_began=time.monotonic()
        for cursor,start_row in enumerate(range(0,len(order),protocol['effective_batch'])):
            ids=order[start_row:start_row+protocol['effective_batch']]
            optimizer.zero_grad(set_to_none=True);total_known=int(train_data['known'][ids].sum())
            for number,small in enumerate(microbatches(ids,train_data['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple'])):
                torch.manual_seed(args.seed*1000003+epoch*10009+cursor*131+number)
                hidden,mask=feature_batch(train_data,small,protocol);output=head(hidden,mask)
                loss=masked_binary_cross_entropy(output['logits'],train_data['y'][small],known_mask=train_data['mask'][small],reduction='sum')
                if not torch.isfinite(loss):raise ValueError('Nonfinite masked training loss')
                (loss/max(1,total_known)).backward();loss_sum+=float(loss.detach())
            observed_sum+=total_known
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            if total_known:optimizer.step()
            rows_seen+=len(ids)
            if cursor%10==0 or cursor+1==steps_epoch:
                dump(directory/'status.json',dict(state='training',**identity,epoch=epoch+1,epochs=protocol['epochs'],
                    step=epoch*steps_epoch+cursor+1,total_steps=protocol['epochs']*steps_epoch,rows=rows_seen,
                    loss=loss_sum/max(1,observed_sum),seconds=time.monotonic()-start,pid=os.getpid(),updated=time.time()))
        logits,val_loss=prediction(head,val,protocol)
        probabilities=torch.tensor(logits,dtype=torch.float64).sigmoid().numpy()
        record=dict(epoch=epoch+1,train_masked_bce=loss_sum/max(1,observed_sum),validation_masked_bce=val_loss,
                    validation_fixed_05=multilabel_metrics(val['labels'],probabilities,.5,val['known']),seconds=time.monotonic()-epoch_began)
        history.append(record)
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1;best_logits=logits.copy()
            atomic_torch_save(cpu_state(head),directory/'head.pt');save_predictions(val,best_logits,directory/'validation.csv')
        dump(directory/'history.json',history)
        atomic_torch_save(dict(identity=identity,head=cpu_state(head),optimizer=optimizer.state_dict(),epoch=epoch+1,
                              best=best,best_epoch=best_epoch,best_logits=best_logits,history=history),resume)
    p=torch.tensor(best_logits,dtype=torch.float64).sigmoid().numpy();thresholds,flags=select_thresholds(val['labels'],p,val['known'])
    gate=val['meta'].binary_probability.to_numpy()>=manifest['parent']['binary_threshold']
    result=dict(**identity,best_epoch=best_epoch,validation_masked_bce=best,thresholds=thresholds,threshold_support_flags=flags,
         validation=multilabel_metrics(val['labels'],p,thresholds,val['known']),validation_fixed_05=multilabel_metrics(val['labels'],p,.5,val['known']),
         validation_gated=multilabel_metrics(val['labels'],p,thresholds,val['known'],gate),
         checkpoint_sha256=sha(directory/'head.pt'),head_parameters=sum(p.numel() for p in head.parameters()),
         binary_parent=manifest['parent'],seconds=time.monotonic()-start,history=history,
         limitation='Exploratory automatically supplemented multi-label annotations; not human gold labels',
         code_sha256=sha(Path(__file__)),heads_sha256=sha(ROOT/'scripts/hanguard/multilabel_heads.py'))
    dump(directory/'selection.json',result);resume.unlink(missing_ok=True)
    dump(directory/'status.json',dict(state='selected',**identity,seconds=result['seconds'],updated=time.time()))



def validate_test_barrier(study,protocol):
    barrier=json.loads((study/'test_barrier.json').read_text())
    if barrier.get('locked') is not True or barrier.get('protocol_sha256')!=sha(study/'protocol.json'):
        raise ValueError('A current locked test barrier is required')
    if barrier.get('test_dataset_sha256')!=sha(Path(protocol['data_dir'])/'test.parquet'):
        raise ValueError('Test labels differ from locked barrier')
    for arm in protocol['all_arms']:
        for seed in protocol['seeds']:
            path=study/'runs'/f'{arm}_s{seed}'/'selection.json'
            relative=str(path.relative_to(study))
            if not path.exists() or barrier.get('selection_sha256',{}).get(relative)!=sha(path):
                raise ValueError(f'All registered selections must be locked before test: {relative}')
    return barrier


def evaluate(args):
    study=Path(args.output);protocol=read_protocol(study);manifest=cache_manifest(study,protocol)
    barrier=validate_test_barrier(study,protocol);directory=study/'runs'/f'{args.arm}_s{args.seed}'
    selection_path=directory/'selection.json';relative=str(selection_path.relative_to(study))
    if barrier.get('locked') is not True or barrier.get('protocol_sha256')!=sha(study/'protocol.json'):raise ValueError('A current locked test barrier is required')
    if barrier.get('selection_sha256',{}).get(relative)!=sha(selection_path):raise ValueError('Selection changed since barrier')
    selection=json.loads(selection_path.read_text())
    if selection['checkpoint_sha256']!=sha(directory/'head.pt'):raise ValueError('Selected head checkpoint changed')
    if selection['protocol_sha256']!=sha(study/'protocol.json') or selection['cache_fingerprint']!=manifest['fingerprint']:raise ValueError('Selected study identity changed')
    if (directory/'test_results.json').exists():return
    data=load_cached_split(study,protocol,manifest,'test')
    if selection['label_manifest_sha256']!=data['label_manifest_sha256']:raise ValueError('Label manifest changed after validation selection')
    head=setup_head(study,protocol,manifest,args.arm,args.seed)
    head.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True))
    logits,bce=prediction(head,data,protocol);p=save_predictions(data,logits,directory/'test.csv')
    thresholds=selection['thresholds'];gate=data['meta'].binary_probability.to_numpy()>=manifest['parent']['binary_threshold']
    sources=data['meta'].source.to_numpy()
    result=dict(arm=args.arm,seed=args.seed,test_masked_bce=bce,test_dataset_sha256=data['dataset_sha256'],thresholds=thresholds,
        metrics=multilabel_metrics(data['labels'],p,thresholds,data['known']),fixed_05=multilabel_metrics(data['labels'],p,.5,data['known']),
        gated=multilabel_metrics(data['labels'],p,thresholds,data['known'],gate),
        gated_fixed_05=multilabel_metrics(data['labels'],p,.5,data['known'],gate),binary_threshold=manifest['parent']['binary_threshold'],
        by_source={str(source):multilabel_metrics(data['labels'][sources==source],p[sources==source],thresholds,data['known'][sources==source]) for source in sorted(set(sources))},
        binary_parent=manifest['parent'],checkpoint_sha256=selection['checkpoint_sha256'])
    dump(directory/'test_results.json',result);dump(directory/'status.json',dict(state='complete',arm=args.arm,seed=args.seed,updated=time.time()))


def preflight(args):
    from scripts.hanguard.multilabel_heads import masked_binary_cross_entropy
    study=Path(args.output);protocol=read_protocol(study);manifest=cache_manifest(study,protocol)
    data=load_cached_split(study,protocol,manifest,'train');head=setup_head(study,protocol,manifest,args.arm,args.seed)
    optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['head_lr']);head.train()
    longest=int(data['lengths'].argmax());ordinary=np.arange(min(128,len(data['labels'])))
    batches=[ordinary,np.asarray([longest])];records=[];torch.cuda.reset_peak_memory_stats()
    for ids in batches:
        began=time.monotonic();optimizer.zero_grad(set_to_none=True);known=int(data['known'][ids].sum());total=0.
        for small in microbatches(ids,data['lengths'],protocol['token_budget'],protocol['max_micro'],protocol['pad_multiple']):
            hidden,mask=feature_batch(data,small,protocol);out=head(hidden,mask)
            loss=masked_binary_cross_entropy(out['logits'],data['y'][small],known_mask=data['mask'][small],reduction='sum')
            (loss/max(1,known)).backward();total+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
        if known and norm<=0:raise AssertionError('No type-head gradient')
        optimizer.step();records.append(dict(rows=len(ids),maximum_tokens=int(data['lengths'][ids].max()),known_elements=known,loss=total/max(1,known),grad_norm=norm,seconds=time.monotonic()-began))
    result=dict(passed=True,arm=args.arm,seed=args.seed,records=records,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,cache_fingerprint=manifest['fingerprint'],binary_backbone_loaded=False)
    dump(study/'preflight'/f'{args.arm}_s{args.seed}.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['extract','train','evaluate','preflight'])
    parser.add_argument('--output',required=True);parser.add_argument('--arm',choices=ARMS,default='last_mlp');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    try:
        if not torch.cuda.is_available():raise RuntimeError('A root-assigned CUDA GPU is required')
        globals()[args.action](args)
    except Exception as exc:
        study=Path(args.output);path=study/'extract_status.json' if args.action=='extract' else study/'runs'/f'{args.arm}_s{args.seed}'/'status.json'
        dump(path,dict(state='failed',action=args.action,arm=args.arm,error=repr(exc),pid=os.getpid(),updated=time.time()))
        raise


if __name__=='__main__':main()
