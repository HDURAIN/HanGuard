"""Protocol invariants independent of a GPU or the downloaded backbone."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.hanguard import repaired_study as study


def test_dynamic_microbatches_keep_every_long_input_and_only_power_of_two_shapes():
    lengths=np.array([1,7,32,65,300,321,903,1600,3202,9000]*17)
    rng=np.random.default_rng(9);order=rng.permutation(len(lengths))
    groups=study.microbatches(order,lengths,8192,64,32)
    np.testing.assert_array_equal(np.concatenate(groups),order)
    for group in groups:
        assert len(group) in [1,2,4,8,16,32,64]
        padded=int(np.ceil(lengths[group].max()/32)*32)
        assert len(group)==1 or padded*len(group)<=8192
    assert sum((lengths[group]==9000).sum() for group in groups)==17


@pytest.mark.parametrize('n',[1,7,128,129,62155])
def test_paired_order_is_complete_reproducible_and_seed_sensitive(n):
    lengths=np.random.default_rng(19).integers(1,3203,size=n)
    first=study.batch_order(lengths,42)
    np.testing.assert_array_equal(first,study.batch_order(lengths,42))
    np.testing.assert_array_equal(np.sort(first),np.arange(n))
    if n>2048: assert not np.array_equal(first,study.batch_order(lengths,43))


def test_accuracy_threshold_matches_exhaustive_with_ties_and_single_class():
    rng=np.random.default_rng(491)
    for n in [1,2,17,100]:
        for _ in range(20):
            y=rng.integers(2,size=n);p=np.round(rng.uniform(size=n),1)
            threshold=study.accuracy_threshold(y,p)
            candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]
            correct=np.array([((p>=t)==y).sum() for t in candidates])
            tied=candidates[correct==correct.max()]
            assert threshold==tied[np.argmin(abs(tied-.5))]
    with pytest.raises(ValueError): study.accuracy_threshold([],[])
    with pytest.raises(ValueError): study.accuracy_threshold([0],[np.nan])


def test_test_barrier_must_be_explicitly_locked(tmp_path):
    with pytest.raises(RuntimeError): study.require_test_barrier(tmp_path)
    path=tmp_path/'test_barrier.json'
    for value in [False,'true',1,None]:
        path.write_text(json.dumps({'locked':value}))
        with pytest.raises(RuntimeError): study.require_test_barrier(tmp_path)
    path.write_text(json.dumps({'locked':True}))
    assert study.require_test_barrier(tmp_path)['locked'] is True


def test_resume_preserves_adam_moments_and_exact_next_update(tmp_path):
    torch.manual_seed(82)
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.03)
    x=torch.randn(7,3);target=torch.randn(7,2)
    def step(m,o):
        o.zero_grad(set_to_none=True);loss=(m(x)-target).square().mean();loss.backward();o.step()
    step(model,optimizer)
    original=copy.deepcopy(model)
    original_optimizer=torch.optim.AdamW(original.parameters(),lr=.03)
    original_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    study.write_resume(tmp_path/'resume.pt',torch.nn.Identity(),model,{'lora':False},optimizer,None,{'cursor':1})
    loaded=torch.load(tmp_path/'resume.pt',weights_only=False)
    model2=torch.nn.Linear(3,2);optimizer2=torch.optim.AdamW(model2.parameters(),lr=.03)
    study.load_weights(torch.nn.Identity(),model2,{'lora':False},loaded)
    optimizer2.load_state_dict(loaded['optimizer'])
    step(original,original_optimizer);step(model2,optimizer2)
    for a,b in zip(original.parameters(),model2.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert loaded['metadata']['cursor']==1


def test_full_batch_scl_gradient_cache_matches_retained_graph_with_dropout():
    pytest.importorskip('scripts.hanguard.repaired_heads')
    from scripts.hanguard.repaired_heads import supervised_contrastive_loss
    torch.manual_seed(173)
    network=torch.nn.Sequential(torch.nn.Linear(9,13),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(13,6))
    replay=copy.deepcopy(network);x=torch.randn(12,9);labels=torch.tensor([0,0,1,1,0,1,1,0,0,1,0,1])
    z=[]
    for i,batch in enumerate(x.split(4)):
        study.seed_micro(42,2,i);z.append(network(batch))
    loss=supervised_contrastive_loss(torch.cat(z),labels,temperature=.2);loss.backward()
    no_graph=[]
    with torch.no_grad():
        for i,batch in enumerate(x.split(4)):
            study.seed_micro(42,2,i);no_graph.append(replay(batch))
    actual,gradient=study.gradient_cache(torch.cat(no_graph),labels,.2)
    for i,batch in enumerate(x.split(4)):
        study.seed_micro(42,2,i);(replay(batch)*gradient[i*4:(i+1)*4]).sum().backward()
    torch.testing.assert_close(actual,loss.detach())
    for direct,cached in zip(network.parameters(),replay.parameters()):
        torch.testing.assert_close(direct.grad,cached.grad,rtol=3e-5,atol=3e-6)


def test_protocol_aliases_and_data_hashes_are_used(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'lr_warmup_ratio':.2,'prototype_warmup_samples':128}))
    (tmp_path/'arms.json').write_text('{}')
    protocol,_=study.json_protocol(tmp_path)
    assert protocol['warmup_fraction']==.2
    assert protocol['shield_warmup_samples']==128


def test_metrics_preserve_confusion_counts_and_empty_subgroups():
    result=study.metrics([0,0,1,1],[.1,.8,.2,.9],.5)
    assert [result[x] for x in ['tp','tn','fp','fn']]==[1,1,1,1]
    assert result['accuracy']==.5 and result['f1']==.5
    assert study.metrics([],[],.5)['auroc'] is None


def test_position_supervision_passes_stable_ids_in_loader_row_order():
    import pandas as pd
    from scripts.hanguard.repaired_heads import ExperimentHead,ShieldState
    torch.manual_seed(9)
    head=ExperimentHead(8,width=4,layers=(2,),dropout=0.)
    ids=['wildguard:original_a','jailbench:b','curated:c']
    state=ShieldState(ids,[3,2,4],[0,1,1],8,mode='dynamic',warmup_steps=0)
    data={'meta':pd.DataFrame({'base_id':ids})}
    indices=np.array([2,0])
    mask=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool)
    features={'layers':(torch.randn(2,4,8),),'mask':mask}
    output=head(features,return_tokens=True)
    loss,stats=study.position_objective(state,output,data,indices,0)
    assert torch.isfinite(loss)
    loss.backward();assert head.classifier.weight.grad.abs().sum()>0
    state.end_macro(0)
    assert state.diagnostics()['total_position_visits']>0


def test_secondary_thresholds_use_validation_only_and_respect_fpr_ties():
    y=np.array([0]*20+[1]*10)
    p=np.array([.1]*19+[.8]+[.8]*10)
    f1,fpr=study.secondary_thresholds(y,p)
    assert study.metrics(y,p,fpr)['fpr']<=.05
    candidates=np.unique(p)
    assert study.metrics(y,p,f1)['f1']==max(study.metrics(y,p,t)['f1'] for t in candidates)


def test_preflight_stresses_real_full_macros_and_both_padding_extremes():
    rng=np.random.default_rng(491)
    lengths=rng.integers(1,440,size=2300)
    lengths[1193]=3202
    protocol=dict(effective_batch=128,token_budget=24576,max_micro=64,pad_multiple=32)
    plan=study.preflight_batches(lengths,42,protocol,phase_index=0,steps=2,include_longest=True)
    order=study.batch_order(lengths,42*1009)
    macros=[order[start:start+128] for start in range(0,len(order),128)]
    workloads=[]
    for macro in macros:
        groups=study.microbatches(macro,lengths,24576,64,32)
        padded=[len(group)*int(np.ceil(lengths[group].max()/32)*32) for group in groups]
        workloads.append((max(padded),sum(padded)))
    reasons={reason for probe in plan for reason in probe['cases']}
    assert reasons=={'ordinary_consecutive_macro','maximum_microbatch_padded_tokens',
                     'maximum_macro_padded_tokens','contains_longest_training_text'}
    for probe in plan:
        np.testing.assert_array_equal(probe['indices'],macros[probe['macro_index']])
        assert probe['maximum_microbatch_padded_tokens']==workloads[probe['macro_index']][0]
        assert probe['total_padded_tokens']==workloads[probe['macro_index']][1]
        if 'contains_longest_training_text' in probe['cases']:
            assert 1193 in probe['indices'] and len(probe['indices'])==128
            assert probe['max_tokens']==3202
        if 'maximum_microbatch_padded_tokens' in probe['cases']:
            assert probe['maximum_microbatch_padded_tokens']==max(value[0] for value in workloads)
        if 'maximum_macro_padded_tokens' in probe['cases']:
            assert probe['total_padded_tokens']==max(value[1] for value in workloads)
    assert [probe['macro_index'] for probe in plan[:2]]==[0,1]


def test_preflight_stress_plan_keeps_short_final_macros_and_phase_order():
    lengths=np.array([9,7,1,2,12,64,1,3202,100])
    protocol=dict(effective_batch=8,token_budget=256,max_micro=8,pad_multiple=32)
    probes=study.preflight_batches(lengths,43,protocol,1,steps=10,include_longest=True)
    assert len(probes)==2
    np.testing.assert_array_equal(np.sort(np.concatenate([probe['indices'] for probe in probes])),np.arange(len(lengths)))
    assert sorted(len(probe['indices']) for probe in probes)==[1,8]
    with pytest.raises(ValueError):study.preflight_batches(lengths,43,protocol,1,steps=0)
