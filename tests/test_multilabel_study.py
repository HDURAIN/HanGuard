"""Multilabel cache identity, partial-label metrics and optimization invariants."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.hanguard import multilabel_study as study


def test_partial_labels_are_not_silently_negative():
    frame=pd.DataFrame({'labels':[[1,0,-1,0,-1],[0,1,1,-1,0]],
                        'label_mask':[[True,True,False,True,False],[True,True,True,False,True]]})
    labels,known=study.label_arrays(frame)
    assert known.sum()==7
    bad=frame.copy();bad['label_mask']=[[True]*5,[True]*5]
    with pytest.raises(ValueError):study.label_arrays(bad)
    with pytest.raises(ValueError):study.label_arrays(pd.DataFrame({'labels':[[0,0,0,0]]}))
    with pytest.raises(ValueError):study.label_arrays(pd.DataFrame({'labels':[[0,0,.5,0,0]]}))
    with pytest.raises(ValueError):study.label_arrays(pd.DataFrame({'labels':[[-1]*5]}))


def test_text_cache_identity_is_independent_of_annotation_but_not_text():
    first=pd.DataFrame({'base_id':['a','b'],'prompt':['你好','完整中文内容'],'labels':[[-1]*5,[-1]*5]})
    annotated=first.copy();annotated['labels']=[[0]*5,[1,0,1,0,0]]
    assert study.text_identity(first)==study.text_identity(annotated)
    altered=annotated.copy();altered.loc[1,'prompt']='截短'
    assert study.text_identity(first)!=study.text_identity(altered)
    assert study.text_identity(first)!=study.text_identity(first.iloc[::-1])


def test_text_frame_keeps_full_prompts_and_sorts_identity(tmp_path):
    path=tmp_path/'train.parquet';long='完整文本。'*1000
    pd.DataFrame({'base_id':['b','a'],'prompt':[long,'短文'],'source':['x','y']}).to_parquet(path,index=False)
    frame=study.text_frame(path)
    assert frame.base_id.tolist()==['a','b'] and frame.prompt.iloc[1]==long
    bad=pd.concat([frame,frame.iloc[:1]],ignore_index=True);bad.to_parquet(path,index=False)
    with pytest.raises(ValueError):study.text_frame(path)


def test_metrics_ignore_unknown_elements_and_strict_accuracy_uses_complete_rows():
    labels=np.array([[1,0,1,0,0],[1,-1,0,-1,0],[-1]*5])
    probabilities=np.array([[.9,.1,.8,.2,.1],[.8,.9,.9,.9,.1],[1]*5])
    result=study.multilabel_metrics(labels,probabilities,.5)
    assert result['known_elements']==8
    assert result['fully_known_rows']==1 and result['exact_match_fully_known']==1.
    assert result['micro_f1']==pytest.approx(6/7)
    assert result['per_class'][1]['known']==1 and result['per_class'][1]['fp']==0
    assert result['per_class'][2]['fp']==1


def test_binary_gate_forces_no_types_even_if_type_threshold_is_zero():
    labels=np.zeros((2,5));p=np.ones((2,5))
    result=study.multilabel_metrics(labels,p,[0]*5,binary_gate=np.array([False,True]))
    assert [item['fp'] for item in result['per_class']]==[1]*5
    assert result['exact_match_fully_known']==.5


def test_f1_thresholds_match_exhaustive_known_only_search():
    rng=np.random.default_rng(839)
    labels=rng.choice([-1,0,1],size=(50,5),p=[.2,.45,.35]);p=np.round(rng.uniform(size=(50,5)),2)
    thresholds,flags=study.select_thresholds(labels,p)
    assert flags==[None]*5
    for col,chosen in enumerate(thresholds):
        mask=labels[:,col]!=-1;y=labels[mask,col];values=p[mask,col]
        candidates=np.r_[np.unique(values),np.nextafter(values.max(),np.inf)]
        def f1(t):
            pred=values>=t;tp=((y==1)&pred).sum();fp=((y==0)&pred).sum();fn=((y==1)&~pred).sum()
            return 2*tp/max(1,2*tp+fp+fn)
        scores=np.array([f1(t) for t in candidates]);ties=candidates[scores==scores.max()]
        assert chosen==ties[np.argmin(abs(ties-.5))]


def test_missing_class_polarity_keeps_fixed_threshold_and_records_limitations():
    labels=np.array([[0,1,-1,0,1],[0,1,-1,1,0]]);p=np.array([[.8,.1,.4,.2,.8],[.7,.2,.8,.9,.1]])
    thresholds,flags=study.select_thresholds(labels,p)
    assert thresholds[:3]==[.5,.5,.5]
    assert flags[:3]==['no_positive_labels','no_negative_labels','no_known_labels']
    result=study.multilabel_metrics(labels,p,thresholds)
    assert result['per_class'][2]['known']==0 and 'no_known_labels' in result['per_class'][2]['support_flags']


def test_description_schema_encodes_only_name_and_definition(tmp_path):
    path=tmp_path/'descriptions.json'
    document={'classes':[{'id':i,'name':f'名称{i}','description':f'定义{i}',
                          'include':['标注专用'],'examples':['不要编码示例']} for i in range(1,6)]}
    path.write_text(json.dumps(document,ensure_ascii=False))
    descriptions,source_hash=study.read_descriptions({'description_file':str(path)})
    assert descriptions==[f'名称{i}：定义{i}' for i in range(1,6)]
    assert source_hash==study.sha(path)


def test_bfloat16_uint16_cache_preserves_raw_features_exactly(tmp_path):
    torch.manual_seed(42);raw=torch.randn(31,17).to(torch.bfloat16)
    path=tmp_path/'raw.npy'
    output=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=tuple(raw.shape))
    output[:]=raw.view(torch.uint16).numpy();output.flush();del output
    stored=np.load(path,mmap_mode='r',allow_pickle=False)
    restored=torch.from_numpy(np.array(stored,copy=True)).view(torch.bfloat16)
    assert torch.equal(raw,restored)


def test_known_element_weighting_matches_full_batch_gradient():
    from scripts.hanguard.multilabel_heads import MultiLabelHead,masked_binary_cross_entropy
    torch.manual_seed(22)
    head=MultiLabelHead(8,width=6,mode='learned_queries',dropout=0.)
    replay=copy.deepcopy(head)
    hidden=torch.randn(7,5,8);mask=torch.ones(7,5,dtype=torch.bool)
    labels=torch.tensor([[1,0,-1,0,-1],[0,1,1,0,0],[-1,-1,1,0,-1],[0]*5,[1]*5,[-1]*5,[0,0,1,1,-1]],dtype=torch.float32)
    known=labels!=-1
    direct=masked_binary_cross_entropy(head(hidden,mask)['logits'],labels,known_mask=known)
    direct.backward()
    for start,end in [(0,2),(2,5),(5,7)]:
        loss=masked_binary_cross_entropy(replay(hidden[start:end],mask[start:end])['logits'],
              labels[start:end],known_mask=known[start:end],reduction='sum')/known.sum()
        loss.backward()
    for a,b in zip(head.parameters(),replay.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=3e-6)


def test_label_manifest_requires_ready_policy_and_exact_file_hash(tmp_path):
    descriptions=['定义'+str(i) for i in range(5)]
    frame=pd.DataFrame({'base_id':['a'],'source':['wildguard_zh'],'prompt':['完整文本'],'labels':[[1,0,-1,0,0]]})
    frame.to_parquet(tmp_path/'train.parquet',index=False)
    protocol={'data_dir':str(tmp_path),'label_descriptions':descriptions,'annotation_policy_sha256':'policy'}
    manifest={'training_ready':True,'descriptions_sha256':study.json_sha(descriptions),'policy_sha256':'policy',
              'splits':{'train':{'sha256':study.sha(tmp_path/'train.parquet'),'rows':1}}}
    path=tmp_path/'label_manifest.json';path.write_text(json.dumps(manifest))
    actual,digest=study.validated_label_manifest(protocol,'train')
    assert actual==manifest and digest==study.sha(path)
    manifest['training_ready']=False;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):study.validated_label_manifest(protocol,'train')
    manifest['training_ready']=True;manifest['descriptions_sha256']='changed';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):study.validated_label_manifest(protocol,'train')
    manifest['descriptions_sha256']=study.json_sha(descriptions);path.write_text(json.dumps(manifest))
    frame.loc[0,'prompt']='被改动';frame.to_parquet(tmp_path/'train.parquet',index=False)
    with pytest.raises(ValueError):study.validated_label_manifest(protocol,'train')


def test_macro_average_excludes_a_class_with_no_observed_labels():
    labels=np.array([[1,0,-1,0,1],[0,1,-1,1,0]])
    probabilities=np.array([[.9,.1,.9,.1,.9],[.1,.9,.1,.9,.1]])
    result=study.multilabel_metrics(labels,probabilities,.5)
    assert result['macro_known_classes']==4
    assert result['macro_f1']==1.
    assert result['macro_f1_all5_zero_undefined']==.8


def test_barrier_binds_every_arm_and_final_test_labels(tmp_path):
    study_dir=tmp_path/'study';data_dir=tmp_path/'data';study_dir.mkdir();data_dir.mkdir()
    protocol={'all_arms':list(study.ARMS),'seeds':[42],'data_dir':str(data_dir)}
    (study_dir/'protocol.json').write_text(json.dumps(protocol))
    (data_dir/'test.parquet').write_bytes(b'fixed finalized test labels')
    selections={}
    for arm in study.ARMS:
        path=study_dir/'runs'/f'{arm}_s42'/'selection.json';path.parent.mkdir(parents=True);path.write_text('{}')
        selections[str(path.relative_to(study_dir))]=study.sha(path)
    barrier={'locked':True,'protocol_sha256':study.sha(study_dir/'protocol.json'),
             'test_dataset_sha256':study.sha(data_dir/'test.parquet'),'selection_sha256':selections}
    path=study_dir/'test_barrier.json';path.write_text(json.dumps(barrier))
    study.validate_test_barrier(study_dir,protocol)
    missing=copy.deepcopy(barrier);missing['selection_sha256'].pop(next(iter(selections)));path.write_text(json.dumps(missing))
    with pytest.raises(ValueError):study.validate_test_barrier(study_dir,protocol)
    path.write_text(json.dumps(barrier));(data_dir/'test.parquet').write_bytes(b'changed after selection')
    with pytest.raises(ValueError):study.validate_test_barrier(study_dir,protocol)
