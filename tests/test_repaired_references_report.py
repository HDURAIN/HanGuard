import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.hanguard.repaired_references import guard_label, length_batches, render_prompt, finish_saved_phase
import scripts.hanguard.repaired_references as reference_module
from scripts.hanguard.report_repaired_study import (
    AuditError, accuracy_threshold, dump, metrics, report, sha256, stratified,
)


def test_guard_parser_and_length_budget_never_drop_long_input():
    assert guard_label('Safety: Unsafe\nCategories: Violent') == 'unsafe'
    assert guard_label('Safety: Controversial\nCategories: None') == 'controversial'
    assert guard_label('Safety: Safe\nSafety: Unsafe') == 'invalid'
    assert guard_label('not unsafe') == 'invalid'
    lengths=[100,8000,200,300,2000]
    batches=list(length_batches(lengths,batch_size=3,max_batch_tokens=1000))
    assert sorted(sum(batches,[]))==list(range(len(lengths)))
    assert any(batch==[1] for batch in batches)
    assert all(len(batch)==1 or len(batch)*max(lengths[i] for i in batch)<=1000 for batch in batches)


def test_fixed_templates_retain_literal_text_and_use_official_flags():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return messages,kwargs
    original='前缀'+'长文本'*3000+'\n"suffix"\u2028尾部'
    messages,kwargs=render_prompt(Tokenizer(),original,'R01')
    assert json.loads(messages[1]['content'].split('\n',1)[1])==original
    assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
    messages,kwargs=render_prompt(Tokenizer(),original,'R02')
    assert messages==[{'role':'user','content':original}]
    assert kwargs==dict(tokenize=False)


def test_threshold_ties_and_rankless_guard_metrics():
    y=np.array([0,1,0,1]);p=np.array([.2,.4,.6,.8])
    threshold=accuracy_threshold(y,p)
    assert threshold==.4
    assert metrics(y,p,threshold)['accuracy']==.75
    assert metrics(y,[0,1,1,1],ranking=False)['auroc'] is None
    frame=pd.DataFrame(dict(label=y,probability=p,source=['a']*4,prompt_tokens=[20,500,1200,3202]))
    groups=stratified(frame,threshold)
    assert groups['length_groups']['gt_370']['rows']==3
    assert groups['length_groups']['gt_2048']['rows']==1


def fixture(tmp_path,ready=False):
    study,data=tmp_path/'study',tmp_path/'data'
    study.mkdir();data.mkdir()
    raw=pd.DataFrame(dict(base_id=['d','a','b','c'],source=['s1','s1','s2','s2'],
        prompt_harm_label=['unharmful','harmful','unharmful','harmful'],prompt_tokens=[20,500,1200,3202]))
    for split in ('train','validation','test'): raw.to_parquet(data/f'{split}.parquet')
    dump(study/'protocol.json',dict(seeds=[42],data_sha256={s:sha256(data/f'{s}.parquet') for s in ('train','validation','test')}))
    dump(study/'arms.json',{'E01':{},'E02':{},'E03':{},'E04':{},'E05':{}})
    if ready:
        frame=raw[['base_id','source','prompt_tokens']].assign(label=[0,1,0,1],probability=[.1,.7,.2,.8])
        paths=[]
        for arm in ('E01','E02','E03','E04','E05'):
            directory=study/'runs'/f'{arm}_s42';directory.mkdir(parents=True)
            frame.to_csv(directory/'validation.csv',index=False)
            frame.iloc[::-1].to_csv(directory/'test.csv',index=False)
            threshold=accuracy_threshold(frame.label,frame.probability)
            dump(directory/'selection.json',dict(arm=arm,seed=42,threshold=threshold,metrics=metrics(frame.label,frame.probability,threshold)))
            paths.append(directory/'selection.json')
        dump(study/'test_barrier.json',dict(locked=True,protocol_sha256=sha256(study/'protocol.json'),
            selection_sha256={str(p.relative_to(study)):sha256(p) for p in paths}))
    return study,data


def test_partial_report_is_explicit_and_contains_no_invented_results(tmp_path):
    study,data=fixture(tmp_path)
    result=report(study,data)
    assert result['completed_runs']==0 and result['selected_runs']==0
    assert not result['complete'] and result['expected_runs']==5
    assert result['aggregates']['E01']['metrics']['accuracy']['mean'] is None
    assert (study/'report.md').exists()


def test_report_recomputes_aligned_metrics_and_requires_frozen_barrier(tmp_path):
    study,data=fixture(tmp_path,True)
    result=report(study,data)
    assert result['completed_runs']==5
    assert not result['complete']  # External references are still pending.
    assert result['aggregates']['E01']['metrics']['accuracy']['mean']==1.
    assert result['acceptance']['lora_depth_order']['positive_means'] is False
    assert result['strata']['E01']['length_groups']['gt_370']['count']==1
    (study/'test_barrier.json').unlink()
    with pytest.raises(AuditError,match='before selection barrier'): report(study,data)


def test_report_rejects_changed_ids_labels_lengths_and_frozen_selection(tmp_path):
    study,data=fixture(tmp_path,True)
    path=study/'runs/E01_s42/test.csv'
    frame=pd.read_csv(path);frame.loc[0,'prompt_tokens']=99;frame.to_csv(path,index=False)
    with pytest.raises(AuditError,match='changed prompt_tokens'):report(study,data)
    frame.loc[0,'prompt_tokens']=3202;frame.to_csv(path,index=False)
    selection_path=study/'runs/E01_s42/selection.json'
    selection=json.loads(selection_path.read_text());selection['unused_note']='changed';dump(selection_path,selection)
    with pytest.raises(AuditError,match='Frozen selection changed'):report(study,data)


def test_report_rejects_test_selected_threshold(tmp_path):
    study,data=fixture(tmp_path,True)
    path=study/'runs/E01_s42/selection.json'
    selection=json.loads(path.read_text());selection['threshold']=.99;dump(path,selection)
    with pytest.raises(AuditError,match='non-validation threshold'):report(study,data)


def test_reference_resume_finishes_atomic_predictions_without_replacing_metrics(tmp_path):
    study,data=fixture(tmp_path)
    out=study/'references/R01';out.mkdir(parents=True)
    hashes={s:sha256(data/f'{s}.parquet') for s in ('train','validation','test')}
    model_hashes={'config.json':'model'}
    dump(out/'protocol.json',dict(data_hashes=hashes,model_hashes=model_hashes,code_sha256=sha256(reference_module.__file__)))
    raw=pd.read_parquet(data/'validation.parquet')
    frame=raw[['base_id','source','prompt_tokens']].assign(label=[0,1,0,1],probability=[.1,.7,.2,.8])
    frame.to_csv(out/'validation.csv',index=False)
    finish_saved_phase(out,data,'R01','validation',hashes,model_hashes)
    assert (out/'selection.json').exists()
    original=(out/'validation_metrics.json').read_bytes()
    finish_saved_phase(out,data,'R01','validation',hashes,model_hashes)
    assert (out/'validation_metrics.json').read_bytes()==original
    assert json.loads((out/'status.json').read_text())['state']=='validation_complete'


def core_fixture(tmp_path,ready=False):
    study,data=fixture(tmp_path)
    arms=['E03','E04','E13','E14','E15','E16']
    protocol=json.loads((study/'protocol.json').read_text())
    protocol.update(all_arms=arms,seeds=[42],references=['R02'])
    dump(study/'protocol.json',protocol)
    # A full available-arm catalog must not expand the six registered jobs.
    dump(study/'arms.json',{f'E{i:02d}':{} for i in range(1,21)})
    if ready:
        raw=pd.read_parquet(data/'test.parquet')
        frame=raw[['base_id','source','prompt_tokens']].assign(label=[0,1,0,1],probability=[.1,.7,.2,.8])
        paths=[]
        for arm in arms:
            directory=study/'runs'/f'{arm}_s42';directory.mkdir(parents=True)
            frame.to_csv(directory/'validation.csv',index=False)
            frame.to_csv(directory/'test.csv',index=False)
            threshold=accuracy_threshold(frame.label,frame.probability)
            dump(directory/'selection.json',dict(arm=arm,seed=42,threshold=threshold,metrics=metrics(frame.label,frame.probability,threshold)))
            paths.append(directory/'selection.json')
        directory=study/'references/R02';directory.mkdir(parents=True)
        frame.drop(columns='probability').assign(safety_label=['safe','unsafe','controversial','unsafe']).to_csv(directory/'test.csv',index=False)
        dump(directory/'selection.json',dict(reference='R02'))
        paths.append(directory/'selection.json')
        dump(study/'test_barrier.json',dict(locked=True,protocol_sha256=sha256(study/'protocol.json'),
            selection_sha256={str(p.relative_to(study)):sha256(p) for p in paths}))
    return study,data,arms


def test_core_partial_report_requires_only_six_jobs_and_guard(tmp_path):
    study,data,arms=core_fixture(tmp_path)
    result=report(study,data)
    assert result['expected_runs']==6 and result['completed_runs']==0 and not result['complete']
    assert result['arms']==arms and list(result['references'])==['R02']
    assert set(result['acceptance'])=={'lora_pseudolabel'}
    assert all({c['left'],c['right']}<=set(arms) for c in result['contrasts'])
    assert 'R01' not in (study/'report.md').read_text()


def test_core_completed_report_is_complete_without_deferred_methods_or_seeds(tmp_path):
    study,data,arms=core_fixture(tmp_path,True)
    result=report(study,data)
    assert result['complete'] and result['completed_runs']==6
    assert result['single_seed_exploration'] is True and result['seeds']==[42]
    assert result['acceptance']['lora_pseudolabel']['all_seed_directions_positive'] is None
    assert result['aggregates']['E03']['metrics']['accuracy']['std'] is None
    assert {('E14','E13'),('E16','E15'),('E13','E03'),('E15','E04')} <= {(c['left'],c['right']) for c in result['contrasts']}
    text=(study/'report.md').read_text()
    assert '全部完成' in text and '单种子探索' in text and '不能作为跨种子' in text
    assert 'R01' not in text and 'E01 ' not in text
