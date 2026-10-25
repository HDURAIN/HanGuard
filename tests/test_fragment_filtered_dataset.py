import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.hanguard import build_fragment_filtered_dataset as builder


def write_json(path,value):
    path.write_text(json.dumps(value))


def write_lines(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.fixture
def inputs(tmp_path):
    parent=tmp_path/'parent'
    parent.mkdir()
    for split in builder.SPLITS:
        frame=pd.DataFrame({
            'base_id':[f'{split}-{i}' for i in range(4)],
            'group_id':[f'group-{split}-{i}' for i in range(4)],
            'split':[split]*4,'source':['wildguard_zh']*4,
            'prompt':['Explain stars.','Make a harmful request.','What is it?','Hello.'],
            'prompt_harm_label':['unharmful','harmful','unharmful','unharmful'],
            'category_id':[0,1,0,0],'type_label_mask':[False,True,False,False],
            'prompt_tokens':[4,5,4,2]})
        frame.to_parquet(parent/f'{split}.parquet',index=False)
    write_json(parent/'split_policy.json',{'same_split':True})
    write_json(parent/'manifest.json',{'dataset_contract':builder.contract.CONTRACT,'dataset_version':'parent'})
    write_json(parent/'release_qa.json',{'release_status':'qa_complete_with_documented_limitations',
        'manifest_sha256':builder.sha(parent/'manifest.json'),
        'split_sha256':{split:builder.sha(parent/f'{split}.parquet') for split in builder.SPLITS}})
    candidates=[{'review_id':f'F{i+1:04d}','base_id':f'{split}-2','split':split,
                 'source':'wildguard_zh','prompt':'What is it?'} for i,split in enumerate(builder.SPLITS)]
    reviews=[{'review_id':row['review_id'],'base_id':row['base_id'],
              'decision':decision,'reason_zh':'Recorded semantic decision'}
             for row,decision in zip(candidates,['remove_fragment','uncertain','keep'])]
    packet=tmp_path/'packet.jsonl';decisions=tmp_path/'reviews.jsonl'
    write_lines(packet,candidates);write_lines(decisions,reviews)
    return parent,packet,decisions,candidates,reviews


def test_filter_preserves_every_retained_value_split_and_uncertain_row(inputs,tmp_path):
    parent,packet,reviews,_,_=inputs
    before={str(p):builder.sha(p) for p in parent.iterdir()}
    output=tmp_path/'derived'
    result=builder.build_dataset(parent,output,packet,[reviews])
    assert result['rows']==11 and result['removed_fragment_rows']==1
    for split in builder.SPLITS:
        original=pd.read_parquet(parent/f'{split}.parquet')
        expected=original[~original.base_id.eq('train-2')].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_parquet(output/f'{split}.parquet'),expected)
    assert 'validation-2' in set(pd.read_parquet(output/'validation.parquet').base_id)
    assert {str(p):builder.sha(p) for p in parent.iterdir()}==before
    assert builder.contract.release_contract(output)['dataset_version']=='derived'


@pytest.mark.parametrize('kind',['missing','duplicate','extra'])
def test_incomplete_or_duplicate_reviews_rejected(inputs,tmp_path,kind):
    _,packet,path,_,reviews=inputs
    if kind=='missing': reviews=reviews[:-1]
    if kind=='duplicate': reviews=reviews+[reviews[0]]
    if kind=='extra': reviews=reviews+[dict(reviews[0],base_id='not-in-packet',review_id='F9999')]
    write_lines(path,reviews)
    with pytest.raises(ValueError,match='Exactly one review'):
        builder.validated_reviews(packet,[path])


def test_changed_prompt_cannot_reuse_old_decision(inputs,tmp_path):
    parent,packet,reviews,candidates,_=inputs
    candidates[0]['prompt']='A different prompt'
    write_lines(packet,candidates)
    with pytest.raises(ValueError,match='prompt or split'):
        builder.build_dataset(parent,tmp_path/'derived',packet,[reviews])
    assert not (tmp_path/'derived').exists()
