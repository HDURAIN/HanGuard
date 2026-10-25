"""CPU integration checks for frozen Guard supervised readouts and test barrier."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch
from scripts.hanguard import qwen3guard_supervised_training as study


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old)

    def test_last_head_matches_parent_and_does_not_depend_on_token_pool(self):
        from scripts.hanguard.clean_ablation_heads import CleanAblationHead
        head=study.OrdinaryHead(6,4,5,'last_mlp',dropout=0.,seed=44)
        parent=CleanAblationHead(6,4,5,'baseline',dropout=0.,seed=44)
        raw=torch.randn(3,6); all_layers=torch.randn(3,32,6); all_layers[:,31]=raw
        torch.testing.assert_close(head(raw,None,None),parent(all_layers))
        head(raw,None,None).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_pooling_padding_invariance_and_no_class_vectors(self):
        for arm in study.ARMS[1:]:
            with self.subTest(arm=arm):
                head=study.OrdinaryHead(6,4,5,arm,dropout=0.)
                head.eval(); tokens=torch.randn(2,3,6); last=torch.randn(2,6)
                valid=torch.ones(2,3,dtype=torch.bool)
                padded=torch.cat([tokens,torch.full((2,5,6),float('nan'))],1)
                mask=torch.cat([valid,torch.zeros(2,5,dtype=torch.bool)],1)
                torch.testing.assert_close(head(last,tokens,valid),head(last,padded,mask),atol=1e-6,rtol=1e-6)
                self.assertFalse(any('category' in n for n,_ in head.named_parameters()))
                with self.assertRaises(ValueError): head(last,tokens,torch.zeros_like(valid))


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old)

    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name); self.output=self.root/'out'; self.output.mkdir()
        self.data=self.root/'data'; self.data.mkdir()
        self.protocol=dict(output=str(self.output),parent_data_dir=str(self.data),model_path='/unused/frozen_model',
            hidden_size=6,tasks=list(study.TASKS),arms=list(study.ARMS),seeds=[42],cpu_threads=1,
            training=dict(width=4,dropout=.1,learning_rate=.001,weight_decay=.001,grad_clip=5.,
                effective_batch_size=5,max_micro_batch_size=4,token_budget=20,epoch_budgets=[2],bootstrap_draws=12),
            data_sha256={},sources={str(Path(study.__file__).resolve()):study.sha(study.__file__)})
        for split in study.SPLITS:
            harm=np.array([i%3!=0 for i in range(20)])
            self.frame=pd.DataFrame(dict(base_id=[f'{split}_{19-i:03d}' for i in range(20)],
                group_id=[f'{split}_g_{i//2}' for i in range(20)],
                source=['wildguard_zh' if i%4==0 else 'chinese_curated' if i%2 else 'jailbench' for i in range(20)],
                prompt_harm_label=np.where(harm,'harmful','unharmful'),
                category_id=np.where(harm,np.arange(20)%5+1,0),type_label_mask=harm))
            self.frame.to_parquet(self.data/f'{split}.parquet',index=False)
            self.protocol['data_sha256'][split]=study.sha(self.data/f'{split}.parquet')
        study.write(self.output/'protocol.json',self.protocol)
        study.write(self.output/'protocol_registration.json',{'protocol_sha256':study.sha(self.output/'protocol.json')})
        for split in study.SPLITS: self.make_cache(split)

    def make_cache(self,split):
        rng=np.random.default_rng({'train':11,'validation':12,'test':13}[split])
        frame=pd.read_parquet(self.data/f'{split}.parquet')
        folder=self.output/'features'/split; prepared=folder/'prepared'; prepared.mkdir(parents=True)
        lengths=np.array([1+i%4 for i in range(20)],np.int64)
        np.save(prepared/'content_lengths.npy',lengths,allow_pickle=False)
        frame.to_parquet(prepared/'metadata.parquet',index=False)
        shards=[dict(shard_index=i,row_start=a,row_end=min(a+7,20)) for i,a in enumerate(range(0,20,7))]
        manifest=dict(rows=20,hidden_size=6,shards=shards,raw_data_sha256=study.sha(self.data/f'{split}.parquet'),
            protocol_sha256=study.sha(self.output/'protocol.json'),
            files={f'prepared/{name}':study.sha(prepared/name) for name in ('content_lengths.npy','metadata.parquet')})
        study.write(folder/'manifest.json',manifest)
        for s in shards:
            a,b=s['row_start'],s['row_end']; d=folder/f"shard_{s['shard_index']:05d}"; d.mkdir()
            offsets=np.concatenate([[0],np.cumsum(lengths[a:b])]).astype(np.int64)
            np.save(d/'offsets.npy',offsets,allow_pickle=False)
            np.save(d/'token_features.npy',rng.normal(size=(int(offsets[-1]),6)).astype(np.float16),allow_pickle=False)
            np.save(d/'last.npy',rng.normal(size=(b-a,6)).astype(np.float16),allow_pickle=False)
            study.write(d/'metadata.json',dict(s,protocol_sha256=study.sha(self.output/'protocol.json'),
                manifest_sha256=study.sha(folder/'manifest.json'),base_ids=frame.base_id.iloc[a:b].tolist()))
            study.write(d/'receipt.json',dict(s,row_start=a,row_end=b,
                protocol_sha256=study.sha(self.output/'protocol.json'),manifest_sha256=study.sha(folder/'manifest.json'),
                files={name:study.sha(d/name) for name in ('offsets.npy','token_features.npy','last.npy','metadata.json')}))

    def test_complete_cache_fit_test_barrier_and_aggregation(self):
        raw=study._load_split(self.output,'train')
        self.assertEqual(raw['meta'].base_id.tolist(),pd.read_parquet(self.data/'train.parquet').base_id.tolist())
        self.assertEqual(len(study._rows(raw,'category')),10)
        with self.assertRaises(ValueError): study._load_split(self.output,'test')
        a=study.fit(self.output,'binary',42)
        self.assertFalse(a['formal_test_opened']); self.assertEqual(a['training_rows'],20)
        self.assertFalse((self.output/'selection_lock.json').exists())
        with self.assertRaises(ValueError): study.evaluate(self.output,'binary',42)
        self.assertIsNone(study.lock_selections(self.output))
        b=study.fit(self.output,'category',42)
        self.assertEqual(b['training_rows'],10)
        self.assertEqual(study.progress(self.output)['selected_heads'],6)
        self.assertFalse((self.output/'selection_lock.json').exists())
        lock=study.lock_selections(self.output); self.assertTrue(lock['complete'])
        self.assertEqual(len(study.selections(self.output)),2)
        for task in study.TASKS:
            result=study.evaluate(self.output,task,42)
            self.assertEqual(result['rows'],20 if task=='binary' else 10)
            again=study.evaluate(self.output,task,42); self.assertEqual(result,again)
            frame=pd.read_parquet(study.job_folder(self.output,task,42)/'test_predictions.parquet')
            self.assertEqual(len(frame),result['rows']); self.assertTrue(frame.base_id.is_unique)
            self.assertNotIn('prompt',frame.columns)
            for arm in study.ARMS:
                probs=frame[[f'{arm}_p{i}' for i in range(2 if task=='binary' else 5)]].to_numpy()
                np.testing.assert_allclose(probs.sum(1),1.,atol=2e-7)
        final=study.summarize(self.output)
        self.assertEqual(len(final['aggregate']),6); self.assertTrue(final['complete'])
        self.assertEqual(study.progress(self.output)['test_complete_heads'],6)
        # Lock binds the selected states and held-out validation threshold.
        selected=study.job_folder(self.output,'binary',42)/'selection.json'
        value=study.read(selected); value['arms']['last_mlp']['threshold']=.123
        study.write(selected,value)
        with self.assertRaises(ValueError): study.verify_test_lock(self.output)

    def test_resumes_from_committed_epoch_without_changing_predictions(self):
        original=study.predict; counter=[0]
        def fail_after_first_epoch(*args,**kwargs):
            counter[0]+=1
            if counter[0]==2: raise RuntimeError('simulated interrupted validation')
            return original(*args,**kwargs)
        with mock.patch.object(study,'predict',side_effect=fail_after_first_epoch):
            with self.assertRaisesRegex(RuntimeError,'simulated'): study.fit(self.output,'binary',42)
        folder=study.job_folder(self.output,'binary',42)
        saved=torch.load(folder/'epoch_resume.pt',map_location='cpu',weights_only=True)
        self.assertEqual(saved['completed_epochs'],1)
        resumed=study.fit(self.output,'binary',42)
        expected=pd.read_parquet(folder/'validation_predictions.parquet')
        for name in ('epoch_resume.pt','selection.json','selected_heads.pt','validation_predictions.parquet','history.json','progress.json'):
            (folder/name).unlink(missing_ok=True)
        clean=study.fit(self.output,'binary',42)
        pd.testing.assert_frame_equal(expected,pd.read_parquet(folder/'validation_predictions.parquet'))
        self.assertEqual(resumed['arms'],clean['arms'])

    def test_malformed_or_modified_features_are_rejected(self):
        d=self.output/'features/train/shard_00000'
        values=np.load(d/'token_features.npy'); values.flat[0]+=1
        np.save(d/'token_features.npy',values,allow_pickle=False)
        cache=study._load_split(self.output,'train')
        with self.assertRaisesRegex(ValueError,'payload changed'): cache['tokens'].get_row(0)

    def test_fresh_prepared_row_order_must_match_raw_data(self):
        path=self.output/'features/train/prepared/metadata.parquet'
        pd.read_parquet(path).iloc[::-1].to_parquet(path,index=False)
        manifest=self.output/'features/train/manifest.json'
        value=study.read(manifest); value['files']['prepared/metadata.parquet']=study.sha(path); study.write(manifest,value)
        with self.assertRaisesRegex(ValueError,'row identities'): study._load_split(self.output,'train')

    def test_unadmitted_cuda_is_rejected_without_cuda_initialization(self):
        with mock.patch.dict('os.environ',{},clear=True), mock.patch('torch.cuda.init') as init:
            with self.assertRaises(RuntimeError): study._admit(self.output,'cuda')
            init.assert_not_called()

    def test_registered_task_counts_are_enforced(self):
        self.protocol['task_rows']={'binary':{'train':20},'category':{'train':11}}
        study.write(self.output/'protocol.json',self.protocol)
        study.write(self.output/'protocol_registration.json',{'protocol_sha256':study.sha(self.output/'protocol.json')})
        with self.assertRaisesRegex(ValueError,'row count'): study._load_split(self.output,'train')

    def test_validation_extension_rule_is_shared_across_all_three_heads(self):
        history=[dict(epoch=e,validation_ce={a:1.-e*.01 for a in study.ARMS}) for e in range(10,13)]
        selected={a:12 for a in study.ARMS}
        ext=study.extension_rule(history,selected,12,40)
        self.assertEqual(ext['new_budget'],24)
        self.assertEqual(set(ext['triggers']),set(study.ARMS))
        self.assertIsNone(study.extension_rule(history,selected,40,40))


if __name__=='__main__': unittest.main()
