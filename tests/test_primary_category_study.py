"""Single-primary-label invariants for the read-only feature-cache study."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from scripts.hanguard import primary_category_study as study
from scripts.hanguard.multilabel_heads import MultiLabelHead


def test_category_targets_use_original_harmful_rows_and_ignore_new_annotations():
    frame = pd.DataFrame({'category_id': ['0', '1', '5'],
                          'prompt_harm_label': ['unharmful', 'harmful', 'harmful'],
                          'labels': [[1]*5, [-1]*5, [0]*5]})
    categories, indices = study.category_arrays(frame)
    assert categories.tolist() == [0, 1, 5]
    assert indices.tolist() == [1, 2]
    assert (categories[indices] - 1).tolist() == [0, 4]
    for value in [None, '6', '-1', '1.5', 'invalid']:
        invalid = frame.copy(); invalid.loc[1, 'category_id'] = value
        with pytest.raises(ValueError):
            study.category_arrays(invalid)
    mismatch = frame.copy(); mismatch.loc[0, 'prompt_harm_label'] = 'harmful'
    with pytest.raises(ValueError, match='contradict'):
        study.category_arrays(mismatch)


def test_softmax_argmax_and_unchanged_binary_gate_cover_all_rows():
    logits = np.array([[5, 4, 3, 2, 1], [0, 0, 0, 0, 8], [3, 3, 0, 0, 0]])
    p, predicted, gated = study.decisions(logits, [.9, .49, .5], .5)
    np.testing.assert_allclose(p.sum(1), 1)
    assert predicted.tolist() == [1, 5, 1]
    assert gated.tolist() == [1, 0, 1]
    metrics = study.metric_pair(np.array([1, 5, 0]), predicted, gated)
    assert metrics['type_metrics']['rows'] == 2
    assert metrics['type_metrics']['accuracy'] == 1
    assert metrics['type_metrics']['macro_f1'] == pytest.approx(2/5)
    assert metrics['gated_metrics']['rows'] == 3
    assert metrics['gated_metrics']['accuracy'] == pytest.approx(1/3)
    assert metrics['type_metrics']['per_class'][1]['support_flags'] == ['no_true_support']
    assert study.metric_pair(np.array([0]), np.array([1]), np.array([0]))['type_metrics']['accuracy'] is None


@pytest.mark.parametrize('mode', study.ARMS)
def test_microbatch_ce_equals_full_harmful_batch_and_safe_rows_rejected(monkeypatch, mode):
    torch.manual_seed(13)
    vectors = torch.randn(5, 8) if mode == 'description_queries' else None
    head = MultiLabelHead(8, width=4, mode=mode, query_vectors=vectors, dropout=0.)
    reference = copy.deepcopy(head)
    x = torch.randn(6, 3, 8)
    categories = np.array([0, 1, 4, 5, 2, 1])
    data = {'categories': categories, 'y': torch.tensor(categories-1), 'lengths': np.full(6, 3)}
    monkeypatch.setattr(study.shared, 'feature_batch', lambda d, i, p: (x[i], torch.ones(len(i), 3, dtype=torch.bool)))
    ids = np.array([1, 2, 3, 4, 5])
    protocol = {'max_micro': 2, 'token_budget': 100, 'pad_multiple': 1}
    optimizer = torch.optim.SGD(head.parameters(), lr=.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    expected_loss = F.cross_entropy(reference(x[ids], torch.ones(5, 3, dtype=torch.bool))['logits'], data['y'][ids])
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    reference_optimizer.step()
    result = study.macro_step(head, optimizer, data, ids, protocol, rng_seed=91)
    assert result['ce'] == pytest.approx(float(expected_loss.detach()), rel=1e-6)
    for actual, expected in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match='harmful'):
        study.macro_step(head, optimizer, data, np.array([0, 1]), protocol, rng_seed=91)


def test_validation_ce_and_saved_predictions_ignore_head_sigmoids(monkeypatch, tmp_path):
    values = torch.tensor([[100., -100., -100., -100., -100.], [0., 0., 0., 0., 4.]])
    class MockHead:
        def eval(self): return self
        def __call__(self, hidden, mask):
            return {'logits': values[hidden[:, 0, 0].long()], 'probabilities': torch.zeros(len(hidden), 5)}
    monkeypatch.setattr(study.shared, 'feature_batch', lambda d, i, p: (torch.tensor(i).reshape(-1, 1, 1), torch.ones(len(i), 1, dtype=torch.bool)))
    meta = pd.DataFrame({'base_id':['safe','harmful'], 'source':['x','y'], 'prompt_sha256':['a','b'],
                         'prompt_tokens':[1,1], 'category_id':[0,5], 'binary_probability':[.1,.9]})
    data = {'lengths':np.array([1,1]), 'categories':np.array([0,5]), 'harmful_indices':np.array([1]), 'meta':meta}
    logits, ce = study.prediction(MockHead(), data, {'token_budget':32,'max_micro':2,'pad_multiple':1})
    assert ce == pytest.approx(float(F.cross_entropy(values[1:], torch.tensor([4]))))
    path = tmp_path/'predictions.csv'
    study.save_predictions(data, logits, path, .5)
    saved = pd.read_csv(path)
    assert len(saved) == 2 and saved.gated_category.tolist() == [0,5]
    np.testing.assert_allclose(saved[[f'p_{i}' for i in range(1,6)]].sum(1), 1.)


def cache_fixture(tmp_path):
    cache = tmp_path/'feature_cache'; cache.mkdir()
    data_dir = tmp_path/'inputs'; data_dir.mkdir()
    frame = pd.DataFrame({'base_id':['a','b'], 'prompt':['完整文本','另外文本'], 'source':['x','y'],
                          'category_id':['0','3'], 'prompt_harm_label':['unharmful','harmful']})
    path = data_dir/'train.parquet'; frame.to_parquet(path, index=False)
    meta = frame[['base_id','source']].copy()
    meta['prompt_sha256'] = frame.prompt.map(study.shared.text_sha)
    meta['prompt_tokens'] = [2,3]; meta['binary_probability'] = [.1,.8]
    meta['original_binary_label'] = [0,1]
    meta.to_parquet(cache/'train_index.parquet', index=False)
    np.save(cache/'train_offsets.npy', np.array([0,2,5]))
    raw = torch.arange(20).reshape(5,4).to(torch.bfloat16)
    np.save(cache/'train_tokens.npy', raw.view(torch.uint16).numpy())
    identity = study.shared.text_identity(frame)
    info = {'rows':2,'tokens':5,'maximum_tokens':3,'text_identity':identity,
            'index_sha256':study.sha(cache/'train_index.parquet'),
            'offsets_sha256':study.sha(cache/'train_offsets.npy'),
            'tokens_sha256':study.sha(cache/'train_tokens.npy')}
    manifest = {'text_identities':{'train':identity},'splits':{'train':info},'hidden_size':4}
    protocol = {'data_dir':str(data_dir),'data_sha256':{'train':study.sha(path)},'rows':{'train':2},
                'max_tokens':4096,'cache_on_gpu':True}
    return protocol, manifest, raw


def test_cache_reads_original_labels_without_annotation_manifest_and_binds_hashes(tmp_path):
    protocol, manifest, raw = cache_fixture(tmp_path)
    data = study.load_cached_split(tmp_path, protocol, manifest, 'train', device='cpu')
    assert data['harmful_indices'].tolist() == [1] and data['y'].tolist() == [-1,2]
    torch.testing.assert_close(data['tokens'], raw)
    assert not data['tokens'].requires_grad
    path = tmp_path/'inputs/train.parquet'; frame = pd.read_parquet(path)
    frame.loc[1,'category_id'] = '4'; frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='labels changed'):
        study.load_cached_split(tmp_path, protocol, manifest, 'train', device='cpu')


def test_cache_detects_text_and_feature_changes_even_with_new_data_hash(tmp_path):
    protocol, manifest, raw = cache_fixture(tmp_path)
    path = tmp_path/'inputs/train.parquet'; frame = pd.read_parquet(path)
    frame.loc[1,'prompt'] = '改变后的完整文本'; frame.to_parquet(path,index=False)
    protocol['data_sha256']['train'] = study.sha(path)
    with pytest.raises(ValueError,match='full prompts'):
        study.load_cached_split(tmp_path, protocol, manifest, 'train', device='cpu')


def test_test_barrier_requires_all_three_selections_and_unchanged_test_data(tmp_path):
    data = tmp_path/'data'; data.mkdir(); (data/'test.parquet').write_bytes(b'original labels')
    protocol = {'all_arms':list(study.ARMS),'seeds':[42],'data_dir':str(data)}
    study.dump(tmp_path/'protocol.json',protocol)
    selections = {}
    for arm in study.ARMS:
        path = tmp_path/'runs'/f'{arm}_s42'/'selection.json'; study.dump(path, {'arm':arm})
        selections[str(path.relative_to(tmp_path))] = study.sha(path)
    barrier = {'locked':True,'protocol_sha256':study.sha(tmp_path/'protocol.json'),
               'test_dataset_sha256':study.sha(data/'test.parquet'),'selection_sha256':selections}
    study.dump(tmp_path/'test_barrier.json',barrier)
    assert study.validate_test_barrier(tmp_path,protocol)['locked']
    first_key = next(iter(selections)); del barrier['selection_sha256'][first_key]
    study.dump(tmp_path/'test_barrier.json',barrier)
    with pytest.raises(ValueError,match='All registered selections'):
        study.validate_test_barrier(tmp_path,protocol)
    barrier['selection_sha256'][first_key] = study.sha(tmp_path/first_key)
    study.dump(tmp_path/'test_barrier.json',barrier); (data/'test.parquet').write_bytes(b'changed labels')
    with pytest.raises(ValueError,match='Test labels'):
        study.validate_test_barrier(tmp_path,protocol)


def test_early_stopping_uses_validation_ce_delta_and_exact_resume_history():
    protocol = dict(early_stopping_patience=3, early_stopping_min_delta=.01)
    history = [{'validation_ce': value} for value in [1., .995, .992, .98, .979, .978]]
    state = study.early_stopping_progress(history, protocol)
    assert state['reference_ce'] == .98 and state['stale_epochs'] == 2 and not state['should_stop']
    history.append({'validation_ce': .977})
    assert study.early_stopping_progress(history, protocol)['should_stop']
    assert not study.early_stopping_progress(history, {})['should_stop']
    assert study.early_stopping_progress(history[:3], protocol)['stale_epochs'] == 2
    with pytest.raises(ValueError, match='early-stopping'):
        study.early_stopping_progress([], dict(early_stopping_patience=-1))


def test_per_arm_width_override_preserves_shared_protocol(monkeypatch, tmp_path):
    protocol = dict(head_width=128, head_width_by_arm=dict(last_mlp=275, learned_queries=128, description_queries=128))
    observed = []
    monkeypatch.setattr(study.shared, 'setup_head', lambda s, p, m, a, seed: observed.append((a, p['head_width'], seed)))
    for arm in study.ARMS:
        study.setup_head(tmp_path, protocol, {}, arm, 44)
    assert observed == [('last_mlp', 275, 44), ('learned_queries', 128, 44), ('description_queries', 128, 44)]
    assert protocol['head_width'] == 128


@pytest.mark.parametrize('resume_after_stop', [False, True])
def test_cpu_training_stops_on_patience_but_selects_actual_minimum_ce(monkeypatch, tmp_path, resume_after_stop):
    protocol = dict(all_arms=list(study.ARMS), seeds=[42], epochs=40, effective_batch=128,
                    head_lr=.001, weight_decay=.01, head_width=128,
                    data_sha256=dict(train='train_hash', validation='validation_hash'),
                    early_stopping_patience=3, early_stopping_min_delta=1e-4)
    study.dump(tmp_path / 'protocol.json', protocol)
    manifest = dict(fingerprint='fixed', parent=dict(binary_threshold=.5))
    meta = pd.DataFrame(dict(base_id=['safe', 'harm'], source=['chinese_curated'] * 2,
                            prompt_sha256=['a', 'b'], prompt_tokens=[1, 1], category_id=[0, 1],
                            binary_probability=[.1, .9]))
    data = dict(categories=np.array([0, 1]), harmful_indices=np.array([1]), lengths=np.array([1, 1]), meta=meta)
    loaded = []
    monkeypatch.setattr(study, 'read_protocol', lambda path: protocol)
    monkeypatch.setattr(study, 'cache_manifest', lambda path, p: manifest)
    monkeypatch.setattr(study, 'load_cached_split', lambda s, p, m, split: (loaded.append(split) or data))
    head = torch.nn.Linear(1, 5, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    monkeypatch.setattr(study, 'setup_head', lambda *args: head)
    def step(head, optimizer, data, ids, protocol, seed):
        with torch.no_grad():
            head.weight.add_(1)
        return dict(ce=.1, rows=len(ids), grad_norm=1.)
    monkeypatch.setattr(study, 'macro_step', step)
    scores = iter([1., .9, .90005, .89995, .9001])
    monkeypatch.setattr(study, 'prediction', lambda *args: (np.ones((2, 5), dtype=np.float32), next(scores)))
    args = SimpleNamespace(output=str(tmp_path), arm='last_mlp', seed=42)
    if resume_after_stop:
        save = study.atomic_torch_save
        def crash_after_save(value, path):
            save(value, path)
            if path.name == 'resume.pt' and value['epoch'] == 5:
                raise RuntimeError('simulated interruption after patience reached')
        monkeypatch.setattr(study, 'atomic_torch_save', crash_after_save)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            study.train(args)
        monkeypatch.setattr(study, 'atomic_torch_save', save)
    study.train(args)
    directory = tmp_path / 'runs/last_mlp_s42'
    result = json.loads((directory / 'selection.json').read_text())
    assert result['best_epoch'] == 4 and result['completed_epochs'] == 5
    assert result['validation_ce'] == .89995 and result['stopped_early']
    assert result['total_steps'] == 5 and result['maximum_steps'] == 40
    assert result['early_stopping']['stale_epochs'] == 3
    checkpoint = torch.load(directory / 'head.pt', weights_only=True)
    assert torch.equal(checkpoint['weight'], torch.full((5, 1), 4.))
    assert loaded == ['train', 'validation'] * (2 if resume_after_stop else 1)
    assert not (directory / 'resume.pt').exists()
