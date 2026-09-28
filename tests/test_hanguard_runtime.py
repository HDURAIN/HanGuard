"""Runtime contract tests; no pretrained model, network, or CUDA required."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import hanguard_model as runtime
import infer
import evaluate


def test_entrypoint_imports_and_help_do_not_import_torch_or_transformers():
    root = Path(runtime.__file__).parent
    script = ('import sys; import hanguard_model,infer,demo_infer,evaluate,server; '
              "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
    subprocess.run([sys.executable, '-c', script], cwd=root, check=True, capture_output=True, text=True)
    for name in ['infer.py', 'demo_infer.py', 'server.py', 'evaluate.py']:
        result = subprocess.run([sys.executable, str(root / name), '--help'], cwd=root / 'tests',
                                check=True, capture_output=True, text=True)
        assert '--model' in result.stdout
    assert runtime.DEFAULT_MANIFEST.is_absolute()


def manifest_fixture(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    for name in ['config.json', 'tokenizer.json']:
        (base / name).write_text('{}')
    folder = tmp_path / 'bundle'
    folder.mkdir()
    weights = {}
    for name in ['adapter', 'binary', 'category']:
        file = folder / (name + '.pt')
        file.write_bytes(name.encode())
        weights[name] = dict(path=file.name, sha256=runtime.file_sha(file))
    spec = dict(schema='hanguard_runtime_1', base_model='../base',
        model_config_sha256=runtime.file_sha(base / 'config.json'),
        tokenizer_sha256=runtime.file_sha(base / 'tokenizer.json'),
        max_tokens=64, pad_multiple=8, token_budget=128,
        binary=dict(mode='fusion', readout='mlp', width=3, layers=[8, 16, 24, 32], threshold=.6,
                    lora=dict(rank=8, alpha=16, dropout=.05, target_modules='test'),
                    adapter=weights['adapter'], head=weights['binary']),
        category=dict(mode='learned_queries', width=3, dropout=.1, seed=43, validation_ce=.4,
                      selection_rule='minimum_validation_ce', head=weights['category']))
    path = folder / 'model.json'
    path.write_text(json.dumps(spec))
    return path, spec


def test_manifest_resolves_its_own_directory_and_checks_artifact_hashes(tmp_path):
    path, spec = manifest_fixture(tmp_path)
    loaded, paths = runtime.read_manifest(path.parent)
    assert loaded == spec
    assert paths['base_model'] == (tmp_path / 'base').resolve()
    assert paths['adapter'] == (path.parent / 'adapter.pt').resolve()
    (path.parent / 'category.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checkpoint hash'):
        runtime.read_manifest(path)


def test_manifest_rejects_different_base_tokenizer_or_test_selection(tmp_path):
    path, spec = manifest_fixture(tmp_path)
    spec['category']['selection_rule'] = 'maximum_test_accuracy'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='validation CE'):
        runtime.read_manifest(path)
    spec['category']['selection_rule'] = 'minimum_validation_ce'
    path.write_text(json.dumps(spec))
    (tmp_path / 'base/tokenizer.json').write_text('changed')
    with pytest.raises(ValueError, match='tokenizer'):
        runtime.read_manifest(path)


def test_checkpoint_choice_uses_validation_only_even_when_test_winner_differs(tmp_path, monkeypatch):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'seeds': [42, 43, 44]}))
    for seed, ce, test in [(42, .52, .9), (43, .47, .7), (44, .54, .99)]:
        directory = tmp_path / 'runs' / f'learned_queries_s{seed}'
        directory.mkdir(parents=True)
        (directory / 'head.pt').write_bytes(str(seed).encode())
        (directory / 'selection.json').write_text(json.dumps(dict(arm='learned_queries', seed=seed,
            validation_ce=ce, protocol_sha256=runtime.file_sha(protocol),
            checkpoint_sha256=runtime.file_sha(directory / 'head.pt'))))
        (directory / 'test_results.json').write_text(json.dumps({'accuracy': test}))
    original = Path.read_text
    def no_test_read(path, *args, **kwargs):
        assert 'test' not in path.name
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', no_test_read)
    selected, value = runtime.select_category_checkpoint(tmp_path)
    assert selected.name == 'learned_queries_s43' and value['seed'] == 43


def test_binary_gate_and_type_argmax_have_no_fixed_category_priority():
    scores = [.05, .1, .5, .3, .05]
    result = runtime.build_result(.6, scores, 42, .6)
    assert result['harmful'] and result['category_id'] == 3
    assert runtime.format_completion(result['harmful_label'], result['category_id']) == '有害\n商业违法违规'
    safe = runtime.build_result(.59, scores, 42, .6)
    assert not safe['harmful'] and safe['category_id'] == 0 and safe['category_label'] == '安全'
    assert safe['category_probabilities']['3'] == .5
    with pytest.raises(ValueError):
        runtime.build_result(float('nan'), scores, 42, .6)
    with pytest.raises(ValueError):
        runtime.build_result(.6, [.1] * 5, 42, .6)
    assert runtime.parse_output('无害\n安全') == ('unharmful', '0')


class StubClassifier:
    def __init__(self):
        self.closed = False
    def predict(self, texts, batch_size=8):
        return [runtime.build_result(.7 if '产品' in text else .2, [.1, .1, .6, .1, .1], len(text), .5) for text in texts]
    def classify(self, text):
        return self.predict([text])[0]
    def close(self):
        self.closed = True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def test_cli_single_text_and_json_output(monkeypatch, capsys):
    model = StubClassifier()
    monkeypatch.setattr(infer, 'load_model', lambda *args, **kwargs: model)
    assert infer.main(['--text', '产品']) == 0
    assert capsys.readouterr().out == '有害\n商业违法违规\n'
    assert infer.main(['--text', '你好', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['category_id'] == 0 and result['safety_label'] == '无害'
    assert model.closed


def test_batch_preserves_original_labels_and_exports_current_probability_columns(monkeypatch, tmp_path):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([dict(prompt='产品', category_id=4, prompt_harm_label='harmful', source='example')]))
    output = tmp_path / 'predictions.jsonl'
    monkeypatch.setattr(infer, 'load_model', lambda *args, **kwargs: StubClassifier())
    assert infer.main(['--input', str(path), '--output', str(output)]) == 0
    row = json.loads(output.read_text())
    assert row['category_id'] == 4 and row['category_pred'] == 3
    assert row['source'] == 'example' and row['category_p_3'] == .6
    assert infer.load_input(output)[0] == row


def test_invalid_input_is_rejected_without_loading_model(monkeypatch, tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('[{"prompt": null}]')
    monkeypatch.setattr(infer, 'load_model', lambda *args, **kwargs: pytest.fail('should not load'))
    with pytest.raises(ValueError, match='prompt'):
        infer.main(['--input', str(path)])


def test_evaluation_separates_ungated_type_accuracy_from_safety_gate():
    rows = [dict(prompt='产品', category_id=3, prompt_harm_label='harmful'),
            dict(prompt='你好', category_id=0, prompt_harm_label='unharmful')]
    prediction = runtime.build_result(.1, [.1, .1, .6, .1, .1], 2, .5)
    exported = infer.records_with_predictions(rows, [prediction, prediction])
    report = evaluate.evaluate_records(exported)
    assert report['binary']['accuracy'] == .5
    assert report['primary_category']['rows'] == 1 and report['primary_category']['accuracy'] == 1.
    assert report['end_to_end']['accuracy'] == .5
    exported[0]['category_p_3'] = .2
    with pytest.raises(ValueError, match='softmax'):
        evaluate.evaluate_records(exported)


def test_http_handlers_share_new_prediction_contract_without_loading_gpu():
    import server
    model = StubClassifier()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        classifier=model, inference_lock=asyncio.Lock(), ready=True)))
    result = asyncio.run(server.classify(server.ClassifyRequest(prompt='产品'), request))
    assert server.Classification(**result).category_id == 3
    result = asyncio.run(server.classify_batch(server.BatchClassifyRequest(prompts=['你好', '产品']), request))
    assert [row['category_id'] for row in result['results']] == [0, 3]
    with pytest.raises(server.HTTPException) as error:
        asyncio.run(server.classify(server.ClassifyRequest(prompt='  '), request))
    assert error.value.status_code == 422


def test_runtime_full_text_order_padding_tuple_hooks_and_cleanup():
    import torch
    from scripts.hanguard.multilabel_heads import MultiLabelHead
    class Batch(dict):
        def to(self, device):
            return self
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {'add_special_tokens': False, 'truncation': False}
            return {'input_ids': [[ord(c) % 11 + 1 for c in text] for text in texts]}
        def pad(self, rows, padding, pad_to_multiple_of, return_tensors):
            maximum = max(len(row['input_ids']) for row in rows)
            maximum = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            return Batch(input_ids=torch.tensor([row['input_ids'] + [0] * (maximum-len(row['input_ids'])) for row in rows]),
                         attention_mask=torch.tensor([row['attention_mask'] + [0] * (maximum-len(row['attention_mask'])) for row in rows]))
    class Layer:
        def register_forward_hook(self, hook):
            self.hook = hook
            self.removed = False
            return SimpleNamespace(remove=lambda: setattr(self, 'removed', True))
    layer = Layer()
    class Tap:
        backbone = SimpleNamespace(layers=[None] * 31 + [layer])
        closed = False
        def __call__(self, batch):
            raw = batch['input_ids'].float().unsqueeze(-1).repeat(1, 1, 4)
            layer.hook(None, None, (raw,))
            return {'logits': batch['attention_mask'].sum(1).float() - 3.}
        def close(self):
            self.closed = True
    tap = Tap()
    model = runtime.HanguardClassifier(tokenizer=Tokenizer(), model=object(), binary_head=object(),
        category_head=MultiLabelHead(4, width=3, mode='learned_queries').eval(), tap=tap,
        manifest={'max_tokens': 8, 'pad_multiple': 2, 'token_budget': 8, 'binary': {'threshold': .5}},
        device=torch.device('cpu'))
    predictions = model.predict(['四个字啊', '一', '两个'], batch_size=3)
    assert [row['input_tokens'] for row in predictions] == [4, 1, 2]
    assert [row['harmful'] for row in predictions] == [True, False, False]
    assert not model._captured
    assert model.predict([]) == []
    with pytest.raises(ValueError, match='不会截断'):
        model.predict(['九个中文汉字超过限制'])
    with pytest.raises(TypeError):
        model.predict('应使用列表')
    model.close()
    model.close()
    assert tap.closed and layer.removed and model.model is None
    with pytest.raises(RuntimeError, match='closed'):
        model.classify('你好')


def test_evaluation_excludes_wildguard_types_but_keeps_all_binary_rows():
    rows = [
        dict(prompt='商业样本', source='chinese_curated', category_id=3, prompt_harm_label='harmful'),
        dict(prompt='个人权益样本', source='jailbench', category_id=4, prompt_harm_label='harmful'),
        dict(prompt='旧映射样本', source='wildguard_zh', category_id=3, prompt_harm_label='harmful'),
        dict(prompt='范围外安全样本', source='wildguard_zh', category_id=0, prompt_harm_label='unharmful'),
        dict(prompt='中文安全样本', source='chinese_curated', category_id=0, prompt_harm_label='unharmful'),
    ]
    positive = runtime.build_result(.9, [.1, .1, .6, .1, .1], 4, .5)
    negative = runtime.build_result(.1, [.1, .1, .6, .1, .1], 4, .5)
    exported = infer.records_with_predictions(rows, [positive] * 3 + [negative] * 2)
    report = evaluate.evaluate_records(exported)
    assert report['binary']['rows'] == 5 and report['binary']['accuracy'] == 1.
    assert report['primary_category']['rows'] == 2 and report['primary_category']['accuracy'] == .5
    assert report['end_to_end']['rows'] == 3 and report['end_to_end']['accuracy'] == pytest.approx(2 / 3)
    assert report['category_scope']['mode'] == 'registered_two_source'
    assert report['category_scope']['excluded_category_rows'] == 2
    assert report['category_scope']['excluded_harmful_rows'] == 1
    assert report['category_scope']['excluded_source_counts'] == {'wildguard_zh': 2}
    assert report['by_source']['wildguard_zh']['binary']['rows'] == 2
    assert report['by_source']['wildguard_zh']['primary_category']['rows'] == 0
    assert report['by_source']['wildguard_zh']['primary_category']['accuracy'] is None
    generic = [{key: value for key, value in row.items() if key != 'source'} for row in exported]
    report = evaluate.evaluate_records(generic)
    assert report['primary_category']['rows'] == 3
    assert report['primary_category']['accuracy'] == pytest.approx(2 / 3)
    assert report['category_scope']['mode'] == 'user_provided_labels_without_source'
    # Outside-scope legacy categories do not gate their valid binary labels.
    exported[2]['category_id'] = -1
    assert evaluate.evaluate_records(exported)['binary']['rows'] == 5
    exported[0].pop('source')
    with pytest.raises(ValueError, match='source'):
        evaluate.evaluate_records(exported)


def test_jsonl_unicode_line_separators_roundtrip_and_evaluation(tmp_path):
    rows = [
        dict(prompt='第一段\u2028第二段\u2029第三段\n末段', category_id=3, prompt_harm_label='harmful'),
        dict(prompt='普通安全文本', category_id=0, prompt_harm_label='unharmful'),
    ]
    scores = [.1, .1, .6, .1, .1]
    exported = infer.records_with_predictions(rows, [
        runtime.build_result(.9, scores, 8, .5), runtime.build_result(.1, scores, 6, .5)])
    path = tmp_path / 'unicode_predictions.jsonl'
    infer.save_output(path, exported)
    raw = path.read_text(encoding='utf-8')
    assert '\u2028' in raw and '\u2029' in raw
    assert raw.count('\n') == 2
    assert infer.load_input(path) == exported
    report_path = tmp_path / 'report.json'
    assert evaluate.main(['--preds', str(path), '--output', str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert report['binary']['rows'] == 2 and report['binary']['accuracy'] == 1.
    assert report['primary_category']['rows'] == 1 and report['primary_category']['accuracy'] == 1.
