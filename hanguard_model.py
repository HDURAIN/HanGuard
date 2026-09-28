"""hanguard: a frozen Qwen3.5 backbone with binary and primary-category heads.

Importing this module does not import torch, load model weights, or initialize
CUDA. The manifest pins one binary checkpoint and one validation-selected type
checkpoint; inference consumes full raw text without a generation prompt.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(ROOT / 'models/Qwen3.5-4B')
DEFAULT_MANIFEST = ROOT / 'models/hanguard/model.json'
DEFAULT_OUTPUT = str(DEFAULT_MANIFEST.parent)
MAX_PROMPT_TOKENS = 4096
CATEGORY_LABELS = {
    '0': '安全',
    '1': '违反社会主义核心价值观的内容',
    '2': '歧视性内容',
    '3': '商业违法违规',
    '4': '侵犯他人合法权益',
    '5': '无法满足特定服务类型的安全需求',
}


def format_completion(harm, category_id):
    """Presentation only: the classifier does not generate or parse label text."""
    category_id = str(category_id)
    if harm not in ('harmful', 'unharmful') or category_id not in CATEGORY_LABELS:
        raise ValueError('Unknown harmfulness or category label')
    if (harm == 'unharmful') != (category_id == '0'):
        raise ValueError('Inconsistent harmfulness and category')
    return ('有害' if harm == 'harmful' else '无害') + '\n' + CATEGORY_LABELS[category_id]


def parse_output(text):
    """Read the public two-line display format, for saved outputs and clients."""
    lines = [line.strip() for line in text.strip().splitlines()]
    if len(lines) != 2 or lines[0] not in ('有害', '无害'):
        raise ValueError('Expected harmfulness and category on two lines')
    category = next((key for key, label in CATEGORY_LABELS.items() if label == lines[1]), None)
    harm = 'harmful' if lines[0] == '有害' else 'unharmful'
    format_completion(harm, category)
    return harm, category


def load_base_model(model_path, **kwargs):
    """Shared lazy loader retained for the current training scripts."""
    from transformers import Qwen3_5ForConditionalGeneration
    kwargs.setdefault('dtype', 'bfloat16')
    return Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def read_manifest(path=DEFAULT_MANIFEST):
    """Validate runtime settings and checkpoint hashes without importing torch."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / 'model.json'
    if not path.is_file():
        raise FileNotFoundError(f'找不到 hanguard 模型清单：{path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'hanguard_runtime_1':
        raise ValueError('Unsupported hanguard runtime manifest')
    base = (path.parent / value['base_model']).resolve()
    for filename, key in [('config.json', 'model_config_sha256'), ('tokenizer.json', 'tokenizer_sha256')]:
        if file_sha(base / filename) != value[key]:
            raise ValueError(f'Base model file differs from the selected model: {filename}')
    for key in ('max_tokens', 'pad_multiple', 'token_budget'):
        _positive_integer(value[key], key)
    padded_max = math.ceil(value['max_tokens'] / value['pad_multiple']) * value['pad_multiple']
    if value['token_budget'] < padded_max:
        raise ValueError('Token budget must accommodate one full-length input')
    binary, category = value['binary'], value['category']
    if (binary.get('mode'), binary.get('readout')) != ('fusion', 'mlp'):
        raise ValueError('The deployed binary head must be the registered fusion MLP')
    if binary.get('layers') != [8, 16, 24, 32]:
        raise ValueError('The deployed binary head requires layers 8,16,24,32')
    if category.get('mode') != 'learned_queries' or category.get('selection_rule') != 'minimum_validation_ce':
        raise ValueError('Deploy a learned-query checkpoint selected by validation CE')
    if type(category.get('seed')) is not int or category['seed'] < 0:
        raise ValueError('Invalid selected category seed')
    if not math.isfinite(category['validation_ce']) or category['validation_ce'] < 0:
        raise ValueError('Invalid category validation CE')
    if not math.isfinite(binary['threshold']) or not 0 <= binary['threshold'] <= 1:
        raise ValueError('Invalid validation-selected binary threshold')
    for head in (binary, category):
        _positive_integer(head['width'], 'head width')
    _positive_integer(binary['lora']['rank'], 'LoRA rank')
    if not isinstance(binary['lora']['target_modules'], (str, list)):
        raise ValueError('LoRA target modules are required')
    weights = {}
    for name, spec in [('adapter', binary['adapter']), ('binary_head', binary['head']), ('category_head', category['head'])]:
        target = (path.parent / spec['path']).resolve()
        if file_sha(target) != spec['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        weights[name] = target
    return value, dict(manifest=path, base_model=base, **weights)


def select_category_checkpoint(study):
    """Choose among the registered learned-query seeds using validation CE only.

    This packaging helper intentionally never opens test predictions or results.
    Runtime inference uses the selected, hash-pinned bundle instead of reselecting.
    """
    study = Path(study)
    protocol_path = study / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    candidates = []
    for seed in protocol['seeds']:
        directory = study / 'runs' / f'learned_queries_s{seed}'
        selection = json.loads((directory / 'selection.json').read_text())
        if selection.get('arm') != 'learned_queries' or selection.get('seed') != seed:
            raise ValueError('Category checkpoint seed or architecture mismatch')
        if selection.get('protocol_sha256') != file_sha(protocol_path):
            raise ValueError('Category checkpoint belongs to another registered protocol')
        ce = selection['validation_ce']
        if not math.isfinite(ce) or ce < 0:
            raise ValueError('Invalid category validation CE')
        if file_sha(directory / 'head.pt') != selection['checkpoint_sha256']:
            raise ValueError('Category checkpoint changed after selection')
        candidates.append((float(ce), seed, directory, selection))
    if not candidates:
        raise ValueError('No registered category seeds')
    _, _, directory, selection = min(candidates, key=lambda item: (item[0], item[1]))
    return directory, selection


def build_result(harmful_probability, category_probabilities, tokens, threshold):
    """Apply the frozen binary threshold, then the five-way argmax."""
    probability = float(harmful_probability)
    scores = [float(value) for value in category_probabilities]
    if (not math.isfinite(probability) or not 0 <= probability <= 1 or len(scores) != 5 or
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores) or
            not math.isclose(sum(scores), 1., abs_tol=1e-5)):
        raise ValueError('Invalid model probabilities')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid harmfulness threshold')
    harmful = probability >= threshold
    category = max(range(5), key=scores.__getitem__) + 1 if harmful else 0
    return dict(harmful=harmful, harmful_label='harmful' if harmful else 'unharmful',
                safety_label='有害' if harmful else '无害', category_id=category,
                category_label=CATEGORY_LABELS[str(category)], harmful_probability=probability,
                category_probabilities={str(index + 1): score for index, score in enumerate(scores)},
                binary_threshold=float(threshold), input_tokens=int(tokens))


class HanguardClassifier:
    """One frozen backbone pass supplies both heads; calls are serialized."""
    def __init__(self, *, tokenizer, model, binary_head, category_head, tap, manifest, device):
        self.tokenizer, self.model = tokenizer, model
        self.binary_head, self.category_head, self.tap = binary_head, category_head, tap
        self.manifest, self.device = manifest, device
        self._lock = threading.RLock()
        self._captured = {}
        self._closed = False
        self._capture_handle = tap.backbone.layers[31].register_forward_hook(self._capture_last)

    def _capture_last(self, module, inputs, output):
        self._captured['last'] = (output[0] if isinstance(output, tuple) else output).detach()

    @classmethod
    def from_manifest(cls, manifest=DEFAULT_MANIFEST, *, device='cuda:0'):
        settings, paths = read_manifest(manifest)
        import torch
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from scripts.hanguard.repaired_heads import ExperimentHead, TapOnline
        from scripts.hanguard.multilabel_heads import MultiLabelHead
        target = torch.device(device)
        if target.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('当前 BF16 模型需要可用 CUDA GPU；请设置 --device 或 CUDA_VISIBLE_DEVICES。')
        tokenizer = AutoTokenizer.from_pretrained(paths['base_model'], local_files_only=True)
        tokenizer.pad_token = '<|im_end|>'
        tokenizer.padding_side = 'right'
        model = load_base_model(paths['base_model'], local_files_only=True,
                                device_map={'': str(target)}, attn_implementation='sdpa')
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        lora = settings['binary']['lora']
        model = get_peft_model(model, LoraConfig(r=lora['rank'], lora_alpha=lora['alpha'],
            lora_dropout=lora['dropout'], target_modules=lora['target_modules'], bias='none'))
        loaded = set_peft_model_state_dict(model, torch.load(paths['adapter'], map_location='cpu', weights_only=True))
        if loaded.unexpected_keys or any('lora_' in name for name in loaded.missing_keys):
            raise ValueError('LoRA checkpoint does not exactly cover the configured adapters')
        model.eval().requires_grad_(False)
        hidden = model.config.text_config.hidden_size
        binary = settings['binary']
        binary_head = ExperimentHead(hidden, mode=binary['mode'], readout=binary['readout'],
            width=binary['width'], layers=tuple(binary['layers'])).to(target)
        binary_head.load_state_dict(torch.load(paths['binary_head'], map_location='cpu', weights_only=True), strict=True)
        binary_head.eval().requires_grad_(False)
        category = settings['category']
        category_head = MultiLabelHead(hidden, width=category['width'], num_labels=5,
            mode='learned_queries', dropout=category.get('dropout', .1), query_seed=category['seed']).to(target)
        category_head.load_state_dict(torch.load(paths['category_head'], map_location='cpu', weights_only=True), strict=True)
        category_head.eval().requires_grad_(False)
        tap = TapOnline(model, binary_head)
        return cls(tokenizer=tokenizer, model=model, binary_head=binary_head,
                   category_head=category_head, tap=tap, manifest=settings, device=target)

    def _encode(self, texts):
        if isinstance(texts, str):
            raise TypeError('predict expects a list of texts; use classify for a single string')
        texts = list(texts)
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'第 {index + 1} 条输入必须是非空文本')
        if not texts:
            return []
        encoded = self.tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
        for index, tokens in enumerate(encoded):
            if not tokens or len(tokens) > self.manifest['max_tokens']:
                raise ValueError(f'第 {index + 1} 条输入有 {len(tokens)} tokens；支持 1–'
                                 f"{self.manifest['max_tokens']} tokens，不会截断文本。")
        return encoded

    def _batches(self, encoded, batch_size):
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
        group = []
        multiple = self.manifest['pad_multiple']
        for index in order:
            padded = math.ceil(len(encoded[index]) / multiple) * multiple
            if group and (len(group) >= batch_size or padded * (len(group) + 1) > self.manifest['token_budget']):
                yield group
                group = []
            group.append(index)
        if group:
            yield group

    def predict(self, texts, *, batch_size=8):
        _positive_integer(batch_size, 'batch_size')
        with self._lock:
            if self._closed:
                raise RuntimeError('Classifier is closed')
            encoded = self._encode(texts)
            if not encoded:
                return []
            import torch
            result = [None] * len(encoded)
            with torch.inference_mode():
                for indices in self._batches(encoded, batch_size):
                    batch = self.tokenizer.pad([dict(input_ids=encoded[index], attention_mask=[1] * len(encoded[index]))
                        for index in indices], padding=True, pad_to_multiple_of=self.manifest['pad_multiple'],
                        return_tensors='pt').to(self.device)
                    self._captured.clear()
                    try:
                        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            binary = self.tap(batch)['logits']
                        raw = self._captured.pop('last')
                        logits = self.category_head(raw, batch['attention_mask'].bool())['logits']
                        harmful = binary.double().sigmoid().cpu().tolist()
                        probabilities = logits.double().softmax(-1).cpu().tolist()
                        for row, index in enumerate(indices):
                            result[index] = build_result(harmful[row], probabilities[row], len(encoded[index]),
                                                         self.manifest['binary']['threshold'])
                    finally:
                        self._captured.clear()
            return result

    def classify(self, text):
        return self.predict([text], batch_size=1)[0]

    def close(self):
        with self._lock:
            if not self._closed:
                self._capture_handle.remove()
                self.tap.close()
                self._captured.clear()
                # Drop all owners of backbone/head references so resident services release the GPU.
                self.tap = self.model = self.binary_head = self.category_head = None
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
