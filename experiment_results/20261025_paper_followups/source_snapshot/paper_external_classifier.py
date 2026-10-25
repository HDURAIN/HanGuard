"""Same-split, full-text MacBERT five-class supervised external reference.

This is an end-to-end encoder baseline, not a matched-budget module ablation.
Long documents use overlapping windows; mean window logits are classified
with one cross-entropy per document. No prompt text is exported.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = ROOT / 'outputs/hanguard_clean_ablation_20261022'
DEFAULT_OUTPUT = ROOT / 'outputs/hanguard_paper_followups_20261025/external'
DEFAULT_MODEL = ROOT / 'models/chinese-macbert-base'
DEFAULT_ARCHIVE = ROOT / 'experiment_results/20261025_paper_followups/external'
SCHEMA = 'hanguard_external_macbert_document_classifier_v1'
SEEDS = (42, 43, 44)
NATIVE_SOURCES = ('chinese_curated', 'jailbench')
EXPECTED_ROWS = dict(train=17440, validation=2180, test=2180)
SETTINGS = dict(lr=2e-5, weight_decay=.01, warmup_fraction=.1, epochs=8,
    patience=2, effective_batch=16, micro_documents=4, micro_windows=4,
    window_token_budget=2048, window_content_tokens=510, window_overlap=128,
    grad_clip=1., allocator_cap_gib=8., bootstrap_draws=2000,
    parameters='float32', cuda_autocast='bfloat16', gradient_checkpointing=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_frame(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _stat(path):
    item = Path(path).stat()
    return dict(size=item.st_size, mtime_ns=item.st_mtime_ns, ctime_ns=item.st_ctime_ns,
                inode=item.st_ino, device=item.st_dev)


def _runtime():
    return {key: importlib.metadata.version(key) for key in
            ('torch', 'transformers', 'tokenizers', 'pandas', 'pyarrow', 'scikit-learn')}


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def windows(tokens, content_tokens=510, overlap=128):
    """Return full-coverage spans; overlap is a count, not an HF stride step."""
    if content_tokens < 1 or not 0 <= overlap < content_tokens:
        raise ValueError('Invalid full-text window settings')
    output, start = [], 0
    while True:
        end = min(start + content_tokens, len(tokens))
        output.append((start, end))
        if end == len(tokens):
            return output
        start += content_tokens - overlap


def bert_window(tokenizer, tokens):
    """Wrap a raw MacBERT window without relying on a legacy tokenizer API.

    Transformers 5's BertTokenizer has no build_inputs_with_special_tokens.
    The pinned model uses single-sequence BERT CLS/content/SEP semantics;
    an empty tokenization still produces one valid CLS/SEP window.
    """
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    vocabulary_size = len(tokenizer)
    if (any(type(value) is not int or not 0 <= value < vocabulary_size
            for value in (cls_id, sep_id)) or cls_id == sep_id):
        raise ValueError('Invalid MacBERT CLS/SEP token IDs')
    if any(type(value) is not int or not 0 <= value < vocabulary_size for value in tokens):
        raise ValueError('Raw window contains invalid token IDs')
    return [cls_id, *tokens, sep_id]


def _native(frame):
    import numpy as np
    needed = {'base_id', 'group_id', 'source', 'prompt', 'category_id',
              'prompt_harm_label', 'type_label_mask'}
    if not needed <= set(frame) or not frame.base_id.is_unique:
        raise ValueError('Invalid parent raw data identities')
    mask = (frame.source.isin(NATIVE_SOURCES) & frame.type_label_mask.eq(True)
            & frame.prompt_harm_label.eq('harmful') & frame.category_id.isin(range(1, 6)))
    result = frame.loc[mask].sort_values('base_id', kind='stable').reset_index(drop=True)
    if (result.group_id.isna().any() or not result.prompt.map(lambda x: isinstance(x, str) and bool(x)).all()
            or not np.isfinite(result.category_id.to_numpy(dtype=float)).all()):
        raise ValueError('Invalid known native harmful examples')
    return result


def _prepare(output, parent, split, tokenizer, *, protocol_sha=None, selection_sha=None):
    import pandas as pd
    raw_path = parent / 'data' / f'{split}.parquet'
    frame = _native(pd.read_parquet(raw_path))
    expected = EXPECTED_ROWS[split]
    if len(frame) != expected:
        raise ValueError(f'Expected {expected} native {split} documents, found {len(frame)}')
    meta = frame[['base_id', 'group_id', 'source', 'category_id']].copy()
    meta['label'] = meta.category_id.astype('int64') - 1
    docs, lengths, counts = [], [], []
    for text in frame.prompt:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        spans = windows(tokens, SETTINGS['window_content_tokens'], SETTINGS['window_overlap'])
        ids = [bert_window(tokenizer, tokens[start:end]) for start, end in spans]
        if any(not row or len(row) > 512 for row in ids):
            raise ValueError('Tokenizer special-token/window configuration exceeds 512')
        covered = set(index for start, end in spans for index in range(start, end))
        if covered != set(range(len(tokens))):
            raise ValueError('Full document token coverage failed')
        docs.append(dict(input_ids=ids, spans=spans, raw_token_count=len(tokens)))
        lengths.append(len(tokens))
        counts.append(len(ids))
    meta['raw_token_count'], meta['window_count'] = lengths, counts
    folder = output / 'data' / split
    folder.mkdir(parents=True, exist_ok=True)
    save_frame(meta, folder / 'metadata.parquet')
    with (folder / 'windows.jsonl').open('w') as stream:
        for doc in docs:
            stream.write(json.dumps(doc, separators=(',', ':')) + '\n')
    manifest = dict(schema=SCHEMA, split=split, documents=len(meta), source_sha256=sha(raw_path),
        files={name: sha(folder / name) for name in ('metadata.parquet', 'windows.jsonl')},
        protocol_sha256=protocol_sha, selection_lock_sha256=selection_sha,
        truncation=False, covered_raw_tokens=sum(lengths), windows=sum(counts),
        long_documents=sum(n > SETTINGS['window_content_tokens'] for n in lengths),
        empty_token_documents=sum(n == 0 for n in lengths), max_raw_tokens=max(lengths),
        max_windows=max(counts), window_content_tokens=SETTINGS['window_content_tokens'],
        overlap=SETTINGS['window_overlap'], pooling='document mean of all window logits before CE')
    write(folder / 'manifest.json', manifest)
    return manifest


def initialize(output=DEFAULT_OUTPUT, parent=DEFAULT_PARENT, model=DEFAULT_MODEL):
    from transformers import AutoTokenizer
    output, parent, model = map(lambda p: Path(p).resolve(), (output, parent, model))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh external-classifier output directory')
    main, main_protocol = read(parent / 'results.json'), read(parent / 'protocol.json')
    parent_sha = sha(parent / 'protocol.json')
    if (not main.get('complete') or main.get('protocol_sha256') != parent_sha
            or read(parent / 'protocol_registration.json')['protocol_sha256'] != parent_sha
            or main.get('rootselection_lock_sha256') != sha(parent / 'selection_lock.json')):
        raise ValueError('Completed parent study and selections must be frozen')
    data_hashes = {split: sha(parent / 'data' / f'{split}.parquet') for split in ('train', 'validation', 'test')}
    if data_hashes != main_protocol['data_sha256']:
        raise ValueError('Parent split bytes changed')
    config, origin = read(model / 'config.json'), read(model / 'MODEL_ORIGIN.json')
    if config.get('model_type') != 'bert' or config.get('max_position_embeddings') != 512:
        raise ValueError('Expected a 512-position Chinese BERT-base encoder')
    if origin.get('repo_id') != 'hfl/chinese-macbert-base' or len(origin.get('revision', '')) != 40:
        raise ValueError('External model must carry the pinned official MacBERT revision')
    for filename, digest in origin.get('files', {}).items():
        path = (model / filename).resolve()
        if not path.is_relative_to(model) or sha(path) != digest:
            raise ValueError('Local MacBERT bytes differ from the pinned model-origin manifest')
    files = sorted(p for p in model.iterdir() if p.is_file())
    if not (model / 'pytorch_model.bin').exists():
        raise ValueError('Pinned official encoder checkpoint is missing')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError('Expected CLS/SEP token overhead of two')
    inputs = {str(path): dict(sha256=sha(path), stat=_stat(path)) for path in
        [Path(__file__).resolve(), parent / 'protocol.json', parent / 'protocol_registration.json',
         parent / 'selection_lock.json', parent / 'results.json',
         *[parent / 'data' / f'{s}.parquet' for s in data_hashes], *files]}
    output.mkdir(parents=True)
    for name in ('seeds', 'workers', 'source_snapshot'):
        (output / name).mkdir()
    manifests = {split: _prepare(output, parent, split, tokenizer) for split in ('train', 'validation')}
    import pandas as pd
    a = pd.read_parquet(output / 'data/train/metadata.parquet')
    b = pd.read_parquet(output / 'data/validation/metadata.parquet')
    for field in ('base_id', 'group_id'):
        if set(a[field]) & set(b[field]):
            raise ValueError('Native train and validation share a family or row')
    shutil.copyfile(__file__, output / 'source_snapshot/paper_external_classifier.py')
    protocol = dict(schema=SCHEMA, output=str(output), parent=str(parent), model=str(model),
        parent_protocol_sha256=parent_sha, data_sha256=data_hashes, seeds=list(SEEDS),
        settings=SETTINGS, model_origin=origin, runtime=_runtime(), inputs=inputs,
        prepared={split: sha(output / 'data' / split / 'manifest.json') for split in manifests},
        snapshot_sha256=sha(output / 'source_snapshot/paper_external_classifier.py'),
        train_rows=EXPECTED_ROWS['train'], validation_rows=EXPECTED_ROWS['validation'], test_rows=EXPECTED_ROWS['test'],
        task='Native known-harmful five-class labels, IDs 1..5 mapped to 0..4',
        supervision='End-to-end encoder fine-tuning plus a new linear 5-class head',
        loss='Mean window logits per document, then document-equally-weighted softmax CE',
        tokenization='No truncation; 510 content tokens, overlap128, CLS/SEP per window',
        selection='Minimum validation CE; maximum8 epochs, patience2; no test selection',
        test_barrier='All three validation-selected checkpoints locked before test tokenization',
        comparison_note='Same labeled split as hanguard; different encoder and end-to-end training budget, not an internal module ablation',
        historical_experiment_inputs=False, created_unix=time.time())
    for filename, item in inputs.items():
        if _stat(filename) != item['stat']:
            raise ValueError('Input changed during external baseline registration')
    write(output / 'protocol.json', protocol)
    write(output / 'protocol_registration.json', dict(protocol_sha256=sha(output / 'protocol.json')))
    write(output / 'state.json', dict(state='registered', updated_unix=time.time()))
    return progress(output)


def verify(output):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('output') != str(output)
            or protocol.get('settings') != SETTINGS or protocol.get('seeds') != list(SEEDS)
            or protocol.get('runtime') != _runtime()
            or read(output / 'protocol_registration.json')['protocol_sha256'] != sha(output / 'protocol.json')):
        raise ValueError('External baseline protocol changed')
    for filename, item in protocol['inputs'].items():
        if _stat(filename) != item['stat'] or sha(filename) != item['sha256']:
            raise ValueError(f'Registered external baseline input changed: {filename}')
    if sha(output / 'source_snapshot/paper_external_classifier.py') != protocol['snapshot_sha256']:
        raise ValueError('External baseline source snapshot changed')
    for split, digest in protocol['prepared'].items():
        if sha(output / 'data' / split / 'manifest.json') != digest:
            raise ValueError('Prepared training/validation manifest changed')
    return protocol


def _selection(output, seed, check_hash=True):
    folder = output / 'seeds' / f's{seed}'
    path = folder / 'selection.json'
    if not path.exists():
        return None
    item = read(path)
    if item.get('seed') != seed or item.get('protocol_sha256') != sha(output / 'protocol.json'):
        raise ValueError('External validation selection belongs to another run')
    if check_hash:
        for name, digest in item['artifacts'].items():
            if sha(folder / name) != digest:
                raise ValueError('External validation-selected artifact changed')
    return item


def lock_selections(output):
    output = Path(output)
    with exclusive(output / '.selection.lock'):
        selected = {str(seed): _selection(output, seed) for seed in SEEDS}
        if any(value is None for value in selected.values()):
            return False
        expected = dict(complete=True, protocol_sha256=sha(output / 'protocol.json'),
            selections={str(seed): sha(output / 'seeds' / f's{seed}' / 'selection.json') for seed in SEEDS},
            artifacts={str(output / 'seeds' / f's{seed}' / name): digest
                       for seed in SEEDS for name, digest in selected[str(seed)]['artifacts'].items()})
        path = output / 'selection_lock.json'
        if path.exists() and read(path) != expected:
            raise ValueError('External test lock changed')
        if not path.exists():
            write(path, expected)
        return True


def verify_lock(output):
    lock = read(output / 'selection_lock.json')
    if lock.get('complete') is not True or lock['protocol_sha256'] != sha(output / 'protocol.json'):
        raise ValueError('All three external seeds must be selected before test')
    if set(lock['selections']) != {str(seed) for seed in SEEDS}:
        raise ValueError('External test lock is missing a seed')
    for seed in SEEDS:
        if sha(output / 'seeds' / f's{seed}' / 'selection.json') != lock['selections'][str(seed)]:
            raise ValueError('External selected record changed after test lock')
    for filename, digest in lock['artifacts'].items():
        if sha(filename) != digest:
            raise ValueError('External selected weights changed after test lock')
    return lock


def load_data(output, split):
    import pandas as pd
    output = Path(output)
    protocol = read(output / 'protocol.json')
    if split == 'test':
        verify_lock(output)
        with exclusive(output / '.test_preparation.lock'):
            path = output / 'data/test/manifest.json'
            if not path.exists():
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(protocol['model'], local_files_only=True)
                _prepare(output, Path(protocol['parent']), split, tokenizer,
                         protocol_sha=sha(output / 'protocol.json'), selection_sha=sha(output / 'selection_lock.json'))
    folder = output / 'data' / split
    manifest = read(folder / 'manifest.json')
    if manifest['source_sha256'] != protocol['data_sha256'][split]:
        raise ValueError('Prepared raw split identity differs')
    if split == 'test' and (manifest['protocol_sha256'] != sha(output / 'protocol.json')
            or manifest['selection_lock_sha256'] != sha(output / 'selection_lock.json')):
        raise ValueError('Prepared test cache was not bound to frozen selections')
    for filename, digest in manifest['files'].items():
        if sha(folder / filename) != digest:
            raise ValueError('Prepared tokenization or metadata changed')
    meta = pd.read_parquet(folder / 'metadata.parquet')
    docs = [json.loads(line) for line in (folder / 'windows.jsonl').read_text().splitlines()]
    if len(meta) != len(docs) or len(meta) != manifest['documents'] or not meta.base_id.is_unique:
        raise ValueError('Prepared document rows differ')
    return dict(meta=meta, docs=docs, manifest=manifest)


def _admission(output, device):
    if str(device) == 'cpu':
        return
    if str(device) not in ('cuda', 'cuda:0'):
        raise ValueError('Only a single UUID-admitted CUDA device is allowed')
    protocol_sha = sha(output / 'protocol.json')
    if os.environ.get('HANGUARD_FOLLOWUP_ADMITTED') != protocol_sha:
        raise RuntimeError('GPU worker must be admitted by the registered follow-up scheduler')
    path = os.environ.get('HANGUARD_FOLLOWUP_ADMISSION_RECORD')
    if not path:
        raise RuntimeError('Missing GPU admission receipt')
    record = read(path)
    proc = Path('/proc') / str(os.getpid())
    stat = (proc / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    expected = dict(pid=os.getpid(), uid=os.getuid(), start_time_ticks=int(fields[19]),
                    command_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest())
    if (any(record.get(key) != value for key, value in expected.items())
            or record.get('followup_protocol_sha256') != protocol_sha
            or record.get('gpu_uuid') != os.environ.get('CUDA_VISIBLE_DEVICES')
            or not str(record.get('gpu_uuid', '')).startswith('GPU-')
            or ',' in str(record.get('gpu_uuid', ''))
            or record.get('allocator_cap_gib') != SETTINGS['allocator_cap_gib']):
        raise RuntimeError('GPU admission identity, UUID, or allocator budget differs')


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


def save_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(_cpu(value), temporary)
    os.replace(temporary, path)


def _seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)


def _rng(device):
    import numpy as np
    import torch
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(0).cpu() if device != 'cpu' else None)


def _restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype='uint32'), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'])
    if device != 'cpu':
        torch.cuda.set_rng_state(state['torch_cuda'])


def _load_model(model_path, seed, device):
    import torch
    from transformers import AutoModelForSequenceClassification
    _seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True,
        num_labels=5, dtype=torch.float32, attn_implementation='sdpa')
    if SETTINGS['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model.to(device)


def _autocast(device):
    import torch
    return nullcontext() if device == 'cpu' else torch.autocast('cuda', dtype=torch.bfloat16)


def document_logits(model, documents, device, window_micro=4, pad_token_id=0):
    """Every window contributes equally to its document's mean logits.

    Encoder gradient checkpointing lets a long singleton document retain all
    window graphs for one document CE. There is no independently labeled-window
    loss and no over-weighting of long documents.
    """
    import torch
    flat = [row for doc in documents for row in doc['input_ids']]
    if not documents or any(not doc['input_ids'] for doc in documents):
        raise ValueError('Nonempty windowed documents required')
    values = []
    for start in range(0, len(flat), window_micro):
        batch = flat[start:start + window_micro]
        length = max(map(len, batch))
        ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(batch):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[index, :len(row)] = 1
        with _autocast(device):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        if logits.shape != (len(batch), 5) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Invalid external classifier logits')
        values.extend(logits.unbind(0))
    pooled, offset = [], 0
    for doc in documents:
        count = len(doc['input_ids'])
        pooled.append(torch.stack(values[offset:offset + count]).mean(0))
        offset += count
    return torch.stack(pooled)


def document_groups(indices, data, window_limit=4):
    current, count, token_count = [], 0, 0
    for index in indices:
        doc_windows = data['docs'][int(index)]['input_ids']
        windows_count = len(doc_windows)
        tokens = sum(map(len, doc_windows))
        if current and (len(current) >= SETTINGS['micro_documents'] or count + windows_count > window_limit
                        or token_count + tokens > SETTINGS['window_token_budget']):
            yield current
            current, count, token_count = [], 0, 0
        current.append(int(index))
        count += windows_count
        token_count += tokens
    if current:
        yield current


def _train_attempt(model, optimizer, data, indices, device, *, seed, epoch, step, window_limit):
    import torch
    from torch.nn import functional as F
    optimizer.zero_grad(set_to_none=True)
    model.train()
    total = 0.
    for micro, group in enumerate(document_groups(indices, data, window_limit)):
        _seed((seed * 1000003 + epoch * 1009 + step * 97 + micro) % (2**63 - 1))
        logits = document_logits(model, [data['docs'][i] for i in group], device, window_limit)
        target = torch.tensor(data['meta'].label.iloc[group].to_numpy(), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target, reduction='sum') / len(indices)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite document classification loss')
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['grad_clip'], error_if_nonfinite=True)
    return total


def train_macro(model, optimizer, data, indices, device, *, seed, epoch, step, window_limit, events):
    import torch
    while True:
        try:
            loss = _train_attempt(model, optimizer, data, indices, device,
                seed=seed, epoch=epoch, step=step, window_limit=window_limit)
            break
        except torch.OutOfMemoryError:
            pass
        # The failed attempt's frame has exited; clear all partial gradients
        # and replay the entire macro before any optimizer/scheduler step.
        optimizer.zero_grad(set_to_none=True)
        if device != 'cpu':
            torch.cuda.empty_cache()
        events.append(dict(epoch=epoch, step=step, window_limit=window_limit,
                           action='replay_entire_document_macro_after_oom'))
        if window_limit == 1:
            raise RuntimeError('Full singleton document cannot fit even with encoder checkpointing; never truncate')
        window_limit = max(1, window_limit // 2)
    # Never retry an optimizer OOM: an optimizer could already have updated a
    # subset of its parameters/state. Only forward/backward OOMs are replayed.
    optimizer.step()
    return loss, window_limit


def predict(model, data, device, window_limit=4):
    import numpy as np
    import torch
    model.eval()
    output = np.empty((len(data['meta']), 5), dtype='float32')
    order = sorted(range(len(output)), key=lambda i: (len(data['docs'][i]['input_ids']), data['meta'].raw_token_count.iloc[i], i))
    with torch.inference_mode():
        for group in document_groups(order, data, window_limit):
            limit = window_limit
            while True:
                try:
                    values = document_logits(model, [data['docs'][i] for i in group], device, limit)
                    output[group] = values.float().cpu().numpy()
                    break
                except torch.OutOfMemoryError:
                    pass
                if device != 'cpu':
                    torch.cuda.empty_cache()
                if limit == 1:
                    raise RuntimeError('Full validation/test document cannot fit; never truncate')
                limit = max(1, limit // 2)
    return output


def _scheduler(optimizer, total_steps):
    import torch
    warmup = max(1, int(total_steps * SETTINGS['warmup_fraction']))
    def factor(step):
        return step / warmup if step < warmup else max(0., (total_steps - step) / max(1, total_steps - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _prediction_frame(data, logits):
    import numpy as np
    probs = np.exp(logits.astype('float64') - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    frame = data['meta'].copy()
    for index in range(5):
        frame[f'p{index}'] = probs[:, index]
    frame['prediction'] = probs.argmax(1) + 1
    return frame


def fit(output=DEFAULT_OUTPUT, seed=42, device='cpu'):
    import numpy as np
    output = Path(output).resolve()
    protocol = verify(output)
    if seed not in SEEDS:
        raise ValueError('Unregistered external seed')
    _admission(output, device)
    folder = output / 'seeds' / f's{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive(folder / '.worker.lock'):
        if (folder / 'test_complete.json').exists():
            return read(folder / 'test_complete.json')
        if _selection(output, seed) is not None:
            if not lock_selections(output):
                return dict(state='waiting_for_other_selections', seed=seed)
            return evaluate(output, seed, device)
        if (output / 'selection_lock.json').exists():
            raise RuntimeError('Cannot fit after test selections have been locked')
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(4)
        train, validation = load_data(output, 'train'), load_data(output, 'validation')
        model = _load_model(protocol['model'], seed, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'], weight_decay=SETTINGS['weight_decay'])
        steps = math.ceil(len(train['meta']) / SETTINGS['effective_batch'])
        scheduler = _scheduler(optimizer, steps * SETTINGS['epochs'])
        identity = dict(protocol_sha256=sha(output / 'protocol.json'), seed=seed, device_type=device,
                        train_manifest_sha256=sha(output / 'data/train/manifest.json'),
                        validation_manifest_sha256=sha(output / 'data/validation/manifest.json'))
        completed, best_ce, best_epoch, stale, history, events = 0, math.inf, 0, 0, [], []
        best_state, window_limit = None, SETTINGS['micro_windows']
        resume = folder / 'epoch_resume.pt'
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('External resume data, protocol, seed or device differs')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            completed, best_ce, best_epoch, stale = saved['completed'], saved['best_ce'], saved['best_epoch'], saved['stale']
            best_state, history, events, window_limit = saved['best_state'], saved['history'], saved['oom_events'], saved['window_limit']
            if (not 1 <= completed <= SETTINGS['epochs'] or len(history) != completed
                    or [item['epoch'] for item in history] != list(range(1, completed + 1))
                    or not 1 <= best_epoch <= completed or not math.isfinite(best_ce)
                    or stale < 0 or window_limit not in (1, 2, 4)):
                raise ValueError('External epoch-resume history or budget is invalid')
            _restore_rng(saved['rng'], device)
        started = time.time()
        while completed < SETTINGS['epochs'] and stale < SETTINGS['patience']:
            if (output / 'STOP').exists():
                raise RuntimeError('External baseline STOP requested')
            order = np.random.default_rng(seed * 1009 + completed).permutation(len(train['meta']))
            losses = []
            for step, start in enumerate(range(0, len(order), SETTINGS['effective_batch'])):
                indices = order[start:start + SETTINGS['effective_batch']].tolist()
                loss, window_limit = train_macro(model, optimizer, train, indices, device,
                    seed=seed, epoch=completed, step=step, window_limit=window_limit, events=events)
                scheduler.step()
                losses.append((loss, len(indices)))
                if step % 25 == 0:
                    write(folder / 'progress.json', dict(state='training', seed=seed, device=device,
                        completed_epochs=completed, current_epoch=completed + 1, budget=SETTINGS['epochs'],
                        macro_step=step + 1, macro_steps=steps, seconds=time.time() - started,
                        window_limit=window_limit, updated_unix=time.time()))
            logits = predict(model, validation, device, window_limit)
            target = torch.tensor(validation['meta'].label.to_numpy(), dtype=torch.long)
            ce = float(F.cross_entropy(torch.from_numpy(logits).double(), target))
            completed += 1
            if ce < best_ce:
                best_ce, best_epoch, stale, best_state = ce, completed, 0, _cpu(model.state_dict())
            else:
                stale += 1
            history.append(dict(epoch=completed,
                training_ce=float(sum(value * count for value, count in losses) / len(train['meta'])), validation_ce=ce))
            save_torch(resume, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), completed=completed, best_ce=best_ce,
                best_epoch=best_epoch, stale=stale, best_state=best_state, history=history,
                oom_events=events, window_limit=window_limit, rng=_rng(device)))
            write(folder / 'history.json', history)
        if best_state is None:
            raise RuntimeError('No finite validation-selected external checkpoint')
        model.load_state_dict(best_state)
        validation_logits = predict(model, validation, device, window_limit)
        save_torch(folder / 'selected_model.pt', dict(protocol_sha256=identity['protocol_sha256'],
            seed=seed, state=best_state, num_labels=5, window_limit=window_limit))
        save_frame(_prediction_frame(validation, validation_logits), folder / 'validation_predictions.parquet')
        selected = dict(schema=SCHEMA, seed=seed, protocol_sha256=identity['protocol_sha256'],
            best_epoch=best_epoch, validation_ce=best_ce, completed_epochs=completed,
            selection_scope='validation_only', formal_test_opened=False,
            parameter_count=sum(p.numel() for p in model.parameters()),
            artifacts={name: sha(folder / name) for name in ('selected_model.pt', 'validation_predictions.parquet')},
            device=device, window_limit=window_limit, oom_events=events)
        write(folder / 'selection.json', selected)
        write(folder / 'progress.json', dict(state='selected', seed=seed, completed_epochs=completed,
                                            best_epoch=best_epoch, device=device, updated_unix=time.time()))
        try:
            lock_selections(output)
        except BlockingIOError:
            pass
        return selected


def metrics(labels, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions,
        labels=list(range(1, 6)), zero_division=0)
    return dict(n=len(labels), accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1.mean()), confusion_matrix=confusion_matrix(labels, predictions, labels=range(1, 6)).tolist(),
        per_class={str(i + 1): dict(precision=float(precision[i]), recall=float(recall[i]),
            f1=float(f1[i]), support=int(support[i])) for i in range(5)})


def _bootstrap(y, candidate, baseline, groups, seed):
    import numpy as np
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    delta = (np.asarray(candidate) == y).astype(float) - (np.asarray(baseline) == y).astype(float)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(SETTINGS['bootstrap_draws']):
        sampled = rng.integers(0, len(counts), len(counts))
        values.append(float(sums[sampled].sum() / counts[sampled].sum()))
    return dict(delta=float(delta.mean()), ci95=np.quantile(values, [.025, .975]).tolist(),
                unit='group_id cluster', groups=len(counts), repetitions=SETTINGS['bootstrap_draws'])


def evaluate(output, seed, device='cpu'):
    import pandas as pd
    import torch
    output = Path(output)
    protocol = verify(output)
    verify_lock(output)
    selected = _selection(output, seed)
    folder = output / 'seeds' / f's{seed}'
    data = load_data(output, 'test')
    saved = torch.load(folder / 'selected_model.pt', map_location='cpu', weights_only=True)
    if saved['protocol_sha256'] != sha(output / 'protocol.json') or saved['seed'] != seed:
        raise ValueError('Selected external checkpoint identity differs')
    model = _load_model(protocol['model'], seed, device)
    model.load_state_dict(saved['state'])
    logits = predict(model, data, device, saved['window_limit'])
    frame = _prediction_frame(data, logits)
    save_frame(frame, folder / 'test_predictions.parquet')
    y = frame.category_id.to_numpy()
    parent_prediction_path = Path(protocol['parent']) / 'heads/category' / f's{seed}' / 'test_predictions.parquet'
    parent_main = read(Path(protocol['parent']) / 'results.json')
    if sha(parent_prediction_path) != parent_main['per_seed'][str(seed)]['category']['predictions_sha256']:
        raise ValueError('Parent category test predictions changed')
    parent_predictions = pd.read_parquet(parent_prediction_path)
    parent = frame[['base_id', 'category_id']].merge(parent_predictions, on='base_id', how='left', validate='one_to_one', suffixes=('', '_parent'))
    if len(parent) != len(frame) or parent.category_id_parent.isna().any() or not parent.category_id.eq(parent.category_id_parent).all():
        raise ValueError('Parent and external five-class test identities differ')
    paired = {}
    for arm in ('category_only', 'fusion_category'):
        pred = parent[[f'{arm}_p{k}' for k in range(5)]].to_numpy().argmax(1) + 1
        paired[arm] = _bootstrap(y, frame.prediction.to_numpy(), pred, frame.group_id, seed)
    result = dict(schema=SCHEMA, seed=seed, complete=True, protocol_sha256=sha(output / 'protocol.json'),
        selection_lock_sha256=sha(output / 'selection_lock.json'), rows=len(frame),
        metrics=metrics(y, frame.prediction),
        by_source={str(source): metrics(group.category_id.to_numpy(), group.prediction.to_numpy())
                   for source, group in frame.groupby('source', sort=True)},
        paired_parent_accuracy=paired, test_window_stats=data['manifest'],
        checkpoint_parameter_count=selected['parameter_count'])
    write(folder / 'test_results.json', result)
    write(folder / 'test_complete.json', dict(complete=True, seed=seed,
        protocol_sha256=sha(output / 'protocol.json'), artifacts={name: sha(folder / name) for name in
        ('test_predictions.parquet', 'test_results.json')}))
    write(folder / 'progress.json', dict(state='evaluated', seed=seed, device=device, updated_unix=time.time()))
    if all((output / 'seeds' / f's{s}' / 'test_complete.json').exists() for s in SEEDS):
        try:
            with exclusive(output / '.summary.lock'):
                summarize(output)
        except BlockingIOError:
            pass
    return result


def progress(output=DEFAULT_OUTPUT):
    output = Path(output)
    if not (output / 'protocol.json').exists():
        return dict(state='not_initialized', pending_jobs=[], completed_jobs=0, total_jobs=6)
    selected = [seed for seed in SEEDS if _selection(output, seed, check_hash=False) is not None]
    tested = [seed for seed in SEEDS if (output / 'seeds' / f's{seed}' / 'test_complete.json').exists()]
    if len(selected) < len(SEEDS):
        pending = [dict(seed=seed, phase='fit') for seed in SEEDS if seed not in selected]
        phase = 'fit'
    else:
        pending = [dict(seed=seed, phase='evaluate') for seed in SEEDS if seed not in tested]
        phase = 'evaluate' if pending else 'complete'
    return dict(state=phase, pending_jobs=pending, completed_jobs=len(selected) + len(tested),
                total_jobs=6, selected_seeds=selected, evaluated_seeds=tested)


def is_complete(output=DEFAULT_OUTPUT):
    path = Path(output) / 'results.json'
    return path.exists() and read(path).get('complete') is True and not progress(output)['pending_jobs']


def summarize(output=DEFAULT_OUTPUT):
    import numpy as np
    output = Path(output).resolve()
    protocol = verify(output)
    verify_lock(output)
    results = {}
    for seed in SEEDS:
        folder = output / 'seeds' / f's{seed}'
        receipt = read(folder / 'test_complete.json')
        if receipt['protocol_sha256'] != sha(output / 'protocol.json'):
            raise ValueError('External test receipt protocol differs')
        for name, digest in receipt['artifacts'].items():
            if sha(folder / name) != digest:
                raise ValueError('External completed test artifact changed')
        results[str(seed)] = read(folder / 'test_results.json')
    aggregate = {}
    for name in ('accuracy', 'macro_f1'):
        values = [results[str(seed)]['metrics'][name] for seed in SEEDS]
        aggregate[name] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), values=values)
    result = dict(schema=SCHEMA, complete=True, protocol_sha256=sha(output / 'protocol.json'),
        model_origin=protocol['model_origin'], rows=2180, seeds=list(SEEDS), per_seed=results,
        aggregate=aggregate, comparison_note=protocol['comparison_note'])
    write(output / 'results.json', result)
    lines = ['# 同划分五分类外部基线：Chinese-MacBERT-base', '',
        '三 seed 已完成；原生两来源已知有害测试样本共 2,180 条。', '',
        '| 指标 | 三 seed 均值 ± 样本标准差 |', '|---|---:|']
    for name, item in aggregate.items():
        lines.append(f"| {name} | {100 * item['mean']:.2f} ± {100 * item['sample_std']:.2f}% |")
    lines += ['', '使用官方 HFL Chinese-MacBERT-base 固定 revision，FP32 参数、CUDA BF16 autocast，端到端微调。',
        '所有原文保留；长文采用 510 内容 token 窗口、128 token 重叠，窗口 logits 均值后按文档计算 CE。',
        '训练/验证/测试与主研究完全相同；三 seed 最小验证 CE 检查点锁定后才处理测试文本。',
        '该对比采用不同模型与训练预算，不是冻结骨干模块的容量匹配消融。', '',
        '官方来源：https://huggingface.co/hfl/chinese-macbert-base；许可证 Apache-2.0。', '']
    (output / 'README.md').write_text('\n'.join(lines))
    archive = DEFAULT_ARCHIVE if output == DEFAULT_OUTPUT.resolve() else output / 'archive'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('protocol.json', 'protocol_registration.json', 'selection_lock.json', 'results.json', 'README.md'):
        shutil.copyfile(output / name, archive / name)
    shutil.copytree(output / 'source_snapshot', archive / 'source_snapshot', dirs_exist_ok=True)
    for seed in SEEDS:
        target = archive / f's{seed}'
        target.mkdir(exist_ok=True)
        for name in ('selection.json', 'test_results.json', 'test_complete.json', 'test_predictions.parquet', 'history.json'):
            shutil.copyfile(output / 'seeds' / f's{seed}' / name, target / name)
    manifest = {str(path.relative_to(archive)): sha(path) for path in sorted(archive.rglob('*'))
                if path.is_file() and path.name != 'manifest.json'}
    write(archive / 'manifest.json', dict(files=manifest, source=str(output), schema=SCHEMA))
    write(output / 'state.json', dict(state='complete', archive=str(archive), updated_unix=time.time()))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('init', 'worker', 'status', 'summarize'):
        child = sub.add_parser(action)
        child.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
        if action == 'init':
            child.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
            child.add_argument('--model', type=Path, default=DEFAULT_MODEL)
        elif action == 'worker':
            child.add_argument('--seed', type=int, required=True)
            child.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    if args.action == 'init':
        result = initialize(args.output, args.parent, args.model)
    elif args.action == 'worker':
        try:
            result = fit(args.output, args.seed, args.device)
        except BlockingIOError:
            return 75
        except Exception as exc:
            write(args.output / 'workers' / f's{args.seed}.error.json', dict(error=repr(exc), updated_unix=time.time()))
            raise
    elif args.action == 'summarize':
        result = summarize(args.output)
    else:
        result = progress(args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
