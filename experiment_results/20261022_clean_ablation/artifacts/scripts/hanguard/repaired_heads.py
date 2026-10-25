"""Fresh, shared readouts and online ShieldHead adaptation for repaired data.

This module loads no model, data, checkpoint or feature cache. Raw block states
are tapped at layers 8/16/24/32 (one-based); every route has one classifier.
ShieldHead prototypes are training-only, detached teachers, not output heads.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LAYERS = (8, 16, 24, 32)
MODES = ('last', 'fusion', 'repeat_last', 'mean', 'attention', 'fusion_attention')


def _fp32(tensor):
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def _zero_linear(width):
    result = nn.Linear(width, width)
    nn.init.zeros_(result.weight)
    nn.init.zeros_(result.bias)
    return result


def _last_indices(mask):
    if mask.ndim != 2 or not mask.bool().any(1).all():
        raise ValueError('Every sentence must have at least one valid token')
    positions = torch.arange(mask.shape[1], device=mask.device)
    return positions[None].expand_as(mask).masked_fill(~mask.bool(), -1).max(1).values


class ExperimentHead(nn.Module):
    """Width-128 MLP, optional residual pooling, and one final classifier.

    ``forward`` accepts {'layers': tuple(raw[B,T,D]), 'mask': bool[B,T]}.
    Sentence-only arms only project the last position. Token arms project in
    chunks, avoiding a stacked [B,4,T,D] float32 feature tensor.
    """
    def __init__(self, hidden_size, mode='last', readout='mlp', width=128,
                 layers=LAYERS, dropout=.1, chunk_size=512, prototypes_per_class=8):
        super().__init__()
        if mode not in MODES or readout not in ('mlp', 'prototype'):
            raise ValueError(f'Unsupported mode/readout: {mode}/{readout}')
        if tuple(sorted(set(layers))) != tuple(layers) or not layers:
            raise ValueError('layers must be nonempty and strictly increasing')
        if chunk_size < 1 or prototypes_per_class < 1:
            raise ValueError('chunk_size and prototypes_per_class must be positive')
        self.hidden_size, self.width = hidden_size, width
        self.mode, self.readout, self.layers = mode, readout, tuple(layers)
        self.chunk_size = chunk_size
        self.has_fusion = mode in ('fusion', 'repeat_last', 'fusion_attention')
        self.layers_for_capture = self.layers if self.has_fusion and mode != 'repeat_last' else (self.layers[-1],)
        self.project = nn.Linear(hidden_size, width)
        self.dropout = nn.Dropout(dropout)
        if readout == 'mlp':
            self.classifier = nn.Linear(width, 1)
        else:
            self.prototypes = nn.Parameter(F.normalize(torch.randn(2, prototypes_per_class, width), dim=-1))
            self.prototype_log_scale = nn.Parameter(torch.tensor(math.log(10.)))
            self.prototype_bias = nn.Parameter(torch.zeros(()))
            self.register_buffer('prototype_temperature', torch.tensor(.1))
        if self.has_fusion:
            self.layer_logits = nn.Parameter(torch.zeros(len(layers)))
            self.layer_values = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
                for _ in layers])
            self.fusion_output = _zero_linear(width)
        if mode in ('mean', 'attention'):
            self.token_project = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
        if mode in ('mean', 'attention', 'fusion_attention'):
            self.pool_output = _zero_linear(width)
        if mode in ('attention', 'fusion_attention'):
            self.attention_tanh = nn.Linear(width, width)
            self.attention_gate = nn.Linear(width, width)
            self.attention_context = nn.Linear(width, 1, bias=False)

    def _project_raw(self, raw, module, normalized=True):
        shape = raw.shape[:-1]
        flattened = raw.reshape(-1, raw.shape[-1])
        chunks = []
        for value in flattened.split(self.chunk_size):
            value = value.float()
            if normalized:
                value = F.layer_norm(value, (value.shape[-1],))
            chunks.append(module(value))
        return torch.cat(chunks).reshape(*shape, self.width)

    def _fuse(self, raw):
        z = F.gelu(self._project_raw(raw[-1], self.project))
        if self.has_fusion:
            source = [raw[-1]] * len(self.layers) if self.mode == 'repeat_last' else raw
            if len(source) != len(self.layers):
                raise ValueError('Captured layer count does not match configured fusion')
            weights = self.layer_logits.softmax(0)
            value = sum(weights[i] * self._project_raw(x, module)
                        for i, (x, module) in enumerate(zip(source, self.layer_values)))
            z = z + self.fusion_output(value)
        return z

    def classify_representation(self, z, dropout=True):
        with _fp32(z):
            if self.readout == 'prototype':
                similarities = torch.einsum('...d,ckd->...ck', F.normalize(z.float(), dim=-1),
                                            F.normalize(self.prototypes, dim=-1))
                weights = (similarities / self.prototype_temperature).softmax(-1)
                scores = (weights * similarities).sum(-1)
                return self.prototype_log_scale.exp() * (scores[..., 1] - scores[..., 0]) + self.prototype_bias
            return self.classifier(self.dropout(z.float()) if dropout else z.float()).squeeze(-1)

    def forward(self, features, return_tokens=False):
        raw = tuple(features['layers'])
        mask = features['mask'].bool()
        if not raw or any(x.shape[:2] != mask.shape or x.shape[-1] != self.hidden_size for x in raw):
            raise ValueError('Raw layer features must align with [batch, tokens] mask and hidden_size')
        last = _last_indices(mask)
        rows = torch.arange(len(mask), device=mask.device)
        with _fp32(raw[-1]):
            token_z = None
            if return_tokens or self.mode == 'fusion_attention':
                token_z = self._fuse(raw)
                z = token_z[rows, last]
            else:
                z = self._fuse(tuple(x[rows, last] for x in raw))
            result = {}
            if self.mode == 'mean':
                # Sum in FP32 without materializing [B,T,D] FP32.
                total = raw[-1].new_zeros((len(mask), self.hidden_size), dtype=torch.float32)
                for begin in range(0, mask.shape[1], self.chunk_size):
                    end = begin + self.chunk_size
                    total = total + (raw[-1][:, begin:end].float() * mask[:, begin:end, None]).sum(1)
                pooled = self._project_raw(total / mask.sum(1)[:, None], self.token_project)
                z = z + self.pool_output(pooled)
            elif self.mode in ('attention', 'fusion_attention'):
                values = token_z if self.mode == 'fusion_attention' else self._project_raw(raw[-1], self.token_project)
                gate = torch.tanh(self.attention_tanh(values)) * torch.sigmoid(self.attention_gate(values))
                scores = self.attention_context(gate).squeeze(-1).masked_fill(~mask, -torch.inf)
                attention = scores.softmax(-1)
                z = z + self.pool_output((attention[..., None] * values).sum(1))
                result['attention_weights'] = attention
            result.update(logits=self.classify_representation(z), representation=z)
            if return_tokens:
                result['token_logits'] = self.classify_representation(token_z)
                with torch.no_grad():
                    result['teacher_probabilities'] = self.classify_representation(token_z.detach(), dropout=False).sigmoid()
                result['prototype_features'] = tuple(x.detach() for x in raw)
                result['mask'] = mask
            return result

    @torch.no_grad()
    def initialize_prototypes(self, z, y, seed=42):
        from sklearn.cluster import KMeans
        if self.readout != 'prototype':
            raise ValueError('This head does not use prototype classification')
        z = torch.as_tensor(z).detach().float().cpu()
        y = torch.as_tensor(y).detach().cpu()
        if z.ndim != 2 or z.shape[1] != self.width or y.shape != (len(z),):
            raise ValueError('Expected training representations [N,width] and labels [N]')
        if not torch.isfinite(z).all() or not ((y == 0) | (y == 1)).all():
            raise ValueError('Representations must be finite and labels binary')
        points = F.normalize(z, dim=-1).numpy()
        centers = []
        for label in (0, 1):
            selected = points[y.numpy() == label]
            if len(selected) < self.prototypes.shape[1]:
                raise ValueError('Too few training samples per prototype class')
            fit = KMeans(n_clusters=self.prototypes.shape[1], random_state=seed, n_init=10).fit(selected)
            centers.append(torch.as_tensor(fit.cluster_centers_))
        self.prototypes.copy_(F.normalize(torch.stack(centers), dim=-1).to(self.prototypes))

    def prototype_regularization(self):
        if self.readout != 'prototype':
            return self.project.weight.sum() * 0.
        centers = F.normalize(self.prototypes, dim=-1)
        pair = centers @ centers.transpose(-1, -2)
        off_diagonal = ~torch.eye(pair.shape[-1], device=pair.device, dtype=torch.bool)
        if not off_diagonal.any():
            return self.prototypes.sum() * 0.
        return F.relu(pair[:, off_diagonal] - .5).square().mean()


class TapOnline:
    """Capture raw block outputs online; preserve the full LoRA gradient graph."""
    def __init__(self, model, head):
        self.model, self.head = model, head
        root = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.backbone = root.model.language_model
        self.captured, self.active = {}, False
        self.last_forward_diagnostics = {}
        self.handles = [self.backbone.layers[layer - 1].register_forward_hook(self._hook(layer))
                        for layer in head.layers_for_capture]

    def _hook(self, layer):
        def capture(module, inputs, output):
            if self.active:
                self.captured[layer] = output[0] if isinstance(output, tuple) else output
        return capture

    def __call__(self, batch, return_tokens=False):
        self.captured.clear()
        mask = batch['attention_mask']
        self.active = True
        try:
            # PEFT's input-requires-grad hook can remain installed while all
            # adapters are frozen. In eval-mode head warmup, checkpointing is
            # inactive, so relying only on parameter.requires_grad would retain
            # an unnecessary full backbone graph. Disable recording explicitly
            # for that phase, while preserving the caller's outer no_grad mode.
            backbone_grad = torch.is_grad_enabled() and any(
                parameter.requires_grad for parameter in self.backbone.parameters())
            with torch.set_grad_enabled(backbone_grad):
                self.backbone(input_ids=batch['input_ids'], attention_mask=mask,
                              position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
        finally:
            self.active = False
        raw = tuple(self.captured[i] for i in self.head.layers_for_capture)
        self.captured.clear()
        self.last_forward_diagnostics = {
            'backbone_grad_enabled': bool(backbone_grad),
            'captured_requires_grad': [value.requires_grad for value in raw],
            'captured_has_grad_fn': [value.grad_fn is not None for value in raw],
        }
        # The head must remain outside the frozen-backbone no_grad context.
        return self.head({'layers': raw, 'mask': mask}, return_tokens=return_tokens)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()


def supervised_contrastive_loss(z, labels, temperature=.1):
    """Supervised contrastive loss on the actual supplied batch, no projection.

    Gradient accumulation is not a larger contrastive batch. The caller must
    supply the complete chosen contrastive batch (or use exact gradient cache).
    Anchors with no other same-class example are excluded, not assigned NaN.
    """
    if temperature <= 0 or z.ndim != 2 or labels.shape != (len(z),):
        raise ValueError('Expected representations [N,D], labels [N], positive temperature')
    if len(z) < 2:
        return z.sum() * 0.
    with _fp32(z):
        normalized = F.normalize(z.float(), dim=-1)
        scores = normalized @ normalized.T / temperature
        diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~diagonal
        counts = positive.sum(1)
        valid = counts > 0
        if not valid.any():
            return z.sum() * 0.
        log_probabilities = scores - scores.masked_fill(diagonal, -torch.inf).logsumexp(1)[:, None]
        return -(log_probabilities.masked_fill(~positive, 0.).sum(1)[valid] / counts[valid]).mean()


class ShieldState:
    """Persistent CPU per-sample/per-token labels and macro-batch prototypes.

    Raw features are never differentiated. Each layer has its own two raw-space
    centers and targets, while one shared prediction receives their mean target.
    Call loss once per microbatch and end_macro once after all accumulation.
    The valid final token is excluded from both auxiliary supervision and teacher
    candidates; it retains the true sentence label. Validation/test IDs are
    rejected and cannot update training state. OOM retries must restore a saved
    macro-boundary checkpoint because loss updates persistent targets eagerly.
    """
    def __init__(self, sample_ids, lengths, labels, hidden_size, layer_count=1,
                 mode='dynamic', device='cpu', warmup_steps=1, ramp_steps=1,
                 topk=32, chunk_size=512):
        if mode not in ('fixed', 'dynamic'):
            raise ValueError('mode must be fixed or dynamic')
        self.ids = tuple(str(x) for x in sample_ids)
        self.index = {value: i for i, value in enumerate(self.ids)}
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        if len(self.index) != len(self.ids) or len(self.lengths) != len(self.ids) or self.labels.shape != self.lengths.shape:
            raise ValueError('Unique sample IDs, lengths and labels must align')
        if (self.lengths < 1).any() or not np.isin(self.labels, (0., 1.)).all():
            raise ValueError('Positive lengths and binary labels are required')
        if layer_count < 1 or topk < 1 or chunk_size < 1:
            raise ValueError('layer_count, topk and chunk_size must be positive')
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)))
        initial = np.repeat(self.labels, self.lengths)
        self.targets = np.repeat(initial[None], layer_count, axis=0)
        self.initial_weight = np.ones_like(self.targets)
        self.centers = torch.zeros(layer_count, 2, hidden_size, device=device)
        self.initialized = np.zeros(layer_count, dtype=bool)
        self.pending = [None] * layer_count
        self.mode, self.topk, self.chunk_size = mode, topk, chunk_size
        self.warmup_steps, self.ramp_steps = max(0, warmup_steps), max(1, ramp_steps)
        self.total_visits, self.disambiguation_visits = 0, 0
        self.last_drift = 0.

    def schedule(self, global_step):
        progress = min(1., max(0., (global_step - self.warmup_steps) / self.ramp_steps))
        return global_step >= self.warmup_steps, .99 - .04 * progress, .98 - .48 * progress

    @torch.no_grad()
    def _collect(self, layer, raw, probability):
        confidence = torch.stack((1. - probability.detach(), probability.detach()))
        values, selected = confidence.topk(min(self.topk, len(raw)), dim=1, sorted=True)
        chosen = raw.detach()[selected].float()
        previous = self.pending[layer]
        if previous is not None:
            values = torch.cat((previous[0], values), dim=1)
            chosen = torch.cat((previous[1], chosen), dim=1)
            values, selected = values.topk(min(self.topk, values.shape[1]), dim=1, sorted=True)
            chosen = chosen[torch.arange(2, device=chosen.device)[:, None], selected]
        self.pending[layer] = values, chosen

    def loss(self, token_logits, raw_features, ids, mask, global_step, confidence=None):
        if token_logits.shape != mask.shape or len(ids) != len(mask):
            raise ValueError('IDs, mask and shared token logits must align')
        if len(raw_features) != len(self.centers):
            raise ValueError('One raw feature tensor is required per prototype layer')
        if any(x.shape[:2] != mask.shape or x.shape[-1] != self.centers.shape[-1] for x in raw_features):
            raise ValueError('Raw features must align with mask and prototype dimension')
        mask = mask.bool()
        _last_indices(mask)
        probability = token_logits.detach().sigmoid() if confidence is None else confidence.detach()
        if probability.shape != mask.shape:
            raise ValueError('Shared classifier confidence must align with mask')
        active, gamma, sigma = self.schedule(global_step)
        losses, disagreement, flips, updates, visits = [], 0., 0, 0, 0
        for row, sample_id in enumerate(ids):
            sample = self.index[str(sample_id)]  # Unknown validation/test IDs fail closed.
            positions = mask[row].nonzero(as_tuple=True)[0]
            if len(positions) != self.lengths[sample]:
                raise ValueError(f'Full token length changed for sample {sample_id}')
            begin, end = self.offsets[sample:sample + 2]
            # The final valid position keeps exclusive true sentence supervision.
            # Both fixed/dynamic auxiliaries and prototype candidates use prefixes.
            positions = positions[:-1]
            end -= 1
            visits += len(positions)
            if not len(positions):
                losses.append(token_logits[row].sum() * 0.)
                continue
            target_layers = []
            for layer, raw in enumerate(raw_features):
                old = self.targets[layer, begin:end]
                target = torch.as_tensor(old.copy(), device=token_logits.device)
                for offset in range(0, len(positions), self.chunk_size):
                    selected = positions[offset:offset + self.chunk_size]
                    hidden = raw[row, selected].detach().float()
                    with torch.no_grad(), _fp32(hidden):
                        if self.mode == 'dynamic':
                            self._collect(layer, hidden, probability[row, selected])
                            if active and self.initialized[layer]:
                                revised = (hidden @ self.centers[layer].T).softmax(-1)[:, 1]
                                before = target[offset:offset + len(selected)]
                                target[offset:offset + len(selected)] = sigma * before + (1. - sigma) * revised
                if self.mode == 'dynamic' and active and self.initialized[layer]:
                    updated = target.detach().cpu().numpy()
                    flips += int(np.sum((old >= .5) != (updated >= .5)))
                    updates += len(updated)
                    self.targets[layer, begin:end] = updated
                    self.initial_weight[layer, begin:end] *= sigma
                target_layers.append(target.detach())
            stack = torch.stack(target_layers)
            target = stack.mean(0)
            disagreement += float(stack.var(0, unbiased=False).sum())
            losses.append(F.binary_cross_entropy_with_logits(token_logits[row, positions].float(), target, reduction='mean'))
        self.total_visits += visits
        self.disambiguation_visits += updates
        stats = {'active': bool(active and self.mode == 'dynamic'), 'gamma': gamma, 'sigma': sigma,
                 'positions': visits, 'updated_layer_positions': updates,
                 'update_flip_rate': flips / max(1, updates),
                 'layer_target_variance': disagreement / max(1, visits)}
        return torch.stack(losses).mean(), stats

    @torch.no_grad()
    def end_macro(self, global_step):
        _, gamma, _ = self.schedule(global_step)
        before = self.centers.clone()
        if self.mode == 'dynamic':
            for layer, pending in enumerate(self.pending):
                if pending is None:
                    continue
                _, features = pending
                # Paper-style sequential normalized EMA, not EMA of a batch mean.
                for token in range(features.shape[1]):
                    self.centers[layer].copy_(F.normalize(gamma * self.centers[layer] + (1. - gamma) * features[:, token], dim=-1))
                self.initialized[layer] = True
        self.pending = [None] * len(self.centers)
        self.last_drift = float((self.centers - before).norm(dim=-1).mean())
        return {'prototype_drift': self.last_drift}

    def diagnostics(self):
        initial = np.repeat(self.labels, self.lengths)
        auxiliary = np.ones(len(initial), dtype=bool)
        auxiliary[self.offsets[1:] - 1] = False
        initial = initial[auxiliary]
        result = []
        for layer, target in enumerate(self.targets):
            target = target[auxiliary]
            p = np.clip(target, 1e-7, 1 - 1e-7)
            count = max(1, len(target))
            result.append({'flip_rate': float(np.sum((target >= .5) != initial) / count),
                           'mean_entropy': float(np.sum(-p * np.log(p) - (1 - p) * np.log(1 - p)) / count),
                           'mean_absolute_label_change': float(np.sum(np.abs(target - initial)) / count),
                           'initial_label_weight': float(self.initial_weight[layer, auxiliary].sum() / count),
                           'prototype_cosine': float(F.cosine_similarity(self.centers[layer, 0:1], self.centers[layer, 1:2]).item()),
                           'initialized': bool(self.initialized[layer])})
        variance, hard_disagreement = 0., 0
        for begin in range(0, self.targets.shape[1], 100000):
            selected = self.targets[:, begin:begin + 100000][:, auxiliary[begin:begin + 100000]]
            variance += float(selected.var(axis=0).sum())
            hard = selected >= .5
            hard_disagreement += int(np.sum(hard.any(axis=0) != hard.all(axis=0)))
        return {'by_layer': result, 'total_position_visits': self.total_visits,
                'disambiguation_layer_visits': self.disambiguation_visits,
                'prototype_drift': self.last_drift,
                'layer_target_variance': variance / max(1, int(auxiliary.sum())),
                'hard_layer_disagreement': hard_disagreement / max(1, int(auxiliary.sum()))}

    def state_dict(self):
        if any(x is not None for x in self.pending):
            raise RuntimeError('Save ShieldState only at a completed macro-batch boundary')
        return {'ids': self.ids, 'lengths': self.lengths.copy(), 'labels': self.labels.copy(),
                'targets': self.targets.copy(), 'initial_weight': self.initial_weight.copy(),
                'centers': self.centers.detach().cpu().clone(), 'initialized': self.initialized.copy(),
                'mode': self.mode, 'warmup_steps': self.warmup_steps, 'ramp_steps': self.ramp_steps,
                'topk': self.topk, 'total_visits': self.total_visits,
                'disambiguation_visits': self.disambiguation_visits, 'last_drift': self.last_drift}

    def load_state_dict(self, state):
        if tuple(state['ids']) != self.ids or not np.array_equal(state['lengths'], self.lengths) or not np.array_equal(state['labels'], self.labels):
            raise ValueError('ShieldState data identities or lengths/labels differ')
        for name in ('mode', 'warmup_steps', 'ramp_steps', 'topk'):
            if state[name] != getattr(self, name):
                raise ValueError(f'ShieldState protocol mismatch: {name}')
        if state['targets'].shape != self.targets.shape or state['centers'].shape != self.centers.shape:
            raise ValueError('ShieldState layer or hidden dimensions differ')
        self.targets[:] = state['targets']
        self.initial_weight[:] = state['initial_weight']
        self.centers.copy_(state['centers'].to(self.centers))
        self.initialized[:] = state['initialized']
        self.total_visits, self.disambiguation_visits = state['total_visits'], state['disambiguation_visits']
        self.last_drift = state['last_drift']
        self.pending = [None] * len(self.centers)
