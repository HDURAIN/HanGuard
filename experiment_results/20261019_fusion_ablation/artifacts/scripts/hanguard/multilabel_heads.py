"""Five-label readouts with shared, label-conditioned token attention.

No backbone, dataset, checkpoint or GPU is loaded by this module. Attention
weights describe this aggregation computation; they are not verified faithful
explanations or ground-truth token labels.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ('last_mlp', 'learned_queries', 'description_queries')


class MultiLabelHead(nn.Module):
    """Global MLP or five query-conditioned readouts with one shared scorer.

    Query modes use the identical trainable architecture:
        query = LayerNorm(project(LayerNorm(fixed_seed)) + free_residual)
    ``description_queries`` gets fixed encoded category descriptions as its
    seeds. ``learned_queries`` gets fixed Gaussian seeds, generated with a local
    RNG. Both have a trainable projection and unrestricted [labels,width]
    residual initialized to zero. Token keys use the same projection and output
    normalization as the queries, preserving their shared input-space alignment
    at initialization. Thus random and semantic initialization have
    exactly equal trainable parameters; no unused dummy parameters are added.

    This random-query control is a factorized freely trainable query adaptation,
    not a parameter-smaller direct embedding lookup. Fixed vectors are detached
    buffers. All five probabilities are independent sigmoids, never a softmax
    over labels. The attention softmax is solely over valid input positions.
    """
    def __init__(self, hidden_size, width=128, num_labels=5, mode='last_mlp',
                 query_vectors=None, dropout=.1, query_seed=42):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown multilabel head mode: {mode}')
        if hidden_size < 1 or width < 1 or num_labels < 1:
            raise ValueError('hidden_size, width and num_labels must be positive')
        self.hidden_size, self.width, self.num_labels = hidden_size, width, num_labels
        self.mode = mode
        if mode == 'last_mlp':
            if query_vectors is not None:
                raise ValueError('The last-token baseline does not consume query vectors')
            self.project = nn.Linear(hidden_size, width)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(width, num_labels)
            return

        if mode == 'description_queries':
            if query_vectors is None:
                raise ValueError('description_queries requires fixed encoded [labels,hidden_size] vectors')
            vectors = torch.as_tensor(query_vectors).detach().float().cpu().clone()
            if vectors.shape != (num_labels, hidden_size) or not torch.isfinite(vectors).all():
                raise ValueError('Description vectors must be finite [num_labels,hidden_size]')
        else:
            if query_vectors is not None:
                raise ValueError('learned_queries uses random seeds, not supplied descriptions')
            generator = torch.Generator(device='cpu').manual_seed(query_seed)
            vectors = torch.randn(num_labels, hidden_size, generator=generator)
        self.register_buffer('query_vectors', vectors)
        self.query_project = nn.Linear(hidden_size, width)
        self.query_residual = nn.Parameter(torch.zeros(num_labels, width))
        self.query_norm = nn.LayerNorm(width)
        self.value_project = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
        # One scorer is broadcast over all category-conditioned representations.
        # Explicit query interaction preserves class identity even for one token.
        self.shared_scorer = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def category_queries(self):
        if self.mode == 'last_mlp':
            raise ValueError('The global last-token baseline has no category queries')
        with torch.autocast(device_type=self.query_vectors.device.type, enabled=False):
            normalized = F.layer_norm(self.query_vectors.float(), (self.hidden_size,))
            return self.query_norm(self.query_project(normalized) + self.query_residual)

    def forward(self, hidden, mask, return_attention=False):
        """Read raw token states [B,T,D] with a bool/0-1 valid-position mask.

        Returns logits/probabilities [B,C] and representation [B,C,width] for
        query modes, or [B,width] for last_mlp. Optional attention is [B,C,T]
        in query modes; the last-token baseline returns attention=None.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError('Expected token features [batch,tokens,hidden_size]')
        if mask.shape != hidden.shape[:2] or hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError('Expected a nonempty [batch,tokens] attention mask')
        valid = mask.bool()
        if not valid.any(1).all():
            raise ValueError('Every input needs at least one valid token')
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if self.mode == 'last_mlp':
                positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand_as(valid)
                last = positions.masked_fill(~valid, -1).max(1).values
                raw = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
                z = F.gelu(self.project(F.layer_norm(raw, (self.hidden_size,))))
                logits = self.classifier(self.dropout(z))
                result = dict(logits=logits, probabilities=logits.sigmoid(), representation=z)
                if return_attention:
                    result['attention'] = None
                return result

            # Clear padding before normalization/projection, including nonfinite
            # padding sentinels, so it cannot pollute pooled values via 0 * NaN.
            raw = hidden.masked_fill(~valid[..., None], 0.).float()
            normalized = F.layer_norm(raw, (self.hidden_size,))
            keys = self.query_norm(self.query_project(normalized))
            values = self.value_project(normalized)
            queries = self.category_queries()
            scores = torch.einsum('cd,btd->bct', queries, keys) / math.sqrt(self.width)
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            pooled = torch.einsum('bct,btd->bcd', weights, values)
            identity = queries[None].expand(len(hidden), -1, -1)
            conditioned = torch.cat((pooled, identity, pooled * identity), dim=-1)
            logits = self.shared_scorer(conditioned).squeeze(-1)
            result = dict(logits=logits, probabilities=logits.sigmoid(), representation=pooled)
            if return_attention:
                result['attention'] = weights
            return result


def masked_binary_cross_entropy(logits, targets, unknown_value=-1, reduction='mean',
                                known_mask=None, pos_weight=None):
    """BCE over known binary labels only; unknown labels never become negatives.

    Mean reduction divides by the number of known sample/category entries.
    ``known_mask`` can additionally suppress entries (e.g. source coverage), but
    cannot make an unknown target known. Fully unknown batches return a
    differentiable zero. Optional pos_weight has the standard [labels] meaning.
    """
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError('Expected matching logits/targets [batch,num_labels]')
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('reduction must be none, mean, or sum')
    targets = targets.to(device=logits.device)
    known = targets != unknown_value
    if known_mask is not None:
        if known_mask.shape != targets.shape:
            raise ValueError('known_mask must match targets')
        known = known & known_mask.to(device=logits.device, dtype=torch.bool)
    observed = targets[known]
    if not ((observed == 0) | (observed == 1)).all():
        raise ValueError('Known labels must be binary; mark unknown entries explicitly')
    # Indexing before BCE prevents invalid or absent targets contributing any
    # loss or gradient. No pseudo-negative target is inserted for unknown cells.
    if not known.any():
        return logits * 0. if reduction == 'none' else logits.sum() * 0.
    positive = None
    if pos_weight is not None:
        positive = torch.as_tensor(pos_weight, device=logits.device, dtype=torch.float32)
        if positive.shape != (logits.shape[1],) or not torch.isfinite(positive).all() or (positive < 0).any():
            raise ValueError('pos_weight must be a finite nonnegative vector [num_labels]')
        positive = positive[None].expand_as(logits)[known]
    with torch.autocast(device_type=logits.device.type, enabled=False):
        losses = F.binary_cross_entropy_with_logits(logits[known].float(), observed.float(),
                                                    pos_weight=positive, reduction='none')
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    result = logits.float() * 0.
    return result.masked_scatter(known, losses)
