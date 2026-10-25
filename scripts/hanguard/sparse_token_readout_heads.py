"""Capacity-matched single-layer readouts for the sparse-evidence experiment.

No model, data or checkpoint is loaded here. All five arms have identical
parameter layouts; only the query/input/normalizer intervention changes.
The entmax-1.5 threshold and Jacobian follow Peters et al. (ACL 2019):
https://aclanthology.org/P19-1146/ and the authors' implementation at
https://github.com/deep-spin/entmax/blob/master/entmax/activations.py .
"""

from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = (
    "repeat_last",
    "static_query_softmax",
    "global_query_softmax",
    "global_query_entmax15",
    "gated_attention_softmax",
)


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, valid: torch.Tensor, dim: int):
        # For shifted x=scores/2, its maximum is 0 and the optimal threshold
        # is in [-1, 0]. Values below -2 can be clipped without changing any
        # positive output; a finite -2 sentinel also avoids inf-inf in moments.
        maximum = scores.masked_fill(~valid, -torch.inf).amax(dim, keepdim=True)
        shifted = ((scores - maximum) / 2).clamp_min(-2).masked_fill(~valid, -2)
        ordered = shifted.sort(dim=dim, descending=True).values
        shape = [1] * scores.ndim
        shape[dim] = scores.shape[dim]
        counts = torch.arange(1, scores.shape[dim] + 1,
                              device=scores.device, dtype=scores.dtype).view(shape)
        mean = ordered.cumsum(dim) / counts
        variance_sum = (ordered.square().cumsum(dim) / counts - mean.square()) * counts
        threshold = mean - ((1 - variance_sum) / counts).clamp_min(0).sqrt()
        support = (ordered >= threshold).sum(dim, keepdim=True)
        tau = threshold.gather(dim, support - 1)
        probabilities = (shifted - tau).clamp_min(0).square().masked_fill(~valid, 0)
        ctx.dim = dim
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        probabilities, = ctx.saved_tensors
        root = probabilities.sqrt()
        weighted = grad_output * root
        correction = weighted.sum(ctx.dim, keepdim=True) / root.sum(ctx.dim, keepdim=True)
        return weighted - root * correction, None, None


def entmax15(scores: torch.Tensor, mask: torch.Tensor | None = None,
             dim: int = -1) -> torch.Tensor:
    """Exact entmax-1.5 with hard padding exclusion and an analytic Jacobian.

    Every reduction slice needs a finite valid score. Padding values may be
    nonfinite. Low-precision scores are computed and returned in float32;
    float64 is preserved for independent numerical gradient checks.
    """
    if not scores.is_floating_point() or scores.ndim < 1:
        raise ValueError("scores must be a floating point tensor with at least one dimension")
    if not -scores.ndim <= dim < scores.ndim or scores.shape[dim] < 1:
        raise ValueError("Invalid or empty normalization dimension")
    dim %= scores.ndim
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    elif mask.shape != scores.shape or mask.device != scores.device:
        raise ValueError("mask must match the score shape and device")
    valid = mask.bool()
    if not valid.any(dim).all():
        raise ValueError("Every normalized slice needs at least one valid score")
    if not torch.isfinite(scores.masked_fill(~valid, 0)).all():
        raise ValueError("Valid scores must be finite")
    work = scores if scores.dtype == torch.float64 else scores.float()
    return _Entmax15.apply(work, valid, dim)


def last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return absolute last valid positions for left, right or mixed padding."""
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 1 or not mask.bool().any(1).all():
        raise ValueError("Every sentence must have at least one valid token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask.bool(), -1).amax(1)


class SparseTokenReadoutHead(nn.Module):
    """A last-token anchor plus task/global-conditioned token evidence.

    ``raw_tokens`` contains a single layer's raw [batch,tokens,hidden] states.
    As in ExperimentHead, each raw vector is feature-normalized before its
    learned projection, and classification uses GELU, dropout and one logit.
    The zero-initialized local residual initially preserves the anchor. Its
    upstream gradients are expected to activate after the first optimizer step.

    ``repeat_last`` substitutes the last valid raw vector at every local slot;
    it has the same nominal parameter count but no extra token information.
    Query weights then cannot affect the pooled value. The static arm retains
    query_global parameters with a zero multiplier; their task gradient is
    intentionally zero. Parameter count matching does not assert equal
    effective trainable function families across these controls.
    """

    def __init__(self, hidden_size: int, mode: str = "global_query_entmax15",
                 width: int = 128, dropout: float = .1, chunk_size: int = 512):
        super().__init__()
        if mode not in ARMS:
            raise ValueError(f"Unsupported sparse token arm: {mode}")
        if hidden_size < 1 or width < 1 or chunk_size < 1 or not 0 <= dropout < 1:
            raise ValueError("Expected positive sizes and dropout in [0,1)")
        self.hidden_size, self.width, self.mode = hidden_size, width, mode
        self.chunk_size = chunk_size
        self.project = nn.Linear(hidden_size, width)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 1)
        self.token_project = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
        self.query_global = nn.Linear(width, width)
        self.task_query = nn.Parameter(torch.empty(width))
        nn.init.uniform_(self.task_query, -1 / math.sqrt(width), 1 / math.sqrt(width))
        self.key = nn.Linear(width, width)
        self.pool_output = nn.Linear(width, width)
        nn.init.zeros_(self.pool_output.weight)
        nn.init.zeros_(self.pool_output.bias)

    def _project_raw(self, raw: torch.Tensor, module: nn.Module) -> torch.Tensor:
        pieces = []
        for piece in raw.reshape(-1, self.hidden_size).split(self.chunk_size):
            # Compute the head in FP32 even with a mixed-precision backbone.
            normalized = F.layer_norm(piece.float(), (self.hidden_size,))
            pieces.append(module(normalized))
        return torch.cat(pieces).reshape(*raw.shape[:-1], self.width)

    def forward(self, raw_tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if (raw_tokens.ndim != 3 or raw_tokens.shape[:2] != mask.shape
                or raw_tokens.shape[-1] != self.hidden_size or raw_tokens.device != mask.device):
            raise ValueError("raw_tokens [B,T,D] and mask [B,T] must align")
        if not raw_tokens.is_floating_point():
            raise ValueError("Raw token states must be floating point")
        valid = mask.bool()
        last = last_valid_indices(valid)
        rows = torch.arange(len(valid), device=valid.device)
        with torch.autocast(device_type=raw_tokens.device.type, enabled=False):
            raw_global = raw_tokens[rows, last]
            global_z = F.gelu(self._project_raw(raw_global, self.project))
            if self.mode == "repeat_last":
                local_raw = raw_global[:, None].expand_as(raw_tokens)
            else:
                # Nonfinite or arbitrary padding never enters any projection.
                local_raw = raw_tokens.masked_fill(~valid[..., None], 0)
            values = self._project_raw(local_raw, self.token_project)
            if self.mode == "gated_attention_softmax":
                # Identical scorer to ExperimentHead(mode='attention') after
                # copying key/query_global/task_query to tanh/gate/context.
                query = self.task_query[None].expand(len(valid), -1)
                gate = torch.tanh(self.key(values)) * torch.sigmoid(self.query_global(values))
                scores = (gate * self.task_query).sum(-1)
            else:
                conditional = self.query_global(global_z)
                query = self.task_query[None] + conditional * (0 if self.mode == "static_query_softmax" else 1)
                scores = (self.key(values) * query[:, None]).sum(-1) / math.sqrt(self.width)
            masked_scores = scores.masked_fill(~valid, -torch.inf)
            if self.mode == "global_query_entmax15":
                attention = entmax15(scores, valid)
            else:
                attention = masked_scores.softmax(-1)
            local_z = (attention[..., None] * values).sum(1)
            z = global_z + self.pool_output(local_z)
            logits = self.classifier(self.dropout(z)).squeeze(-1)
            return dict(logits=logits, representation=z, global_representation=global_z,
                        local_representation=local_z, query=query, attention_scores=masked_scores,
                        attention_weights=attention, last_indices=last,
                        support_size=(attention > 0).sum(-1))


def parameter_count(head: nn.Module) -> int:
    return sum(parameter.numel() for parameter in head.parameters())


def initialization_audit(heads: dict[str, nn.Module]) -> dict:
    """Fail unless state layout, values and nominal parameter counts match."""
    if tuple(heads) != ARMS:
        raise ValueError("Expected all five arms in their registered order")
    reference = heads[ARMS[0]].state_dict()
    count = parameter_count(heads[ARMS[0]])
    digest = hashlib.sha256()
    for name, value in reference.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for arm in ARMS[1:]:
        state = heads[arm].state_dict()
        if list(state) != list(reference) or parameter_count(heads[arm]) != count:
            raise ValueError(f"{arm} has a different capacity/state layout")
        if any(state[name].shape != value.shape or not torch.equal(state[name], value)
               for name, value in reference.items()):
            raise ValueError(f"{arm} initialization differs")
    return dict(arms=list(ARMS), parameters=count, initialization_sha256=digest.hexdigest(),
                effective_capacity_caveat="static query projection and repeat query route are intentionally inactive")


def build_matched_heads(hidden_size: int, width: int = 128, seed: int = 42,
                        dropout: float = .1, device: str | torch.device = "cpu",
                        chunk_size: int = 512) -> tuple[dict[str, SparseTokenReadoutHead], dict]:
    """Initialize once without perturbing caller RNG, then copy all weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        template = SparseTokenReadoutHead(hidden_size, ARMS[0], width, dropout, chunk_size)
    heads = {}
    for arm in ARMS:
        head = copy.deepcopy(template)
        head.mode = arm
        heads[arm] = head.to(device)
    return heads, initialization_audit(heads)
