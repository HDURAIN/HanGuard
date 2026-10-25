"""Fresh nine-configuration frozen backbone readouts.

Only source definitions are reused from earlier work; no experiment data,
cache, learned state or selected layer is loaded by this module.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.hanguard.multilabel_heads import MultiLabelHead


MAIN_ARMS = ('baseline', 'fusion_only', 'category_only', 'fusion_category')
ARMS = MAIN_ARMS + ('fusion_repeat_last', 'category_fusion_repeat_last',
                    'category_shared', 'fusion_category_shared', 'probe_selected_single')
LAYERS = (8, 16, 24, 32)


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    """Count actual parameters, excluding the fixed layer/query buffers."""
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


class FixedLayerFusion(nn.Module):
    """Independent per-layer projections and an initially zero residual map."""

    layers = LAYERS

    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.hidden_size, self.width = hidden_size, width
        self.register_buffer('weights', torch.full((len(self.layers),), .25))
        self.layer_values = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.LayerNorm(width))
            for _ in self.layers
        ])
        self.fusion_output = nn.Linear(width, width)
        nn.init.zeros_(self.fusion_output.weight)
        nn.init.zeros_(self.fusion_output.bias)

    def forward(self, last_layers: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(last_layers.float(), (self.hidden_size,))
        value = sum(self.weights[i] * module(normalized[:, i])
                    for i, module in enumerate(self.layer_values))
        return self.fusion_output(value)


class CleanAblationHead(nn.Module):
    """MLP, fixed fusion, category queries, or both, with paired initialization.

    ``last_layers`` holds all 32 raw last-valid-token layer states,
    [B,32,D]. Query arms additionally require all raw L32 token states [B,T,D]
    and their valid-token mask [B,T]. No token is truncated in this module.
    Fusion adds the same sentence-level residual to every category's pooled
    representation before the existing query/pooled interaction scorer.

    A/B have identical initial MLP state, and C/D identical initial query
    state, independent of construction order or global RNG state. The added
    fusion map starts at zero, making paired initial logits identical. Every
    trainable parameter belongs to a used path; there are no count-matching
    dummy parameters and no learnable layer weights or best-layer anchor.
    """

    def __init__(self, hidden_size: int, width: int = 128,
                 num_classes: int = 2, arm: str = 'baseline',
                 dropout: float = .1, seed: int = 42, selected_layer: int = 32):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f'Unknown four-arm head: {arm}')
        if hidden_size < 1 or width < 1 or num_classes not in (2, 5):
            raise ValueError('Positive hidden_size/width and num_classes=2 or 5 required')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.hidden_size, self.width = hidden_size, width
        self.num_classes, self.arm = num_classes, arm
        self.seed, self.dropout_rate = int(seed), float(dropout)
        self.has_category = arm in ('category_only', 'fusion_category', 'category_fusion_repeat_last', 'category_shared', 'fusion_category_shared')
        self.has_fusion = arm in ('fusion_only', 'fusion_category', 'fusion_repeat_last', 'category_fusion_repeat_last', 'fusion_category_shared')
        self.shared_attention = arm in ('category_shared', 'fusion_category_shared')
        self.repeat_last = arm in ('fusion_repeat_last', 'category_fusion_repeat_last')
        if not 1 <= int(selected_layer) <= 32:
            raise ValueError('selected_layer must be in 1..32')
        self.selected_layer = int(selected_layer) if arm == 'probe_selected_single' else 32
        self.layers = LAYERS
        # Initialize on CPU without consuming the caller's RNG or touching CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            if self.has_category:
                self.query_head = MultiLabelHead(
                    hidden_size, width=width, num_labels=num_classes,
                    mode='learned_queries', dropout=dropout, query_seed=self.seed)
            else:
                self.project = nn.Linear(hidden_size, width)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(width, num_classes)
            if self.has_fusion:
                self.branch = FixedLayerFusion(hidden_size, width)

    def parameter_count(self, trainable_only: bool = True) -> int:
        return parameter_count(self, trainable_only=trainable_only)

    def metadata(self) -> dict:
        """JSON-safe architecture description for experiment registration."""
        return dict(
            architecture='clean_nine_arm_fixed_fusion_category_query_v1', arm=self.arm,
            hidden_size=self.hidden_size, width=self.width,
            num_classes=self.num_classes, seed=self.seed,
            dropout=self.dropout_rate, has_category=self.has_category,
            has_fusion=self.has_fusion, output='raw_logits',
            objective='softmax_cross_entropy',
            global_readout=f'L{self.selected_layer}_last_valid_token',
            shared_token_attention=self.shared_attention, repeat_last_capacity_control=self.repeat_last,
            category_readout='L32_all_valid_token_learned_queries' if self.has_category else None,
            query_implementation='MultiLabelHead.learned_queries' if self.has_category else None,
            fusion_layers=([32] * 4 if self.repeat_last else list(LAYERS)) if self.has_fusion else [],
            fusion_weights=[.25] * 4 if self.has_fusion else [],
            learnable_layer_weights=False, best_layer_anchor=False,
            fusion_initialization='zero_output_residual' if self.has_fusion else None,
            trainable_parameters=self.parameter_count(),
            fusion_parameters=parameter_count(self.branch) if self.has_fusion else 0,
            token_truncation=False,
        )

    def _validate(self, last_layers, tokens, mask):
        if (not isinstance(last_layers, torch.Tensor) or last_layers.ndim != 3 or
                last_layers.shape[1:] != (32, self.hidden_size) or
                last_layers.shape[0] < 1):
            raise ValueError('Expected nonempty last_layers [batch,32,hidden_size]')
        if not self.has_category:
            return
        if (tokens is None or mask is None or tokens.ndim != 3 or
                tokens.shape[0] != last_layers.shape[0] or
                tokens.shape[-1] != self.hidden_size or tokens.shape[1] < 1 or
                mask.shape != tokens.shape[:2]):
            raise ValueError('Query arms require aligned tokens [batch,tokens,hidden_size] and mask')
        if tokens.device != last_layers.device or mask.device != tokens.device:
            raise ValueError('Token features, layer features and mask must share a device')
        if not mask.bool().any(1).all():
            raise ValueError('Every input needs at least one valid token')

    def forward(self, last_layers: torch.Tensor,
                tokens: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        self._validate(last_layers, tokens, mask)
        all_layers = last_layers
        branch_layers = all_layers[:, [31] * 4] if self.repeat_last else all_layers[:, [layer - 1 for layer in LAYERS]]
        with torch.autocast(device_type=last_layers.device.type, enabled=False):
            if not self.has_category:
                raw = all_layers[:, self.selected_layer - 1].float()
                pooled = F.gelu(self.project(F.layer_norm(raw, (self.hidden_size,))))
                if self.has_fusion:
                    pooled = pooled + self.branch(branch_layers)
                return self.classifier(self.dropout(pooled))

            # Identical operations and weights to MultiLabelHead.learned_queries.
            # Clear padding before any normalization: zero attention times NaN
            # would otherwise contaminate an otherwise valid pooled value.
            valid = mask.bool()
            raw = tokens.masked_fill(~valid[..., None], 0.).float()
            normalized = F.layer_norm(raw, (self.hidden_size,))
            query_head = self.query_head
            keys = query_head.query_norm(query_head.query_project(normalized))
            values = query_head.value_project(normalized)
            queries = query_head.category_queries()
            attention_queries = queries.mean(0, keepdim=True).expand_as(queries) if self.shared_attention else queries
            scores = torch.einsum('cd,btd->bct', attention_queries, keys) / math.sqrt(self.width)
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            pooled = torch.einsum('bct,btd->bcd', weights, values)
            if self.has_fusion:
                pooled = pooled + self.branch(branch_layers)[:, None, :]
            identity = queries[None].expand(len(tokens), -1, -1)
            conditioned = torch.cat((pooled, identity, pooled * identity), dim=-1)
            return query_head.shared_scorer(conditioned).squeeze(-1)
