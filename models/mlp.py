"""No-message-passing baseline for relational privilege audits."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureSpec, NodeFeatureEncoder


class NodeMLPClassifier(nn.Module):
    """Use the shared node encoder without reading graph edges.

    The hidden depth, residual updates, normalisation and classifier mirror the
    relational classifiers as closely as possible.  The intended comparison is
    therefore whether neighbourhood message passing adds a disparity beyond
    the information already present in the selected node features.
    """

    model_name = "mlp"
    supports_attention = False

    def __init__(
        self,
        num_classes: int,
        feature_specs: Mapping[str, FeatureSpec],
        hidden_dim: int = 128,
        branch_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **_,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.feature_encoder = NodeFeatureEncoder(feature_specs, branch_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        del edge_index, edge_type
        state, _ = self.feature_encoder(features)
        for layer, norm in zip(self.layers, self.norms):
            state = norm(state + self.dropout(F.gelu(layer(state))))
        return self.classifier(state)
