"""Relation-aware graph convolution baseline using the shared feature encoder."""

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .features import FeatureSpec, NodeFeatureEncoder


class RelationalGCNClassifier(nn.Module):
    model_name = "rgcn"
    supports_attention = False

    def __init__(self, num_relations: int, num_classes: int, feature_specs: Mapping[str, FeatureSpec],
                 hidden_dim: int = 128, branch_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, num_bases: int = 30, **_) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("Current sampled training uses exactly two message-passing layers")
        self.feature_encoder = NodeFeatureEncoder(feature_specs, branch_dim, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, edge_index: torch.Tensor, edge_type: torch.Tensor, **_) -> torch.Tensor:
        h, _ = self.feature_encoder(features)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.dropout(F.gelu(conv(h, edge_index, edge_type))))
        return self.classifier(h)
