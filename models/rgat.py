"""Relation-aware graph attention classifier."""

from typing import Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureSpec, NodeFeatureEncoder

try:
    from torch_geometric.nn import RGATConv
except ImportError:
    RGATConv = None


class RelationalGATClassifier(nn.Module):
    model_name = "rgat"
    supports_attention = True

    def __init__(self, num_relations: int, num_classes: int, feature_specs: Mapping[str, FeatureSpec],
                 hidden_dim: int = 128, branch_dim: int = 64, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.2, attention_dropout: float = 0.1, **_) -> None:
        super().__init__()
        if RGATConv is None:
            raise ImportError("RGATConv requires a current torch-geometric installation")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.feature_encoder = NodeFeatureEncoder(feature_specs, branch_dim, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            RGATConv(hidden_dim, hidden_dim, num_relations, heads=heads, concat=False, dropout=attention_dropout)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: Mapping[str, torch.Tensor], edge_index: torch.Tensor,
                edge_type: torch.Tensor, return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        h, feature_gates = self.feature_encoder(features)
        attention_layers = []
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if return_attention_weights:
                # Keep the exact state that RGATConv receives.  Alpha alone is
                # only an allocation coefficient; value-aware analysis needs
                # the corresponding relation-transformed source state W_r h_j.
                input_node_state = h
                updated, (attention_edge_index, alpha) = conv(h, edge_index, edge_type, return_attention_weights=True)
                # ``RGATConv`` may append synthetic self loops before returning
                # its attention edge index.  Keep the original typed edges too,
                # so downstream exports can exclude synthetic loops instead of
                # accidentally assigning them a relation ID.
                attention_layers.append({
                    "layer": index,
                    "edge_index": attention_edge_index,
                    "edge_type": edge_type,
                    "input_edge_index": edge_index,
                    "input_edge_type": edge_type,
                    "input_node_state": input_node_state,
                    "alpha": alpha,
                })
            else:
                updated = conv(h, edge_index, edge_type)
            h = norm(h + self.dropout(F.gelu(updated)))
        logits = self.classifier(h)
        return (logits, {"feature_gates": feature_gates, "attention_layers": attention_layers}) if return_attention_weights else logits
