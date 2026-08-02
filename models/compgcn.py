"""CompGCN-style relational message passing for person-node classification.

The graph remains a directed, multi-relational *person* graph.  Each relation
type has a learned embedding, which is composed with the source person's
representation before the message reaches the destination person.  This is a
node-classification use of CompGCN: it does not change the task into link
prediction or introduce occupation labels as input features.
"""

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from .features import FeatureSpec, NodeFeatureEncoder


class CompGCNLayer(nn.Module):
    """One relation-compositional, mean-aggregation message-passing layer."""

    def __init__(self, hidden_dim: int, composition: str, dropout: float) -> None:
        super().__init__()
        if composition not in {"mult", "sub"}:
            raise ValueError("composition must be 'mult' or 'sub'")
        self.composition = composition
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.root = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_state: torch.Tensor,
        relation_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        source_state = node_state[source]
        edge_relation = relation_state[edge_type]
        if self.composition == "mult":
            composed = source_state * edge_relation
        else:
            composed = source_state - edge_relation

        messages = self.message(composed)
        aggregated = scatter(messages, destination, dim=0, dim_size=node_state.size(0), reduce="mean")
        updated_nodes = self.root(node_state) + self.dropout(aggregated)
        updated_relations = self.relation_update(relation_state)
        return updated_nodes, updated_relations


class RelationalCompGCNClassifier(nn.Module):
    """Two-layer CompGCN classifier with the shared, schema-driven encoder."""

    model_name = "compgcn"
    supports_attention = False

    def __init__(
        self,
        num_relations: int,
        num_classes: int,
        feature_specs: Mapping[str, FeatureSpec],
        hidden_dim: int = 128,
        branch_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        compgcn_composition: str = "mult",
        **_,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("Current sampled training uses exactly two message-passing layers")
        self.composition = compgcn_composition
        self.feature_encoder = NodeFeatureEncoder(feature_specs, branch_dim, hidden_dim, dropout)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.convs = nn.ModuleList(
            CompGCNLayer(hidden_dim, compgcn_composition, dropout) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        node_state, _ = self.feature_encoder(features)
        relation_state = self.relation_embedding.weight
        for conv, norm in zip(self.convs, self.norms):
            updated_nodes, relation_state = conv(node_state, relation_state, edge_index, edge_type)
            node_state = norm(node_state + self.dropout(F.gelu(updated_nodes)))
            relation_state = F.gelu(relation_state)
        return self.classifier(node_state)
