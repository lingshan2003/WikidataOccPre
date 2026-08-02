"""Graph models used by this project.

``RGCNModel`` is kept for the original experiment.  New experiments should use
``RelationalGATClassifier`` below: it supports typed, extensible node features
and can return per-edge attention weights for later explanation.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

try:
    from torch_geometric.nn import RGATConv
except ImportError:  # Makes the old R-GCN baseline usable with older PyG.
    RGATConv = None
from config import HIDDEN_DIM, NUM_BASES, DROPOUT_RATE


class RGCNModel(nn.Module):
    def __init__(self, num_relations, hidden_dim, feature_num_classes):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feature_name: nn.Embedding(num_classes, hidden_dim)
            for feature_name, num_classes in feature_num_classes.items()
        })
        
        in_channels = hidden_dim * len(self.embeddings)
        self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations, num_bases=NUM_BASES)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=NUM_BASES)
        self.classifier = nn.Linear(hidden_dim, feature_num_classes['occupation'])

    def forward(self, features_dict, edge_index, edge_type):
        embeddings_list = []
        for feature_name, emb_layer in self.embeddings.items():
            embeddings_list.append(emb_layer(features_dict[feature_name]))
        combined_features = torch.cat(embeddings_list, dim=1)

        h = self.conv1(combined_features, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT_RATE, training=self.training)
        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT_RATE, training=self.training)

        return self.classifier(h)


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one node feature branch.

    ``categorical`` values must be integer IDs in ``[0, cardinality)``.  Reserve
    an explicit ID for missing/unknown values in the data pipeline; it must not
    be silently conflated with a valid country or occupation.

    ``numeric`` and ``vector`` values are float tensors of shape
    ``[num_nodes, input_dim]``.  Numeric values should be normalised using
    training-node statistics by the data pipeline.
    """

    kind: str
    input_dim: int = 1
    cardinality: Optional[int] = None
    optional: bool = False

    def validate(self) -> None:
        if self.kind not in {"categorical", "numeric", "vector"}:
            raise ValueError(f"Unknown feature kind: {self.kind}")
        if self.kind == "categorical" and (self.cardinality is None or self.cardinality < 2):
            raise ValueError("A categorical feature needs cardinality >= 2")
        if self.kind != "categorical" and self.input_dim < 1:
            raise ValueError("A numeric/vector feature needs input_dim >= 1")


class NodeFeatureEncoder(nn.Module):
    """Encode and gate any registered set of node attributes.

    Adding a new attribute requires only adding a :class:`FeatureSpec` when the
    model is constructed and supplying its tensor at forward time.  Optional
    features get a learned missing vector when absent, so an older data export
    can still be used with a newer model definition.
    """

    def __init__(
        self,
        feature_specs: Mapping[str, FeatureSpec],
        branch_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not feature_specs:
            raise ValueError("At least one node feature is required")

        self.feature_specs = dict(feature_specs)
        self.feature_names = tuple(self.feature_specs)
        self.branch_dim = branch_dim
        self.encoders = nn.ModuleDict()
        self.missing_vectors = nn.ParameterDict()

        for name, spec in self.feature_specs.items():
            spec.validate()
            if spec.kind == "categorical":
                self.encoders[name] = nn.Embedding(spec.cardinality, branch_dim)
            else:
                self.encoders[name] = nn.Sequential(
                    nn.Linear(spec.input_dim, branch_dim),
                    nn.GELU(),
                    nn.LayerNorm(branch_dim),
                )
            if spec.optional:
                self.missing_vectors[name] = nn.Parameter(torch.empty(branch_dim))
                nn.init.normal_(self.missing_vectors[name], std=0.02)

        fusion_dim = branch_dim * len(self.feature_names)
        self.feature_gate = nn.Linear(fusion_dim, len(self.feature_names))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, features: Mapping[str, torch.Tensor], num_nodes: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if num_nodes is None:
            for value in features.values():
                num_nodes = value.size(0)
                break
        if num_nodes is None:
            raise ValueError("num_nodes is required when no feature tensor is supplied")

        branches = []
        for name in self.feature_names:
            spec = self.feature_specs[name]
            value = features.get(name)
            if value is None:
                if not spec.optional:
                    raise KeyError(f"Missing required feature '{name}'")
                branch = self.missing_vectors[name].unsqueeze(0).expand(num_nodes, -1)
            elif spec.kind == "categorical":
                if value.dim() != 1:
                    raise ValueError(f"Categorical feature '{name}' must have shape [num_nodes]")
                branch = self.encoders[name](value.long())
            else:
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                if value.dim() != 2 or value.size(1) != spec.input_dim:
                    raise ValueError(
                        f"Feature '{name}' must have shape [num_nodes, {spec.input_dim}]"
                    )
                branch = self.encoders[name](value.float())
            branches.append(branch)

        stacked = torch.stack(branches, dim=1)  # [N, feature_count, branch_dim]
        flattened = stacked.flatten(start_dim=1)
        gate = torch.softmax(self.feature_gate(flattened), dim=-1)
        fused = self.fusion((stacked * gate.unsqueeze(-1)).flatten(start_dim=1))
        return fused, {name: gate[:, index] for index, name in enumerate(self.feature_names)}


class RelationalGATClassifier(nn.Module):
    """Two-or-more-layer relation-aware GAT for occupation classification.

    The model deliberately returns attention weights only on request.  These
    weights are an explanation candidate, not a causal proof; evaluate selected
    edges with deletion tests or a learned edge-mask explainer as well.
    """

    def __init__(
        self,
        num_relations: int,
        num_classes: int,
        feature_specs: Mapping[str, FeatureSpec],
        hidden_dim: int = 128,
        branch_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if RGATConv is None:
            raise ImportError("RelationalGATClassifier requires torch-geometric with RGATConv")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.feature_encoder = NodeFeatureEncoder(feature_specs, branch_dim, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            RGATConv(
                hidden_dim,
                hidden_dim,
                num_relations=num_relations,
                heads=heads,
                concat=False,
                dropout=attention_dropout,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
        """Return node logits, optionally with attention aligned to input edges.

        ``edge_index`` and ``edge_type`` can be a sampled subgraph during
        training.  For explanation, retain the sampled graph's global ``n_id``
        so a local edge can be mapped back to the original people.
        """
        h, feature_gates = self.feature_encoder(features)
        attention_layers = []

        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if return_attention_weights:
                updated, (attention_edge_index, alpha) = conv(
                    h, edge_index, edge_type, return_attention_weights=True
                )
                attention_layers.append(
                    {
                        "layer": layer_index,
                        "edge_index": attention_edge_index,
                        "edge_type": edge_type,
                        "alpha": alpha,
                    }
                )
            else:
                updated = conv(h, edge_index, edge_type)
            h = norm(h + self.dropout(F.gelu(updated)))

        logits = self.classifier(h)
        if not return_attention_weights:
            return logits
        return logits, {"feature_gates": feature_gates, "attention_layers": attention_layers}
