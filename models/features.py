"""Schema-driven encoders shared by every graph model."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FeatureSpec:
    """One categorical, numeric, or pre-computed vector node feature."""

    kind: str
    input_dim: int = 1
    cardinality: Optional[int] = None
    optional: bool = False
    semantic_embeddings: Optional[torch.Tensor] = None
    unknown_id: Optional[int] = None

    def validate(self) -> None:
        if self.kind not in {"categorical", "numeric", "vector", "semantic_categorical"}:
            raise ValueError(f"Unknown feature kind: {self.kind}")
        if self.kind == "categorical" and (self.cardinality is None or self.cardinality < 2):
            raise ValueError("A categorical feature needs cardinality >= 2")
        if self.kind in {"numeric", "vector"} and self.input_dim < 1:
            raise ValueError("A numeric/vector feature needs input_dim >= 1")
        if self.kind == "semantic_categorical":
            if self.semantic_embeddings is None or self.semantic_embeddings.dim() != 2:
                raise ValueError("A semantic categorical feature needs a rank-2 fixed embedding table")
            if self.unknown_id is None:
                raise ValueError("A semantic categorical feature needs an unknown ID")
            if self.cardinality != self.semantic_embeddings.size(0):
                raise ValueError("Semantic categorical cardinality must match its embedding table")
            if self.input_dim != self.semantic_embeddings.size(1):
                raise ValueError("Semantic categorical input_dim must match its embedding dimension")


class NodeFeatureEncoder(nn.Module):
    """Encode feature branches and learn a per-node fusion gate."""

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
        self.encoders = nn.ModuleDict()
        self.missing_vectors = nn.ParameterDict()
        for name, spec in self.feature_specs.items():
            spec.validate()
            if spec.kind == "categorical":
                self.encoders[name] = nn.Embedding(spec.cardinality, branch_dim)
            elif spec.kind == "semantic_categorical":
                # This table is a fixed external semantic prior. It is a buffer
                # rather than a parameter, so optimizer steps cannot rewrite it.
                self.register_buffer(f"{name}_semantic_embeddings", spec.semantic_embeddings.float())
                self.encoders[name] = nn.Sequential(
                    nn.Linear(spec.input_dim, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim)
                )
            else:
                self.encoders[name] = nn.Sequential(
                    nn.Linear(spec.input_dim, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim)
                )
            if spec.optional or spec.kind == "semantic_categorical":
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
            num_nodes = next(iter(features.values())).size(0) if features else None
        if num_nodes is None:
            raise ValueError("num_nodes is required when no feature tensor is supplied")
        branches = []
        for name in self.feature_names:
            spec, value = self.feature_specs[name], features.get(name)
            if value is None:
                if not spec.optional:
                    raise KeyError(f"Missing required feature '{name}'")
                branch = self.missing_vectors[name].unsqueeze(0).expand(num_nodes, -1)
            elif spec.kind == "categorical":
                if value.dim() != 1:
                    raise ValueError(f"Categorical feature '{name}' must have shape [num_nodes]")
                branch = self.encoders[name](value.long())
            elif spec.kind == "semantic_categorical":
                if value.dim() != 1:
                    raise ValueError(f"Semantic categorical feature '{name}' must have shape [num_nodes]")
                ids = value.long()
                table = getattr(self, f"{name}_semantic_embeddings")
                if ids.numel() and (ids.min() < 0 or ids.max() >= table.size(0)):
                    raise ValueError(f"Semantic categorical feature '{name}' contains an out-of-range ID")
                projected = self.encoders[name](table[ids])
                unknown = self.missing_vectors[name].unsqueeze(0).expand(num_nodes, -1)
                branch = torch.where((ids != spec.unknown_id).unsqueeze(-1), projected, unknown)
            else:
                value = value.unsqueeze(-1) if value.dim() == 1 else value
                if value.dim() != 2 or value.size(1) != spec.input_dim:
                    raise ValueError(f"Feature '{name}' must have shape [num_nodes, {spec.input_dim}]")
                branch = self.encoders[name](value.float())
            branches.append(branch)

        stacked = torch.stack(branches, dim=1)
        flattened = stacked.flatten(start_dim=1)
        gate = torch.softmax(self.feature_gate(flattened), dim=-1)
        fused = self.fusion((stacked * gate.unsqueeze(-1)).flatten(start_dim=1))
        return fused, {name: gate[:, index] for index, name in enumerate(self.feature_names)}


def build_feature_specs(feature_schema: Mapping[str, Mapping], metadata: Mapping) -> Dict[str, FeatureSpec]:
    """Construct model specs, including fixed semantic tables stored in an artifact."""
    semantic_tables = metadata.get("semantic_feature_tables", {})
    specs = {}
    for name, definition in feature_schema.items():
        kind = definition["kind"]
        table = None
        if kind == "semantic_categorical":
            table_key = definition.get("semantic_table_key")
            if not table_key or table_key not in semantic_tables:
                raise ValueError(f"Semantic feature '{name}' is missing its fixed embedding table")
            table = semantic_tables[table_key]
            if not isinstance(table, torch.Tensor):
                raise ValueError(f"Semantic table '{table_key}' must be a torch.Tensor")
        specs[name] = FeatureSpec(
            kind=kind,
            cardinality=definition.get("cardinality"),
            input_dim=definition.get("input_dim", 1),
            optional=definition.get("optional", False),
            semantic_embeddings=table,
            unknown_id=definition.get("unknown_id"),
        )
    return specs
