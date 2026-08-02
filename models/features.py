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

    def validate(self) -> None:
        if self.kind not in {"categorical", "numeric", "vector"}:
            raise ValueError(f"Unknown feature kind: {self.kind}")
        if self.kind == "categorical" and (self.cardinality is None or self.cardinality < 2):
            raise ValueError("A categorical feature needs cardinality >= 2")
        if self.kind != "categorical" and self.input_dim < 1:
            raise ValueError("A numeric/vector feature needs input_dim >= 1")


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
            else:
                self.encoders[name] = nn.Sequential(
                    nn.Linear(spec.input_dim, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim)
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
