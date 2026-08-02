"""HGT encoder and DistMult decoder for Person--Occupation links."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from models.features import FeatureSpec, NodeFeatureEncoder

PERSON = "person"
OCCUPATION = "occupation"


class HeteroOccupationLinkPredictor(nn.Module):
    """Encode heterogeneous neighbourhoods and rank occupation link targets."""

    model_name = "hgt_distmult"

    def __init__(
        self,
        metadata: Tuple[list[str], list[Tuple[str, str, str]]],
        num_occupations: int,
        country_cardinality: int,
        temporal_dim: int,
        hidden_dim: int = 128,
        branch_dim: int = 64,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.person_encoder = NodeFeatureEncoder(
            {
                "country": FeatureSpec("categorical", cardinality=country_cardinality),
                "temporal": FeatureSpec("numeric", input_dim=temporal_dim),
            },
            branch_dim,
            hidden_dim,
            dropout,
        )
        self.occupation_embedding = nn.Embedding(num_occupations, hidden_dim)
        nn.init.xavier_uniform_(self.occupation_embedding.weight)
        self.convs = nn.ModuleList(
            HGTConv({PERSON: hidden_dim, OCCUPATION: hidden_dim}, hidden_dim, metadata, heads=heads)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({PERSON: nn.LayerNorm(hidden_dim), OCCUPATION: nn.LayerNorm(hidden_dim)})
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder_relation = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.decoder_relation, std=0.02)

    def encode(self, batch) -> Dict[str, torch.Tensor]:
        person, _ = self.person_encoder(
            {"country": batch[PERSON].country, "temporal": batch[PERSON].temporal}
        )
        states = {
            PERSON: person,
            OCCUPATION: self.occupation_embedding(batch[OCCUPATION].n_id),
        }
        for conv, norms in zip(self.convs, self.norms):
            updated = conv(states, batch.edge_index_dict)
            states = {
                node_type: norms[node_type](states[node_type] + self.dropout(F.gelu(updated[node_type])))
                for node_type in states
            }
        return states

    def score_pairs(self, person_state: torch.Tensor, occupation_ids: torch.Tensor) -> torch.Tensor:
        occupation_state = self.occupation_embedding(occupation_ids)
        return ((person_state * self.decoder_relation) * occupation_state).sum(dim=-1)

    def score_all_occupations(self, person_state: torch.Tensor) -> torch.Tensor:
        return (person_state * self.decoder_relation) @ self.occupation_embedding.weight.t()
