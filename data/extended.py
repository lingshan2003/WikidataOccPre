"""Preparation utilities for ``Q_R_Q_extended.txt``.

The source is edge-centric: a person's attributes repeat once for every
incident edge.  This module converts it into one canonical node table and one
typed edge table. Target occupations remain in the canonical node table;
preparation/training controls which occupations are visible to the model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Node1", "Relation", "Node2", "Node1_Birth", "Node1_Death",
    "Node2_Birth", "Node2_Death", "Node1_occ_level1", "Node1_occ_level2",
    "Node1_occ_level3", "Node2_occ_level1", "Node2_occ_level2",
    "Node2_occ_level3", "Node1_Country", "Node2_Country",
}


@dataclass
class GraphTables:
    """Canonical tables and mappings used by a graph training pipeline."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    attribute_conflicts: pd.DataFrame


def _first_present(values: pd.Series):
    values = values.dropna()
    return values.iloc[0] if not values.empty else np.nan


class ExtendedGraphLoader:
    """Read the current CSV export and create stable node and edge tables."""

    def __init__(self, path: str = "Q_R_Q_extended.txt") -> None:
        self.path = Path(path)

    def read(self) -> GraphTables:
        raw = pd.read_csv(self.path)
        missing = REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"Extended graph is missing columns: {sorted(missing)}")

        raw = raw.dropna(subset=["Node1", "Relation", "Node2"]).copy()
        node_one = self._endpoint_attributes(raw, "Node1")
        node_two = self._endpoint_attributes(raw, "Node2")
        candidates = pd.concat([node_one, node_two], ignore_index=True)
        nodes, conflicts = self._canonicalise_nodes(candidates)

        edges = raw.loc[:, ["Node1", "Relation", "Node2"]].rename(
            columns={"Node1": "source", "Relation": "relation", "Node2": "target"}
        )
        edges = edges.astype({"source": "string", "relation": "string", "target": "string"})
        edges = self._with_reverse_edges(edges)

        node_ids = {node: index for index, node in enumerate(nodes["node_id"])}
        relation_ids = {relation: index for index, relation in enumerate(sorted(edges["relation"].unique()))}
        edges["source_id"] = edges["source"].map(node_ids)
        edges["target_id"] = edges["target"].map(node_ids)
        edges["relation_id"] = edges["relation"].map(relation_ids)
        return GraphTables(nodes, edges, node_ids, relation_ids, conflicts)

    @staticmethod
    def _endpoint_attributes(raw: pd.DataFrame, endpoint: str) -> pd.DataFrame:
        prefix = f"{endpoint}_"
        return raw.loc[:, [
            endpoint, f"{prefix}Birth", f"{prefix}Death", f"{prefix}occ_level1",
            f"{prefix}occ_level2", f"{prefix}occ_level3", f"{prefix}Country",
        ]].rename(columns={
            endpoint: "node_id",
            f"{prefix}Birth": "birth_year",
            f"{prefix}Death": "death_year",
            f"{prefix}occ_level1": "occupation_level1",
            f"{prefix}occ_level2": "occupation_level2",
            f"{prefix}occ_level3": "occupation_level3",
            f"{prefix}Country": "country",
        })

    @staticmethod
    def _canonicalise_nodes(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collapse repeated endpoint attributes without per-node Python loops."""
        columns = [column for column in candidates.columns if column != "node_id"]
        grouped = candidates.groupby("node_id", sort=False)
        # Pandas GroupBy.first skips missing values, which is precisely the
        # canonicalisation policy. The vectorised implementation matters for
        # the roughly 1.4M endpoint records in the real export.
        nodes = grouped[columns].first().reset_index()

        conflict_rows: List[Dict[str, object]] = []
        uniqueness = grouped[columns].nunique(dropna=True)
        for column in columns:
            conflict_ids = uniqueness.index[uniqueness[column] > 1]
            if len(conflict_ids) == 0:
                continue
            conflicted = candidates.loc[
                candidates["node_id"].isin(conflict_ids), ["node_id", column]
            ]
            values_by_node = conflicted.groupby("node_id", sort=False)[column].agg(
                lambda values: list(pd.unique(values.dropna()))
            )
            conflict_rows.extend(
                {"node_id": node_id, "attribute": column, "values": values}
                for node_id, values in values_by_node.items()
            )
        return nodes, pd.DataFrame(conflict_rows)

    @staticmethod
    def _with_reverse_edges(edges: pd.DataFrame) -> pd.DataFrame:
        reverse = edges.rename(columns={"source": "target", "target": "source"}).copy()
        reverse["relation"] = reverse["relation"] + "__rev"
        return pd.concat([edges, reverse], ignore_index=True)


def make_numeric_features(nodes: pd.DataFrame, train_nodes: Optional[Iterable[str]] = None) -> np.ndarray:
    """Build normalised temporal attributes: birth, death, age and missing flags.

    If ``train_nodes`` is provided, mean/std are fit using only those nodes.
    Occupation is encoded separately because it has transductive masking rules.
    """
    birth = pd.to_numeric(nodes["birth_year"], errors="coerce").to_numpy(dtype=np.float32)
    death = pd.to_numeric(nodes["death_year"], errors="coerce").to_numpy(dtype=np.float32)
    age = death - birth
    raw = np.column_stack([birth, death, age])
    missing = ~np.isfinite(raw)

    if train_nodes is None:
        fit_rows = np.ones(len(nodes), dtype=bool)
    else:
        fit_rows = nodes["node_id"].isin(set(train_nodes)).to_numpy()
    mean = np.nanmean(raw[fit_rows], axis=0)
    std = np.nanstd(raw[fit_rows], axis=0)
    std[std == 0] = 1.0
    normalised = (np.where(missing, mean, raw) - mean) / std
    return np.column_stack([normalised, missing.astype(np.float32)]).astype(np.float32)


def encode_categorical(values: pd.Series, missing_token: str = "__MISSING__") -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode a categorical attribute with a stable, explicit missing ID."""
    clean = values.fillna(missing_token).astype(str)
    vocabulary = [missing_token] + sorted(value for value in clean.unique() if value != missing_token)
    mapping = {value: index for index, value in enumerate(vocabulary)}
    return clean.map(mapping).to_numpy(dtype=np.int64), mapping
