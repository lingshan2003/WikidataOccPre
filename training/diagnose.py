#!/usr/bin/env python3
"""Diagnose graph coverage and relation-level occupation signal.

This is an analysis-only command: it reads a prepared artifact and optional
prediction CSV, writes CSV/JSON reports, and never trains a model or modifies
the graph artifact.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score, normalized_mutual_info_score

from training.relation_controls import RELATION_GROUPS, base_relation_name, relation_pair_keys
from training.tie_taxonomy import (
    DEFAULT_TIE_TAXONOMY_PATH,
    TieTaxonomy,
    load_tie_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/graph_data.pt")
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument(
        "--predictions",
        help="Optional test_predictions.csv from a completed train run; enables accuracy/F1-by-coverage reports",
    )
    parser.add_argument(
        "--homophily-label-split",
        choices=["train", "all"],
        default="train",
        help="Use only visible training labels (recommended) or all retained labels for relation homophily",
    )
    parser.add_argument("--min-relation-support", type=int, default=20)
    parser.add_argument(
        "--tie-taxonomy",
        default=str(DEFAULT_TIE_TAXONOMY_PATH),
        help="Versioned inherited/acquired taxonomy JSON used for tie-group audits",
    )
    return parser.parse_args()


def degree_arrays(edge_index: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed out/in degree and distinct undirected-neighbour degree."""
    source, target = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    out_degree = np.bincount(source, minlength=num_nodes)
    in_degree = np.bincount(target, minlength=num_nodes)
    low, high = np.minimum(source, target), np.maximum(source, target)
    keep = low != high
    pair_keys = np.unique((low[keep] * num_nodes) + high[keep])
    if pair_keys.size == 0:
        return out_degree, in_degree, np.zeros(num_nodes, dtype=np.int64)
    pair_source, pair_target = pair_keys // num_nodes, pair_keys % num_nodes
    neighbour_degree = np.bincount(
        np.concatenate([pair_source, pair_target]), minlength=num_nodes
    )
    return out_degree, in_degree, neighbour_degree


def component_arrays(edge_index: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return component ID and component size per node using unique undirected pairs."""
    source, target = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    low, high = np.minimum(source, target), np.maximum(source, target)
    keep = low != high
    pair_keys = np.unique((low[keep] * num_nodes) + high[keep])
    if pair_keys.size == 0:
        components = np.arange(num_nodes, dtype=np.int64)
        return components, np.ones(num_nodes, dtype=np.int64)
    pair_source, pair_target = pair_keys // num_nodes, pair_keys % num_nodes
    adjacency = coo_matrix(
        (np.ones(pair_source.size * 2, dtype=np.uint8),
         (np.concatenate([pair_source, pair_target]), np.concatenate([pair_target, pair_source]))),
        shape=(num_nodes, num_nodes),
    )
    _, component_id = connected_components(adjacency.tocsr(), directed=False, return_labels=True)
    component_counts = np.bincount(component_id)
    return component_id.astype(np.int64), component_counts[component_id].astype(np.int64)


def visible_occupation_coverage(
    edge_index: torch.Tensor, visible_train_nodes: np.ndarray, num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Count direct visible neighbours and mark reachability within two message-passing hops."""
    source, target = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    direct_count = np.bincount(target[visible_train_nodes[source]], minlength=num_nodes)
    one_hop = np.zeros(num_nodes, dtype=bool)
    one_hop[target[visible_train_nodes[source]]] = True
    two_hop = np.zeros(num_nodes, dtype=bool)
    two_hop[target[one_hop[source]]] = True
    return direct_count, one_hop | two_hop


def relation_base_ids(relation_to_id: Mapping[str, int]) -> Tuple[np.ndarray, Dict[int, str]]:
    """Map directed relation IDs to base relation IDs and names."""
    max_relation_id = max(relation_to_id.values())
    base_names = sorted({base_relation_name(name) for name in relation_to_id})
    base_to_id = {name: index for index, name in enumerate(base_names)}
    directed_to_base = np.full(max_relation_id + 1, -1, dtype=np.int64)
    for relation, relation_id in relation_to_id.items():
        directed_to_base[int(relation_id)] = base_to_id[base_relation_name(relation)]
    return directed_to_base, {index: name for name, index in base_to_id.items()}


def group_for_relation(relation: str) -> str:
    for group, members in RELATION_GROUPS.items():
        if relation in members:
            return group
    return "other"


def relation_homophily(
    data,
    relation_to_id: Mapping[str, int],
    label_mask: np.ndarray,
    min_support: int,
    tie_taxonomy: Optional[TieTaxonomy] = None,
) -> pd.DataFrame:
    """Estimate label agreement and mutual information per base relation.

    Reverse-edge duplicates are collapsed first.  By default callers pass the
    training mask, so held-out labels do not influence this diagnostic.
    """
    if min_support < 1:
        raise ValueError("--min-relation-support must be positive")
    source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    labels = data.y.detach().cpu().numpy().astype(np.int64, copy=False)
    directed_to_base, base_id_to_name = relation_base_ids(relation_to_id)
    edge_relation_ids = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_base_ids = directed_to_base[edge_relation_ids]
    if (edge_base_ids < 0).any():
        raise ValueError("edge_type contains a relation ID absent from metadata")
    pair_keys = relation_pair_keys(data, relation_to_id)
    global_labels = labels[label_mask & (labels >= 0)]
    if global_labels.size == 0:
        raise ValueError("No retained labels are available for homophily analysis")
    _, global_counts = np.unique(global_labels, return_counts=True)
    independent_same_label_rate = float(np.square(global_counts / global_counts.sum()).sum())

    valid_edge = label_mask[source] & label_mask[target] & (labels[source] >= 0) & (labels[target] >= 0)
    records: List[Dict[str, object]] = []
    for base_id, relation in base_id_to_name.items():
        indices = np.flatnonzero(valid_edge & (edge_base_ids == base_id))
        if indices.size == 0:
            continue
        # Selecting a single representative removes generated reverse edges.
        _, first = np.unique(pair_keys[indices], return_index=True)
        indices = indices[first]
        if indices.size < min_support:
            continue
        source_labels, target_labels = labels[source[indices]], labels[target[indices]]
        same_label_rate = float(np.mean(source_labels == target_labels))
        records.append({
            "relation": relation,
            "relation_group": group_for_relation(relation),
            "tie_group": tie_taxonomy.group_for_base_relation(relation) if tie_taxonomy else None,
            "labeled_train_pairs": int(indices.size),
            "same_label_rate": same_label_rate,
            "independent_same_label_rate": independent_same_label_rate,
            "same_label_lift": same_label_rate / independent_same_label_rate
            if independent_same_label_rate else float("nan"),
            "mutual_information": float(mutual_info_score(source_labels, target_labels)),
            "normalized_mutual_information": float(normalized_mutual_info_score(source_labels, target_labels)),
        })
    return pd.DataFrame(records).sort_values(
        ["same_label_lift", "labeled_train_pairs"], ascending=[False, False]
    ).reset_index(drop=True) if records else pd.DataFrame(columns=[
        "relation", "relation_group", "labeled_train_pairs", "same_label_rate",
        "tie_group", "independent_same_label_rate", "same_label_lift", "mutual_information",
        "normalized_mutual_information",
    ])


def _tie_groups_for_edges(
    edge_relation_ids: np.ndarray,
    relation_to_id: Mapping[str, int],
    tie_taxonomy: TieTaxonomy,
) -> np.ndarray:
    """Return the inherited/acquired label for each directed edge relation ID."""
    max_relation_id = max(relation_to_id.values())
    by_relation_id = np.full(max_relation_id + 1, "", dtype=object)
    for relation, relation_id in relation_to_id.items():
        by_relation_id[int(relation_id)] = tie_taxonomy.group_for_base_relation(relation)
    if edge_relation_ids.size and (
        edge_relation_ids.min() < 0 or edge_relation_ids.max() >= len(by_relation_id)
    ):
        raise ValueError("edge_type contains a relation ID absent from metadata")
    result = by_relation_id[edge_relation_ids]
    if np.any(result == ""):
        raise ValueError("Tie taxonomy did not cover every edge relation ID")
    return result


def tie_group_coverage(
    data,
    relation_to_id: Mapping[str, int],
    tie_taxonomy: TieTaxonomy,
    visible_train_nodes: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Audit direct inherited/acquired messages and visible training neighbours.

    Counts retain the graph's directed message semantics.  The accompanying
    relation-pair count collapses generated reverse edges, so density controls
    can be checked against the same unit that training removes at random.
    """
    source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    relation_ids = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    groups = _tie_groups_for_edges(relation_ids, relation_to_id, tie_taxonomy)
    pair_keys = relation_pair_keys(data, relation_to_id)
    num_nodes = int(data.num_nodes)
    columns: Dict[str, np.ndarray] = {}
    records: List[Dict[str, object]] = []
    for group in ("inherited", "acquired"):
        edge_mask = groups == group
        visible_mask = edge_mask & visible_train_nodes[source]
        direct_count = np.bincount(target[edge_mask], minlength=num_nodes)
        visible_count = np.bincount(target[visible_mask], minlength=num_nodes)
        columns[f"{group}_direct_messages"] = direct_count.astype(np.int64, copy=False)
        columns[f"{group}_visible_train_occupation_messages"] = visible_count.astype(np.int64, copy=False)
        incident_nodes = (
            np.unique(np.concatenate([source[edge_mask], target[edge_mask]]))
            if edge_mask.any() else np.empty(0, dtype=np.int64)
        )
        records.append({
            "tie_group": group,
            "directed_edges": int(edge_mask.sum()),
            "relation_pairs": int(np.unique(pair_keys[edge_mask]).size),
            "distinct_incident_nodes": int(incident_nodes.size),
            "nodes_with_direct_message": int((direct_count > 0).sum()),
            "visible_train_source_messages": int(visible_mask.sum()),
            "nodes_with_visible_train_occupation_message": int((visible_count > 0).sum()),
        })
    inherited_visible = columns["inherited_visible_train_occupation_messages"] > 0
    acquired_visible = columns["acquired_visible_train_occupation_messages"] > 0
    exposure = np.full(num_nodes, "neither", dtype=object)
    exposure[inherited_visible & ~acquired_visible] = "inherited_only"
    exposure[~inherited_visible & acquired_visible] = "acquired_only"
    exposure[inherited_visible & acquired_visible] = "both"
    columns["tie_exposure"] = exposure
    return columns, pd.DataFrame(records)


def tie_group_homophily(
    data,
    relation_to_id: Mapping[str, int],
    tie_taxonomy: TieTaxonomy,
    label_mask: np.ndarray,
    min_support: int,
) -> pd.DataFrame:
    """Estimate train-label agreement for each complete tie category."""
    source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    relation_ids = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    labels = data.y.detach().cpu().numpy().astype(np.int64, copy=False)
    groups = _tie_groups_for_edges(relation_ids, relation_to_id, tie_taxonomy)
    pair_keys = relation_pair_keys(data, relation_to_id)
    global_labels = labels[label_mask & (labels >= 0)]
    if global_labels.size == 0:
        raise ValueError("No retained labels are available for tie-group homophily analysis")
    _, global_counts = np.unique(global_labels, return_counts=True)
    independent_same_label_rate = float(np.square(global_counts / global_counts.sum()).sum())
    valid = label_mask[source] & label_mask[target] & (labels[source] >= 0) & (labels[target] >= 0)
    records: List[Dict[str, object]] = []
    for group in ("inherited", "acquired"):
        indices = np.flatnonzero(valid & (groups == group))
        if indices.size:
            _, first = np.unique(pair_keys[indices], return_index=True)
            indices = indices[first]
        source_labels, target_labels = labels[source[indices]], labels[target[indices]]
        record: Dict[str, object] = {
            "tie_group": group,
            "labeled_train_pairs": int(indices.size),
            "independent_same_label_rate": independent_same_label_rate,
            "same_label_rate": None,
            "same_label_lift": None,
            "mutual_information": None,
            "normalized_mutual_information": None,
        }
        if indices.size >= min_support:
            same_label_rate = float(np.mean(source_labels == target_labels))
            record.update({
                "same_label_rate": same_label_rate,
                "same_label_lift": same_label_rate / independent_same_label_rate,
                "mutual_information": float(mutual_info_score(source_labels, target_labels)),
                "normalized_mutual_information": float(normalized_mutual_info_score(source_labels, target_labels)),
            })
        records.append(record)
    return pd.DataFrame(records)


def value_bucket(values: np.ndarray, kind: str) -> np.ndarray:
    """Fixed, interpretable buckets used for degree and visible-neighbour strata."""
    if kind == "degree":
        labels = np.full(values.shape, "11+", dtype=object)
        labels[values <= 10] = "6-10"
        labels[values <= 5] = "3-5"
    elif kind == "visible_neighbours":
        labels = np.full(values.shape, "6+", dtype=object)
        labels[values <= 5] = "3-5"
    else:
        raise ValueError(f"Unknown bucket kind: {kind}")
    labels[values == 2] = "2"
    labels[values == 1] = "1"
    labels[values == 0] = "0"
    return labels


def prediction_strata(predictions: pd.DataFrame, node_report: pd.DataFrame) -> pd.DataFrame:
    """Report optional prediction quality by graph degree and label visibility."""
    required = {"node_index", "true_label", "prediction", "confidence"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["node_index"] = predictions["node_index"].astype(np.int64)
    if predictions["node_index"].duplicated().any():
        raise ValueError("Prediction CSV contains duplicate node_index values")
    report = predictions.merge(node_report, on="node_index", how="left", validate="one_to_one")
    if report[["distinct_neighbour_degree", "visible_train_occupation_neighbours"]].isna().any().any():
        raise ValueError("Prediction CSV contains node indices absent from the graph artifact")
    records: List[Dict[str, object]] = []
    for name, bucket in {
        "distinct_neighbour_degree": value_bucket(report["distinct_neighbour_degree"].to_numpy(), "degree"),
        "visible_train_occupation_neighbours": value_bucket(
            report["visible_train_occupation_neighbours"].to_numpy(), "visible_neighbours"
        ),
        "within_two_hops_of_visible_train_occupation": report[
            "within_two_hops_of_visible_train_occupation"
        ].map({True: "yes", False: "no"}).to_numpy(),
        "tie_exposure": report["tie_exposure"].to_numpy(),
    }.items():
        grouped = report.assign(_bucket=bucket).groupby("_bucket", sort=True)
        for bucket_name, rows in grouped:
            records.append({
                "stratifier": name,
                "bucket": bucket_name,
                "nodes": int(len(rows)),
                "accuracy": float(accuracy_score(rows["true_label"], rows["prediction"])),
                "macro_f1": float(f1_score(rows["true_label"], rows["prediction"], average="macro", zero_division=0)),
                "mean_confidence": float(rows["confidence"].mean()),
            })
    return pd.DataFrame(records)


def split_coverage_summary(node_report: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Summarise degree and visible occupation coverage separately for each split."""
    result: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test", "unlabeled"):
        rows = node_report[node_report["split"] == split]
        if rows.empty:
            continue
        result[split] = {
            "nodes": int(len(rows)),
            "median_distinct_neighbour_degree": float(rows["distinct_neighbour_degree"].median()),
            "share_zero_distinct_neighbours": float((rows["distinct_neighbour_degree"] == 0).mean()),
            "share_without_visible_training_occupation_neighbour": float(
                (rows["visible_train_occupation_neighbours"] == 0).mean()
            ),
            "share_within_two_hops_of_visible_training_occupation": float(
                rows["within_two_hops_of_visible_train_occupation"].mean()
            ),
            **{
                f"share_tie_exposure_{exposure}": float((rows["tie_exposure"] == exposure).mean())
                for exposure in ("neither", "inherited_only", "acquired_only", "both")
            },
        }
    return result


def main() -> None:
    args = parse_args()
    if args.min_relation_support < 1:
        raise ValueError("--min-relation-support must be positive")
    bundle = torch.load(args.data, map_location="cpu", weights_only=False)
    data, metadata = bundle["data"], bundle["metadata"]
    required = {"edge_index", "edge_type", "y", "train_mask", "val_mask", "test_mask"}
    if not all(hasattr(data, name) for name in required):
        raise ValueError("Expected a current prepared node-classification artifact")
    target_feature = metadata.get("target_column")
    if not target_feature or not hasattr(data, target_feature):
        raise ValueError("Artifact is missing the target occupation feature required for coverage diagnostics")
    if target_feature not in metadata.get("occupation_unknown_ids", {}):
        raise ValueError("Artifact is missing occupation unknown-ID metadata")
    tie_taxonomy = load_tie_taxonomy(args.tie_taxonomy, metadata["relation_to_id"])

    num_nodes = int(data.num_nodes)
    out_degree, in_degree, neighbour_degree = degree_arrays(data.edge_index, num_nodes)
    component_id, component_size = component_arrays(data.edge_index, num_nodes)
    visible_train_nodes = (
        data.train_mask.detach().cpu().numpy()
        & (data.y.detach().cpu().numpy() >= 0)
        & (getattr(data, target_feature).detach().cpu().numpy()
           != int(metadata["occupation_unknown_ids"][target_feature]))
    )
    visible_neighbours, within_two_hops = visible_occupation_coverage(
        data.edge_index, visible_train_nodes, num_nodes
    )
    tie_columns, tie_edge_summary = tie_group_coverage(
        data, metadata["relation_to_id"], tie_taxonomy, visible_train_nodes
    )
    split = np.full(num_nodes, "unlabeled", dtype=object)
    split[data.train_mask.detach().cpu().numpy()] = "train"
    split[data.val_mask.detach().cpu().numpy()] = "val"
    split[data.test_mask.detach().cpu().numpy()] = "test"
    node_report = pd.DataFrame({
        "node_index": np.arange(num_nodes, dtype=np.int64),
        "split": split,
        "directed_out_degree": out_degree,
        "directed_in_degree": in_degree,
        "distinct_neighbour_degree": neighbour_degree,
        "component_id": component_id,
        "component_size": component_size,
        "visible_train_occupation_neighbours": visible_neighbours,
        "within_two_hops_of_visible_train_occupation": within_two_hops,
        **tie_columns,
    })
    label_mask = data.train_mask.detach().cpu().numpy()
    if args.homophily_label_split == "all":
        label_mask = data.y.detach().cpu().numpy() >= 0
    homophily = relation_homophily(
        data, metadata["relation_to_id"], label_mask, args.min_relation_support, tie_taxonomy
    )
    tie_homophily = tie_group_homophily(
        data, metadata["relation_to_id"], tie_taxonomy, label_mask, args.min_relation_support
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    node_report.to_csv(output_dir / "node_diagnostics.csv", index=False)
    homophily.to_csv(output_dir / "relation_homophily.csv", index=False)
    tie_edge_summary.to_csv(output_dir / "tie_group_edge_summary.csv", index=False)
    tie_homophily.to_csv(output_dir / "tie_group_homophily.csv", index=False)
    summary = {
        "data": str(args.data),
        "target_column": target_feature,
        "nodes": num_nodes,
        "directed_edges": int(data.edge_index.size(1)),
        "directed_edges_per_node": float(data.edge_index.size(1) / num_nodes),
        "unique_undirected_neighbour_pairs": int(neighbour_degree.sum() // 2),
        "largest_component_nodes": int(component_size.max()),
        "largest_component_share": float(component_size.max() / num_nodes),
        "connected_components": int(np.unique(component_id).size),
        "visible_training_occupation_nodes": int(visible_train_nodes.sum()),
        "tie_taxonomy": tie_taxonomy.manifest(),
        "homophily_label_split": args.homophily_label_split,
        "min_relation_support": args.min_relation_support,
        "split_coverage": split_coverage_summary(node_report),
        "tie_group_edge_summary": str(output_dir / "tie_group_edge_summary.csv"),
        "tie_group_homophily": str(output_dir / "tie_group_homophily.csv"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if args.predictions:
        prediction_report = prediction_strata(pd.read_csv(args.predictions), node_report)
        prediction_report.to_csv(output_dir / "prediction_strata.csv", index=False)
        summary["predictions"] = str(args.predictions)
        summary["prediction_strata"] = str(output_dir / "prediction_strata.csv")
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
