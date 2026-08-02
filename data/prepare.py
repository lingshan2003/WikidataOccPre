#!/usr/bin/env python3
"""Prepare a leak-safe PyG graph for occupation prediction.

The script is intentionally separate from training. It makes every irreversible
data decision -- label filtering, node split, reverse relations and feature
encoding -- inspectable and reproducible before a GPU job is started.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .extended import ExtendedGraphLoader, encode_categorical, make_numeric_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Q_R_Q_extended.txt", help="Extended CSV edge export")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for prepared artifacts")
    parser.add_argument("--target-level", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=20,
        help="Classes with fewer labeled people remain in the graph but receive y=-1",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(args: argparse.Namespace) -> None:
    ratios = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratios, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    if args.min_class_count < 3:
        raise ValueError("min-class-count must be at least 3 for three-way splitting")


def stratified_node_split(
    class_ids: np.ndarray,
    eligible_mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split labeled nodes class-by-class while retaining every class in train."""
    train = np.zeros(len(class_ids), dtype=bool)
    val = np.zeros(len(class_ids), dtype=bool)
    test = np.zeros(len(class_ids), dtype=bool)
    rng = np.random.default_rng(seed)

    for class_id in np.unique(class_ids[eligible_mask]):
        members = np.flatnonzero(eligible_mask & (class_ids == class_id))
        rng.shuffle(members)
        count = len(members)

        # Eligible classes have >= min_class_count >= 3.  Preserve one example
        # in each split, then allocate the remaining nodes by requested ratio.
        requested_val = max(1, int(round(count * val_ratio)))
        requested_test = max(1, int(round(count * test_ratio)))
        val_count = min(requested_val, count - 2)
        test_count = min(requested_test, count - val_count - 1)
        train_count = count - val_count - test_count
        if train_count < 1:
            raise RuntimeError(f"Could not keep class {class_id} in training")

        train[members[:train_count]] = True
        val[members[train_count:train_count + val_count]] = True
        test[members[train_count + val_count:]] = True

    return train, val, test


def encode_labels(nodes: pd.DataFrame, target_column: str, min_class_count: int):
    raw_labels = nodes[target_column]
    counts = raw_labels.value_counts(dropna=True)
    retained_labels = sorted(counts[counts >= min_class_count].index.astype(str).tolist())
    label_to_id = {label: index for index, label in enumerate(retained_labels)}

    clean = raw_labels.astype("string")
    known = clean.notna() & clean.isin(label_to_id)
    class_ids = np.full(len(nodes), -1, dtype=np.int64)
    class_ids[known.to_numpy()] = clean[known].map(label_to_id).to_numpy(dtype=np.int64)
    return class_ids, known.to_numpy(), label_to_id, counts


def build_pyg_data(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    class_ids: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
):
    country, country_to_id = encode_categorical(nodes["country"])
    temporal = make_numeric_features(nodes, nodes.loc[train_mask, "node_id"])

    edge_index = torch.tensor(
        edges[["source_id", "target_id"]].to_numpy(dtype=np.int64).T,
        dtype=torch.long,
    )
    data = Data(
        edge_index=edge_index,
        edge_type=torch.tensor(edges["relation_id"].to_numpy(dtype=np.int64), dtype=torch.long),
        y=torch.tensor(class_ids, dtype=torch.long),
        num_nodes=len(nodes),
    )
    data.country = torch.tensor(country, dtype=torch.long)
    data.temporal = torch.tensor(temporal, dtype=torch.float)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    return data, country_to_id


def write_tables(output_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, conflicts: pd.DataFrame) -> None:
    """Use CSV intentionally: it needs no optional PyArrow dependency."""
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)
    conflicts.to_csv(output_dir / "attribute_conflicts.csv", index=False)


def main() -> None:
    args = parse_args()
    validate_ratios(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input} ...")
    tables = ExtendedGraphLoader(args.input).read()
    edges = tables.edges.drop_duplicates(
        subset=["source_id", "relation_id", "target_id"], keep="first"
    ).reset_index(drop=True)
    target_column = f"occupation_level{args.target_level}"
    class_ids, eligible, label_to_id, all_counts = encode_labels(
        tables.nodes, target_column, args.min_class_count
    )
    train_mask, val_mask, test_mask = stratified_node_split(
        class_ids,
        eligible,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    data, country_to_id = build_pyg_data(
        tables.nodes, edges, class_ids, train_mask, val_mask, test_mask
    )

    torch.save(
        {
            "data": data,
            "metadata": {
                "target_column": target_column,
                "num_relations": len(tables.relation_to_id),
                "num_classes": len(label_to_id),
                "feature_schema": {
                    "country": {"kind": "categorical", "cardinality": len(country_to_id)},
                    "temporal": {"kind": "numeric", "input_dim": int(data.temporal.size(1))},
                },
                "country_to_id": country_to_id,
                "label_to_id": label_to_id,
                "relation_to_id": tables.relation_to_id,
                "seed": args.seed,
            },
        },
        output_dir / "graph_data.pt",
    )
    write_tables(output_dir, tables.nodes, edges, tables.attribute_conflicts)

    retained_counts = all_counts[all_counts.index.astype(str).isin(label_to_id)]
    retained_counts.rename_axis("occupation").rename("count").to_csv(
        output_dir / "class_stats.csv"
    )
    relation_counts = edges["relation"].value_counts().rename_axis("relation").rename("count")
    relation_counts.to_csv(output_dir / "relation_stats.csv")
    report = {
        "input": str(args.input),
        "target_column": target_column,
        "nodes": int(data.num_nodes),
        "edges_after_reverse_and_deduplication": int(data.edge_index.size(1)),
        "relation_types": len(tables.relation_to_id),
        "target_classes_retained": len(label_to_id),
        "labeled_nodes_retained": int(eligible.sum()),
        "ignored_unlabeled_or_rare_nodes": int((~eligible).sum()),
        "train_nodes": int(train_mask.sum()),
        "val_nodes": int(val_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "attribute_conflicts": int(len(tables.attribute_conflicts)),
        "seed": args.seed,
        "min_class_count": args.min_class_count,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Prepared graph saved to {output_dir / 'graph_data.pt'}")


if __name__ == "__main__":
    main()
