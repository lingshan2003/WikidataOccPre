#!/usr/bin/env python3
"""Build a heterogeneous Person--Occupation graph for link prediction.

Person--person social edges are always retained.  Only occupation edges for
training people are part of the message-passing graph; validation and test
occupation edges are held out as prediction targets.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from data.extended import ExtendedGraphLoader, encode_categorical, make_numeric_features


PERSON = "person"
OCCUPATION = "occupation"
HAS_OCCUPATION = (PERSON, "has_occupation", OCCUPATION)
REV_HAS_OCCUPATION = (OCCUPATION, "rev_has_occupation", PERSON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="Q_R_Q_extended.txt")
    parser.add_argument("--output-dir", default="link_artifacts/level3")
    parser.add_argument("--target-level", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_split(
    labels: np.ndarray, eligible: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    masks = [np.zeros(len(labels), dtype=bool) for _ in range(3)]
    for label in np.unique(labels[eligible]):
        members = np.flatnonzero(eligible & (labels == label))
        if len(members) < 3:
            raise ValueError("Every retained occupation needs at least three people")
        rng.shuffle(members)
        val_count = min(max(1, round(len(members) * val_ratio)), len(members) - 2)
        test_count = min(max(1, round(len(members) * test_ratio)), len(members) - val_count - 1)
        train_count = len(members) - val_count - test_count
        masks[0][members[:train_count]] = True
        masks[1][members[train_count:train_count + val_count]] = True
        masks[2][members[train_count + val_count:]] = True
    return tuple(masks)  # type: ignore[return-value]


def encode_occupations(nodes: pd.DataFrame, target_column: str, min_class_count: int):
    counts = nodes[target_column].value_counts(dropna=True)
    vocabulary = sorted(counts[counts >= min_class_count].index.astype(str).tolist())
    label_to_id = {label: index for index, label in enumerate(vocabulary)}
    raw = nodes[target_column].astype("string")
    eligible = raw.notna() & raw.isin(label_to_id)
    labels = np.full(len(nodes), -1, dtype=np.int64)
    labels[eligible.to_numpy()] = raw[eligible].map(label_to_id).to_numpy(dtype=np.int64)
    return labels, eligible.to_numpy(), label_to_id, counts


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, labels: np.ndarray, train, val, test):
    data = HeteroData()
    country, country_to_id = encode_categorical(nodes["country"])
    data[PERSON].num_nodes = len(nodes)
    data[PERSON].country = torch.tensor(country, dtype=torch.long)
    data[PERSON].temporal = torch.tensor(
        make_numeric_features(nodes, nodes.loc[train, "node_id"]), dtype=torch.float
    )
    data[PERSON].y = torch.tensor(labels, dtype=torch.long)
    data[PERSON].train_mask = torch.tensor(train, dtype=torch.bool)
    data[PERSON].val_mask = torch.tensor(val, dtype=torch.bool)
    data[PERSON].test_mask = torch.tensor(test, dtype=torch.bool)

    for relation_id, relation_edges in edges.groupby("relation_id", sort=False):
        edge_index = relation_edges[["source_id", "target_id"]].to_numpy(dtype=np.int64).T
        # HeteroData uses relation names as type keys. Avoid raw ``__rev``
        # names, since double underscores have special meaning in PyG.
        data[PERSON, f"social_{int(relation_id)}", PERSON].edge_index = torch.tensor(
            np.ascontiguousarray(edge_index), dtype=torch.long
        ).contiguous()

    num_occupations = int(labels.max()) + 1
    data[OCCUPATION].num_nodes = num_occupations
    pairs = np.vstack([np.flatnonzero(labels >= 0), labels[labels >= 0]])
    train_pairs = pairs[:, train[pairs[0]]]
    val_pairs = pairs[:, val[pairs[0]]]
    test_pairs = pairs[:, test[pairs[0]]]
    data[HAS_OCCUPATION].edge_index = torch.tensor(
        np.ascontiguousarray(train_pairs), dtype=torch.long
    ).contiguous()
    data[REV_HAS_OCCUPATION].edge_index = torch.tensor(
        np.ascontiguousarray(train_pairs[[1, 0]]), dtype=torch.long
    ).contiguous()
    return data, country_to_id, train_pairs, val_pairs, test_pairs


def main() -> None:
    args = parse_args()
    if args.min_class_count < 3:
        raise ValueError("--min-class-count must be at least 3")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.input} ...")
    tables = ExtendedGraphLoader(args.input).read()
    edges = tables.edges.drop_duplicates(
        subset=["source_id", "relation_id", "target_id"], keep="first"
    ).reset_index(drop=True)
    target_column = f"occupation_level{args.target_level}"
    labels, eligible, label_to_id, counts = encode_occupations(
        tables.nodes, target_column, args.min_class_count
    )
    train, val, test = stratified_split(
        labels, eligible, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    graph, country_to_id, train_pairs, val_pairs, test_pairs = build_graph(
        tables.nodes, edges, labels, train, val, test
    )
    bundle = {
        "data": graph,
        "splits": {
            "train_pos": torch.tensor(train_pairs, dtype=torch.long),
            "val_pos": torch.tensor(val_pairs, dtype=torch.long),
            "test_pos": torch.tensor(test_pairs, dtype=torch.long),
        },
        "metadata": {
            "target_column": target_column,
            "num_occupations": len(label_to_id),
            "label_to_id": label_to_id,
            "country_to_id": country_to_id,
            "social_relation_to_id": {
                str(relation_id): relation
                for relation_id, relation in edges[["relation_id", "relation"]].drop_duplicates().itertuples(index=False)
            },
            "link_edge_type": HAS_OCCUPATION,
            "reverse_link_edge_type": REV_HAS_OCCUPATION,
            "seed": args.seed,
        },
    }
    torch.save(bundle, output_dir / "hetero_graph.pt")
    tables.nodes.to_csv(output_dir / "persons.csv", index=False)
    pd.DataFrame({"occupation_id": range(len(label_to_id)), "occupation": list(label_to_id)}).to_csv(
        output_dir / "occupations.csv", index=False
    )
    report = {
        "target_column": target_column,
        "person_nodes": int(graph[PERSON].num_nodes),
        "occupation_nodes": len(label_to_id),
        "social_edges": int(len(edges)),
        "social_relation_types": int(len(edges["relation"].unique())),
        "train_occupation_edges": int(train_pairs.shape[1]),
        "val_occupation_edges": int(val_pairs.shape[1]),
        "test_occupation_edges": int(test_pairs.shape[1]),
        "protocol": "Only train Person--Occupation edges are in the message-passing graph; all person--person social edges remain.",
        "seed": args.seed,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    counts.rename_axis("occupation").rename("count").to_csv(output_dir / "raw_class_counts.csv")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Heterogeneous graph saved to {output_dir / 'hetero_graph.pt'}")


if __name__ == "__main__":
    main()
