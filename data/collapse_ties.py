#!/usr/bin/env python3
"""Collapse a prepared graph's base relations into inherited/acquired ties.

The prepared graph remains directed: ``inherited_ties`` and ``acquired_ties``
each receive a generated ``__rev`` relation.  GraphMask's base-relation report
therefore contains exactly two rows per layer while the R-GCN can still learn
different weights for the two message directions.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from training.tie_taxonomy import load_tie_taxonomy, sha256_file


DIRECTED_RELATION_TO_ID = {
    "inherited_ties": 0,
    "inherited_ties__rev": 1,
    "acquired_ties": 2,
    "acquired_ties__rev": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Source prepared graph_data.pt")
    parser.add_argument(
        "--tie-taxonomy",
        required=True,
        help="Inherited/acquired taxonomy resolved against the source relation vocabulary",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collapsed_relation(relation: str, taxonomy) -> str:
    group = taxonomy.group_for_base_relation(relation)
    suffix = "__rev" if relation.endswith("__rev") else ""
    return f"{group}_ties{suffix}"


def collapse_tie_artifact(
    data_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
) -> Dict:
    data_path = data_path.resolve()
    taxonomy_path = taxonomy_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    data = copy.deepcopy(bundle["data"])
    metadata = copy.deepcopy(bundle["metadata"])
    source_relation_to_id = {
        str(relation): int(relation_id)
        for relation, relation_id in metadata["relation_to_id"].items()
    }
    taxonomy = load_tie_taxonomy(taxonomy_path, source_relation_to_id)

    old_id_to_relation = {
        relation_id: relation for relation, relation_id in source_relation_to_id.items()
    }
    lookup = torch.empty(max(old_id_to_relation) + 1, dtype=torch.long)
    for relation_id, relation in old_id_to_relation.items():
        lookup[relation_id] = DIRECTED_RELATION_TO_ID[_collapsed_relation(relation, taxonomy)]

    old_edge_index = data.edge_index.detach().cpu()
    old_edge_type = data.edge_type.detach().cpu().long()
    remapped_edge_type = lookup[old_edge_type]

    # Replacing several labels by one category can create identical messages.
    # Retain the first source/category/target occurrence, matching prepare.py's
    # original triple-deduplication policy.
    source_ids = old_edge_index[0].numpy().astype(np.int64, copy=False)
    target_ids = old_edge_index[1].numpy().astype(np.int64, copy=False)
    type_ids = remapped_edge_type.numpy().astype(np.int64, copy=False)
    keys = (
        (source_ids * int(data.num_nodes) + target_ids)
        * len(DIRECTED_RELATION_TO_ID)
        + type_ids
    )
    _, first_indices = np.unique(keys, return_index=True)
    keep_indices = np.sort(first_indices).astype(np.int64, copy=False)
    keep = torch.from_numpy(keep_indices)
    data.edge_index = old_edge_index[:, keep]
    data.edge_type = remapped_edge_type[keep]

    manifest = {
        "operation": "collapse_base_relations_to_inherited_acquired_ties",
        "source_data": str(data_path),
        "source_data_sha256": sha256_file(data_path),
        "source_taxonomy": taxonomy.manifest(),
        "base_relation_groups": {
            "inherited_ties": list(taxonomy.inherited),
            "acquired_ties": list(taxonomy.acquired),
        },
        "directed_relation_to_id": dict(DIRECTED_RELATION_TO_ID),
        "direction_policy": "preserve_generated_reverse_direction",
        "duplicate_policy": "deduplicate_source_collapsed_relation_target",
        "edges_before": int(old_edge_type.numel()),
        "edges_after": int(data.edge_type.numel()),
        "duplicates_removed": int(old_edge_type.numel() - data.edge_type.numel()),
    }
    metadata["source_relation_to_id_before_tie_collapse"] = source_relation_to_id
    metadata["relation_to_id"] = dict(DIRECTED_RELATION_TO_ID)
    metadata["num_relations"] = len(DIRECTED_RELATION_TO_ID)
    metadata["tie_relation_collapse"] = manifest
    torch.save({"data": data, "metadata": metadata}, output_dir / "graph_data.pt")

    source_edges_path = data_path.parent / "edges.csv"
    if not source_edges_path.is_file():
        raise FileNotFoundError(f"Missing source edge table: {source_edges_path}")
    edges = pd.read_csv(source_edges_path)
    if len(edges) != int(old_edge_type.numel()):
        raise ValueError(
            f"{source_edges_path} has {len(edges)} rows but graph_data.pt has "
            f"{old_edge_type.numel()} edges"
        )
    if not np.array_equal(edges["source_id"].to_numpy(dtype=np.int64), source_ids):
        raise ValueError("edges.csv source_id order differs from graph_data.pt")
    if not np.array_equal(edges["target_id"].to_numpy(dtype=np.int64), target_ids):
        raise ValueError("edges.csv target_id order differs from graph_data.pt")
    if not np.array_equal(
        edges["relation_id"].to_numpy(dtype=np.int64), old_edge_type.numpy()
    ):
        raise ValueError("edges.csv relation_id order differs from graph_data.pt")

    id_to_relation = {
        relation_id: relation for relation, relation_id in DIRECTED_RELATION_TO_ID.items()
    }
    edges = edges.iloc[keep_indices].copy()
    edges["relation_id"] = data.edge_type.numpy()
    edges["relation"] = edges["relation_id"].map(id_to_relation)
    edges.to_csv(output_dir / "edges.csv", index=False)
    edges["relation"].value_counts().rename_axis("relation").rename("count").to_csv(
        output_dir / "relation_stats.csv"
    )

    for filename in ("nodes.csv", "class_stats.csv", "attribute_conflicts.csv"):
        source = data_path.parent / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)

    source_summary = data_path.parent / "split_summary.json"
    summary = (
        json.loads(source_summary.read_text(encoding="utf-8"))
        if source_summary.is_file()
        else {}
    )
    summary.update({
        "relation_types": len(DIRECTED_RELATION_TO_ID),
        "base_relation_types": 2,
        "edges_after_binary_tie_collapse": int(data.edge_type.numel()),
        "tie_relation_collapse": manifest,
    })
    _write_json(output_dir / "split_summary.json", summary)
    _write_json(output_dir / "relation_collapse_manifest.json", manifest)
    _write_json(output_dir / "binary_tie_taxonomy.json", {
        "name": "binary_inherited_acquired_ties_v1",
        "version": 1,
        "description": "Artifact-specific taxonomy after binary tie collapse.",
        "groups": {
            "inherited": ["inherited_ties"],
            "acquired": ["acquired_ties"],
        },
    })
    return manifest


def main() -> None:
    args = parse_args()
    manifest = collapse_tie_artifact(
        Path(args.data), Path(args.tie_taxonomy), Path(args.output_dir)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
