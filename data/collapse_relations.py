#!/usr/bin/env python3
"""Collapse a prepared graph according to a complete multi-group taxonomy.

Each source base relation is replaced by its taxonomy group while generated
``__rev`` directions remain distinct.  This is useful when the model itself,
rather than only a post-hoc report, should operate on a coarser relation
vocabulary such as the acquired-tie subgroups.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from training.tie_taxonomy import load_relation_taxonomy, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Source prepared graph_data.pt")
    parser.add_argument(
        "--relation-taxonomy",
        required=True,
        help="Complete multi-group taxonomy resolved against the source relation vocabulary",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ordered_groups(groups: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Keep inherited first for readability and make the remaining IDs stable."""
    names = sorted(groups)
    if "inherited" in names:
        names.remove("inherited")
        names.insert(0, "inherited")
    return tuple(names)


def _directed_relation_mapping(group_names: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for group in group_names:
        if group.endswith("__rev"):
            raise ValueError("Taxonomy group names may not end with '__rev'")
        mapping[group] = len(mapping)
        mapping[f"{group}__rev"] = len(mapping)
    return mapping


def _collapsed_relation(relation: str, taxonomy) -> str:
    group = taxonomy.group_for_base_relation(relation)
    return f"{group}__rev" if relation.endswith("__rev") else group


def collapse_relation_artifact(
    data_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Create a standalone graph artifact whose relation types are taxonomy groups."""
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
    taxonomy = load_relation_taxonomy(taxonomy_path, source_relation_to_id)
    group_names = _ordered_groups(taxonomy.groups)
    if len(group_names) < 2:
        raise ValueError("Relation collapse requires at least two non-empty taxonomy groups")
    empty_groups = [group for group in group_names if not taxonomy.groups[group]]
    if empty_groups:
        raise ValueError(
            "Every collapsed relation group must contain at least one source relation; "
            f"empty groups={empty_groups}"
        )
    if "inherited" not in group_names:
        raise ValueError(
            "A training-compatible collapsed artifact requires an 'inherited' group"
        )
    directed_relation_to_id = _directed_relation_mapping(group_names)

    old_id_to_relation = {
        relation_id: relation for relation, relation_id in source_relation_to_id.items()
    }
    expected_ids = set(range(len(old_id_to_relation)))
    if set(old_id_to_relation) != expected_ids:
        raise ValueError(
            "Source relation IDs must be consecutive from zero before relation collapse"
        )
    lookup = torch.empty(len(old_id_to_relation), dtype=torch.long)
    for relation_id, relation in old_id_to_relation.items():
        lookup[relation_id] = directed_relation_to_id[
            _collapsed_relation(relation, taxonomy)
        ]

    old_edge_index = data.edge_index.detach().cpu()
    old_edge_type = data.edge_type.detach().cpu().long()
    remapped_edge_type = lookup[old_edge_type]

    # Multiple source labels can become the same source/group/target message.
    # Match data preparation's first-occurrence triple deduplication policy.
    source_ids = old_edge_index[0].numpy().astype(np.int64, copy=False)
    target_ids = old_edge_index[1].numpy().astype(np.int64, copy=False)
    type_ids = remapped_edge_type.numpy().astype(np.int64, copy=False)
    keys = (
        (source_ids * int(data.num_nodes) + target_ids)
        * len(directed_relation_to_id)
        + type_ids
    )
    _, first_indices = np.unique(keys, return_index=True)
    keep_indices = np.sort(first_indices).astype(np.int64, copy=False)
    keep = torch.from_numpy(keep_indices)
    data.edge_index = old_edge_index[:, keep]
    data.edge_type = remapped_edge_type[keep]

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
        relation_id: relation
        for relation, relation_id in directed_relation_to_id.items()
    }
    edges = edges.iloc[keep_indices].copy()
    edges["relation_id"] = data.edge_type.numpy()
    edges["relation"] = edges["relation_id"].map(id_to_relation)

    manifest: Dict[str, object] = {
        "operation": "collapse_base_relations_to_taxonomy_groups",
        "source_data": str(data_path),
        "source_data_sha256": sha256_file(data_path),
        "source_taxonomy": taxonomy.manifest(),
        "base_relation_groups": {
            group: list(taxonomy.groups[group]) for group in group_names
        },
        "directed_relation_to_id": directed_relation_to_id,
        "direction_policy": "preserve_generated_reverse_direction",
        "duplicate_policy": "deduplicate_source_collapsed_relation_target",
        "edges_before": int(old_edge_type.numel()),
        "edges_after": int(data.edge_type.numel()),
        "duplicates_removed": int(old_edge_type.numel() - data.edge_type.numel()),
    }
    metadata["source_relation_to_id_before_taxonomy_collapse"] = source_relation_to_id
    metadata["relation_to_id"] = directed_relation_to_id
    metadata["num_relations"] = len(directed_relation_to_id)
    metadata["relation_taxonomy_collapse"] = manifest

    edges.to_csv(output_dir / "edges.csv", index=False)
    edges["relation"].value_counts().rename_axis("relation").rename("count").to_csv(
        output_dir / "relation_stats.csv"
    )
    torch.save({"data": data, "metadata": metadata}, output_dir / "graph_data.pt")

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
        "relation_types": len(directed_relation_to_id),
        "base_relation_types": len(group_names),
        "edges_after_relation_taxonomy_collapse": int(data.edge_type.numel()),
        "relation_taxonomy_collapse": manifest,
    })
    _write_json(output_dir / "split_summary.json", summary)
    _write_json(output_dir / "relation_collapse_manifest.json", manifest)
    _write_json(output_dir / "collapsed_relation_taxonomy.json", {
        "name": f"{taxonomy.name}_collapsed_artifact",
        "version": taxonomy.version,
        "description": "Artifact-specific singleton taxonomy after relation collapse.",
        "groups": {group: [group] for group in group_names},
    })

    # training/train.py always records the inherited/acquired taxonomy.  This
    # artifact-specific file keeps that contract valid while the model itself
    # sees the finer relation labels.
    acquired_groups = [group for group in group_names if group != "inherited"]
    _write_json(output_dir / "collapsed_tie_taxonomy.json", {
        "name": f"{taxonomy.name}_collapsed_ties",
        "version": taxonomy.version,
        "description": (
            "Artifact-specific inherited/acquired provenance after multi-group collapse."
        ),
        "groups": {
            "inherited": ["inherited"],
            "acquired": acquired_groups,
        },
    })
    return manifest


def main() -> None:
    args = parse_args()
    manifest = collapse_relation_artifact(
        Path(args.data), Path(args.relation_taxonomy), Path(args.output_dir)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
