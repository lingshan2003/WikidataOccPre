"""Build genuinely period-contained occupation-graph artifacts.

The earlier cohort intervention keeps the complete graph and only changes a
selected set of cohort-incident edges.  That is useful as a *global-graph
local-intervention* sensitivity analysis, but it is not a historical-period
experiment.  This module implements the latter explicitly:

1. select people whose known life interval overlaps one configured period, or
   whose sole known birth/death endpoint lies in that period;
2. retain only message edges for which both endpoints are in that period;
3. rebuild feature encodings and a fresh within-period 70/10/20 split; and
4. save one independently auditable artifact for each period.

Relation IDs and the label vocabulary are retained from the source artifact.
This keeps the definition of a Level-1 class and the inherited/acquired
taxonomy stable across periods, while feature vocabularies and split masks are
fitted only inside the selected period.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from data.prepare import build_pyg_data, stratified_node_split
from training.life_periods import (
    LifePeriod,
    LifePeriodConfig,
    life_period_membership,
    load_life_period_config,
    sha256_file,
)


ARTIFACT_SCHEMA_VERSION = 3
INDUCED_EDGE_POLICY = "retain_only_edges_with_both_endpoints_in_selected_life_period"
SPLIT_POLICY = "fresh_within_period_stratified_70_10_20"
REQUIRED_NODE_COLUMNS = {
    "node_id",
    "birth_year",
    "death_year",
    "occupation_level1",
    "occupation_level2",
    "occupation_level3",
    "country",
}
REQUIRED_EDGE_COLUMNS = {"source", "relation", "target", "source_id", "target_id", "relation_id"}


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _require_source_tables(source_data: Path, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and verify tables whose row order must match the source graph."""
    nodes_path, edges_path = source_data.parent / "nodes.csv", source_data.parent / "edges.csv"
    if not nodes_path.is_file() or not edges_path.is_file():
        raise FileNotFoundError(
            "A period-induced artifact needs nodes.csv and edges.csv alongside the source graph_data.pt: "
            f"missing={[str(path) for path in (nodes_path, edges_path) if not path.is_file()]}"
        )
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    missing_nodes, missing_edges = REQUIRED_NODE_COLUMNS - set(nodes), REQUIRED_EDGE_COLUMNS - set(edges)
    if missing_nodes:
        raise ValueError(f"Source node table lacks required columns: {sorted(missing_nodes)}")
    if missing_edges:
        raise ValueError(f"Source edge table lacks required columns: {sorted(missing_edges)}")
    if len(nodes) != int(data.num_nodes):
        raise ValueError(
            f"Source node-table row count {len(nodes)} differs from graph node count {data.num_nodes}"
        )
    if len(edges) != int(data.edge_index.size(1)):
        raise ValueError(
            f"Source edge-table row count {len(edges)} differs from graph edge count {data.edge_index.size(1)}"
        )

    edge_index = data.edge_index.detach().cpu().numpy()
    edge_type = data.edge_type.detach().cpu().numpy()
    table_sources = edges["source_id"].to_numpy(dtype=np.int64)
    table_targets = edges["target_id"].to_numpy(dtype=np.int64)
    table_relations = edges["relation_id"].to_numpy(dtype=np.int64)
    if not (
        np.array_equal(table_sources, edge_index[0])
        and np.array_equal(table_targets, edge_index[1])
        and np.array_equal(table_relations, edge_type)
    ):
        raise ValueError(
            "Source edges.csv is not in exactly the same order as graph_data.pt edge_index/edge_type; "
            "refusing to construct an incorrectly reindexed period graph"
        )
    if not (
        np.array_equal(nodes.iloc[table_sources]["node_id"].to_numpy(), edges["source"].astype(str).to_numpy())
        and np.array_equal(nodes.iloc[table_targets]["node_id"].to_numpy(), edges["target"].astype(str).to_numpy())
    ):
        raise ValueError("Source node IDs do not agree with source_id/target_id in edges.csv")
    return nodes, edges


def _validate_source_bundle(source_data: Path) -> Tuple[object, Mapping[str, object], pd.DataFrame, pd.DataFrame]:
    if not source_data.is_file():
        raise FileNotFoundError(f"Source graph artifact does not exist: {source_data}")
    bundle = torch.load(source_data, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping) or "data" not in bundle or "metadata" not in bundle:
        raise ValueError(f"Source artifact must contain data and metadata: {source_data}")
    data, metadata = bundle["data"], bundle["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Source artifact metadata must be an object: {source_data}")
    required_data = ("edge_index", "edge_type", "y", "num_nodes")
    missing_data = [name for name in required_data if not hasattr(data, name)]
    if missing_data:
        raise ValueError(f"Source graph lacks required fields {missing_data}: {source_data}")
    required_metadata = ("target_column", "label_to_id", "num_classes", "relation_to_id", "num_relations")
    missing_metadata = [name for name in required_metadata if name not in metadata]
    if missing_metadata:
        raise ValueError(f"Source metadata lacks required fields {missing_metadata}: {source_data}")
    relation_to_id = metadata["relation_to_id"]
    if not isinstance(relation_to_id, Mapping):
        raise ValueError("Source metadata relation_to_id must be an object")
    relation_ids = sorted(int(value) for value in relation_to_id.values())
    if relation_ids != list(range(int(metadata["num_relations"]))):
        raise ValueError("Source relation_to_id must use consecutive IDs from 0 to num_relations - 1")
    label_to_id = metadata["label_to_id"]
    if not isinstance(label_to_id, Mapping):
        raise ValueError("Source metadata label_to_id must be an object")
    label_ids = sorted(int(value) for value in label_to_id.values())
    if label_ids != list(range(int(metadata["num_classes"]))):
        raise ValueError("Source label_to_id must use consecutive IDs from 0 to num_classes - 1")
    if len(data.y) != int(data.num_nodes):
        raise ValueError("Source y length differs from graph node count")
    nodes, edges = _require_source_tables(source_data, data)
    return data, metadata, nodes, edges


def _source_conflicts(source_data: Path, node_ids: Iterable[str]) -> pd.DataFrame:
    """Filter optional source conflict audit rows without requiring that old artifacts have it."""
    conflicts_path = source_data.parent / "attribute_conflicts.csv"
    if not conflicts_path.is_file():
        return pd.DataFrame(columns=["node_id", "attribute", "values"])
    try:
        conflicts = pd.read_csv(conflicts_path)
    except pd.errors.EmptyDataError:
        # The canonical loader writes an empty file when it detected no
        # conflicts.  Treat that as the intended empty audit table rather
        # than rejecting an otherwise valid source artifact.
        return pd.DataFrame(columns=["node_id", "attribute", "values"])
    if "node_id" not in conflicts:
        raise ValueError(f"attribute_conflicts.csv lacks node_id: {conflicts_path}")
    return conflicts.loc[conflicts["node_id"].isin(set(node_ids))].reset_index(drop=True)


def _period_metadata(
    source_data: Path,
    source_hash: str,
    period_config: LifePeriodConfig,
    period: LifePeriod,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    local_min_class_count: int,
    source_nodes: int,
    source_edges: int,
    period_nodes: int,
    period_edges: int,
    incident_cross_period_edges: int,
    local_excluded_class_counts: Mapping[str, int],
    source_life_date_status_counts: Mapping[str, int],
    selected_node_membership_count_counts: Mapping[str, int],
) -> Dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_data": str(source_data.resolve()),
        "source_data_sha256": source_hash,
        "life_period_config": period_config.manifest(),
        "selected_life_period": period.manifest(),
        "node_policy": "select_nodes_with_life_interval_overlap_or_known_date_endpoint_in_selected_life_period",
        "edge_policy": INDUCED_EDGE_POLICY,
        "split_policy": SPLIT_POLICY,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": split_seed,
            "local_min_class_count": local_min_class_count,
            "rare_class_treatment": "set_y_to_minus_one_before_within_period_stratified_split",
        },
        "source_graph": {"nodes": source_nodes, "directed_edges": source_edges},
        "period_graph": {
            "nodes": period_nodes,
            "directed_edges": period_edges,
            "excluded_directed_edges": source_edges - period_edges,
            "incident_cross_period_directed_edges": incident_cross_period_edges,
        },
        "local_rare_classes_excluded_from_supervision": dict(local_excluded_class_counts),
        "source_life_date_status_counts": dict(source_life_date_status_counts),
        "selected_node_life_period_membership_count_counts": dict(selected_node_membership_count_counts),
        "relation_id_policy": "preserve_source_relation_to_id_mapping_even_when_some_types_are_absent",
        "label_vocabulary_policy": "preserve_source_label_to_id_mapping",
    }


def _existing_artifact_matches(
    output_dir: Path,
    source_hash: str,
    period_config: LifePeriodConfig,
    period: LifePeriod,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    local_min_class_count: int,
) -> bool:
    artifact_path = output_dir / "graph_data.pt"
    if not artifact_path.is_file():
        return False
    bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
    metadata = bundle.get("metadata") if isinstance(bundle, Mapping) else None
    details = metadata.get("period_induced_artifact") if isinstance(metadata, Mapping) else None
    if not isinstance(details, Mapping):
        raise ValueError(
            f"Existing output is not a period-induced artifact; choose a new output root: {output_dir}"
        )
    config = details.get("life_period_config")
    selected = details.get("selected_life_period")
    split = details.get("split")
    try:
        same_split = (
            isinstance(split, Mapping)
            and split.get("seed") == split_seed
            and np.isclose(float(split.get("train_ratio")), train_ratio)
            and np.isclose(float(split.get("val_ratio")), val_ratio)
            and np.isclose(float(split.get("test_ratio")), test_ratio)
            and split.get("local_min_class_count") == local_min_class_count
        )
    except (TypeError, ValueError):
        same_split = False
    expected = (
        details.get("source_data_sha256") == source_hash
        and isinstance(config, Mapping)
        and config.get("sha256") == period_config.sha256
        and isinstance(selected, Mapping)
        and selected.get("id") == period.identifier
        and same_split
    )
    if not expected:
        raise ValueError(
            f"Existing period artifact has incompatible source/period/split provenance: {output_dir}; "
            "choose a new output root rather than overwriting it"
        )
    return True


def build_period_induced_artifact(
    source_data: str | Path,
    output_root: str | Path,
    period_config: LifePeriodConfig,
    period_id: str,
    *,
    split_seed: int = 20260814,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    local_min_class_count: int = 3,
) -> Dict[str, object]:
    """Create one period-induced artifact, or verify and reuse an exact existing one."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    if local_min_class_count < 3:
        raise ValueError("local_min_class_count must be at least 3 for a train/validation/test split")
    source_data, output_root = Path(source_data).resolve(), Path(output_root).resolve()
    period = period_config.period(period_id)
    source_hash = sha256_file(source_data)
    output_dir = output_root / period.identifier
    if output_dir.exists() and _existing_artifact_matches(
        output_dir,
        source_hash,
        period_config,
        period,
        split_seed,
        train_ratio,
        val_ratio,
        test_ratio,
        local_min_class_count,
    ):
        summary_path = output_dir / "split_summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Existing period artifact lacks split_summary.json: {output_dir}")
        result = dict(_read_json(summary_path))
        result["artifact_dir"] = str(output_dir)
        result["reused_existing"] = True
        return result
    if output_dir.exists():
        raise ValueError(
            f"Output directory exists but is incomplete: {output_dir}; remove it manually or use a new output root"
        )

    source_graph, source_metadata, source_nodes, source_edges = _validate_source_bundle(source_data)
    memberships, life_audit = life_period_membership(source_nodes, period_config)
    selected_mask = memberships[period.identifier]
    if not selected_mask.any():
        raise ValueError(f"Life period {period.identifier!r} has no nodes meeting its life-date membership rule")
    source_indices = np.flatnonzero(selected_mask)
    source_life_date_status_counts = {
        str(status): int(count)
        for status, count in life_audit["life_interval_status"].value_counts().items()
    }
    selected_node_membership_count_counts = {
        str(count): int(total)
        for count, total in life_audit.loc[selected_mask, "life_period_membership_count"].value_counts().sort_index().items()
    }
    source_to_period = np.full(int(source_graph.num_nodes), -1, dtype=np.int64)
    source_to_period[source_indices] = np.arange(len(source_indices), dtype=np.int64)

    source_edge_index = source_graph.edge_index.detach().cpu().numpy()
    retain_edge = selected_mask[source_edge_index[0]] & selected_mask[source_edge_index[1]]
    incident_cross_period = (
        (selected_mask[source_edge_index[0]] | selected_mask[source_edge_index[1]]) & ~retain_edge
    )
    period_edges = source_edges.loc[retain_edge].copy().reset_index(drop=True)
    period_edges["source_id"] = source_to_period[period_edges["source_id"].to_numpy(dtype=np.int64)]
    period_edges["target_id"] = source_to_period[period_edges["target_id"].to_numpy(dtype=np.int64)]
    if (period_edges[["source_id", "target_id"]].to_numpy() < 0).any():
        raise RuntimeError("An excluded source node survived induced-subgraph remapping")

    period_nodes = source_nodes.iloc[source_indices].copy().reset_index(drop=True)
    period_nodes.insert(0, "source_node_index", source_indices)
    period_nodes = pd.concat([period_nodes, life_audit.iloc[source_indices].reset_index(drop=True)], axis=1)
    period_nodes["life_period"] = period.identifier
    period_nodes["life_period_label"] = period.label

    source_y = source_graph.y.detach().cpu().numpy().astype(np.int64, copy=True)
    period_y = source_y[source_indices].copy()
    label_to_id = {str(label): int(label_id) for label, label_id in source_metadata["label_to_id"].items()}
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}
    known_counts = {
        class_id: int(np.sum(period_y == class_id))
        for class_id in np.unique(period_y[period_y >= 0])
    }
    excluded_class_ids = {
        class_id for class_id, count in known_counts.items() if count < local_min_class_count
    }
    local_excluded_class_counts = {
        id_to_label[class_id]: known_counts[class_id] for class_id in sorted(excluded_class_ids)
    }
    if excluded_class_ids:
        period_y[np.isin(period_y, list(excluded_class_ids))] = -1
    eligible = period_y >= 0
    if not eligible.any():
        raise ValueError(
            f"Life period {period.identifier!r} has no supervised classes with at least "
            f"{local_min_class_count} members"
        )
    train_mask, val_mask, test_mask = stratified_node_split(
        period_y, eligible, train_ratio, val_ratio, test_ratio, split_seed
    )
    period_graph, country_to_id, occupation_schema, occupation_unknown_ids, occupation_vocabularies = build_pyg_data(
        period_nodes,
        period_edges,
        period_y,
        train_mask,
        val_mask,
        test_mask,
    )
    if int(period_graph.num_nodes) != len(source_indices):
        raise RuntimeError("Period graph node count changed while rebuilding features")

    details = _period_metadata(
        source_data,
        source_hash,
        period_config,
        period,
        split_seed,
        train_ratio,
        val_ratio,
        test_ratio,
        local_min_class_count,
        int(source_graph.num_nodes),
        int(source_graph.edge_index.size(1)),
        int(period_graph.num_nodes),
        int(period_graph.edge_index.size(1)),
        int(incident_cross_period.sum()),
        local_excluded_class_counts,
        source_life_date_status_counts,
        selected_node_membership_count_counts,
    )
    metadata = {
        "target_column": source_metadata["target_column"],
        "num_relations": int(source_metadata["num_relations"]),
        "num_classes": int(source_metadata["num_classes"]),
        "feature_schema": {
            **occupation_schema,
            "country": {"kind": "categorical", "cardinality": len(country_to_id)},
            "temporal": {"kind": "numeric", "input_dim": int(period_graph.temporal.size(1))},
        },
        "country_to_id": country_to_id,
        "label_to_id": label_to_id,
        "occupation_unknown_ids": occupation_unknown_ids,
        "occupation_vocabularies": occupation_vocabularies,
        "relation_to_id": {str(name): int(value) for name, value in source_metadata["relation_to_id"].items()},
        "seed": split_seed,
        "period_induced_artifact": details,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    # Build atomically inside the requested output root: a killed preparation
    # never leaves a directory that a later experiment runner could mistake for
    # a complete, valid period artifact.
    temporary_dir = output_root / f".{period.identifier}.building-{os.getpid()}"
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary preparation directory already exists: {temporary_dir}")
    temporary_dir.mkdir()
    try:
        torch.save({"data": period_graph, "metadata": metadata}, temporary_dir / "graph_data.pt")
        period_nodes.to_csv(temporary_dir / "nodes.csv", index=False)
        period_edges.to_csv(temporary_dir / "edges.csv", index=False)
        _source_conflicts(source_data, period_nodes["node_id"]).to_csv(
            temporary_dir / "attribute_conflicts.csv", index=False
        )
        retained_counts = pd.Series(period_y[period_y >= 0]).value_counts().sort_index()
        pd.DataFrame({
            "occupation": [id_to_label[int(class_id)] for class_id in retained_counts.index],
            "count": retained_counts.to_numpy(dtype=np.int64),
        }).to_csv(temporary_dir / "class_stats.csv", index=False)
        period_edges["relation"].value_counts().rename_axis("relation").rename("count").to_csv(
            temporary_dir / "relation_stats.csv"
        )
        summary = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_dir": str(output_dir),
            "target_column": metadata["target_column"],
            "period_id": period.identifier,
            "period_label": period.label,
            "period_start_year": period.start,
            "period_end_year": period.end,
            "nodes": int(period_graph.num_nodes),
            "directed_edges": int(period_graph.edge_index.size(1)),
            "labeled_nodes_before_local_rare_filter": int(np.sum(source_y[source_indices] >= 0)),
            "labeled_nodes_retained": int(eligible.sum()),
            "locally_rare_nodes_excluded_from_supervision": int(np.sum(~eligible) - np.sum(source_y[source_indices] < 0)),
            "local_rare_classes_excluded_from_supervision": local_excluded_class_counts,
            "train_nodes": int(train_mask.sum()),
            "val_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "active_target_classes": int(len(np.unique(period_y[period_y >= 0]))),
            "life_period_config": period_config.manifest(),
            "period_induced_artifact": details,
        }
        (temporary_dir / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_dir, output_dir)
    except BaseException:
        # Leave the temporary directory for inspection instead of deleting
        # incomplete data silently.  It is deliberately never accepted as an
        # artifact by the runner because it does not use the cohort ID path.
        raise
    return {**summary, "reused_existing": False}


def prepare_period_induced_artifacts(
    source_data: str | Path,
    output_root: str | Path,
    period_config_path: str | Path | None,
    period_ids: Sequence[str] | None = None,
    **split_kwargs: object,
) -> List[Dict[str, object]]:
    """Prepare every requested period, validating all IDs before writing any artifact."""
    config = load_life_period_config(period_config_path)
    selected_ids = tuple(period_ids) if period_ids else config.identifiers
    unknown = set(selected_ids) - set(config.identifiers)
    if unknown:
        raise ValueError(f"Unknown requested life periods: {sorted(unknown)}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Life periods may be requested only once")
    return [
        build_period_induced_artifact(
            source_data,
            output_root,
            config,
            period_id,
            **split_kwargs,
        )
        for period_id in selected_ids
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True, help="Canonical complete-graph graph_data.pt")
    parser.add_argument("--output-root", required=True, help="Root that will contain one directory per period")
    parser.add_argument(
        "--life-periods",
        default="config/historical_life_periods_v2.json",
        help="Versioned life-period overlap configuration JSON",
    )
    parser.add_argument(
        "--periods",
        default="all",
        help="Comma-separated period IDs, or all (default)",
    )
    parser.add_argument("--split-seed", type=int, default=20260814)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--local-min-class-count",
        type=int,
        default=3,
        help="Within-period labeled support required to retain a class for the three-way split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = None if args.periods.strip().casefold() == "all" else tuple(
        item.strip() for item in args.periods.split(",") if item.strip()
    )
    reports = prepare_period_induced_artifacts(
        args.source_data,
        args.output_root,
        args.life_periods,
        requested,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        local_min_class_count=args.local_min_class_count,
    )
    for report in reports:
        state = "reused" if report["reused_existing"] else "prepared"
        print(
            f"[{state}] {report['period_id']}: nodes={report['nodes']}, "
            f"directed_edges={report['directed_edges']}, "
            f"split={report['train_nodes']}/{report['val_nodes']}/{report['test_nodes']}"
        )


if __name__ == "__main__":
    main()
