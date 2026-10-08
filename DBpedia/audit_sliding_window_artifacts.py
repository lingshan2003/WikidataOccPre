#!/usr/bin/env python3
"""Independently reconstruct selected windows and compare saved graph artifacts."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline, write_json


def interval_mask(birth, death, start, end):
    complete = np.isfinite(birth) & np.isfinite(death)
    valid = complete & (birth <= death)
    birth_only = np.isfinite(birth) & ~np.isfinite(death)
    death_only = ~np.isfinite(birth) & np.isfinite(death)
    return ((valid & (birth <= end) & (death >= start)) |
            (birth_only & (birth >= start) & (birth <= end)) |
            (death_only & (death >= start) & (death <= end)))


def audit(config, periods=None):
    pipeline = Pipeline(SimpleNamespace(config=config, periods=periods, representations="multi_group",
                        device=None, include_full=False, stage="collapse"))
    if pipeline.period_config.get("calendar_layout") != "sliding_windows" or "full" in pipeline.contexts:
        raise ValueError("Audit requires sliding windows without a full-graph context")
    source_path = pipeline.path(pipeline.config["source_data"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    nodes = pd.read_csv(source_path.parent / "nodes.csv")
    edges = pd.read_csv(source_path.parent / "edges.csv")
    birth = pd.to_numeric(nodes.birth_year, errors="coerce").to_numpy(dtype=float)
    death = pd.to_numeric(nodes.death_year, errors="coerce").to_numpy(dtype=float)
    membership_count = np.zeros(len(nodes), dtype=np.int64)
    for period in pipeline.periods.values():
        membership_count += interval_mask(birth, death, period["start"], period["end"])
    source_ids = edges.source_id.to_numpy(dtype=np.int64)
    target_ids = edges.target_id.to_numpy(dtype=np.int64)
    group_for = {relation: group for group, relations in pipeline.multi["groups"].items() for relation in relations}
    rows, groups, checks = [], [], []
    for context in pipeline.contexts:
        passed = []
        def check(name, condition):
            checks.append({"context": context, "check": name, "passed": bool(condition)})
            passed.append(bool(condition))
        row = {"context": context, "passed": False}
        try:
            pipeline.validate_period(context)
            collapsed_bundle = pipeline.validate_collapsed(context, "multi_group")
            period = pipeline.periods[context]
            mask = interval_mask(birth, death, period["start"], period["end"])
            selected = np.flatnonzero(mask)
            original_path = pipeline.source(context)
            original = torch.load(original_path, map_location="cpu", weights_only=False)
            original_nodes = pd.read_csv(original_path.parent / "nodes.csv")
            original_edges = pd.read_csv(original_path.parent / "edges.csv")
            artifact, _, _ = pipeline.directories(context, "multi_group")
            grouped_nodes = pd.read_csv(artifact / "nodes.csv")
            grouped_edges = pd.read_csv(artifact / "edges.csv")
            check("exact_node_membership_and_source_indices", np.array_equal(selected, original_nodes.source_node_index.to_numpy()))
            check("exact_node_URI_order", np.array_equal(nodes.node_id.to_numpy()[selected], original_nodes.node_id.to_numpy()))
            check("membership_counts_across_all_annual_windows", np.array_equal(membership_count[selected], original_nodes.life_period_membership_count.to_numpy()))
            retained = mask[source_ids] & mask[target_ids]
            expected_edges = edges.loc[retained].copy().reset_index(drop=True)
            local = np.full(len(nodes), -1, dtype=np.int64)
            local[selected] = np.arange(len(selected))
            expected_edges.source_id = local[expected_edges.source_id.to_numpy()]
            expected_edges.target_id = local[expected_edges.target_id.to_numpy()]
            edge_columns = ["source", "relation", "target", "source_id", "target_id", "relation_id"]
            check("all_and_only_induced_edges_with_correct_local_indices", expected_edges[edge_columns].equals(original_edges[edge_columns]))
            check("source_graph_matches_source_tables", int(source["data"].num_nodes) == len(nodes) and
                  np.array_equal(source["data"].edge_index.numpy(), edges[["source_id", "target_id"]].to_numpy().T) and
                  np.array_equal(source["data"].edge_type.numpy(), edges.relation_id.to_numpy()))
            check("window_tensor_matches_saved_edges", int(original["data"].num_nodes) == len(selected) and
                  np.array_equal(original["data"].edge_index.numpy(), original_edges[["source_id", "target_id"]].to_numpy().T) and
                  np.array_equal(original["data"].edge_type.numpy(), original_edges.relation_id.to_numpy()))
            mapping = collapsed_bundle["metadata"]["relation_to_id"]
            expected_grouped = expected_edges.copy()
            expected_grouped.relation = [group_for[r.removesuffix("__rev")] + ("__rev" if r.endswith("__rev") else "") for r in expected_edges.relation]
            expected_grouped.relation_id = expected_grouped.relation.map(mapping)
            expected_grouped = expected_grouped.drop_duplicates(["source_id", "relation", "target_id"], keep="first").reset_index(drop=True)
            check("exact_group_mapping_reverse_direction_and_triple_dedup", expected_grouped[edge_columns].equals(grouped_edges[edge_columns]))
            check("seven_groups_fourteen_directed_relation_ids", set(mapping) == set(pipeline.multi["groups"]) | {g + "__rev" for g in pipeline.multi["groups"]} and
                  set(mapping.values()) == set(range(14)))
            check("grouped_node_table_unchanged", original_nodes.equals(grouped_nodes))
            graph, cg = original["data"], collapsed_bundle["data"]
            check("all_node_tensors_labels_and_features_unchanged_by_grouping", all(
                torch.equal(graph[key], cg[key]) if isinstance(graph[key], torch.Tensor) else graph[key] == cg[key]
                for key in graph.keys() if key not in ("edge_index", "edge_type")))
            source_y = source["data"].y.numpy()[selected].copy()
            counts = pd.Series(source_y[source_y >= 0]).value_counts()
            minimum = pipeline.config["prepare"]["local_min_class_count"]
            source_y[np.isin(source_y, counts[counts < minimum].index)] = -1
            check("labels_match_source_with_only_local_rare_class_filter", np.array_equal(source_y, graph.y.numpy()))
            train, val, test = (graph[key].numpy() for key in ("train_mask", "val_mask", "test_mask"))
            check("splits_disjoint_cover_supervised_nodes_and_nonempty", not np.any((train & val) | (train & test) | (val & test)) and
                  np.array_equal(train | val | test, source_y >= 0) and all(m.any() for m in (train, val, test)))
            expected_split = np.zeros((3, len(selected)), dtype=bool)
            split_config = pipeline.config["prepare"]
            rng = np.random.default_rng(split_config["split_seed"])
            for class_id in np.unique(source_y[source_y >= 0]):
                members = np.flatnonzero(source_y == class_id)
                rng.shuffle(members)
                val_count = min(max(1, round(len(members) * split_config["val_ratio"])), len(members)-2)
                test_count = min(max(1, round(len(members) * split_config["test_ratio"])), len(members)-val_count-1)
                train_count = len(members)-val_count-test_count
                expected_split[0, members[:train_count]] = True
                expected_split[1, members[train_count:train_count+val_count]] = True
                expected_split[2, members[train_count+val_count:]] = True
            check("exact_seeded_per_class_70_10_20_split_with_small_class_rounding", np.array_equal(expected_split, np.stack((train, val, test))))
            unknown = original["metadata"]["occupation_unknown_ids"]
            check("heldout_occupation_features_are_masked", all(bool(torch.all(graph[key][torch.from_numpy(~train)] == value)) for key, value in unknown.items()))
            incident = np.unique(np.concatenate([original_edges.source_id.to_numpy(), original_edges.target_id.to_numpy()]))
            base_edges = original_edges.loc[~original_edges.relation.str.endswith("__rev")]
            base_grouped = grouped_edges.loc[~grouped_edges.relation.str.endswith("__rev")]
            row.update({"center_year": (period["start"] + period["end"]) // 2,
                        "start": period["start"], "end": period["end"], "nodes": len(selected),
                        "original_triples": len(base_edges), "grouped_triples": len(base_grouped),
                        "directed_messages": len(grouped_edges), "isolated_nodes": len(selected)-len(incident),
                        "train_nodes": int(train.sum()), "val_nodes": int(val.sum()), "test_nodes": int(test.sum()),
                        "active_classes": len(np.unique(source_y[source_y >= 0])),
                        "complete_dates": int((np.isfinite(birth[selected]) & np.isfinite(death[selected])).sum()),
                        "birth_only": int((np.isfinite(birth[selected]) & ~np.isfinite(death[selected])).sum()),
                        "death_only": int((~np.isfinite(birth[selected]) & np.isfinite(death[selected])).sum()),
                        "passed": all(passed), "checks": len(passed)})
            for group in pipeline.multi["groups"]:
                groups.append({"context": context, "group": group,
                               "raw_original_triples": int(base_edges.relation.isin(pipeline.multi["groups"][group]).sum()),
                               "grouped_original_triples": int((base_grouped.relation == group).sum())})
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
    return pipeline, {"passed": bool(rows) and all(r["passed"] for r in rows), "windows": rows, "checks": checks,
                      "group_counts": groups, "config": pipeline.config, "hash_verification": "not performed",
                      "scope": "Graph construction, grouping, supervision and held-out feature masking; no claim about trained predictive performance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dbpedia_grouped_sliding_pilot_4years_v1.json")
    parser.add_argument("--periods")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    pipeline, result = audit(args.config, args.periods)
    output = pipeline.path(args.output_dir) if args.output_dir else pipeline.path(pipeline.config["graphmask_root"]) / "pilot_audit"
    write_json(output / "graph_audit.json", result)
    for name, rows in (("window_audit", result["windows"]), ("checks", result["checks"]), ("group_counts", result["group_counts"])):
        pd.DataFrame(rows).to_csv(output / f"{name}.tsv", sep="\t", index=False)
    print(json.dumps({"passed": result["passed"], "windows": result["windows"], "report": str(output)}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
