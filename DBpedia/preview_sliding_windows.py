#!/usr/bin/env python3
"""Count window graphs from source tables, without allocating graph artifacts or training."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.life_periods import load_life_period_config, life_period_membership


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def preview(config_path):
    config_path = path(config_path)
    pipeline = json.loads(config_path.read_text())
    windows = load_life_period_config(path(pipeline["period_config"]))
    if windows.calendar_layout != "sliding_windows":
        raise ValueError("Preview expects calendar_layout=sliding_windows")
    source = path(pipeline["source_data"]).parent
    nodes = pd.read_csv(source / "nodes.csv")
    edges = pd.read_csv(source / "edges.csv")
    memberships, audit = life_period_membership(nodes, windows)
    # Stats are original directed triples, excluding generated reverse messages.
    edges = edges.loc[~edges.relation.str.endswith("__rev")].copy()
    sources = edges.source_id.to_numpy(dtype=np.int64)
    targets = edges.target_id.to_numpy(dtype=np.int64)
    if (len(sources) and (min(sources.min(), targets.min()) < 0 or max(sources.max(), targets.max()) >= len(nodes))):
        raise ValueError("Edge endpoint index outside node table")
    if not (np.array_equal(nodes.node_id.to_numpy()[sources], edges.source.to_numpy()) and
            np.array_equal(nodes.node_id.to_numpy()[targets], edges.target.to_numpy())):
        raise ValueError("Source URI/index mapping differs from nodes.csv")
    taxonomy = json.loads(path(pipeline["multi_group_taxonomy"]).read_text())["groups"]
    group_names = list(taxonomy)
    relation_group = {r: i for i, rs in taxonomy.items() for r in rs}
    if not set(edges.relation) <= set(relation_group):
        raise ValueError("Source predicates missing from multi-group taxonomy")
    groups = np.array([group_names.index(relation_group[r]) for r in edges.relation], dtype=np.int64)
    collapsed = np.unique(np.column_stack((sources, targets, groups)), axis=0)
    cs, ct, cg = collapsed.T
    previous_nodes = previous_edges = None
    rows, group_rows = [], []
    for period in windows.periods:
        mask = memberships[period.identifier]
        edge_mask = mask[sources] & mask[targets]
        collapsed_mask = mask[cs] & mask[ct]
        incident = np.unique(np.concatenate((sources[edge_mask], targets[edge_mask])))
        row = {
            "period": period.identifier, "center_year": (period.start + period.end) // 2,
            "window_start": period.start, "window_end": period.end,
            "nodes": int(mask.sum()), "original_triples": int(edge_mask.sum()),
            "multi_group_triples_after_collapse": int(collapsed_mask.sum()),
            "isolated_nodes": int(mask.sum() - len(incident)),
            "complete_life_interval_nodes": int((mask & (audit.life_interval_status == "valid_life_interval").to_numpy()).sum()),
            "birth_endpoint_only_nodes": int((mask & (audit.life_interval_status == "birth_endpoint_only").to_numpy()).sum()),
            "death_endpoint_only_nodes": int((mask & (audit.life_interval_status == "death_endpoint_only").to_numpy()).sum()),
            "previous_center_year": None if previous_nodes is None else rows[-1]["center_year"],
            "nodes_entered": None if previous_nodes is None else int((mask & ~previous_nodes).sum()),
            "nodes_exited": None if previous_nodes is None else int((previous_nodes & ~mask).sum()),
            "node_jaccard_previous": None if previous_nodes is None else float((mask & previous_nodes).sum() / (mask | previous_nodes).sum()),
            "original_triple_jaccard_previous": None if previous_edges is None else float((edge_mask & previous_edges).sum() / max(1, (edge_mask | previous_edges).sum())),
        }
        rows.append(row)
        for index, group in enumerate(group_names):
            raw = int((edge_mask & (groups == index)).sum())
            merged = int((collapsed_mask & (cg == index)).sum())
            group_rows.append({"period": period.identifier, "center_year": row["center_year"], "group": group,
                               "original_triples": raw, "multi_group_triples_after_collapse": merged,
                               "raw_graph_group_share": raw / row["original_triples"] if row["original_triples"] else None,
                               "collapsed_graph_group_share": merged / row["multi_group_triples_after_collapse"] if row["multi_group_triples_after_collapse"] else None})
        previous_nodes, previous_edges = mask, edge_mask
    return rows, group_rows, {
        "pipeline_config": str(config_path.resolve()), "source_tables": [str((source / n).resolve()) for n in ("nodes.csv", "edges.csv")],
        "window_count": len(rows), "source_nodes": len(nodes), "source_original_triples": len(edges),
        "membership": windows.manifest(),
        "counts_unit": "Original directed triples excluding __rev; grouped triples deduplicated by source/group/target. These are graph composition counts, not learned GraphMask retention.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json")
    parser.add_argument("--output-dir", default="artifacts/dbpedia_sliding_window_preview_2026_10_07")
    args = parser.parse_args()
    rows, groups, metadata = preview(args.config)
    output = path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "window_sizes.tsv", sep="\t", index=False)
    pd.DataFrame(groups).to_csv(output / "window_relation_groups.tsv", sep="\t", index=False)
    (output / "preview_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Counted {len(rows)} windows: {output}")
    for row in rows:
        if row["center_year"] in (1900, 1901, 1902, 1920, 1950, 1980, 2000):
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
