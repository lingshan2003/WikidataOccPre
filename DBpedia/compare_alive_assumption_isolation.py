#!/usr/bin/env python3
"""Compare DBpedia sliding-window isolation before and after an alive-through assumption.

The comparison uses the shared life-period membership implementation and counts
only original directed predicates (generated ``__rev`` message edges are excluded).
It writes per-window counts and rates so newly enrolled isolated nodes cannot be
mistaken for a reduction in isolation among the original window members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.life_periods import life_period_membership, load_life_period_config


DEFAULT_SOURCE_DIR = ROOT / "artifacts" / "dbpedia_2022_priority_v1"
DEFAULT_OLD_CONFIG = ROOT / "config" / "dbpedia_life_windows_1900_2000_pm20_step1_v1.json"
DEFAULT_NEW_CONFIG = ROOT / "config" / "dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "dbpedia_alive2026_isolation_2026_10_10"


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _window_stats(mask: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    selected_edges = mask[sources] & mask[targets]
    incident = np.bincount(
        np.concatenate((sources[selected_edges], targets[selected_edges])),
        minlength=len(mask),
    )
    isolated = mask & (incident == 0)
    node_count = int(mask.sum())
    isolated_count = int(isolated.sum())
    return (
        {
            "nodes": node_count,
            "original_directed_triples": int(selected_edges.sum()),
            "isolated_nodes": isolated_count,
            "isolation_rate": _ratio(isolated_count, node_count),
        },
        isolated,
        incident,
    )


def _load_periods(config_path: Path):
    config = load_life_period_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("calendar_layout") != "sliding_windows":
        raise ValueError(f"Expected sliding_windows in {config_path}")
    return config, payload


def compare(source_dir: Path, old_config_path: Path, new_config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old_config, old_payload = _load_periods(old_config_path)
    new_config, new_payload = _load_periods(new_config_path)
    expected_assumption = {"born_after": 1920, "alive_through": 2026}
    if old_payload.get("birth_only_alive_assumption") is not None:
        raise ValueError("The baseline config must not contain a birth-only alive assumption")
    if new_payload.get("birth_only_alive_assumption") != expected_assumption:
        raise ValueError(
            "New period config must set birth_only_alive_assumption to "
            f"{expected_assumption!r}; got {new_payload.get('birth_only_alive_assumption')!r}"
        )
    if old_config.identifiers != new_config.identifiers:
        raise ValueError("Old and new configs must contain the same period IDs in the same order")
    old_bounds = [(p.start, p.end) for p in old_config.periods]
    new_bounds = [(p.start, p.end) for p in new_config.periods]
    if old_bounds != new_bounds:
        raise ValueError("Old and new configs must have identical window boundaries")

    nodes = pd.read_csv(source_dir / "nodes.csv")
    edges = pd.read_csv(source_dir / "edges.csv")
    if not {"node_id", "birth_year", "death_year"}.issubset(nodes.columns):
        raise ValueError("nodes.csv must contain node_id, birth_year, and death_year")
    if not {"source_id", "target_id", "source", "target", "relation"}.issubset(edges.columns):
        raise ValueError("edges.csv lacks the required endpoint or relation columns")

    # The source edge table includes generated reverse messages for model use;
    # exclude those so each original predicate contributes once.
    edges = edges.loc[~edges["relation"].astype(str).str.endswith("__rev")].reset_index(drop=True)
    sources = edges["source_id"].to_numpy(dtype=np.int64)
    targets = edges["target_id"].to_numpy(dtype=np.int64)
    node_ids = nodes["node_id"].to_numpy()
    if (np.any(sources < 0) or np.any(targets < 0) or
            np.any(sources >= len(nodes)) or np.any(targets >= len(nodes))):
        raise ValueError("Edge endpoint indices fall outside nodes.csv")
    if not (np.array_equal(node_ids[sources], edges["source"].to_numpy()) and
            np.array_equal(node_ids[targets], edges["target"].to_numpy())):
        raise ValueError("Source URI/index mapping differs between node and edge tables")

    old_memberships, _ = life_period_membership(nodes, old_config)
    new_memberships, _ = life_period_membership(nodes, new_config)
    rows: list[dict[str, Any]] = []
    for old_period, new_period in zip(old_config.periods, new_config.periods):
        old_mask = old_memberships[old_period.identifier]
        new_mask = new_memberships[new_period.identifier]
        if not np.all(~old_mask | new_mask):
            raise AssertionError(
                f"Alive-through assumption unexpectedly removed members from {old_period.identifier}"
            )

        old_stats, old_isolated, _ = _window_stats(old_mask, sources, targets)
        new_stats, new_isolated, _ = _window_stats(new_mask, sources, targets)
        entrants = new_mask & ~old_mask
        rescued_old_isolated = old_isolated & old_mask & ~new_isolated
        remaining_old_isolated = old_isolated & old_mask & new_isolated
        newly_isolated_old_members = old_mask & ~old_isolated & new_isolated
        newly_added_isolated = entrants & new_isolated
        if newly_isolated_old_members.any():
            raise AssertionError(
                f"Adding nodes/edges made previously connected members isolated in {old_period.identifier}"
            )
        if int(old_isolated.sum()) != int(rescued_old_isolated.sum() + remaining_old_isolated.sum()):
            raise AssertionError(f"Original isolation transition does not balance for {old_period.identifier}")
        if int(new_isolated.sum()) != int(remaining_old_isolated.sum() + newly_added_isolated.sum()):
            raise AssertionError(f"New isolation accounting does not balance for {old_period.identifier}")

        rows.append({
            "period_id": old_period.identifier,
            "label": old_period.label,
            "start": old_period.start,
            "end": old_period.end,
            "center_year": (old_period.start + old_period.end) // 2,
            "old_nodes": old_stats["nodes"],
            "new_nodes": new_stats["nodes"],
            "newly_added_nodes": int(entrants.sum()),
            "old_original_directed_triples": old_stats["original_directed_triples"],
            "new_original_directed_triples": new_stats["original_directed_triples"],
            "added_original_directed_triples": (
                new_stats["original_directed_triples"] - old_stats["original_directed_triples"]
            ),
            "old_isolated_nodes": old_stats["isolated_nodes"],
            "new_isolated_nodes": new_stats["isolated_nodes"],
            "new_minus_old_isolated_nodes": new_stats["isolated_nodes"] - old_stats["isolated_nodes"],
            "rescued_old_isolated_nodes": int(rescued_old_isolated.sum()),
            "remaining_old_isolated_nodes": int(remaining_old_isolated.sum()),
            "newly_added_isolated_nodes": int(newly_added_isolated.sum()),
            "old_isolation_rate": old_stats["isolation_rate"],
            "new_isolation_rate": new_stats["isolation_rate"],
            "new_isolation_rate_on_old_members": _ratio(int((old_mask & new_isolated).sum()), old_stats["nodes"]),
            "new_entrant_isolation_rate": _ratio(int(newly_added_isolated.sum()), int(entrants.sum())),
            "rescued_fraction_of_old_isolated": _ratio(int(rescued_old_isolated.sum()), old_stats["isolated_nodes"]),
            "new_minus_old_isolation_rate_pp": (
                (new_stats["isolation_rate"] - old_stats["isolation_rate"]) * 100
                if old_stats["isolation_rate"] is not None and new_stats["isolation_rate"] is not None
                else None
            ),
        })

    summary = {
        "source_dir": str(source_dir.resolve()),
        "old_window_config": str(old_config_path.resolve()),
        "new_window_config": str(new_config_path.resolve()),
        "assumption": expected_assumption,
        "window_count": len(rows),
        "aggregate_unit": "Node-window and edge-window observations; overlapping windows count the same person or relation repeatedly, not unique people or facts.",
        "edge_definition": "Only rows whose relation does not end with __rev; original directed triples are counted once.",
        "isolation_definition": "A selected node with zero incident original directed triples in the induced node set, before model training or masking.",
        "interpretation": (
            "Absolute isolated-node totals include newly enrolled nodes. Rescue among old isolated nodes, "
            "newly enrolled isolated nodes, overall rates, and the rate on the fixed old-node cohort are "
            "reported separately; a lower overall rate alone is not treated as evidence that old nodes were rescued."
        ),
        "aggregate": {
            "old_isolated_nodes_across_windows": int(sum(r["old_isolated_nodes"] for r in rows)),
            "new_isolated_nodes_across_windows": int(sum(r["new_isolated_nodes"] for r in rows)),
            "rescued_old_isolated_nodes_across_windows": int(sum(r["rescued_old_isolated_nodes"] for r in rows)),
            "remaining_old_isolated_nodes_across_windows": int(sum(r["remaining_old_isolated_nodes"] for r in rows)),
            "newly_added_isolated_nodes_across_windows": int(sum(r["newly_added_isolated_nodes"] for r in rows)),
            "newly_added_nodes_across_windows": int(sum(r["newly_added_nodes"] for r in rows)),
            "added_original_directed_triples_across_windows": int(sum(r["added_original_directed_triples"] for r in rows)),
        },
        "windows": rows,
    }
    return summary, rows


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    x = [row["center_year"] for row in rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axes[0].plot(x, [r["old_isolated_nodes"] for r in rows], label="Old isolated nodes", color="#5470a5", linewidth=1.8)
    axes[0].plot(x, [r["new_isolated_nodes"] for r in rows], label="New isolated nodes", color="#d26b3c", linewidth=1.8)
    axes[0].set(title="Absolute isolated-node counts", xlabel="Window center year", ylabel="Isolated nodes")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=.2)

    axes[1].plot(x, [r["old_isolation_rate"] for r in rows], label="Old rate", color="#5470a5", linewidth=1.8)
    axes[1].plot(x, [r["new_isolation_rate"] for r in rows], label="New overall rate", color="#d26b3c", linewidth=1.8)
    axes[1].plot(x, [r["new_isolation_rate_on_old_members"] for r in rows], label="New rate on old members", color="#3a8f68", linewidth=1.8)
    axes[1].plot(x, [r["new_entrant_isolation_rate"] for r in rows], label="New entrants' rate", color="#8a63a8", linewidth=1.5, alpha=.9)
    axes[1].set(title="Isolation proportions by cohort", xlabel="Window center year", ylabel="Isolated / selected")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=.2)

    fig.suptitle("DBpedia: isolation with an explicit alive-through-2026 assumption\nMissing death date and birth year > 1920", fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"alive2026_isolation_comparison.{extension}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--old-window-config", type=Path, default=DEFAULT_OLD_CONFIG)
    parser.add_argument("--new-window-config", type=Path, default=DEFAULT_NEW_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()

    source_dir = resolve(args.source_dir)
    old_config_path = resolve(args.old_window_config)
    new_config_path = resolve(args.new_window_config)
    output_dir = resolve(args.output_dir)
    summary, rows = compare(source_dir, old_config_path, new_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "window_isolation_comparison.tsv", sep="\t", index=False)
    plot(rows, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "window_count": len(rows), "aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
