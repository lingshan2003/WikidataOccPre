#!/usr/bin/env python3
"""Measure pre-GraphMask isolation and trace people when a life window widens."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.audit_sliding_window_artifacts import interval_mask, effective_death_years


def window_stats(node_mask, sources, targets):
    keep = node_mask[sources] & node_mask[targets]
    incident = np.bincount(np.concatenate((sources[keep], targets[keep])), minlength=len(node_mask))
    isolated = node_mask & (incident == 0)
    return {"nodes": int(node_mask.sum()), "original_triples": int(keep.sum()),
            "isolated_nodes": int(isolated.sum()),
            "isolation_rate": float(isolated.sum() / node_mask.sum()) if node_mask.any() else None}, isolated, incident


def analyze(source_dir, window_config, output_dir):
    nodes = pd.read_csv(source_dir / "nodes.csv")
    edges = pd.read_csv(source_dir / "edges.csv")
    edges = edges.loc[~edges.relation.str.endswith("__rev")].reset_index(drop=True)
    sources = edges.source_id.to_numpy(dtype=np.int64)
    targets = edges.target_id.to_numpy(dtype=np.int64)
    if not (np.array_equal(nodes.node_id.to_numpy()[sources], edges.source.to_numpy()) and
            np.array_equal(nodes.node_id.to_numpy()[targets], edges.target.to_numpy())):
        raise ValueError("Source URI/index mapping differs between node and edge tables")
    birth = pd.to_numeric(nodes.birth_year, errors="coerce").to_numpy(float)
    death = pd.to_numeric(nodes.death_year, errors="coerce").to_numpy(float)
    windows = json.loads(window_config.read_text())
    if windows.get("calendar_layout") != "sliding_windows" or windows.get("partial_date_policy") != "include_known_endpoint_periods":
        raise ValueError("Expected the life-window experiment policy")
    assumption = windows.get("birth_only_alive_assumption")
    effective_death, _ = effective_death_years(birth, death, assumption)
    date_eligible = ((np.isfinite(birth) & np.isfinite(effective_death) & (effective_death >= birth)) |
                     (np.isfinite(birth) ^ np.isfinite(effective_death)))
    small = interval_mask(birth, death, 1930, 1970, assumption)
    wide = interval_mask(birth, death, 1900, 2000, assumption)
    narrow_stats, narrow_isolated, narrow_incident = window_stats(small, sources, targets)
    wide_stats, wide_isolated, wide_incident = window_stats(wide, sources, targets)
    full_stats, full_isolated, _ = window_stats(np.ones(len(nodes), dtype=bool), sources, targets)
    rescued = narrow_isolated & ~wide_isolated
    remaining = narrow_isolated & wide_isolated
    newly_added = wide & ~small
    newly_added_isolated = wide_isolated & newly_added
    if not np.all(~small | wide):
        raise AssertionError("Expected nested node sets")
    if np.any((small & ~narrow_isolated) & wide_isolated):
        raise AssertionError("Widening an induced graph cannot disconnect an existing node")
    if int(wide_isolated.sum()) != int(remaining.sum() + newly_added_isolated.sum()):
        raise AssertionError("Isolation transition accounting does not balance")

    annual = []
    for period in windows["periods"]:
        mask = interval_mask(birth, death, period["start"], period["end"], assumption)
        stats, _, _ = window_stats(mask, sources, targets)
        annual.append({"context": period["id"], "center_year": (period["start"]+period["end"])//2,
                       "start": period["start"], "end": period["end"], **stats})
    sensitivity = []
    for half_width in (0, 5, 10, 20, 30, 40, 50):
        mask = interval_mask(birth, death, 1950-half_width, 1950+half_width, assumption)
        stats, isolated, _ = window_stats(mask, sources, targets)
        sensitivity.append({"center_year": 1950, "half_width": half_width,
                            "start": 1950-half_width, "end": 1950+half_width, **stats,
                            "isolated_among_original_1950_nodes": int((small & isolated).sum()),
                            "original_1950_nodes_present": int((small & mask).sum()),
                            "rescued_original_1950_isolated_nodes": int((narrow_isolated & ~isolated & mask).sum()) if half_width >= 20 else None})

    # Unique neighbours are counted separately from multi-predicate triples.
    pairs = np.unique(np.concatenate((np.column_stack((sources, targets)), np.column_stack((targets, sources)))), axis=0)
    degree = np.bincount(pairs[:, 0], minlength=len(nodes))
    eligible_neighbors = np.bincount(pairs[date_eligible[pairs[:, 1]], 0], minlength=len(nodes))
    reason = np.full(len(nodes), "", dtype=object)
    reason[wide_isolated & (degree == 0)] = "already_isolated_in_full_source"
    reason[wide_isolated & (degree > 0) & (eligible_neighbors == 0)] = "all_neighbors_missing_or_invalid_dates"
    reason[wide_isolated & (degree > 0) & (eligible_neighbors == degree)] = "all_neighbors_date_eligible_but_outside_1900_2000"
    reason[wide_isolated & (eligible_neighbors > 0) & (eligible_neighbors < degree)] = "mixed_date_exclusion_and_outside_window"
    people = nodes.loc[wide].copy()
    people.insert(0, "source_node_index", np.flatnonzero(wide))
    people["in_1930_1970"] = small[wide]
    people["isolated_in_1930_1970"] = narrow_isolated[wide]
    people["isolated_in_1900_2000"] = wide_isolated[wide]
    people["rescued_by_widening"] = rescued[wide]
    people["incident_original_triples_1930_1970"] = narrow_incident[wide]
    people["incident_original_triples_1900_2000"] = wide_incident[wide]
    people["full_graph_unique_neighbors"] = degree[wide]
    people["wide_isolation_reason"] = reason[wide]

    summary = {
        "source_dir": str(source_dir.resolve()), "window_config": str(window_config.resolve()),
        "definition": "Selected node with zero incident induced edges, before GraphMask. Original predicates only; generated reverse messages and group deduplication do not change whether a node has a neighbour.",
        "date_policy": "Valid complete life interval intersects window; sole known endpoint lies inside inclusive window; both missing or death before birth excluded. " +
                       ("Explicit birth-only alive assumption applies to membership only." if assumption else "No missing-death extrapolation."),
        "birth_only_alive_assumption": assumption,
        "small_1930_1970": narrow_stats, "wide_1900_2000": wide_stats, "full_source": full_stats,
        "transitions": {"rescued_original_isolated": int(rescued.sum()), "remaining_original_isolated": int(remaining.sum()),
                        "rescued_fraction_of_original_isolated": float(rescued.sum()/narrow_isolated.sum()),
                        "remaining_isolation_rate_on_fixed_original_nodes": float(remaining.sum()/small.sum()),
                        "newly_added_nodes": int(newly_added.sum()), "newly_added_isolated": int(newly_added_isolated.sum()),
                        "net_isolated_count_reduction": int(narrow_isolated.sum()-wide_isolated.sum()),
                        "net_isolated_count_reduction_fraction": float((narrow_isolated.sum()-wide_isolated.sum())/narrow_isolated.sum())},
        "wide_isolation_reasons": pd.Series(reason[wide_isolated]).value_counts().to_dict(),
        "annual_min_isolation_rate": min(annual, key=lambda r: r["isolation_rate"]),
        "annual_max_isolation_rate": max(annual, key=lambda r: r["isolation_rate"]),
        "annual_min_isolated_count": min(annual, key=lambda r: r["isolated_nodes"]),
        "annual_max_isolated_count": max(annual, key=lambda r: r["isolated_nodes"]),
        "scope": "Graph composition comparison only; no model training and no change to the running sliding-window experiment.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    pd.DataFrame(annual).to_csv(output_dir / "annual_window_isolation.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity).to_csv(output_dir / "center1950_window_width_sensitivity.tsv", sep="\t", index=False)
    people.to_csv(output_dir / "people_isolation_transitions.csv.gz", index=False, compression="gzip")
    return summary, annual, sensitivity


def plot(summary, annual, sensitivity, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = [r["center_year"] for r in annual]
    y = [r["isolation_rate"] for r in annual]
    axes[0].plot(x, y, color="#2673ad", linewidth=2)
    axes[0].scatter([1950], [summary["small_1930_1970"]["isolation_rate"]], color="#2673ad", zorder=3)
    axes[0].axhline(summary["wide_1900_2000"]["isolation_rate"], color="#d16c23", linestyle="--", label=f"1900–2000 window: {summary['wide_1900_2000']['isolation_rate']:.2%}")
    axes[0].set(title="Annual sliding windows (center ±20 years)", xlabel="Window center year", ylabel="Isolated nodes / selected nodes")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    axes[1].plot([2*r["half_width"]+1 for r in sensitivity], [r["isolation_rate"] for r in sensitivity], marker="o", color="#2673ad")
    axes[1].scatter([41, 101], [summary["small_1930_1970"]["isolation_rate"], summary["wide_1900_2000"]["isolation_rate"]], color=["#2673ad", "#d16c23"], s=65, zorder=3)
    for x0, value, text in ((41, summary["small_1930_1970"]["isolation_rate"], f"1930–1970\n{summary['small_1930_1970']['isolation_rate']:.2%}"),
                            (101, summary["wide_1900_2000"]["isolation_rate"], f"1900–2000\n{summary['wide_1900_2000']['isolation_rate']:.2%}")):
        axes[1].annotate(text, (x0, value), xytext=((-10, 12) if x0 == 101 else (10, 12)),
                         ha="right" if x0 == 101 else "left", textcoords="offset points", fontsize=10)
    axes[1].set(title="Widening the window centered on 1950", xlabel="Inclusive window length (years)", ylabel="Isolated nodes / selected nodes")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("DBpedia: isolation introduced by life-window induced subgraphs", fontsize=14)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"window_isolation_comparison.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="artifacts/dbpedia_2022_priority_v1")
    parser.add_argument("--window-config", default="config/dbpedia_life_windows_1900_2000_pm20_step1_v1.json")
    parser.add_argument("--output-dir", default="artifacts/dbpedia_window_isolation_2026_10_09")
    args = parser.parse_args()
    def resolved(value):
        p = Path(value)
        return p if p.is_absolute() else ROOT / p
    summary, annual, sensitivity = analyze(resolved(args.source_dir), resolved(args.window_config), resolved(args.output_dir))
    plot(summary, annual, sensitivity, resolved(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
