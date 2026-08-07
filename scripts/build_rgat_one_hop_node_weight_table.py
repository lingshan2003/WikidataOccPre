#!/usr/bin/env python3
"""Build node-first L1 pair weight tables from a raw RGAT root export.

For every test target node, rows that differ only in ``source_visibility`` are
first merged for the same ``(source L1, relation, target L1)`` pair.  The main
``mean_a`` estimator is then the mean of those node-level attention masses over
target nodes that have at least one matching edge.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, Iterable, Iterator, Mapping, NamedTuple, TextIO


class NodePair(NamedTuple):
    experiment: str
    seed: str
    checkpoint: str
    target_l1_id: int
    target_l1: str
    relation_id: int
    relation: str
    source_l1_id: int
    source_l1: str


class PairAccumulator:
    def __init__(self) -> None:
        self.node_count = 0
        self.edge_count = 0
        self.mass_sum = 0.0
        self.mass_square_sum = 0.0

    def add(self, mass: float, edge_count: int) -> None:
        self.node_count += 1
        self.edge_count += edge_count
        self.mass_sum += mass
        self.mass_square_sum += mass * mass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="rgat_one_hop")
    parser.add_argument("--split", default="test")
    parser.add_argument("--message-passing-layer", type=int, default=1)
    return parser.parse_args()


def open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def roster_budgets(
    path: Path,
    experiment: str,
    split: str,
    layer: int,
) -> Dict[tuple[str, str, int, str], Dict[str, float]]:
    budgets: Dict[tuple[str, str, int, str], Dict[str, float]] = defaultdict(
        lambda: {"target_n": 0, "total_mass": 0.0, "typed_mass": 0.0}
    )
    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            if (
                row["experiment"] != experiment
                or row["split"] != split
                or int(row["message_passing_layer"]) != layer
            ):
                continue
            key = (
                row["seed"],
                row["checkpoint"],
                int(row["target_l1_id"]),
                row["target_l1"],
            )
            value = budgets[key]
            value["target_n"] += 1
            value["total_mass"] += float(row["total_attention_mass"])
            value["typed_mass"] += float(row["typed_attention_mass"])
    return dict(budgets)


def collapsed_node_pairs(
    path: Path,
    experiment: str,
    split: str,
    layer: int,
) -> Iterator[tuple[NodePair, int, float]]:
    """Yield one record per target node and pair, summing visibility groups."""
    current_identity = None
    current_pair = None
    current_edge_count = 0
    current_mass = 0.0

    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            if (
                row["experiment"] != experiment
                or row["split"] != split
                or int(row["message_passing_layer"]) != layer
            ):
                continue
            pair = NodePair(
                experiment=row["experiment"],
                seed=row["seed"],
                checkpoint=row["checkpoint"],
                target_l1_id=int(row["target_l1_id"]),
                target_l1=row["target_l1"],
                relation_id=int(row["relation_id"]),
                relation=row["relation"],
                source_l1_id=int(row["source_l1_id"]),
                source_l1=row["source_l1"],
            )
            identity = (row["checkpoint"], int(row["root_index"]), pair)
            if current_identity is not None and identity != current_identity:
                assert current_pair is not None
                yield current_pair, current_edge_count, current_mass
                current_edge_count = 0
                current_mass = 0.0
            current_identity = identity
            current_pair = pair
            current_edge_count += int(row["candidate_edge_count"])
            current_mass += float(row["attention_mass"])

    if current_identity is not None:
        assert current_pair is not None
        yield current_pair, current_edge_count, current_mass


def per_seed_rows(
    accumulators: Mapping[NodePair, PairAccumulator],
    budgets: Mapping[tuple[str, str, int, str], Mapping[str, float]],
) -> list[Dict[str, object]]:
    rows = []
    for pair, acc in sorted(
        accumulators.items(),
        key=lambda item: (
            item[0].source_l1_id,
            item[0].target_l1_id,
            item[0].relation_id,
            item[0].seed,
        ),
    ):
        budget_key = (pair.seed, pair.checkpoint, pair.target_l1_id, pair.target_l1)
        budget = budgets.get(budget_key)
        if budget is None:
            raise RuntimeError(f"Missing target roster budget for {budget_key}")
        n = acc.node_count
        mean_a = acc.mass_sum / n
        variance = max(acc.mass_square_sum / n - mean_a * mean_a, 0.0)
        target_n = int(budget["target_n"])
        total_mass = float(budget["total_mass"])
        typed_mass = float(budget["typed_mass"])
        rows.append({
            "source_l1": pair.source_l1,
            "target_l1": pair.target_l1,
            "relation": pair.relation,
            "seed": pair.seed,
            "mean_a": mean_a,
            "n": n,
            "node_mass_std": math.sqrt(variance),
            "matching_edge_n": acc.edge_count,
            "matching_edges_per_node": acc.edge_count / n,
            "target_n": target_n,
            "coverage": n / target_n,
            "mean_a_all_target": acc.mass_sum / target_n,
            "attention_share_of_Ot_budget": acc.mass_sum / total_mass,
            "attention_share_of_Ot_typed_budget": acc.mass_sum / typed_mass,
            "attention_mass_sum": acc.mass_sum,
            "source_l1_id": pair.source_l1_id,
            "target_l1_id": pair.target_l1_id,
            "relation_id": pair.relation_id,
            "experiment": pair.experiment,
            "checkpoint": pair.checkpoint,
        })
    return rows


def summary_rows(seed_rows: Iterable[Mapping[str, object]]) -> list[Dict[str, object]]:
    grouped: Dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            row["source_l1_id"], row["source_l1"],
            row["target_l1_id"], row["target_l1"],
            row["relation_id"], row["relation"],
        )
        grouped[key].append(row)

    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][4])):
        mean_values = [float(row["mean_a"]) for row in rows]
        shares = [float(row["attention_share_of_Ot_budget"]) for row in rows]
        n_values = [int(row["n"]) for row in rows]
        edge_values = [int(row["matching_edge_n"]) for row in rows]
        target_values = [int(row["target_n"]) for row in rows]
        support_is_constant = len(set(n_values)) == 1
        edges_are_constant = len(set(edge_values)) == 1
        targets_are_constant = len(set(target_values)) == 1
        output.append({
            "source_l1": key[1],
            "target_l1": key[3],
            "relation": key[5],
            "mean_a": fmean(mean_values),
            "n": n_values[0] if support_is_constant else fmean(n_values),
            "mean_a_seed_std": stdev(mean_values) if len(mean_values) > 1 else 0.0,
            "matching_edge_n": edge_values[0] if edges_are_constant else fmean(edge_values),
            "matching_edges_per_node": fmean(float(row["matching_edges_per_node"]) for row in rows),
            "target_n": target_values[0] if targets_are_constant else fmean(target_values),
            "coverage": fmean(float(row["coverage"]) for row in rows),
            "mean_a_all_target": fmean(float(row["mean_a_all_target"]) for row in rows),
            "attention_share_of_Ot_budget": fmean(shares),
            "attention_share_of_Ot_budget_seed_std": stdev(shares) if len(shares) > 1 else 0.0,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "source_l1_id": key[0],
            "target_l1_id": key[2],
            "relation_id": key[4],
        })
    return output


def validation_summary(
    seed_rows: Iterable[Mapping[str, object]],
    budgets: Mapping[tuple[str, str, int, str], Mapping[str, float]],
) -> Dict[str, object]:
    pair_mass: Dict[tuple[str, str, int, str], float] = defaultdict(float)
    pair_share: Dict[tuple[str, str, int, str], float] = defaultdict(float)
    for row in seed_rows:
        key = (
            str(row["seed"]),
            str(row["checkpoint"]),
            int(row["target_l1_id"]),
            str(row["target_l1"]),
        )
        pair_mass[key] += float(row["attention_mass_sum"])
        pair_share[key] += float(row["attention_share_of_Ot_budget"])

    checks = []
    for key, budget in sorted(budgets.items()):
        total = float(budget["total_mass"])
        typed = float(budget["typed_mass"])
        checks.append({
            "seed": key[0],
            "target_l1": key[3],
            "target_n": int(budget["target_n"]),
            "total_attention_budget": total,
            "typed_attention_budget": typed,
            "pair_attention_mass_sum": pair_mass.get(key, 0.0),
            "pair_share_sum": pair_share.get(key, 0.0),
            "pair_vs_typed_abs_error": abs(pair_mass.get(key, 0.0) - typed),
            "total_vs_target_n_abs_error": abs(total - float(budget["target_n"])),
        })
    return {
        "checks": checks,
        "max_pair_vs_typed_abs_error": max(row["pair_vs_typed_abs_error"] for row in checks),
        "max_total_vs_target_n_abs_error": max(row["total_vs_target_n_abs_error"] for row in checks),
        "max_pair_share_sum_abs_error": max(abs(row["pair_share_sum"] - 1.0) for row in checks),
    }


def main() -> None:
    args = parse_args()
    sparse_path = args.input_dir / "root_direct_attention_sparse_by_seed.csv.gz"
    roster_path = args.input_dir / "root_attention_roster_by_seed.csv.gz"
    for path in (sparse_path, roster_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    budgets = roster_budgets(
        roster_path, args.experiment, args.split, args.message_passing_layer
    )
    if not budgets:
        raise RuntimeError("No matching target-node roster rows")

    accumulators: Dict[NodePair, PairAccumulator] = defaultdict(PairAccumulator)
    collapsed_count = 0
    for pair, edge_count, mass in collapsed_node_pairs(
        sparse_path, args.experiment, args.split, args.message_passing_layer
    ):
        accumulators[pair].add(mass, edge_count)
        collapsed_count += 1
    if not accumulators:
        raise RuntimeError("No matching sparse node-pair rows")

    seed_rows = per_seed_rows(accumulators, budgets)
    summaries = summary_rows(seed_rows)
    validation = validation_summary(seed_rows, budgets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_fields = [
        "source_l1", "target_l1", "relation", "seed", "mean_a", "n",
        "node_mass_std", "matching_edge_n", "matching_edges_per_node", "target_n",
        "coverage", "mean_a_all_target", "attention_share_of_Ot_budget",
        "attention_share_of_Ot_typed_budget", "attention_mass_sum", "source_l1_id",
        "target_l1_id", "relation_id", "experiment", "checkpoint",
    ]
    summary_fields = [
        "source_l1", "target_l1", "relation", "mean_a", "n", "mean_a_seed_std",
        "matching_edge_n", "matching_edges_per_node", "target_n", "coverage",
        "mean_a_all_target", "attention_share_of_Ot_budget",
        "attention_share_of_Ot_budget_seed_std", "seed_count", "seeds",
        "source_l1_id", "target_l1_id", "relation_id",
    ]
    seed_path = args.output_dir / "rgat_one_hop_l1_pair_node_weight_by_seed.csv"
    summary_path = args.output_dir / "rgat_one_hop_l1_pair_node_weight_summary.csv"
    manifest_path = args.output_dir / "rgat_one_hop_l1_pair_node_weight_manifest.json"
    write_csv(seed_path, seed_rows, seed_fields)
    write_csv(summary_path, summaries, summary_fields)
    manifest_path.write_text(json.dumps({
        "input_sparse": str(sparse_path.resolve()),
        "input_roster": str(roster_path.resolve()),
        "experiment": args.experiment,
        "split": args.split,
        "message_passing_layer": args.message_passing_layer,
        "source_visibility_policy": "sum every visibility category before node aggregation",
        "unlabeled_source_policy": "retain as __UNLABELED__ instead of dropping",
        "mean_a_definition": (
            "For each (source L1, relation, target L1), sum matching head-averaged alpha "
            "inside each target node, then average over target nodes with at least one matching edge."
        ),
        "n_definition": "Number of target nodes with at least one matching edge; not an edge count.",
        "attention_share_of_Ot_budget_definition": (
            "Pair attention mass summed over all target-L1 nodes divided by their total incoming attention budget."
        ),
        "collapsed_target_pair_records": collapsed_count,
        "per_seed_rows": len(seed_rows),
        "summary_rows": len(summaries),
        "validation": validation,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(seed_path)
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
