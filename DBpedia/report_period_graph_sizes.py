#!/usr/bin/env python3
"""Count relation-bearing people and original triples in DBpedia period graphs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "period_id", "period_label", "nonisolated_nodes", "original_triples",
    "all_period_nodes", "isolated_nodes", "generated_reverse_edges",
    "unique_unordered_person_pairs", "nonisolated_components", "two_person_components",
)


def graph_components(nodes: set[int], pairs: set[tuple[int, int]]) -> tuple[int, int]:
    parent = {node: node for node in nodes}
    size = {node: 1 for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target in pairs:
        first, second = find(source), find(target)
        if first != second:
            if size[first] < size[second]:
                first, second = second, first
            parent[second] = first
            size[first] += size[second]
    roots = {find(node) for node in nodes}
    return len(roots), sum(size[root] == 2 for root in roots)


def period_row(root: Path, period: dict) -> dict:
    period_id = period["id"]
    directory = root / period_id
    summary = json.loads((directory / "split_summary.json").read_text(encoding="utf-8"))
    originals: set[tuple[int, str, int]] = set()
    reverses: set[tuple[int, str, int]] = set()
    endpoints: set[int] = set()
    person_pairs: set[tuple[int, int]] = set()
    with (directory / "edges.csv").open(encoding="utf-8", newline="") as handle:
        for edge in csv.DictReader(handle):
            source, target = int(edge["source_id"]), int(edge["target_id"])
            relation = edge["relation"]
            if relation.endswith("__rev"):
                reverses.add((source, relation.removesuffix("__rev"), target))
            else:
                originals.add((source, relation, target))
                endpoints.update((source, target))
                person_pairs.add((min(source, target), max(source, target)))
    if len(originals) + len(reverses) != summary["directed_edges"]:
        raise ValueError(f"Directed edge count disagrees with summary for {period_id}")
    if {(target, relation, source) for source, relation, target in originals} != reverses:
        raise ValueError(f"Original and generated reverse edges do not pair for {period_id}")
    if len(originals) != len(reverses):
        raise ValueError(f"Original/reverse edge counts differ for {period_id}")
    if endpoints and (min(endpoints) < 0 or max(endpoints) >= summary["nodes"]):
        raise ValueError(f"An edge endpoint is outside the node table for {period_id}")
    components, dyads = graph_components(endpoints, person_pairs)
    return {
        "period_id": period_id,
        "period_label": period["label"],
        "nonisolated_nodes": len(endpoints),
        "original_triples": len(originals),
        "all_period_nodes": summary["nodes"],
        "isolated_nodes": summary["nodes"] - len(endpoints),
        "generated_reverse_edges": len(reverses),
        "unique_unordered_person_pairs": len(person_pairs),
        "nonisolated_components": components,
        "two_person_components": dyads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-root", type=Path, required=True)
    parser.add_argument("--period-config", type=Path, required=True)
    args = parser.parse_args()
    periods = json.loads(args.period_config.read_text(encoding="utf-8"))["periods"]
    rows = [period_row(args.period_root, period) for period in periods]
    output = args.period_root / "graph_sizes.tsv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[graph sizes] {len(rows)} periods: {output}")
    for row in rows:
        print(
            f"{row['period_id']}: nodes_with_edges={row['nonisolated_nodes']}, "
            f"original_triples={row['original_triples']}, isolates={row['isolated_nodes']}"
        )


if __name__ == "__main__":
    main()
