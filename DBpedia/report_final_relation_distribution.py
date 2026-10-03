#!/usr/bin/env python3
"""Report relation proportions in the unpartitioned final DBpedia graph."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("external_data/dbpedia/processed"))
    args = parser.parse_args()
    graph_dir = args.processed_dir / "04_graph"
    counts = Counter()
    groups = Counter()
    metadata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with gzip.open(graph_dir / "graph_edges.tsv.gz", "rt", encoding="utf-8", newline="") as handle:
        for edge in csv.DictReader(handle, delimiter="\t"):
            predicate = edge["predicate_uri"].rsplit("/", 1)[-1]
            counts[predicate] += 1
            groups[edge["social_group"]] += 1
            metadata[predicate].add((edge["social_group"], edge["selection_scope"]))
    total = sum(counts.values())
    summary = json.loads((graph_dir / "summary.json").read_text(encoding="utf-8"))
    if total != summary["graph_edges"] or len(counts) != summary["graph_predicates"]:
        raise ValueError("Relation counts disagree with the final graph summary")
    for predicate, values in metadata.items():
        if len(values) != 1:
            raise ValueError(f"Predicate has inconsistent group/scope assignments: {predicate}")

    relation_rows = []
    for rank, (predicate, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1):
        group, scope = next(iter(metadata[predicate]))
        relation_rows.append({
            "rank": rank,
            "predicate": predicate,
            "original_triples": count,
            "share_percent": f"{100 * count / total:.3f}",
            "social_group": group,
            "selection_scope": scope,
        })
    group_rows = [
        {"social_group": group, "original_triples": count, "share_percent": f"{100 * count / total:.3f}"}
        for group, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))
    ]
    relation_path = graph_dir / "final_relation_distribution.tsv"
    group_path = graph_dir / "final_relation_group_distribution.tsv"
    write_tsv(relation_path, (
        "rank", "predicate", "original_triples", "share_percent", "social_group", "selection_scope"
    ), relation_rows)
    write_tsv(group_path, ("social_group", "original_triples", "share_percent"), group_rows)
    print(f"[final relations] {total} original triples, {len(relation_rows)} predicates: {relation_path}")
    print(f"[final relations] {len(group_rows)} groups: {group_path}")


if __name__ == "__main__":
    main()
