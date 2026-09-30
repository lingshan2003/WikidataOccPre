#!/usr/bin/env python3
"""Inventory reviewed role classes and their evidence/selected-node counts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from occupations import ROLE_TYPES


SOURCES = ("dbo_role_type", "wikidata_role_type")
FIELDS = (
    "source", "type_id", "label_en", "label_zh", "accepted_by_rule",
    "evidence_nodes_in_final_graph", "selected_as_unique_occupation_nodes",
)


def tsv_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def local_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].removesuffix(">")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("external_data/dbpedia/processed"))
    parser.add_argument("--q-review", type=Path, default=Path("DBpedia/rules/q_class_review.tsv"))
    args = parser.parse_args()

    graph_nodes = {
        row["person_uri"] for row in tsv_rows(args.processed_dir / "04_graph/graph_nodes.tsv.gz")
    }
    review = {
        row["qid"]: row for row in tsv_rows(args.q_review)
        if row["decision"] == "accept"
    }
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in tsv_rows(args.processed_dir / "04_graph/candidate_evidence.tsv.gz"):
        if row["person_uri"] in graph_nodes and row["candidate_source"] in SOURCES:
            evidence[(row["candidate_source"], local_name(row["raw_value"]))].add(
                row["person_uri"]
            )
    chosen = Counter()
    for row in tsv_rows(args.processed_dir / "model_input/occupation_assignments.tsv.gz"):
        if row["primary_source"] in SOURCES:
            chosen[(row["primary_source"], local_name(row["chosen_raw_value"]))] += 1

    rows = []
    for name in sorted(ROLE_TYPES):
        source = "dbo_role_type"
        rows.append({
            "source": source,
            "type_id": f"dbo:{name}",
            "label_en": name,
            "label_zh": "",
            "accepted_by_rule": "yes",
            "evidence_nodes_in_final_graph": len(evidence[source, name]),
            "selected_as_unique_occupation_nodes": chosen[source, name],
        })
    for qid, definition in sorted(review.items(), key=lambda item: item[0]):
        source = "wikidata_role_type"
        rows.append({
            "source": source,
            "type_id": qid,
            "label_en": definition["label_en"],
            "label_zh": definition["label_zh"],
            "accepted_by_rule": "yes",
            "evidence_nodes_in_final_graph": len(evidence[source, qid]),
            "selected_as_unique_occupation_nodes": chosen[source, qid],
        })
    summary = json.loads((args.processed_dir / "model_input/summary.json").read_text(encoding="utf-8"))
    for source in SOURCES:
        selected = sum(
            row["selected_as_unique_occupation_nodes"] for row in rows
            if row["source"] == source
        )
        if selected != summary["nodes_by_primary_source"][source]:
            raise ValueError(f"Chosen {source} types disagree with the DBpedia graph summary")
    output = args.processed_dir / "model_input/occupation_type_inventory.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[occupation type inventory] {len(rows)} reviewed classes: {output}")


if __name__ == "__main__":
    main()
