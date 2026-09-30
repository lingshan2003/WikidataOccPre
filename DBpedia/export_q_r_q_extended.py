#!/usr/bin/env python3
"""Export the reviewed DBpedia graph in Q_R_Q_extended.txt's 15-column schema.

One occupation is assigned per graph node. Source priority is dbo:occupation,
dbo:profession, explicit dbo role type, then reviewed Wikidata Q role type.
Within the chosen source, the earliest source-file assertion wins. The complete
set of alternatives remains in the companion assignment audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .common import DBO, gzip_rows, require_files, tsv_writer, write_json
else:
    from common import DBO, gzip_rows, require_files, tsv_writer, write_json


ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("occupation", "profession", "dbo_role_type", "wikidata_role_type")
COLUMNS = [
    "Node1", "Relation", "Node2", "Node1_Birth", "Node1_Death",
    "Node2_Birth", "Node2_Death", "Node1_occ_level1", "Node1_occ_level2",
    "Node1_occ_level3", "Node2_occ_level1", "Node2_occ_level2",
    "Node2_occ_level3", "Node1_Country", "Node2_Country",
]
MISSING_TOKENS = {
    "", "-", "missing", "unknown", "none", "null", "nan", "n/a", "na",
    "not available", "not known", "not specified",
}


def occupation_label(readable: str) -> str:
    """Merge spelling/case variants without imposing a career taxonomy."""
    value = readable.strip().replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value).casefold()
    if value in MISSING_TOKENS:
        raise ValueError(f"Chosen occupation is an empty/missing token: {readable!r}")
    return value


def run(processed: Path, output_dir: Path) -> dict:
    graph_dir = processed / "04_graph"
    nodes_file = graph_dir / "graph_nodes.tsv.gz"
    edges_file = graph_dir / "graph_edges.tsv.gz"
    evidence_file = graph_dir / "candidate_evidence.tsv.gz"
    require_files(nodes_file, edges_file, evidence_file)

    nodes = {}
    primary = {}
    for row in gzip_rows(nodes_file):
        uri = row["person_uri"]
        if uri in nodes:
            raise ValueError(f"Duplicate graph node: {uri}")
        if row["primary_source"] not in SOURCES:
            raise ValueError(f"Unexpected primary source for {uri}: {row['primary_source']}")
        nodes[uri] = row
        primary[uri] = row["primary_source"]
    if not nodes:
        raise ValueError("Final DBpedia graph has no nodes")

    evidence = defaultdict(dict)
    for row in gzip_rows(evidence_file):
        uri = row["person_uri"]
        if uri not in nodes or row["candidate_source"] != primary[uri]:
            continue
        raw = row["raw_value"]
        line = int(row["source_line"])
        previous = evidence[uri].get(raw)
        if previous is None or line < previous[0]:
            evidence[uri][raw] = (line, row["readable_value"], row["source_file"])

    assignments = {}
    source_counts = Counter()
    labels = Counter()
    multiple = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_file = output_dir / "occupation_assignments.tsv.gz"
    with gzip.open(audit_file, "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "person_uri", "occupation_label", "primary_source", "chosen_raw_value",
            "chosen_readable_value", "chosen_source_file", "chosen_source_line",
            "primary_raw_value_count", "all_primary_raw_values_json",
        ])
        for uri in sorted(nodes):
            facts = evidence.get(uri)
            if not facts:
                raise ValueError(f"No evidence in the declared primary source for graph node {uri}")
            chosen_raw, (line, readable, source_file) = min(
                facts.items(), key=lambda item: (item[1][0], item[0])
            )
            label = occupation_label(readable)
            assignments[uri] = label
            source = primary[uri]
            source_counts[source] += 1
            labels[label] += 1
            multiple[source] += len(facts) > 1
            writer.writerow({
                "person_uri": uri,
                "occupation_label": label,
                "primary_source": source,
                "chosen_raw_value": chosen_raw,
                "chosen_readable_value": readable,
                "chosen_source_file": source_file,
                "chosen_source_line": line,
                "primary_raw_value_count": len(facts),
                "all_primary_raw_values_json": json.dumps(sorted(facts), ensure_ascii=False),
            })

    relation_mapping = {}
    relation_counts = Counter()
    output_file = output_dir / "DBpedia_R_R_extended.csv"
    temporary_file = output_dir / ".DBpedia_R_R_extended.csv.tmp"
    try:
        with temporary_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for edge in gzip_rows(edges_file):
                subject, object_ = edge["subject_uri"], edge["object_uri"]
                if subject not in nodes or object_ not in nodes:
                    raise ValueError("Final DBpedia edge has an endpoint missing from graph_nodes")
                predicate = edge["predicate_uri"]
                if not predicate.startswith(DBO):
                    raise ValueError(f"Non-dbo relation needs an explicit name mapping: {predicate}")
                relation = predicate[len(DBO):]
                previous = relation_mapping.get(relation)
                if previous is not None and previous != predicate:
                    raise ValueError(f"Relation local-name collision: {relation}")
                relation_mapping[relation] = predicate
                relation_counts[relation] += 1
                left, right = nodes[subject], nodes[object_]
                writer.writerow({
                    "Node1": subject,
                    "Relation": relation,
                    "Node2": object_,
                    "Node1_Birth": left["birth_year"],
                    "Node1_Death": left["death_year"],
                    "Node2_Birth": right["birth_year"],
                    "Node2_Death": right["death_year"],
                    "Node1_occ_level1": assignments[subject],
                    "Node1_occ_level2": "",
                    "Node1_occ_level3": "",
                    "Node2_occ_level1": assignments[object_],
                    "Node2_occ_level2": "",
                    "Node2_occ_level3": "",
                    "Node1_Country": "",
                    "Node2_Country": "",
                })
        os.replace(temporary_file, output_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()

    with (output_dir / "relation_mapping.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["relation", "predicate_uri", "original_directed_edges"])
        for relation, count in sorted(relation_counts.items()):
            writer.writerow({
                "relation": relation,
                "predicate_uri": relation_mapping[relation],
                "original_directed_edges": count,
            })
    with (output_dir / "occupation_label_counts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["occupation_label", "graph_nodes"])
        for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"occupation_label": label, "graph_nodes": count})

    result = {
        "output": str(output_file),
        "schema": "Q_R_Q_extended.txt 15-column edge-centric CSV",
        "graph_nodes": len(nodes),
        "original_directed_edges": sum(relation_counts.values()),
        "relation_types": len(relation_counts),
        "assigned_occupation_labels": len(labels),
        "nodes_by_primary_source": dict(source_counts),
        "nodes_with_multiple_values_in_primary_source": dict(multiple),
        "source_priority": list(SOURCES),
        "within_source_tie_rule": "Choose the distinct raw value with the earliest source_file line; raw URI breaks an exact line tie.",
        "label_normalization": "Read source readable_value; replace underscores, split lower-to-upper CamelCase, collapse spaces, casefold. No occupational taxonomy is applied.",
        "compatibility_columns": "The selected single occupation occupies occupation_level1 solely for the existing preparation interface. Level2, level3, and country are empty; this is not the Wikidata Level-1 taxonomy.",
    }
    write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "external_data/dbpedia/processed")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    processed = args.processed_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else processed / "model_input"
    print(json.dumps(run(processed, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
