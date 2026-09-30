"""Stage 1: extract the broad type-based person candidate universe."""

from __future__ import annotations

import csv
import gzip
from collections import Counter
from pathlib import Path

from common import DBR, RDF_TYPE, triples, uri_from_token, uri_warning, write_json


TYPE_BITS = {
    b"<http://dbpedia.org/ontology/Person>": 1,
    b"<http://schema.org/Person>": 2,
    b"<http://xmlns.com/foaf/0.1/Person>": 4,
    b"<http://www.wikidata.org/entity/Q5>": 8,
}
TYPE_COLUMNS = (
    ("dbo_person", 1),
    ("schema_person", 2),
    ("foaf_person", 4),
    ("wikidata_q5_type", 8),
)


def run(type_dump: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    people: dict[str, int] = {}
    matched = Counter()
    scanned = malformed = 0
    for line_no, subject, predicate, obj in triples(type_dump):
        scanned = line_no
        if subject is None:
            malformed += 1
            continue
        if predicate != RDF_TYPE:
            continue
        bit = TYPE_BITS.get(obj)
        if bit is None:
            continue
        uri = uri_from_token(subject)
        people[uri] = people.get(uri, 0) | bit
        matched[obj.decode("ascii")] += 1

    type_counts = Counter()
    warning_counts = Counter()
    path = output_dir / "person_candidates.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["resource_uri", "resource_local_name", *(name for name, _ in TYPE_COLUMNS), "uri_warning"])
        for uri in sorted(people):
            mask = people[uri]
            warning = uri_warning(uri)
            warning_counts[warning or "none"] += 1
            for name, bit in TYPE_COLUMNS:
                type_counts[name] += bool(mask & bit)
            writer.writerow([
                uri, uri.removeprefix(DBR) if uri.startswith(DBR) else "",
                *(int(bool(mask & bit)) for _, bit in TYPE_COLUMNS), warning,
            ])

    result = {
        "stage": "01_people",
        "input_file": str(type_dump),
        "input_rows": scanned,
        "malformed_rows": malformed,
        "candidate_uris": len(people),
        "type_flag_distinct_uri_counts": dict(type_counts),
        "matched_type_triples": dict(matched),
        "uri_warning_counts": dict(warning_counts),
        "candidate_rule": "Union of dbo:Person, schema:Person, foaf:Person and wikidata:Q5 types in the transitive type dump.",
        "output": str(path),
    }
    write_json(output_dir / "summary.json", result)
    return result
