"""Stage 3: resolve birth/death years and retain date-qualified social edges."""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from common import DBO, gzip_rows, parse_date_literal, triples, tsv_writer, uri_from_token, write_json


DATE_PREDICATES = {
    f"<{DBO}birthDate>".encode(): ("birth", "birthDate"),
    f"<{DBO}birthYear>".encode(): ("birth", "birthYear"),
    f"<{DBO}deathDate>".encode(): ("death", "deathDate"),
    f"<{DBO}deathYear>".encode(): ("death", "deathYear"),
}
NODE_COLUMNS = [
    "person_uri", "birth_year", "death_year", "birth_sources", "death_sources",
    "birth_year_conflict", "death_year_conflict", "chronology_warning",
]


def run(literal_dump: Path, social_edge_file: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = set()
    input_predicates = Counter()
    for row in gzip_rows(social_edge_file):
        participants.update((row["subject_uri"], row["object_uri"]))
        input_predicates[row["predicate_uri"]] += 1

    evidence = defaultdict(lambda: {"birth": set(), "death": set()})
    source_flags = defaultdict(set)
    date_facts, parse_flags = Counter(), Counter()
    scanned = malformed = 0
    with gzip.open(output_dir / "date_evidence.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "person_uri", "predicate", "raw_literal", "source_line", "parsed_year", "quality_flag",
        ])
        for line_no, subject, predicate, value in triples(literal_dump):
            scanned = line_no
            if subject is None:
                malformed += 1
                continue
            info = DATE_PREDICATES.get(predicate)
            if info is None:
                continue
            uri = uri_from_token(subject)
            if uri not in participants:
                continue
            side, name = info
            year, _shape, flag = parse_date_literal(value)
            if year is not None:
                evidence[uri][side].add(year)
            source_flags[uri].add(name)
            date_facts[name] += 1
            parse_flags[flag or "valid"] += 1
            writer.writerow({
                "person_uri": uri,
                "predicate": name,
                "raw_literal": value.decode("utf-8", "replace"),
                "source_line": line_no,
                "parsed_year": "" if year is None else year,
                "quality_flag": flag,
            })

    person_rows = {}
    usable = set()
    year_stats = Counter()
    for uri in participants:
        facts = evidence[uri]
        birth = next(iter(facts["birth"])) if len(facts["birth"]) == 1 else None
        death = next(iter(facts["death"])) if len(facts["death"]) == 1 else None
        birth_conflict = len(facts["birth"]) > 1
        death_conflict = len(facts["death"]) > 1
        warning = ""
        if birth is not None and death is not None:
            if death < birth:
                warning = "death_before_birth"
            elif death - birth > 125:
                warning = "lifespan_over_125_years"
        person_rows[uri] = {
            "person_uri": uri,
            "birth_year": "" if birth is None else birth,
            "death_year": "" if death is None else death,
            "birth_sources": "|".join(sorted(source_flags[uri] & {"birthDate", "birthYear"})),
            "death_sources": "|".join(sorted(source_flags[uri] & {"deathDate", "deathYear"})),
            "birth_year_conflict": int(birth_conflict),
            "death_year_conflict": int(death_conflict),
            "chronology_warning": warning,
        }
        if birth is not None or death is not None:
            usable.add(uri)
        year_stats["birth"] += birth is not None
        year_stats["death"] += death is not None
        year_stats["both"] += birth is not None and death is not None
        year_stats["birth_conflict"] += birth_conflict
        year_stats["death_conflict"] += death_conflict
        if warning:
            year_stats[warning] += 1

    with gzip.open(output_dir / "all_participant_years.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, NODE_COLUMNS + ["has_either_unique_year"])
        for uri in sorted(participants):
            writer.writerow({**person_rows[uri], "has_either_unique_year": int(uri in usable)})

    coverage = Counter()
    group_flow, predicate_flow = defaultdict(Counter), defaultdict(Counter)
    retained = set()
    with gzip.open(social_edge_file, "rt", encoding="utf-8", newline="") as source, gzip.open(
        output_dir / "graph_edges.tsv.gz", "wt", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = tsv_writer(destination, reader.fieldnames)
        for row in reader:
            s, o = row["subject_uri"], row["object_uri"]
            covered = int(s in usable) + int(o in usable)
            coverage[covered] += 1
            group_flow[row["social_group"]][covered] += 1
            predicate_flow[row["predicate_uri"]][covered] += 1
            if covered == 2:
                writer.writerow(row)
                retained.update((s, o))

    with gzip.open(output_dir / "graph_nodes.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, NODE_COLUMNS)
        for uri in sorted(retained):
            writer.writerow(person_rows[uri])

    with (output_dir / "relation_flow.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "predicate_uri", "selected_edges", "neither_endpoint_with_year",
            "one_endpoint_with_year", "both_endpoints_with_year",
        ])
        for predicate in sorted(input_predicates, key=lambda uri: (-input_predicates[uri], uri)):
            counts = predicate_flow[predicate]
            writer.writerow({
                "predicate_uri": predicate,
                "selected_edges": input_predicates[predicate],
                "neither_endpoint_with_year": counts[0],
                "one_endpoint_with_year": counts[1],
                "both_endpoints_with_year": counts[2],
            })
    with (output_dir / "group_flow.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "social_group", "selected_edges", "neither_endpoint_with_year",
            "one_endpoint_with_year", "both_endpoints_with_year",
        ])
        for group, counts in sorted(group_flow.items()):
            writer.writerow({
                "social_group": group,
                "selected_edges": sum(counts.values()),
                "neither_endpoint_with_year": counts[0],
                "one_endpoint_with_year": counts[1],
                "both_endpoints_with_year": counts[2],
            })

    result = {
        "stage": "03_dates",
        "literal_dump": str(literal_dump),
        "literal_rows_scanned": scanned,
        "malformed_literal_rows": malformed,
        "input_social_edges": sum(input_predicates.values()),
        "input_social_participants": len(participants),
        "date_facts_by_predicate": dict(date_facts),
        "date_parse_flags": dict(parse_flags),
        "people_with_birth_year": year_stats["birth"],
        "people_with_death_year": year_stats["death"],
        "people_with_both_years": year_stats["both"],
        "people_with_either_year": len(usable),
        "people_with_birth_year_conflict": year_stats["birth_conflict"],
        "people_with_death_year_conflict": year_stats["death_conflict"],
        "chronology_warning_counts": {k: year_stats[k] for k in ("death_before_birth", "lifespan_over_125_years")},
        "edge_endpoint_coverage": {str(n): coverage[n] for n in range(3)},
        "output_edges": coverage[2],
        "output_nodes": len(retained),
        "output_predicates": sum(bool(counts[2]) for counts in predicate_flow.values()),
        "eligibility_rule": "Each endpoint has at least one unique parseable birth or death year from birthDate/birthYear/deathDate/deathYear. Conflicted side does not qualify; other valid side can qualify.",
        "period_rule_note": "Signed birth_year and death_year are preserved. Historical-period membership is not assigned during extraction.",
    }
    write_json(output_dir / "summary.json", result)
    return result
