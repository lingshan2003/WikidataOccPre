#!/usr/bin/env python3
"""Rebuild the Freebase Easy person graph in auditable, resumable stages.

All exports are CPU-only. Every selected raw fact keeps its source line and
original direction. A person with one OR MORE Profession values is eligible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import groupby
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTS = ROOT / "external_data/freebase_easy/freebase-easy-latest/facts.txt"
DEFAULT_LINKS = ROOT / "external_data/freebase_easy/freebase-easy-latest/freebase-links.txt"
DEFAULT_OUTPUT = ROOT / "external_data/freebase/processed"
DEFAULT_RULES = Path(__file__).with_name("relation_rules.json")
STAGES = ("people", "cohort", "pairs", "relations", "final", "stats")
DIRS = {name: f"{i:02d}_{name}" for i, name in enumerate(STAGES, 1)}
OUTPUT_FILES = {
    "people": ("person_names.txt",),
    "cohort": ("person_attribute_facts.tsv", "cohort_people.tsv", "cohort_person_names.txt"),
    "pairs": ("person_name_pairs.tsv", "cohort_name_pairs.tsv", "predicate_coverage.tsv"),
    "relations": ("review_candidate_facts.tsv", "main_relation_facts.tsv"),
    "final": ("nodes.csv", "main_relation_facts.csv", "name_id_status.tsv"),
    "stats": ("occupation_people_counts.tsv", "relation_counts.tsv"),
}
PAIR_HEADER = b"source_line\tsubject_name\tpredicate\tobject_name\n"
ATTR_HEADER = b"source_line\tperson_name\tpredicate\traw_value\n"
COHORT_HEADER = b"name\tprofessions_json\tbirth_years_json\tdeath_years_json\tbirth_literals_json\tdeath_literals_json\n"
BIRTH = {b"Date of birth", b"Date of Birth"}
DEATH = {b"Date of death", b"Date of Death"}
DATE_RE = re.compile(
    rb'^"(?P<lex>[^"]+)"\^\^<https?://www\.w3\.org/2001/XMLSchema#'
    rb'(?P<type>date|gYear|gYearMonth|dateTime)>$'
)
LEX_RE = re.compile(
    rb'^(?P<year>-?\d{4,6})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?'
    rb'(?:T(?P<hour>\d{2})(?::(?P<minute>\d{2}))?'
    rb'(?::(?P<second>\d{2}))?(?:Z|[+-]\d{2}:\d{2})?)?$'
)
URI_PREFIX = b"<http://rdf.freebase.com/ns/"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--resume", action="store_true", help="skip complete stages after an interrupted --stage all run")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def encode_json(values) -> bytes:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def token(value: bytes) -> bytes:
    value = value.strip()
    return value[1:-1] if value.startswith(b"<") and value.endswith(b">") else value


def fact(raw: bytes) -> tuple[bytes, bytes, bytes] | None:
    fields = raw.rstrip(b"\r\n").split(b"\t", 3)
    if len(fields) != 4 or fields[3].strip() != b".":
        return None
    subject, predicate, obj = (token(value) for value in fields[:3])
    return (subject, predicate, obj) if subject and predicate and obj else None


def parse_year(value: bytes) -> tuple[int | None, str]:
    """Reuse the structural year rule from the previous date audit."""
    match = DATE_RE.fullmatch(value)
    if match is None:
        return None, "unsupported_literal"
    lex = LEX_RE.fullmatch(match.group("lex"))
    if lex is None:
        return None, "unsupported_lexical_form"
    for field, low, high in (("month", 1, 12), ("day", 1, 31),
                             ("hour", 0, 23), ("minute", 0, 59), ("second", 0, 59)):
        part = lex.group(field)
        if part is not None and not low <= int(part) <= high:
            return None, f"{field}_out_of_range"
    return int(lex.group("year")), "parsed"


def rows(path: Path, header: bytes, fields: int, allow_empty: tuple[int, ...] = ()): 
    with path.open("rb") as handle:
        if handle.readline() != header:
            raise SystemExit(f"Unexpected header: {path}")
        for line, raw in enumerate(handle, 2):
            values = raw.rstrip(b"\r\n").split(b"\t", fields - 1)
            if len(values) != fields or any(not value for i, value in enumerate(values)
                                                  if i not in allow_empty):
                raise SystemExit(f"Malformed TSV row {line}: {path}")
            yield values


def read_names(path: Path) -> set[bytes]:
    with path.open("rb") as handle:
        return {line.rstrip(b"\r\n") for line in handle if line.strip()}


def require_same_facts(base: Path, facts: Path) -> None:
    marker = base / DIRS["people"] / "summary.json"
    if not marker.is_file():
        raise SystemExit(f"Missing completed people stage: {marker}")
    prior = json.loads(marker.read_text(encoding="utf-8"))
    if (prior.get("status") != "complete_scan" or prior.get("facts") != str(facts)
            or prior.get("facts_bytes") != facts.stat().st_size
            or prior.get("facts_mtime_ns") != facts.stat().st_mtime_ns):
        raise SystemExit("facts.txt changed since the people stage; use a new output directory")


def directory(base: Path, name: str) -> Path:
    path = base / DIRS[name]
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Stage output exists: {path}. Completed stages are kept; use a new output directory to rerun.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"Missing input file: {resolved}")
    return resolved


def write_summary(path: Path, data: dict) -> None:
    (path / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": data["stage"], **{key: data[key] for key in
                      ("status", "lines_scanned", "cohort_people", "all_person_pairs",
                       "review_facts", "main_facts", "final_facts") if key in data}},
                     ensure_ascii=False), flush=True)


def progress(args: argparse.Namespace, stage: str, lines: int, count: int) -> None:
    if lines % args.progress_every == 0:
        print(f"[{stage}] scanned {lines:,} lines; matched {count:,}", file=sys.stderr, flush=True)


def stage_people(args: argparse.Namespace, base: Path) -> None:
    facts = source(args.facts)
    output = directory(base, "people")
    started = time.monotonic()
    unsorted = output / "person_names.unsorted.partial"
    lines = malformed = person_facts = 0
    with facts.open("rb") as inp, unsorted.open("wb", buffering=1024 * 1024) as out:
        for lines, raw in enumerate(inp, 1):
            triple = fact(raw)
            if triple is None:
                malformed += 1
            elif triple[1] == b"is-a" and triple[2] == b"Person":
                out.write(triple[0] + b"\n")
                person_facts += 1
            progress(args, "people", lines, person_facts)
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    names = output / "person_names.txt"
    subprocess.run(["sort", "-u", "-T", str(output), "-o", str(names), str(unsorted)],
                   env=env, check=True)
    unsorted.unlink()
    unique = sum(1 for _ in names.open("rb"))
    write_summary(output, {"stage": "people", "status": "complete_scan", "facts": str(facts),
                           "facts_bytes": facts.stat().st_size, "facts_mtime_ns": facts.stat().st_mtime_ns,
                           "lines_scanned": lines,
                           "malformed_rows": malformed, "person_type_facts": person_facts,
                           "person_names": unique, "elapsed_seconds": round(time.monotonic()-started, 2)})


def stage_cohort(args: argparse.Namespace, base: Path) -> None:
    facts = source(args.facts)
    require_same_facts(base, facts)
    people_path = base / DIRS["people"] / "person_names.txt"
    persons = read_names(source(people_path))
    output = directory(base, "cohort")
    started = time.monotonic()
    raw_attrs = output / "person_attribute_facts.tsv"
    sortable = output / "by_name.unsorted.partial"
    lines = malformed = attributes = 0
    with facts.open("rb") as inp, raw_attrs.open("wb", buffering=1024*1024) as attrs, \
            sortable.open("wb", buffering=1024*1024) as by_name:
        attrs.write(ATTR_HEADER)
        for lines, raw in enumerate(inp, 1):
            triple = fact(raw)
            if triple is None:
                malformed += 1
                continue
            name, predicate, value = triple
            if name in persons and (predicate == b"Profession" or predicate in BIRTH or predicate in DEATH):
                line = str(lines).encode("ascii")
                attrs.write(b"\t".join((line, name, predicate, value)) + b"\n")
                by_name.write(b"\t".join((name, predicate, value, line)) + b"\n")
                attributes += 1
            progress(args, "cohort", lines, attributes)
    sorted_path = output / "by_name.sorted.partial"
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(["sort", "-t", "\t", "-k1,1", "-T", str(output), "-o",
                    str(sorted_path), str(sortable)], env=env, check=True)
    sortable.unlink()

    counts: Counter[str] = Counter()
    date_issues: Counter[str] = Counter()
    cohort_path = output / "cohort_people.tsv"
    names_path = output / "cohort_person_names.txt"
    issues_path = output / "date_parse_issues.tsv"
    with sorted_path.open("rb") as inp, cohort_path.open("wb", buffering=1024*1024) as cohort, \
            names_path.open("wb", buffering=1024*1024) as names, \
            issues_path.open("wb", buffering=1024*1024) as issues:
        cohort.write(COHORT_HEADER)
        issues.write(b"source_line\tperson_name\tpredicate\traw_value\treason\n")
        def records():
            for raw in inp:
                parts = raw.rstrip(b"\r\n").split(b"\t", 3)
                if len(parts) != 4:
                    raise SystemExit("Malformed intermediate attribute row")
                yield parts
        for name, group in groupby(records(), key=lambda item: item[0]):
            occupations: set[bytes] = set()
            years = {"birth": set(), "death": set()}
            literals = {"birth": set(), "death": set()}
            for _, predicate, value, line in group:
                if predicate == b"Profession":
                    occupations.add(value)
                else:
                    kind = "birth" if predicate in BIRTH else "death"
                    literals[kind].add(value)
                    year, reason = parse_year(value)
                    if year is None:
                        date_issues[reason] += 1
                        issues.write(b"\t".join((line, name, predicate, value, reason.encode())) + b"\n")
                    else:
                        years[kind].add(year)
            if occupations:
                counts["person_names_with_profession"] += 1
            if years["birth"] or years["death"]:
                counts["person_names_with_parsed_date"] += 1
            if not occupations or not (years["birth"] or years["death"]):
                continue
            counts["cohort_people"] += 1
            counts["multi_profession_people"] += len(occupations) > 1
            counts["birth_only_people"] += bool(years["birth"] and not years["death"])
            counts["death_only_people"] += bool(years["death"] and not years["birth"])
            counts["birth_and_death_people"] += bool(years["birth"] and years["death"])
            counts["conflicting_birth_year_people"] += len(years["birth"]) > 1
            counts["conflicting_death_year_people"] += len(years["death"]) > 1
            fields = (
                name,
                encode_json([show(v) for v in sorted(occupations)]),
                encode_json(sorted(years["birth"])),
                encode_json(sorted(years["death"])),
                encode_json([show(v) for v in sorted(literals["birth"])]),
                encode_json([show(v) for v in sorted(literals["death"])]),
            )
            cohort.write(b"\t".join(fields) + b"\n")
            names.write(name + b"\n")
    sorted_path.unlink()
    write_summary(output, {"stage": "cohort", "status": "complete_scan", "facts": str(facts),
                           "facts_bytes": facts.stat().st_size, "facts_mtime_ns": facts.stat().st_mtime_ns,
                           "lines_scanned": lines, "malformed_rows": malformed,
                           "person_names_loaded": len(persons), "attribute_facts": attributes,
                           **counts, "date_parse_issues": dict(date_issues),
                           "elapsed_seconds": round(time.monotonic()-started, 2),
                           "rule": "is-a Person name; >=1 distinct Profession; >=1 structurally parsed birth or death year"})


def stage_pairs(args: argparse.Namespace, base: Path) -> None:
    facts = source(args.facts)
    require_same_facts(base, facts)
    people = read_names(source(base / DIRS["people"] / "person_names.txt"))
    cohort = read_names(source(base / DIRS["cohort"] / "cohort_person_names.txt"))
    if not cohort <= people:
        raise SystemExit("Cohort includes a name not present in Person candidates")
    output = directory(base, "pairs")
    started = time.monotonic()
    all_counts: Counter[bytes] = Counter()
    cohort_counts: Counter[bytes] = Counter()
    lines = malformed = all_pairs = cohort_pairs = 0
    with facts.open("rb") as inp, (output / "person_name_pairs.tsv").open("wb", buffering=1024*1024) as all_out, \
            (output / "cohort_name_pairs.tsv").open("wb", buffering=1024*1024) as cohort_out:
        all_out.write(PAIR_HEADER)
        cohort_out.write(PAIR_HEADER)
        for lines, raw in enumerate(inp, 1):
            triple = fact(raw)
            if triple is None:
                malformed += 1
                continue
            subject, predicate, obj = triple
            if subject in people and obj in people:
                record = b"\t".join((str(lines).encode("ascii"), subject, predicate, obj)) + b"\n"
                all_out.write(record)
                all_counts[predicate] += 1
                all_pairs += 1
                if subject in cohort and obj in cohort:
                    cohort_out.write(record)
                    cohort_counts[predicate] += 1
                    cohort_pairs += 1
            progress(args, "pairs", lines, cohort_pairs)
    with (output / "predicate_coverage.tsv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp, delimiter="\t")
        writer.writerow(("predicate", "all_person_name_pair_facts", "cohort_name_pair_facts"))
        for predicate in sorted(all_counts, key=lambda v: (-cohort_counts[v], -all_counts[v], v)):
            writer.writerow((show(predicate), all_counts[predicate], cohort_counts[predicate]))
    write_summary(output, {"stage": "pairs", "status": "complete_scan", "facts": str(facts),
                           "facts_bytes": facts.stat().st_size, "facts_mtime_ns": facts.stat().st_mtime_ns,
                           "lines_scanned": lines, "malformed_rows": malformed,
                           "all_person_pairs": all_pairs, "cohort_person_pairs": cohort_pairs,
                           "all_pair_predicates": len(all_counts), "cohort_pair_predicates": len(cohort_counts),
                           "elapsed_seconds": round(time.monotonic()-started, 2),
                           "note": "All predicates, self-links and reverse duplicate facts are preserved; matching names alone do not prove two human entities."})


def relation_rules(path: Path) -> dict[bytes, dict]:
    config = json.loads(source(path).read_text(encoding="utf-8"))
    result = {}
    for row in config["relations"]:
        key = row["predicate"].encode("utf-8")
        if key in result or row["tier"] not in ("priority", "semantic_review"):
            raise SystemExit("Invalid or duplicate relation rule")
        result[key] = row
    return result


def stage_relations(args: argparse.Namespace, base: Path) -> None:
    rules = relation_rules(args.rules)
    path = source(base / DIRS["pairs"] / "cohort_name_pairs.tsv")
    output = directory(base, "relations")
    started = time.monotonic()
    review_counts: Counter[bytes] = Counter()
    main_counts: Counter[bytes] = Counter()
    lines = 0
    with (output / "review_candidate_facts.tsv").open("wb", buffering=1024*1024) as reviewed, \
            (output / "main_relation_facts.tsv").open("wb", buffering=1024*1024) as main:
        reviewed.write(b"source_line\tsubject_name\tpredicate\tobject_name\treview_tier\n")
        main.write(PAIR_HEADER)
        for source_line, subject, predicate, obj in rows(path, PAIR_HEADER, 4):
            lines += 1
            rule = rules.get(predicate)
            if rule is None:
                continue
            reviewed.write(b"\t".join((source_line, subject, predicate, obj,
                                         rule["tier"].encode("ascii"))) + b"\n")
            review_counts[predicate] += 1
            if rule.get("main"):
                main.write(b"\t".join((source_line, subject, predicate, obj)) + b"\n")
                main_counts[predicate] += 1
    with (output / "review_predicate_counts.tsv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp, delimiter="\t")
        writer.writerow(("predicate", "tier", "review_facts", "main_facts"))
        for predicate, count in sorted(review_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow((show(predicate), rules[predicate]["tier"], count, main_counts[predicate]))
    write_summary(output, {"stage": "relations", "status": "complete_subset_scan",
                           "cohort_pairs_read": lines, "configured_review_predicates": len(rules),
                           "review_facts": sum(review_counts.values()),
                           "configured_main_predicates": sum(bool(v.get("main")) for v in rules.values()),
                           "main_facts": sum(main_counts.values()),
                           "review_predicates_with_rows": len(review_counts),
                           "main_predicates_with_rows": len(main_counts),
                           "rules": str(source(args.rules)), "rules_bytes": source(args.rules).stat().st_size,
                           "rules_mtime_ns": source(args.rules).stat().st_mtime_ns,
                           "elapsed_seconds": round(time.monotonic()-started, 2),
                           "note": "These are raw triples. No ID, self-loop, direction, or reciprocity filtering was applied."})


def link_name(value: bytes) -> bytes | None:
    if not value:
        return None
    if not value.startswith(b":d:"):
        return value
    end = value.find(b":", 3)
    return value[end+1:] if end > 3 and value[3:end].isdigit() and end+1 < len(value) else None


def link_id(value: bytes) -> bytes | None:
    if value.startswith(URI_PREFIX) and value.endswith(b">"):
        identifier = value[len(URI_PREFIX):-1]
        if identifier.startswith((b"m.", b"g/")):
            return identifier
    return None


def load_profiles(path: Path, wanted: set[bytes]) -> dict[bytes, tuple[str, list, list, list]]:
    result = {}
    for name, professions, births, deaths, _, _ in rows(path, COHORT_HEADER, 6):
        if name in wanted:
            result[name] = (show(name), json.loads(professions), json.loads(births), json.loads(deaths))
    if len(result) != len(wanted):
        raise SystemExit(f"Missing {len(wanted)-len(result)} relation endpoint profiles")
    return result


def stage_final(args: argparse.Namespace, base: Path) -> None:
    links = source(args.links)
    main_path = source(base / DIRS["relations"] / "main_relation_facts.tsv")
    cohort_path = source(base / DIRS["cohort"] / "cohort_people.tsv")
    rules = relation_rules(args.rules)
    output = directory(base, "final")
    started = time.monotonic()
    endpoints = set()
    raw_count = 0
    for _, subject, _, obj in rows(main_path, PAIR_HEADER, 4):
        raw_count += 1
        endpoints.update((subject, obj))
    profiles = load_profiles(cohort_path, endpoints)
    first_id: dict[bytes, bytes] = {}
    ambiguous: set[bytes] = set()
    links_read = malformed_links = recognized_links = 0
    with links.open("rb") as inp:
        for links_read, raw in enumerate(inp, 1):
            progress(args, "links", links_read, recognized_links)
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or fields[1] != b"freebase-entity" or fields[3].strip() != b".":
                malformed_links += 1
                continue
            name = link_name(fields[0])
            if name not in endpoints:
                continue
            identifier = link_id(fields[2])
            if identifier is None:
                continue
            recognized_links += 1
            previous = first_id.setdefault(name, identifier)
            if previous != identifier:
                ambiguous.add(name)
    unique_ids = {name: identifier for name, identifier in first_id.items() if name not in ambiguous}
    with (output / "name_id_status.tsv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp, delimiter="\t")
        writer.writerow(("name", "id_status", "freebase_id"))
        for name in sorted(endpoints):
            status = "ambiguous" if name in ambiguous else ("unique" if name in unique_ids else "missing")
            writer.writerow((show(name), status, show(unique_ids[name]) if name in unique_ids else ""))
    node_columns = ("Name", "FreebaseID", "IDStatus", "Professions", "BirthYears", "DeathYears")
    with (output / "nodes.csv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp)
        writer.writerow(node_columns)
        for name in sorted(endpoints):
            label, occupations, births, deaths = profiles[name]
            status = "ambiguous" if name in ambiguous else ("unique" if name in unique_ids else "missing")
            writer.writerow((label, show(unique_ids[name]) if name in unique_ids else "", status,
                             json.dumps(occupations, ensure_ascii=False), json.dumps(births), json.dumps(deaths)))
    fact_columns = ("SourceLine", "Node1_Name", "Node1_ID", "Node1_IDStatus", "RawPredicate",
                    "Relation", "RelationGroup", "Node2_Name", "Node2_ID", "Node2_IDStatus",
                    "Node1_Professions", "Node1_BirthYears", "Node1_DeathYears",
                    "Node2_Professions", "Node2_BirthYears", "Node2_DeathYears")
    written = 0
    with (output / "main_relation_facts.csv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp)
        writer.writerow(fact_columns)
        for source_line, subject, predicate, obj in rows(main_path, PAIR_HEADER, 4):
            rule = rules[predicate]
            left, left_prof, left_birth, left_death = profiles[subject]
            right, right_prof, right_birth, right_death = profiles[obj]
            left_status = "ambiguous" if subject in ambiguous else ("unique" if subject in unique_ids else "missing")
            right_status = "ambiguous" if obj in ambiguous else ("unique" if obj in unique_ids else "missing")
            writer.writerow((int(source_line), left, show(unique_ids[subject]) if subject in unique_ids else "",
                             left_status, show(predicate), rule["main"], rule["group"], right,
                             show(unique_ids[obj]) if obj in unique_ids else "", right_status,
                             json.dumps(left_prof, ensure_ascii=False), json.dumps(left_birth), json.dumps(left_death),
                             json.dumps(right_prof, ensure_ascii=False), json.dumps(right_birth), json.dumps(right_death)))
            written += 1
    if written != raw_count:
        raise SystemExit("Final fact count differs from raw main relation count")
    write_summary(output, {"stage": "final", "status": "complete_link_scan",
                           "links": str(links), "links_bytes": links.stat().st_size,
                           "links_mtime_ns": links.stat().st_mtime_ns,
                           "rules": str(source(args.rules)), "rules_bytes": source(args.rules).stat().st_size,
                           "rules_mtime_ns": source(args.rules).stat().st_mtime_ns,
                           "link_rows_read": links_read,
                           "malformed_link_rows": malformed_links, "recognized_link_rows_for_endpoints": recognized_links,
                           "relation_endpoint_names": len(endpoints),
                           "names_with_unique_id": len(unique_ids), "names_with_ambiguous_id": len(ambiguous),
                           "names_without_id": len(endpoints)-len(first_id),
                           "final_facts": written, "elapsed_seconds": round(time.monotonic()-started, 2),
                           "note": "All raw main relation facts and all multi-profession values remain; IDs are provisional and missing/ambiguous IDs do not remove facts."})


def stage_stats(args: argparse.Namespace, base: Path) -> None:
    cohort_path = source(base / DIRS["cohort"] / "cohort_people.tsv")
    pair_path = source(base / DIRS["pairs"] / "cohort_name_pairs.tsv")
    review_path = source(base / DIRS["relations"] / "review_candidate_facts.tsv")
    main_path = source(base / DIRS["relations"] / "main_relation_facts.tsv")
    output = directory(base, "stats")
    started = time.monotonic()
    cohort = multi = birth_only = death_only = both_dates = 0
    occ_people: Counter[str] = Counter()
    occ_multiplicity: Counter[int] = Counter()
    for _, occupations, births, deaths, _, _ in rows(cohort_path, COHORT_HEADER, 6):
        cohort += 1
        values = json.loads(occupations)
        has_birth, has_death = bool(json.loads(births)), bool(json.loads(deaths))
        occ_multiplicity[len(values)] += 1
        multi += len(values) > 1
        birth_only += has_birth and not has_death
        death_only += has_death and not has_birth
        both_dates += has_birth and has_death
        occ_people.update(values)
    all_pairs = Counter()
    for _, _, predicate, _ in rows(pair_path, PAIR_HEADER, 4):
        all_pairs[show(predicate)] += 1
    reviewed = Counter()
    review_header = b"source_line\tsubject_name\tpredicate\tobject_name\treview_tier\n"
    for _, _, predicate, _, tier in rows(review_path, review_header, 5):
        reviewed[(show(predicate), show(tier))] += 1
    main = Counter()
    directed: dict[bytes, set[tuple[bytes, bytes]]] = defaultdict(set)
    participants: dict[bytes, set[bytes]] = defaultdict(set)
    self_loops = Counter()
    all_nodes = set()
    parents: dict[bytes, bytes] = {}

    def find(node: bytes) -> bytes:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: bytes, right: bytes) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a

    for _, subject, predicate, obj in rows(main_path, PAIR_HEADER, 4):
        main[show(predicate)] += 1
        directed[predicate].add((subject, obj))
        participants[predicate].update((subject, obj))
        all_nodes.update((subject, obj))
        union(subject, obj)
        if subject == obj:
            self_loops[predicate] += 1
    with (output / "occupation_people_counts.tsv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp, delimiter="\t")
        writer.writerow(("raw_profession", "cohort_people"))
        for value, count in sorted(occ_people.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow((value, count))
    with (output / "relation_counts.tsv").open("w", encoding="utf-8", newline="") as inp:
        writer = csv.writer(inp, delimiter="\t")
        writer.writerow(("predicate", "cohort_pair_facts", "review_facts", "main_facts",
                         "unique_directed_name_pairs", "unique_unordered_name_pairs",
                         "reciprocal_unordered_name_pairs", "self_facts", "participants"))
        for predicate, count in sorted(all_pairs.items(), key=lambda kv: (-kv[1], kv[0])):
            pairs = directed[predicate.encode()]
            unordered = {tuple(sorted(pair)) for pair in pairs}
            reciprocals = sum((a != b and (b, a) in pairs) for a, b in pairs) // 2
            writer.writerow((predicate, count, sum(v for (p, _), v in reviewed.items() if p == predicate),
                             main[predicate], len(pairs), len(unordered), reciprocals,
                             self_loops[predicate.encode()], len(participants[predicate.encode()])))
    component_sizes = Counter(find(node) for node in all_nodes)
    summary = {"stage": "stats", "status": "complete_descriptive_scan",
               "cohort_people": cohort, "multi_profession_people": multi,
               "birth_only_people": birth_only, "death_only_people": death_only,
               "birth_and_death_people": both_dates,
               "cohort_pair_facts": sum(all_pairs.values()),
               "review_facts": sum(reviewed.values()), "main_facts": sum(main.values()),
               "main_relation_participants": len(all_nodes),
               "main_graph_components_by_name": len(component_sizes),
               "main_graph_largest_component_by_name": max(component_sizes.values(), default=0),
               "cohort_predicates": len(all_pairs), "review_predicates": len(reviewed),
               "main_predicates": len(main),
               "occupation_multiplicity_people": dict(sorted(occ_multiplicity.items())),
               "main_facts_by_predicate": dict(sorted(main.items())),
               "elapsed_seconds": round(time.monotonic()-started, 2),
               "notes": ["Occupation counts count people once per raw value; multi-profession people contribute to multiple values.",
                         "Relation facts and pairs are based on name matching, not validated stable entity identity.",
                         "All raw directions are retained; reciprocal counts are descriptive, not automatic deduplication."]}
    write_summary(output, summary)


def main() -> None:
    args = arguments()
    base = args.output_dir.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    actions = {"people": stage_people, "cohort": stage_cohort, "pairs": stage_pairs,
               "relations": stage_relations, "final": stage_final, "stats": stage_stats}
    for name in STAGES if args.stage == "all" else (args.stage,):
        if args.resume:
            stage_dir = base / DIRS[name]
            marker = stage_dir / "summary.json"
            if marker.is_file():
                summary = json.loads(marker.read_text(encoding="utf-8"))
                inputs_current = True
                for label, input_path in (("facts", args.facts), ("links", args.links), ("rules", args.rules)):
                    if label not in summary:
                        continue
                    current = source(input_path)
                    inputs_current &= (summary[label] == str(current)
                                       and summary.get(f"{label}_bytes") == current.stat().st_size
                                       and summary.get(f"{label}_mtime_ns") == current.stat().st_mtime_ns)
                if summary.get("stage") == name and summary.get("status", "").startswith("complete") \
                        and inputs_current \
                        and all((stage_dir / filename).is_file() for filename in OUTPUT_FILES[name]):
                    print(f"Skipping completed stage {name}", file=sys.stderr, flush=True)
                    continue
        print(f"Starting {name}", file=sys.stderr, flush=True)
        actions[name](args, base)


if __name__ == "__main__":
    main()
