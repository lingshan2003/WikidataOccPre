#!/usr/bin/env python3
"""Audit birth/death year parsing and relation-participant coverage.

Reads only exported person_attributes.tsv and provisional relation candidates.
No GPU, no facts.txt scan, and no automatic person or edge exclusion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import re
import sys
import time


ATTRIBUTE_HEADER = b"source_line\tperson_name\tpredicate\traw_value"
PAIR_HEADER = b"source_line\taudit_tier\tsubject_name\tpredicate\tobject_name"
BIRTH = {b"Date of birth", b"Date of Birth"}
DEATH = {b"Date of death", b"Date of Death"}
DATE_PREDICATES = BIRTH | DEATH
TYPED_LITERAL = re.compile(
    rb'^"(?P<lex>[^"]+)"\^\^<https?://www\.w3\.org/2001/XMLSchema#'
    rb'(?P<dtype>date|gYear|gYearMonth|dateTime)>$'
)
LEXICAL_DATE = re.compile(
    rb'^(?P<year>-?\d{4,6})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?'
    rb'(?:T(?P<hour>\d{2})(?::(?P<minute>\d{2}))?'
    rb'(?::(?P<second>\d{2}))?(?:Z|[+-]\d{2}:\d{2})?)?$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--relation-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def parse_year(value: bytes) -> tuple[int | None, bytes, bool]:
    literal = TYPED_LITERAL.fullmatch(value)
    if literal is None:
        return None, b"unsupported_literal", False
    lexical = LEXICAL_DATE.fullmatch(literal.group("lex"))
    if lexical is None:
        return None, b"unsupported_lexical_form", False
    year = int(lexical.group("year"))
    month = lexical.group("month")
    day = lexical.group("day")
    hour = lexical.group("hour")
    minute = lexical.group("minute")
    second = lexical.group("second")
    if month is not None and not 1 <= int(month) <= 12:
        return None, b"month_out_of_range", False
    if day is not None and not 1 <= int(day) <= 31:
        return None, b"day_out_of_range", False
    if hour is not None and int(hour) > 23:
        return None, b"hour_out_of_range", False
    if minute is not None and int(minute) > 59:
        return None, b"minute_out_of_range", False
    if second is not None and int(second) > 59:
        return None, b"second_out_of_range", False
    dtype = literal.group("dtype")
    canonical = (
        (dtype == b"gYear" and month is None and hour is None)
        or (dtype == b"gYearMonth" and month is not None and day is None and hour is None)
        or (dtype == b"date" and day is not None and hour is None)
        or (dtype == b"dateTime" and day is not None and hour is not None)
    )
    return year, dtype, canonical


def load_groups(path: Path) -> dict[str, set[bytes]]:
    groups = {"priority": set(), "semantic_review": set()}
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected relation candidate table header")
        for raw in handle:
            fields = raw.rstrip(b"\r\n").split(b"\t", 4)
            if len(fields) != 5 or not all(fields) or fields[1] not in (b"priority", b"semantic_review"):
                raise SystemExit("Malformed relation candidate row")
            _, tier, subject, _, obj = fields
            group = groups[show(tier)]
            group.add(subject)
            group.add(obj)
    groups["union"] = groups["priority"] | groups["semantic_review"]
    return groups


def main() -> None:
    args = parse_args()
    attributes_path = args.attributes.expanduser().resolve()
    pairs_path = args.relation_candidates.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (attributes_path, pairs_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    groups = load_groups(pairs_path)
    print("loaded relation participants: " + ", ".join(
        f"{name}={len(members):,}" for name, members in groups.items()
    ), file=sys.stderr, flush=True)
    participants = groups["union"]
    profession_people: set[bytes] = set()
    raw_birth_people: set[bytes] = set()
    raw_death_people: set[bytes] = set()
    parsed_years: dict[bytes, dict[bytes, set[int]]] = {
        b"birth": defaultdict(set), b"death": defaultdict(set)
    }
    date_counts: dict[bytes, Counter[str]] = defaultdict(Counter)
    parse_error_examples: dict[tuple[bytes, bytes], list[bytes]] = defaultdict(list)
    rows = malformed = 0
    audit_year = date.today().year
    output.mkdir(parents=True, exist_ok=True)
    parsed_partial = output / "parsed_year_facts.tsv.partial"
    parsed_final = output / "parsed_year_facts.tsv"
    with attributes_path.open("rb") as attributes, parsed_partial.open("wb", buffering=1024 * 1024) as parsed_out:
        if attributes.readline().rstrip(b"\r\n") != ATTRIBUTE_HEADER:
            raise SystemExit("Unexpected person attribute table header")
        parsed_out.write(b"source_line\tperson_name\tpredicate\tyear\tdate_datatype\tcanonical_shape\n")
        for raw in attributes:
            rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            source_line, person, predicate, value = fields
            in_group = person in participants
            if predicate == b"Profession":
                if in_group:
                    profession_people.add(person)
            elif predicate in DATE_PREDICATES:
                item = date_counts[predicate]
                item["facts"] += 1
                is_birth = predicate in BIRTH
                if in_group:
                    (raw_birth_people if is_birth else raw_death_people).add(person)
                year, dtype_or_reason, canonical = parse_year(value)
                if year is None:
                    item["unparsed"] += 1
                    examples = parse_error_examples[(predicate, dtype_or_reason)]
                    if len(examples) < 20:
                        examples.append(raw.rstrip(b"\r\n"))
                else:
                    item["parsed_year"] += 1
                    if not canonical:
                        item["noncanonical_shape"] += 1
                    if year <= 0:
                        item["nonpositive_year"] += 1
                    if is_birth and year > audit_year:
                        item["future_birth_year"] += 1
                    parsed_out.write(source_line + b"\t" + person + b"\t" + predicate
                                     + b"\t" + str(year).encode("ascii") + b"\t"
                                     + dtype_or_reason + b"\t" + (b"yes" if canonical else b"no") + b"\n")
                    if in_group:
                        parsed_years[b"birth" if is_birth else b"death"][person].add(year)
            if rows % args.progress_every == 0:
                print(f"read {rows:,} attribute rows", file=sys.stderr, flush=True)
    parsed_partial.replace(parsed_final)

    good_birth = {person for person, years in parsed_years[b"birth"].items() if len(years) == 1}
    good_death = {person for person, years in parsed_years[b"death"].items() if len(years) == 1}
    conflict_birth = set(parsed_years[b"birth"]) - good_birth
    conflict_death = set(parsed_years[b"death"]) - good_death
    good_date = good_birth | good_death
    with (output / "predicate_date_parse.tsv").open("w", encoding="utf-8") as handle:
        columns = ("facts", "parsed_year", "unparsed", "noncanonical_shape",
                   "nonpositive_year", "future_birth_year")
        handle.write("predicate\t" + "\t".join(columns) + "\n")
        for predicate in sorted(DATE_PREDICATES):
            item = date_counts[predicate]
            handle.write(show(predicate) + "\t" + "\t".join(str(item[key]) for key in columns) + "\n")
    with (output / "date_parse_error_examples.tsv").open("wb") as handle:
        handle.write(b"source_line\tperson_name\tpredicate\traw_value\treason\n")
        for (predicate, reason), examples in sorted(parse_error_examples.items()):
            for raw in examples:
                handle.write(raw + b"\t" + reason + b"\n")
    with (output / "relation_group_date_funnel.tsv").open("w", encoding="utf-8") as handle:
        handle.write("group\tparticipants\twith_profession\twith_raw_birth_or_death\t"
                     "with_unique_parsed_birth_or_death\twith_profession_and_unique_parsed_date\t"
                     "with_conflicting_birth_years\twith_conflicting_death_years\n")
        for group_name, members in groups.items():
            raw_date = raw_birth_people | raw_death_people
            handle.write(f"{group_name}\t{len(members)}\t{len(members & profession_people)}\t"
                         f"{len(members & raw_date)}\t{len(members & good_date)}\t"
                         f"{len(members & profession_people & good_date)}\t"
                         f"{len(members & conflict_birth)}\t{len(members & conflict_death)}\n")
    anomaly_counts: Counter[str] = Counter()
    anomaly_examples: list[dict[str, object]] = []
    for person in sorted(good_birth & good_death):
        birth = next(iter(parsed_years[b"birth"][person]))
        death = next(iter(parsed_years[b"death"][person]))
        gap = death - birth
        reason = "death_before_birth" if gap < 0 else ("age_over_125" if gap > 125 else "")
        if reason:
            anomaly_counts[reason] += 1
            if len(anomaly_examples) < 50:
                anomaly_examples.append({"person_name": show(person), "birth_year": birth,
                                         "death_year": death, "reason": reason})
    (output / "temporal_anomaly_examples.json").write_text(
        json.dumps(anomaly_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete_subset_audit" if malformed == 0 else "subset_audit_with_malformed_rows",
        "attributes": str(attributes_path),
        "relation_candidates": str(pairs_path),
        "attribute_rows_read": rows,
        "malformed_rows": malformed,
        "relation_participants_union": len(participants),
        "relation_participants_with_conflicting_birth_years": len(conflict_birth),
        "relation_participants_with_conflicting_death_years": len(conflict_death),
        "relation_participants_with_death_before_birth": anomaly_counts["death_before_birth"],
        "relation_participants_with_age_over_125": anomaly_counts["age_over_125"],
        "audit_year": audit_year,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Parsed year means a typed literal with a structurally readable year; it is not a verified historical date.",
            "A noncanonical datatype/shape combination is counted and retained for review.",
            "Profession here means raw fact presence, not a validated or mapped occupation label.",
            "No people or relation facts are removed by this audit.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
