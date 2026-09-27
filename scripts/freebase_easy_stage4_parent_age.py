#!/usr/bin/env python3
"""Audit Freebase Easy Children edges using birth years and explicit character types.

Reads facts.txt once on CPU. It reports evidence and samples; it does not
automatically discard edges or produce a training graph.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
import time


EXPLICIT_CHARACTER_TYPES = {
    b"Fictional Character", b"TV Character", b"Film character",
    b"Book Character", b"Theatre Character", b"Music video character",
    b"Comic Book Character", b"Poem character", b"Opera Character",
    b"Video Game Character",
}
DATE_YEAR = re.compile(
    rb'^"(-?\d{4,6})(?:-\d{2}(?:-\d{2})?)?"\^\^'
    rb'<http://www\.w3\.org/2001/XMLSchema#(?:date|gYear|gYearMonth)>$'
)
SAMPLE_LIMIT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--persons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lines", type=int, help="Optional probe; omit for full scan")
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.max_lines is not None and args.max_lines < 1:
        parser.error("--max-lines must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def token(value: bytes) -> bytes:
    value = value.strip()
    if value.startswith(b"<") and value.endswith(b">"):
        return value[1:-1]
    return value


def string(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def mark(bits: bytearray, index: int) -> None:
    bits[index >> 3] |= 1 << (index & 7)


def marked(bits: bytearray, index: int) -> bool:
    return bool(bits[index >> 3] & (1 << (index & 7)))


def reservoir(
    rows: list[dict[str, object]], seen: int, row: dict[str, object], rng: random.Random
) -> None:
    position = seen - 1 if seen <= SAMPLE_LIMIT else rng.randrange(seen)
    if position >= SAMPLE_LIMIT:
        return
    if seen <= SAMPLE_LIMIT:
        rows.append(row)
    else:
        rows[position] = row


def main() -> None:
    args = parse_args()
    facts_path = args.facts.expanduser().resolve()
    persons_path = args.persons.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (facts_path, persons_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    person_ids: dict[bytes, int] = {}
    names: list[bytes] = []
    with persons_path.open("rb") as handle:
        for raw in handle:
            entity = raw.rstrip(b"\r\n")
            if entity and entity not in person_ids:
                person_ids[entity] = len(names)
                names.append(entity)
    print(f"loaded {len(names):,} Person candidates", file=sys.stderr, flush=True)
    character_bits = bytearray((len(names) + 7) // 8)
    character_type_facts: Counter[bytes] = Counter()
    birth_years: dict[int, int] = {}
    conflicting_birth_ids: set[int] = set()
    birth_unparsed_examples: list[str] = []
    child_edges: set[tuple[int, int]] = set()
    lines = bytes_read = malformed = birth_person_facts = birth_parsed_facts = 0
    child_facts = child_person_pair_facts = 0
    output.mkdir(parents=True, exist_ok=True)

    with facts_path.open("rb") as facts:
        for raw in facts:
            if args.max_lines is not None and lines >= args.max_lines:
                break
            lines += 1
            bytes_read += len(raw)
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) < 3:
                malformed += 1
                continue
            predicate = token(fields[1])
            if predicate not in (b"is-a", b"Date of birth", b"Children"):
                if lines % args.progress_every == 0:
                    print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)
                continue
            subject, obj = token(fields[0]), token(fields[2])
            if not subject or not obj:
                malformed += 1
                continue
            subject_id = person_ids.get(subject)
            if predicate == b"is-a":
                if subject_id is not None and obj in EXPLICIT_CHARACTER_TYPES:
                    mark(character_bits, subject_id)
                    character_type_facts[obj] += 1
            elif predicate == b"Date of birth":
                if subject_id is not None:
                    birth_person_facts += 1
                    match = DATE_YEAR.fullmatch(obj)
                    if match is None:
                        if len(birth_unparsed_examples) < SAMPLE_LIMIT:
                            birth_unparsed_examples.append(string(obj))
                    else:
                        birth_parsed_facts += 1
                        year = int(match.group(1))
                        previous = birth_years.get(subject_id)
                        if previous is not None and previous != year:
                            conflicting_birth_ids.add(subject_id)
                        else:
                            birth_years[subject_id] = year
            else:
                child_facts += 1
                object_id = person_ids.get(obj)
                if subject_id is not None and object_id is not None:
                    child_person_pair_facts += 1
                    child_edges.add((subject_id, object_id))
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)

    category_counts: Counter[str] = Counter()
    gap_counts: Counter[int] = Counter()
    samples: dict[str, list[dict[str, object]]] = {
        key: [] for key in (
            "nonpositive_gap", "gap_1_to_11", "gap_12_to_80", "gap_above_80",
            "missing_birth", "conflicting_birth", "explicit_character_type"
        )
    }
    rng = random.Random(20260927)
    self_edges = sum(parent == child for parent, child in child_edges)
    reciprocal_directed = sum(
        1 for parent, child in child_edges
        if parent != child and (child, parent) in child_edges
    )
    for parent, child in child_edges:
        if parent == child or (child, parent) in child_edges:
            continue
        parent_year = birth_years.get(parent)
        child_year = birth_years.get(child)
        row: dict[str, object] = {
            "parent": string(names[parent]),
            "child": string(names[child]),
            "parent_birth_year": parent_year,
            "child_birth_year": child_year,
        }
        if marked(character_bits, parent) or marked(character_bits, child):
            category_counts["explicit_character_type"] += 1
            seen = category_counts["explicit_character_type"]
            reservoir(samples["explicit_character_type"], seen, row, rng)
        if parent in conflicting_birth_ids or child in conflicting_birth_ids:
            category = "conflicting_birth"
        elif parent_year is None or child_year is None:
            category = "missing_birth"
        else:
            gap = child_year - parent_year
            gap_counts[gap] += 1
            row["age_gap_years"] = gap
            if gap <= 0:
                category = "nonpositive_gap"
            elif gap < 12:
                category = "gap_1_to_11"
            elif gap <= 80:
                category = "gap_12_to_80"
            else:
                category = "gap_above_80"
        category_counts[category] += 1
        reservoir(samples[category], category_counts[category], row, rng)

    with (output / "age_gap_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("child_minus_parent_birth_year\tedges\n")
        for gap, count in sorted(gap_counts.items()):
            handle.write(f"{gap}\t{count}\n")
    with (output / "explicit_character_types.tsv").open("w", encoding="utf-8") as handle:
        handle.write("type_object\tperson_subject_facts\n")
        for obj, count in character_type_facts.most_common():
            handle.write(f"{string(obj)}\t{count}\n")
    (output / "age_anomaly_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    structural_edges = len(child_edges) - self_edges - reciprocal_directed
    summary = {
        "status": "complete_scan" if bytes_read == facts_path.stat().st_size else "sample_only",
        "facts": str(facts_path),
        "persons": str(persons_path),
        "person_candidates_loaded": len(names),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "birth_person_facts": birth_person_facts,
        "birth_parsed_facts": birth_parsed_facts,
        "birth_unparsed_facts": birth_person_facts - birth_parsed_facts,
        "birth_unparsed_examples": birth_unparsed_examples,
        "people_with_parsed_birth_year": len(birth_years),
        "people_with_conflicting_birth_years": len(conflicting_birth_ids),
        "person_candidates_with_explicit_character_type": sum(
            byte.bit_count() for byte in character_bits
        ),
        "children_all_facts": child_facts,
        "children_person_pair_facts": child_person_pair_facts,
        "children_unique_directed_pairs": len(child_edges),
        "children_self_pairs": self_edges,
        "children_reciprocal_unordered_pairs": reciprocal_directed // 2,
        "children_structurally_unambiguous_directed_pairs": structural_edges,
        "structurally_unambiguous_pair_categories": dict(category_counts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "The subject of Children is provisionally interpreted as parent, object as child.",
            "Birth-year gaps are diagnostics, not automatic exclusions; missing years are reported separately.",
            "Explicit character types are a narrower audit flag, not proof that every marked entity is fictional.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
