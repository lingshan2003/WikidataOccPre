#!/usr/bin/env python3
"""Extract raw occupation, life-date and type facts for provisional Person names.

One CPU-only facts.txt pass retains the original predicate, value and source
line. It does not require a relation, occupation, parseable date, or MID and
does not create labels or a training graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time


PREDICATES = (
    b"Profession", b"Occupation", b"Date of birth", b"Date of Birth",
    b"Date of death", b"Date of Death", b"Gender",
    b"Country of nationality", b"Place of birth", b"Place of death",
    b"is-a", b"kg/object_profile/prominent_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--persons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-coverage", type=Path,
                        help="Optional stage2 predicate_coverage.tsv for full-scan validation")
    parser.add_argument("--max-lines", type=int, help="Optional short probe; omit for full export")
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


def mark(bits: bytearray, index: int) -> None:
    bits[index >> 3] |= 1 << (index & 7)


def bit_count(bits: bytearray) -> int:
    return sum(value.bit_count() for value in bits)


def load_expected(path: Path) -> dict[str, tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["predicate"]: (int(row["person_subject_facts"]),
                               int(row["unique_person_subjects"]))
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> None:
    args = parse_args()
    facts_path = args.facts.expanduser().resolve()
    persons_path = args.persons.expanduser().resolve()
    expected_path = args.expected_coverage.expanduser().resolve() if args.expected_coverage else None
    output = args.output_dir.expanduser().resolve()
    for source in (facts_path, persons_path, expected_path):
        if source is not None and not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    person_ids: dict[bytes, int] = {}
    with persons_path.open("rb") as persons:
        for raw in persons:
            name = raw.rstrip(b"\r\n")
            if name and name not in person_ids:
                person_ids[name] = len(person_ids)
    print(f"loaded {len(person_ids):,} provisional Person names", file=sys.stderr, flush=True)
    bits_size = (len(person_ids) + 7) // 8
    subject_bits = {predicate: bytearray(bits_size) for predicate in PREDICATES}
    counts = dict.fromkeys(PREDICATES, 0)
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "person_attributes.tsv.partial"
    final_path = output / "person_attributes.tsv"
    lines = bytes_read = malformed = exported = 0
    with facts_path.open("rb") as facts, partial_path.open("wb", buffering=1024 * 1024) as out:
        out.write(b"source_line\tperson_name\tpredicate\traw_value\n")
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
            if predicate not in subject_bits:
                if lines % args.progress_every == 0:
                    print(f"scanned {lines:,} facts; exported {exported:,} person attributes",
                          file=sys.stderr, flush=True)
                continue
            name, value = token(fields[0]), fields[2].strip()
            if not name or not value:
                malformed += 1
                continue
            person_id = person_ids.get(name)
            if person_id is not None:
                out.write(str(lines).encode("ascii") + b"\t" + name + b"\t"
                          + predicate + b"\t" + value + b"\n")
                counts[predicate] += 1
                mark(subject_bits[predicate], person_id)
                exported += 1
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} facts; exported {exported:,} person attributes",
                      file=sys.stderr, flush=True)
    partial_path.replace(final_path)

    coverage = {
        predicate.decode("utf-8"): {
            "person_subject_facts": counts[predicate],
            "unique_person_subjects": bit_count(subject_bits[predicate]),
        }
        for predicate in PREDICATES
    }
    with (output / "predicate_attribute_coverage.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tperson_subject_facts\tunique_person_subjects\n")
        for name, item in coverage.items():
            handle.write(f"{name}\t{item['person_subject_facts']}\t{item['unique_person_subjects']}\n")
    complete = bytes_read == facts_path.stat().st_size
    mismatches: list[dict[str, object]] = []
    if complete and expected_path is not None:
        expected = load_expected(expected_path)
        for name, observed in coverage.items():
            if name in expected and expected[name] != (
                observed["person_subject_facts"], observed["unique_person_subjects"]
            ):
                mismatches.append({"predicate": name, "expected": expected[name],
                                   "exported": observed})
    summary = {
        "status": ("complete_scan" if complete else "sample_only") if not mismatches else "complete_scan_with_count_mismatches",
        "facts": str(facts_path),
        "persons": str(persons_path),
        "expected_coverage": str(expected_path) if expected_path else None,
        "person_candidate_names_loaded": len(person_ids),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "person_attribute_facts_exported": exported,
        "expected_coverage_mismatches": mismatches,
        "output_file": str(final_path),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Raw values and predicate spelling are preserved; date and occupation quality are not yet validated.",
            "No relation, profession, date, or MID criterion was used to drop Person candidates.",
            "Gender, nationality, places and type facts are retained for later audit, not model features by default.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
