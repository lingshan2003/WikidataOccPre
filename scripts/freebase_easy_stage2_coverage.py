#!/usr/bin/env python3
"""Audit selected Freebase Easy predicates against Stage 1 Person candidates.

Reads the extracted facts.txt once, uses CPU only, and writes small aggregate
reports and reproducible reservoir samples. No graph or training data is made.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time


PREDICATES = (
    "Profession", "Occupation", "Gender", "Date of birth", "Date of Birth",
    "Date of death", "Place of birth", "Country of nationality", "Children",
    "Sibling", "Spouse (or domestic partner)", "Married To", "Parent",
)
KINSHIP = ("Children", "Sibling", "Spouse (or domestic partner)", "Married To", "Parent")
OCCUPATION = ("Profession", "Occupation")


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path, help="Extracted facts.txt")
    parser.add_argument("--persons", required=True, type=Path, help="Stage 1 person_candidate_entities.txt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-lines", type=int, help="Optional format probe; omit for full scan")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.max_lines is not None and args.max_lines < 1:
        parser.error("--max-lines must be positive")
    if args.sample_size < 0 or args.progress_every < 1:
        parser.error("--sample-size must be nonnegative and --progress-every must be positive")
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


def cardinality(bits: bytearray) -> int:
    return sum(byte.bit_count() for byte in bits)


def as_int(bits: bytearray) -> int:
    return int.from_bytes(bits, byteorder="little")


def maybe_sample(
    samples: list[dict[str, object]], seen: int, limit: int, rng: random.Random,
    line_no: int, subject: bytes, predicate: bytes, obj: bytes,
) -> None:
    if not limit:
        return
    position = seen - 1 if seen <= limit else rng.randrange(seen)
    if position >= limit:
        return
    row = {
        "line": line_no,
        "subject": string(subject),
        "predicate": string(predicate),
        "object": string(obj),
    }
    if seen <= limit:
        samples.append(row)
    else:
        samples[position] = row


class Coverage:
    def __init__(self, people_count: int) -> None:
        bits_size = (people_count + 7) // 8
        self.rows = 0
        self.subject_person_facts = 0
        self.object_person_facts = 0
        self.both_person_facts = 0
        self.subject_bits = bytearray(bits_size)
        self.object_bits = bytearray(bits_size)
        self.both_subject_bits = bytearray(bits_size)
        self.both_object_bits = bytearray(bits_size)
        self.all_samples: list[dict[str, object]] = []
        self.person_subject_samples: list[dict[str, object]] = []
        self.person_pair_samples: list[dict[str, object]] = []
        self.person_subject_values: Counter[bytes] = Counter()

    def report(self) -> dict[str, int]:
        return {
            "all_facts": self.rows,
            "person_subject_facts": self.subject_person_facts,
            "unique_person_subjects": cardinality(self.subject_bits),
            "person_object_facts": self.object_person_facts,
            "unique_person_objects": cardinality(self.object_bits),
            "person_pair_facts": self.both_person_facts,
            "unique_pair_subjects": cardinality(self.both_subject_bits),
            "unique_pair_objects": cardinality(self.both_object_bits),
        }


def write_tsv(path: Path, rows: dict[str, dict[str, int]]) -> None:
    columns = list(next(iter(rows.values())))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("predicate\t" + "\t".join(columns) + "\n")
        for predicate, values in rows.items():
            handle.write(predicate + "\t" + "\t".join(str(values[key]) for key in columns) + "\n")


def main() -> None:
    args = args_from_cli()
    facts_path = args.facts.expanduser().resolve()
    persons_path = args.persons.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (facts_path, persons_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    person_index: dict[bytes, int] = {}
    with persons_path.open("rb") as handle:
        for raw in handle:
            name = raw.rstrip(b"\r\n")
            if name and name not in person_index:
                person_index[name] = len(person_index)
    print(f"loaded {len(person_index):,} Person candidates", file=sys.stderr, flush=True)
    coverage = {name.encode(): Coverage(len(person_index)) for name in PREDICATES}
    rng = random.Random(20260927)
    output.mkdir(parents=True, exist_ok=True)
    lines = malformed = matched = bytes_read = 0

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
            stats = coverage.get(predicate)
            if stats is None:
                if lines % args.progress_every == 0:
                    print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)
                continue
            subject, obj = token(fields[0]), token(fields[2])
            if not subject or not obj:
                malformed += 1
                continue
            matched += 1
            stats.rows += 1
            maybe_sample(stats.all_samples, stats.rows, args.sample_size, rng,
                         lines, subject, predicate, obj)
            subject_id = person_index.get(subject)
            object_id = person_index.get(obj)
            if subject_id is not None:
                stats.subject_person_facts += 1
                mark(stats.subject_bits, subject_id)
                maybe_sample(stats.person_subject_samples, stats.subject_person_facts,
                             args.sample_size, rng, lines, subject, predicate, obj)
                if predicate in (b"Profession", b"Occupation"):
                    stats.person_subject_values[obj] += 1
            if object_id is not None:
                stats.object_person_facts += 1
                mark(stats.object_bits, object_id)
            if subject_id is not None and object_id is not None:
                stats.both_person_facts += 1
                mark(stats.both_subject_bits, subject_id)
                mark(stats.both_object_bits, object_id)
                maybe_sample(stats.person_pair_samples, stats.both_person_facts,
                             args.sample_size, rng, lines, subject, predicate, obj)
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows; target facts {matched:,}",
                      file=sys.stderr, flush=True)

    rows = {string(name): stats.report() for name, stats in coverage.items()}
    write_tsv(output / "predicate_coverage.tsv", rows)
    samples = {
        string(name): {
            "all_facts": stats.all_samples,
            "person_subject_facts": stats.person_subject_samples,
            "person_pair_facts": stats.person_pair_samples,
        }
        for name, stats in coverage.items()
    }
    (output / "relation_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "occupation_values.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tobject\tperson_subject_fact_count\n")
        for name in OCCUPATION:
            counts = coverage[name.encode()].person_subject_values
            for obj, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                handle.write(f"{name}\t{string(obj)}\t{count}\n")

    profession = as_int(coverage[b"Profession"].subject_bits)
    birth = as_int(coverage[b"Date of birth"].subject_bits) | as_int(
        coverage[b"Date of Birth"].subject_bits
    )
    kinship = 0
    for name in KINSHIP:
        stats = coverage[name.encode()]
        kinship |= as_int(stats.both_subject_bits) | as_int(stats.both_object_bits)
    summary = {
        "status": "complete_scan" if bytes_read == facts_path.stat().st_size else "sample_only",
        "facts": str(facts_path),
        "person_candidates": str(persons_path),
        "person_candidates_loaded": len(person_index),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "target_predicate_facts": matched,
        "distinct_profession_values_for_person_subjects": len(coverage[b"Profession"].person_subject_values),
        "person_with_profession": profession.bit_count(),
        "person_with_birth_date": birth.bit_count(),
        "person_in_person_pair_kinship_fact": kinship.bit_count(),
        "person_with_profession_and_kinship": (profession & kinship).bit_count(),
        "person_with_profession_birth_and_kinship": (profession & birth & kinship).bit_count(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Counts use provisional is-a Person names; human identity and data quality are not yet verified.",
            "Person-pair kinship counts only facts where both endpoints occur in the Person candidate list.",
            "Occupation value frequencies count facts, not unique people; relations are not yet deduplicated or mapped.",
            "Reservoir samples are reproducible but require manual semantic review.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
