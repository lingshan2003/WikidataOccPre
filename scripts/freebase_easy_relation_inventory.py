#!/usr/bin/env python3
"""Discover all Freebase Easy predicates that connect Person candidates.

This CPU-only scan inventories every predicate and samples person-to-person
facts. It deliberately does not call a name match a valid social relation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time


ATTRIBUTE_PREDICATES = (b"Profession", b"Date of birth", b"Date of death")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--persons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lines", type=int, help="Optional probe; omit for full scan")
    parser.add_argument("--sample-size", type=int, default=5)
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


def count_bits(bits: bytearray) -> int:
    return sum(value.bit_count() for value in bits)


class PredicateStats:
    __slots__ = ("facts", "person_subject_facts", "person_pair_facts", "samples")

    def __init__(self) -> None:
        self.facts = 0
        self.person_subject_facts = 0
        self.person_pair_facts = 0
        self.samples: list[dict[str, object]] = []


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
    with persons_path.open("rb") as handle:
        for raw in handle:
            entity = raw.rstrip(b"\r\n")
            if entity and entity not in person_ids:
                person_ids[entity] = len(person_ids)
    print(f"loaded {len(person_ids):,} Person candidates", file=sys.stderr, flush=True)
    masks = {
        predicate: bytearray((len(person_ids) + 7) // 8)
        for predicate in ATTRIBUTE_PREDICATES
    }
    stats: dict[bytes, PredicateStats] = defaultdict(PredicateStats)
    rng = random.Random(20260927)
    lines = bytes_read = malformed = all_pair_facts = 0
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
            subject, predicate, obj = (token(field) for field in fields[:3])
            if not subject or not predicate or not obj:
                malformed += 1
                continue
            item = stats[predicate]
            item.facts += 1
            subject_id = person_ids.get(subject)
            if subject_id is not None:
                item.person_subject_facts += 1
                if predicate in masks:
                    mark(masks[predicate], subject_id)
                if person_ids.get(obj) is not None:
                    item.person_pair_facts += 1
                    all_pair_facts += 1
                    if args.sample_size:
                        seen = item.person_pair_facts
                        position = seen - 1 if seen <= args.sample_size else rng.randrange(seen)
                    else:
                        position = -1
                    if position >= 0 and position < args.sample_size:
                        sample: dict[str, object] = {
                            "line": lines,
                            "subject": string(subject),
                            "object": string(obj),
                        }
                        if seen <= args.sample_size:
                            item.samples.append(sample)
                        else:
                            item.samples[position] = sample
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows; name-matched pairs {all_pair_facts:,}",
                      file=sys.stderr, flush=True)

    ordered = sorted(stats.items(), key=lambda entry: (-entry[1].person_pair_facts, -entry[1].facts, entry[0]))
    with (output / "predicate_inventory.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tall_facts\tperson_subject_facts\tname_matched_person_pair_facts\tpair_fraction_of_all_facts\n")
        for predicate, item in ordered:
            fraction = item.person_pair_facts / item.facts
            handle.write(
                f"{string(predicate)}\t{item.facts}\t{item.person_subject_facts}\t"
                f"{item.person_pair_facts}\t{fraction:.6f}\n"
            )
    pair_samples = {
        string(predicate): item.samples
        for predicate, item in ordered if item.person_pair_facts
    }
    (output / "person_pair_samples.json").write_text(
        json.dumps(pair_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    profession = int.from_bytes(masks[b"Profession"], "little")
    birth = int.from_bytes(masks[b"Date of birth"], "little")
    death = int.from_bytes(masks[b"Date of death"], "little")
    summary = {
        "status": "complete_scan" if bytes_read == facts_path.stat().st_size else "sample_only",
        "facts": str(facts_path),
        "persons": str(persons_path),
        "person_candidates_loaded": len(person_ids),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "distinct_predicates": len(stats),
        "predicates_with_name_matched_person_pairs": sum(
            item.person_pair_facts > 0 for item in stats.values()
        ),
        "name_matched_person_pair_facts_all_predicates": all_pair_facts,
        "person_with_profession": count_bits(masks[b"Profession"]),
        "person_with_birth_date": count_bits(masks[b"Date of birth"]),
        "person_with_death_date": count_bits(masks[b"Date of death"]),
        "person_with_profession_and_birth_or_death": (
            profession & (birth | death)
        ).bit_count(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "A matching Person name in both positions does not establish a valid human-to-human social relation.",
            "Profession, is-a, locations, and other attribute predicates may have name collisions.",
            "The inventory does not select relations or filter final graph nodes.",
            "Birth/death counts indicate raw fact presence, not yet parsed or verified dates.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
