#!/usr/bin/env python3
"""Audit Freebase Easy person-to-person family predicates and fiction overlap.

This is a read-only, CPU-only quality check. It does not discard entities or
construct a training graph. Treat the output as evidence for later rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time


RELATIONS = (
    b"Children", b"Sibling", b"Spouse (or domestic partner)", b"Married To", b"Parent"
)
EXAMPLE_LIMIT = 20


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


def name(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def mark(bits: bytearray, index: int) -> None:
    bits[index >> 3] |= 1 << (index & 7)


def marked(bits: bytearray, index: int) -> bool:
    return bool(bits[index >> 3] & (1 << (index & 7)))


def pair_names(pair: tuple[int, int], names: list[bytes]) -> list[str]:
    return [name(names[pair[0]]), name(names[pair[1]])]


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
    persons: dict[bytes, int] = {}
    names: list[bytes] = []
    with persons_path.open("rb") as handle:
        for raw in handle:
            entity = raw.rstrip(b"\r\n")
            if entity and entity not in persons:
                persons[entity] = len(names)
                names.append(entity)
    print(f"loaded {len(names):,} Person candidates", file=sys.stderr, flush=True)
    fiction_bits = bytearray((len(names) + 7) // 8)
    fiction_type_facts: Counter[bytes] = Counter()
    relation_facts: Counter[bytes] = Counter()
    person_pair_facts: Counter[bytes] = Counter()
    edges: dict[bytes, set[tuple[int, int]]] = {relation: set() for relation in RELATIONS}
    lines = bytes_read = malformed = 0
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
            if predicate not in edges and predicate != b"is-a":
                if lines % args.progress_every == 0:
                    print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)
                continue
            subject, obj = token(fields[0]), token(fields[2])
            if not subject or not obj:
                malformed += 1
                continue
            if predicate == b"is-a":
                obj_lower = obj.lower()
                if b"fiction" in obj_lower or b"character" in obj_lower:
                    person_id = persons.get(subject)
                    if person_id is not None:
                        mark(fiction_bits, person_id)
                        fiction_type_facts[obj] += 1
            else:
                relation_facts[predicate] += 1
                subject_id = persons.get(subject)
                object_id = persons.get(obj)
                if subject_id is not None and object_id is not None:
                    person_pair_facts[predicate] += 1
                    edges[predicate].add((subject_id, object_id))
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)

    fiction_people = sum(byte.bit_count() for byte in fiction_bits)
    report: dict[str, dict[str, int]] = {}
    examples: dict[str, dict[str, list[list[str]]]] = {}
    for predicate in RELATIONS:
        pairs = edges[predicate]
        reciprocal_directed = sum(
            1 for subject_id, object_id in pairs
            if subject_id != object_id and (object_id, subject_id) in pairs
        )
        self_pairs = sum(subject_id == object_id for subject_id, object_id in pairs)
        fiction_pairs = sum(
            marked(fiction_bits, subject_id) or marked(fiction_bits, object_id)
            for subject_id, object_id in pairs
        )
        participants = bytearray(len(fiction_bits))
        for subject_id, object_id in pairs:
            mark(participants, subject_id)
            mark(participants, object_id)
        report[name(predicate)] = {
            "all_facts": relation_facts[predicate],
            "person_pair_facts": person_pair_facts[predicate],
            "unique_directed_person_pairs": len(pairs),
            "duplicate_person_pair_facts": person_pair_facts[predicate] - len(pairs),
            "self_pairs": self_pairs,
            "reciprocal_unordered_pairs": reciprocal_directed // 2,
            "unique_unordered_person_pairs": len(pairs) - reciprocal_directed // 2,
            "unique_person_participants": sum(byte.bit_count() for byte in participants),
            "unique_directed_pairs_with_fiction_like_type": fiction_pairs,
        }
        selected = {
            "self_pairs": [],
            "reciprocal_pairs": [],
            "fiction_like_pairs": [],
        }
        for pair in sorted(pairs):
            subject_id, object_id = pair
            if subject_id == object_id and len(selected["self_pairs"]) < EXAMPLE_LIMIT:
                selected["self_pairs"].append(pair_names(pair, names))
            if (subject_id < object_id and (object_id, subject_id) in pairs
                    and len(selected["reciprocal_pairs"]) < EXAMPLE_LIMIT):
                selected["reciprocal_pairs"].append(pair_names(pair, names))
            if ((marked(fiction_bits, subject_id) or marked(fiction_bits, object_id))
                    and len(selected["fiction_like_pairs"]) < EXAMPLE_LIMIT):
                selected["fiction_like_pairs"].append(pair_names(pair, names))
            if all(len(group) >= EXAMPLE_LIMIT for group in selected.values()):
                break
        examples[name(predicate)] = selected

    with (output / "relation_quality.tsv").open("w", encoding="utf-8") as handle:
        columns = list(next(iter(report.values())))
        handle.write("predicate\t" + "\t".join(columns) + "\n")
        for predicate, values in report.items():
            handle.write(predicate + "\t" + "\t".join(str(values[key]) for key in columns) + "\n")
    with (output / "fiction_like_types.tsv").open("w", encoding="utf-8") as handle:
        handle.write("type_object\tperson_subject_facts\n")
        for obj, count in fiction_type_facts.most_common():
            handle.write(f"{name(obj)}\t{count}\n")
    (output / "anomaly_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete_scan" if bytes_read == facts_path.stat().st_size else "sample_only",
        "facts": str(facts_path),
        "person_candidates": str(persons_path),
        "person_candidates_loaded": len(names),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "person_candidates_with_fiction_like_type": fiction_people,
        "fiction_like_rule": "Any is-a object containing fiction or character, case-insensitive; audit flag only.",
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Reciprocal pairs are expected for symmetric Sibling and Spouse predicates; they are suspicious for Children.",
            "Fiction-like type overlap is not an automatic exclusion rule.",
            "Candidate names are provisional entity keys; verify identity through freebase-links.txt before merging sources.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
