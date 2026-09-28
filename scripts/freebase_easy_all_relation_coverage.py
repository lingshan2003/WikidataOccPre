#!/usr/bin/env python3
"""Count unique people and attribute overlap for every name-matched predicate.

The candidate tiers are provisional. This read-only, CPU-only scan does not
turn name matches into social edges or filter the final graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time


ATTRIBUTES = (b"Profession", b"Date of birth", b"Date of death")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--persons", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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


def bit_count(bits: bytearray) -> int:
    return sum(value.bit_count() for value in bits)


def sample_row(
    rows: list[dict[str, object]], seen: int, limit: int,
    line_no: int, subject: bytes, obj: bytes, rng: random.Random,
) -> None:
    if not limit:
        return
    position = seen - 1 if seen <= limit else rng.randrange(seen)
    if position >= limit:
        return
    row: dict[str, object] = {
        "line": line_no,
        "subject": string(subject),
        "object": string(obj),
    }
    if seen <= limit:
        rows.append(row)
    else:
        rows[position] = row


def load_inventory(path: Path) -> dict[bytes, int]:
    counts: dict[bytes, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pairs = int(row["name_matched_person_pair_facts"])
            if pairs > 0:
                counts[row["predicate"].encode("utf-8")] = pairs
    if not counts:
        raise SystemExit("Inventory has no predicates with person-pair name matches")
    return counts


def main() -> None:
    args = parse_args()
    facts_path = args.facts.expanduser().resolve()
    persons_path = args.persons.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (facts_path, persons_path, inventory_path, config_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    inventory = load_inventory(inventory_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    priority = {item.encode("utf-8") for item in config["priority"]}
    review = {item.encode("utf-8") for item in config["semantic_review"]}
    if priority & review:
        raise SystemExit("Priority and semantic_review predicates must not overlap")
    unknown = (priority | review) - inventory.keys()
    if unknown:
        raise SystemExit(f"Configured predicates absent from inventory: {sorted(map(string, unknown))}")

    started = time.monotonic()
    person_ids: dict[bytes, int] = {}
    with persons_path.open("rb") as handle:
        for raw in handle:
            entity = raw.rstrip(b"\r\n")
            if entity and entity not in person_ids:
                person_ids[entity] = len(person_ids)
    print(f"loaded {len(person_ids):,} Person candidates", file=sys.stderr, flush=True)
    bits_size = (len(person_ids) + 7) // 8
    participants = {predicate: bytearray(bits_size) for predicate in inventory}
    attribute_bits = {predicate: bytearray(bits_size) for predicate in ATTRIBUTES}
    pair_facts = dict.fromkeys(inventory, 0)
    samples = {predicate: [] for predicate in priority | review}
    rng = random.Random(20260928)
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
            if predicate not in inventory and predicate not in attribute_bits:
                if lines % args.progress_every == 0:
                    print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)
                continue
            subject, obj = token(fields[0]), token(fields[2])
            if not subject or not obj:
                malformed += 1
                continue
            subject_id = person_ids.get(subject)
            if subject_id is not None:
                if predicate in attribute_bits:
                    mark(attribute_bits[predicate], subject_id)
                if predicate in inventory:
                    object_id = person_ids.get(obj)
                    if object_id is not None:
                        pair_facts[predicate] += 1
                        mark(participants[predicate], subject_id)
                        mark(participants[predicate], object_id)
                        if predicate in samples:
                            limit = 30 if predicate in priority else 15
                            sample_row(samples[predicate], pair_facts[predicate], limit,
                                       lines, subject, obj, rng)
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows", file=sys.stderr, flush=True)

    profession = int.from_bytes(attribute_bits[b"Profession"], "little")
    birth = int.from_bytes(attribute_bits[b"Date of birth"], "little")
    death = int.from_bytes(attribute_bits[b"Date of death"], "little")
    dated = birth | death
    rows: list[dict[str, object]] = []
    for predicate in sorted(inventory, key=lambda item: (-inventory[item], item)):
        people = int.from_bytes(participants[predicate], "little")
        tier = "priority" if predicate in priority else (
            "semantic_review" if predicate in review else "not_selected"
        )
        rows.append({
            "predicate": string(predicate),
            "audit_tier": tier,
            "person_pair_facts": pair_facts[predicate],
            "unique_person_participants": people.bit_count(),
            "participants_with_profession": (people & profession).bit_count(),
            "participants_with_birth": (people & birth).bit_count(),
            "participants_with_death": (people & death).bit_count(),
            "participants_with_profession_and_birth_or_death": (
                people & profession & dated
            ).bit_count(),
        })
    with (output / "predicate_person_coverage.tsv").open("w", encoding="utf-8") as handle:
        columns = list(rows[0])
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    (output / "review_samples.json").write_text(
        json.dumps({string(key): value for key, value in samples.items()},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    unions: dict[str, int] = {}
    for label, selected in (("priority", priority), ("semantic_review", review),
                            ("all_name_matched", set(inventory))):
        union = 0
        for predicate in selected:
            union |= int.from_bytes(participants[predicate], "little")
        unions[label] = union
    group_coverage = {
        label: {
            "person_participants": people.bit_count(),
            "participants_with_profession": (people & profession).bit_count(),
            "participants_with_birth_or_death": (people & dated).bit_count(),
            "participants_with_profession_and_birth_or_death": (
                people & profession & dated
            ).bit_count(),
        }
        for label, people in unions.items()
    }
    mismatches = [
        {"predicate": string(predicate), "inventory": expected, "rescan": pair_facts[predicate]}
        for predicate, expected in inventory.items()
        if expected != pair_facts[predicate]
    ] if bytes_read == facts_path.stat().st_size else []
    summary = {
        "status": "complete_scan" if bytes_read == facts_path.stat().st_size else "sample_only",
        "facts": str(facts_path),
        "person_candidates": str(persons_path),
        "inventory": str(inventory_path),
        "config": str(config_path),
        "person_candidates_loaded": len(person_ids),
        "predicates_audited": len(inventory),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "people_with_profession": bit_count(attribute_bits[b"Profession"]),
        "people_with_birth": bit_count(attribute_bits[b"Date of birth"]),
        "people_with_death": bit_count(attribute_bits[b"Date of death"]),
        "group_coverage": group_coverage,
        "inventory_count_mismatches": mismatches,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "All pair memberships are based on Freebase Easy entity-name matches, not verified stable IDs.",
            "Priority and review tiers are exploratory, not a frozen training-graph whitelist.",
            "Birth/death coverage indicates a raw fact, not necessarily a valid parsed date.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
