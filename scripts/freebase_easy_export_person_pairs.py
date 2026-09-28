#!/usr/bin/env python3
"""Export every fact whose subject and object match provisional Person names.

This is a raw, CPU-only extraction. It keeps all predicates, including false
person-pair matches caused by names shared with types, places, or occupations.
No profession or birth/death filter is applied. Freebase IDs are audited later.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--persons", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, help="Prior full predicate_inventory.tsv for count validation")
    parser.add_argument("--max-lines", type=int, help="Optional short probe; omit for full extraction")
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


def load_expected_counts(path: Path) -> dict[bytes, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["predicate"].encode("utf-8"): int(row["name_matched_person_pair_facts"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> None:
    args = parse_args()
    facts_path = args.facts.expanduser().resolve()
    persons_path = args.persons.expanduser().resolve()
    inventory_path = args.inventory.expanduser().resolve() if args.inventory else None
    output = args.output_dir.expanduser().resolve()
    for source in (facts_path, persons_path, inventory_path):
        if source is not None and not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    with persons_path.open("rb") as handle:
        persons = {raw.rstrip(b"\r\n") for raw in handle if raw.strip()}
    print(f"loaded {len(persons):,} provisional Person names", file=sys.stderr, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[bytes] = Counter()
    lines = bytes_read = malformed = pair_facts = 0
    partial_path = output / "person_name_pairs.tsv.partial"
    final_path = output / "person_name_pairs.tsv"
    with facts_path.open("rb") as facts, partial_path.open("wb", buffering=1024 * 1024) as pairs:
        pairs.write(b"source_line\tsubject_name\tpredicate\tobject_name\n")
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
            if subject in persons and obj in persons:
                pairs.write(str(lines).encode("ascii") + b"\t" + subject + b"\t"
                            + predicate + b"\t" + obj + b"\n")
                counts[predicate] += 1
                pair_facts += 1
            if lines % args.progress_every == 0:
                print(f"scanned {lines:,} rows; exported {pair_facts:,} raw name pairs",
                      file=sys.stderr, flush=True)
    partial_path.replace(final_path)

    with (output / "predicate_pair_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tname_matched_pair_facts\n")
        for predicate, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{predicate.decode('utf-8', errors='replace')}\t{count}\n")

    complete = bytes_read == facts_path.stat().st_size
    mismatches: list[dict[str, object]] = []
    if complete and inventory_path is not None:
        expected = load_expected_counts(inventory_path)
        for predicate in sorted(expected.keys() | counts.keys()):
            if expected.get(predicate, 0) != counts.get(predicate, 0):
                mismatches.append({
                    "predicate": predicate.decode("utf-8", errors="replace"),
                    "inventory": expected.get(predicate, 0),
                    "export": counts.get(predicate, 0),
                })
    summary = {
        "status": "complete_scan" if complete else "sample_only",
        "facts": str(facts_path),
        "persons": str(persons_path),
        "inventory": str(inventory_path) if inventory_path else None,
        "person_candidate_names_loaded": len(persons),
        "lines_scanned": lines,
        "bytes_scanned": bytes_read,
        "malformed_rows": malformed,
        "raw_name_pair_facts_exported": pair_facts,
        "predicates_with_raw_name_pairs": len(counts),
        "inventory_count_mismatches": mismatches,
        "output_file": str(final_path),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Every predicate is retained; name matching is not proof of two human entities or a social tie.",
            "No profession, date, relation whitelist, or Freebase ID filter was applied.",
            "source_line refers to the one-based line number in facts.txt.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
