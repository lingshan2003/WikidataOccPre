#!/usr/bin/env python3
"""Audit structure and link-ID coverage of an attribute-filtered raw pair subset.

This does not infer relation semantics or establish that a name-matched Easy
fact refers to the Freebase ID supplied by freebase-links.txt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time


PAIR_HEADER = b"source_line\tsubject_name\tpredicate\tobject_name"
STATUS_HEADER = b"name\tstatus\tmid\tdistinct_mid_count"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="cohort_name_pairs.tsv from freebase_easy_filter_person_cohort.py")
    parser.add_argument("--name-mids", type=Path, required=True,
                        help="name_mid_audit_v3/name_mid_status.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def pairs_from(path: Path):
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected pair header")
        for number, raw in enumerate(handle, 2):
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                raise SystemExit(f"Malformed pair row at line {number}")
            yield fields


def main() -> None:
    options = cli()
    pair_path = options.pairs.expanduser().resolve()
    status_path = options.name_mids.expanduser().resolve()
    output = options.output_dir.expanduser().resolve()
    for path in (pair_path, status_path):
        if not path.is_file():
            raise SystemExit(f"Input file does not exist: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    participants: set[bytes] = set()
    input_rows = 0
    for _, subject, _, obj in pairs_from(pair_path):
        input_rows += 1
        participants.add(subject)
        participants.add(obj)
    print(f"loaded {input_rows:,} pair rows, {len(participants):,} names", file=sys.stderr, flush=True)

    status: dict[bytes, tuple[bytes, bytes]] = {}
    with status_path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != STATUS_HEADER:
            raise SystemExit("Unexpected name-MID status header")
        for number, raw in enumerate(handle, 2):
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not fields[0]:
                raise SystemExit(f"Malformed name-MID row at line {number}")
            name, label, mid, _ = fields
            if name not in participants:
                continue
            if name in status:
                raise SystemExit(f"Duplicate name-MID status row: {show(name)}")
            if label not in (b"unique", b"ambiguous", b"missing") or (label == b"unique") != bool(mid):
                raise SystemExit(f"Invalid name-MID status at line {number}")
            status[name] = label, mid

    counts: dict[bytes, Counter[str]] = defaultdict(Counter)
    directed: dict[bytes, set[tuple[bytes, bytes]]] = defaultdict(set)
    mid_directed: dict[bytes, set[tuple[bytes, bytes]]] = defaultdict(set)
    example_counts: Counter[tuple[bytes, str]] = Counter()
    examples: list[tuple[bytes, bytes, bytes, bytes, str]] = []
    for line, subject, predicate, obj in pairs_from(pair_path):
        item = counts[predicate]
        item["raw_rows"] += 1
        directed[predicate].add((subject, obj))
        sub_label, sub_mid = status.get(subject, (b"unlisted", b""))
        obj_label, obj_mid = status.get(obj, (b"unlisted", b""))
        if subject == obj:
            item["same_name_self_rows"] += 1
            reason = "same_name_self"
        elif sub_mid and obj_mid and sub_mid == obj_mid:
            item["same_id_distinct_name_rows"] += 1
            reason = "same_id_distinct_name"
        elif not sub_mid or not obj_mid:
            item["id_unresolved_rows"] += 1
            reason = "id_unresolved"
        else:
            reason = ""
        if sub_mid and obj_mid:
            item["both_unique_id_rows"] += 1
            mid_directed[predicate].add((sub_mid, obj_mid))
        elif sub_mid or obj_mid:
            item["one_unique_id_rows"] += 1
        else:
            item["neither_unique_id_rows"] += 1
        if sub_label == b"unlisted" or obj_label == b"unlisted":
            item["unlisted_status_rows"] += 1
        if reason and example_counts[(predicate, reason)] < 5:
            examples.append((line, subject, predicate, obj, reason))
            example_counts[(predicate, reason)] += 1

    output.mkdir(parents=True, exist_ok=True)
    columns = (
        "raw_rows", "unique_directed_name_pairs", "reciprocal_unordered_name_pairs",
        "same_name_self_rows", "both_unique_id_rows", "one_unique_id_rows",
        "neither_unique_id_rows", "unique_directed_id_pairs", "same_id_distinct_name_rows",
        "unlisted_status_rows",
    )
    with (output / "predicate_structure.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\t" + "\t".join(columns) + "\n")
        for predicate, item in sorted(counts.items(), key=lambda pair: (-pair[1]["raw_rows"], pair[0])):
            names = directed[predicate]
            item["unique_directed_name_pairs"] = len(names)
            item["reciprocal_unordered_name_pairs"] = sum(
                a < b and (b, a) in names for a, b in names
            )
            item["unique_directed_id_pairs"] = len(mid_directed[predicate])
            handle.write(show(predicate) + "\t" + "\t".join(str(item[column]) for column in columns) + "\n")
    with (output / "anomaly_examples.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source_line\tsubject_name\tpredicate\tobject_name\treason\n")
        for line, subject, predicate, obj, reason in examples:
            handle.write("\t".join(map(show, (line, subject, predicate, obj))) + f"\t{reason}\n")

    totals: Counter[str] = Counter()
    for item in counts.values():
        totals.update(item)
    summary = {
        "status": "complete_subset_audit" if len(status) == len(participants) else "name_id_status_incomplete",
        "pairs": str(pair_path), "name_mids": str(status_path),
        "pair_rows_read": input_rows, "participant_names": len(participants),
        "names_with_status": len(status),
        "names_with_unique_id": sum(label == b"unique" for label, _ in status.values()),
        "names_with_ambiguous_id": sum(label == b"ambiguous" for label, _ in status.values()),
        "names_with_missing_id": sum(label == b"missing" for label, _ in status.values()),
        "predicates_seen": len(counts),
        "raw_rows_with_both_unique_ids": totals["both_unique_id_rows"],
        "raw_rows_with_one_unique_id": totals["one_unique_id_rows"],
        "raw_rows_with_neither_unique_id": totals["neither_unique_id_rows"],
        "same_name_self_rows": totals["same_name_self_rows"],
        "same_id_distinct_name_rows": totals["same_id_distinct_name_rows"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Every predicate remains in this structural audit; this is not a relation whitelist.",
            "Reciprocal counts are unordered name pairs with both directions present, excluding self-pairs.",
            "A unique link-file ID is provisional evidence and does not resolve every Easy fact's identity.",
            "No pair is removed; raw pair and name-MID inputs remain unchanged.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
