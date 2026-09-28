#!/usr/bin/env python3
"""Annotate provisional Freebase Easy relation candidates with link-file IDs.

Reads only the candidate relation subset and name_mid_status.tsv. Missing or
ambiguous IDs are retained and counted; no relation or person is filtered.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time


STATUS_HEADER = b"name\tstatus\tmid\tdistinct_mid_count"
PAIR_HEADER = b"source_line\taudit_tier\tsubject_name\tpredicate\tobject_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--name-mids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def main() -> None:
    args = parse_args()
    pair_path = args.pairs.expanduser().resolve()
    status_path = args.name_mids.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (pair_path, status_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    unique_mids: dict[bytes, bytes] = {}
    ambiguous_names: set[bytes] = set()
    missing_names: set[bytes] = set()
    status_rows = invalid_status_rows = 0
    with status_path.open("rb") as status_file:
        if status_file.readline().rstrip(b"\r\n") != STATUS_HEADER:
            raise SystemExit("Unexpected name MID status header")
        for raw in status_file:
            status_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not fields[0]:
                invalid_status_rows += 1
                continue
            name, status, mid, _ = fields
            if status == b"unique" and mid:
                unique_mids[name] = mid
            elif status == b"ambiguous" and not mid:
                ambiguous_names.add(name)
            elif status == b"missing" and not mid:
                missing_names.add(name)
            else:
                invalid_status_rows += 1
    if invalid_status_rows or status_rows != len(unique_mids) + len(ambiguous_names) + len(missing_names):
        raise SystemExit("MID status file has invalid or duplicate rows")
    print(f"loaded {len(unique_mids):,} unique-ID names and {len(ambiguous_names):,} ambiguous names",
          file=sys.stderr, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    counts: dict[bytes, Counter[str]] = defaultdict(Counter)
    examples: dict[bytes, list[bytes]] = defaultdict(list)
    participants: set[bytes] = set()
    rows = malformed = unlisted_status_endpoints = 0
    annotated_partial = output / "relation_candidates_with_mids.tsv.partial"
    annotated_final = output / "relation_candidates_with_mids.tsv"
    with pair_path.open("rb") as pairs, annotated_partial.open("wb", buffering=1024 * 1024) as annotated:
        if pairs.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected relation candidate table header")
        annotated.write(b"source_line\taudit_tier\tsubject_name\tsubject_id_status\tsubject_mid\t"
                        b"predicate\tobject_name\tobject_id_status\tobject_mid\n")
        for raw in pairs:
            rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 4)
            if len(fields) != 5 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            source_line, tier, subject, predicate, obj = fields
            participants.add(subject)
            participants.add(obj)
            subject_mid = unique_mids.get(subject, b"")
            object_mid = unique_mids.get(obj, b"")
            subject_status = b"unique" if subject_mid else (
                b"ambiguous" if subject in ambiguous_names else (
                    b"missing" if subject in missing_names else b"unlisted"))
            object_status = b"unique" if object_mid else (
                b"ambiguous" if obj in ambiguous_names else (
                    b"missing" if obj in missing_names else b"unlisted"))
            unlisted_status_endpoints += (subject_status == b"unlisted") + (object_status == b"unlisted")
            annotated.write(source_line + b"\t" + tier + b"\t" + subject + b"\t"
                            + subject_status + b"\t" + subject_mid + b"\t" + predicate
                            + b"\t" + obj + b"\t" + object_status + b"\t" + object_mid + b"\n")
            item = counts[predicate]
            item["facts"] += 1
            if subject_mid and object_mid:
                item["both_unique_id"] += 1
                if subject_mid == object_mid and subject != obj:
                    item["distinct_names_same_mid"] += 1
            elif subject_status == b"ambiguous" or object_status == b"ambiguous":
                item["ambiguous_endpoint"] += 1
            elif subject_mid or object_mid:
                item["one_unique_id"] += 1
            else:
                item["neither_unique_id"] += 1
            if (not subject_mid or not object_mid) and len(examples[predicate]) < 5:
                examples[predicate].append(raw.rstrip(b"\r\n"))
            if rows % args.progress_every == 0:
                print(f"joined {rows:,} candidate rows", file=sys.stderr, flush=True)
    annotated_partial.replace(annotated_final)

    columns = ("facts", "both_unique_id", "one_unique_id", "neither_unique_id",
               "ambiguous_endpoint", "distinct_names_same_mid")
    with (output / "predicate_mid_coverage.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\t" + "\t".join(columns) + "\n")
        for predicate in sorted(counts, key=lambda name: (-counts[name]["facts"], name)):
            item = counts[predicate]
            handle.write(show(predicate) + "\t" + "\t".join(str(item[column]) for column in columns) + "\n")
    with (output / "missing_id_pair_examples.tsv").open("wb") as handle:
        handle.write(PAIR_HEADER + b"\n")
        for predicate in sorted(examples):
            for raw in examples[predicate]:
                handle.write(raw + b"\n")

    totals = Counter()
    for item in counts.values():
        totals.update(item)
    summary = {
        "status": "complete_subset_join" if malformed == 0 and unlisted_status_endpoints == 0 else "subset_join_with_input_mismatch",
        "pairs": str(pair_path),
        "name_mids": str(status_path),
        "name_status_rows_loaded": status_rows,
        "unique_id_names_loaded": len(unique_mids),
        "ambiguous_names_loaded": len(ambiguous_names),
        "candidate_rows_read": rows,
        "malformed_candidate_rows": malformed,
        "candidate_endpoints_absent_from_status": unlisted_status_endpoints,
        "candidate_distinct_names": len(participants),
        "candidate_names_with_unique_id": sum(name in unique_mids for name in participants),
        "candidate_names_with_ambiguous_id": sum(name in ambiguous_names for name in participants),
        "candidate_names_without_id": sum(name in missing_names for name in participants),
        "candidate_names_absent_from_status": sum(name not in unique_mids and name not in ambiguous_names and name not in missing_names for name in participants),
        "candidate_facts_with_both_unique_ids": totals["both_unique_id"],
        "candidate_facts_with_one_unique_id": totals["one_unique_id"],
        "candidate_facts_with_neither_unique_id": totals["neither_unique_id"],
        "candidate_facts_with_ambiguous_endpoint": totals["ambiguous_endpoint"],
        "candidate_facts_with_distinct_names_same_mid": totals["distinct_names_same_mid"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "No candidate relation fact was excluded because an ID is missing or ambiguous.",
            "A unique link-file ID is provisional evidence; Freebase Easy facts themselves remain keyed by names.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
