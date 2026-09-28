#!/usr/bin/env python3
"""Audit provisional relation predicates using exported Freebase Easy name pairs.

This reads only person_name_pairs.tsv, not the complete facts.txt. Configured
predicates remain candidates, not an approved social-graph whitelist.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time


HEADER = b"source_line\tsubject_name\tpredicate\tobject_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def label(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def main() -> None:
    args = parse_args()
    pair_path = args.pairs.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (pair_path, config_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    priority = {name.encode("utf-8") for name in config["priority"]}
    review = {name.encode("utf-8") for name in config["semantic_review"]}
    if priority & review:
        raise SystemExit("priority and semantic_review overlap")
    selected = priority | review
    tiers = {name: b"priority" for name in priority}
    tiers.update({name: b"semantic_review" for name in review})

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    row_counts: Counter[bytes] = Counter()
    directed: dict[bytes, set[tuple[bytes, bytes]]] = defaultdict(set)
    selected_rows = raw_rows = malformed = 0
    all_pair_predicates: set[bytes] = set()
    candidate_path = output / "relation_candidates_v1.tsv"
    president_path = output / "president_all.tsv"
    with pair_path.open("rb") as source, candidate_path.open("wb") as candidates, \
            president_path.open("wb") as president:
        header = source.readline().rstrip(b"\r\n")
        if header != HEADER:
            raise SystemExit(f"Unexpected input header: {label(header)!r}")
        candidates.write(b"source_line\taudit_tier\tsubject_name\tpredicate\tobject_name\n")
        president.write(HEADER + b"\n")
        for raw in source:
            raw_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            source_line, subject, predicate, obj = fields
            all_pair_predicates.add(predicate)
            if predicate in selected:
                candidates.write(source_line + b"\t" + tiers[predicate] + b"\t"
                                 + subject + b"\t" + predicate + b"\t" + obj + b"\n")
                if predicate == b"President":
                    president.write(raw)
                row_counts[predicate] += 1
                directed[predicate].add((subject, obj))
                selected_rows += 1
            if raw_rows % args.progress_every == 0:
                print(f"read {raw_rows:,} raw name pairs; kept {selected_rows:,} audit candidates",
                      file=sys.stderr, flush=True)

    with (output / "predicate_structure.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\taudit_tier\tfacts\tunique_directed_pairs\tduplicate_directed_facts\tself_pairs\treciprocal_unordered_pairs\tunique_participants\n")
        for predicate in sorted(selected, key=lambda item: (-row_counts[item], item)):
            pairs = directed[predicate]
            self_pairs = sum(a == b for a, b in pairs)
            reciprocal = sum(a < b and (b, a) in pairs for a, b in pairs)
            people = {name for pair in pairs for name in pair}
            handle.write(f"{label(predicate)}\t{label(tiers[predicate])}\t{row_counts[predicate]}\t"
                         f"{len(pairs)}\t{row_counts[predicate] - len(pairs)}\t{self_pairs}\t"
                         f"{reciprocal}\t{len(people)}\n")

    with (output / "cross_predicate_overlap.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate_a\tpredicate_b\tsame_direction_pairs\treverse_direction_pairs\n")
        names = sorted(selected)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = directed[left], directed[right]
                smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
                same = sum(pair in larger for pair in smaller)
                reverse = sum((obj, subject) in larger for subject, obj in smaller)
                if same or reverse:
                    handle.write(f"{label(left)}\t{label(right)}\t{same}\t{reverse}\n")

    summary = {
        "status": "complete_subset_scan" if malformed == 0 else "subset_scan_with_malformed_rows",
        "pairs": str(pair_path),
        "config": str(config_path),
        "raw_name_pair_rows_read": raw_rows,
        "malformed_rows": malformed,
        "all_predicates_seen": len(all_pair_predicates),
        "configured_predicates": len(selected),
        "configured_predicates_with_rows": sum(bool(row_counts[name]) for name in selected),
        "candidate_rows_written": selected_rows,
        "president_rows_written": row_counts[b"President"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Counts use name-matched provisional Person records; no MID or human-identity check was applied.",
            "Candidate rows are for auditing and are not a frozen training-graph whitelist.",
            "Cross-predicate overlaps compare exact subject/object names; reverse counts compare swapped names.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
