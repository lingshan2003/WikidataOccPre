#!/usr/bin/env python3
"""Export small, source-traceable kinship conflicts from a relation subset.

Reads only the filtered relation_candidates_v1.tsv. It flags structural and
cross-predicate conflicts; no edge is removed and no direction is guessed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time


HEADER = b"source_line\taudit_tier\tsubject_name\tpredicate\tobject_name"
CHILD = b"Children"
SIBLING = b"Sibling"
PARTNER = b"Spouse (or domestic partner)"
KINSHIP = {CHILD, SIBLING, PARTNER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs_path = args.pairs.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not pairs_path.is_file():
        raise SystemExit(f"Input file does not exist: {pairs_path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    grouped: dict[tuple[bytes, bytes], list[tuple[bytes, bytes, bytes, bytes]]] = defaultdict(list)
    rows_read = malformed = kinship_rows = 0
    with pairs_path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != HEADER:
            raise SystemExit("Unexpected relation candidate header")
        for raw in handle:
            rows_read += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 4)
            if len(fields) != 5 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            line, _, subject, predicate, obj = fields
            if predicate not in KINSHIP:
                continue
            kinship_rows += 1
            key = (subject, obj) if subject <= obj else (obj, subject)
            grouped[key].append((line, subject, predicate, obj))

    reason_counts: Counter[str] = Counter()
    flagged_pairs = 0
    flagged_rows = 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "kinship_conflict_rows.tsv").open("wb") as handle:
        handle.write(b"pair_name_a\tpair_name_b\treasons\tsource_line\tsubject_name\tpredicate\tobject_name\n")
        for (name_a, name_b), rows in sorted(grouped.items()):
            predicates = {predicate for _, _, predicate, _ in rows}
            child_directions = {(subject, obj) for _, subject, predicate, obj in rows if predicate == CHILD}
            reasons: list[str] = []
            if name_a == name_b and CHILD in predicates:
                reasons.append("child_self")
            if name_a != name_b and (name_a, name_b) in child_directions and (name_b, name_a) in child_directions:
                reasons.append("child_reciprocal")
            if CHILD in predicates and SIBLING in predicates:
                reasons.append("child_sibling")
            if CHILD in predicates and PARTNER in predicates:
                reasons.append("child_partner")
            if SIBLING in predicates and PARTNER in predicates:
                reasons.append("sibling_partner")
            if not reasons:
                continue
            flagged_pairs += 1
            reason_counts.update(reasons)
            reason_cell = ",".join(reasons).encode("ascii")
            for line, subject, predicate, obj in sorted(rows, key=lambda row: int(row[0])):
                handle.write(b"\t".join((name_a, name_b, reason_cell, line, subject, predicate, obj)) + b"\n")
                flagged_rows += 1

    with (output / "reason_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("reason\tunique_unordered_name_pairs\n")
        for reason in ("child_self", "child_reciprocal", "child_sibling", "child_partner", "sibling_partner"):
            handle.write(f"{reason}\t{reason_counts[reason]}\n")
    summary = {
        "status": "complete_subset_audit" if malformed == 0 else "subset_audit_with_malformed_rows",
        "pairs": str(pairs_path), "rows_read": rows_read, "malformed_rows": malformed,
        "kinship_rows_read": kinship_rows,
        "flagged_unordered_name_pairs": flagged_pairs,
        "flagged_source_rows": flagged_rows,
        "reason_pair_counts": dict(reason_counts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "A pair may have multiple reasons, so reason counts need not sum to flagged pair count.",
            "Sibling and partner reciprocal records are expected and are not flagged by themselves.",
            "No pair is removed; these name-based flags require semantic and identity review.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
