#!/usr/bin/env python3
"""Make a review-only Freebase profession crosswalk draft from old graph labels.

Exact case-insensitive text matches are candidate evidence, not approved
semantic mappings. This script never chooses a final Freebase training label.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freebase-values", type=Path, required=True)
    parser.add_argument("--old-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def main() -> None:
    args = parse_args()
    values_path = args.freebase_values.expanduser().resolve()
    old_nodes_path = args.old_nodes.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (values_path, old_nodes_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    old: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    with old_nodes_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            level3 = row["occupation_level3"].strip()
            if level3 and level3 not in ("Missing", "Other"):
                old[normalize(level3)][(
                    row["occupation_level1"], row["occupation_level2"]
                )] += 1

    with values_path.open("r", encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle, delimiter="\t"))
    required = {"raw_value", "facts", "unique_people", "priority_people", "union_people"}
    if not values or not required.issubset(values[0]):
        raise SystemExit("Unexpected Freebase profession-values columns")
    values.sort(key=lambda row: (-int(row["priority_people"]), -int(row["facts"]), row["raw_value"]))

    output.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    priority_pair_counts: Counter[str] = Counter()
    columns = (
        "raw_value", "facts", "unique_people", "priority_people", "union_people",
        "draft_status", "suggested_level1", "suggested_level2",
        "old_supporting_nodes", "old_alternatives", "review_decision",
        "approved_level1", "approved_level2", "review_notes",
    )
    with (output / "profession_crosswalk_draft.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in values:
            mappings = old.get(normalize(row["raw_value"]))
            suggested_l1 = suggested_l2 = ""
            old_support = 0
            alternatives = ""
            if not mappings:
                status = "unmatched"
            elif len(mappings) > 1:
                status = "ambiguous_old_mapping"
                alternatives = " | ".join(
                    f"{l1} > {l2} ({count})"
                    for (l1, l2), count in mappings.most_common()
                )
                old_support = sum(mappings.values())
            else:
                (suggested_l1, suggested_l2), old_support = next(iter(mappings.items()))
                status = ("old_other_bucket" if suggested_l1 in ("Other", "Missing")
                          or suggested_l2 in ("Other", "Missing") else "exact_text_candidate")
            status_counts[status] += 1
            priority_pair_counts[status] += int(row["priority_people"])
            writer.writerow({
                **{key: row[key] for key in required},
                "draft_status": status,
                "suggested_level1": suggested_l1,
                "suggested_level2": suggested_l2,
                "old_supporting_nodes": old_support,
                "old_alternatives": alternatives,
                "review_decision": "",
                "approved_level1": "",
                "approved_level2": "",
                "review_notes": "",
            })

    summary = {
        "status": "review_draft_only",
        "freebase_values": str(values_path),
        "old_nodes": str(old_nodes_path),
        "freebase_raw_values": len(values),
        "old_distinct_level3_strings": len(old),
        "status_value_counts": dict(status_counts),
        "status_priority_person_value_pair_counts": dict(priority_pair_counts),
        "notes": [
            "An exact text match is not an approved semantic crosswalk.",
            "Counts are person/value pairs, not distinct people covered by a draft mapping.",
            "Other/Missing and ambiguous old mappings require explicit decisions.",
            "Final label policy must handle multiple occupations per person and be shared across sources.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
