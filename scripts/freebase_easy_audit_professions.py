#!/usr/bin/env python3
"""Deduplicate Freebase Easy Person--Profession facts and audit label multiplicity.

Reads only exported attributes and provisional relation candidates. An
external sort deduplicates person/value pairs without keeping them all in RAM.
No occupation mapping, person exclusion, graph construction, or GPU use.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ATTRIBUTE_HEADER = b"source_line\tperson_name\tpredicate\traw_value"
PAIR_HEADER = b"source_line\taudit_tier\tsubject_name\tpredicate\tobject_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--relation-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def load_groups(path: Path) -> dict[str, set[bytes]]:
    groups = {"priority": set(), "semantic_review": set()}
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected relation candidate table header")
        for raw in handle:
            fields = raw.rstrip(b"\r\n").split(b"\t", 4)
            if len(fields) != 5 or not all(fields) or fields[1] not in (b"priority", b"semantic_review"):
                raise SystemExit("Malformed relation candidate row")
            _, tier, subject, _, obj = fields
            group = groups[show(tier)]
            group.add(subject)
            group.add(obj)
    groups["union"] = groups["priority"] | groups["semantic_review"]
    return groups


def main() -> None:
    args = parse_args()
    attributes_path = args.attributes.expanduser().resolve()
    pairs_path = args.relation_candidates.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (attributes_path, pairs_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    groups = load_groups(pairs_path)
    output.mkdir(parents=True, exist_ok=True)
    raw_pairs = output / "person_profession_pairs_unsorted.tmp"
    sorted_pairs = output / "person_profession_pairs_unique.tmp"
    fact_counts: Counter[bytes] = Counter()
    occupation_facts = 0
    attribute_rows = malformed = profession_facts = 0
    with attributes_path.open("rb") as attributes, raw_pairs.open("wb", buffering=1024 * 1024) as raw_out, \
            (output / "occupation_facts.tsv").open("wb") as occupations:
        if attributes.readline().rstrip(b"\r\n") != ATTRIBUTE_HEADER:
            raise SystemExit("Unexpected person attribute table header")
        occupations.write(ATTRIBUTE_HEADER + b"\n")
        for raw in attributes:
            attribute_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            _, person, predicate, value = fields
            if predicate == b"Profession":
                raw_out.write(person + b"\t" + value + b"\n")
                fact_counts[value] += 1
                profession_facts += 1
            elif predicate == b"Occupation":
                occupations.write(raw)
                occupation_facts += 1
            if attribute_rows % args.progress_every == 0:
                print(f"read {attribute_rows:,} attributes; {profession_facts:,} Profession facts",
                      file=sys.stderr, flush=True)

    sort_env = os.environ.copy()
    sort_env["LC_ALL"] = "C"
    print("deduplicating person/profession pairs with external sort", file=sys.stderr, flush=True)
    subprocess.run(["sort", "-u", "-T", str(output), "-o", str(sorted_pairs), str(raw_pairs)],
                   check=True, env=sort_env)
    raw_pairs.unlink()

    unique_value_people: Counter[bytes] = Counter()
    group_value_people: dict[str, Counter[bytes]] = {name: Counter() for name in groups}
    multiplicity: dict[str, Counter[int]] = {name: Counter() for name in ("all", *groups)}
    rng = random.Random(20260928)
    multi_examples: list[dict[str, object]] = []
    multi_people_seen = unique_pairs = unique_people = 0
    current_person: bytes | None = None
    current_values: list[bytes] = []

    def finish_person() -> None:
        nonlocal unique_people, multi_people_seen
        if current_person is None:
            return
        unique_people += 1
        count = len(current_values)
        multiplicity["all"][count] += 1
        for group_name, members in groups.items():
            if current_person in members:
                multiplicity[group_name][count] += 1
        if count > 1:
            multi_people_seen += 1
            position = (multi_people_seen - 1 if multi_people_seen <= 30
                        else rng.randrange(multi_people_seen))
            if position < 30:
                example = {"person_name": show(current_person),
                           "distinct_profession_values": [show(value) for value in current_values]}
                if multi_people_seen <= 30:
                    multi_examples.append(example)
                else:
                    multi_examples[position] = example

    with sorted_pairs.open("rb") as unique_in, \
            (output / "person_professions.tsv").open("wb", buffering=1024 * 1024) as final_pairs:
        final_pairs.write(b"person_name\tprofession_value\n")
        for raw in unique_in:
            person, separator, value = raw.rstrip(b"\r\n").partition(b"\t")
            if not separator or not person or not value:
                raise SystemExit("Malformed sorted person/profession pair")
            if person != current_person:
                finish_person()
                current_person = person
                current_values = []
            current_values.append(value)
            final_pairs.write(person + b"\t" + value + b"\n")
            unique_value_people[value] += 1
            for group_name, members in groups.items():
                if person in members:
                    group_value_people[group_name][value] += 1
            unique_pairs += 1
        finish_person()
    sorted_pairs.unlink()

    with (output / "profession_values.tsv").open("w", encoding="utf-8") as handle:
        handle.write("raw_value\tfacts\tunique_people\tpriority_people\treview_people\tunion_people\n")
        for value, count in fact_counts.most_common():
            handle.write(f"{show(value)}\t{count}\t{unique_value_people[value]}\t"
                         f"{group_value_people['priority'][value]}\t"
                         f"{group_value_people['semantic_review'][value]}\t"
                         f"{group_value_people['union'][value]}\n")
    with (output / "profession_multiplicity.tsv").open("w", encoding="utf-8") as handle:
        handle.write("distinct_values_per_person\tall_people\tpriority_people\treview_people\tunion_people\n")
        for count in sorted(multiplicity["all"]):
            handle.write(f"{count}\t{multiplicity['all'][count]}\t"
                         f"{multiplicity['priority'][count]}\t"
                         f"{multiplicity['semantic_review'][count]}\t"
                         f"{multiplicity['union'][count]}\n")
    (output / "multi_profession_examples.json").write_text(
        json.dumps(multi_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    group_coverage = {
        name: {
            "participants": len(members),
            "with_profession": sum(counter.values()),
            "with_one_distinct_profession": counter[1],
            "with_multiple_distinct_professions": sum(value for count, value in counter.items() if count > 1),
        }
        for name, members in groups.items()
        for counter in (multiplicity[name],)
    }
    summary = {
        "status": "complete_subset_audit" if malformed == 0 else "subset_audit_with_malformed_rows",
        "attributes": str(attributes_path),
        "relation_candidates": str(pairs_path),
        "attribute_rows_read": attribute_rows,
        "malformed_attribute_rows": malformed,
        "profession_facts": profession_facts,
        "unique_person_profession_pairs": unique_pairs,
        "duplicate_person_profession_facts": profession_facts - unique_pairs,
        "people_with_profession": unique_people,
        "distinct_profession_values": len(fact_counts),
        "occupation_facts_kept_separate": occupation_facts,
        "group_coverage": group_coverage,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Multi-profession counts use distinct raw values per person, not a normalized occupation ontology.",
            "Broad and narrow profession strings may overlap semantically; no model label mapping was applied.",
            "Occupation predicate facts remain separate because they are sparse and semantically unreviewed.",
            "No person or relation candidate was removed.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
