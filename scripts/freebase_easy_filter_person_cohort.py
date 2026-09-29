#!/usr/bin/env python3
"""Explore an attribute-complete Person-name cohort using exported subsets only.

The main cohort has a raw Profession fact and at least one parseable birth or
death year. A stricter birth-year cohort is reported for comparison. All raw
pair predicates are retained, including false name matches with type names.
No Freebase ID or relation-semantic validation is implied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

from freebase_easy_audit_dates import BIRTH, DEATH, parse_year


PAIR_HEADER = b"source_line\tsubject_name\tpredicate\tobject_name"
ATTRIBUTE_HEADER = b"source_line\tperson_name\tpredicate\traw_value"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="person_pairs_raw_v1/person_name_pairs.tsv")
    parser.add_argument("--attributes", type=Path, required=True,
                        help="person_attributes_v1/person_attributes.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-predicate", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    result = parser.parse_args()
    if result.samples_per_predicate < 0 or result.progress_every < 1:
        parser.error("--samples-per-predicate must be nonnegative and --progress-every positive")
    return result


def display(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def main() -> None:
    options = args()
    pair_path = options.pairs.expanduser().resolve()
    attribute_path = options.attributes.expanduser().resolve()
    output = options.output_dir.expanduser().resolve()
    for path in (pair_path, attribute_path):
        if not path.is_file():
            raise SystemExit(f"Input file does not exist: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    participants: set[bytes] = set()
    first_pass_rows = malformed_pairs = 0
    with pair_path.open("rb") as pairs:
        if pairs.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected pair header")
        for raw in pairs:
            first_pass_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed_pairs += 1
                continue
            participants.add(fields[1])
            participants.add(fields[3])
    print(f"loaded {len(participants):,} raw-pair participant names", file=sys.stderr, flush=True)

    profession: set[bytes] = set()
    birth: set[bytes] = set()
    death: set[bytes] = set()
    attribute_rows = malformed_attributes = unparsed_date_facts = 0
    with attribute_path.open("rb") as attributes:
        if attributes.readline().rstrip(b"\r\n") != ATTRIBUTE_HEADER:
            raise SystemExit("Unexpected attribute header")
        for raw in attributes:
            attribute_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed_attributes += 1
                continue
            _, person, predicate, value = fields
            if person not in participants:
                continue
            if predicate == b"Profession":
                profession.add(person)
            elif predicate in BIRTH or predicate in DEATH:
                year, _, _ = parse_year(value)
                if year is None:
                    unparsed_date_facts += 1
                elif predicate in BIRTH:
                    birth.add(person)
                else:
                    death.add(person)
            if attribute_rows % options.progress_every == 0:
                print(f"read {attribute_rows:,} attribute rows", file=sys.stderr, flush=True)

    dated = birth | death
    cohort = profession & dated
    birth_cohort = profession & birth
    print(f"attribute-complete cohort: {len(cohort):,} names", file=sys.stderr, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "cohort_person_names.txt").open("wb") as handle:
        for person in sorted(cohort):
            handle.write(person + b"\n")

    predicate_counts: dict[bytes, Counter[str]] = defaultdict(Counter)
    sampled: dict[bytes, list[tuple[bytes, bytes, bytes, bytes]]] = defaultdict(list)
    rng = random.Random(20260929)
    endpoints_in_filtered_pairs: set[bytes] = set()
    second_pass_rows = filtered_rows = birth_filtered_rows = 0
    filtered_partial = output / "cohort_name_pairs.tsv.partial"
    filtered_final = output / "cohort_name_pairs.tsv"
    with pair_path.open("rb") as pairs, filtered_partial.open("wb", buffering=1024 * 1024) as selected:
        if pairs.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected pair header on second pass")
        selected.write(PAIR_HEADER + b"\n")
        for raw in pairs:
            second_pass_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                continue
            source_line, subject, predicate, obj = fields
            item = predicate_counts[predicate]
            item["all_raw_pairs"] += 1
            subject_ok, object_ok = subject in cohort, obj in cohort
            if subject_ok or object_ok:
                item["at_least_one_in_cohort"] += 1
            if subject_ok and object_ok:
                item["both_in_cohort"] += 1
                filtered_rows += 1
                selected.write(raw)
                endpoints_in_filtered_pairs.add(subject)
                endpoints_in_filtered_pairs.add(obj)
                if options.samples_per_predicate:
                    examples = sampled[predicate]
                    if len(examples) < options.samples_per_predicate:
                        examples.append((source_line, subject, predicate, obj))
                    else:
                        index = rng.randrange(item["both_in_cohort"])
                        if index < options.samples_per_predicate:
                            examples[index] = (source_line, subject, predicate, obj)
            if subject in birth_cohort and obj in birth_cohort:
                item["both_in_birth_cohort"] += 1
                birth_filtered_rows += 1
    filtered_partial.replace(filtered_final)
    if first_pass_rows != second_pass_rows:
        raise SystemExit("Input pair file changed between passes")

    with (output / "predicate_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tall_raw_pairs\tat_least_one_in_cohort\t"
                     "both_in_cohort\tboth_in_birth_cohort\n")
        for predicate, item in sorted(predicate_counts.items(),
                                      key=lambda pair: (-pair[1]["both_in_cohort"], pair[0])):
            handle.write(display(predicate) + "\t" + "\t".join(str(item[column]) for column in (
                "all_raw_pairs", "at_least_one_in_cohort", "both_in_cohort",
                "both_in_birth_cohort")) + "\n")
    samples = {
        display(predicate): [
            {"source_line": int(line), "subject_name": display(subject),
             "predicate": display(relation), "object_name": display(obj)}
            for line, subject, relation, obj in examples
        ]
        for predicate, examples in sorted(sampled.items())
    }
    (output / "predicate_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete_subset_filter" if malformed_pairs == 0 and malformed_attributes == 0 else "input_malformed_rows_present",
        "pairs": str(pair_path), "attributes": str(attribute_path),
        "filter_rule": "raw Profession fact and at least one structurally parsed birth or death year",
        "raw_pair_rows_read": first_pass_rows,
        "malformed_pair_rows": malformed_pairs,
        "attribute_rows_read": attribute_rows,
        "malformed_attribute_rows": malformed_attributes,
        "unparsed_date_facts_for_pair_participants": unparsed_date_facts,
        "raw_pair_participant_names": len(participants),
        "participants_with_profession": len(profession),
        "participants_with_parsed_birth_year": len(birth),
        "participants_with_parsed_death_year": len(death),
        "cohort_names_with_profession_and_birth_or_death": len(cohort),
        "birth_only_cohort_names_with_profession_and_birth": len(birth_cohort),
        "filtered_raw_pair_rows": filtered_rows,
        "filtered_raw_pair_participants": len(endpoints_in_filtered_pairs),
        "birth_only_filtered_raw_pair_rows": birth_filtered_rows,
        "predicates_with_filtered_raw_pairs": sum(item["both_in_cohort"] > 0 for item in predicate_counts.values()),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "All predicates are retained, including type/occupation/name collisions; relation review comes next.",
            "Names with a Profession and parsed year are provisional, not validated human identities or mapped occupation labels.",
            "A parsed year is structural evidence, not a verified historical birth/death date.",
            "The birth-only comparison is stricter and is not used to write cohort_name_pairs.tsv.",
            "The original pair and attribute exports are never changed.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
