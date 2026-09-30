"""Stage 2: extract person-candidate pairs and reviewed social relations."""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from common import gzip_rows, token_for_uri, triples, tsv_writer, uri_from_token, write_json


PAIR_COLUMNS = [
    "source_line", "subject_uri", "predicate_uri", "object_uri",
    "subject_dbo_person", "object_dbo_person",
    "subject_uri_warning", "object_uri_warning",
]
EDGE_COLUMNS = PAIR_COLUMNS + ["social_group", "selection_scope"]


def load_policy(path: Path):
    rules = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            uri = row["predicate_uri"]
            if uri in rules:
                raise ValueError(f"Duplicate predicate policy: {uri}")
            if row["decision"] not in {"include", "exclude"}:
                raise ValueError(f"Invalid predicate decision: {uri}")
            rules[uri] = row
    if not rules:
        raise ValueError("Empty relation policy")
    return rules


def run(
    object_dump: Path,
    person_file: Path,
    policy_file: Path,
    output_dir: Path,
    allow_unreviewed: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    people = {}
    for row in gzip_rows(person_file):
        bits = int(row["dbo_person"]) | (2 if row["uri_warning"] else 0)
        people[token_for_uri(row["resource_uri"])] = bits
    rules = load_policy(policy_file)
    raw_counts, cleaned_counts = Counter(), Counter()
    excluded_counts, selected_raw_counts = Counter(), Counter()
    unknown_counts = Counter()
    removed = Counter()
    retained_people = set()
    exact_selected = set()
    scanned = malformed = raw_pairs = 0

    with gzip.open(output_dir / "raw_person_pairs.tsv.gz", "wt", encoding="utf-8", newline="") as raw_handle, gzip.open(
        output_dir / "social_edges.tsv.gz", "wt", encoding="utf-8", newline=""
    ) as social_handle:
        raw_writer = tsv_writer(raw_handle, PAIR_COLUMNS)
        social_writer = tsv_writer(social_handle, EDGE_COLUMNS)
        for line_no, subject, predicate, obj in triples(object_dump):
            scanned = line_no
            if subject is None:
                malformed += 1
                continue
            subject_bits, object_bits = people.get(subject), people.get(obj)
            if subject_bits is None or object_bits is None:
                continue
            raw_pairs += 1
            s, p, o = uri_from_token(subject), uri_from_token(predicate), uri_from_token(obj)
            raw_counts[p] += 1
            pair = {
                "source_line": line_no,
                "subject_uri": s,
                "predicate_uri": p,
                "object_uri": o,
                "subject_dbo_person": int(bool(subject_bits & 1)),
                "object_dbo_person": int(bool(object_bits & 1)),
                "subject_uri_warning": int(bool(subject_bits & 2)),
                "object_uri_warning": int(bool(object_bits & 2)),
            }
            raw_writer.writerow(pair)
            rule = rules.get(p)
            if rule is None:
                unknown_counts[p] += 1
                continue
            if rule["decision"] == "exclude":
                excluded_counts[rule["group_or_reason"]] += 1
                continue
            selected_raw_counts[p] += 1
            if s == o:
                removed["self_loops"] += 1
                continue
            key = (s, p, o)
            if key in exact_selected:
                removed["exact_duplicate_triples"] += 1
                continue
            exact_selected.add(key)
            cleaned_counts[p] += 1
            retained_people.update((s, o))
            social_writer.writerow({
                **pair,
                "social_group": rule["group_or_reason"],
                "selection_scope": rule["selection_scope"],
            })

    with (output_dir / "predicate_flow.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "predicate_uri", "decision", "group_or_reason", "selection_scope",
            "raw_person_pair_triples", "selected_before_dedup", "selected_clean",
        ])
        for predicate in sorted(raw_counts, key=lambda uri: (-raw_counts[uri], uri)):
            rule = rules.get(predicate)
            writer.writerow({
                "predicate_uri": predicate,
                "decision": rule["decision"] if rule else "unreviewed",
                "group_or_reason": rule["group_or_reason"] if rule else "unreviewed",
                "selection_scope": rule["selection_scope"] if rule else "",
                "raw_person_pair_triples": raw_counts[predicate],
                "selected_before_dedup": selected_raw_counts[predicate],
                "selected_clean": cleaned_counts[predicate],
            })
    with (output_dir / "unreviewed_predicates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["predicate_uri", "raw_person_pair_triples"])
        for predicate, n in unknown_counts.most_common():
            writer.writerow({"predicate_uri": predicate, "raw_person_pair_triples": n})

    result = {
        "stage": "02_relations",
        "object_dump": str(object_dump),
        "person_candidates": str(person_file),
        "relation_policy": str(policy_file),
        "object_rows_scanned": scanned,
        "malformed_object_rows": malformed,
        "raw_person_pair_triples": raw_pairs,
        "raw_person_pair_predicates": len(raw_counts),
        "excluded_by_reason": dict(excluded_counts),
        "unreviewed_predicate_triples": sum(unknown_counts.values()),
        "unreviewed_predicate_count": len(unknown_counts),
        "selected_before_dedup": sum(selected_raw_counts.values()),
        "removed_from_selected": dict(removed),
        "selected_social_edges": sum(cleaned_counts.values()),
        "selected_social_predicates": len(cleaned_counts),
        "selected_social_participants": len(retained_people),
        "selection_rule": "Reviewed social predicates only; currentMember and musical associated acts excluded; self-loops and identical directed s-p-o duplicates removed.",
    }
    if unknown_counts and not allow_unreviewed:
        raise ValueError(
            f"Found {len(unknown_counts)} unreviewed predicates; inspect "
            f"{output_dir / 'unreviewed_predicates.tsv'} and update relation_policy.tsv"
        )
    write_json(output_dir / "summary.json", result)
    return result
