#!/usr/bin/env python3
"""Build a Q_R_Q-style Freebase Easy edge export from the strict cohort.

Reads only previously exported pair, attribute, and name-to-ID tables. The
result is exploratory and does not invent L1/L2/L3 occupation labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

from freebase_easy_audit_dates import BIRTH, DEATH, parse_year


PAIR_HEADER = b"source_line\tsubject_name\tpredicate\tobject_name"
ATTRIBUTE_HEADER = b"source_line\tperson_name\tpredicate\traw_value"
STATUS_HEADER = b"name\tstatus\tmid\tdistinct_mid_count"
WIDE_COLUMNS = [
    "Node1", "Relation", "Node2", "RelationGroup", "Node1_Name", "Node2_Name",
    "Node1_Birth", "Node1_Death", "Node1_Occupation", "Node2_Birth",
    "Node2_Death", "Node2_Occupation", "SourcePredicates", "SourceLines",
    "SupportRows",
]


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="strict-cohort cohort_name_pairs.tsv")
    parser.add_argument("--attributes", type=Path, required=True,
                        help="person_attributes_v1/person_attributes.tsv")
    parser.add_argument("--name-mids", type=Path, required=True,
                        help="name_mid_audit_v3/name_mid_status.tsv")
    parser.add_argument("--config", type=Path, required=True,
                        help="config/freebase_easy_relation_export_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rules(path: Path) -> dict[bytes, dict[str, object]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    normalised = {}
    for entry in config["relations"]:
        predicate = entry["predicate"].encode("utf-8")
        if predicate in result:
            raise SystemExit(f"Duplicate predicate in config: {entry['predicate']}")
        relation = entry["relation"]
        symmetric = entry["symmetric"]
        if relation in normalised and normalised[relation] != (entry["group"], symmetric):
            raise SystemExit(f"Inconsistent definition of relation: {relation}")
        normalised[relation] = (entry["group"], symmetric)
        result[predicate] = entry
    if not result:
        raise SystemExit("Relation config is empty")
    return result


def parse_rows(path: Path, expected_header: bytes, fields: int):
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != expected_header:
            raise SystemExit(f"Unexpected header in {path}")
        for lineno, raw in enumerate(handle, 2):
            values = raw.rstrip(b"\r\n").split(b"\t", fields - 1)
            required = (0, 1, 3) if expected_header == STATUS_HEADER else tuple(range(fields))
            if len(values) != fields or any(not values[index] for index in required):
                raise SystemExit(f"Malformed row in {path} at line {lineno}")
            yield values


def decoded(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def json_cell(values) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = options()
    paths = {
        "pairs": args.pairs.expanduser().resolve(),
        "attributes": args.attributes.expanduser().resolve(),
        "name_mids": args.name_mids.expanduser().resolve(),
        "config": args.config.expanduser().resolve(),
    }
    output = args.output_dir.expanduser().resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name}: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty: {output}")
    started = time.monotonic()
    rules = read_rules(paths["config"])

    raw_selected = []
    names: set[bytes] = set()
    pair_rows = 0
    for source_line, subject, predicate, obj in parse_rows(paths["pairs"], PAIR_HEADER, 4):
        if not source_line.isdigit():
            raise SystemExit("Non-numeric source_line in pair table")
        pair_rows += 1
        if predicate in rules:
            raw_selected.append((source_line, subject, predicate, obj))
            names.add(subject)
            names.add(obj)
    print(f"selected {len(raw_selected):,} of {pair_rows:,} raw pair rows", file=sys.stderr, flush=True)

    ids: dict[bytes, bytes] = {}
    status_found: set[bytes] = set()
    for name, label, mid, _ in parse_rows(paths["name_mids"], STATUS_HEADER, 4):
        if name not in names:
            continue
        if name in status_found:
            raise SystemExit(f"Duplicate name status: {decoded(name)}")
        status_found.add(name)
        if label == b"unique" and mid:
            ids[name] = mid
        elif label not in (b"ambiguous", b"missing") or mid:
            raise SystemExit(f"Invalid name status: {decoded(name)}")
    id_names: dict[bytes, set[bytes]] = defaultdict(set)
    for name, mid in ids.items():
        id_names[mid].add(name)
    shared_name_ids = {mid for mid, mapped_names in id_names.items() if len(mapped_names) > 1}

    professions: dict[bytes, set[bytes]] = defaultdict(set)
    births: dict[bytes, set[int]] = defaultdict(set)
    deaths: dict[bytes, set[int]] = defaultdict(set)
    attribute_rows = 0
    for source_line, name, predicate, value in parse_rows(paths["attributes"], ATTRIBUTE_HEADER, 4):
        if not source_line.isdigit() or not value:
            raise SystemExit("Malformed attribute source_line or value")
        attribute_rows += 1
        if name not in names:
            continue
        if predicate == b"Profession":
            professions[name].add(value)
        elif predicate in BIRTH or predicate in DEATH:
            year, _, _ = parse_year(value)
            if year is not None:
                (births if predicate in BIRTH else deaths)[name].add(year)
    print(f"read {attribute_rows:,} attribute rows", file=sys.stderr, flush=True)

    def attr_issue(name: bytes) -> str | None:
        if not professions[name] or not (births[name] or deaths[name]):
            return "endpoint_attribute_missing"
        if len(professions[name]) > 1:
            return "endpoint_multiple_professions"
        if len(births[name]) > 1 or len(deaths[name]) > 1:
            return "endpoint_conflicting_dates"
        return None

    pre_exclusion: dict[int, str] = {}
    child_directions: dict[tuple[bytes, bytes], set[tuple[bytes, bytes]]] = defaultdict(set)
    kinship_types: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    advisor_directions: dict[tuple[bytes, bytes], set[tuple[bytes, bytes]]] = defaultdict(set)
    for index, (source_line, subject, predicate, obj) in enumerate(raw_selected):
        sub_id, obj_id = ids.get(subject), ids.get(obj)
        if not sub_id or not obj_id:
            pre_exclusion[index] = "endpoint_id_unresolved"
            continue
        if sub_id in shared_name_ids or obj_id in shared_name_ids:
            pre_exclusion[index] = "id_shared_by_multiple_names"
            continue
        if sub_id == obj_id:
            pre_exclusion[index] = "self_link"
            continue
        if issue := attr_issue(subject) or attr_issue(obj):
            pre_exclusion[index] = issue
            continue
        rule = rules[predicate]
        pair = tuple(sorted((sub_id, obj_id)))
        relation = str(rule["relation"])
        if relation in ("child", "sibling", "partner"):
            kinship_types[pair].add(relation)
            if relation == "child":
                child_directions[pair].add((sub_id, obj_id))
        if relation == "academic_advisor_raw":
            advisor_directions[pair].add((sub_id, obj_id))

    conflict_kinship = {
        pair for pair, types in kinship_types.items()
        if len(types) > 1 or len(child_directions[pair]) > 1
    }
    conflict_advisor = {
        pair for pair, directions in advisor_directions.items() if len(directions) > 1
    }
    print(f"flagged {len(conflict_kinship):,} kinship and {len(conflict_advisor):,} advisor pairs",
          file=sys.stderr, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    decisions: Counter[str] = Counter()
    edges: dict[tuple[str, bytes, bytes], list[tuple[bytes, bytes]]] = defaultdict(list)
    edge_groups: dict[str, str] = {}
    id_to_name: dict[bytes, bytes] = {}
    with (output / "row_audit.tsv").open("wb") as audit:
        audit.write(b"source_line\tsubject_name\tpredicate\tobject_name\tdecision\n")
        for index, (source_line, subject, predicate, obj) in enumerate(raw_selected):
            decision = pre_exclusion.get(index)
            if decision is None:
                sub_id, obj_id = ids[subject], ids[obj]
                rule = rules[predicate]
                relation = str(rule["relation"])
                pair = tuple(sorted((sub_id, obj_id)))
                if relation in ("child", "sibling", "partner") and pair in conflict_kinship:
                    decision = "kinship_pair_conflict"
                elif relation == "academic_advisor_raw" and pair in conflict_advisor:
                    decision = "advisor_reciprocal_conflict"
                else:
                    if rule.get("reverse", False):
                        sub_id, obj_id = obj_id, sub_id
                        subject, obj = obj, subject
                    if rule["symmetric"] and sub_id > obj_id:
                        sub_id, obj_id = obj_id, sub_id
                        subject, obj = obj, subject
                    for entity_id, name in ((sub_id, subject), (obj_id, obj)):
                        old = id_to_name.setdefault(entity_id, name)
                        if old != name:
                            raise SystemExit(f"Distinct names mapped to one ID: {decoded(entity_id)}")
                    key = (relation, sub_id, obj_id)
                    edges[key].append((source_line, predicate))
                    edge_groups[relation] = str(rule["group"])
                    decision = "included_support"
            decisions[decision] += 1
            audit.write(b"\t".join((source_line, raw_selected[index][1], predicate,
                                    raw_selected[index][3], decision.encode("ascii"))) + b"\n")

    def node_details(entity_id: bytes) -> tuple[str, str, str, str, str]:
        name = id_to_name[entity_id]
        birth = next(iter(births[name])) if births[name] else ""
        death = next(iter(deaths[name])) if deaths[name] else ""
        occupation = decoded(next(iter(professions[name])))
        warnings = []
        if isinstance(birth, int) and isinstance(death, int):
            if death < birth:
                warnings.append("death_before_birth")
            elif death - birth > 125:
                warnings.append("age_over_125")
        return decoded(name), str(birth), str(death), occupation, json_cell(warnings)

    edge_nodes = {entity_id for _, left, right in edges for entity_id in (left, right)}
    with (output / "nodes_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("NodeID", "Name", "Birth", "Death", "Occupation", "DateWarnings"))
        for entity_id in sorted(edge_nodes):
            writer.writerow((decoded(entity_id), *node_details(entity_id)))
    with (output / "Q_R_Q_extended_freebase_v1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(WIDE_COLUMNS)
        for (relation, left, right), support in sorted(edges.items()):
            left_name, left_birth, left_death, left_prof, _ = node_details(left)
            right_name, right_birth, right_death, right_prof, _ = node_details(right)
            writer.writerow((
                decoded(left), relation, decoded(right), edge_groups[relation],
                left_name, right_name, left_birth, left_death, left_prof,
                right_birth, right_death, right_prof,
                json_cell(sorted({decoded(predicate) for _, predicate in support})),
                json_cell(sorted(int(line) for line, _ in support)), len(support),
            ))

    per_relation_edges = Counter(relation for relation, _, _ in edges)
    per_group_edges = Counter(edge_groups[relation] for relation, _, _ in edges)
    summary = {
        "status": "complete_export",
        "inputs": {name: str(path) for name, path in paths.items()},
        "raw_pair_rows_read": pair_rows,
        "selected_raw_pair_rows": len(raw_selected),
        "excluded_by_predicate_rows": pair_rows - len(raw_selected),
        "attribute_rows_read": attribute_rows,
        "selected_endpoint_names": len(names),
        "selected_endpoint_names_with_unique_id": len(ids),
        "selected_endpoint_names_without_status": len(names - status_found),
        "link_ids_shared_by_multiple_selected_names": len(shared_name_ids),
        "conflicting_kinship_unordered_id_pairs": len(conflict_kinship),
        "reciprocal_academic_advisor_unordered_id_pairs": len(conflict_advisor),
        "row_decisions": dict(decisions),
        "exported_unique_edges": len(edges),
        "exported_unique_nodes": len(edge_nodes),
        "edge_counts_by_relation": dict(sorted(per_relation_edges.items())),
        "edge_counts_by_group": dict(sorted(per_group_edges.items())),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Node1/Node2 are link-file Freebase IDs, provisional evidence for the original Easy name facts.",
            "Occupation is the sole raw Profession value for each exported node; people with multiple values are excluded and counted in row_decisions.",
            "Symmetric relations are written once per unordered ID pair; directed relations retain their raw direction except the configured inverse philosophy predicate.",
            "Conflicting kinship pairs and reciprocal Academic advisor pairs are omitted from this exploratory export and listed in row_audit.tsv.",
            "This CSV is Q_R_Q-like, not directly compatible with the old L1/L2/L3 training loader.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
