"""Build a fresh CBDB person-pair table from the original SQLite database."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "external_data/cbdb/2026.09.14/cbdb_20260914.sqlite3"
DEFAULT_RULES = ROOT / "config/cbdb_occupation_whitelist_v1.json"
DEFAULT_OUT = ROOT / "external_data/cbdb/2026.09.14/flat_triples_v1"
MAX_YEAR = 2026  # Release year; catches obvious five-digit birth-year errors.

FIELDS = [
    "Node1", "Relation", "Node2", "RelationGroup", "RelationLabelZh",
    "Node1_NameZh", "Node1_Birth", "Node1_Death", "Node1_Occupation",
    "Node1_OccupationSource", "Node2_NameZh", "Node2_Birth", "Node2_Death",
    "Node2_Occupation", "Node2_OccupationSource", "SupportRows",
]


def load_people(conn: sqlite3.Connection) -> tuple[dict[int, tuple], Counter]:
    people = {}
    counts = Counter()
    for person_id, name, birth, death in conn.execute(
        "SELECT c_personid,c_name_chn,c_birthyear,c_deathyear FROM BIOG_MAIN"
    ):
        counts["all_people"] += 1
        birth = birth if birth not in (None, 0) else None
        death = death if death not in (None, 0) else None
        if birth is None and death is None:
            counts["no_literal_birth_or_death"] += 1
            continue
        if (birth is not None and birth > MAX_YEAR) or (
            death is not None and death > MAX_YEAR
        ) or (birth is not None and death is not None and birth > death):
            counts["invalid_birth_or_death"] += 1
            continue
        people[person_id] = (name or "", birth, death)
    counts["people_with_usable_birth_or_death"] = len(people)
    return people, counts


def load_occupations(
    conn: sqlite3.Connection, people: dict[int, tuple], rules_path: Path
) -> tuple[dict[int, tuple[str, str]], Counter]:
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    code_to_label = {}
    for label, codes in rules.items():
        for code in codes:
            if code in code_to_label:
                raise ValueError(f"STATUS code {code} mapped twice")
            code_to_label[code] = label

    official_ids = {
        person_id
        for (person_id,) in conn.execute(
            "SELECT DISTINCT c_personid FROM POSTED_TO_OFFICE_DATA "
            "WHERE c_personid>0 AND c_office_id>0"
        )
    }
    status_labels: dict[int, set[str]] = defaultdict(set)
    for person_id, code in conn.execute(
        "SELECT c_personid,c_status_code FROM STATUS_DATA WHERE c_personid>0"
    ):
        label = code_to_label.get(code)
        if label:
            status_labels[person_id].add(label)

    occupations = {}
    counts = Counter()
    for person_id in people:
        labels = status_labels.get(person_id, set())
        if person_id in official_ids:
            occupations[person_id] = ("做官", "POSTED_TO_OFFICE_DATA")
        elif "做官" in labels:
            occupations[person_id] = ("做官", "STATUS_DATA")
        elif len(labels) == 1:
            occupations[person_id] = (next(iter(labels)), "STATUS_DATA")
        elif len(labels) > 1:
            counts["ambiguous_nonofficial_status_people"] += 1
        if person_id in occupations:
            counts[f"occupation_{occupations[person_id][0]}"] += 1
    counts["people_with_date_and_clear_occupation"] = len(occupations)
    counts["people_with_date_but_no_clear_occupation"] = (
        len(people) - len(occupations)
    )
    return occupations, counts


def collect_triples(
    conn: sqlite3.Connection, eligible: set[int]
) -> tuple[Counter[tuple[str, int, int, int]], Counter]:
    triples: Counter[tuple[str, int, int, int]] = Counter()
    counts = Counter()
    for source, target, code in conn.execute(
        "SELECT c_personid,c_kin_id,c_kin_code FROM KIN_DATA"
    ):
        counts["kinship_raw_rows"] += 1
        if source <= 0 or target <= 0 or source == target or code <= 0:
            counts["kinship_invalid_or_untyped_rows"] += 1
        elif source not in eligible or target not in eligible:
            counts["kinship_missing_endpoint_requirements"] += 1
        else:
            triples[("KIN", code, source, target)] += 1
            counts["kinship_eligible_rows"] += 1

    for source, target, code, source_kin, source_kin_id, target_kin, target_kin_id, tertiary in conn.execute(
        "SELECT c_personid,c_assoc_id,c_assoc_code,c_kin_code,c_kin_id,"
        "c_assoc_kin_code,c_assoc_kin_id,c_tertiary_personid FROM ASSOC_DATA"
    ):
        counts["association_raw_rows"] += 1
        if any(value not in (None, 0) for value in (
            source_kin, source_kin_id, target_kin, target_kin_id, tertiary
        )):
            counts["association_indirect_or_tertiary_rows"] += 1
            continue
        if source <= 0 or target <= 0 or source == target or code <= 0:
            counts["association_invalid_or_untyped_rows"] += 1
        elif source not in eligible or target not in eligible:
            counts["association_missing_endpoint_requirements"] += 1
        else:
            triples[("ASSOC", code, source, target)] += 1
            counts["association_eligible_rows"] += 1
    counts["distinct_directed_typed_triples"] = len(triples)
    counts["collapsed_duplicate_source_rows"] = (
        counts["kinship_eligible_rows"] + counts["association_eligible_rows"]
        - len(triples)
    )
    return triples, counts


def export(
    conn: sqlite3.Connection, output_dir: Path,
    people: dict[int, tuple], occupations: dict[int, tuple[str, str]],
    triples: Counter[tuple[str, int, int, int]], summary: Counter,
) -> None:
    kin_names = dict(conn.execute(
        "SELECT c_kincode,c_kinrel_chn FROM KINSHIP_CODES"
    ))
    assoc_names = dict(conn.execute(
        "SELECT c_assoc_code,c_assoc_desc_chn FROM ASSOC_CODES"
    ))
    assoc_groups = dict(conn.execute(
        "SELECT c_assoc_code,substr(c_assoc_type_code,1,2) "
        "FROM ASSOC_CODE_TYPE_REL"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "person_relation_triples.tsv.gz"
    group_counts = Counter()
    output_nodes = set()
    official_official = 0
    with gzip.open(target, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(FIELDS)
        for (kind, code, source, target_id), support in sorted(triples.items()):
            source_name, source_birth, source_death = people[source]
            target_name, target_birth, target_death = people[target_id]
            source_occupation, source_evidence = occupations[source]
            target_occupation, target_evidence = occupations[target_id]
            output_nodes.update((source, target_id))
            if source_occupation == target_occupation == "做官":
                official_official += 1
            group = "kinship" if kind == "KIN" else (
                "association_" + assoc_groups.get(code, "unclassified")
            )
            group_counts[group] += 1
            writer.writerow([
                f"CBDB:{source}", f"{kind}:{code}", f"CBDB:{target_id}",
                group, (kin_names if kind == "KIN" else assoc_names).get(code, ""),
                source_name, source_birth if source_birth is not None else "",
                source_death if source_death is not None else "", source_occupation,
                source_evidence, target_name,
                target_birth if target_birth is not None else "",
                target_death if target_death is not None else "", target_occupation,
                target_evidence, support,
            ])
    summary["triple_relation_groups"] = dict(sorted(group_counts.items()))
    summary["output_unique_people"] = len(output_nodes)
    summary["output_node_occupations"] = dict(sorted(Counter(
        occupations[person_id][0] for person_id in output_nodes
    ).items()))
    summary["output_official_official_triples"] = official_official
    summary["output"] = str(target)
    (output_dir / "summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--occupation-rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    conn = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    try:
        people, person_counts = load_people(conn)
        occupations, occupation_counts = load_occupations(
            conn, people, args.occupation_rules
        )
        triples, triple_counts = collect_triples(conn, set(occupations))
        summary = Counter()
        summary.update(person_counts)
        summary.update(occupation_counts)
        summary.update(triple_counts)
        export(conn, args.output_dir, people, occupations, triples, summary)
        print(json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
