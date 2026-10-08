#!/usr/bin/env python3
"""Export date-qualified CBDB people with all distinct recorded office titles."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cbdb_build_flat_triples import (  # noqa: E402
    DEFAULT_DB, DEFAULT_RULES, MAX_YEAR, load_occupations, load_people,
)

DEFAULT_OUT = ROOT / "external_data/cbdb/2026.09.14/detailed_occupations_v1"
UNKNOWN_NAMES = {
    "未詳", "未详", "不詳", "不详", "未知", "不明", "[未詳]", "[未详]",
    "unknown", "[unknown]", "[missing data]",
}
BASE_FIELDS = [
    "Node", "PersonID", "NameZh", "Birth", "Death", "OccupationCount",
    "SpecificOfficeCount", "OfficeCodeCount", "UnknownOfficeCodeCount",
    "OccupationSource", "OccupationResolution", "LegacyBroadOccupation",
    "OfficeCodes", "StatusOccupationCodes",
]
LONG_FIELDS = [
    "Node", "PersonID", "OccSlot", "OccupationCode", "OccupationNameZh",
    "OccupationKind", "Source", "OfficeID", "OfficeDynastyCode",
    "OfficeDynastyNameZh", "SupportRows", "EvidenceStatusCodes",
]


@dataclass(frozen=True)
class Occupation:
    code: str
    label: str
    kind: str
    source: str
    support: int
    office_id: int | None = None
    dynasty: int | None = None
    dynasty_name: str = ""
    status_codes: tuple[int, ...] = ()


def office_name_status(name: str | None, code_exists: bool = True) -> str:
    if not code_exists:
        return "office_code_not_found"
    if not name or not name.strip():
        return "office_name_empty"
    if name.strip().lower() in UNKNOWN_NAMES:
        return "office_name_unknown"
    return "specific_office"


def load_status_evidence(conn, eligible, rules_path):
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    mapping = {}
    for label, codes in rules.items():
        for code in codes:
            if code in mapping:
                raise ValueError(f"STATUS code {code} mapped twice")
            mapping[code] = label
    evidence = defaultdict(Counter)
    for person, code in conn.execute("SELECT c_personid,c_status_code FROM STATUS_DATA"):
        if person in eligible and code in mapping:
            evidence[person][code] += 1
    return evidence, mapping


def extract(conn, people, rules_path):
    """Reuse the old eligibility policy; replace the official broad label by titles."""
    legacy, legacy_counts = load_occupations(conn, people, Path(rules_path))
    status_evidence, status_mapping = load_status_evidence(conn, people, rules_path)
    dynasties = dict(conn.execute("SELECT c_dy,c_dynasty_chn FROM DYNASTIES"))
    office_codes = {
        row[0]: {"dynasty": row[1], "name": row[2] or ""}
        for row in conn.execute("SELECT c_office_id,c_dy,c_office_chn FROM OFFICE_CODES")
    }
    offices = defaultdict(dict)
    for person, office, support in conn.execute(
        "SELECT c_personid,c_office_id,count(*) FROM POSTED_TO_OFFICE_DATA "
        "WHERE c_personid>0 AND c_office_id>0 GROUP BY c_personid,c_office_id"
    ):
        if person in people:
            offices[person][office] = support

    occupations = {}
    for person in sorted(people):
        entries = []
        for office, support in sorted(offices.get(person, {}).items()):
            info = office_codes.get(office, {})
            if office_name_status(info.get("name"), office in office_codes) != "specific_office":
                continue
            entries.append(Occupation(
                code=f"OFFICE:{office}", label=info["name"].strip(),
                kind="specific_office", source="POSTED_TO_OFFICE_DATA",
                support=support, office_id=office, dynasty=info["dynasty"],
                dynasty_name=dynasties.get(info["dynasty"], "") or "",
            ))
        if not entries and person in legacy:
            label, source = legacy[person]
            codes = tuple(sorted(
                code for code in status_evidence.get(person, {})
                if status_mapping[code] == label
            ))
            is_unknown_office = bool(offices.get(person))
            entries = [Occupation(
                code=("OFFICE_UNKNOWN:" if is_unknown_office else "STATUS_CLASS:") + label,
                label=label,
                kind="office_unknown_broad" if is_unknown_office else "status_broad",
                source=source,
                support=(sum(offices[person].values()) if is_unknown_office else
                         sum(status_evidence[person][code] for code in codes)),
                status_codes=codes,
            )]
        if entries:
            occupations[person] = entries
    if set(occupations) != set(legacy):
        raise ValueError("Detailed export unexpectedly changed occupation eligibility")
    return {
        "occupations": occupations, "offices": offices, "office_codes": office_codes,
        "dynasties": dynasties, "legacy": legacy, "legacy_counts": legacy_counts,
        "status_evidence": status_evidence, "status_mapping": status_mapping,
    }


def person_base(person, people, data):
    name, birth, death = people[person]
    entries = data["occupations"].get(person, [])
    offices = data["offices"].get(person, {})
    specific = sum(e.kind == "specific_office" for e in entries)
    status_codes = data["status_evidence"].get(person, {})
    if entries:
        resolution = "specific_office" if specific else entries[0].kind
        source = entries[0].source
    else:
        labels = {data["status_mapping"][code] for code in status_codes}
        resolution = "ambiguous_nonofficial_status" if len(labels) > 1 else "no_occupation"
        source = ""
    return [
        f"CBDB:{person}", person, name, birth, death, len(entries), specific,
        len(offices), len(offices) - specific, source, resolution,
        data["legacy"].get(person, ("", ""))[0],
        ";".join(f"OFFICE:{code}" for code in sorted(offices)),
        ";".join(f"STATUS:{code}" for code in sorted(status_codes)),
    ]


def write_csv(path, fields, rows):
    # BOM permits direct Chinese CSV viewing in Excel without changing values.
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def export_posting_evidence(conn, output, eligible):
    cursor = conn.execute(
        "SELECT * FROM POSTED_TO_OFFICE_DATA WHERE c_personid>0 AND c_office_id>0 "
        "ORDER BY c_personid,c_office_id,c_posting_id"
    )
    fields = [column[0] for column in cursor.description]
    person_index = fields.index("c_personid")
    count = 0
    with gzip.open(output / "office_postings_source.tsv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(fields)
        for row in cursor:
            if row[person_index] in eligible:
                writer.writerow(row)
                count += 1
    return count


def write_exports(conn, output, people, person_counts, data, database, rules_path):
    output = Path(output)
    occupations, offices, office_codes = (data[k] for k in ("occupations", "offices", "office_codes"))
    slots = max((len(entries) for entries in occupations.values()), default=0)
    fields = BASE_FIELDS + [field for i in range(1, slots + 1) for field in (f"occ{i}", f"occ{i}_code")]

    def wide_rows(specific_only=False):
        for person, entries in sorted(occupations.items()):
            if specific_only and entries[0].kind != "specific_office":
                continue
            row = person_base(person, people, data)
            row += [value for e in entries for value in (e.label, e.code)]
            row += [""] * (2 * (slots - len(entries)))
            yield row

    write_csv(output / "people_with_occupations.csv", fields, wide_rows())
    write_csv(output / "people_with_specific_offices.csv", fields, wide_rows(True))
    write_csv(output / "date_filtered_people.csv", BASE_FIELDS,
              (person_base(person, people, data) for person in sorted(people)))
    write_csv(output / "officials_without_specific_office.csv", BASE_FIELDS,
              (person_base(person, people, data) for person, entries in sorted(occupations.items())
               if entries[0].kind != "specific_office" and entries[0].label == "做官"))

    entry_count = 0
    with gzip.open(output / "person_occupations_long.tsv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(LONG_FIELDS)
        for person, entries in sorted(occupations.items()):
            for slot, e in enumerate(entries, 1):
                writer.writerow([
                    f"CBDB:{person}", person, slot, e.code, e.label, e.kind, e.source,
                    e.office_id, e.dynasty, e.dynasty_name, e.support,
                    ";".join(f"STATUS:{code}" for code in e.status_codes),
                ])
                entry_count += 1

    office_people, office_rows = Counter(), Counter()
    for person_offices in offices.values():
        office_people.update(person_offices.keys())
        office_rows.update(person_offices)
    write_csv(output / "office_inventory.csv", [
        "OfficeCode", "OfficeID", "OfficeNameZh", "OfficeDynastyCode",
        "OfficeDynastyNameZh", "NameStatus", "DateQualifiedPeople", "SourceRows",
    ], (
        [f"OFFICE:{office}", office, office_codes.get(office, {}).get("name", ""),
         office_codes.get(office, {}).get("dynasty"),
         data["dynasties"].get(office_codes.get(office, {}).get("dynasty"), ""),
         office_name_status(office_codes.get(office, {}).get("name"), office in office_codes),
         office_people[office], office_rows[office]]
        for office in sorted(office_people)
    ))
    evidence_rows = export_posting_evidence(conn, output, people)
    if evidence_rows != sum(office_rows.values()):
        raise ValueError("Posting evidence count does not match aggregated source rows")

    kinds = Counter(entries[0].kind for entries in occupations.values())
    known_pairs = sum(e.kind == "specific_office" for entries in occupations.values() for e in entries)
    known_codes = {e.office_id for entries in occupations.values() for e in entries if e.office_id is not None}
    unknown_offices = {office for office in office_people
                       if office_name_status(office_codes.get(office, {}).get("name"), office in office_codes) != "specific_office"}
    histogram = Counter(len(entries) for entries in occupations.values())
    specific_histogram = Counter(len(entries) for entries in occupations.values() if entries[0].kind == "specific_office")
    broad_only_officials = sum(entries[0].label == "做官" and entries[0].kind != "specific_office"
                               for entries in occupations.values())
    summary = {
        "schema_version": "cbdb_detailed_occupations_v1",
        "database": str(Path(database).resolve()), "occupation_rules": str(Path(rules_path).resolve()),
        "date_filter": {**dict(person_counts), "max_year": MAX_YEAR},
        "legacy_occupation_filter": dict(data["legacy_counts"]),
        "people_with_occupations": len(occupations),
        "people_without_occupations": len(people) - len(occupations),
        "people_with_positive_office_code": len(offices),
        "people_with_specific_offices": kinds["specific_office"],
        "people_with_multiple_specific_offices": sum(n for k, n in specific_histogram.items() if k > 1),
        "officials_without_specific_office": broad_only_officials,
        "people_with_only_unknown_office_names": kinds["office_unknown_broad"],
        "status_only_officials": broad_only_officials - kinds["office_unknown_broad"],
        "occupation_resolution_people": dict(sorted(kinds.items())),
        "distinct_person_office_pairs": sum(len(v) for v in offices.values()),
        "specific_person_office_pairs": known_pairs,
        "distinct_used_office_codes": len(office_people),
        "distinct_specific_office_codes": len(known_codes),
        "unknown_office_codes": sorted(unknown_offices),
        "unknown_person_office_pairs": sum(office_people[o] for o in unknown_offices),
        "unknown_posting_rows": sum(office_rows[o] for o in unknown_offices),
        "posting_source_rows": evidence_rows,
        "long_occupation_rows": entry_count, "max_occupations_per_person": slots,
        "occupation_count_histogram": dict(sorted(histogram.items())),
        "specific_office_count_histogram": dict(sorted(specific_histogram.items())),
        "policies": {
            "population": "All literal-Birth/Death date-qualified BIOG_MAIN people; no relation endpoint or period restriction.",
            "office_dedup_key": "(c_personid,c_office_id); different office IDs remain distinct even with the same name.",
            "slot_order": "Increasing office ID; occ1 is not a primary, earliest, highest or most important office.",
            "unknown_office": "Unknown/missing names are excluded from specific slots but retained in source evidence and the office inventory. If no specific title remains, retain legacy broad 做官 with an explicit resolution flag.",
            "status_fallback": "Reuse the previous v1 whitelist/official priority/unique nonofficial class policy when no specific title is available. Do not promote STATUS descriptions to specific office titles.",
            "office_scope": "All recorded positive office codes, including ranks, honorary titles and titles of nobility. No rank hierarchy or modern occupational recoding.",
            "temporal_scope": "Person dates are filtered; all their office history is retained. Appointment years are preserved raw, not matched to a period or used to order slots.",
        },
        "files": {p.name: p.stat().st_size for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# CBDB 具体官职与职业数据 v1\n\n"
        f"日期合格 {len(people):,} 人，有职业 {len(occupations):,} 人，其中有具体官职 {kinds['specific_office']:,} 人。\n\n"
        "`people_with_occupations.csv` 是主表：每人一行，保留姓名、生卒年、来源，以及 occ1/occ1_code、occ2/occ2_code 等全部职业槽位。\n"
        "`people_with_specific_offices.csv` 仅含至少有一项具体官职的人。两个宽表均不截断官职列表。\n"
        "`date_filtered_people.csv` 保留全部日期合格人物及职业资格，未获职业标签者也可追溯。\n"
        "`person_occupations_long.tsv.gz` 每人每个职业一行，保存槽位、官职码、朝代、来源及支持记录数。\n"
        "`office_inventory.csv` 是实际出现的官职码表与频数，含未详码。\n"
        "`office_postings_source.tsv.gz` 保留日期合格人物全部有效正官职码的原始任官行，含原任官起止年、posting_id、官职性质与文献来源。\n"
        "`officials_without_specific_office.csv` 单列仅知做官、缺具体官名的人，不把这些人当作已有具体官职。\n\n"
        "## 口径\n\n"
        "日期复用旧脚本：生卒年至少一项非0非NULL，排除超过2026及生年晚于卒年；保留负年份。不限旧实验图的关系端点。\n"
        "有具体官名时以任官记录为准，按官职ID去重，同名异码不合并。原库官名与朝代保留，不统一为做官。\n"
        "未详/空白/缺码不是具体官职。只知做官者仍保留宽类并标明 OccupationResolution；非任官者沿用旧 STATUS 白名单和冲突处理。\n"
        "occ1、occ2 按代码升序排列，编号没有主次、官阶或任职时间含义。保留的是个人全部任官史，尚未将具体任职绑定时期。\n"
        "职位、散官、虚衔、爵位均保留原貌，尚未归并为现代职业大类。多个 STATUS 非官宽类冲突者仍不赋标签。\n\n"
        "CSV 使用 UTF-8 BOM，TSV 使用 UTF-8 并作 gzip 压缩。空白年份表示缺失，不能当作0年。\n",
        encoding="utf-8",
    )
    return summary


def build(database=DEFAULT_DB, rules_path=DEFAULT_RULES, output=DEFAULT_OUT):
    database, output = Path(database).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new --output-dir: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        people, person_counts = load_people(conn)
        data = extract(conn, people, rules_path)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
            staging = Path(tmp) / "data"
            staging.mkdir()
            summary = write_exports(conn, staging, people, person_counts, data, database, rules_path)
            staging.rename(output)
        return summary
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--occupation-rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.database, args.occupation_rules, args.output_dir)
    print(json.dumps({k: summary[k] for k in (
        "people_with_occupations", "people_with_specific_offices", "officials_without_specific_office",
        "specific_person_office_pairs", "distinct_specific_office_codes", "max_occupations_per_person",
    )}, ensure_ascii=False, indent=2))
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
