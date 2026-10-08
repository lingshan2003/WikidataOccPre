#!/usr/bin/env python3
"""Extract BHHT occupation fields without inferring a keyword taxonomy.

The person-level hierarchy is authoritative for that person. Observed main
keyword/hierarchy pairs are evidence, not the authors' classification rules.
The source gzip is consumed to EOF, which also checks its gzip CRC.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "external_data/notable_people/bhht_2022"
FIELDS = ["wikidata_code", "name", "level1_main_occ", "level2_main_occ",
          "level2_second_occ", "level3_main_occ", "level3_all_occ",
          "freq_main_occ", "freq_second_occ"]
MISSING = {"", "Missing"}
INVALID_UTF8 = re.compile("[\udc80-\udcff]")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_author_hierarchy(root):
    """Parse the final aggregation block, not individual override statements."""
    source = root / "raw/author_code/7_prog_occupations_B.do"
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    in_l1 = in_l2 = False
    parents, labels = {}, {}
    for line_number, line in enumerate(lines, 1):
        if "*AGGREGATION LEVEL 1" in line:
            in_l1 = True
        elif "*AGGREGATION LEVEL 2" in line:
            in_l1, in_l2 = False, True
        elif in_l2 and line.strip() == "}":
            break
        if in_l1:
            match = re.search(r'occupation_1_L1_B="([^"]+)" if (.*)', line)
            if match:
                for code in re.findall(r'final_occupation_L2_B=="([^"]+)"', match[2]):
                    if code in parents:
                        raise ValueError(f"Duplicate author L2 code: {code}")
                    parents[code] = match[1], line_number
        if in_l2:
            match = re.search(r'replace `x\'=\"([^\"]+)\" if `x\'==\"([^\"]+)\"', line)
            if match:
                labels[match[2]] = match[1], line_number
    if set(parents) != set(labels) or not parents:
        raise ValueError("Author hierarchy blocks do not reconcile")
    rows = [{"level1": parents[code][0], "level2": labels[code][0],
             "author_level2_code": code, "evidence_type": "author_executable_aggregation_rule",
             "source_file": "7_prog_occupations_B.do", "l1_source_line": parents[code][1],
             "l2_source_line": labels[code][1]} for code in parents]
    return sorted(rows, key=lambda r: (r["level1"], r["level2"]))


def extract_keyword_rules(root):
    """Extract literal lists and execution order; do not execute Stata rules."""
    out = root / "derived"
    out.mkdir(parents=True, exist_ok=True)
    hierarchy = extract_author_hierarchy(root)
    write_csv(out / "author_level1_level2_hierarchy.csv", list(hierarchy[0]), hierarchy)
    parents = {r["author_level2_code"]: r for r in hierarchy}
    relative = "Data_Construction/wikidata/prog_occ.do"
    lines = (root / "raw/author_code" / relative).read_text(encoding="utf-8-sig").splitlines()
    definitions = {}
    execution_order = None
    for number, line in enumerate(lines, 1):
        match = re.match(r'global\s+(\w+)\s+"([^"]*)"', line)
        if match:
            if match[1] in definitions:
                raise ValueError(f"Repeated Stata global: {match[1]}")
            # Preserve non-ASCII whitespace within a literal; do not repair source patterns.
            definitions[match[1]] = re.findall(r"[^ \t\r\n]+", match[2]), number
        if execution_order is None and re.match(r"foreach x in .*\{", line.strip()):
            execution_order = line.strip().split(" in ", 1)[1].split("{", 1)[0].split()
    if execution_order is None or set(parents) - set(execution_order):
        raise ValueError("Author matching loop does not cover the aggregation categories")
    if set(parents) - set(definitions):
        raise ValueError("Missing author keyword globals")
    rows = []
    by_literal = defaultdict(set)
    for group_order, code in enumerate(execution_order, 1):
        if code not in parents:
            continue
        literals, source_line = definitions[code]
        for word_order, literal in enumerate(literals, 1):
            by_literal[literal].add(code)
            rows.append({"level1": parents[code]["level1"], "level2": parents[code]["level2"],
                         "level3_keyword": literal, "author_level2_code": code,
                         "match_type": "literal_substring_stata_strpos",
                         "category_priority": group_order, "keyword_priority": word_order,
                         "rule_priority": len(rows) + 1,
                         "assignment_condition": "only_if_current_occupation_category_is_empty",
                         "normalization": "lowercase_then_ascii_spaces_to_underscores_then_wrap_with_underscores",
                         "source_file": relative, "source_line": source_line})
    write_csv(out / "author_keyword_l123_rules.csv", list(rows[0]), rows)
    collisions = [{"level3_keyword": literal, "parent_count": len(codes),
                   "level2_codes_json": json.dumps(sorted(codes)),
                   "policy": "preserve_author_first_match_order_for_reproduction_review_before_transfer"}
                  for literal, codes in sorted(by_literal.items()) if len(codes) > 1]
    write_csv(out / "author_keyword_cross_category_collisions.csv",
              ["level3_keyword", "parent_count", "level2_codes_json", "policy"], collisions)
    inactive = [{"defined_global": code, "literal": literal, "source_line": source_line,
                 "reason": "global_not_referenced_in_matching_loop"}
                for code, (literals, source_line) in definitions.items() if code not in execution_order
                for literal in literals]
    write_csv(out / "author_keyword_unused_globals.csv",
              ["defined_global", "literal", "source_line", "reason"], inactive)
    summary = {"source_doi": "https://doi.org/10.21410/7E4/YLG6YR", "dataset_version": "2.3",
               "author_keyword_source": relative, "active_level2_count": len(parents),
               "active_rule_rows_including_duplicates": len(rows),
               "unique_active_literal_count": len(by_literal),
               "active_literals_in_multiple_level2_groups": len(collisions),
               "defined_but_unused_global_terms": len(inactive),
               "undefined_globals_referenced_by_loop": [code for code in execution_order if code not in definitions],
               "scope": "Wikidata occupation label substring rules, not a complete person-main-label algorithm or a regex dictionary.",
               "not_applied_to_freebase": True}
    (out / "author_keyword_rules_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract(root):
    source = root / "raw/cross-verified-database.csv.gz"
    out = root / "derived"
    out.mkdir(parents=True, exist_ok=True)
    official = extract_author_hierarchy(root)
    write_csv(out / "author_level1_level2_hierarchy.csv", list(official[0]), official)
    expected_parent = {r["level2"]: r["level1"] for r in official}
    l1_count, pairs, triples = Counter(), Counter(), Counter()
    missing = Counter()
    examples = defaultdict(list)
    qids = set()
    count = 0
    encoding_issues = []
    repaired_rows = 0
    target = out / "person_occupation_l123.csv.gz"
    with gzip.open(source, "rt", encoding="utf-8-sig", errors="surrogateescape", newline="") as stream, \
         gzip.open(target, "wt", encoding="utf-8", newline="", compresslevel=3) as dest:
        reader = csv.DictReader(stream)
        absent = set(FIELDS) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"Missing required source fields: {sorted(absent)}")
        writer = csv.DictWriter(dest, fieldnames=FIELDS)
        writer.writeheader()
        for r in reader:
            count += 1
            repaired = False
            for field, value in r.items():
                if INVALID_UTF8.search(value):
                    raw_bytes = value.encode("utf-8", errors="surrogateescape")
                    encoding_issues.append({"source_record_number": count,
                                            "wikidata_code": r["wikidata_code"],
                                            "field": field, "raw_utf8_bytes_hex": raw_bytes.hex()})
                    r[field] = raw_bytes.decode("utf-8", errors="replace")
                    repaired = True
            repaired_rows += repaired
            qid = r["wikidata_code"]
            if not qid or qid in qids:
                raise ValueError(f"Empty or duplicate source Wikidata ID: {qid!r}")
            qids.add(qid)
            writer.writerow({f: r[f] for f in FIELDS})
            a, b, c = (r[f] for f in ["level1_main_occ", "level2_main_occ", "level3_main_occ"])
            l1_count[a] += 1
            for f in FIELDS[2:]:
                missing[f] += r[f] in MISSING
            pairs[a, b] += 1
            triples[a, b, c] += 1
            if len(examples[a, b, c]) < 3:
                examples[a, b, c].append(qid)
    assert sum(l1_count.values()) == sum(pairs.values()) == sum(triples.values()) == count
    write_csv(out / "source_utf8_issues.csv",
              ["source_record_number", "wikidata_code", "field", "raw_utf8_bytes_hex"], encoding_issues)
    parents = defaultdict(set)
    l2_parents = defaultdict(set)
    for a, b, c in triples:
        if a not in MISSING and b not in MISSING and c not in MISSING:
            parents[c].add((a, b))
            l2_parents[b].add(a)
    write_csv(out / "observed_main_occupation_l123.csv",
              ["level1_main_occ", "level2_main_occ", "level3_main_occ", "person_count",
               "observed_parent_count", "evidence_type", "example_wikidata_codes"],
              ({"level1_main_occ": a, "level2_main_occ": b, "level3_main_occ": c,
                "person_count": n, "observed_parent_count": len(parents.get(c, set())),
                "evidence_type": "observed_person_main_fields_not_author_rule",
                "example_wikidata_codes": "|".join(examples[a, b, c])}
               for (a, b, c), n in sorted(triples.items())))
    write_csv(out / "level1_level2_inventory.csv",
              ["level1_main_occ", "level2_main_occ", "person_count", "observed_l1_parent_count"],
              ({"level1_main_occ": a, "level2_main_occ": b, "person_count": n,
                "observed_l1_parent_count": len(l2_parents.get(b, set()))}
               for (a, b), n in sorted(pairs.items())))
    write_csv(out / "main_keyword_parent_candidates.csv",
              ["level3_main_occ", "observed_parent_count", "observed_parents_json", "status"],
              ({"level3_main_occ": c, "observed_parent_count": len(ps),
                "observed_parents_json": json.dumps(sorted(ps), ensure_ascii=False),
                "status": "unique_observed_parent_not_author_rule" if len(ps) == 1 else "ambiguous_observed_parents"}
               for c, ps in sorted(parents.items())))
    exceptions = [{"level1_main_occ": a, "level2_main_occ": b, "level3_main_occ": c,
                   "person_count": n, "aggregation_rule_level1": expected_parent[b],
                   "policy": "preserve_published_person_fields_do_not_override"}
                  for (a, b, c), n in sorted(triples.items())
                  if b in expected_parent and a not in MISSING and a != expected_parent[b]]
    write_csv(out / "published_person_hierarchy_exceptions.csv",
              ["level1_main_occ", "level2_main_occ", "level3_main_occ", "person_count",
               "aggregation_rule_level1", "policy"], exceptions)
    metadata = json.loads((root / "raw/cross_verified_dataverse_metadata.json").read_text())
    version = metadata["data"]["latestVersion"]
    source_file = next(f["dataFile"] for f in version["files"]
                       if f["dataFile"]["filename"] == source.name)
    if source.stat().st_size != source_file["filesize"]:
        raise ValueError("Downloaded size differs from official metadata")
    summary = {
        "source_doi": "https://doi.org/10.21410/7E4/RDAG3O",
        "paper_doi": "https://doi.org/10.1038/s41597-022-01369-4",
        "dataset_version": f"{version['versionNumber']}.{version['versionMinorNumber']}",
        "source_file_id": source_file["id"],
        "source_bytes": source.stat().st_size,
        "source_gzip_crc_checked_by_full_read": True,
        "person_count": count, "unique_wikidata_ids": len(qids),
        "level1_counts_including_missing": dict(l1_count.most_common()),
        "valid_level1_count": sum(a not in MISSING for a in l1_count),
        "valid_level2_count": len(l2_parents),
        "author_aggregation_level1_count": len({r["level1"] for r in official}),
        "author_aggregation_level2_count": len(official),
        "published_person_hierarchy_exception_count": sum(r["person_count"] for r in exceptions),
        "distinct_observed_main_triples_including_missing": len(triples),
        "valid_main_keywords": len(parents),
        "keywords_with_multiple_observed_parent_pairs": sum(len(ps) > 1 for ps in parents.values()),
        "missing_field_counts": dict(missing),
        "source_invalid_utf8_rows": repaired_rows,
        "source_invalid_utf8_fields": len(encoding_issues),
        "encoding_policy": "Original gzip unchanged. Invalid UTF-8 bytes replaced with U+FFFD only in derived exports; exact field bytes retained in source_utf8_issues.csv.",
        "caution": "No author keyword rule is inferred from person co-occurrence. level3_all_occ is preserved verbatim, not split or assigned to the person's main class.",
    }
    (out / "extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--rules-only", action="store_true", help="Extract downloaded author keyword rules without rereading all people")
    args = parser.parse_args()
    if args.rules_only:
        extract_keyword_rules(args.root)
    else:
        extract(args.root)
        if (args.root / "raw/author_code/Data_Construction/wikidata/prog_occ.do").exists():
            extract_keyword_rules(args.root)
