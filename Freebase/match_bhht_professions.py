#!/usr/bin/env python3
"""Look up Freebase profession values against the authors' BHHT rules.

Writes separate review tables. No source occupations, existing crosswalks,
person labels, or training artifacts are modified. Main-person keyword
co-occurrence is auxiliary evidence and never supplies an author rule.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/freebase_profession_review_2026-10-03"
BHHT = ROOT / "external_data/notable_people/bhht_2022/derived"
DEFAULT_OUT = ROOT / "docs/freebase_bhht_match_2026-10-08"
VERSION = "freebase_bhht_author_lookup_v1"
MISSING = {"", "Missing"}


def read_rows(path, delimiter="\t"):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_compact_csv(path, records):
    fields = ["职业", "图内人数", "查表方式", "查表状态", "候选L1", "候选L2",
              "匹配L3词项", "候选分类组合", "旧LLM候选L1", "与旧LLM关系", "复核标记"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for r in records:
            writer.writerow([r["raw_value"], r["graph_people"], r["lookup_method"], r["match_status"],
                             r["candidate_level1"], r["candidate_level2"],
                             "; ".join(json.loads(r["candidate_keywords_json"])),
                             "; ".join(" / ".join(p) for p in json.loads(r["candidate_parent_pairs_json"])),
                             r["previous_llm_level1"], r["previous_llm_comparison"],
                             "; ".join(json.loads(r["review_flags_json"]))])


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_label(value):
    # Match the author's normalization; no aliases, stemming, punctuation
    # removal, parenthesis removal, hyphen conversion, or fuzzy matching.
    return value.lower().replace(" ", "_")


def parent_pairs(rules):
    return sorted({(r["level1"], r["level2"]) for r in rules})


def load_rules(path):
    rules = list(read_rows(path, ","))
    for r in rules:
        r["rule_priority"] = int(r["rule_priority"])
        if r["match_type"] != "literal_substring_stata_strpos" or not r["level3_keyword"]:
            raise ValueError("Unexpected or empty source matching rule")
    rules.sort(key=lambda r: r["rule_priority"])
    if [r["rule_priority"] for r in rules] != list(range(1, len(rules) + 1)):
        raise ValueError("Author rule priority is not contiguous")
    return rules


def load_observed(path):
    index = defaultdict(list)
    for r in read_rows(path, ","):
        if any(r[f] in MISSING for f in ("level1_main_occ", "level2_main_occ", "level3_main_occ")):
            continue
        index[normalize_label(r["level3_main_occ"])].append({
            "level1": r["level1_main_occ"], "level2": r["level2_main_occ"],
            "level3": r["level3_main_occ"], "person_count": int(r["person_count"]),
            "example_wikidata_codes": r["example_wikidata_codes"],
        })
    return index


def lookup(value, rules, observed):
    key = normalize_label(value)
    normalized = "_" + key + "_"
    substring = [r for r in rules if r["level3_keyword"] in normalized]
    # Leading/trailing underscores in an author literal are boundary markers.
    # They are removed only for the whole-label equality test, not substring matching.
    exact = [r for r in substring if r["level3_keyword"].strip("_") == key]
    selected = exact or substring
    pairs = parent_pairs(selected)
    l1s = sorted({p[0] for p in pairs})
    l2s = sorted({p[1] for p in pairs})
    method = "author_exact" if exact else "author_substring" if substring else "no_author_rule"
    if not selected:
        status = "no_author_rule"
    elif len(l1s) > 1:
        status = ("exact" if exact else "substring") + "_cross_l1_conflict"
    elif len(l2s) > 1:
        status = ("exact" if exact else "substring") + "_same_l1_multiple_l2"
    else:
        status = "exact_unique" if exact else "substring_unique_candidate"
    auxiliary = observed.get(key, [])
    auxiliary_pairs = {(r["level1"], r["level2"]) for r in auxiliary}
    selected_set = set(pairs)
    first = substring[0] if substring else None
    suggested_l1 = l1s[0] if len(l1s) == 1 else ""
    suggested_l2 = l2s[0] if len(pairs) == 1 else ""
    flags = []
    if not selected:
        flags.append("no_author_rule")
    elif not exact:
        flags.append("substring_requires_transfer_review")
    if len(l1s) > 1:
        flags.append("multiple_author_l1_candidates")
    elif len(l2s) > 1:
        flags.append("multiple_author_l2_candidates")
    reference_disagrees = bool(selected_set and auxiliary_pairs - selected_set)
    reference_l1_disagrees = bool(l1s and {p[0] for p in auxiliary_pairs} - set(l1s))
    if reference_disagrees:
        flags.append("person_main_reference_disagrees_not_used_to_override")
    first_differs = bool(exact and len(pairs) == 1 and
                         (first["level1"], first["level2"]) != pairs[0])
    first_l1_differs = bool(exact and suggested_l1 and first["level1"] != suggested_l1)
    if first_differs:
        flags.append("exact_lookup_differs_from_full_author_first_match")
    return {
        "normalized_label": key, "lookup_method": method, "match_status": status,
        "candidate_level1": suggested_l1, "candidate_level2": suggested_l2,
        "candidate_parent_pairs_json": encoded(pairs),
        "candidate_level1_count": len(l1s), "candidate_level2_count": len(l2s),
        "candidate_keywords_json": encoded(list(dict.fromkeys(r["level3_keyword"] for r in selected))),
        "selected_rule_priorities_json": encoded([r["rule_priority"] for r in selected]),
        "all_substring_parent_pairs_json": encoded(parent_pairs(substring)),
        "all_substring_rule_priorities_json": encoded([r["rule_priority"] for r in substring]),
        "author_first_match_level1": first["level1"] if first else "",
        "author_first_match_level2": first["level2"] if first else "",
        "author_first_match_keyword": first["level3_keyword"] if first else "",
        "exact_overrides_first_match_l1": int(first_l1_differs),
        "exact_overrides_first_match_parent": int(first_differs),
        "observed_main_reference_json": encoded(auxiliary),
        "observed_reference_disagrees": int(reference_disagrees),
        "observed_reference_l1_disagrees": int(reference_l1_disagrees),
        "lookup_flags_json": encoded(flags),
    }, substring, exact


def comparison(old, candidate):
    if not candidate:
        return "no_unique_author_l1_candidate"
    if old["status"] == "proposed":
        return "agrees_with_previous_llm" if old["proposed_level1"] == candidate else "differs_from_previous_llm"
    if old["status"] == "out_of_scope" and candidate == "Other":
        return "formal_author_other_previously_out_of_scope"
    return "author_candidate_previous_" + old["status"]


def person_coverage(nodes_path, inventory, matched):
    verified = Counter()
    people_by_status = Counter()
    resolution = {"exact_only": Counter(), "exact_and_unique_l1_substring": Counter()}
    count = 0
    for r in read_rows(nodes_path, ","):
        count += 1
        raw = json.loads(r["Professions"])
        if not raw or len(raw) != len(set(raw)):
            raise ValueError("Expected nonempty unique professions per person")
        verified.update(raw)
        people_by_status.update({matched[v]["match_status"] for v in raw})
        for policy in resolution:
            usable = [matched[v]["candidate_level1"] if (
                matched[v]["candidate_level1"] and (
                    policy != "exact_only" or matched[v]["lookup_method"] == "author_exact")) else "" for v in raw]
            if any(not label for label in usable):
                state = "incomplete_mapping"
            else:
                state = "unique_l1_candidate" if len(set(usable)) == 1 else "multiple_l1_candidates"
            resolution[policy][state] += 1
    expected = Counter({r["raw_value"]: int(r["graph_people"]) for r in inventory})
    if verified != expected:
        raise ValueError("1817-value inventory no longer matches current graph nodes")
    return {"graph_people": count, "person_profession_pairs": sum(verified.values()),
            "unique_people_with_at_least_one_profession_by_match_status": dict(people_by_status),
            "person_candidate_resolution_not_frozen_labels": {k: dict(v) for k, v in resolution.items()}}


def report(out, summary, records):
    labels = {
        "exact_unique": "精确同词，作者 L1/L2 唯一",
        "exact_same_l1_multiple_l2": "精确同词，L1 唯一但 L2 冲突",
        "exact_cross_l1_conflict": "精确同词，跨 L1 冲突",
        "substring_unique_candidate": "仅子串命中，单一 L1/L2 候选",
        "substring_same_l1_multiple_l2": "仅子串命中，L1 唯一但 L2 冲突",
        "substring_cross_l1_conflict": "仅子串命中，跨 L1 冲突",
        "no_author_rule": "无作者规则命中",
    }
    lines = ["# Freebase 1,817 种职业的 BHHT 作者规则查表结果", "",
             "本轮仅生成独立查表、对照和复核材料；没有修改上一版职业 crosswalk、人物标签或训练结果，也没有执行新的 LLM 分类。", "",
             "## 统计", "", "| 查表结果 | 职业种数 | 人物—职业关联次数 |", "|---|---:|---:|"]
    for status, label in labels.items():
        lines.append(f"| {label} | {summary['match_status_counts'].get(status, 0):,} | {summary['profession_occurrences_by_status'].get(status, 0):,} |")
    lines.extend(["", "关联次数允许同一人物重复计入，不能当作不重复人数。图中 100,756 人，职业种数 1,817；两者均用原始节点表重新核验。", "",
                  "## 匹配口径", "",
                  "1. 标签仅小写、ASCII 空格转下划线。作者词项的首尾下划线只在完整标签相等测试中作为边界标记去除。保留括号、标点、连字符、词形和非 ASCII 字符，不做模糊匹配或新同义词推断。", 
                  "2. 完整标签精确命中优先；无精确时，严格按作者的两端补下划线后字面子串规则查表。保留全部候选，跨类时不按第一条强制定类。", 
                  "3. `candidate_level1` 仅在当前查表层级的 L1 唯一时填写，`candidate_level2` 仅在 L1/L2 组合唯一时填写。它们是职业词的候选分类，不是人物主职业或冻结训练标签。", 
                  "4. 作者完整子串分类器的首命中另存 `author_first_match_*`。它可能与本轮“精确优先”的结果不同，不能把两套口径混称为作者原始分类器。", 
                  "5. 人物主字段只做独立参考。主关键词对应多个分类或与规则不一致时标记复核，不从同人共现或多数频数强行补出作者规则。", 
                  "6. `Other` 是作者的正式分类，与旧标签中未映射的临时兜底不同。旧 LLM 的一致/不一致只做对照，不参与本轮规则选择。", "",
                  f"人物主字段参考与当前候选组合不一致：{summary['observed_reference_disagreement_professions']} 种（其中 L1 不一致 {summary['observed_reference_l1_disagreement_professions']} 种）。这些标记不能自动推翻作者规则，但需要在冻结标签前复核。", 
                  f"在精确结果唯一的职业中，{summary['exact_priority_differs_from_author_first_parent']} 种的父类组合不同于作者完整子串首命中，其中 L1 不同 {summary['exact_priority_differs_from_author_first_l1']} 种。", "",
                  "## 文件", "", 
                  "- `profession_bhht_matches_compact.csv`：便于查看的 11 列简表，1,817 行，中文表头，含候选 L1/L2/L3、旧标签和复核标记。", 
                  "- `profession_bhht_matches.tsv`：全部 1,817 种职业的完整主表，保留 L1/L2/L3 候选、全部候选组、原始规则编号、旧 LLM 映射与参考证据。", 
                  "- `exact_author_matches.tsv`：543 种精确同词命中，含 2 种跨 L1 冲突。精确唯一的行仍需留意参考一致性标记。", 
                  "- `substring_candidates.tsv`：912 种仅子串命中的候选，全部需要转用复核。", 
                  "- `conflicts_for_review.tsv`：多候选、来源参考不一致、精确与首命中类别不同、或与旧 LLM 的唯一 L1 不一致的行，按图内频次排序。", 
                  "- `unmatched_for_llm.tsv`：362 种无作者规则命中的职业，供后续补词/LLM分类；部分有非权威人物主字段参考，单独标记。", 
                  "- `author_llm_l1_disagreements.tsv`：本轮唯一 L1 候选与旧 `proposed` LLM 标签不同的行。", 
                  "- `rule_evidence.tsv`：每个匹配词项一行，含作者原词、源码路径/行号、原始优先顺序，以及是否精确、是否本轮所选层级。", 
                  "- `summary.json`：完整计数、输入路径/大小/修改时间与人物覆盖统计。", "",
                  "## 优先复核的跨大类子串冲突", "", "| 职业 | 图内人数 | 候选 L1/L2 |", "|---|---:|---|"])
    for r in [r for r in records if r["match_status"] == "substring_cross_l1_conflict"][:15]:
        lines.append(f"| {r['raw_value'].replace('|', '/')} | {int(r['graph_people']):,} | {r['candidate_parent_pairs_json'].replace('|', '/')} |")
    lines.extend(["", "## 未命中的高频职业", "", "| 职业 | 图内人数 | 旧状态 |", "|---|---:|---|"])
    for r in [r for r in records if r["match_status"] == "no_author_rule"][:15]:
        lines.append(f"| {r['raw_value'].replace('|', '/')} | {int(r['graph_people']):,} | {r['previous_llm_status']} |")
    lines.extend(["", "## 单一候选与旧 LLM 提议不同的高频职业", "",
                  "单一子串候选不等于可靠分类。例如 Disc jockey 只命中 `jockey` 得到体育候选；本轮只保留规则证据和差异，不自动覆盖文化标签。旧 LLM 也不是裁决依据，须结合具体职业语义复核。", "",
                  "| 职业 | 图内人数 | 本轮候选 L1 | 旧 LLM L1 | 查表方式 |", "|---|---:|---|---|---|"])
    for r in [r for r in records if r["previous_llm_comparison"] == "differs_from_previous_llm"][:15]:
        lines.append(f"| {r['raw_value'].replace('|', '/')} | {int(r['graph_people']):,} | {r['candidate_level1']} | {r['previous_llm_level1']} | {r['lookup_method']} |")
    lines.extend(["", "## 来源与复现", "",
                  "作者数据/规则：BHHT，https://doi.org/10.21410/7E4/RDAG3O (v2.2)；https://doi.org/10.21410/7E4/YLG6YR (v2.3)。职业规则来自 `Data_Construction/wikidata/prog_occ.do`，L2→L1 来自 `7_prog_occupations_B.do:1413–1437`。许可与署名见 `external_data/notable_people/bhht_2022/README.md`。", "",
                  "```bash", "python Freebase/match_bhht_professions.py", "```", "",
                  "下一步先复核冲突与高影响子串候选，再对未命中项补词/LLM分类；本表不自动选择多职业人物的主职业。", ""])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    inventory = list(read_rows(args.inventory))
    if len(inventory) != 1817 or len({r["raw_value"] for r in inventory}) != 1817:
        raise ValueError("Expected exactly 1817 distinct current-graph professions")
    previous_rows = list(read_rows(args.previous))
    previous = {r["raw_value"]: r for r in previous_rows}
    if len(previous) != len(previous_rows) or set(previous) != {r["raw_value"] for r in inventory}:
        raise ValueError("Previous crosswalk and current inventory differ")
    rules = load_rules(args.rules)
    observed = load_observed(args.observed)
    records, evidence = [], []
    for source in sorted(inventory, key=lambda r: int(r["rank"])):
        value = source["raw_value"]
        found, sub, exact = lookup(value, rules, observed)
        old = previous[value]
        diff = comparison(old, found["candidate_level1"])
        selected_ids = set(json.loads(found["selected_rule_priorities_json"]))
        exact_ids = {r["rule_priority"] for r in exact}
        row = {"rank": int(source["rank"]), "raw_value": value,
               "graph_people": int(source["graph_people"]), **found,
               "previous_llm_level1": old["proposed_level1"],
               "previous_llm_status": old["status"], "previous_llm_confidence": old["confidence"],
               "previous_llm_comparison": diff,
               "review_flags_json": encoded(json.loads(found["lookup_flags_json"]) +
                                             (["previous_llm_l1_disagreement"] if diff == "differs_from_previous_llm" else [])),
               "auxiliary_reference_only_without_author_rule": int(not sub and bool(observed.get(normalize_label(value)))),
               "mapping_version": VERSION, "label_status": "lookup_candidate_not_frozen",
               "sample_people_json": source["sample_people_json"]}
        records.append(row)
        for r in sub:
            evidence.append({"raw_value": value, "graph_people": row["graph_people"],
                             "rule_priority": r["rule_priority"], "level1": r["level1"], "level2": r["level2"],
                             "level3_keyword": r["level3_keyword"], "is_whole_label_exact": int(r["rule_priority"] in exact_ids),
                             "in_selected_lookup_tier": int(r["rule_priority"] in selected_ids),
                             "is_full_author_first_match": int(r is sub[0]),
                             "match_type": r["match_type"], "category_priority": r["category_priority"],
                             "keyword_priority": r["keyword_priority"], "source_file": r["source_file"],
                             "source_line": r["source_line"]})
    indexed = {r["raw_value"]: r for r in records}
    coverage = person_coverage(args.nodes, inventory, indexed)
    methods = Counter(r["lookup_method"] for r in records)
    statuses = Counter(r["match_status"] for r in records)
    occurrences = Counter()
    for r in records:
        occurrences[r["match_status"]] += r["graph_people"]
    assert sum(statuses.values()) == 1817 and sum(occurrences.values()) == coverage["person_profession_pairs"]
    summary = {"version": VERSION, **coverage, "professions": len(records), "author_rule_rows": len(rules),
               "lookup_method_counts": dict(methods), "match_status_counts": dict(statuses),
               "profession_occurrences_by_status": dict(occurrences),
               "previous_llm_comparison_counts": dict(Counter(r["previous_llm_comparison"] for r in records)),
               "observed_reference_disagreement_professions": sum(r["observed_reference_disagrees"] for r in records),
               "observed_reference_l1_disagreement_professions": sum(r["observed_reference_l1_disagrees"] for r in records),
               "exact_priority_differs_from_author_first_parent": sum(r["exact_overrides_first_match_parent"] for r in records),
               "exact_priority_differs_from_author_first_l1": sum(r["exact_overrides_first_match_l1"] for r in records),
               "no_author_rule_with_auxiliary_observed_reference": sum(r["auxiliary_reference_only_without_author_rule"] for r in records),
               "inputs": {name: {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                          for name, path in [("inventory", args.inventory), ("previous_llm", args.previous),
                                             ("rules", args.rules), ("observed_reference", args.observed), ("nodes", args.nodes)]},
               "existing_crosswalk_or_training_labels_modified": False, "new_llm_classification_performed": False}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    write_compact_csv(out / "profession_bhht_matches_compact.csv", records)
    write_rows(out / "profession_bhht_matches.tsv", fields, records)
    write_rows(out / "exact_author_matches.tsv", fields, [r for r in records if r["lookup_method"] == "author_exact"])
    write_rows(out / "substring_candidates.tsv", fields, [r for r in records if r["lookup_method"] == "author_substring"])
    write_rows(out / "unmatched_for_llm.tsv", fields, [r for r in records if r["lookup_method"] == "no_author_rule"])
    write_rows(out / "conflicts_for_review.tsv", fields,
               [r for r in records if r["candidate_level1_count"] > 1 or r["candidate_level2_count"] > 1 or
                r["observed_reference_disagrees"] or r["exact_overrides_first_match_parent"] or
                r["previous_llm_comparison"] == "differs_from_previous_llm"])
    write_rows(out / "author_llm_l1_disagreements.tsv", fields,
               [r for r in records if r["previous_llm_comparison"] == "differs_from_previous_llm"])
    write_rows(out / "rule_evidence.tsv", list(evidence[0]), evidence)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report(out, summary, records)
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=REVIEW / "profession_inventory.tsv")
    parser.add_argument("--previous", type=Path, default=REVIEW / "profession_l1_crosswalk_draft.tsv")
    parser.add_argument("--rules", type=Path, default=BHHT / "author_keyword_l123_rules.csv")
    parser.add_argument("--observed", type=Path, default=BHHT / "observed_main_occupation_l123.csv")
    parser.add_argument("--nodes", type=Path, default=ROOT / "external_data/freebase/descriptive_v2_local/05_final/nodes.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
