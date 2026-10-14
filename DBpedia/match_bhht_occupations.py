#!/usr/bin/env python3
"""Build an auditable BHHT lookup draft for the current DBpedia occupations.

No graph, source occupation, split, training label, or existing result is edited.
The lookup shares Freebase's exact/substring evidence code. Author-priority
results are recorded separately; this draft is not the authors' full algorithm.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Freebase.match_bhht_professions import load_observed, load_rules, lookup

BHHT = ROOT / "external_data/notable_people/bhht_2022/derived"
MODEL = ROOT / "external_data/dbpedia/processed/model_input"
VERSION = "dbpedia_bhht_lookup_draft_v1"
STATUS_NAMES = {
    "exact_unique": "精确匹配，L1/L2唯一",
    "substring_unique_candidate": "子串匹配，L1/L2唯一",
    "substring_same_l1_multiple_l2": "子串匹配，L1唯一、L2冲突",
    "substring_cross_l1_conflict": "子串匹配，跨L1冲突",
    "exact_same_l1_multiple_l2": "精确匹配，L1唯一、L2冲突",
    "exact_cross_l1_conflict": "精确匹配，跨L1冲突",
    "no_author_rule": "未命中作者规则",
}


def rows(path, delimiter="\t"):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_rows(path, records, fields=None):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]),
                                delimiter="," if ".csv" in path.name else "\t")
        writer.writeheader()
        writer.writerows(records)


def match_occupation(value, rules, observed):
    found, substring, exact = lookup(value, rules, observed)
    selected_ids = set(json.loads(found["selected_rule_priorities_json"]))
    selected = [r for r in substring if r["rule_priority"] in selected_ids]
    terms = list(dict.fromkeys(r["level3_keyword"].strip("_") for r in selected))
    candidates = [{"level1": r["level1"], "level2": r["level2"],
                   "level3_keyword": r["level3_keyword"], "rule_priority": r["rule_priority"]}
                  for r in selected]
    flags = json.loads(found["lookup_flags_json"])
    if len(terms) > 1:
        flags.append("multiple_level3_keyword_candidates")
    # L1 is the primary target. L2/L3 disagreement alone does not block L1.
    # Only an uncertain transfer, a cross-L1 ambiguity, or actual L1 evidence
    # disagreement enters the L1 review queue.
    needs_review = (found["lookup_method"] != "author_exact" or
                    found["candidate_level1_count"] != 1 or
                    bool(found["observed_reference_l1_disagrees"]) or
                    bool(found["exact_overrides_first_match_l1"]))
    l1_flags = []
    if found["lookup_method"] == "no_author_rule":
        l1_flags.append("no_author_rule")
    elif found["lookup_method"] == "author_substring":
        l1_flags.append("substring_requires_l1_transfer_review")
    if found["candidate_level1_count"] > 1:
        l1_flags.append("multiple_author_l1_candidates")
    if found["observed_reference_l1_disagrees"]:
        l1_flags.append("observed_reference_l1_disagrees")
    if found["exact_overrides_first_match_l1"]:
        l1_flags.append("exact_lookup_differs_from_author_first_l1")
    first = substring[0] if substring else None
    l3 = terms[0].replace("_", " ") if len(terms) == 1 else ""
    stable_fine = (not needs_review and bool(found["candidate_level2"]) and
                   not found["exact_overrides_first_match_parent"] and not found["observed_reference_disagrees"])
    l1_lookup_status = ("unmatched" if not selected else "cross_l1_conflict" if found["candidate_level1_count"] > 1 else
                        "exact_l1_unique" if exact else "substring_l1_unique")
    result = {**found, "candidate_level3_keyword_display": l3,
              "candidate_level3_keyword": selected[0]["level3_keyword"] if len(terms) == 1 else "",
              "candidate_level3_count": len(terms), "l1_lookup_status": l1_lookup_status,
              "candidate_l123_json": dump(candidates), "review_flags_json": dump(flags),
              "l1_review_flags_json": dump(l1_flags),
              "l1_status": "unmatched" if not selected else "conflicting" if found["candidate_level1_count"] > 1 else "unique_candidate",
              "l2_status": "unmatched" if not selected else "conflicting" if found["candidate_level2_count"] > 1 else "unique_candidate",
              "l3_status": "unmatched" if not selected else "multiple_keywords" if len(terms) > 1 else "unique_keyword",
              "needs_llm_review": int(needs_review), "llm_review_status": "not_reviewed",
              "strict_level1": found["candidate_level1"] if not needs_review else "",
              "strict_level2": found["candidate_level2"] if stable_fine else "",
              "strict_level3_keyword_display": l3 if stable_fine else "",
              "strict_level3_keyword": selected[0]["level3_keyword"] if stable_fine and len(terms) == 1 else "",
              "author_first_match_level3_keyword": first["level3_keyword"] if first else "",
              "all_substring_l123_json": dump([{"level1": r["level1"], "level2": r["level2"],
                                                "level3_keyword": r["level3_keyword"],
                                                "rule_priority": r["rule_priority"]} for r in substring]),
              "mapping_status": "pending_l1_review" if needs_review else "exact_l1_lookup_candidate",
              "mapping_version": VERSION}
    return result, substring, exact


def validate_inputs(inventory, assignments, nodes):
    expected = Counter({r["occupation_label"]: int(r["graph_nodes"]) for r in inventory})
    if len(expected) != len(inventory) or any(v <= 0 for v in expected.values()):
        raise ValueError("Occupation inventory has duplicates or invalid counts")
    assigned = {r["person_uri"]: r for r in assignments}
    if len(assigned) != len(assignments):
        raise ValueError("Duplicate person in assignment audit")
    node_map = {r["node_id"]: r for r in nodes}
    if len(node_map) != len(nodes) or set(node_map) != set(assigned):
        raise ValueError("Current source nodes and occupation-assignment people differ")
    if Counter(r["occupation_label"] for r in assignments) != expected:
        raise ValueError("Assignment audit and occupation inventory differ")
    for person, row in assigned.items():
        if node_map[person]["occupation_level1"] != row["occupation_label"]:
            raise ValueError(f"Current source occupation differs: {person}")
    return assigned


def run(args):
    inventory = list(rows(args.inventory))
    assignments = list(rows(args.assignments))
    nodes = list(rows(args.nodes, ","))
    assigned = validate_inputs(inventory, assignments, nodes)
    rules = load_rules(args.rules)
    observed = load_observed(args.observed)
    hierarchy = list(rows(args.hierarchy, ","))
    parents = {(r["level1"], r["level2"]) for r in hierarchy}
    if any((r["level1"], r["level2"]) not in parents for r in rules):
        raise ValueError("Rule parent absent from author L1/L2 hierarchy")
    out = args.output_dir
    if out.resolve() in {args.nodes.parent.resolve(), args.assignments.parent.resolve(), BHHT.resolve()}:
        raise ValueError("Output must be separate from source tables")
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "occupation_bhht_matches.tsv"
    if existing.exists() and any(any(r.get(k) for k in ["review_level1", "review_level2", "review_level3_keyword", "review_decision", "review_reason"]) for r in rows(existing)):
        raise ValueError("Existing human/LLM review fields are populated; use a new output directory")
    examples, sources, raw_values, multi = defaultdict(list), defaultdict(Counter), defaultdict(list), Counter()
    for r in assignments:
        key = r["occupation_label"]
        sources[key][r["primary_source"]] += 1
        multi[key] += int(int(r["primary_raw_value_count"]) > 1)
        if len(examples[key]) < 3:
            examples[key].append(r["person_uri"])
        if r["chosen_raw_value"] not in raw_values[key] and len(raw_values[key]) < 5:
            raw_values[key].append(r["chosen_raw_value"])
    records, evidence = [], []
    for rank, source in enumerate(sorted(inventory, key=lambda r: (-int(r["graph_nodes"]), r["occupation_label"])), 1):
        value = source["occupation_label"]
        found, substring, exact = match_occupation(value, rules, observed)
        record = {"rank": rank, "occupation_label": value, "graph_nodes": int(source["graph_nodes"]),
                  **found, "primary_source_counts_json": dump(sources[value]),
                  "multi_primary_value_nodes": multi[value], "sample_people_json": dump(examples[value]),
                  "sample_chosen_raw_values_json": dump(raw_values[value]),
                  "review_level1": "", "review_level2": "", "review_level3_keyword": "",
                  "review_decision": "", "review_reason": ""}
        records.append(record)
        ids = set(json.loads(found["selected_rule_priorities_json"]))
        exact_ids = {r["rule_priority"] for r in exact}
        for rule in substring:
            evidence.append({"occupation_label": value, "graph_nodes": record["graph_nodes"],
                             **rule, "whole_label_exact": int(rule["rule_priority"] in exact_ids),
                             "in_selected_lookup_tier": int(rule["rule_priority"] in ids),
                             "author_first_priority_match": int(rule is substring[0])})
    index = {r["occupation_label"]: r for r in records}
    status_counts = Counter(r["match_status"] for r in records)
    status_nodes = Counter()
    for r in records:
        status_nodes[r["match_status"]] += r["graph_nodes"]
    review = sorted([r for r in records if r["needs_llm_review"]], key=lambda r: (-r["graph_nodes"], r["rank"]))
    strict = [r for r in records if not r["needs_llm_review"]]
    l1_counts, l1_nodes = Counter(), Counter()
    for r in records:
        l1_counts[r["l1_lookup_status"]] += 1
        l1_nodes[r["l1_lookup_status"]] += r["graph_nodes"]
    summary = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "occupations": len(records), "graph_nodes": len(nodes), "author_rules": len(rules),
               "match_status_counts": dict(status_counts), "graph_nodes_by_match_status": dict(status_nodes),
               "l1_lookup_status_counts": dict(l1_counts), "graph_nodes_by_l1_lookup_status": dict(l1_nodes),
               "strict_exact_candidate_occupations": len(strict),
               "strict_exact_candidate_nodes": sum(r["graph_nodes"] for r in strict),
               "pending_review_occupations": len(review), "pending_review_nodes": sum(r["graph_nodes"] for r in review),
               "original_priority_vs_exact_parent_disagreements": sum(r["exact_overrides_first_match_parent"] for r in records),
               "original_priority_vs_exact_parent_disagreement_nodes": sum(r["graph_nodes"] for r in records if r["exact_overrides_first_match_parent"]),
               "observed_reference_disagreement_occupations": sum(r["observed_reference_disagrees"] for r in records),
               "l1_exact_candidate_with_l2_conflict_occupations": sum(not r["needs_llm_review"] and r["candidate_level2_count"] > 1 for r in records),
               "unique_l1_candidate_occupations": sum(r["candidate_level1_count"] == 1 for r in records),
               "unique_l1_candidate_nodes": sum(r["graph_nodes"] for r in records if r["candidate_level1_count"] == 1),
               "multi_primary_value_nodes": sum(multi.values()), "primary_source_nodes": dict(Counter(r["primary_source"] for r in assignments)),
               "training_labels_modified": False, "llm_classification_performed": False,
               "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "inputs": {k: {"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                          for k, p in [("inventory", args.inventory), ("assignments", args.assignments), ("nodes", args.nodes),
                                       ("rules", args.rules), ("observed", args.observed), ("hierarchy", args.hierarchy)]}}
    assert sum(status_counts.values()) == len(records) and sum(status_nodes.values()) == len(nodes)
    write_rows(out / "occupation_bhht_matches.tsv", records)
    write_rows(out / "llm_review_queue.tsv", review, list(records[0]))
    write_rows(out / "priority_disagreements.tsv", [r for r in records if r["exact_overrides_first_match_parent"]], list(records[0]))
    write_rows(out / "rule_evidence.tsv", evidence, ["occupation_label", "graph_nodes", *rules[0],
               "whole_label_exact", "in_selected_lookup_tier", "author_first_priority_match"])
    person_rows = []
    for node in nodes:
        source = assigned[node["node_id"]]
        match = index[source["occupation_label"]]
        person_rows.append({"person_uri": node["node_id"], "occupation_label": source["occupation_label"],
                            "primary_source": source["primary_source"], "chosen_raw_value": source["chosen_raw_value"],
                            "primary_raw_value_count": source["primary_raw_value_count"],
                            "additional_source_values_need_review": int(int(source["primary_raw_value_count"]) > 1),
                            **{k: match[k] for k in ["candidate_level1", "candidate_level2", "candidate_level3_keyword",
                                                   "strict_level1", "strict_level2", "strict_level3_keyword", "l1_lookup_status", "needs_llm_review"]}})
    write_rows(out / "person_mapping_preview.csv.gz", person_rows)
    with (out / "llm_review_queue.jsonl").open("w", encoding="utf-8") as stream:
        for r in review:
            stream.write(dump({k: r[k] for k in ["occupation_label", "graph_nodes", "match_status", "candidate_l123_json",
                                                "all_substring_l123_json", "author_first_match_level1", "author_first_match_level2",
                                                "review_flags_json", "l1_review_flags_json", "sample_chosen_raw_values_json", "sample_people_json",
                                                "observed_main_reference_json"]}) + "\n")
    prompt = """请审核 DBpedia 职业到 BHHT 作者 L1/L2/L3 的候选映射。
输入每行按 occupation_label 为唯一键；人数用于审核排序，不是分类证据。
本轮优先解决L1。L1一致但L2/L3不唯一时，先保留L1，细层级可以留空，不因细分冲突拒绝L1。
依据职业的完整语义、原始URI、作者允许的父类别和候选词项判断。作者首次子串命中是复现参考，不是语义真值。
不要把 exact、substring 或观察数据参考当作已人工验证。不要因多个词同属L1就随意指定L2/L3。
需要区分职业、领域、机构/职务和多职业复合字符串。没有足够上下文时标 pending_context，不要强行归 Other。
L1/L2必须属于提供的 author_level1_level2_hierarchy.csv 的有效父对子集。
L3若确定，请返回作者规则中真实存在且父类一致的 level3_keyword 原文；若作者没有合适的细词，可以仅确定 L1/L2，L3留空。
每条返回JSON：occupation_label, decision(mapped/ambiguous/out_of_scope/pending_context), level1, level2, level3_keyword, reason, evidence, alternative_candidates。
复合多职业可返回多个候选，不能在这一步默默选主职业。没有外部检索证据时，将evidence标为semantic_reasoning_only。
输出属于LLM建议，不能标为人工确认或自动写入训练标签。\n"""
    (out / "LLM_REVIEW_PROMPT.md").write_text(prompt, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # JSON is an analysis intermediate for the separately reproducible XLSX builder.
    (out / "workbook_data.json").write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False), encoding="utf-8")
    lines = ["# DBpedia → BHHT L1/L2/L3 首版查表", "", "这是一份职业词表映射草案。未修改人物的原职业选择、图、split或训练输入，未执行本轮LLM分类。", "",
             "## 结果", "", "| 查表结果 | 职业种数 | 对应人物数 |", "|---|---:|---:|"]
    for status, name in [("exact_l1_unique", "精确匹配，L1唯一（不因L2冲突阻塞）"),
                         ("substring_l1_unique", "子串匹配，L1唯一候选"),
                         ("cross_l1_conflict", "跨L1冲突"), ("unmatched", "未命中作者规则")]:
        lines.append(f"| {name} | {l1_counts[status]:,} | {l1_nodes[status]:,} |")
    lines += ["", f"合计 {len(records):,} 种已选职业，{len(nodes):,} 人。每人沿用已有唯一职业选择，表中人物数互斥。",
              f"严格L1候选（精确匹配且L1唯一、无L1复核标记）：{len(strict):,} 种，{summary['strict_exact_candidate_nodes']:,} 人。L2冲突不阻塞L1。",
              f"待LLM复核：{len(review):,} 种，{summary['pending_review_nodes']:,} 人。精确命中也不等于职业语义已经验证。", "",
              "## 方法与层级含义", "",
              "- 完全沿用Freebase的作者查表函数：小写、ASCII空格替换下划线、两端加下划线；保留作者词项原本的边界标记做literal substring。未加模糊匹配、同义词或最长词优先。",
              "- 精确全词候选优先于仅子串候选，所有命中证据仍保留。这是独立查表草案，不等于作者按优先级首次赋值的完整算法。author_first_match_*记录原算法赢家，priority_disagreements.tsv专列差异。",
              "- L1和L2来自作者规则。L3是命中的作者关键词，而非另建的精细职业本体。多个关键词可能共享L1/L2；此时保留所有L3候选，不随意选一个。",
              "- 本轮L1优先。同L1多L2时保留candidate_level1，L2留空；跨L1时L1/L2悬置；多个L3时单值candidate_level3_keyword留空。strict_level1只要求精确匹配且L1唯一、无L1优先级/参考冲突，不因细分类冲突阻塞L1。",
              "- L3字段明确命名为level3_keyword，保留原作者的边界下划线；*_display仅是去除边界后的显示词，不是另建的职业本体。L2/关键词若与作者首命中父对不一致或观察参考有差异，仅保留候选，不放入strict细层字段，L1仍可单独保留。",
              "- 观察到的BHHT人物主职业字段只作辅助对照。artist等同词存在观察父类例外的标签仍进入复核队列；不会据此覆盖作者规则。",
              "- 人物层面的多职业选择未在此重做。源主来源存在多值的人数为 " + f"{summary['multi_primary_value_nodes']:,}；人物预览另列该标记，仍保持原来源优先级与原唯一职业。", "",
              "## 文件与复现", "",
              "- occupation_bhht_matches.tsv：全部职业、分层候选、作者首次命中、规则编号、源URI和人物例子。",
              "- llm_review_queue.tsv/jsonl：按图内人数降序排列的待审项；LLM_REVIEW_PROMPT.md提供受限层级和输出协议。",
              "- rule_evidence.tsv：每个职业的所有作者规则命中、精确标记、选中tier和source_line。",
              f"- person_mapping_preview.csv.gz：{len(nodes):,}人的候选标签预览；不作为训练输入。",
              "- summary.json：口径、输入路径/大小/mtime、Git版本和覆盖统计。",
              "- dbpedia_bhht_mapping_review.xlsx：用于查看候选与填写独立审核字段。",
              "", "```bash", "python DBpedia/match_bhht_occupations.py", "```", "",
              "工作簿生成脚本另保存在 DBpedia/build_bhht_mapping_review.mjs，依赖 @oai/artifact-tool。将依赖包链接到 artifacts/dbpedia_bhht_review_workbook_20261014/node_modules 后执行：", "",
              "```bash", "node DBpedia/build_bhht_mapping_review.mjs", "```", "",
              "工作簿若已存在，生成器会拒绝覆盖，以保留填写过的审核字段。需重建时使用新的输出目录。CSV/TSV映射脚本仅依赖Python标准库，无需Excel运行库。", "",
              "数据与规则来自本地BHHT派生表；作者源程序位置和行号保留在逐条证据中。Freebase共享查表函数用于保持口径一致。下一步先审核高影响子串、跨父类和未命中项，再讨论冻结新标签和新训练实验。"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=MODEL / "occupation_label_counts.tsv")
    parser.add_argument("--assignments", type=Path, default=MODEL / "occupation_assignments.tsv.gz")
    parser.add_argument("--nodes", type=Path, default=ROOT / "artifacts/dbpedia_2022_priority_v1/nodes.csv")
    parser.add_argument("--rules", type=Path, default=BHHT / "author_keyword_l123_rules.csv")
    parser.add_argument("--observed", type=Path, default=BHHT / "observed_main_occupation_l123.csv")
    parser.add_argument("--hierarchy", type=Path, default=BHHT / "author_level1_level2_hierarchy.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/dbpedia_bhht_match_2026-10-14")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
