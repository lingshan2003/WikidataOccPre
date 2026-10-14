#!/usr/bin/env python3
"""Validate and merge LLM L1 proposals for conflicting/unmatched DBpedia terms.

Only the two explicitly authorized lookup groups receive a semantic review.
Original mappings, raw occupations, graphs and training artifacts stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/dbpedia_bhht_l1_semantic_review_2026-10-14"
LEVELS = {"Culture", "Discovery/Science", "Leadership", "Sports/Games", "Other"}
TARGET_STATUSES = {"cross_l1_conflict", "unmatched"}
REVIEW_FIELDS = ["rank", "occupation_label", "graph_nodes", "semantic_level1", "review_status",
                 "confidence", "candidate_l1s_json", "semantic_rationale", "evidence", "reviewer"]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))


def write(path, records, fields=None):
    fields = fields or list(records[0])
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "wt", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="," if ".csv" in path.name else "\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def validate_decision(row):
    for key in REVIEW_FIELDS:
        if key not in row:
            raise ValueError(f"Missing review field {key}")
    if not row["semantic_rationale"].strip() or not row["evidence"].strip():
        raise ValueError(f"Missing rationale/evidence: {row['occupation_label']}")
    values = json.loads(row["candidate_l1s_json"])
    if not isinstance(values, list) or len(values) != len(set(values)) or set(values) - LEVELS:
        raise ValueError(f"Invalid candidate L1s: {row['occupation_label']}")
    if row["review_status"] == "resolved":
        if row["semantic_level1"] not in LEVELS or row["confidence"] not in {"high", "medium"} or values != [row["semantic_level1"]]:
            raise ValueError(f"Invalid resolved decision: {row['occupation_label']}")
    elif row["review_status"] != "needs_context" or row["semantic_level1"] or row["confidence"] != "low":
        raise ValueError(f"Invalid pending decision: {row['occupation_label']}")


def validate_batch(inputs, outputs, reviewer):
    expected = {(r["rank"], r["occupation_label"], r["graph_nodes"]) for r in inputs}
    actual = [(r["rank"], r["occupation_label"], r["graph_nodes"]) for r in outputs]
    if len(expected) != len(inputs) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Batch rows are missing, duplicated, or mismatched")
    for r in outputs:
        if r["reviewer"] != reviewer:
            raise ValueError(f"Wrong reviewer provenance: {r['occupation_label']}")
        validate_decision(r)


def merge(source, directory):
    original = read(source)
    index = {r["occupation_label"]: r for r in original}
    if len(index) != len(original):
        raise ValueError("Duplicate source occupations")
    selected = {k for k, r in index.items() if r["l1_lookup_status"] in TARGET_STATUSES}
    reviews, batch_counts = {}, []
    for batch in range(1, 4):
        inputs = read(directory / f"batch_{batch}_input.tsv")
        outputs = read(directory / f"batch_{batch}_review.tsv")
        validate_batch(inputs, outputs, f"dbpedia_l1_batch_{batch}")
        if set(reviews) & {r["occupation_label"] for r in outputs}:
            raise ValueError("Batch overlap")
        for r in inputs:
            if r["occupation_label"] not in selected or r["rank"] != index[r["occupation_label"]]["rank"] or r["graph_nodes"] != index[r["occupation_label"]]["graph_nodes"]:
                raise ValueError("Batch input differs from current frozen lookup")
        reviews.update({r["occupation_label"]: r for r in outputs})
        batch_counts.append({"batch": batch, "rows": len(outputs), "status_counts": dict(Counter(r["review_status"] for r in outputs))})
    if set(reviews) != selected:
        raise ValueError("Review coverage differs from authorized two-group scope")
    overrides = directory / "primary_adjudications.tsv"
    adjudications = read(overrides) if overrides.exists() else []
    if len({r["occupation_label"] for r in adjudications}) != len(adjudications):
        raise ValueError("Duplicate primary adjudications")
    for r in adjudications:
        validate_decision(r)
        if r["occupation_label"] not in selected or r["reviewer"] != "dbpedia_l1_primary":
            raise ValueError("Out-of-scope primary adjudication")
        old = index[r["occupation_label"]]
        if r["rank"] != old["rank"] or r["graph_nodes"] != old["graph_nodes"]:
            raise ValueError("Primary input mismatch")
    overrides_by_label = {r["occupation_label"]: r for r in adjudications}
    reviewed, crosswalk = [], []
    for raw, original_row in index.items():
        decision = overrides_by_label.get(raw, reviews.get(raw))
        if decision:
            review = {**decision, "review_scope": original_row["l1_lookup_status"],
                      "batch_reviewer": reviews[raw]["reviewer"], "human_confirmed": 0, "l2_l3_review": "deferred",
                      **{key: original_row.get(key, "[]") for key in ["candidate_parent_pairs_json", "sample_people_json",
                                                                     "sample_chosen_raw_values_json", "primary_source_counts_json"]}}
            reviewed.append(review)
            label = decision["semantic_level1"]
            origin = "llm_semantic_proposal" if label else "pending_context"
        else:
            review = {"semantic_level1": "", "review_status": "outside_this_review_scope", "confidence": "",
                      "candidate_l1s_json": original_row["candidate_parent_pairs_json"], "semantic_rationale": "", "evidence": "", "reviewer": ""}
            label = original_row["candidate_level1"]
            origin = "original_unique_l1_lookup_candidate"
        crosswalk.append({**original_row, "level1": label, "level1_origin": origin,
                          **{key: review[key] for key in ["semantic_level1", "review_status", "confidence", "semantic_rationale", "evidence", "reviewer"]},
                          "semantic_candidate_l1s_json": review["candidate_l1s_json"] if decision else json.dumps(sorted({p[0] for p in json.loads(original_row["candidate_parent_pairs_json"])})),
                          "primary_adjudicated": int(raw in overrides_by_label), "human_confirmed": 0, "l2_l3_review": "deferred"})
    reviewed.sort(key=lambda r: int(r["rank"]))
    resolved = [r for r in reviewed if r["review_status"] == "resolved"]
    pending = [r for r in reviewed if r["review_status"] == "needs_context"]
    counts, people = Counter(), Counter()
    for r in crosswalk:
        counts[r["level1"] or "Pending"] += 1
        people[r["level1"] or "Pending"] += int(r["graph_nodes"])
    summary = {"version": "dbpedia_bhht_llm_l1_two_group_review_v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": sorted(TARGET_STATUSES), "all_occupations": len(original), "all_graph_nodes": sum(int(r["graph_nodes"]) for r in original),
               "reviewed_occupations": len(reviewed), "reviewed_graph_nodes": sum(int(r["graph_nodes"]) for r in reviewed),
               "semantic_resolved": len(resolved), "semantic_resolved_graph_nodes": sum(int(r["graph_nodes"]) for r in resolved),
               "needs_context": len(pending), "needs_context_graph_nodes": sum(int(r["graph_nodes"]) for r in pending),
               "untouched_lookup_occupations": len(original) - len(reviewed), "batch_counts": batch_counts,
               "primary_adjudications": len(adjudications), "level1_occupation_counts": dict(counts), "level1_person_counts": dict(people),
               "l1_candidate_or_proposal_occupations": sum(bool(r["level1"]) for r in crosswalk),
               "l1_candidate_or_proposal_people": sum(int(r["graph_nodes"]) for r in crosswalk if r["level1"]),
               "training_labels_modified": False, "human_confirmation_performed": False, "l2_l3_semantically_reviewed": False,
               "source_lookup": str(source.resolve()), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    assert len(resolved) + len(pending) == len(reviewed)
    assert sum(counts.values()) == len(original) and sum(people.values()) == summary["all_graph_nodes"]
    write(directory / "semantic_review_943.tsv", reviewed)
    write(directory / "needs_context.tsv", pending, list(reviewed[0]))
    write(directory / "occupation_l1_semantic_crosswalk.tsv", crosswalk)
    compact = [{"职业": r["occupation_label"], "图内人数": r["graph_nodes"], "L1": r["level1"],
                "映射来源": r["level1_origin"], "LLM审核状态": r["review_status"], "置信度": r["confidence"],
                "审核理由": r["semantic_rationale"], "证据": r["evidence"]} for r in crosswalk]
    write(directory / "occupation_l1_semantic_crosswalk_compact.csv", compact)
    pending_compact = [{"职业": r["occupation_label"], "图内人数": r["graph_nodes"],
                        "未定原因": r["semantic_rationale"], "可能L1": r["candidate_l1s_json"],
                        "原始职业URI": r["sample_chosen_raw_values_json"], "人物例子": r["sample_people_json"]}
                       for r in pending]
    write(directory / "needs_context_compact.csv", pending_compact,
          ["职业", "图内人数", "未定原因", "可能L1", "原始职业URI", "人物例子"])
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# DBpedia 跨L1冲突及未命中职业的LLM复核", "", "只裁决L1，L2/L3不纳入本轮判断。三个独立LLM批次由主代理合并验收；所有结果仍为LLM建议，未人工逐项批准、未改训练标签。", "",
             "| 范围/结果 | 职业种数 | 对应人物数 |", "|---|---:|---:|",
             f"| 本轮审核两类词 | {len(reviewed):,} | {summary['reviewed_graph_nodes']:,} |",
             f"| LLM给出L1建议 | {len(resolved):,} | {summary['semantic_resolved_graph_nodes']:,} |",
             f"| 仍需岗位/实体语境 | {len(pending):,} | {summary['needs_context_graph_nodes']:,} |", "",
             "Other是BHHT正式类，仅用于能确定的普通工种、服务及相应身份。机构、专名污染、残缺词和真正跨域的描述不以Other兜底。职业字段中的专业活动或学科可以给出明确注明领域推断的中等置信度L1建议。", "",
             "旧表中的唯一L1查表候选保持原样；不把未开展本轮语义审查的词标成已审。semantic_review_943.tsv保留逐项LLM理由与证据，needs_context.tsv列出未能确定的项，occupation_l1_semantic_crosswalk.tsv为全部词的合并草案。L3仍只是上轮作者关键词参考。", "",
             f"合并后 {summary['l1_candidate_or_proposal_people']:,}/{summary['all_graph_nodes']:,} 人具有L1查表候选或LLM建议（{summary['l1_candidate_or_proposal_people']/summary['all_graph_nodes']:.2%}）。这个分母沿用原图每人已有的唯一职业选择，不将未审的1,474个唯一查表候选说成新的LLM结果。",
             f"occupation_l1_semantic_crosswalk_compact.csv是全部{len(original):,}个职业的查看版；needs_context_compact.csv附原URI和人物例子，便于针对{len(pending):,}个未定词补岗位信息。", "",
             "主代理裁决只覆盖本轮943条输入，批次原始结果保留在batch_*_review.tsv。各批次的输入/输出唯一键、人数、完整性、合法L1、候选集合、置信度、原因、证据和审核者均校验。", "",
             "```bash", "python DBpedia/merge_bhht_l1_review.py", "```", ""]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/dbpedia_bhht_match_2026-10-14/occupation_bhht_matches.tsv")
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    merge(args.source, args.directory)


if __name__ == "__main__":
    main()
