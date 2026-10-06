"""Inventory and audit a draft L1 crosswalk for the current Freebase relation graph.

This tool only writes separate review artifacts. It never edits extracted data,
freezes training labels, or interprets agent proposals as human approval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "external_data/freebase/descriptive_v2_local/05_final"
REVIEW = PROJECT / "docs/freebase_profession_review_2026-10-03"
DERIVED = PROJECT / "external_data/freebase/profession_l1_review_v1"
LEVELS = {"Culture", "Discovery/Science", "Leadership", "Sports/Games"}


def rows(path: Path, delimiter: str = "\t"):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def write_rows(path: Path, fields: list[str], records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def inventory():
    counts = Counter()
    examples = defaultdict(list)
    n = 0
    for row in rows(INPUT / "nodes.csv", ","):
        n += 1
        professions = json.loads(row["Professions"])
        assert len(professions) == len(set(professions)) and professions
        counts.update(professions)
        for value in professions:
            if len(examples[value]) < 3:
                examples[value].append(row["Name"])
    assert n == 100756 and len(counts) == 1817
    old = defaultdict(Counter)
    old_l2 = defaultdict(Counter)
    for row in rows(PROJECT / "artifacts/nodes.csv", ","):
        value = norm(row["occupation_level3"])
        old[value][row["occupation_level1"]] += 1
        old_l2[row["occupation_level1"]][row["occupation_level2"]] += 1
    draft = {r["raw_value"]: r for r in rows(
        PROJECT / "docs/freebase_easy_audit_2026-09-28/profession_crosswalk_draft_v1/profession_crosswalk_draft.tsv")}
    records = []
    for rank, (value, count) in enumerate(counts.most_common(), 1):
        hint = draft[value]
        records.append({
            "rank": rank, "raw_value": value, "graph_people": count,
            "sample_people_json": json.dumps(examples[value], ensure_ascii=False),
            "legacy_l1_counts_json": json.dumps(old.get(norm(value), {}), ensure_ascii=False),
            "legacy_draft_status": hint["draft_status"],
            "legacy_suggested_level1": hint["suggested_level1"],
            "legacy_suggested_level2": hint["suggested_level2"],
            "batch_id": f"batch_{(rank - 1) % 3 + 1}",
        })
    write_rows(REVIEW / "profession_inventory.tsv", list(records[0]), records)
    for batch in range(1, 4):
        write_rows(REVIEW / f"batch_{batch}_input.tsv", list(records[0]),
                   [r for r in records if r["batch_id"] == f"batch_{batch}"])
    (REVIEW / "legacy_level2_reference.json").write_text(
        json.dumps(old_l2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"people": n, "professions": len(counts),
                      "person_profession_pairs": sum(counts.values()),
                      "review_dir": str(REVIEW)}, ensure_ascii=False))


def period(row):
    years = json.loads(row["BirthYears"])
    if len(years) != 1 or years[0] > 2026:
        return "missing_or_invalid_birth"
    year = years[0]
    for end, label in [(1500, "<=1500"), (1900, "1501-1900"), (1920, "1901-1920"),
                       (1940, "1921-1940"), (1960, "1941-1960"), (1980, "1961-1980"),
                       (2000, "1981-2000"), (2026, "2001-2026")]:
        if year <= end:
            return label


def resolve_professions(raw, mapping):
    """An unresolved value blocks a unique label even if mapped values agree."""
    if not raw or len(raw) != len(set(raw)):
        raise ValueError("Expected a nonempty set of raw professions")
    labels = sorted({mapping[v]["proposed_level1"] for v in raw if mapping[v]["status"] == "proposed"})
    blocked = [v for v in raw if mapping[v]["status"] != "proposed"]
    if blocked:
        status = "incomplete_mapping"
    elif len(labels) == 1:
        status = "proposed_single_l1"
    else:
        assert len(labels) > 1
        status = "proposed_multiple_l1"
    chosen = labels[0] if status == "proposed_single_l1" else ""
    return labels, blocked, status, chosen


def merge():
    inv = {r["raw_value"]: r for r in rows(REVIEW / "profession_inventory.tsv")}
    mapping = {}
    for batch in range(1, 4):
        for row in rows(REVIEW / f"batch_{batch}_review.tsv"):
            value = row["raw_value"]
            assert value in inv and value not in mapping, f"Unknown/duplicate profession: {value}"
            assert inv[value]["batch_id"] == f"batch_{batch}", f"Wrong batch: {value}"
            assert row["status"] in {"proposed", "needs_review", "out_of_scope"}
            assert row["confidence"] in {"high", "medium", "low"}
            assert row["rationale"] and row["evidence"] and row["rule_id"]
            if row["status"] == "proposed":
                assert row["proposed_level1"] in LEVELS and row["confidence"] != "low"
            else:
                assert not row["proposed_level1"], f"Unresolved profession has label: {value}"
            mapping[value] = row
    assert set(mapping) == set(inv), f"Missing professions: {sorted(set(inv) - set(mapping))}"
    primary_reviewed = set()
    changes = []
    primary_path = REVIEW / "primary_review.tsv"
    if primary_path.exists():
        for row in rows(primary_path):
            value = row["raw_value"]
            assert value in mapping and value not in primary_reviewed
            assert row["status"] in {"proposed", "needs_review", "out_of_scope"}
            assert row["confidence"] in {"high", "medium", "low"}
            assert row["rationale"] and row["evidence"] and row["rule_id"]
            if row["status"] == "proposed":
                assert row["proposed_level1"] in LEVELS and row["confidence"] != "low"
            else:
                assert not row["proposed_level1"]
            previous = mapping[value]
            if any(row[key] != previous[key] for key in ("proposed_level1", "status", "confidence")):
                changes.append({"raw_value": value, "before_level1": previous["proposed_level1"],
                                "before_status": previous["status"], "before_confidence": previous["confidence"],
                                "after_level1": row["proposed_level1"], "after_status": row["status"],
                                "after_confidence": row["confidence"], "rationale": row["rationale"]})
            mapping[value] = row
            primary_reviewed.add(value)
    write_rows(REVIEW / "primary_changes.tsv", ["raw_value", "before_level1", "before_status", "before_confidence",
               "after_level1", "after_status", "after_confidence", "rationale"], changes)
    fields = ["rank", "raw_value", "graph_people", "proposed_level1", "status", "confidence",
              "rationale", "evidence", "rule_id", "batch_id", "legacy_draft_status",
              "legacy_suggested_level1", "legacy_suggested_level2", "legacy_l1_counts_json",
              "sample_people_json", "primary_agent_reviewed", "human_review_status", "review_version"]
    combined = []
    for value, source in inv.items():
        item = {**source, **mapping[value], "graph_people": source["graph_people"],
                "batch_id": source["batch_id"], "human_review_status": "not_reviewed",
                "primary_agent_reviewed": int(value in primary_reviewed),
                "review_version": "freebase_l1_agent_draft_v1"}
        combined.append({key: item[key] for key in fields})
    write_rows(REVIEW / "profession_l1_crosswalk_draft.tsv", fields, combined)
    write_rows(REVIEW / "professions_needing_review.tsv", fields,
               [r for r in combined if r["status"] != "proposed" or r["confidence"] != "high"])
    people = []
    raw_dist, mapped_dist, statuses, label_counts, unresolved = (Counter() for _ in range(5))
    by_period = defaultdict(Counter)
    name_status = {}
    name_raw_single = {}
    sole_blocker = Counter()
    for source in rows(INPUT / "nodes.csv", ","):
        raw = json.loads(source["Professions"])
        labels, blocked, status, chosen = resolve_professions(raw, mapping)
        name_status[source["Name"]] = status
        name_raw_single[source["Name"]] = len(raw) == 1
        raw_dist[len(raw)] += 1
        if not blocked:
            mapped_dist[len(labels)] += 1
        statuses[status] += 1
        label_counts.update([chosen] if chosen else [])
        unresolved.update(blocked)
        if len(blocked) == 1 and len(labels) == 1:
            sole_blocker[blocked[0]] += 1
        era = period(source)
        by_period[era]["people"] += 1
        by_period[era][status] += 1
        by_period[era]["single_raw"] += len(raw) == 1
        people.append({
            "person_name": source["Name"], "freebase_id": source["FreebaseID"],
            "raw_professions_json": source["Professions"],
            "proposed_l1_set_json": json.dumps(labels, ensure_ascii=False),
            "unresolved_professions_json": json.dumps(blocked, ensure_ascii=False),
            "proposed_single_l1": chosen, "resolution_status": status,
            "human_review_status": "not_reviewed", "birth_period": era,
            "review_version": "freebase_l1_agent_draft_v1",
        })
    assert len(people) == 100756 and len(name_status) == len(people)
    write_rows(DERIVED / "person_l1_audit.tsv", list(people[0]), people)
    priority_rows = [{"raw_value": value, "people_with_this_blocker": count,
                      "sole_blocker_with_one_known_l1_people": sole_blocker[value],
                      "status": mapping[value]["status"], "rationale": mapping[value]["rationale"]}
                     for value, count in unresolved.most_common()]
    write_rows(REVIEW / "unresolved_person_coverage.tsv", ["raw_value", "people_with_this_blocker",
               "sole_blocker_with_one_known_l1_people", "status", "rationale"], priority_rows)
    groups = defaultdict(Counter)
    members = defaultdict(set)
    for row in rows(INPUT / "main_relation_facts.csv", ","):
        group = row["RelationGroup"]
        a, b = row["Node1_Name"], row["Node2_Name"]
        assert a in name_status and b in name_status
        groups[group]["raw_facts"] += 1
        groups[group]["both_proposed_single_l1"] += (
            name_status[a] == name_status[b] == "proposed_single_l1")
        groups[group]["both_single_raw"] += name_raw_single[a] and name_raw_single[b]
        members[group].update((a, b))
    group_rows = [{"relation_group": group, **dict(counts), "people": len(members[group]),
                   "proposed_single_l1_people": sum(name_status[x] == "proposed_single_l1" for x in members[group]),
                   "single_raw_people": sum(name_raw_single[x] for x in members[group])}
                  for group, counts in sorted(groups.items())]
    write_rows(REVIEW / "relation_group_coverage.tsv", list(group_rows[0]), group_rows)
    period_rows = [{"birth_period": era, **{k: counts[k] for k in
                   ("people", "single_raw", "proposed_single_l1", "proposed_multiple_l1", "incomplete_mapping")}}
                   for era, counts in sorted(by_period.items())]
    write_rows(REVIEW / "birth_period_coverage.tsv", list(period_rows[0]), period_rows)
    report = {
        "version": "freebase_l1_agent_draft_v1", "status": "agent_review_draft_not_training_labels",
        "input_nodes": str(INPUT / "nodes.csv"), "input_facts": str(INPUT / "main_relation_facts.csv"),
        "people": len(people), "professions": len(mapping),
        "mapping_status_counts": dict(Counter(r["status"] for r in combined)),
        "mapping_confidence_counts": dict(Counter(r["confidence"] for r in combined)),
        "primary_agent_reviewed_professions": len(primary_reviewed),
        "primary_agent_changes": len(changes),
        "person_resolution_status_counts": dict(statuses), "single_l1_label_counts": dict(label_counts),
        "raw_profession_count_distribution": dict(sorted(raw_dist.items())),
        "l1_count_distribution_complete_mapping_only": dict(sorted(mapped_dist.items())),
        "top_unresolved_professions_people_counts": unresolved.most_common(25),
        "top_sole_blocker_with_one_known_l1": sole_blocker.most_common(25),
        "relation_groups": group_rows, "birth_periods": period_rows,
        "audit_output": str(DERIVED / "person_l1_audit.tsv"),
        "policies": ["Preserve all raw people, professions and relation facts.",
                     "Every raw profession must have a proposed valid L1 for a provisional single label.",
                     "Other/Missing and non-proposed mappings block a provisional single label.",
                     "No labels are inferred from neighbours or person identities.",
                     "All mapping proposals are agent drafts, not human-approved or frozen training inputs."],
    }
    (REVIEW / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_report(report)
    print(json.dumps({k: report[k] for k in ("people", "professions", "mapping_status_counts",
                                           "person_resolution_status_counts", "single_l1_label_counts")}, ensure_ascii=False))


def render_report(summary):
    n = summary["people"]
    states = summary["person_resolution_status_counts"]
    lines = ["# Freebase 当前关系图职业 L1 映射初审", "",
             "日期：2026-10-03。状态：**智能体候选草案，尚未人工批准或冻结训练标签**。", "",
             "## 范围与方法", "",
             f"只处理当前 11 个主分析谓词涉及的 {n:,} 个人物名称、1,817 个原始职业值。保留全部人物、原始职业和 145,876 条原始关系事实。", "",
             "候选目标是已有项目四个 L1：Culture、Discovery/Science、Leadership、Sports/Games。Leadership 沿用旧本体的广义范围，包含政治、法律、军事、宗教、贵族及企业经营，不等于狭义的组织领导岗位。", "",
             "三批由 GPT-6 Luna（high）并行提出词义/规则候选，保留各批方法与证据；主智能体复核高频词及跨批边界。自动规则、旧文本候选、智能体语义判断分别记录，均不冒充人工核验。没有逐个人核实职业事实或身份。", "",
             f"主智能体二次复核 {summary['primary_agent_reviewed_professions']:,} 个值，改变类别/状态/置信度 {summary['primary_agent_changes']:,} 项。前 100 个词覆盖 90.50% 人物—职业值配对，前 150 个覆盖 93.98%；这不是独立人物覆盖率。", "",
             "## 职业值初审", "", "| 状态 | 职业值数 |", "| --- | ---: |"]
    for status, count in sorted(summary["mapping_status_counts"].items()):
        lines.append(f"| {status} | {count:,} |")
    lines += ["", "## 人物层面的候选标签", "",
              "只有全部原始职业均有有效大类候选时才判定映射完整。任何待审值或四类之外的值都阻止赋唯一标签，不把其静默删去。", "",
              "| 状态 | 人数 | 占全部人物 |", "| --- | ---: | ---: |",
              f"| 原始职业只有一个（对照，仍须检查本体归属） | 33,254 | {33254/n:.2%} |"]
    for status in ("proposed_single_l1", "proposed_multiple_l1", "incomplete_mapping"):
        count = states.get(status, 0)
        lines.append(f"| {status} | {count:,} | {count/n:.2%} |")
    lines += ["", "多职业可落入同一 L1 不代表存在唯一细职业或主职业；未完整映射也不代表无职业。", "",
              "| 唯一候选大类 | 人数 |", "| --- | ---: |"]
    for label, count in sorted(summary["single_l1_label_counts"].items()):
        lines.append(f"| {label} | {count:,} |")
    lines += ["", "## 按关系组的覆盖", "",
              "原始关系全部保留。两端均为唯一候选大类的事实数仅用于衡量标签覆盖，不是删边指令，且含原始双向重复、自环。", "",
              "| 关系组 | 人物 | 原始单职业人物 | 唯一候选 L1 人物 | 原始事实 | 两端原始单职业 | 两端唯一候选 L1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["relation_groups"]:
        lines.append("| " + " | ".join(str(row[k]) for k in ("relation_group", "people", "single_raw_people",
                      "proposed_single_l1_people", "raw_facts", "both_single_raw", "both_proposed_single_l1")) + " |")
    lines += ["", "## 按出生时期的覆盖", "",
              "时期只按唯一出生年分箱，不能解释为职业或关系发生时期。", "",
              "| 出生时期 | 人物 | 原始单职业 | 唯一候选 L1 | 多候选 L1 | 映射未完整 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["birth_periods"]:
        lines.append("| " + " | ".join(str(row[k]) for k in ("birth_period", "people", "single_raw", "proposed_single_l1",
                      "proposed_multiple_l1", "incomplete_mapping")) + " |")
    lines += ["", "## 优先复核的阻塞值", "",
              "下表按受该未解决职业影响的人数排序，同一人可出现于多行。更完整的优先表见 unresolved_person_coverage.tsv；单一阻塞值的计数仅表示复核潜在价值，不保证能够获得唯一标签。", "",
              "| 原始职业 | 受影响人数 |", "| --- | ---: |"]
    for value, count in summary["top_unresolved_professions_people_counts"]:
        lines.append(f"| {value.replace('|', '/')} | {count:,} |")
    lines += ["", "## 文件与复现", "",
              "- profession_l1_crosswalk_draft.tsv：全部 1,817 行候选映射，含原值、人数、理由、证据、规则及待人工审核标记。",
              "- professions_needing_review.tsv：非候选或非高置信度的待审项。",
              "- primary_review.tsv / primary_changes.tsv：主智能体复核结果与对原批次决定的修订记录。",
              "- batch_N_input.tsv / batch_N_review.tsv：各批输入和原始子智能体决定。",
              "- summary.json、relation_group_coverage.tsv、birth_period_coverage.tsv：精确计数与覆盖。",
              f"- `{summary['audit_output']}`：全部 {n:,} 人的派生审计表，原始职业数组仍在，训练图未生成。", "",
              "```bash", "python Freebase/review_professions.py inventory", "# 子智能体/审核者修改各批及 primary_review.tsv 后：",
              "python Freebase/review_professions.py merge", "python -m unittest Freebase.test_review_professions -v", "```", "",
              "后续优先审定高频边界与中置信度条目，再冻结本体和映射；要比较保守/宽松方案，另开版本并保留本版。不得按人物邻居、预测结果或测试性能修改目标标签。"]
    (REVIEW / "review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "merge"])
    args = parser.parse_args()
    inventory() if args.stage == "inventory" else merge()
