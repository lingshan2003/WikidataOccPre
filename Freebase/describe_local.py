#!/usr/bin/env python3
"""Describe the downloaded Freebase Easy v2 cohort and raw relation facts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "external_data/freebase/descriptive_v2_local"
DEFAULT_OUTPUT = ROOT / "docs/freebase_easy_audit_2026-10-01/descriptive_report.md"
RULES = Path(__file__).with_name("relation_rules.json")
SYMMETRIC = {"sibling", "partner", "peer", "celebrity_friend", "celebrity_romantic_relationship"}
PERIODS = (
    ("≤1500", None, 1500), ("1501–1900", 1501, 1900),
    ("1901–1920", 1901, 1920), ("1921–1940", 1921, 1940),
    ("1941–1960", 1941, 1960), ("1961–1980", 1961, 1980),
    ("1981–2000", 1981, 2000), ("2001–2026", 2001, 2026),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def summary(root: Path, stage: str) -> dict:
    return json.loads((root / stage / "summary.json").read_text(encoding="utf-8"))


def csv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def tsv_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def period(years: list[int]) -> str:
    if not years:
        return "缺出生年"
    if len(years) != 1:
        return "出生年冲突"
    year = years[0]
    if year > 2026:
        return "晚于2026"
    return next(label for label, start, end in PERIODS if (start is None or year >= start) and year <= end)


def table(headers: tuple[str, ...], data: list[tuple]) -> list[str]:
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in data)]


def main() -> None:
    args = options()
    root = args.input_dir.expanduser().resolve()
    summaries = {stage: summary(root, stage) for stage in
                 ("01_people", "02_cohort", "03_pairs", "04_relations", "05_final", "06_stats")}
    rules = {row["predicate"]: row for row in json.loads(RULES.read_text(encoding="utf-8"))["relations"]}
    cohort_counts = Counter()
    cohort_occupations = Counter()
    for row in tsv_rows(root / "02_cohort/cohort_people.tsv"):
        cohort_counts["rows"] += 1
        occupations = json.loads(row["professions_json"])
        if not occupations:
            raise ValueError("Cohort person without a profession")
        cohort_counts["multi_profession"] += len(occupations) > 1
        cohort_occupations.update(occupations)
    if cohort_counts["rows"] != summaries["02_cohort"]["cohort_people"]:
        raise ValueError("Cohort count disagrees with stage summary")

    all_predicates = Counter()
    all_pair_count = 0
    pair_lines = set()
    for row in tsv_rows(root / "03_pairs/cohort_name_pairs.tsv"):
        all_pair_count += 1
        all_predicates[row["predicate"]] += 1
        pair_lines.add(row["source_line"])
    if all_pair_count != summaries["03_pairs"]["cohort_person_pairs"]:
        raise ValueError("Cohort pair count disagrees with stage summary")

    review_count = 0
    review_predicates = Counter()
    for row in tsv_rows(root / "04_relations/review_candidate_facts.tsv"):
        review_count += 1
        review_predicates[row["predicate"]] += 1
        if row["source_line"] not in pair_lines:
            raise ValueError("Review fact missing from cohort pairs")
    if review_count != summaries["04_relations"]["review_facts"]:
        raise ValueError("Review count disagrees with stage summary")

    nodes: dict[str, dict] = {}
    node_periods = Counter()
    main_occupations = Counter()
    date_flags = Counter()
    for row in csv_rows(root / "05_final/nodes.csv"):
        name = row["Name"]
        if name in nodes:
            raise ValueError(f"Duplicate node name: {name}")
        occupations = json.loads(row["Professions"])
        births = json.loads(row["BirthYears"])
        deaths = json.loads(row["DeathYears"])
        if not occupations or not (births or deaths):
            raise ValueError("Final node violates cohort rule")
        nodes[name] = {"births": births, "deaths": deaths, "professions": occupations,
                       "id_status": row["IDStatus"]}
        node_periods[period(births)] += 1
        main_occupations.update(occupations)
        date_flags["multi_profession"] += len(occupations) > 1
        date_flags["death_before_birth"] += bool(births and deaths and deaths[0] < births[0])
        date_flags["age_over_125"] += bool(births and deaths and deaths[0] - births[0] > 125)
        date_flags["future_death"] += bool(deaths and deaths[0] > 2026)
    if len(nodes) != summaries["05_final"]["relation_endpoint_names"]:
        raise ValueError("Final node count disagrees with stage summary")

    raw = Counter()
    relation = Counter()
    groups = Counter()
    nonself_groups = Counter()
    unique_edges: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    same_period_facts = Counter()
    period_facts_other = Counter()
    self_facts = Counter()
    endpoint_names = set()
    for row in csv_rows(root / "05_final/main_relation_facts.csv"):
        name1, name2 = row["Node1_Name"], row["Node2_Name"]
        if name1 not in nodes or name2 not in nodes or row["SourceLine"] not in pair_lines:
            raise ValueError("Final fact has unknown node or source line")
        predicate = row["RawPredicate"]
        rule = rules[predicate]
        if rule.get("main") != row["Relation"] or rule.get("group") != row["RelationGroup"]:
            raise ValueError("Final fact differs from local relation rules")
        raw[predicate] += 1
        relation[row["Relation"]] += 1
        group = row["RelationGroup"]
        groups[group] += 1
        endpoint_names.update((name1, name2))
        p1, p2 = period(nodes[name1]["births"]), period(nodes[name2]["births"])
        if p1 == p2 and p1 in {label for label, _, _ in PERIODS}:
            same_period_facts[p1] += 1
        elif p1 in {label for label, _, _ in PERIODS} and p2 in {label for label, _, _ in PERIODS}:
            period_facts_other["不同出生时期"] += 1
        else:
            period_facts_other["至少一端出生年缺失或异常"] += 1
        if name1 == name2:
            self_facts[group] += 1
            continue
        nonself_groups[group] += 1
        if rule.get("inverse_when_normalizing"):
            name1, name2 = name2, name1
        if row["Relation"] in SYMMETRIC:
            name1, name2 = sorted((name1, name2))
        unique_edges[group].add((row["Relation"], name1, name2))
    main_total = sum(raw.values())
    if (main_total != summaries["05_final"]["final_facts"]
            or main_total != summaries["06_stats"]["main_facts"]
            or len(endpoint_names) != len(nodes)):
        raise ValueError("Main relation totals disagree with summaries or node table")
    if dict(raw) != summaries["06_stats"]["main_facts_by_predicate"]:
        raise ValueError("Predicate counts disagree with stage stats")

    pct = lambda n, denominator: f"{n / denominator * 100:.2f}%" if denominator else "—"
    lines = ["# Freebase Easy v2 描述性统计", "",
             "数据：服务器 `Freebase/extract.py` 六阶段输出，下载包 `freebase_descriptive_v2.tar.gz`。"
             "统计单位是 Freebase Easy 的**名称匹配记录**；MID 只是辅助线索。",
             "", "## 抽取漏斗", ""]
    funnel = [
        ("`is-a Person` 候选名称", f"{summaries['01_people']['person_names']:,}"),
        ("≥1 个职业且有出生或死亡可解析年份", f"{cohort_counts['rows']:,}"),
        ("两端均在上述人物集合的原始三元组", f"{all_pair_count:,}（{len(all_predicates)} 种谓词）"),
        ("20 个初审谓词的原始三元组", f"{review_count:,}"),
        ("11 个主分析谓词的原始三元组", f"{main_total:,}（{len(relation)} 个规范关系名）"),
        ("11 谓词涉及的不同人物名称", f"{len(nodes):,}"),
    ]
    lines += table(("步骤", "数量"), funnel)
    lines += ["", f"598,078 人中有多个原始职业的为 **{cohort_counts['multi_profession']:,} 人**"
             f"（{pct(cohort_counts['multi_profession'], cohort_counts['rows'])}）；"
             "这些人物及其所有职业均保留。", "",
             "## 关系频次", "",
             f"下表以 11 个主分析谓词的 **{main_total:,} 条原始事实**为分母。双向存储仍算两条；"
             "这不是去重边数。", ""]
    lines += table(("原始谓词", "原始事实", "占比"),
                   [(name, f"{count:,}", pct(count, main_total)) for name, count in raw.most_common(10)])
    lines += ["", f"剩余第 11 个谓词为 `Martial Art Instructor(s)`：{raw['Martial Art Instructor(s)']:,} 条。",
             "", "作为关系筛选前的参照，全部 60 种谓词中前十名如下，分母为 148,445 条：", ""]
    lines += table(("原始谓词", "原始事实", "占比"),
                   [(name, f"{count:,}", pct(count, all_pair_count)) for name, count in all_predicates.most_common(10)])
    lines += ["", "## 关系组", "",
             f"11 谓词共 {main_total:,} 条原始事实，其中自环 {sum(self_facts.values()):,} 条。"
             "去自环后保留原始方向和反向重复；最后一列再按名称对折叠对称关系，"
             "并把哲学影响的逆向谓词调整方向后与相同规范关系合并。", ""]
    lines += table(("关系组", "原始事实", "去自环事实", "去自环占比", "规范去重名称边"),
                   [(group, f"{groups[group]:,}", f"{nonself_groups[group]:,}",
                     pct(nonself_groups[group], main_total - sum(self_facts.values())),
                     f"{len(unique_edges[group]):,}")
                    for group in ("kinship", "influence", "education", "social")])
    lines += ["", f"合计规范去重名称边 **{sum(map(len, unique_edges.values())):,}** 条。"
             "它是描述性去重计数，尚未处理同名实体、虚构人物或亲属冲突。", "",
             "## 按出生时期", "",
             "时期只依据**唯一出生年**划分；缺出生年者不被推定出生时期。"
             "右列仅计两端人物都出生在同一时期的原始关系事实，故跨时期关系另列。"
             "这些数字不代表关系发生的年份。", ""]
    lines += table(("出生时期", "主分析关系涉及人物", "两端同出生时期的原始事实"),
                   [(label, f"{node_periods[label]:,}", f"{same_period_facts[label]:,}")
                    for label, _, _ in PERIODS])
    lines += ["", f"未能分配出生时期：缺出生年 {node_periods['缺出生年']:,} 人，"
             f"出生年晚于 2026 年 {node_periods['晚于2026']:,} 人；"
             f"跨时期原始事实 {period_facts_other['不同出生时期']:,} 条，"
             f"至少一端无可用出生时期 {period_facts_other['至少一端出生年缺失或异常']:,} 条。",
             "", "## 数据质量与解释范围", "",
             f"- 关系人物中多职业者 {date_flags['multi_profession']:,} / {len(nodes):,} 人；"
             f"唯一 MID 的人物 {summaries['05_final']['names_with_unique_id']:,} 人，"
             f"缺 MID 的人物 {summaries['05_final']['names_without_id']:,} 人。",
             f"- 已发现卒年早于生年 {date_flags['death_before_birth']:,} 人、"
             f"生卒差超过 125 年 {date_flags['age_over_125']:,} 人、"
             f"卒年晚于 2026 年 {date_flags['future_death']:,} 人。以上仅为结构/数值审计，未自动删人。",
             "- 当前 11 谓词由亲属/伴侣占主导；`Influenced By` 表示思想或艺术影响，"
             "不等于双方实际相识。`Academic advisor` 保留原始方向。",
             "- 这次 `01_people` 摘要给出 3,970,878 个名称，先前旧版摘要为 3,970,877 个；"
             "报告统一采用本次完整重跑的摘要。压缩包未包含完整 Person 名单，无法在本地定位这一名称差异。",
             "", "## 核对", "",
             f"- `cohort_people.tsv`：{cohort_counts['rows']:,} 行，等于 02 阶段摘要。",
             f"- `cohort_name_pairs.tsv`：{all_pair_count:,} 行，等于 03 阶段摘要。",
             f"- `review_candidate_facts.tsv`：{review_count:,} 行，等于 04 阶段摘要。",
             f"- `main_relation_facts.csv`：{main_total:,} 行，等于 05/06 阶段摘要；"
             f"其人物集合与 `nodes.csv` 的 {len(nodes):,} 个名称完全相同。",
             ""]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
