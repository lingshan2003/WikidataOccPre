#!/usr/bin/env python3
"""Report official-QID overlap using the production life-window membership rule."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from training.life_periods import load_life_period_config, life_period_membership


def main():
    out = ROOT / "outputs/person_overlap_20261020"
    summary = json.loads((out / "official_overlap_summary.json").read_text())
    db = pd.read_csv(summary["source_files"]["dbpedia"])
    wd = pd.read_csv(summary["source_files"]["wikidata"])
    with (out / "confirmed_overlap_people.csv").open(encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    mapping = {r["dbpedia_uri"]: r["wikidata_id"] for r in pairs}
    calendar = load_life_period_config(ROOT / "config/l1_only_life_windows_1900_2000_pm20_step1_alive2026_v1.json")
    dm, _ = life_period_membership(db, calendar)
    wm, _ = life_period_membership(wd, calendar)
    def measure(label, dmask, wmask):
        dactive = set(db.loc[dmask, "node_id"])
        wactive = set(wd.loc[wmask, "node_id"])
        dshared = {uri for uri in dactive if uri in mapping and mapping[uri] in wactive}
        shared = {mapping[uri] for uri in dshared}
        return dict(scope=label, dbpedia_nodes=len(dactive), wikidata_nodes=len(wactive),
                    matched_dbpedia_nodes=len(dshared), shared_unique_qids=len(shared),
                    dbpedia_overlap_pct=100 * len(dshared) / len(dactive),
                    wikidata_overlap_pct=100 * len(shared) / len(wactive))
    annual = [measure(p.identifier, dm[p.identifier], wm[p.identifier]) for p in calendar.periods]
    # Confirm that reconstructed memberships describe the completed runs.
    for dataset, key in (("dbpedia", "dbpedia_nodes"), ("wikidata", "wikidata_nodes")):
        for row in annual:
            p = ROOT / "artifacts/l1_only_results_20261020" / dataset / "relations" / row["scope"] / "multi_group/split_summary.json"
            expected = json.loads(p.read_text())["nodes"]
            if row[key] != expected:
                raise ValueError(f"Membership differs from actual run: {dataset}/{row['scope']}")
    selected = [measure("full_source_graph", np.ones(len(db), dtype=bool), np.ones(len(wd), dtype=bool)),
                measure("union_of_1900_2000_windows", np.logical_or.reduce(list(dm.values())), np.logical_or.reduce(list(wm.values()))) ]
    selected += [r for r in annual if r["scope"] in ("center_1900", "center_1950", "center_2000")]
    if selected[0]["shared_unique_qids"] != summary["matched_unique_wikidata_persons"]:
        raise ValueError("Full-graph intersection disagrees with official mapping summary")
    for name, rows in (("annual_window_overlap.csv", annual), ("overlap_by_scope.csv", selected)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    names = {"full_source_graph": "完整源图", "union_of_1900_2000_windows": "1900–2000滑窗涉及人物的并集",
             "center_1900": "1900窗口（1880–1920）", "center_1950": "1950窗口（1930–1970）", "center_2000": "2000窗口（1980–2020）"}
    lines = ["# DBpedia 与 Wikidata 人物重合统计", "", "按DBpedia官方 owl:sameAs 的唯一 Wikidata QID 匹配；未使用名称候选填补身份映射。", "",
             "| 范围 | DBpedia节点 | Wikidata节点 | 共同人物（QID去重） | 对应DBpedia节点 | DBpedia重合比例 | Wikidata重合比例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected:
        lines.append(f"| {names[r['scope']]} | {r['dbpedia_nodes']:,} | {r['wikidata_nodes']:,} | {r['shared_unique_qids']:,} | {r['matched_dbpedia_nodes']:,} | {r['dbpedia_overlap_pct']:.2f}% | {r['wikidata_overlap_pct']:.2f}% |")
    lines += ["", f"官方唯一QID映射覆盖 {summary['dbpedia_nodes_with_unique_official_qid']:,}/{summary['dbpedia_nodes']:,} 个DBpedia节点。"
              f"未取得QID的节点 {summary['dbpedia_nodes_with_no_official_qid']:,} 个，多QID歧义节点 {summary['dbpedia_nodes_with_multiple_official_qids']:,} 个，均未自动计入重合。",
              "", "映射来自2026-10-20查询的当前官方端点；DBpedia人物源图来自2022-12版本。未映射、映射歧义和旧QID重定向可能使直接QID交集低估实际人物重合，不能把未匹配项全部解释为独有人物。",
              "", "滑窗统计复用生产代码 training.life_periods.life_period_membership，并逐窗核对两套已完成实验的节点数。仍沿用±20年、出生>1920且死亡日期缺失时存活至2026的构图规则。并集按人物去重，不是101窗人数相加。",
              "", "明细：confirmed_overlap_people.csv；逐窗统计：annual_window_overlap.csv；官方原始查询及结果：official_mapping_batches/。",
              "", "复现：先运行 scripts/audit_dbpedia_wikidata_overlap.py，再用项目Python环境运行 scripts/summarize_person_overlap_windows.py。已完成的官方查询会从缓存复用，无需重新查询。"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
