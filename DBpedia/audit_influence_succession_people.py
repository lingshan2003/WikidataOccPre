#!/usr/bin/env python3
"""List people and raw/canonical pairs behind DBpedia influence/succession ties.

Uses only existing CSV graph artifacts. Canonical pairs are an audit view and
do not modify graphs, group mappings or trained models.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ("influenced", "influencedBy", "successor", "predecessor")
THEMES = {"influenced": "influence", "influencedBy": "influence", "successor": "succession", "predecessor": "succession"}
MEANINGS = {
    "influenced": "A influenced B: A influences B",
    "influencedBy": "A influencedBy B: B influences A",
    "successor": "A successor B: B succeeds A",
    "predecessor": "A predecessor B: A succeeds B",
}
EXAMPLES = (
    ("Aristotle", "influencedBy", "Plato"),
    ("Immanuel_Kant", "influencedBy", "David_Hume"),
    ("Bertrand_Russell", "influenced", "Alan_Turing"),
    ("Aaron_Beck", "influenced", "Martin_Seligman"),
    ("Christian_Michelsen", "successor", "Gunnar_Knudsen"),
    ("John_McEwen", "successor", "John_Gorton"),
    ("Pope_Benedict_XVI", "successor", "Pope_Francis"),
    ("Jay_Cutler_(bodybuilder)", "successor", "Phil_Heath"),
    ("Iris_Kyle", "predecessor", "Lenda_Murray"),
    ("Iris_Kyle", "successor", "Andrea_Shaw"),
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def name(uri):
    return unquote(uri.rsplit("/", 1)[-1]).replace("_", " ")


def canonical(row):
    source, target = row["source"], row["target"]
    if row["relation"] in ("influencedBy", "predecessor"):
        source, target = target, source
    return THEMES[row["relation"]], source, target


def markdown(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "artifacts/dbpedia_2022_priority_v1")
    parser.add_argument("--period-root", type=Path, default=ROOT / "artifacts/dbpedia_2022_priority_periods_20y_v1")
    parser.add_argument("--period-config", type=Path, default=ROOT / "config/dbpedia_life_periods_20y_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dbpedia_influence_succession_people_2026_10_07")
    args = parser.parse_args()
    nodes = {r["node_id"]: r for r in read_csv(args.graph_dir / "nodes.csv")}
    original = [r for r in read_csv(args.graph_dir / "edges.csv") if r["relation"] in RELATIONS]
    keys = {(r["source"], r["relation"], r["target"]) for r in original}
    if len(keys) != len(original):
        raise ValueError("Duplicate original triples")
    if any(r[k] not in nodes for r in original for k in ("source", "target")):
        raise ValueError("A relationship endpoint is missing from nodes.csv")
    periods = json.loads(args.period_config.read_text(encoding="utf-8"))["periods"]
    memberships, period_rows = defaultdict(list), []
    for p in periods:
        path = args.period_root / p["id"] / "edges.csv"
        if not path.is_file():
            raise ValueError(f"Missing period graph: {path}")
        rows = [r for r in read_csv(path) if r["relation"] in RELATIONS]
        for r in rows:
            key = (r["source"], r["relation"], r["target"])
            if key not in keys:
                raise ValueError(f"Period edge absent from the full graph: {key}")
            memberships[key].append(p["id"])
        count = Counter(r["relation"] for r in rows)
        pairs = {canonical(r) for r in rows}
        period_rows.append({"period": p["id"], **{r: count[r] for r in RELATIONS},
                            "original_triples": len(rows), "canonical_influence_pairs": sum(t == "influence" for t, _, _ in pairs),
                            "canonical_succession_pairs": sum(t == "succession" for t, _, _ in pairs)})
    canonical_records, raw_rows = defaultdict(list), []
    relation_count = Counter(r["relation"] for r in original)
    source_degrees, target_degrees = Counter(), Counter()
    for r in original:
        theme, first, second = canonical(r)
        canonical_records[(theme, first, second)].append(r)
        source_degrees[(r["source"], r["relation"])] += 1
        target_degrees[(r["target"], r["relation"])] += 1
        row = {"source_name": name(r["source"]), "relation": r["relation"], "target_name": name(r["target"]),
               "source_uri": r["source"], "target_uri": r["target"], "theme": theme,
               "canonical_first_name": name(first), "canonical_second_name": name(second),
               "canonical_direction": "influencer_to_influenced" if theme == "influence" else "earlier_to_later_holder",
               "period_graphs": ";".join(memberships[(r["source"], r["relation"], r["target"])])}
        for prefix in ("source", "target"):
            node = nodes[r[prefix]]
            for field in ("occupation_level1", "birth_year", "death_year"):
                row[prefix + "_" + field] = node[field]
            row[prefix + "_uri_has_double_underscore"] = "__" in r[prefix]
        raw_rows.append(row)
    canonical_rows, incoming, outgoing = [], Counter(), Counter()
    for (theme, first, second), supporting in sorted(canonical_records.items()):
        outgoing[(theme, first)] += 1
        incoming[(theme, second)] += 1
        canonical_rows.append({"theme": theme, "first_name": name(first), "second_name": name(second),
                               "first_uri": first, "second_uri": second,
                               "direction": "influencer_to_influenced" if theme == "influence" else "earlier_to_later_holder",
                               "supporting_original_triples": len(supporting),
                               "original_predicates": ";".join(sorted({r["relation"] for r in supporting})),
                               "period_graphs": ";".join(p["id"] for p in periods if any(p["id"] in memberships[(r["source"], r["relation"], r["target"])] for r in supporting)),
                               "first_occupation_label": nodes[first]["occupation_level1"], "second_occupation_label": nodes[second]["occupation_level1"]})
    people = sorted({r[k] for r in original for k in ("source", "target")})
    people_rows = []
    for uri in people:
        row = {"name": name(uri), "uri": uri, "occupation_label": nodes[uri]["occupation_level1"],
               "birth_year": nodes[uri]["birth_year"], "death_year": nodes[uri]["death_year"],
               "uri_has_double_underscore": "__" in uri}
        for r in RELATIONS:
            row[r + "_as_source"] = source_degrees[(uri, r)]
            row[r + "_as_target"] = target_degrees[(uri, r)]
        for theme in ("influence", "succession"):
            row[theme + "_canonical_out_degree"] = outgoing[(theme, uri)]
            row[theme + "_canonical_in_degree"] = incoming[(theme, uri)]
        people_rows.append(row)
    rank_rows = []
    for theme in ("influence", "succession"):
        for direction, counter in (("outgoing", outgoing), ("incoming", incoming)):
            ranked = sorted(((uri, count) for (t, uri), count in counter.items() if t == theme), key=lambda item: (-item[1], item[0]))
            for rank, (uri, count) in enumerate(ranked, 1):
                rank_rows.append({"theme": theme, "direction": direction, "rank": rank, "name": name(uri),
                                  "uri": uri, "occupation_label": nodes[uri]["occupation_level1"], "canonical_neighbors": count})
    relation_summary = [{"relation": rel, "meaning": MEANINGS[rel], "original_triples": relation_count[rel],
                         "distinct_endpoint_nodes": len({r[k] for r in original if r["relation"] == rel for k in ("source", "target")}),
                         "share_of_four_relation_triples": relation_count[rel] / len(original)} for rel in RELATIONS]
    theme_summary = []
    for theme in ("influence", "succession"):
        rr = [r for r in original if THEMES[r["relation"]] == theme]
        participants = {r[k] for r in rr for k in ("source", "target")}
        theme_summary.append({"theme": theme, "original_triples": len(rr), "distinct_endpoint_nodes": len(participants),
                              "canonical_pairs": sum(t == theme for t, _, _ in canonical_records),
                              "endpoint_occupation_labels": dict(Counter(nodes[u]["occupation_level1"] for u in participants).most_common())})
    examples = []
    for source, rel, target in EXAMPLES:
        match = next((r for r in raw_rows if r["source_uri"].rsplit("/", 1)[-1] == source and r["relation"] == rel and r["target_uri"].rsplit("/", 1)[-1] == target), None)
        if match is None:
            raise ValueError(f"Example not present in dataset: {(source, rel, target)}")
        examples.append(match)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("original_triples", raw_rows), ("canonical_pairs", canonical_rows), ("people", people_rows),
                           ("people_rankings", rank_rows), ("relation_summary", relation_summary), ("period_relation_summary", period_rows), ("examples", examples)):
        write_csv(output / (filename + ".csv"), rows)
    summary = {"graph_dir": str(args.graph_dir.resolve()), "period_root": str(args.period_root.resolve()),
               "original_triples": len(original), "distinct_endpoint_nodes": len(people), "canonical_pairs": len(canonical_rows),
               "relations": relation_summary, "themes": theme_summary,
               "name_policy": "Decoded DBpedia URI local name, not a separately retrieved label",
               "node_policy": "Distinct URI nodes; double-underscore identifiers preserved, no person/entity resolution",
               "canonical_policy": "Reverse influencedBy and predecessor, deduplicate identical theme/source/target; do not merge opposite succession directions or infer office/term/event identity",
               "scope": "Original graph records, excluding generated __rev message types; not GraphMask importance scores"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    top = [r for r in rank_rows if r["theme"] == "influence" and r["direction"] == "outgoing" and r["rank"] <= 10]
    report = [
        "# DBpedia influence / succession 人物探查",
        f"原图四种关系共 {len(original):,} 条三元组，涉及 {len(people):,} 个不同 URI 节点。未统计自动生成的 __rev。",
        markdown(["关系", "原始三元组", "涉及节点", "方向解释"], [[r["relation"], r["original_triples"], r["distinct_endpoint_nodes"], r["meaning"]] for r in relation_summary]),
        "## 图中真实存在的例子",
        markdown(["Source", "Relation", "Target", "Source 标签", "Target 标签"], [[r["source_name"], r["relation"], r["target_name"], r["source_occupation_level1"], r["target_occupation_level1"]] for r in examples]),
        "## 统一影响方向后的高连接人物",
        "下表按不同被影响者 URI 数排序。同一事实的 influenced 与反向 influencedBy 记录只计一次。",
        markdown(["人物", "不同被影响者节点数", "当前职业标签"], [[r["name"], r["canonical_neighbors"], r["occupation_label"]] for r in top]),
        "## 主题构成",
        markdown(["主题", "原始三元组", "统一方向后的节点对", "不同参与节点"], [[r["theme"], r["original_triples"], r["canonical_pairs"], r["distinct_endpoint_nodes"]] for r in theme_summary]),
        "影响组主要覆盖 scientist、philosopher、economist 等当前职业标签。接替组主要覆盖 politician、cleric，也包含体育头衔前后任等情形，因此不应只解释为政治任职接替。",
        "## 文件与口径",
        "original_triples.csv 保留完整人物对、URI、职业标签、生卒年、实际出现的时期图；people.csv 是全体参与节点名单；people_rankings.csv 是统一方向后的完整排名；canonical_pairs.csv 保留统一方向节点对及原始谓词支持信息；period_relation_summary.csv 按原始时期图统计。",
        "职业字段是当前实验按优先级选择的单标签，不是人物完整身份。名字来自 URI 解码，含 __ 的节点保留原样，未擅自合并成同一自然人。节点对缺少具体任期或头衔上下文，统一方向去重只用于检查节点对，不代表去重后的历史事件次数。",
        "Influenced 关系不要求双方同时在世，Successor/Predecessor 也不意味着师承或思想影响。时期图按人物生命区间重叠构造，不是关系发生时间。这里的度数和名单描述原始图，不能当作 GraphMask 的人物重要性排名。",
        "方向定义参考 [influenced](https://dbpedia.org/ontology/influenced)、[influencedBy](https://mappings.dbpedia.org/index.php/OntologyProperty:InfluencedBy)、[successor](https://mappings.dbpedia.org/index.php/OntologyProperty:Successor)、[predecessor](https://dbpedia.org/ontology/predecessor)。",
        "## 复现\n\n```bash\n.venv/bin/python DBpedia/audit_influence_succession_people.py\n```",
    ]
    (output / "README.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    bundle = output / "dbpedia_influence_succession_people.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        for p in output.iterdir():
            if p.is_file() and p.suffix != ".zip":
                archive.write(p, p.name)
        archive.write(Path(__file__), "audit_influence_succession_people.py")
    print(json.dumps({"original_triples": len(original), "distinct_endpoint_nodes": len(people), "canonical_pairs": len(canonical_rows),
                      "themes": [{k: v for k, v in t.items() if k != "endpoint_occupation_labels"} for t in theme_summary],
                      "output_dir": str(output), "bundle_bytes": bundle.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
