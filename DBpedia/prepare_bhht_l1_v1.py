#!/usr/bin/env python3
"""Export an independent DBpedia L1 dataset with user-authorized Other fallback.

Uses only the standard library. Preserves original people, relation facts, dates,
and the existing single-occupation selection; does not create a split or train.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dbpedia_bhht_provisional_single_l1_v1"
LEVELS = {"Culture", "Discovery/Science", "Leadership", "Sports/Games", "Other"}
SOURCE = ROOT / "external_data/dbpedia/processed/model_input"
DEFAULT_MAPPING = ROOT / "docs/dbpedia_bhht_l1_semantic_review_2026-10-14/occupation_l1_semantic_crosswalk.tsv"
DEFAULT_OUTPUT = ROOT / "external_data/dbpedia/bhht_l1_provisional_v1"
NODE_FIELDS = ["node_id", "birth_year", "death_year", "occupation_level1", "occupation_level2", "occupation_level3", "country"]


def rows(path, delimiter=None):
    op = gzip.open if path.suffix == ".gz" else open
    sep = delimiter or ("\t" if ".tsv" in path.name else ",")
    with op(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=sep)


def write_rows(path, records, fields=None):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(records[0]),
                                delimiter="\t" if ".tsv" in path.name else ",")
        writer.writeheader()
        writer.writerows(records)


def load_mapping(path):
    mapping = {}
    for source in rows(path):
        label, level, state = source["occupation_label"], source["level1"], source["review_status"]
        if not label or label in mapping or int(source["graph_nodes"]) <= 0:
            raise ValueError(f"Invalid/duplicate occupation: {label!r}")
        if state not in {"resolved", "needs_context", "outside_this_review_scope"}:
            raise ValueError(f"Unexpected L1 review state: {label!r}")
        fallback = state == "needs_context"
        if fallback and level or not fallback and level not in LEVELS:
            raise ValueError(f"Inconsistent original L1/state: {label!r}")
        mapping[label] = {**source, "training_level1": "Other" if fallback else level,
                          "temporary_other_fallback": int(fallback),
                          "mapping_application": "temporary_unresolved_to_other" if fallback else source["level1_origin"],
                          "dataset_version": VERSION, "training_label_status": "user_authorized_provisional_v1"}
    if not mapping:
        raise ValueError("Empty occupation crosswalk")
    return mapping


def map_nodes(nodes, mapping, assignments):
    audit_by_person = {r["person_uri"]: r for r in assignments}
    if len(audit_by_person) != len(assignments):
        raise ValueError("Duplicate person in source occupation audit")
    seen, frequencies, output, audit = set(), Counter(), [], []
    for source in nodes:
        person, occupation = source["node_id"], source["occupation_level1"]
        if not person or person in seen:
            raise ValueError(f"Invalid/duplicate source node: {person!r}")
        seen.add(person)
        if occupation not in mapping or person not in audit_by_person:
            raise ValueError(f"Missing occupation mapping or assignment audit: {person}")
        assignment = audit_by_person[person]
        if assignment["occupation_label"] != occupation:
            raise ValueError(f"Original chosen occupation differs: {person}")
        decision = mapping[occupation]
        frequencies[occupation] += 1
        node = {key: source[key] for key in NODE_FIELDS}
        node.update(occupation_level1=decision["training_level1"], occupation_level2="", occupation_level3="")
        output.append(node)
        audit.append({**assignment, "original_bhht_level1": decision["level1"],
                      "training_level1": decision["training_level1"],
                      "temporary_other_fallback": decision["temporary_other_fallback"],
                      "original_l1_review_status": decision["review_status"],
                      "original_l1_confidence": decision["confidence"],
                      "original_l1_origin": decision["level1_origin"],
                      "semantic_rationale": decision["semantic_rationale"],
                      "mapping_application": decision["mapping_application"], "dataset_version": VERSION})
    expected = Counter({key: int(r["graph_nodes"]) for key, r in mapping.items()})
    if frequencies != expected or seen != set(audit_by_person):
        raise ValueError("Current graph population/frequencies differ from reviewed crosswalk")
    return output, audit


def rewrite_relations(source_path, output_path, facts_path, nodes, original_nodes):
    index = {r["node_id"]: r for r in nodes}
    original = {r["node_id"]: r for r in original_nodes}
    seen, relation_counts = set(), Counter()
    with source_path.open(encoding="utf-8-sig", newline="") as src, output_path.open("w", encoding="utf-8", newline="") as dst, facts_path.open("w", encoding="utf-8", newline="") as facts:
        reader = csv.DictReader(src)
        expected = {"Node1", "Relation", "Node2", "Node1_occ_level1", "Node2_occ_level1",
                    "Node1_occ_level2", "Node2_occ_level2", "Node1_occ_level3", "Node2_occ_level3"}
        if not expected <= set(reader.fieldnames or []):
            raise ValueError("Invalid extended input schema")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        fact_writer = csv.DictWriter(facts, fieldnames=["source_node", "relation", "target_node"], lineterminator="\n")
        fact_writer.writeheader()
        for row in reader:
            for side in ("Node1", "Node2"):
                person = row[side]
                if person not in index or row[f"{side}_occ_level1"] != original[person]["occupation_level1"]:
                    raise ValueError(f"Input edge occupation/node differs from original table: {person}")
                seen.add(person)
                row[f"{side}_occ_level1"] = index[person]["occupation_level1"]
                row[f"{side}_occ_level2"] = row[f"{side}_occ_level3"] = ""
            relation_counts[row["Relation"]] += 1
            writer.writerow(row)
            fact_writer.writerow({"source_node": row["Node1"], "relation": row["Relation"], "target_node": row["Node2"]})
    if seen != set(index):
        raise ValueError("Edge-centric input and original node populations differ")
    return relation_counts


def verify_export(source_path, output_path):
    count = 0
    # Independently compare every row, allowing changes only in occupation slots.
    a, b = rows(source_path), rows(output_path)
    from itertools import zip_longest
    for old, new in zip_longest(a, b):
        if old is None or new is None:
            raise ValueError("Relation facts were added or removed")
        for key in old:
            if "_occ_level" not in key and old[key] != new[key]:
                raise ValueError(f"Non-occupation field changed: {key}")
        if new["Node1_occ_level1"] not in LEVELS or new["Node2_occ_level1"] not in LEVELS:
            raise ValueError("Unassigned/invalid provisional L1")
        if any(new[key] for key in new if key.endswith(("_occ_level2", "_occ_level3"))):
            raise ValueError("Unexpected unreviewed fine labels")
        count += 1
    return count


def build(args):
    inputs = {"mapping": args.mapping, "source_nodes": args.source_nodes, "source_input": args.source_input,
              "assignments": args.assignments, "relation_mapping": args.relation_mapping}
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing dataset directory: {output}")
    archive = output.parent / (output.name + ".tar.gz")
    if archive.exists():
        raise ValueError(f"Refusing to overwrite existing data package: {archive}")
    mapping = load_mapping(args.mapping)
    original_nodes = list(rows(args.source_nodes))
    nodes, audit = map_nodes(original_nodes, mapping, list(rows(args.assignments)))
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "nodes.csv", nodes, NODE_FIELDS)
    write_rows(output / "node_label_audit.tsv.gz", audit)
    write_rows(output / "occupation_crosswalk.tsv", list(mapping.values()))
    write_rows(output / "occupation_crosswalk_compact.csv", [{"原职业": r["occupation_label"],
               "第一版L1": r["training_level1"], "图内节点数": r["graph_nodes"],
               "临时Other兜底": r["temporary_other_fallback"], "原审核状态": r["review_status"],
               "原语义L1": r["level1"], "原置信度": r["confidence"]} for r in mapping.values()])
    relations = rewrite_relations(args.source_input, output / "DBpedia_R_R_extended.csv", output / "relations.csv", nodes, original_nodes)
    original_relation_counts = {r["relation"]: int(r["original_directed_edges"]) for r in rows(args.relation_mapping)}
    if dict(relations) != original_relation_counts:
        raise ValueError("Original relation distribution changed")
    edges = verify_export(args.source_input, output / "DBpedia_R_R_extended.csv")
    assert edges == sum(relations.values())
    for old, new in zip(original_nodes, nodes):
        for key in ["node_id", "birth_year", "death_year", "country"]:
            if old[key] != new[key]:
                raise ValueError(f"Original node attribute changed: {key}")
    shutil.copy2(args.relation_mapping, output / "relation_mapping.tsv")
    total, formal, temporary = Counter(), Counter(), Counter()
    for r in audit:
        total[r["training_level1"]] += 1
        (temporary if r["temporary_other_fallback"] else formal)[r["training_level1"]] += 1
    write_rows(output / "level1_counts.tsv", [{"level1": label, "graph_nodes": total[label],
               "original_candidate_or_llm_nodes": formal[label], "temporary_other_nodes": temporary[label]}
              for label in sorted(LEVELS)])
    generator = Path(__file__).resolve()
    shutil.copy2(generator, output / "generator_snapshot.py")
    generator_revision = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(generator)], cwd=ROOT, text=True).strip()
    record = {"dataset_version": VERSION, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
              "graph_nodes": len(nodes), "original_relation_facts": edges, "base_relation_types": len(relations),
              "level1_classes": len(total), "level1_node_counts": dict(total),
              "occupation_values": len(mapping), "temporary_other_occupation_values": sum(r["temporary_other_fallback"] for r in mapping.values()),
              "temporary_other_nodes": temporary["Other"], "formal_other_nodes": formal["Other"],
              "temporary_other_node_fraction": temporary["Other"] / len(nodes),
              "label_policy": "preserve_selected_occupation_and_existing_l1_candidate_or_llm_proposal; unresolved_to_temporary_other",
              "l2_l3_policy": "deferred_and_empty_in_dataset", "dates": "original_recorded_dates_preserved; no_death_date_imputation",
              "nodes_relations_dates_country_preserved": True, "old_datasets_or_results_modified": False,
              "split_created": False, "graph_data_pt_created": False, "training_started": False,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "generator_git_revision": generator_revision or None, "generator_snapshot": "generator_snapshot.py",
              "inputs": {key: {"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for key, path in inputs.items()},
              "archive_name": archive.name}
    (output / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ["# DBpedia BHHT L1 第一版数据", "",
            f"{len(nodes):,} 人、{edges:,} 条原方向关系事实、{len(relations)} 种基础关系、{len(total)} 个L1职业类别。", "",
            f"Other共{total['Other']:,}人：已有查表候选/LLM建议Other为{formal['Other']:,}人，待补语境暂归Other为{temporary['Other']:,}人（全图{temporary['Other']/len(nodes):.2%}），对应{record['temporary_other_occupation_values']}种职业。",
            "原始审核状态、空的语义L1及未定理由全部保留。temporary_other_fallback=1只表示用户授权的临时训练标签，不表示LLM已确定为Other。", "",
            "已有唯一L1查表候选和已给出L1的LLM提案沿用上轮结果。人物原唯一职业选择、节点顺序、关系及全部原始生卒/国家字段保持不变。L2/L3暂空，不从不一致候选硬填。", "",
            "nodes.csv是逐人数据，relations.csv只含原方向事实，DBpedia_R_R_extended.csv兼容既有15列准备接口。预处理时生成的反向边不在本数据包中重复记录。",
            "occupation_crosswalk_compact.csv是逐职业查看表。occupation_crosswalk.tsv保留原职业和审查证据，新增training_level1与临时Other标记；node_label_audit.tsv.gz将这些标记落实到每个原人物。level1_counts.tsv区分已有Other与临时Other。generator_snapshot.py保留生成器源码快照，常规复现仍从仓库DBpedia/prepare_bhht_l1_v1.py执行。", "",
            "这一步仅生成数据，未划分新的训练/验证/测试集，未准备PyG权重或启动训练；新标签不能直接套用旧82类DBpedia模型。滑窗的出生>1920、缺死亡日期延至2026规则应继续在构图阶段应用，此处不修改原始死亡日期。", "",
            "```bash", "python DBpedia/prepare_bhht_l1_v1.py", "```", ""]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.is_file())
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=output.name + "/" + path.name)
    with tarfile.open(archive, "r:gz") as tar:
        if {m.name for m in tar.getmembers()} != {output.name + "/" + p.name for p in files}:
            raise ValueError("Data-package contents mismatch")
        for member in tar.getmembers():
            with tar.extractfile(member) as stream:
                while stream.read(1024 * 1024):
                    pass
    print(json.dumps({**record, "output_dir": str(output), "archive": str(archive), "archive_bytes": archive.stat().st_size}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--source-nodes", type=Path, default=ROOT / "artifacts/dbpedia_2022_priority_v1/nodes.csv")
    parser.add_argument("--source-input", type=Path, default=SOURCE / "DBpedia_R_R_extended.csv")
    parser.add_argument("--assignments", type=Path, default=SOURCE / "occupation_assignments.tsv.gz")
    parser.add_argument("--relation-mapping", type=Path, default=SOURCE / "relation_mapping.tsv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
