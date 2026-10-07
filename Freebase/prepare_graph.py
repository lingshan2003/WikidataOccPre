#!/usr/bin/env python3
"""Build a separately versioned, provisional Freebase single-label artifact.

The remaining multi/uncertain people receive an explicitly arbitrary temporary
selection, as requested by the user. Raw occupations and the selection reason
remain in an audit; no raw extraction or review artifact is overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Freebase.review_professions import resolve_professions

DEFAULT_CONFIG = ROOT / "config/freebase_grouped_rgcn_graphmask_20y_v1.json"
SYMMETRIC = {"sibling", "partner", "peer", "celebrity_friend", "celebrity_romantic_relationship"}
POLICIES = {"mapped_first_other_fallback", "mapped_first_raw_fallback"}
VERSION = "freebase_provisional_single_l1_v1"


def path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_rows(p, delimiter=","):
    with Path(p).open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def write_rows(p, fields, records, delimiter=","):
    with Path(p).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(records)


def stamp(p):
    stat = p.stat()
    return {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_label(raw, mapping, previous, policy):
    """Select by recorded array order, never popularity, rarity or neighbours."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown label policy: {policy}")
    if not raw:
        raise ValueError("This cohort requires at least one recorded profession")
    labels, blocked, state, unique = resolve_professions(raw, mapping)
    if previous["resolution_status"] != state or previous["proposed_single_l1"] != unique:
        raise ValueError("Person audit and crosswalk disagree; regenerate/review the audit first")
    if json.loads(previous["proposed_l1_set_json"]) != labels or json.loads(previous["unresolved_professions_json"]) != blocked:
        raise ValueError("Person audit's candidate/unresolved sets differ from crosswalk")
    candidates = [v for v in raw if mapping[v]["status"] == "proposed"]
    if unique:
        return unique, candidates[0], "preserved_unique_candidate"
    if candidates:
        chosen = candidates[0]
        return mapping[chosen]["proposed_level1"], chosen, "forced_first_mapped_profession"
    chosen = raw[0]
    return ("Other" if policy == "mapped_first_other_fallback" else f"Raw::{chosen}"), chosen, "forced_unmapped_profession"


def single_year(raw, maximum):
    years = json.loads(raw)
    if len(years) != 1:
        return "", "missing" if not years else "conflicting_years"
    year = years[0]
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"Unexpected extracted year: {year!r}")
    if year > maximum:
        return "", "future_year"
    return year, "observed"


def load_node_tables(nodes_path, audit_path, crosswalk_path, *, policy, maximum_year):
    mapping = {}
    valid_l1 = {"Culture", "Discovery/Science", "Leadership", "Sports/Games"}
    for r in read_rows(crosswalk_path, "\t"):
        value = r["raw_value"]
        if value in mapping or r["status"] not in {"proposed", "needs_review", "out_of_scope"}:
            raise ValueError(f"Invalid/duplicate crosswalk value: {value}")
        if r["status"] == "proposed" and r["proposed_level1"] not in valid_l1:
            raise ValueError(f"Invalid proposed L1: {value}")
        if r["status"] != "proposed" and r["proposed_level1"]:
            raise ValueError(f"Unresolved crosswalk value has a label: {value}")
        mapping[value] = r
    audit = {}
    for r in read_rows(audit_path, "\t"):
        if r["person_name"] in audit:
            raise ValueError("Duplicate person in occupation audit")
        audit[r["person_name"]] = r
    nodes, selection, date_audit = [], [], []
    seen = set()
    for r in read_rows(nodes_path):
        name = r["Name"]
        if not name or name in seen or name not in audit:
            raise ValueError(f"Duplicate/empty/missing person: {name!r}")
        seen.add(name)
        raw = json.loads(r["Professions"])
        previous = audit[name]
        if json.loads(previous["raw_professions_json"]) != raw or previous["freebase_id"] != r["FreebaseID"]:
            raise ValueError(f"Audit raw occupations/ID differ for {name!r}")
        label, chosen, reason = select_label(raw, mapping, previous, policy)
        birth, birth_status = single_year(r["BirthYears"], maximum_year)
        death, death_status = single_year(r["DeathYears"], maximum_year)
        nodes.append({"node_id": name, "birth_year": birth, "death_year": death,
                      "occupation_level1": label, "occupation_level2": "", "occupation_level3": "",
                      "country": "", "freebase_id": r["FreebaseID"], "freebase_id_status": r["IDStatus"],
                      "raw_professions_json": r["Professions"], "selected_raw_profession": chosen,
                      "label_selection_reason": reason, "original_resolution_status": previous["resolution_status"]})
        selection.append({"person_name": name, "freebase_id": r["FreebaseID"],
                          "raw_professions_json": r["Professions"], "selected_raw_profession": chosen,
                          "temporary_target_label": label, "selection_reason": reason,
                          "original_resolution_status": previous["resolution_status"],
                          "selected_mapping_status": mapping[chosen]["status"],
                          "selected_mapping_confidence": mapping[chosen]["confidence"],
                          "label_policy": policy, "label_status": "experimental_provisional"})
        date_audit.append({"person_name": name, "birth_years_json": r["BirthYears"],
                           "death_years_json": r["DeathYears"], "birth_status": birth_status,
                           "death_status": death_status,
                           "death_before_birth": bool(birth != "" and death != "" and death < birth)})
    if seen != set(audit):
        raise ValueError("Node/audit person sets differ")
    return nodes, selection, date_audit


def normalize_relations(facts_path, rules_path, names):
    rules = {r["predicate"]: r for r in read_json(rules_path)["relations"] if r.get("main")}
    normalized, raw_counts = {}, Counter()
    self_loops = 0
    for r in read_rows(facts_path):
        predicate = r["RawPredicate"]
        if predicate not in rules:
            raise ValueError(f"Unexpected raw relation: {predicate}")
        rule = rules[predicate]
        a, b, relation = r["Node1_Name"], r["Node2_Name"], rule["main"]
        if a not in names or b not in names or r["Relation"] != relation or r["RelationGroup"] != rule["group"]:
            raise ValueError("Relation endpoint/config does not match extraction")
        raw_counts[relation] += 1
        if a == b:
            self_loops += 1
            continue
        if rule.get("inverse_when_normalizing"):
            a, b = b, a
        if relation in SYMMETRIC:
            a, b = sorted((a, b))
        key = (a, relation, b)
        if key not in normalized:
            normalized[key] = {"source": a, "relation": relation, "target": b,
                               "first_source_line": r["SourceLine"], "source_fact_count": 0}
        normalized[key]["source_fact_count"] += 1
    return list(normalized.values()), sorted({r["main"] for r in rules.values()}), {
        "raw_relation_facts": sum(raw_counts.values()), "raw_facts_by_relation": dict(raw_counts),
        "self_loop_facts_removed": self_loops, "normalized_nonself_relation_edges": len(normalized),
        "duplicates_removed_after_self_loops": sum(raw_counts.values()) - self_loops - len(normalized),
    }


def resolve_input_directory(configured):
    """Accept an explicit override; bridge only the two known export layouts."""
    override = os.environ.get("FREEBASE_INPUT_DIR")
    selected = path(override or configured)
    files = ("nodes.csv", "main_relation_facts.csv")
    local = path("external_data/freebase/descriptive_v2_local/05_final")
    server = path("external_data/freebase/processed/05_final")
    # Do not replace a custom path or mix a partial export with another batch.
    if not override and selected == local and not any((selected / f).exists() for f in files):
        if all((server / f).is_file() for f in files):
            print(f"[input] local download layout absent; using server export: {server}", flush=True)
            return server
    return selected


def settings(config, output_dir, tables_only):
    c = read_json(config)
    s = dict(c["source_prepare"])
    if s["label_policy"] not in POLICIES or s["min_class_count"] < 3:
        raise ValueError("Invalid label policy or min_class_count < 3")
    if not all(0 < s[k] < 1 for k in ("train_ratio", "val_ratio", "test_ratio")) or abs(sum(s[k] for k in ("train_ratio", "val_ratio", "test_ratio")) - 1) > 1e-8:
        raise ValueError("Source train/val/test ratios must be positive and sum to 1")
    output = path(output_dir) if output_dir else path(c["source_data"]).parent
    input_dir = resolve_input_directory(s["input_dir"])
    if os.environ.get("FREEBASE_INPUT_DIR") or input_dir != path(s["input_dir"]):
        s["input_dir"] = str(input_dir)
    inputs = {"nodes": input_dir / "nodes.csv", "facts": input_dir / "main_relation_facts.csv",
              "audit": path(s["audit_file"]), "crosswalk": path(s["crosswalk_file"]),
              "relation_rules": path(s.get("relation_rules", "Freebase/relation_rules.json"))}
    missing = [(key, p) for key, p in inputs.items() if not p.is_file()]
    if missing:
        details = "\n".join(f"  {key}: {p}" for key, p in missing)
        raise FileNotFoundError("Missing Freebase input files:\n" + details +
            "\nSet FREEBASE_INPUT_DIR to the directory containing nodes.csv and main_relation_facts.csv. "
            "The person_l1_audit.tsv is generated locally and ignored by Git; copy it to its configured path "
            "along with profession_l1_crosswalk_draft.tsv before running.")
    if any(output == p.parent or output in p.parents for p in inputs.values()):
        raise ValueError("Output directory overlaps an input directory")
    signature = {"schema_version": VERSION, "source_prepare": s, "tables_only": tables_only,
                 "inputs": {k: stamp(v) for k, v in inputs.items()},
                 "relation_rules": read_json(inputs["relation_rules"])}
    return s, output, inputs, signature


def prepare(config, output_dir=None, *, tables_only=False):
    s, output, inputs, signature = settings(config, output_dir, tables_only)
    complete = ("nodes.csv", "edges.csv", "split_summary.json", "source_prepare_manifest.json", "label_selection_audit.tsv",
                "date_audit.tsv", "normalized_relation_facts.csv", "class_stats.csv", "relation_stats.csv", "attribute_conflicts.csv")
    if not tables_only:
        complete += ("graph_data.pt",)
    manifest = output / "source_prepare_manifest.json"
    if output.exists():
        if all((output / f).is_file() and (output / f).stat().st_size > 0 for f in complete) and read_json(manifest) == signature:
            print(f"[reuse] compatible source artifact: {output}")
            return read_json(output / "split_summary.json")
        raise ValueError(f"Existing source output is incomplete/incompatible: {output}; use a new output directory")
    nodes, selection, date_audit = load_node_tables(inputs["nodes"], inputs["audit"], inputs["crosswalk"],
        policy=s["label_policy"], maximum_year=s.get("maximum_year", 2026))
    facts, vocabulary, relation_stats = normalize_relations(inputs["facts"], inputs["relation_rules"], {n["node_id"] for n in nodes})
    node_to_id = {n["node_id"]: i for i, n in enumerate(nodes)}
    relation_to_id = {r: i for i, r in enumerate(sorted(vocabulary + [r + "__rev" for r in vocabulary]))}
    edges = []
    for fact in facts:
        for a, relation, b in ((fact["source"], fact["relation"], fact["target"]),
                               (fact["target"], fact["relation"] + "__rev", fact["source"])):
            edges.append({"source": a, "relation": relation, "target": b,
                          "source_id": node_to_id[a], "target_id": node_to_id[b], "relation_id": relation_to_id[relation]})
    label_counts = Counter(n["occupation_level1"] for n in nodes)
    summary = {"target_column": "occupation_level1", "nodes": len(nodes),
               "edges_after_reverse_and_deduplication": len(edges), "relation_types": len(relation_to_id),
               "provisional_label_counts": dict(label_counts),
               "selection_reason_counts": dict(Counter(r["selection_reason"] for r in selection)),
               "original_resolution_status_counts": dict(Counter(r["original_resolution_status"] for r in selection)),
               "label_policy": s["label_policy"], "label_status": "experimental_provisional_not_ground_truth",
               "date_audit_status_counts": {k: dict(Counter(r[k] for r in date_audit)) for k in ("birth_status", "death_status")},
               "date_validation_maximum_year": s.get("maximum_year", 2026),
               "maximum_observed_year": max((n[k] for n in nodes for k in ("birth_year", "death_year") if n[k] != ""), default=None),
               **relation_stats}
    graph = metadata = None
    if not tables_only:
        import pandas as pd
        import torch
        from data.prepare import build_pyg_data, encode_labels, stratified_node_split
        node_frame, edge_frame = pd.DataFrame(nodes), pd.DataFrame(edges)
        for k in ("birth_year", "death_year"):
            node_frame[k] = pd.to_numeric(node_frame[k], errors="coerce")
        for k in ("occupation_level2", "occupation_level3", "country"):
            node_frame[k] = node_frame[k].replace("", pd.NA)
        labels, eligible, label_to_id, _ = encode_labels(node_frame, "occupation_level1", s["min_class_count"])
        split = stratified_node_split(labels, eligible, s["train_ratio"], s["val_ratio"], s["test_ratio"], s["seed"])
        graph, country_to_id, schema, unknown_ids, vocabularies = build_pyg_data(node_frame, edge_frame, labels, *split)
        metadata = {"target_column": "occupation_level1", "num_relations": len(relation_to_id),
                    "num_classes": len(label_to_id), "label_to_id": label_to_id, "relation_to_id": relation_to_id,
                    "feature_schema": {**schema, "country": {"kind": "categorical", "cardinality": len(country_to_id)},
                                       "temporal": {"kind": "numeric", "input_dim": int(graph.temporal.size(1))}},
                    "country_to_id": country_to_id, "occupation_unknown_ids": unknown_ids,
                    "occupation_vocabularies": vocabularies, "seed": s["seed"],
                    "freebase_provisional_labels": summary, "source_prepare": signature,
                    "identity_policy": "original_unique_name_keys_with_auxiliary_freebase_ids"}
        summary.update({"target_classes_retained": len(label_to_id), "labeled_nodes_retained": int(eligible.sum()),
                        "ignored_unlabeled_or_rare_nodes": int((~eligible).sum()), "seed": s["seed"],
                        "min_class_count": s["min_class_count"],
                        **{k: int(v.sum()) for k, v in zip(("train_nodes", "val_nodes", "test_nodes"), split)},
                        "occupation_feature_protocol": "Only training labels visible as neighbour L1; val/test L1 unknown; current seed L1 masked at every forward. L2/L3 and country missing."})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging directories let a rerun recover after a killed writer.
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    write_rows(temporary / "nodes.csv", list(nodes[0]), nodes)
    write_rows(temporary / "edges.csv", ["source", "relation", "target", "source_id", "target_id", "relation_id"], edges)
    write_rows(temporary / "normalized_relation_facts.csv", ["source", "relation", "target", "first_source_line", "source_fact_count"], facts)
    write_rows(temporary / "label_selection_audit.tsv", list(selection[0]), selection, "\t")
    write_rows(temporary / "date_audit.tsv", list(date_audit[0]), date_audit, "\t")
    write_rows(temporary / "attribute_conflicts.csv", ["node_id", "attribute", "values"], [])
    write_rows(temporary / "class_stats.csv", ["occupation", "count"], [{"occupation": k, "count": v} for k, v in sorted(label_counts.items())])
    edge_counts = Counter(r["relation"] for r in edges)
    write_rows(temporary / "relation_stats.csv", ["relation", "count"], [{"relation": k, "count": v} for k, v in sorted(edge_counts.items())])
    (temporary / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (temporary / "source_prepare_manifest.json").write_text(json.dumps(signature, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if graph is not None:
        import torch
        torch.save({"data": graph, "metadata": metadata}, temporary / "graph_data.pt")
    os.replace(temporary, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--tables-only", action="store_true", help="Use a separate output directory to inspect provisional labels without PyTorch")
    args = p.parse_args()
    try:
        prepare(path(args.config), args.output_dir, tables_only=args.tables_only)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
