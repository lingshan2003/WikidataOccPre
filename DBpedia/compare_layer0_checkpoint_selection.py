#!/usr/bin/env python3
"""Compare the original GraphMask selection with the Layer-0-enabled supplement.

Only reads JSON/CSV statistics; no torch import or weight hashing is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def selected_record(directory, policy, threshold):
    history = read_json(directory / "training_history.json")["history"]
    candidates = [r for r in history
                  if r["validation"]["relative_macro_f1_difference"] <= threshold
                  and r["validation"]["hard_retention_rate"] is not None
                  and (policy == "any-stage" or r["enabled_through_layer"] == 0)]
    if not candidates:
        raise ValueError(f"No faithful {policy} candidate: {directory}")
    # Stable min reproduces the trainer's strict '<' tie-breaking rule.
    selected = min(candidates, key=lambda r: r["validation"]["hard_retention_rate"])
    validation = read_json(directory / "validation.json")
    for recorded, final in [(selected["validation"], validation),
                            *zip(selected["validation"]["layers"], validation["layers"])]:
        a, b = recorded["hard_retention_rate"], final["hard_retention_rate"]
        if a != b and (a is None or b is None or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"Selected epoch disagrees with final validation: {directory}")
    return selected, validation


def snapshot(directory, policy, threshold):
    manifest = read_json(directory / "manifest.json")
    selected, validation = selected_record(directory, policy, threshold)
    report_dir = directory / "test_report"
    metrics = read_json(report_dir / "test_metrics.json")
    report_manifest = read_json(report_dir / "manifest.json")
    for key in ("source_checkpoint", "data"):
        if report_manifest[key] != manifest[key]:
            raise ValueError(f"Report/probe {key} differs: {directory}")
    if policy == "all-layers-enabled":
        metadata = manifest.get("selected_checkpoint") or {}
        if metadata.get("policy") != policy or metadata.get("enabled_layers") != [True, True]:
            raise ValueError(f"Expected both layer gates enabled: {directory}")
        if metadata.get("global_epoch") != selected["global_epoch"]:
            raise ValueError(f"Selected epoch metadata disagrees with history: {directory}")
        if report_manifest.get("selected_checkpoint") != metadata or report_manifest.get("enabled_layers") != [True, True]:
            raise ValueError(f"Report/probe selection metadata differs: {directory}")
    layers = {row["layer"]: row for row in metrics["layers"]}
    fields = {
        "selected_epoch": selected["global_epoch"],
        "layer0_enabled": selected["enabled_through_layer"] == 0,
        "validation_relative_f1_difference": validation["relative_macro_f1_difference"],
        "validation_hard_retention_rate": validation["hard_retention_rate"],
        "test_macro_f1": metrics["masked"]["macro_f1"],
        "test_accuracy": metrics["masked"]["accuracy"],
        "prediction_agreement": metrics["prediction_agreement"],
        "test_hard_retention_rate": metrics["hard_retention_rate"],
        "layer0_hard_retention_rate": layers[0]["hard_retention_rate"],
        "layer1_hard_retention_rate": layers[1]["hard_retention_rate"],
    }
    groups = {}
    with (report_dir / "relations_base.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["layer"]) == 0:
                groups[row["relation"]] = row
    return fields, groups, manifest, report_manifest


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dbpedia_multi_group_layer0_enabled_5periods_v1.json")
    parser.add_argument("--original-root", default="runs_graphmask/dbpedia_grouped_20y_v1")
    parser.add_argument("--supplement-root", help="Override new probe root (e.g. extracted download)")
    parser.add_argument("--output-dir", help="Write comparison tables separately from the received reports")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    old_root = resolve(args.original_root)
    new_root = resolve(args.supplement_root or config["graphmask_root"])
    threshold = config["graphmask_train"]["max_relative_macro_f1_diff"]
    taxonomy = read_json(resolve(config["multi_group_taxonomy"]))
    rows, group_rows = [], []
    for period in config["periods"]:
        relative = Path(period) / "multi_group" / f"seed_{config['seed']}"
        old, old_groups, old_manifest, old_report = snapshot(old_root / relative, "any-stage", threshold)
        new, new_groups, new_manifest, new_report = snapshot(new_root / relative, "all-layers-enabled", threshold)
        for key in ("source_checkpoint", "data", "fanouts", "seed"):
            if old_manifest[key] != new_manifest[key]:
                raise ValueError(f"Different {key} across runs: {period}")
        for key in ("split", "fanouts", "sampling_seed"):
            if old_report[key] != new_report[key]:
                raise ValueError(f"Different report {key} across runs: {period}")
        for key in set(old_manifest["training_config"]) | set(new_manifest["training_config"]):
            if key not in ("output_dir", "checkpoint_selection", "device") and old_manifest["training_config"].get(key) != new_manifest["training_config"].get(key):
                raise ValueError(f"Different GraphMask training option {key} across runs: {period}")
        row = {"context": period, "representation": "multi_group", "seed": config["seed"]}
        row.update({"original_" + k: v for k, v in old.items()})
        row.update({"supplement_" + k: v for k, v in new.items()})
        rows.append(row)
        for group in taxonomy["groups"]:
            item = {"context": period, "group": group, "layer": 0}
            for label, groups in (("original", old_groups), ("supplement", new_groups)):
                obs = groups.get(group, {})
                item[label + "_message_observations"] = int(obs.get("message_observations", 0))
                for metric in ("hard_retention_rate", "retained_edge_share", "mean_keep_probability"):
                    item[label + "_" + metric] = float(obs[metric]) if obs.get(metric) else None
            group_rows.append(item)
    output = resolve(args.output_dir) if args.output_dir else new_root
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "checkpoint_selection_comparison.tsv", rows)
    write_table(output / "layer0_group_comparison.tsv", group_rows)
    (output / "checkpoint_selection_comparison.json").write_text(
        json.dumps({"original_root": str(old_root), "supplement_root": str(new_root),
                    "selection_metric": "Minimum overall validation hard retention subject to fidelity and stage eligibility",
                    "validation_fidelity_threshold": threshold, "periods": rows, "layer0_groups": group_rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(rows)} periods: {output / 'checkpoint_selection_comparison.tsv'}")


if __name__ == "__main__":
    main()
