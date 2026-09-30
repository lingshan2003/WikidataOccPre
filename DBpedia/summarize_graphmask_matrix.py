#!/usr/bin/env python3
"""Summarize the full-graph and life-period DBpedia GraphMask test reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = (
    "context", "period_label", "nodes", "directed_edges", "labeled_nodes",
    "classes", "test_nodes", "graphmask_roots", "original_accuracy",
    "original_macro_f1", "masked_accuracy", "masked_macro_f1",
    "prediction_agreement", "hard_retention_rate", "mean_keep_probability",
    "mean_kl", "report_dir",
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required summary/report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-artifact", type=Path, required=True)
    parser.add_argument("--period-root", type=Path, required=True)
    parser.add_argument("--graphmask-root", type=Path, required=True)
    parser.add_argument("--period-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    periods = read_json(args.period_config)["periods"]
    contexts = [("full", "Full graph", args.full_artifact)] + [
        (period["id"], period["label"], args.period_root / period["id"])
        for period in periods
    ]
    rows = []
    for context, label, artifact_dir in contexts:
        artifact = read_json(artifact_dir / "split_summary.json")
        report_dir = args.graphmask_root / context / f"seed_{args.seed}" / "test_report"
        metrics = read_json(report_dir / "test_metrics.json")
        if metrics.get("split") != "test":
            raise ValueError(f"Expected a test report for {context}: {report_dir}")
        test_nodes = artifact["test_nodes"]
        if metrics["roots"] != test_nodes or metrics["labeled_roots"] != test_nodes:
            raise ValueError(
                f"GraphMask test roots for {context} do not match the graph split: "
                f"{metrics['roots']}/{metrics['labeled_roots']} vs {test_nodes}"
            )
        rows.append({
            "context": context,
            "period_label": label,
            "nodes": artifact["nodes"],
            "directed_edges": artifact.get(
                "directed_edges", artifact.get("edges_after_reverse_and_deduplication")
            ),
            "labeled_nodes": artifact["labeled_nodes_retained"],
            "classes": artifact.get(
                "active_target_classes", artifact.get("target_classes_retained")
            ),
            "test_nodes": test_nodes,
            "graphmask_roots": metrics["roots"],
            "original_accuracy": metrics["original"]["accuracy"],
            "original_macro_f1": metrics["original"]["macro_f1"],
            "masked_accuracy": metrics["masked"]["accuracy"],
            "masked_macro_f1": metrics["masked"]["macro_f1"],
            "prediction_agreement": metrics["prediction_agreement"],
            "hard_retention_rate": metrics["hard_retention_rate"],
            "mean_keep_probability": metrics["mean_keep_probability"],
            "mean_kl": metrics["mean_kl"],
            "report_dir": str(report_dir),
        })

    args.graphmask_root.mkdir(parents=True, exist_ok=True)
    tsv_path = args.graphmask_root / "graphmask_matrix_summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.graphmask_root / "graphmask_matrix_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {len(rows)} graph contexts: {tsv_path} and {json_path}")


if __name__ == "__main__":
    main()
