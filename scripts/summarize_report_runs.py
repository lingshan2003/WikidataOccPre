#!/usr/bin/env python3
"""Summarise per-seed report runs into reproducible mean/std CSV tables."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_report/level3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Report root does not exist: {root}")
    records: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/seed_*/metrics.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        test = payload.get("test")
        if not test:
            continue
        records.append({
            "experiment": path.parent.parent.name,
            "seed": path.parent.name.removeprefix("seed_"),
            **{metric: float(test[metric]) for metric in METRICS},
            "metrics_path": str(path),
        })
    if not records:
        raise RuntimeError("No completed test metrics found. Runs must omit --skip-test.")

    seed_path = root / "report_seed_metrics.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["experiment"]), []).append(record)
    summaries = []
    for experiment, rows in grouped.items():
        summary: Dict[str, object] = {
            "experiment": experiment,
            "seeds_completed": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    summaries.sort(key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    summary_path = root / "report_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {seed_path} and {summary_path}")
    for row in summaries:
        print(
            f"{row['experiment']:42s} "
            f"accuracy={float(row['accuracy_mean']):.4f}±{float(row['accuracy_std']):.4f} "
            f"macro_f1={float(row['macro_f1_mean']):.4f}±{float(row['macro_f1_std']):.4f} "
            f"n={row['seeds_completed']}"
        )


if __name__ == "__main__":
    main()
