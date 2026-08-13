#!/usr/bin/env python3
"""Stratify completed Level-1 tie-audit predictions by editable birth cohorts.

This analysis never trains or re-evaluates a GNN.  It joins each completed
run's ``test_predictions.csv`` to the ``nodes.csv`` paired with the canonical
graph artifact, then repeats the existing same-model/same-seed comparisons
inside each birth cohort.  The final relation-specific statistic is:

    Macro-F1(random matched to group) - Macro-F1(drop group)

Positive values mean deleting the named tie group harms that birth cohort more
than deleting an equal number of cohort-incident random relation pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ``python scripts/<name>.py`` places scripts/ rather than the project root on
# sys.path.  Keep this standalone server entry point able to import training/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.birth_cohorts import (
    DEFAULT_BIRTH_COHORT_CONFIG_PATH,
    MISSING_COHORT_ID,
    load_artifact_birth_cohorts,
    load_birth_cohort_config,
)


MODELS = ("rgcn", "rgat")
SEEDS = ("42", "43", "44")
CONDITIONS = (
    "without_inherited",
    "random_matched_inherited",
    "without_acquired",
    "random_matched_acquired",
)
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_report/level1/tie_audit")
    parser.add_argument("--baseline-root", default="runs_report/level1")
    parser.add_argument("--data", default="artifacts/level1_hierarchy/graph_data.pt")
    parser.add_argument("--birth-cohorts", default=str(DEFAULT_BIRTH_COHORT_CONFIG_PATH))
    parser.add_argument(
        "--targeted-root",
        default=None,
        help=(
            "Optional root of cohort-incident deletion runs named "
            "<model>__cohort_<cohort_id>__<condition>. When set, summarise only "
            "the matching cohort's test nodes rather than stratifying global ablations."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <root>/birth_cohort_summary",
    )
    return parser.parse_args()


def _prediction_path(root: Path, model: str, condition: str, seed: str) -> Path:
    if condition == "full":
        return root / f"{model}_baseline" / f"seed_{seed}" / "test_predictions.csv"
    return root / f"{model}__occupation_neighbours__{condition}" / f"seed_{seed}" / "test_predictions.csv"


def _metrics_path(root: Path, model: str, condition: str, seed: str) -> Path:
    if condition == "full":
        return root / f"{model}_baseline" / f"seed_{seed}" / "metrics.json"
    return root / f"{model}__occupation_neighbours__{condition}" / f"seed_{seed}" / "metrics.json"


def _targeted_prediction_path(root: Path, model: str, cohort_id: str, condition: str, seed: str) -> Path:
    return root / f"{model}__cohort_{cohort_id}__{condition}" / f"seed_{seed}" / "test_predictions.csv"


def _targeted_metrics_path(root: Path, model: str, cohort_id: str, condition: str, seed: str) -> Path:
    return root / f"{model}__cohort_{cohort_id}__{condition}" / f"seed_{seed}" / "metrics.json"


def _read_predictions(path: Path) -> Dict[int, Tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed test predictions are missing: {path}")
    required = {"node_index", "true_label", "prediction"}
    result: Dict[int, Tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction file missing {sorted(missing)}: {path}")
        for row in reader:
            node_index = int(row["node_index"])
            if node_index in result:
                raise ValueError(f"Duplicate node_index {node_index} in {path}")
            result[node_index] = (row["true_label"], row["prediction"])
    if not result:
        raise ValueError(f"Prediction file contains no rows: {path}")
    return result


def _metric_values(true_labels: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    labels = sorted(set(true_labels) | set(predictions))
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predictions, labels=labels, average="weighted", zero_division=0)),
    }


def _label_support(true_labels: Sequence[str]) -> Dict[str, object]:
    counts = dict(sorted(Counter(true_labels).items()))
    return {
        "n_true_labels": len(counts),
        "min_true_label_support": min(counts.values()),
        "true_label_support": json.dumps(counts, ensure_ascii=False, sort_keys=True),
    }


def _read_provenance(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metrics JSON must contain an object: {path}")
    relation = payload.get("relation_perturbation")
    return relation if isinstance(relation, Mapping) else {}


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    keys: List[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _mean_std(rows: Sequence[Mapping[str, object]], key: str) -> Tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def per_cohort_records(
    root: Path,
    baseline_root: Path,
    cohort_nodes,
    cohort_manifest: Mapping[str, object],
) -> List[Dict[str, object]]:
    cohort_by_node = cohort_nodes.set_index("node_index")
    records: List[Dict[str, object]] = []
    reference_labels: Dict[Tuple[str, str], Dict[int, str]] = {}
    for model in MODELS:
        for seed in SEEDS:
            for condition in ("full", *CONDITIONS):
                run_root = baseline_root if condition == "full" else root
                prediction_path = _prediction_path(run_root, model, condition, seed)
                metrics_path = _metrics_path(run_root, model, condition, seed)
                predictions = _read_predictions(prediction_path)
                labels = {node_index: true_label for node_index, (true_label, _) in predictions.items()}
                reference_key = (model, seed)
                if reference_key in reference_labels and labels != reference_labels[reference_key]:
                    raise ValueError(
                        f"Test nodes or true labels differ from complete-graph baseline: {prediction_path}"
                    )
                reference_labels[reference_key] = labels
                unknown_nodes = set(predictions) - set(cohort_by_node.index)
                if unknown_nodes:
                    raise ValueError(
                        f"Prediction contains node indices absent from nodes.csv, e.g. {sorted(unknown_nodes)[:5]}"
                    )
                provenance = _read_provenance(metrics_path)
                selected = cohort_by_node.loc[list(predictions)]
                for cohort_id, frame in selected.groupby("birth_cohort", sort=False):
                    node_indices = frame.index.to_list()
                    true_labels = [predictions[index][0] for index in node_indices]
                    predicted_labels = [predictions[index][1] for index in node_indices]
                    metrics = _metric_values(true_labels, predicted_labels)
                    records.append({
                        "model": model,
                        "seed": seed,
                        "condition": condition,
                        "birth_cohort": cohort_id,
                        "birth_cohort_label": str(frame["birth_cohort_label"].iloc[0]),
                        "included_in_cohort_hypothesis": bool(frame["included_in_cohort_hypothesis"].iloc[0]),
                        "n_test_nodes": len(node_indices),
                        **_label_support(true_labels),
                        "prediction_path": str(prediction_path),
                        "metrics_path": str(metrics_path),
                        "birth_cohort_config_name": cohort_manifest["name"],
                        "birth_cohort_config_sha256": cohort_manifest["sha256"],
                        "data_sha256": provenance.get("data_sha256"),
                        "tie_taxonomy_sha256": (
                            provenance.get("tie_taxonomy", {}).get("sha256")
                            if isinstance(provenance.get("tie_taxonomy"), Mapping) else None
                        ),
                        **metrics,
                    })
    return records


def targeted_cohort_records(
    targeted_root: Path,
    baseline_root: Path,
    cohort_nodes,
    cohort_manifest: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Score only the cohort whose incident edges were removed in each run."""
    cohort_by_node = cohort_nodes.set_index("node_index")
    configured_cohorts = cohort_manifest["bins"]
    if not isinstance(configured_cohorts, list):
        raise ValueError("Invalid birth-cohort manifest")
    records: List[Dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            baseline_path = _prediction_path(baseline_root, model, "full", seed)
            baseline_predictions = _read_predictions(baseline_path)
            for cohort_definition in configured_cohorts:
                cohort_id = str(cohort_definition["id"])
                cohort_nodes_in_test = [
                    node_index for node_index in baseline_predictions
                    if str(cohort_by_node.at[node_index, "birth_cohort"]) == cohort_id
                ]
                if not cohort_nodes_in_test:
                    raise ValueError(f"No test nodes belong to configured cohort {cohort_id!r}")
                cohort_label = str(cohort_by_node.at[cohort_nodes_in_test[0], "birth_cohort_label"])
                for condition in ("full", *CONDITIONS):
                    if condition == "full":
                        prediction_path = baseline_path
                        metrics_path = _metrics_path(baseline_root, model, condition, seed)
                        predictions = baseline_predictions
                    else:
                        prediction_path = _targeted_prediction_path(
                            targeted_root, model, cohort_id, condition, seed
                        )
                        metrics_path = _targeted_metrics_path(
                            targeted_root, model, cohort_id, condition, seed
                        )
                        predictions = _read_predictions(prediction_path)
                        baseline_labels = {
                            node_index: baseline_predictions[node_index][0]
                            for node_index in baseline_predictions
                        }
                        labels = {node_index: item[0] for node_index, item in predictions.items()}
                        if labels != baseline_labels:
                            raise ValueError(
                                f"Test nodes or true labels differ from complete-graph baseline: {prediction_path}"
                            )
                        provenance = _read_provenance(metrics_path)
                        edge_cohort = provenance.get("edge_cohort")
                        if not isinstance(edge_cohort, Mapping) or edge_cohort.get("selected_cohort_id") != cohort_id:
                            raise ValueError(
                                f"Run lacks matching cohort-edge provenance ({cohort_id!r}): {metrics_path}"
                            )
                    provenance = _read_provenance(metrics_path)
                    true_labels = [predictions[node_index][0] for node_index in cohort_nodes_in_test]
                    predicted_labels = [predictions[node_index][1] for node_index in cohort_nodes_in_test]
                    records.append({
                        "model": model,
                        "seed": seed,
                        "condition": condition,
                        "birth_cohort": cohort_id,
                        "birth_cohort_label": cohort_label,
                        "included_in_cohort_hypothesis": True,
                        "n_test_nodes": len(cohort_nodes_in_test),
                        **_label_support(true_labels),
                        "prediction_path": str(prediction_path),
                        "metrics_path": str(metrics_path),
                        "birth_cohort_config_name": cohort_manifest["name"],
                        "birth_cohort_config_sha256": cohort_manifest["sha256"],
                        "data_sha256": provenance.get("data_sha256"),
                        "tie_taxonomy_sha256": (
                            provenance.get("tie_taxonomy", {}).get("sha256")
                            if isinstance(provenance.get("tie_taxonomy"), Mapping) else None
                        ),
                        "intervention_scope": (
                            provenance.get("edge_cohort", {}).get("edge_scope")
                            if isinstance(provenance.get("edge_cohort"), Mapping) else "complete_graph_baseline"
                        ),
                        **_metric_values(true_labels, predicted_labels),
                    })
    return records


def paired_records(records: Iterable[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    indexed = {
        (str(row["model"]), str(row["seed"]), str(row["condition"]), str(row["birth_cohort"])): row
        for row in records
    }
    deltas: List[Dict[str, object]] = []
    specificity: List[Dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            cohort_ids = sorted({
                key[3] for key in indexed if key[0] == model and key[1] == seed and key[2] == "full"
            })
            for cohort_id in cohort_ids:
                baseline = indexed[(model, seed, "full", cohort_id)]
                for condition in CONDITIONS:
                    current = indexed.get((model, seed, condition, cohort_id))
                    if current is None:
                        raise ValueError(f"Missing cohort result: {model}/{seed}/{condition}/{cohort_id}")
                    if int(current["n_test_nodes"]) != int(baseline["n_test_nodes"]):
                        raise ValueError(f"Test-node count changed within a paired cohort: {model}/{seed}/{cohort_id}")
                    row = {
                        "model": model,
                        "seed": seed,
                        "condition": condition,
                        "birth_cohort": cohort_id,
                        "birth_cohort_label": baseline["birth_cohort_label"],
                        "included_in_cohort_hypothesis": baseline["included_in_cohort_hypothesis"],
                        "n_test_nodes": baseline["n_test_nodes"],
                        "n_true_labels": baseline["n_true_labels"],
                    }
                    for metric in METRICS:
                        row[f"baseline_{metric}"] = baseline[metric]
                        row[f"{metric}_delta"] = float(current[metric]) - float(baseline[metric])
                    deltas.append(row)
                for tie_group, dropped, matched_random in (
                    ("inherited", "without_inherited", "random_matched_inherited"),
                    ("acquired", "without_acquired", "random_matched_acquired"),
                ):
                    dropped_row = indexed[(model, seed, dropped, cohort_id)]
                    control_row = indexed[(model, seed, matched_random, cohort_id)]
                    specificity.append({
                        "model": model,
                        "seed": seed,
                        "tie_group": tie_group,
                        "birth_cohort": cohort_id,
                        "birth_cohort_label": baseline["birth_cohort_label"],
                        "included_in_cohort_hypothesis": baseline["included_in_cohort_hypothesis"],
                        "n_test_nodes": baseline["n_test_nodes"],
                        "n_true_labels": baseline["n_true_labels"],
                        "macro_f1_drop": dropped_row["macro_f1"],
                        "macro_f1_random_matched": control_row["macro_f1"],
                        "relationship_specific_macro_f1_loss": (
                            float(control_row["macro_f1"]) - float(dropped_row["macro_f1"])
                        ),
                    })
    return deltas, specificity


def summarise(rows: Iterable[Mapping[str, object]], group_keys: Sequence[str], numeric_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    result: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        summary = {field: value for field, value in zip(group_keys, key)}
        first = group[0]
        for field in ("birth_cohort_label", "included_in_cohort_hypothesis", "n_test_nodes", "n_true_labels"):
            if field in first:
                values = {str(row[field]) for row in group}
                if len(values) != 1:
                    raise ValueError(f"Inconsistent {field} within summary group {key}")
                summary[field] = first[field]
        summary["seeds_completed"] = len(group)
        summary["seeds"] = ",".join(str(row["seed"]) for row in group)
        for numeric in numeric_keys:
            summary[f"{numeric}_mean"], summary[f"{numeric}_std"] = _mean_std(group, numeric)
        result.append(summary)
    return result


def main() -> None:
    args = parse_args()
    root, baseline_root, data_path = Path(args.root), Path(args.baseline_root), Path(args.data)
    targeted_root = Path(args.targeted_root) if args.targeted_root else None
    if not root.is_dir() or not baseline_root.is_dir():
        raise FileNotFoundError("Both --root and --baseline-root must be existing directories")
    if targeted_root is not None and not targeted_root.is_dir():
        raise FileNotFoundError(f"--targeted-root must be an existing directory: {targeted_root}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Canonical Level-1 artifact is required: {data_path}")
    cohort_config = load_birth_cohort_config(args.birth_cohorts)
    cohort_nodes = load_artifact_birth_cohorts(data_path, cohort_config)
    default_output = targeted_root / "birth_cohort_summary" if targeted_root is not None else root / "birth_cohort_summary"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_manifest = cohort_config.manifest()
    records = (
        targeted_cohort_records(targeted_root, baseline_root, cohort_nodes, cohort_manifest)
        if targeted_root is not None
        else per_cohort_records(root, baseline_root, cohort_nodes, cohort_manifest)
    )
    deltas, specificity = paired_records(records)
    condition_summary = summarise(
        deltas,
        ("model", "condition", "birth_cohort"),
        tuple(f"{metric}_delta" for metric in METRICS),
    )
    specificity_summary = summarise(
        specificity,
        ("model", "tie_group", "birth_cohort"),
        ("relationship_specific_macro_f1_loss",),
    )
    _write_csv(output_dir / "birth_cohort_test_metrics.csv", records)
    _write_csv(output_dir / "birth_cohort_paired_deltas_by_seed.csv", deltas)
    _write_csv(output_dir / "birth_cohort_paired_summary.csv", condition_summary)
    _write_csv(output_dir / "birth_cohort_relation_specificity_by_seed.csv", specificity)
    _write_csv(output_dir / "birth_cohort_relation_specificity_summary.csv", specificity_summary)
    (output_dir / "birth_cohort_manifest.json").write_text(
        json.dumps({
            "birth_cohort_config": cohort_manifest,
            "data": str(data_path.resolve()),
            "targeted_root": str(targeted_root) if targeted_root is not None else None,
            "missing_cohort_id": MISSING_COHORT_ID,
            "missing_cohort_policy": "reported but excluded from cohort hypothesis comparison",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Birth-cohort reports written to {output_dir}")
    for row in specificity_summary:
        if row["included_in_cohort_hypothesis"]:
            print(
                f"{row['model']} {row['birth_cohort']} {row['tie_group']}: "
                f"specific_macro_f1_loss={float(row['relationship_specific_macro_f1_loss_mean']):+.4f}"
            )


if __name__ == "__main__":
    main()
