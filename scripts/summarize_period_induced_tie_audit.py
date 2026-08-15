#!/usr/bin/env python3
"""Summarise a fresh-baseline, life-period-induced inherited/acquired tie audit.

Unlike the global-graph birth-cohort summary, each period here contains every
person whose known life interval overlaps that period, has its own graph
artifact and own complete-graph (`full`) baseline. Comparisons are permitted
only within identical period artifact, model and model seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.life_periods import load_life_period_config, sha256_file
from training.tie_taxonomy import load_tie_taxonomy


MODELS = ("rgcn", "rgat")
SEEDS = ("42", "43", "44")
CONDITIONS = (
    "full",
    "without_inherited",
    "random_matched_inherited",
    "without_acquired",
    "random_matched_acquired",
)
METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
EDGE_POLICY = "retain_only_edges_with_both_endpoints_in_selected_life_period"
EXACT_RANDOM_CONTROL_UNIT = "original_edge_instance_plus_generated_reverse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_report/level1/life_period_induced_tie_audit_v2")
    parser.add_argument("--artifact-root", default="artifacts/level1_life_period_induced_v2")
    parser.add_argument("--life-periods", default="config/historical_life_periods_v2.json")
    parser.add_argument("--tie-taxonomy", default="config/tie_taxonomy_ascribed_family_v1.json")
    parser.add_argument(
        "--occupation-feature-levels",
        default="1,2,3",
        help="Exact occupation_feature_levels value required in every run's protocol",
    )
    parser.add_argument(
        "--expected-target-column",
        default="occupation_level1",
        help="Expected artifact target_column; use an empty string to skip this guard",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _period_range(start: object, end: object) -> str:
    if start is None:
        return f"through {end} CE"
    if end is None:
        return f"{start} CE onward"
    return f"{start}\u2013{end} CE"


def load_period_artifact(
    artifact_root: Path,
    period,
    life_period_config,
    taxonomy_path: str | Path,
    expected_target_column: str | None = None,
) -> Dict[str, object]:
    data_path = artifact_root / period.identifier / "graph_data.pt"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing period artifact: {data_path}")
    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("metadata"), Mapping):
        raise ValueError(f"Invalid period artifact: {data_path}")
    data, metadata = bundle["data"], bundle["metadata"]
    details = metadata.get("period_induced_artifact")
    if not isinstance(details, Mapping):
        raise ValueError(f"Artifact is not a fresh period-induced graph: {data_path}")
    selected, config = details.get("selected_life_period"), details.get("life_period_config")
    if not isinstance(selected, Mapping) or selected.get("id") != period.identifier:
        raise ValueError(f"Period artifact declares the wrong selected period: {data_path}")
    if not isinstance(config, Mapping) or config.get("sha256") != life_period_config.sha256:
        raise ValueError(f"Period artifact life-period config hash differs: {data_path}")
    if details.get("edge_policy") != EDGE_POLICY:
        raise ValueError(f"Period artifact does not use the required induced-edge policy: {data_path}")
    if expected_target_column and metadata.get("target_column") != expected_target_column:
        raise ValueError(
            f"Period artifact target_column={metadata.get('target_column')!r}; "
            f"expected {expected_target_column!r}: {data_path}"
        )
    if not all(hasattr(data, field) for field in ("train_mask", "val_mask", "test_mask", "y")):
        raise ValueError(f"Period artifact lacks fresh split masks: {data_path}")
    masks = data.train_mask | data.val_mask | data.test_mask
    if torch.any(data.train_mask & data.val_mask) or torch.any(data.train_mask & data.test_mask) or torch.any(data.val_mask & data.test_mask):
        raise ValueError(f"Period split masks overlap: {data_path}")
    if not torch.equal(masks, data.y >= 0):
        raise ValueError(f"Period split masks do not partition exactly the supervised nodes: {data_path}")
    taxonomy = load_tie_taxonomy(taxonomy_path, metadata["relation_to_id"])
    return {
        "period_id": period.identifier,
        "period_label": period.label,
        "period_start_year": period.start,
        "period_end_year": period.end,
        "period_year_range": _period_range(period.start, period.end),
        "artifact_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "source_data_sha256": details.get("source_data_sha256"),
        "taxonomy_sha256": taxonomy.sha256,
        "life_period_config_sha256": life_period_config.sha256,
        "artifact_nodes": int(data.num_nodes),
        "artifact_directed_edges": int(data.edge_index.size(1)),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
        "split_seed": details.get("split", {}).get("seed") if isinstance(details.get("split"), Mapping) else None,
    }


def _metrics(payload: Mapping[str, object], path: Path) -> Dict[str, float]:
    test = payload.get("test")
    if not isinstance(test, Mapping):
        raise ValueError(f"Completed test metrics are required: {path}")
    missing = [metric for metric in METRICS if metric not in test]
    if missing:
        raise ValueError(f"Test metrics are missing {missing}: {path}")
    return {metric: float(test[metric]) for metric in METRICS}


def _validate_protocol(
    config: Mapping[str, object], path: Path, model: str, seed: str, occupation_feature_levels: str
) -> None:
    expected = {
        "model": model,
        "seed": int(seed),
        "occupation_feature_levels": occupation_feature_levels,
        "auxiliary_features": "none",
        "feature_mode": "selected",
        "occupation_representation": "categorical",
        "num_neighbors": "15,10",
        "train_mode": "sampled",
        "eval_mode": "sampled",
        "loss": "cross_entropy",
        "train_root_sampling": "uniform",
        "early_stop_metric": "macro_f1",
        "hidden_dim": 128,
        "branch_dim": 64,
        "heads": 4,
        "edge_cohort_config": None,
        "edge_cohort_id": None,
    }
    mismatches = {key: {"expected": value, "actual": config.get(key)} for key, value in expected.items() if config.get(key) != value}
    if model == "rgcn" and config.get("rgcn_backend") != "fast":
        mismatches["rgcn_backend"] = {"expected": "fast", "actual": config.get("rgcn_backend")}
    if mismatches:
        raise ValueError(f"Run protocol differs from the period-audit protocol: {path}: {mismatches}")


def _expected_perturbation(condition: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if condition == "full":
        return (), ()
    if condition == "without_inherited":
        return ("inherited",), ()
    if condition == "without_acquired":
        return ("acquired",), ()
    if condition == "random_matched_inherited":
        return (), ("inherited",)
    if condition == "random_matched_acquired":
        return (), ("acquired",)
    raise ValueError(f"Unexpected condition: {condition}")


def _validate_perturbation(
    payload: Mapping[str, object], path: Path, artifact: Mapping[str, object], condition: str
) -> Mapping[str, object]:
    perturbation = payload.get("relation_perturbation")
    if not isinstance(perturbation, Mapping):
        raise ValueError(f"Missing relation perturbation provenance: {path}")
    taxonomy = perturbation.get("tie_taxonomy")
    if perturbation.get("data_sha256") != artifact["data_sha256"]:
        raise ValueError(f"Run uses a different period artifact: {path}")
    if not isinstance(taxonomy, Mapping) or taxonomy.get("sha256") != artifact["taxonomy_sha256"]:
        raise ValueError(f"Run uses a different tie taxonomy: {path}")
    if perturbation.get("edge_cohort") is not None:
        raise ValueError(f"Period-induced run must not use the old cohort-incident deletion option: {path}")
    expected_drop, expected_random = _expected_perturbation(condition)
    if tuple(perturbation.get("dropped_tie_groups", ())) != expected_drop:
        raise ValueError(f"Unexpected direct tie deletion for {condition}: {path}")
    if tuple(perturbation.get("random_drop_matched_tie_groups", ())) != expected_random:
        raise ValueError(f"Unexpected matched random control for {condition}: {path}")
    before, after = perturbation.get("edge_count_before"), perturbation.get("edge_count_after_random_drop")
    if not isinstance(before, int) or not isinstance(after, int):
        raise ValueError(f"Run lacks actual edge-count provenance: {path}")
    if condition == "full" and before != after:
        raise ValueError(f"Full baseline unexpectedly changed its graph: {path}")
    if condition != "full" and after >= before:
        raise ValueError(f"Ablation/control did not remove any edges: {path}")
    return perturbation


def _load_predictions(path: Path, expected_nodes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: Dict[int, Tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"node_index", "true_label", "prediction"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction file is missing {sorted(missing)}: {path}")
        for row in reader:
            node_index = int(row["node_index"])
            if node_index in rows:
                raise ValueError(f"Duplicate node_index={node_index} in {path}")
            rows[node_index] = (row["true_label"], row["prediction"])
    expected = sorted(int(node) for node in expected_nodes)
    if sorted(rows) != expected:
        raise ValueError(f"Prediction test-node set differs from the period artifact: {path}")
    true = np.asarray([rows[node][0] for node in expected], dtype=object)
    prediction = np.asarray([rows[node][1] for node in expected], dtype=object)
    return np.asarray(expected, dtype=np.int64), true, prediction


def _bootstrap_delta(
    true: np.ndarray, baseline_prediction: np.ndarray, condition_prediction: np.ndarray,
    draws: int, seed: int,
) -> np.ndarray:
    if draws <= 0:
        return np.empty(0, dtype=float)
    labels = sorted(set(true) | set(baseline_prediction) | set(condition_prediction))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        index = rng.integers(0, len(true), size=len(true))
        baseline_f1 = f1_score(true[index], baseline_prediction[index], labels=labels, average="macro", zero_division=0)
        condition_f1 = f1_score(true[index], condition_prediction[index], labels=labels, average="macro", zero_division=0)
        samples[draw] = float(condition_f1 - baseline_f1)
    return samples


def _interval(samples: np.ndarray) -> Tuple[float | None, float | None]:
    if not len(samples):
        return None, None
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def load_records(
    root: Path,
    artifact_root: Path,
    life_period_config,
    taxonomy_path: str | Path,
    bootstrap_draws: int,
    bootstrap_seed: int,
    occupation_feature_levels: str = "1,2,3",
    expected_target_column: str | None = "occupation_level1",
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for period in life_period_config.periods:
        artifact = load_period_artifact(
            artifact_root, period, life_period_config, taxonomy_path, expected_target_column
        )
        artifact_path = Path(str(artifact["artifact_path"]))
        bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
        expected_test = torch.where(bundle["data"].test_mask)[0].detach().cpu().tolist()
        for model in MODELS:
            for seed in SEEDS:
                per_condition: Dict[str, Tuple[Mapping[str, object], Path, np.ndarray, np.ndarray, Mapping[str, object]]] = {}
                for condition in CONDITIONS:
                    run_dir = root / f"{model}__period_{period.identifier}__{condition}" / f"seed_{seed}"
                    metrics_path, prediction_path = run_dir / "metrics.json", run_dir / "test_predictions.csv"
                    if not metrics_path.is_file() or not prediction_path.is_file():
                        raise FileNotFoundError(f"Completed metrics and predictions are required: {run_dir}")
                    payload = _read_json(metrics_path)
                    config = payload.get("run_config")
                    if not isinstance(config, Mapping):
                        raise ValueError(f"Missing run_config: {metrics_path}")
                    _validate_protocol(config, metrics_path, model, seed, occupation_feature_levels)
                    perturbation = _validate_perturbation(payload, metrics_path, artifact, condition)
                    nodes, true, prediction = _load_predictions(prediction_path, expected_test)
                    per_condition[condition] = (payload, prediction_path, true, prediction, perturbation)
                    records.append({
                        "model": model,
                        "seed": seed,
                        "condition": condition,
                        **artifact,
                        "metrics_path": str(metrics_path),
                        "checkpoint_path": str(run_dir / "best_model.pt") if (run_dir / "best_model.pt").is_file() else None,
                        "test_predictions_path": str(prediction_path),
                        "actual_directed_edges_removed": int(perturbation["edge_count_before"]) - int(perturbation["edge_count_after_random_drop"]),
                        "dropped_relation_pair_count": perturbation.get("dropped_relation_pair_count"),
                        "random_edge_instance_pairs": perturbation.get("random_edge_instance_pairs"),
                        "random_control_unit": perturbation.get("random_control_unit"),
                        **_metrics(payload, metrics_path),
                    })
                baseline_true, baseline_prediction = per_condition["full"][2], per_condition["full"][3]
                for condition, (_, _, current_true, current_prediction, _) in per_condition.items():
                    if not np.array_equal(baseline_true, current_true):
                        raise ValueError(f"True labels differ from baseline for {model}/{period.identifier}/{seed}/{condition}")
                    # Stash per-seed paired bootstrap samples only in memory; CSVs
                    # get scalar CIs below, never opaque array serialisations.
                    for row in records[-len(CONDITIONS):]:
                        if row["condition"] == condition:
                            row["_bootstrap_samples"] = _bootstrap_delta(
                                baseline_true, baseline_prediction, current_prediction,
                                bootstrap_draws, bootstrap_seed + int(seed),
                            ) if condition != "full" else np.zeros(bootstrap_draws, dtype=float)
                            break
                for tie_group, direct_condition, random_condition in (
                    ("inherited", "without_inherited", "random_matched_inherited"),
                    ("acquired", "without_acquired", "random_matched_acquired"),
                ):
                    direct = per_condition[direct_condition][4]
                    random = per_condition[random_condition][4]
                    direct_removed = int(direct["edge_count_before"]) - int(direct["edge_count_after_random_drop"])
                    random_removed = int(random["edge_count_before"]) - int(random["edge_count_after_random_drop"])
                    if direct_removed != random_removed:
                        raise ValueError(
                            f"Exact random control mismatch for {model}/{period.identifier}/seed_{seed}/{tie_group}: "
                            f"direct={direct_removed} directed edges, random={random_removed}"
                        )
                    if random.get("random_control_unit") != EXACT_RANDOM_CONTROL_UNIT:
                        raise ValueError(
                            f"Random control does not use exact original-edge/reverse units: "
                            f"{model}/{period.identifier}/seed_{seed}/{tie_group}"
                        )
                    if int(random.get("random_edge_instance_pairs", -1)) * 2 != random_removed:
                        raise ValueError(
                            f"Random control does not record exactly two directed edges per selected unit: "
                            f"{model}/{period.identifier}/seed_{seed}/{tie_group}"
                        )
    return records


def summarise(records: Iterable[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    record_index = {
        (str(row["model"]), str(row["period_id"]), str(row["seed"]), str(row["condition"])): row
        for row in records
    }
    for row in records:
        grouped[(str(row["model"]), str(row["period_id"]), str(row["condition"]))].append(row)
    summaries: List[Dict[str, object]] = []
    specificity: List[Dict[str, object]] = []
    for (model, period_id, condition), group in sorted(grouped.items()):
        if len(group) != len(SEEDS):
            raise ValueError(f"Expected {len(SEEDS)} seeds for {model}/{period_id}/{condition}")
        first = group[0]
        fixed = (
            "period_label", "period_start_year", "period_end_year", "period_year_range", "artifact_path",
            "data_sha256", "source_data_sha256", "taxonomy_sha256", "life_period_config_sha256", "artifact_nodes",
            "artifact_directed_edges", "train_nodes", "val_nodes", "test_nodes", "split_seed",
        )
        for field in fixed:
            if len({str(row[field]) for row in group}) != 1:
                raise ValueError(f"Inconsistent {field} across seeds for {model}/{period_id}/{condition}")
        summary = {
            "model": model,
            "period_id": period_id,
            "condition": condition,
            "seeds_completed": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda row: int(str(row["seed"])))),
            **{field: first[field] for field in fixed},
        }
        baseline_group = [record_index[(model, period_id, str(row["seed"]), "full")] for row in group]
        for metric in METRICS:
            baseline_values = [float(row[metric]) for row in baseline_group]
            current_values = [float(row[metric]) for row in group]
            deltas = [current - baseline for current, baseline in zip(current_values, baseline_values)]
            summary[f"baseline_{metric}_mean"] = mean(baseline_values)
            summary[f"baseline_{metric}_std"] = stdev(baseline_values) if len(baseline_values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean(current_values)
            summary[f"{metric}_std"] = stdev(current_values) if len(current_values) > 1 else 0.0
            summary[f"{metric}_delta_mean"] = mean(deltas)
            summary[f"{metric}_delta_std"] = stdev(deltas) if len(deltas) > 1 else 0.0
        samples = [np.asarray(row["_bootstrap_samples"], dtype=float) for row in group]
        mean_samples = np.mean(np.stack(samples), axis=0) if samples and all(len(sample) for sample in samples) else np.empty(0)
        low, high = _interval(mean_samples)
        summary["bootstrap_available"] = bool(len(mean_samples))
        summary["bootstrap_draws"] = int(len(mean_samples))
        summary["macro_f1_delta_bootstrap_ci_low"] = low
        summary["macro_f1_delta_bootstrap_ci_high"] = high
        summaries.append(summary)

    for model in MODELS:
        for period_id in sorted({str(row["period_id"]) for row in records}):
            for seed in SEEDS:
                for tie_group, dropped, random_control in (
                    ("inherited", "without_inherited", "random_matched_inherited"),
                    ("acquired", "without_acquired", "random_matched_acquired"),
                ):
                    drop_row = record_index[(model, period_id, seed, dropped)]
                    random_row = record_index[(model, period_id, seed, random_control)]
                    specificity.append({
                        "model": model,
                        "period_id": period_id,
                        "seed": seed,
                        "tie_group": tie_group,
                        "period_year_range": drop_row["period_year_range"],
                        "data_sha256": drop_row["data_sha256"],
                        "taxonomy_sha256": drop_row["taxonomy_sha256"],
                        "drop_macro_f1": drop_row["macro_f1"],
                        "random_matched_macro_f1": random_row["macro_f1"],
                        "relationship_specific_macro_f1_loss": float(random_row["macro_f1"]) - float(drop_row["macro_f1"]),
                    })
    return summaries, specificity


def main() -> None:
    args = parse_args()
    root, artifact_root = Path(args.root), Path(args.artifact_root)
    if not root.is_dir() or not artifact_root.is_dir():
        raise FileNotFoundError("Both --root and --artifact-root must be existing directories")
    life_period_config = load_life_period_config(args.life_periods)
    records = load_records(
        root,
        artifact_root,
        life_period_config,
        args.tie_taxonomy,
        args.bootstrap_draws,
        args.bootstrap_seed,
        args.occupation_feature_levels,
        args.expected_target_column or None,
    )
    source_hashes = {row["source_data_sha256"] for row in records}
    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError(
            "Period artifacts must all descend from the same canonical source-data hash; "
            f"got {sorted(str(value) for value in source_hashes)}"
        )
    summaries, specificity = summarise(records)
    for row in records:
        row.pop("_bootstrap_samples", None)
    output_dir = Path(args.output_dir) if args.output_dir else root / "period_induced_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "period_induced_test_metrics.csv", records)
    _write_csv(output_dir / "period_induced_condition_summary.csv", summaries)
    _write_csv(output_dir / "period_induced_relation_specificity_by_seed.csv", specificity)
    (output_dir / "period_induced_manifest.json").write_text(json.dumps({
        "analysis": "fresh_life_period_overlap_graphs_with_fresh_within_period_splits",
        "edge_policy": EDGE_POLICY,
        "life_period_config": life_period_config.manifest(),
        "tie_taxonomy": str(Path(args.tie_taxonomy).resolve()),
        "tie_taxonomy_sha256": sha256_file(Path(args.tie_taxonomy)),
        "artifact_root": str(artifact_root.resolve()),
        "run_root": str(root.resolve()),
        "occupation_feature_levels": args.occupation_feature_levels,
        "expected_target_column": args.expected_target_column or None,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Period-induced reports written to {output_dir}")


if __name__ == "__main__":
    main()
