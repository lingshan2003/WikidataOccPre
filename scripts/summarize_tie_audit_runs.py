#!/usr/bin/env python3
"""Compare new tie-ablation runs with the existing Level-1 baselines.

The six complete-graph baselines are reused rather than retrained.  Every
ablation is paired with the complete-graph run using the same model and seed.
When both sides of a pair retain ``test_predictions.csv``, the report also
computes a paired test-node bootstrap interval for the Macro-F1 difference.
When either prediction file is unavailable, the comparison remains valid for
the saved test metrics but transparently falls back to seed-paired mean/std.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
MODELS = ("rgcn", "rgat")
SEEDS = ("42", "43", "44")
CONDITIONS = (
    "without_inherited",
    "random_matched_inherited",
    "without_acquired",
    "random_matched_acquired",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_report/level1/tie_audit")
    parser.add_argument(
        "--baseline-root",
        default="runs_report/level1",
        help="Directory containing rgcn_baseline/ and rgat_baseline/ seed folders",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=2000,
        help="Paired test-node bootstrap draws for Macro-F1 differences (default: 2000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260813,
        help="Random seed for paired test-node bootstrap (default: 20260813)",
    )
    return parser.parse_args()


def experiment_parts(experiment: str) -> Tuple[str, str, str]:
    parts = experiment.split("__")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"Tie-audit experiment name must be '<model>__<feature_regime>__<condition>', got {experiment!r}"
        )
    return parts[0], parts[1], parts[2]


def _read_payload(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metrics JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metrics JSON must contain an object: {path}")
    return payload


def _metrics(payload: Mapping[str, object], path: Path) -> Dict[str, float]:
    test = payload.get("test")
    if not isinstance(test, Mapping):
        raise ValueError(f"Completed test metrics are required: {path}")
    missing = [metric for metric in METRICS if metric not in test]
    if missing:
        raise ValueError(f"Test metrics are missing {missing}: {path}")
    return {metric: float(test[metric]) for metric in METRICS}


def _run_config(payload: Mapping[str, object], path: Path) -> Mapping[str, object]:
    config = payload.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Missing run_config: {path}")
    return config


def _validate_shared_protocol(
    config: Mapping[str, object],
    path: Path,
    model: str,
    seed: str,
) -> None:
    """Reject a legacy run that cannot act as this audit's exact baseline."""
    expected = {
        "model": model,
        "seed": int(seed),
        "occupation_feature_levels": "1,2,3",
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
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if model == "rgcn" and config.get("rgcn_backend") != "fast":
        mismatches["rgcn_backend"] = {"expected": "fast", "actual": config.get("rgcn_backend")}
    if mismatches:
        raise ValueError(f"Baseline protocol does not match this audit: {path}: {mismatches}")


def _taxonomy_provenance(payload: Mapping[str, object], path: Path) -> Tuple[str, str]:
    perturbation = payload.get("relation_perturbation")
    if not isinstance(perturbation, Mapping):
        raise ValueError(f"Missing relation_perturbation provenance: {path}")
    data_sha256 = perturbation.get("data_sha256")
    taxonomy = perturbation.get("tie_taxonomy")
    taxonomy_sha256 = taxonomy.get("sha256") if isinstance(taxonomy, Mapping) else None
    if not isinstance(data_sha256, str) or not isinstance(taxonomy_sha256, str):
        raise ValueError(
            f"Tie-ablation run lacks data/taxonomy hashes; rerun with the current training command: {path}"
        )
    return data_sha256, taxonomy_sha256


def _optional_artifact_path(metrics_path: Path, filename: str) -> str | None:
    """Return a run artifact path if it was preserved alongside metrics."""
    path = metrics_path.parent / filename
    return str(path) if path.is_file() else None


def load_legacy_baselines(root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for model in MODELS:
        for seed in SEEDS:
            metrics_path = root / f"{model}_baseline" / f"seed_{seed}" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Required existing baseline is missing: {metrics_path}")
            payload = _read_payload(metrics_path)
            _validate_shared_protocol(_run_config(payload, metrics_path), metrics_path, model, seed)
            records.append({
                "experiment": f"{model}_baseline",
                "model": model,
                "feature_regime": "occupation_neighbours",
                "condition": "full",
                "seed": seed,
                "metrics_path": str(metrics_path),
                "checkpoint_path": _optional_artifact_path(metrics_path, "best_model.pt"),
                "test_predictions_path": _optional_artifact_path(metrics_path, "test_predictions.csv"),
                "baseline_provenance": "legacy_metrics_config",
                "data_sha256": None,
                "tie_taxonomy_sha256": None,
                **_metrics(payload, metrics_path),
            })
    return records


def load_ablation_records(root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for metrics_path in sorted(root.glob("*/seed_*/metrics.json")):
        payload = _read_payload(metrics_path)
        experiment = metrics_path.parent.parent.name
        model, feature_regime, condition = experiment_parts(experiment)
        if model not in MODELS or feature_regime != "occupation_neighbours" or condition not in CONDITIONS:
            raise ValueError(f"Unexpected audit run directory: {metrics_path}")
        seed = metrics_path.parent.name.removeprefix("seed_")
        if seed not in SEEDS:
            raise ValueError(f"Unexpected audit seed in {metrics_path}; expected one of {SEEDS}")
        _validate_shared_protocol(_run_config(payload, metrics_path), metrics_path, model, seed)
        data_sha256, taxonomy_sha256 = _taxonomy_provenance(payload, metrics_path)
        records.append({
            "experiment": experiment,
            "model": model,
            "feature_regime": feature_regime,
            "condition": condition,
            "seed": seed,
            "metrics_path": str(metrics_path),
            "checkpoint_path": _optional_artifact_path(metrics_path, "best_model.pt"),
            "test_predictions_path": _optional_artifact_path(metrics_path, "test_predictions.csv"),
            "baseline_provenance": "current_data_and_taxonomy_hashes",
            "data_sha256": data_sha256,
            "tie_taxonomy_sha256": taxonomy_sha256,
            **_metrics(payload, metrics_path),
        })
    if not records:
        raise RuntimeError(f"No completed tie-ablation metrics found below {root}")
    return records


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _load_predictions(path: str) -> Dict[int, Tuple[str, str]]:
    """Load test labels and predictions keyed by immutable graph node index."""
    required = {"node_index", "true_label", "prediction"}
    rows: Dict[int, Tuple[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"Prediction file is missing {sorted(missing)}: {path}")
        for raw in reader:
            node_index = int(raw["node_index"])
            if node_index in rows:
                raise ValueError(f"Duplicate node_index={node_index} in prediction file: {path}")
            rows[node_index] = (raw["true_label"], raw["prediction"])
    if not rows:
        raise ValueError(f"Prediction file has no test rows: {path}")
    return rows


def _paired_prediction_arrays(
    baseline_path: str,
    ablation_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Ensure two runs predict exactly the same test nodes and labels."""
    baseline_rows = _load_predictions(baseline_path)
    ablation_rows = _load_predictions(ablation_path)
    if baseline_rows.keys() != ablation_rows.keys():
        only_baseline = sorted(baseline_rows.keys() - ablation_rows.keys())[:5]
        only_ablation = sorted(ablation_rows.keys() - baseline_rows.keys())[:5]
        raise ValueError(
            "Baseline and ablation test-node sets differ: "
            f"only_baseline={only_baseline}, only_ablation={only_ablation}"
        )
    node_indices = sorted(baseline_rows)
    baseline_true = np.asarray([baseline_rows[index][0] for index in node_indices], dtype=object)
    ablation_true = np.asarray([ablation_rows[index][0] for index in node_indices], dtype=object)
    if not np.array_equal(baseline_true, ablation_true):
        raise ValueError("Baseline and ablation true labels differ for one or more shared test nodes")
    baseline_prediction = np.asarray([baseline_rows[index][1] for index in node_indices], dtype=object)
    ablation_prediction = np.asarray([ablation_rows[index][1] for index in node_indices], dtype=object)
    labels = sorted(set(baseline_true) | set(baseline_prediction) | set(ablation_prediction))
    return baseline_true, baseline_prediction, ablation_prediction, np.asarray(node_indices), labels


def _bootstrap_macro_f1_delta(
    true_labels: np.ndarray,
    baseline_prediction: np.ndarray,
    ablation_prediction: np.ndarray,
    labels: Sequence[str],
    draws: int,
    random_seed: int,
) -> np.ndarray:
    if draws <= 0:
        return np.empty(0, dtype=float)
    generator = np.random.default_rng(random_seed)
    sample_size = len(true_labels)
    deltas = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = generator.integers(0, sample_size, size=sample_size)
        baseline_score = f1_score(
            true_labels[sample], baseline_prediction[sample], labels=list(labels), average="macro", zero_division=0
        )
        ablation_score = f1_score(
            true_labels[sample], ablation_prediction[sample], labels=list(labels), average="macro", zero_division=0
        )
        deltas[draw] = float(ablation_score - baseline_score)
    return deltas


def _interval(samples: np.ndarray) -> Tuple[float | None, float | None]:
    if not len(samples):
        return None, None
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def paired_records(
    baselines: Iterable[Mapping[str, object]],
    ablations: Iterable[Mapping[str, object]],
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 20260813,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    baseline_by_key = {
        (str(record["model"]), str(record["seed"])): record for record in baselines
    }
    indexed: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for record in ablations:
        key = (str(record["model"]), str(record["condition"]), str(record["seed"]))
        if key in indexed:
            raise ValueError(f"Duplicate ablation run for {key}")
        indexed[key] = record

    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                baseline = baseline_by_key.get((model, seed))
                ablation = indexed.get((model, condition, seed))
                if baseline is None or ablation is None:
                    raise ValueError(
                        f"Missing same-seed pair for model={model}, condition={condition}, seed={seed}"
                    )
                row: Dict[str, object] = {
                    "model": model,
                    "feature_regime": "occupation_neighbours",
                    "condition": condition,
                    "seed": seed,
                    "baseline_metrics_path": baseline["metrics_path"],
                    "ablation_metrics_path": ablation["metrics_path"],
                    "baseline_checkpoint_path": baseline.get("checkpoint_path"),
                    "ablation_checkpoint_path": ablation.get("checkpoint_path"),
                    "baseline_test_predictions_path": baseline.get("test_predictions_path"),
                    "ablation_test_predictions_path": ablation.get("test_predictions_path"),
                    "comparison_provenance": "same_seed_legacy_baseline_config",
                    "ablation_data_sha256": ablation["data_sha256"],
                    "tie_taxonomy_sha256": ablation["tie_taxonomy_sha256"],
                }
                for metric in METRICS:
                    row[f"baseline_{metric}"] = float(baseline[metric])
                    row[f"{metric}_delta"] = float(ablation[metric]) - float(baseline[metric])
                baseline_prediction_path = baseline.get("test_predictions_path")
                ablation_prediction_path = ablation.get("test_predictions_path")
                if isinstance(baseline_prediction_path, str) and isinstance(ablation_prediction_path, str):
                    true_labels, baseline_prediction, ablation_prediction, _, labels = _paired_prediction_arrays(
                        baseline_prediction_path, ablation_prediction_path
                    )
                    samples = _bootstrap_macro_f1_delta(
                        true_labels,
                        baseline_prediction,
                        ablation_prediction,
                        labels,
                        draws=bootstrap_draws,
                        random_seed=bootstrap_seed + int(seed),
                    )
                    ci_low, ci_high = _interval(samples)
                    row.update({
                        "bootstrap_available": bool(len(samples)),
                        "bootstrap_draws": len(samples),
                        "macro_f1_delta_bootstrap_ci_low": ci_low,
                        "macro_f1_delta_bootstrap_ci_high": ci_high,
                        "_bootstrap_macro_f1_delta_samples": samples,
                    })
                else:
                    row.update({
                        "bootstrap_available": False,
                        "bootstrap_draws": 0,
                        "macro_f1_delta_bootstrap_ci_low": None,
                        "macro_f1_delta_bootstrap_ci_high": None,
                        "_bootstrap_macro_f1_delta_samples": np.empty(0, dtype=float),
                    })
                rows.append(row)

    summaries: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["condition"]))].append(row)
    for (model, condition), group in sorted(grouped.items()):
        if len({str(row["ablation_data_sha256"]) for row in group}) != 1:
            raise ValueError(f"Ablation artifact hash differs across seeds for {model}/{condition}")
        if len({str(row["tie_taxonomy_sha256"]) for row in group}) != 1:
            raise ValueError(f"Tie taxonomy hash differs across seeds for {model}/{condition}")
        summary: Dict[str, object] = {
            "model": model,
            "feature_regime": "occupation_neighbours",
            "condition": condition,
            "seeds_completed": len(group),
            "seeds": ",".join(str(row["seed"]) for row in group),
            "comparison_provenance": "same_seed_legacy_baseline_config",
            "ablation_data_sha256": group[0]["ablation_data_sha256"],
            "tie_taxonomy_sha256": group[0]["tie_taxonomy_sha256"],
        }
        for metric in METRICS:
            deltas = [float(row[f"{metric}_delta"]) for row in group]
            summary[f"{metric}_delta_mean"] = mean(deltas)
            summary[f"{metric}_delta_std"] = stdev(deltas) if len(deltas) > 1 else 0.0
        bootstrap_samples = [row["_bootstrap_macro_f1_delta_samples"] for row in group]
        if bootstrap_samples and all(len(samples) for samples in bootstrap_samples):
            stacked = np.stack(bootstrap_samples)
            ci_low, ci_high = _interval(np.mean(stacked, axis=0))
            summary.update({
                "bootstrap_available": True,
                "bootstrap_draws": int(stacked.shape[1]),
                "macro_f1_delta_bootstrap_ci_low": ci_low,
                "macro_f1_delta_bootstrap_ci_high": ci_high,
                "comparison_provenance": "same_seed_legacy_baseline_predictions",
            })
        else:
            summary.update({
                "bootstrap_available": False,
                "bootstrap_draws": 0,
                "macro_f1_delta_bootstrap_ci_low": None,
                "macro_f1_delta_bootstrap_ci_high": None,
            })
        summaries.append(summary)
    for row in rows:
        row.pop("_bootstrap_macro_f1_delta_samples")
    return rows, summaries


def main() -> None:
    args = parse_args()
    root, baseline_root = Path(args.root), Path(args.baseline_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Tie-audit root does not exist: {root}")
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"Baseline root does not exist: {baseline_root}")
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be non-negative")
    baselines = load_legacy_baselines(baseline_root)
    ablations = load_ablation_records(root)
    per_seed, summaries = paired_records(
        baselines,
        ablations,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    baseline_path = root / "tie_audit_reused_baseline_metrics.csv"
    ablation_path = root / "tie_audit_ablation_seed_metrics.csv"
    paired_path = root / "tie_audit_paired_deltas_by_seed.csv"
    summary_path = root / "tie_audit_paired_summary.csv"
    _write_csv(baseline_path, baselines)
    _write_csv(ablation_path, ablations)
    _write_csv(paired_path, per_seed)
    _write_csv(summary_path, summaries)
    print(f"Wrote {baseline_path}, {ablation_path}, {paired_path}, and {summary_path}")
    for row in summaries:
        interval = ""
        if row["bootstrap_available"]:
            interval = (
                "; paired test-node 95% CI="
                f"[{float(row['macro_f1_delta_bootstrap_ci_low']):+.4f}, "
                f"{float(row['macro_f1_delta_bootstrap_ci_high']):+.4f}]"
            )
        print(
            f"{row['model']} {row['condition']}: "
            f"macro_f1_delta={float(row['macro_f1_delta_mean']):+.4f}±{float(row['macro_f1_delta_std']):.4f} "
            f"(same-seed legacy-baseline comparison{interval})"
        )


if __name__ == "__main__":
    main()
