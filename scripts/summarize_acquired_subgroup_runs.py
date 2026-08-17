#!/usr/bin/env python3
"""Summarise R-GCN acquired-tie subgroup ablations against matched controls.

The script accepts complete-graph runs (whose full baseline is reused) and
induced life-period runs (whose full baseline was trained on each period
artifact).  It rejects an unequal random control rather than silently treating
graph-density loss as a relationship-specific effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from training.life_periods import load_life_period_config


METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
SEEDS = ("42", "43", "44")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("full", "period"), required=True)
    parser.add_argument("--root", required=True, help="Root containing the subgroup ablation runs")
    parser.add_argument("--baseline-root", required=True, help="Root containing matching R-GCN full baselines")
    parser.add_argument("--relation-taxonomy", required=True)
    parser.add_argument("--groups", required=True, help="Comma-separated acquired subgroup IDs")
    parser.add_argument("--life-periods", help="Required when --scope period")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_groups(value: str) -> Tuple[str, ...]:
    groups = tuple(item.strip() for item in value.split(",") if item.strip())
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("--groups must contain one or more unique subgroup IDs")
    if "inherited" in groups:
        raise ValueError("This report is reserved for acquired subgroups, not inherited")
    return groups


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid metrics JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Metrics JSON must contain an object: {path}")
    return payload


def _test_metrics(payload: Mapping[str, object], path: Path) -> Dict[str, float]:
    test = payload.get("test")
    if not isinstance(test, Mapping):
        raise ValueError(f"Completed test metrics are required: {path}")
    missing = [metric for metric in METRICS if metric not in test]
    if missing:
        raise ValueError(f"Test metrics missing {missing}: {path}")
    return {metric: float(test[metric]) for metric in METRICS}


def _run_config(payload: Mapping[str, object], path: Path, seed: str) -> Mapping[str, object]:
    config = payload.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Missing run_config: {path}")
    expected = {
        "model": "rgcn",
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
        "rgcn_backend": "fast",
    }
    mismatches = {key: {"expected": value, "actual": config.get(key)}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"R-GCN protocol differs from the subgroup audit: {path}: {mismatches}")
    return config


def _perturbation(payload: Mapping[str, object], path: Path) -> Mapping[str, object]:
    value = payload.get("relation_perturbation")
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing relation_perturbation provenance: {path}")
    return value


def _prediction_rows(path: Path) -> Dict[int, Tuple[str, str]]:
    required = {"node_index", "true_label", "prediction"}
    rows: Dict[int, Tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Prediction file missing {sorted(missing)}: {path}")
        for row in reader:
            index = int(row["node_index"])
            if index in rows:
                raise ValueError(f"Duplicate node_index={index} in prediction file: {path}")
            rows[index] = (row["true_label"], row["prediction"])
    if not rows:
        raise ValueError(f"Prediction file has no rows: {path}")
    return rows


def _paired_arrays(left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    left, right = _prediction_rows(left_path), _prediction_rows(right_path)
    if left.keys() != right.keys():
        raise ValueError(f"Prediction node sets differ: {left_path} versus {right_path}")
    indices = sorted(left)
    true_left = np.asarray([left[index][0] for index in indices], dtype=object)
    true_right = np.asarray([right[index][0] for index in indices], dtype=object)
    if not np.array_equal(true_left, true_right):
        raise ValueError(f"Prediction true labels differ: {left_path} versus {right_path}")
    prediction_left = np.asarray([left[index][1] for index in indices], dtype=object)
    prediction_right = np.asarray([right[index][1] for index in indices], dtype=object)
    labels = sorted(set(true_left) | set(prediction_left) | set(prediction_right))
    return true_left, prediction_left, prediction_right, labels


def _bootstrap_macro_f1_delta(
    true: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    labels: Sequence[str],
    draws: int,
    seed: int,
) -> np.ndarray:
    if draws <= 0:
        return np.empty(0, dtype=float)
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = generator.integers(0, len(true), size=len(true))
        left_score = f1_score(true[selected], left[selected], labels=list(labels), average="macro", zero_division=0)
        right_score = f1_score(true[selected], right[selected], labels=list(labels), average="macro", zero_division=0)
        samples[draw] = float(right_score - left_score)
    return samples


def _interval(samples: np.ndarray) -> Tuple[float | None, float | None]:
    if not len(samples):
        return None, None
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return float(lower), float(upper)


def _experiment_name(scope: str, context: str, condition: str) -> str:
    return (
        f"rgcn__occupation_neighbours__{condition}"
        if scope == "full" else f"rgcn__period_{context}__{condition}"
    )


def _baseline_path(scope: str, baseline_root: Path, context: str, seed: str) -> Path:
    experiment = "rgcn_baseline" if scope == "full" else _experiment_name(scope, context, "full")
    return baseline_root / experiment / f"seed_{seed}" / "metrics.json"


def _run_path(root: Path, scope: str, context: str, condition: str, seed: str) -> Path:
    return root / _experiment_name(scope, context, condition) / f"seed_{seed}" / "metrics.json"


def _run_record(path: Path, seed: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed run is missing: {path}")
    payload = _read_json(path)
    _run_config(payload, path, seed)
    perturbation = _perturbation(payload, path)
    return {
        "metrics_path": str(path),
        "predictions_path": str(path.parent / "test_predictions.csv")
        if (path.parent / "test_predictions.csv").is_file() else None,
        "metrics": _test_metrics(payload, path),
        "perturbation": perturbation,
    }


def _validate_subgroup_run(
    record: Mapping[str, object],
    group: str,
    condition_kind: str,
    expected_taxonomy_sha256: str,
) -> None:
    perturbation = record["perturbation"]
    assert isinstance(perturbation, Mapping)
    taxonomy = perturbation.get("relation_taxonomy")
    if not isinstance(taxonomy, Mapping) or taxonomy.get("sha256") != expected_taxonomy_sha256:
        raise ValueError(f"Run has a different or missing relation taxonomy: {record['metrics_path']}")
    groups = taxonomy.get("groups")
    if not isinstance(groups, Mapping) or group not in groups:
        raise ValueError(f"Relation taxonomy does not contain subgroup {group!r}: {record['metrics_path']}")
    dropped = tuple(perturbation.get("dropped_relation_taxonomy_groups", ()))
    random_matched = tuple(perturbation.get("random_drop_matched_relation_taxonomy_groups", ()))
    if condition_kind == "direct":
        valid = dropped == (group,) and not random_matched
    else:
        valid = random_matched == (group,) and not dropped
        valid = valid and perturbation.get("random_control_unit") == "original_edge_instance_plus_generated_reverse"
        valid = valid and int(perturbation.get("random_edge_instance_pairs", 0)) > 0
    if not valid:
        raise ValueError(f"Unexpected {condition_kind} perturbation for subgroup {group!r}: {record['metrics_path']}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 0:
        raise ValueError("--bootstrap-draws must be non-negative")
    root, baseline_root, taxonomy_path = Path(args.root), Path(args.baseline_root), Path(args.relation_taxonomy)
    if not root.is_dir() or not baseline_root.is_dir() or not taxonomy_path.is_file():
        raise FileNotFoundError("--root, --baseline-root, and --relation-taxonomy must exist")
    groups = _parse_groups(args.groups)
    if args.scope == "period":
        if not args.life_periods:
            raise ValueError("--life-periods is required with --scope period")
        contexts = tuple(period.identifier for period in load_life_period_config(args.life_periods).periods)
    else:
        contexts = ("full",)
    taxonomy_sha256 = _sha256_file(taxonomy_path)

    condition_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    for context_index, context in enumerate(contexts):
        for group_index, group in enumerate(groups):
            for seed in SEEDS:
                baseline = _run_record(_baseline_path(args.scope, baseline_root, context, seed), seed)
                direct = _run_record(_run_path(root, args.scope, context, f"without_{group}", seed), seed)
                random = _run_record(_run_path(root, args.scope, context, f"random_matched_{group}", seed), seed)
                _validate_subgroup_run(direct, group, "direct", taxonomy_sha256)
                _validate_subgroup_run(random, group, "random", taxonomy_sha256)
                baseline_p, direct_p, random_p = baseline["perturbation"], direct["perturbation"], random["perturbation"]
                assert isinstance(baseline_p, Mapping) and isinstance(direct_p, Mapping) and isinstance(random_p, Mapping)
                direct_removed = int(direct_p["edge_count_before"]) - int(direct_p["edge_count_after_random_drop"])
                random_removed = int(random_p["edge_count_before"]) - int(random_p["edge_count_after_random_drop"])
                if direct_removed <= 0 or direct_removed != random_removed:
                    raise ValueError(
                        f"Exact random-control mismatch for {context}/{group}/seed_{seed}: "
                        f"direct={direct_removed}, random={random_removed} directed edges"
                    )
                if int(random_p["random_edge_instance_pairs"]) * 2 != random_removed:
                    raise ValueError(f"Random control does not remove two directed edges per unit: {random['metrics_path']}")
                if direct_p.get("data_sha256") != random_p.get("data_sha256"):
                    raise ValueError(f"Direct and random runs use different graph artifacts: {context}/{group}/seed_{seed}")
                baseline_hash = baseline_p.get("data_sha256")
                if isinstance(baseline_hash, str) and baseline_hash != direct_p.get("data_sha256"):
                    raise ValueError(f"Baseline and subgroup runs use different graph artifacts: {context}/{group}/seed_{seed}")
                baseline_edges = baseline_p.get("edge_count_before")
                if baseline_edges is not None and int(baseline_edges) != int(direct_p["edge_count_before"]):
                    raise ValueError(f"Baseline and subgroup runs have different pre-ablation edge counts: {context}/{group}/seed_{seed}")

                for kind, record in (("full", baseline), ("direct", direct), ("random_matched", random)):
                    metrics = record["metrics"]
                    assert isinstance(metrics, Mapping)
                    condition_rows.append({
                        "scope": args.scope,
                        "context": context,
                        "subgroup": group,
                        "seed": seed,
                        "condition": kind,
                        "metrics_path": record["metrics_path"],
                        "test_predictions_path": record["predictions_path"],
                        "data_sha256": (record["perturbation"] or {}).get("data_sha256"),
                        "relation_taxonomy_sha256": taxonomy_sha256 if kind != "full" else None,
                        "directed_edges_removed": 0 if kind == "full" else direct_removed,
                        **metrics,
                    })

                comparisons = (
                    ("direct_minus_full", baseline, direct),
                    ("random_minus_full", baseline, random),
                    ("random_minus_direct", direct, random),
                )
                for comparison_index, (comparison, left, right) in enumerate(comparisons):
                    left_metrics, right_metrics = left["metrics"], right["metrics"]
                    assert isinstance(left_metrics, Mapping) and isinstance(right_metrics, Mapping)
                    samples = np.empty(0, dtype=float)
                    left_prediction, right_prediction = left["predictions_path"], right["predictions_path"]
                    if isinstance(left_prediction, str) and isinstance(right_prediction, str) and args.bootstrap_draws:
                        true, left_values, right_values, labels = _paired_arrays(Path(left_prediction), Path(right_prediction))
                        samples = _bootstrap_macro_f1_delta(
                            true, left_values, right_values, labels, args.bootstrap_draws,
                            args.bootstrap_seed + context_index * 10_000 + group_index * 100 + int(seed) + comparison_index,
                        )
                    ci_low, ci_high = _interval(samples)
                    comparison_rows.append({
                        "scope": args.scope,
                        "context": context,
                        "subgroup": group,
                        "seed": seed,
                        "comparison": comparison,
                        "left_metrics_path": left["metrics_path"],
                        "right_metrics_path": right["metrics_path"],
                        "directed_edges_removed": direct_removed,
                        "relation_taxonomy_sha256": taxonomy_sha256,
                        **{f"{metric}_delta": float(right_metrics[metric]) - float(left_metrics[metric]) for metric in METRICS},
                        "bootstrap_available": bool(len(samples)),
                        "bootstrap_draws": int(len(samples)),
                        "macro_f1_delta_bootstrap_ci_low": ci_low,
                        "macro_f1_delta_bootstrap_ci_high": ci_high,
                        "_bootstrap_samples": samples,
                    })

    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(str(row["context"]), str(row["subgroup"]), str(row["comparison"]))].append(row)
    for (context, group, comparison), rows in sorted(grouped.items()):
        summary: Dict[str, object] = {
            "scope": args.scope,
            "context": context,
            "subgroup": group,
            "comparison": comparison,
            "seeds_completed": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
            "directed_edges_removed_mean": mean(float(row["directed_edges_removed"]) for row in rows),
            "relation_taxonomy_sha256": taxonomy_sha256,
        }
        for metric in METRICS:
            values = [float(row[f"{metric}_delta"]) for row in rows]
            summary[f"{metric}_delta_mean"] = mean(values)
            summary[f"{metric}_delta_std"] = stdev(values) if len(values) > 1 else 0.0
        samples = [row["_bootstrap_samples"] for row in rows]
        if samples and all(len(value) for value in samples):
            ci_low, ci_high = _interval(np.mean(np.stack(samples), axis=0))
            summary.update({"bootstrap_available": True, "bootstrap_draws": len(samples[0]),
                            "macro_f1_delta_bootstrap_ci_low": ci_low, "macro_f1_delta_bootstrap_ci_high": ci_high})
        else:
            summary.update({"bootstrap_available": False, "bootstrap_draws": 0,
                            "macro_f1_delta_bootstrap_ci_low": None, "macro_f1_delta_bootstrap_ci_high": None})
        summary_rows.append(summary)

    for row in comparison_rows:
        row.pop("_bootstrap_samples")
    manifest = {
        "scope": args.scope,
        "groups": list(groups),
        "contexts": list(contexts),
        "relation_taxonomy_path": str(taxonomy_path.resolve()),
        "relation_taxonomy_sha256": taxonomy_sha256,
        "random_control_unit": "original_edge_instance_plus_generated_reverse",
        "bootstrap_draws_requested": args.bootstrap_draws,
    }
    output_dir = root / ("acquired_subgroup_summary" if args.scope == "full" else "period_acquired_subgroup_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "acquired_subgroup_condition_metrics.csv", condition_rows)
    _write_csv(output_dir / "acquired_subgroup_paired_deltas_by_seed.csv", comparison_rows)
    _write_csv(output_dir / "acquired_subgroup_paired_summary.csv", summary_rows)
    (output_dir / "acquired_subgroup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote acquired subgroup summaries to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['context']} {row['subgroup']} {row['comparison']}: "
            f"macro_f1_delta={float(row['macro_f1_delta_mean']):+.4f}±{float(row['macro_f1_delta_std']):.4f}"
        )


if __name__ == "__main__":
    main()
