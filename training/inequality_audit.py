#!/usr/bin/env python3
"""Measure whether relational message passing amplifies occupational disparities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from training.diagnose import tie_group_coverage, value_bucket
from training.tie_taxonomy import DEFAULT_TIE_TAXONOMY_PATH, load_tie_taxonomy


PRIMARY_MLP = "mlp_baseline"
PRIMARY_GNN = "rgcn_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tie-taxonomy", default=str(DEFAULT_TIE_TAXONOMY_PATH))
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence using natural logarithms."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / first.sum()
    second = second / second.sum()
    midpoint = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        keep = left > 0
        return float(np.sum(left[keep] * np.log(left[keep] / right[keep])))

    return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, num_classes: int, ece_bins: int
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    row = np.arange(len(labels))
    true_probability = probabilities[row, labels].clip(1e-12, 1.0)
    brier = np.square(probabilities).sum(axis=1) - 2.0 * true_probability + 1.0
    return {
        "nodes": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels,
            predictions,
            labels=np.arange(num_classes),
            average="macro",
            zero_division=0,
        )),
        "nll": float(-np.log(true_probability).mean()),
        "multiclass_brier": float(brier.mean()),
        "ece": expected_calibration_error(labels, probabilities, ece_bins),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def privilege_frame(data, metadata: Mapping[str, object], taxonomy) -> pd.DataFrame:
    target_feature = str(metadata["target_column"])
    unknown_id = int(metadata["occupation_unknown_ids"][target_feature])
    visible_train = (
        data.train_mask.detach().cpu().numpy()
        & (data.y.detach().cpu().numpy() >= 0)
        & (getattr(data, target_feature).detach().cpu().numpy() != unknown_id)
    )
    tie_columns, _ = tie_group_coverage(
        data, metadata["relation_to_id"], taxonomy, visible_train
    )
    inherited_visible = tie_columns["inherited_visible_train_occupation_messages"]
    inherited_direct = tie_columns["inherited_direct_messages"]
    relation_ids = {
        int(identifier)
        for relation, identifier in metadata["relation_to_id"].items()
        if taxonomy.group_for_base_relation(relation) == "inherited"
    }
    source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_type = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    inherited_edge = np.isin(edge_type, np.asarray(sorted(relation_ids), dtype=np.int64))
    node_count = int(data.num_nodes)
    directed_neighbour_keys = np.unique(target[inherited_edge] * node_count + source[inherited_edge])
    inherited_neighbours = np.bincount(
        directed_neighbour_keys // node_count, minlength=node_count
    )
    return pd.DataFrame({
        "node_index": np.arange(int(data.num_nodes), dtype=np.int64),
        "kin_information_access": np.where(inherited_visible > 0, "yes", "no"),
        "tie_exposure": tie_columns["tie_exposure"],
        "inherited_visible_messages": inherited_visible.astype(np.int64, copy=False),
        "inherited_visible_bucket": value_bucket(inherited_visible, "visible_neighbours"),
        "inherited_direct_messages": inherited_direct.astype(np.int64, copy=False),
        "inherited_distinct_neighbours": inherited_neighbours.astype(np.int64, copy=False),
        "inherited_degree_bucket": value_bucket(inherited_neighbours, "degree"),
    })


def discover_runs(root: Path) -> list[dict[str, object]]:
    records = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        probability_path = manifest_path.parent / "probabilities.npz"
        if not probability_path.is_file():
            continue
        if "condition" not in manifest or "seed" not in manifest:
            continue
        records.append({
            "condition": str(manifest["condition"]),
            "seed": int(manifest["seed"]),
            "manifest_path": manifest_path,
            "probability_path": probability_path,
            "manifest": manifest,
        })
    if not records:
        raise FileNotFoundError(f"No inequality prediction manifests found below {root}")
    keys = [(record["condition"], record["seed"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Prediction root contains duplicate condition/seed manifests")
    return records


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        result = {key: payload[key] for key in ("node_index", "label", "probabilities")}
    if result["probabilities"].ndim != 2:
        raise ValueError(f"Expected a [node,class] probability matrix in {path}")
    if len(result["node_index"]) != len(result["label"]) or len(result["label"]) != len(result["probabilities"]):
        raise ValueError(f"Prediction arrays have inconsistent lengths in {path}")
    if len(np.unique(result["node_index"])) != len(result["node_index"]):
        raise ValueError(f"Prediction arrays contain duplicate node indices in {path}")
    return result


def align_to_reference(arrays: dict[str, np.ndarray], nodes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order = np.argsort(arrays["node_index"])
    sorted_nodes = arrays["node_index"][order]
    positions = np.searchsorted(sorted_nodes, nodes)
    if np.any(positions >= len(sorted_nodes)) or not np.array_equal(sorted_nodes[positions], nodes):
        raise ValueError("Runs do not contain exactly the same audit nodes")
    aligned = order[positions]
    if not np.array_equal(arrays["label"][aligned], labels):
        raise ValueError("Runs disagree on true labels for the audit nodes")
    return arrays["probabilities"][aligned]


def distribution_metrics(
    labels: np.ndarray, probabilities: np.ndarray, kin_access: np.ndarray, num_classes: int
) -> dict[str, float]:
    yes, no = kin_access == "yes", kin_access == "no"
    if not yes.any() or not no.any():
        raise ValueError("Both kin-information-access groups must be non-empty")
    true_yes = np.bincount(labels[yes], minlength=num_classes).astype(np.float64)
    true_no = np.bincount(labels[no], minlength=num_classes).astype(np.float64)
    soft_yes, soft_no = probabilities[yes].mean(axis=0), probabilities[no].mean(axis=0)
    pred = probabilities.argmax(axis=1)
    hard_yes = np.bincount(pred[yes], minlength=num_classes).astype(np.float64)
    hard_no = np.bincount(pred[no], minlength=num_classes).astype(np.float64)
    true_js = js_divergence(true_yes, true_no)
    soft_js = js_divergence(soft_yes, soft_no)
    hard_js = js_divergence(hard_yes, hard_no)
    error_yes = float(np.mean(pred[yes] != labels[yes]))
    error_no = float(np.mean(pred[no] != labels[no]))
    return {
        "true_group_js": true_js,
        "soft_prediction_group_js": soft_js,
        "argmax_prediction_group_js": hard_js,
        "soft_distribution_amplification": soft_js - true_js,
        "argmax_distribution_amplification": hard_js - true_js,
        "error_kin_access_yes": error_yes,
        "error_kin_access_no": error_no,
        "kin_access_error_gap_no_minus_yes": error_no - error_yes,
    }


def class_amplification_records(
    condition: str,
    seed: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    kin_access: np.ndarray,
    id_to_label: Mapping[int, str],
) -> list[dict[str, object]]:
    yes, no = kin_access == "yes", kin_access == "no"
    num_classes = probabilities.shape[1]
    smoothing = 0.5

    def smoothed_true(mask: np.ndarray) -> np.ndarray:
        counts = np.bincount(labels[mask], minlength=num_classes).astype(np.float64)
        return (counts + smoothing) / (counts.sum() + smoothing * num_classes)

    def smoothed_soft(mask: np.ndarray) -> np.ndarray:
        counts = probabilities[mask].sum(axis=0, dtype=np.float64)
        return (counts + smoothing) / (counts.sum() + smoothing * num_classes)

    true_yes, true_no = smoothed_true(yes), smoothed_true(no)
    pred_yes, pred_no = smoothed_soft(yes), smoothed_soft(no)
    data_log_odds = np.log(true_yes / (1.0 - true_yes)) - np.log(true_no / (1.0 - true_no))
    model_log_odds = np.log(pred_yes / (1.0 - pred_yes)) - np.log(pred_no / (1.0 - pred_no))
    support_yes = np.bincount(labels[yes], minlength=num_classes)
    support_no = np.bincount(labels[no], minlength=num_classes)
    return [{
        "condition": condition,
        "seed": seed,
        "class_id": class_id,
        "occupation": id_to_label[class_id],
        "true_support_kin_yes": int(support_yes[class_id]),
        "true_support_kin_no": int(support_no[class_id]),
        "data_log_odds_gap_yes_minus_no": float(data_log_odds[class_id]),
        "model_log_odds_gap_yes_minus_no": float(model_log_odds[class_id]),
        "directional_log_odds_amplification": float(model_log_odds[class_id] - data_log_odds[class_id]),
    } for class_id in range(num_classes)]


def bootstrap_message_passing_difference(
    labels: np.ndarray,
    kin_access: np.ndarray,
    mlp: np.ndarray,
    rgcn: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    if draws <= 0:
        return {"bootstrap_draws": 0, "error_gap_ci_low": None, "error_gap_ci_high": None,
                "distribution_ci_low": None, "distribution_ci_high": None}
    rng = np.random.default_rng(seed)
    yes_positions, no_positions = np.flatnonzero(kin_access == "yes"), np.flatnonzero(kin_access == "no")
    error_differences, distribution_differences = [], []
    for _ in range(draws):
        sampled_yes = rng.choice(yes_positions, size=len(yes_positions), replace=True)
        sampled_no = rng.choice(no_positions, size=len(no_positions), replace=True)
        sampled = np.concatenate([sampled_yes, sampled_no])
        sampled_access = kin_access[sampled]
        mlp_metrics = distribution_metrics(labels[sampled], mlp[sampled], sampled_access, mlp.shape[1])
        rgcn_metrics = distribution_metrics(labels[sampled], rgcn[sampled], sampled_access, rgcn.shape[1])
        error_differences.append(
            rgcn_metrics["kin_access_error_gap_no_minus_yes"]
            - mlp_metrics["kin_access_error_gap_no_minus_yes"]
        )
        distribution_differences.append(
            rgcn_metrics["soft_distribution_amplification"]
            - mlp_metrics["soft_distribution_amplification"]
        )
    error_low, error_high = np.quantile(error_differences, [0.025, 0.975])
    distribution_low, distribution_high = np.quantile(distribution_differences, [0.025, 0.975])
    return {
        "bootstrap_draws": int(draws),
        "error_gap_ci_low": float(error_low),
        "error_gap_ci_high": float(error_high),
        "distribution_ci_low": float(distribution_low),
        "distribution_ci_high": float(distribution_high),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 0 or args.ece_bins < 2:
        raise ValueError("bootstrap draws must be non-negative and ECE bins at least two")
    data_path, predictions_root = Path(args.data), Path(args.predictions_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    data, metadata = bundle["data"], bundle["metadata"]
    taxonomy = load_tie_taxonomy(args.tie_taxonomy, metadata["relation_to_id"])
    privilege = privilege_frame(data, metadata, taxonomy)
    test_nodes = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy().astype(np.int64)
    test_labels = data.y[test_nodes].cpu().numpy().astype(np.int64)
    if np.any(test_labels < 0):
        raise ValueError("Test mask contains ignored labels")
    test_privilege = privilege.set_index("node_index").loc[test_nodes].reset_index()
    test_privilege.to_csv(output_dir / "test_node_privilege_groups.csv", index=False)
    kin_access = test_privilege["kin_information_access"].to_numpy()
    num_classes = int(metadata["num_classes"])
    id_to_label = {int(identifier): str(label) for label, identifier in metadata["label_to_id"].items()}

    run_records, group_records, class_records = [], [], []
    probabilities_by_key: dict[tuple[str, int], np.ndarray] = {}
    manifests = discover_runs(predictions_root)
    for discovered in manifests:
        condition, seed = str(discovered["condition"]), int(discovered["seed"])
        arrays = load_arrays(discovered["probability_path"])
        probabilities = align_to_reference(arrays, test_nodes, test_labels)
        if probabilities.shape[1] != num_classes:
            raise ValueError(f"{condition} seed {seed} uses a different class count")
        probabilities_by_key[(condition, seed)] = probabilities
        overall = classification_metrics(test_labels, probabilities, num_classes, args.ece_bins)
        distribution = distribution_metrics(test_labels, probabilities, kin_access, num_classes)
        run_records.append({"condition": condition, "seed": seed, **overall, **distribution})
        for stratifier in (
            "kin_information_access", "tie_exposure", "inherited_visible_bucket", "inherited_degree_bucket"
        ):
            values = test_privilege[stratifier].to_numpy()
            for group in sorted(np.unique(values).tolist()):
                selected = values == group
                group_records.append({
                    "condition": condition,
                    "seed": seed,
                    "stratifier": stratifier,
                    "group": group,
                    **classification_metrics(
                        test_labels[selected], probabilities[selected], num_classes, args.ece_bins
                    ),
                })
        class_records.extend(class_amplification_records(
            condition, seed, test_labels, probabilities, kin_access, id_to_label
        ))

    runs = pd.DataFrame(run_records).sort_values(["condition", "seed"]).reset_index(drop=True)
    runs.to_csv(output_dir / "run_metrics.csv", index=False)
    pd.DataFrame(group_records).to_csv(output_dir / "group_metrics.csv", index=False)
    pd.DataFrame(class_records).to_csv(output_dir / "class_directional_amplification.csv", index=False)

    pair_records = []
    seeds = sorted({seed for condition, seed in probabilities_by_key if condition == PRIMARY_GNN})
    run_lookup = runs.set_index(["condition", "seed"])
    for seed in seeds:
        if (PRIMARY_MLP, seed) not in probabilities_by_key:
            raise ValueError(f"Missing {PRIMARY_MLP} seed {seed} required by the primary comparison")
        rgcn_row, mlp_row = run_lookup.loc[(PRIMARY_GNN, seed)], run_lookup.loc[(PRIMARY_MLP, seed)]
        record = {
            "comparison": "message_passing_rgcn_minus_mlp",
            "seed": seed,
            "macro_f1_delta": float(rgcn_row["macro_f1"] - mlp_row["macro_f1"]),
            "message_passing_error_gap_amplification": float(
                rgcn_row["kin_access_error_gap_no_minus_yes"]
                - mlp_row["kin_access_error_gap_no_minus_yes"]
            ),
            "message_passing_soft_distribution_amplification": float(
                rgcn_row["soft_distribution_amplification"]
                - mlp_row["soft_distribution_amplification"]
            ),
        }
        record.update(bootstrap_message_passing_difference(
            test_labels,
            kin_access,
            probabilities_by_key[(PRIMARY_MLP, seed)],
            probabilities_by_key[(PRIMARY_GNN, seed)],
            args.bootstrap_draws,
            args.bootstrap_seed + seed,
        ))
        pair_records.append(record)

        for condition in sorted({condition for condition, candidate_seed in probabilities_by_key if candidate_seed == seed}):
            if condition in {PRIMARY_MLP, PRIMARY_GNN}:
                continue
            condition_row = run_lookup.loc[(condition, seed)]
            pair_records.append({
                "comparison": f"{condition}_minus_rgcn_full",
                "seed": seed,
                "macro_f1_delta": float(condition_row["macro_f1"] - rgcn_row["macro_f1"]),
                "message_passing_error_gap_amplification": float(
                    condition_row["kin_access_error_gap_no_minus_yes"]
                    - rgcn_row["kin_access_error_gap_no_minus_yes"]
                ),
                "message_passing_soft_distribution_amplification": float(
                    condition_row["soft_distribution_amplification"]
                    - rgcn_row["soft_distribution_amplification"]
                ),
                "bootstrap_draws": 0,
                "error_gap_ci_low": None,
                "error_gap_ci_high": None,
                "distribution_ci_low": None,
                "distribution_ci_high": None,
            })

    pairs = pd.DataFrame(pair_records).sort_values(["comparison", "seed"]).reset_index(drop=True)
    pairs.to_csv(output_dir / "paired_comparisons_by_seed.csv", index=False)
    summary = pairs.groupby("comparison", sort=True).agg(
        seeds=("seed", "nunique"),
        macro_f1_delta_mean=("macro_f1_delta", "mean"),
        macro_f1_delta_std=("macro_f1_delta", "std"),
        error_gap_amplification_mean=("message_passing_error_gap_amplification", "mean"),
        error_gap_amplification_std=("message_passing_error_gap_amplification", "std"),
        soft_distribution_amplification_mean=("message_passing_soft_distribution_amplification", "mean"),
        soft_distribution_amplification_std=("message_passing_soft_distribution_amplification", "std"),
    ).reset_index()
    summary.to_csv(output_dir / "paired_comparison_summary.csv", index=False)

    audit_manifest = {
        "format_version": 1,
        "data": str(data_path.resolve()),
        "predictions_root": str(predictions_root.resolve()),
        "tie_taxonomy": taxonomy.manifest(),
        "primary_group": "kin_information_access: at least one model-visible training occupation message over an inherited tie",
        "primary_comparison": f"{PRIMARY_GNN} minus {PRIMARY_MLP}",
        "test_nodes": int(len(test_nodes)),
        "kin_access_yes": int(np.sum(kin_access == "yes")),
        "kin_access_no": int(np.sum(kin_access == "no")),
        "conditions": sorted(runs["condition"].unique().tolist()),
        "seeds": sorted(runs["seed"].unique().astype(int).tolist()),
        "bootstrap_draws": int(args.bootstrap_draws),
        "interpretation": (
            "Positive message-passing amplification means RGCN increases the no-kin versus kin error gap "
            "or the between-group occupational-distribution divergence relative to the no-message MLP. "
            "This is a model audit, not a causal estimate of social inequality."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_manifest, handle, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print(f"Wrote inequality audit to {output_dir}")


if __name__ == "__main__":
    main()
