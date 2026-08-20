#!/usr/bin/env python3
"""Report direct GraphMask message dependence by L1 occupation pair and relation.

For every test root ``v``, this command considers only typed messages whose
destination is ``v`` and aggregates them by
``(layer, source L1, exact directed relation, target L1)``.  The frozen model
and an already-trained GraphMask probe are reused; no parameters are fitted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from training.attention_common import (
    git_revision,
    root_indices,
    sha256_file,
    source_visibility_codes,
)
from training.graphmask.common import (
    graphmask_kl,
    load_probe,
    load_run_context,
    make_loader,
    resolve_device,
    resolve_fanouts,
    set_seed,
    validate_probe_sources,
    write_json,
)
from training.train import batch_features


VISIBILITY_NAMES = {
    0: "visible_train",
    1: "hidden_validation_or_test",
    2: "missing_or_unknown",
}
COHORT_ALL = "all_test"
COHORT_CORRECT = "correct_only"


class DirectMessageGroup(NamedTuple):
    relation_id: int
    source_l1_id: int
    visibility_id: int
    candidate_message_n: int
    hard_retained_message_n: int
    expected_retained_mass: float


class PairKey(NamedTuple):
    layer: int
    relation_id: int
    relation: str
    source_l1_id: int
    source_l1: str
    target_l1_id: int
    target_l1: str


@dataclass
class PairAggregate:
    candidate_message_n: int = 0
    hard_retained_message_n: int = 0
    expected_retained_mass: float = 0.0
    candidate_node_count: int = 0
    retained_node_count: int = 0

    def add_root_group(
        self,
        candidate_message_n: int,
        hard_retained_message_n: int,
        expected_retained_mass: float,
    ) -> None:
        """Add one already-collapsed root/pair observation."""
        candidate = int(candidate_message_n)
        hard = int(hard_retained_message_n)
        if candidate < 1:
            raise ValueError("A root/pair group must contain at least one candidate message")
        if hard < 0 or hard > candidate:
            raise ValueError("Hard-retained messages must lie between zero and candidates")
        expected = float(expected_retained_mass)
        if expected < 0.0 or expected > candidate + 1e-6:
            raise ValueError("Expected retained mass must lie between zero and candidates")
        self.candidate_message_n += candidate
        self.hard_retained_message_n += hard
        self.expected_retained_mass += expected
        self.candidate_node_count += 1
        if hard:
            self.retained_node_count += 1


@dataclass
class Budget:
    candidate_message_n: int = 0
    hard_retained_message_n: int = 0
    expected_retained_mass: float = 0.0

    def add(self, stats: PairAggregate) -> None:
        self.candidate_message_n += stats.candidate_message_n
        self.hard_retained_message_n += stats.hard_retained_message_n
        self.expected_retained_mass += stats.expected_retained_mass


SUMMARY_FIELDS = [
    "cohort",
    "normalization_scope",
    "layer",
    "source_l1_id",
    "source_l1",
    "relation_id",
    "relation",
    "target_l1_id",
    "target_l1",
    "candidate_message_n",
    "hard_retained_message_n",
    "expected_retained_mass",
    "candidate_node_count",
    "retained_node_count",
    "target_node_count",
    "candidate_node_coverage",
    "retained_node_coverage",
    "hard_retention_rate",
    "mean_keep_probability",
    "opportunity_share_of_Ot_budget",
    "retained_share_of_Ot_budget",
    "expected_retained_share_of_Ot_budget",
    "opportunity_share_within_R_Ot",
    "retained_share_within_R_Ot",
    "expected_retained_share_within_R_Ot",
    "retention_lift_within_R_Ot",
    "expected_lift_within_R_Ot",
    "support_eligible",
]


RAW_FIELDS = [
    "split",
    "root_index",
    "root_id",
    "target_l1_id",
    "target_l1",
    "original_prediction_id",
    "original_prediction",
    "masked_prediction_id",
    "masked_prediction",
    "original_correct",
    "layer",
    "relation_id",
    "relation",
    "source_l1_id",
    "source_l1",
    "source_visibility",
    "candidate_message_n",
    "hard_retained_message_n",
    "expected_retained_mass",
]


BUDGET_FIELDS = [
    "cohort",
    "normalization_scope",
    "budget_scope",
    "layer",
    "target_l1_id",
    "target_l1",
    "relation_id",
    "relation",
    "target_node_count",
    "candidate_message_n",
    "hard_retained_message_n",
    "expected_retained_mass",
    "hard_retention_rate",
    "mean_keep_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument(
        "--num-neighbors",
        default="auto",
        help="auto reuses the probe fan-outs; full or explicit fan-outs override them",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Must be one so every direct message is assigned to exactly one prediction root",
    )
    parser.add_argument("--min-node-support", type=int, default=100)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Reserved for clustered node bootstrap; this report currently requires zero",
    )
    parser.add_argument("--seed", type=int, default=None, help="Neighbour-sampling seed")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def group_direct_messages(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_labels: torch.Tensor,
    node_visibility: torch.Tensor,
    hard_gate: torch.Tensor,
    keep_probability: torch.Tensor,
    class_count: int,
    root_local_index: int = 0,
) -> list[DirectMessageGroup]:
    """Collapse direct typed messages for one root without crossing visibility groups."""
    edge_count = int(edge_index.size(1))
    if edge_type.numel() != edge_count:
        raise ValueError("edge_type and edge_index disagree on the message count")
    if hard_gate.numel() != edge_count or keep_probability.numel() != edge_count:
        raise ValueError("GraphMask gates and traced edges disagree on the message count")
    if node_labels.numel() != node_visibility.numel():
        raise ValueError("Node labels and visibility codes must have the same length")
    if class_count < 1:
        raise ValueError("class_count must be positive")

    selected = edge_index[1].eq(int(root_local_index))
    if not bool(selected.any()):
        return []
    source_indices = edge_index[0, selected].long()
    relation_ids = edge_type[selected].long()
    source_labels = node_labels[source_indices].long()
    visibility = node_visibility[source_indices].long()
    if bool(((visibility < 0) | (visibility > 2)).any()):
        raise ValueError("Source visibility codes must be zero, one, or two")
    source_slots = source_labels.clone()
    source_slots[source_slots < 0] = class_count
    if bool(source_slots.gt(class_count).any()):
        raise ValueError("Source labels exceed the artifact class count")

    encoded = ((relation_ids * (class_count + 1) + source_slots) * 3) + visibility
    unique, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
    candidates = torch.bincount(inverse, minlength=unique.numel())
    hard_sums = torch.zeros(unique.numel(), dtype=torch.float64, device=edge_index.device)
    hard_sums.scatter_add_(0, inverse, hard_gate[selected].to(dtype=torch.float64))
    probability_sums = torch.zeros_like(hard_sums)
    probability_sums.scatter_add_(
        0, inverse, keep_probability[selected].to(dtype=torch.float64)
    )

    groups: list[DirectMessageGroup] = []
    for encoded_value, candidate, hard, probability in zip(
        unique.detach().cpu().tolist(),
        candidates.detach().cpu().tolist(),
        hard_sums.detach().cpu().tolist(),
        probability_sums.detach().cpu().tolist(),
    ):
        visibility_id = int(encoded_value % 3)
        decoded = int(encoded_value // 3)
        source_slot = int(decoded % (class_count + 1))
        relation_id = int(decoded // (class_count + 1))
        source_l1_id = -1 if source_slot == class_count else source_slot
        hard_count = int(round(float(hard)))
        if abs(float(hard) - hard_count) > 1e-6:
            raise RuntimeError("GraphMask eval gates must be binary for hard edge counts")
        groups.append(DirectMessageGroup(
            relation_id=relation_id,
            source_l1_id=source_l1_id,
            visibility_id=visibility_id,
            candidate_message_n=int(candidate),
            hard_retained_message_n=hard_count,
            expected_retained_mass=float(probability),
        ))
    return groups


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def summarize_pair_aggregates(
    aggregates: Mapping[PairKey, PairAggregate],
    target_node_counts: Mapping[tuple[int, int, str], int],
    cohort: str,
    min_node_support: int,
    normalization_scope: str = "all_known_sources",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute target-wide shares and relation-target conditional lift."""
    target_budgets: dict[tuple[int, int, str], Budget] = defaultdict(Budget)
    relation_target_budgets: dict[tuple[int, int, str, int, str], Budget] = defaultdict(Budget)
    for target_key in target_node_counts:
        target_budgets[target_key]
    for key, stats in aggregates.items():
        target_key = (key.layer, key.target_l1_id, key.target_l1)
        relation_target_key = (
            key.layer,
            key.target_l1_id,
            key.target_l1,
            key.relation_id,
            key.relation,
        )
        target_budgets[target_key].add(stats)
        relation_target_budgets[relation_target_key].add(stats)

    rows: list[dict[str, object]] = []
    for key, stats in sorted(aggregates.items()):
        target_key = (key.layer, key.target_l1_id, key.target_l1)
        relation_target_key = (
            key.layer,
            key.target_l1_id,
            key.target_l1,
            key.relation_id,
            key.relation,
        )
        target_budget = target_budgets[target_key]
        relation_target_budget = relation_target_budgets[relation_target_key]
        target_node_count = int(target_node_counts.get(target_key, 0))
        opportunity_within_relation = _ratio(
            stats.candidate_message_n,
            relation_target_budget.candidate_message_n,
        )
        retained_within_relation = _ratio(
            stats.hard_retained_message_n,
            relation_target_budget.hard_retained_message_n,
        )
        expected_within_relation = _ratio(
            stats.expected_retained_mass,
            relation_target_budget.expected_retained_mass,
        )
        rows.append({
            "cohort": cohort,
            "normalization_scope": normalization_scope,
            "layer": key.layer,
            "source_l1_id": key.source_l1_id,
            "source_l1": key.source_l1,
            "relation_id": key.relation_id,
            "relation": key.relation,
            "target_l1_id": key.target_l1_id,
            "target_l1": key.target_l1,
            "candidate_message_n": stats.candidate_message_n,
            "hard_retained_message_n": stats.hard_retained_message_n,
            "expected_retained_mass": stats.expected_retained_mass,
            "candidate_node_count": stats.candidate_node_count,
            "retained_node_count": stats.retained_node_count,
            "target_node_count": target_node_count,
            "candidate_node_coverage": _ratio(
                stats.candidate_node_count, target_node_count
            ),
            "retained_node_coverage": _ratio(
                stats.retained_node_count, target_node_count
            ),
            "hard_retention_rate": _ratio(
                stats.hard_retained_message_n, stats.candidate_message_n
            ),
            "mean_keep_probability": _ratio(
                stats.expected_retained_mass, stats.candidate_message_n
            ),
            "opportunity_share_of_Ot_budget": _ratio(
                stats.candidate_message_n, target_budget.candidate_message_n
            ),
            "retained_share_of_Ot_budget": _ratio(
                stats.hard_retained_message_n,
                target_budget.hard_retained_message_n,
            ),
            "expected_retained_share_of_Ot_budget": _ratio(
                stats.expected_retained_mass,
                target_budget.expected_retained_mass,
            ),
            "opportunity_share_within_R_Ot": opportunity_within_relation,
            "retained_share_within_R_Ot": retained_within_relation,
            "expected_retained_share_within_R_Ot": expected_within_relation,
            "retention_lift_within_R_Ot": (
                _ratio(retained_within_relation, opportunity_within_relation)
                if retained_within_relation is not None
                and opportunity_within_relation is not None
                else None
            ),
            "expected_lift_within_R_Ot": (
                _ratio(expected_within_relation, opportunity_within_relation)
                if expected_within_relation is not None
                and opportunity_within_relation is not None
                else None
            ),
            "support_eligible": stats.candidate_node_count >= min_node_support,
        })

    budget_rows: list[dict[str, object]] = []
    for key, budget in sorted(target_budgets.items()):
        layer, target_l1_id, target_l1 = key
        budget_rows.append(_budget_row(
            cohort,
            normalization_scope,
            "target_occupation",
            layer,
            target_l1_id,
            target_l1,
            None,
            None,
            int(target_node_counts.get(key, 0)),
            budget,
        ))
    for key, budget in sorted(relation_target_budgets.items()):
        layer, target_l1_id, target_l1, relation_id, relation = key
        budget_rows.append(_budget_row(
            cohort,
            normalization_scope,
            "relation_target_occupation",
            layer,
            target_l1_id,
            target_l1,
            relation_id,
            relation,
            int(target_node_counts.get((layer, target_l1_id, target_l1), 0)),
            budget,
        ))
    return rows, budget_rows


def _budget_row(
    cohort: str,
    normalization_scope: str,
    budget_scope: str,
    layer: int,
    target_l1_id: int,
    target_l1: str,
    relation_id: int | None,
    relation: str | None,
    target_node_count: int,
    budget: Budget,
) -> dict[str, object]:
    return {
        "cohort": cohort,
        "normalization_scope": normalization_scope,
        "budget_scope": budget_scope,
        "layer": layer,
        "target_l1_id": target_l1_id,
        "target_l1": target_l1,
        "relation_id": relation_id,
        "relation": relation,
        "target_node_count": target_node_count,
        "candidate_message_n": budget.candidate_message_n,
        "hard_retained_message_n": budget.hard_retained_message_n,
        "expected_retained_mass": budget.expected_retained_mass,
        "hard_retention_rate": _ratio(
            budget.hard_retained_message_n, budget.candidate_message_n
        ),
        "mean_keep_probability": _ratio(
            budget.expected_retained_mass, budget.candidate_message_n
        ),
    }


def ranked_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Rank only support-eligible all-test rows under the two primary estimands."""
    eligible = [
        dict(row)
        for row in rows
        if row["cohort"] == COHORT_ALL and bool(row["support_eligible"])
    ]
    global_groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    conditional_groups: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        global_groups[(int(row["layer"]), int(row["target_l1_id"]))].append(row)
        conditional_groups[(
            int(row["layer"]),
            int(row["relation_id"]),
            int(row["target_l1_id"]),
        )].append(row)

    for group in global_groups.values():
        group.sort(key=lambda row: (
            -float(row["retained_share_of_Ot_budget"] or 0.0),
            -int(row["hard_retained_message_n"]),
            str(row["relation"]),
            str(row["source_l1"]),
        ))
        for rank, row in enumerate(group, start=1):
            row["global_dependency_rank_within_Ot"] = rank
    for group in conditional_groups.values():
        group.sort(key=lambda row: (
            -float(row["retention_lift_within_R_Ot"] or 0.0),
            -int(row["candidate_node_count"]),
            str(row["source_l1"]),
        ))
        for rank, row in enumerate(group, start=1):
            row["conditional_lift_rank_within_R_Ot"] = rank

    return sorted(eligible, key=lambda row: (
        int(row["layer"]),
        int(row["target_l1_id"]),
        int(row.get("global_dependency_rank_within_Ot", 0)),
    ))


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _node_ids(data_path: Path, count: int) -> list[str]:
    path = data_path.parent / "nodes.csv"
    if not path.is_file():
        return [str(index) for index in range(count)]
    with path.open(newline="", encoding="utf-8") as handle:
        values = [str(row["node_id"]) for row in csv.DictReader(handle)]
    if len(values) != count:
        raise ValueError(f"{path} has {len(values)} rows but graph_data.pt has {count} nodes")
    return values


def _metric_block(labels: list[int], predictions: list[int]) -> dict[str, float | None]:
    if not labels:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def _update_pair(
    aggregates: dict[PairKey, PairAggregate],
    key: PairKey,
    candidate: int,
    hard: int,
    expected: float,
) -> None:
    aggregates.setdefault(key, PairAggregate()).add_root_group(
        candidate, hard, expected
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must be one for root-specific direct-message attribution")
    if args.min_node_support < 1:
        raise ValueError("--min-node-support must be positive")
    if args.bootstrap_replicates != 0:
        raise ValueError("--bootstrap-replicates must be zero in this report version")

    device = resolve_device(args.device)
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    probe_path = Path(args.probe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, checkpoint, restored = load_run_context(
        data_path, checkpoint_path, device
    )
    if metadata.get("target_column") != "occupation_level1":
        raise ValueError(
            "graphmask-occupation-pair-report requires an occupation_level1 artifact"
        )
    if restored.model_name != "rgat":
        raise ValueError("This first occupation-pair report is scoped to an RGAT checkpoint")
    probe, probe_metadata = load_probe(probe_path, device)
    validate_probe_sources(probe_metadata, data_path, checkpoint_path)
    if len(probe.gates) != len(restored.adapter.layers):
        raise ValueError("Probe depth does not match the source checkpoint")
    if args.num_neighbors == "auto":
        fanouts = [int(value) for value in probe_metadata["fanouts"]]
    else:
        fanouts = resolve_fanouts(checkpoint_path, checkpoint, args.num_neighbors)
    report_seed = int(probe_metadata["seed"] if args.seed is None else args.seed)
    set_seed(report_seed)

    roots = root_indices(data, args.split)
    loader = make_loader(
        data, roots, fanouts, batch_size=1, shuffle=False, num_workers=0
    )
    node_ids = _node_ids(data_path, int(data.num_nodes))
    id_to_relation = {
        int(index): str(relation)
        for relation, index in metadata["relation_to_id"].items()
    }
    id_to_label = {
        int(index): str(label) for label, index in metadata["label_to_id"].items()
    }
    class_count = int(metadata["num_classes"])

    cohort_aggregates: dict[str, dict[PairKey, PairAggregate]] = {
        COHORT_ALL: {},
        COHORT_CORRECT: {},
    }
    visibility_aggregates: dict[
        str, dict[str, dict[PairKey, PairAggregate]]
    ] = {
        COHORT_ALL: {name: {} for name in VISIBILITY_NAMES.values()},
        COHORT_CORRECT: {name: {} for name in VISIBILITY_NAMES.values()},
    }
    target_node_counts: dict[str, dict[tuple[int, int, str], int]] = {
        COHORT_ALL: defaultdict(int),
        COHORT_CORRECT: defaultdict(int),
    }
    unknown_audit: dict[int, Budget] = defaultdict(Budget)
    layer_hard = [0.0 for _ in probe.gates]
    layer_probability = [0.0 for _ in probe.gates]
    layer_count = [0 for _ in probe.gates]
    labels: list[int] = []
    original_predictions: list[int] = []
    masked_predictions: list[int] = []
    all_original_predictions: list[int] = []
    all_masked_predictions: list[int] = []
    divergences: list[float] = []
    correct_root_count = 0

    raw_path = output_dir / "root_direct_graphmask_pairs.csv.gz"
    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
        raw_writer.writeheader()
        with torch.no_grad():
            probe.eval()
            for batch in loader:
                batch = batch.to(device)
                if int(batch.batch_size) != 1:
                    raise RuntimeError("The occupation-pair loader yielded multiple roots")
                features = batch_features(
                    batch,
                    restored.feature_schema,
                    metadata["occupation_unknown_ids"],
                )
                original_logits, traces = restored.adapter.trace(
                    features, batch.edge_index, batch.edge_type
                )
                gates, probabilities, _, _ = probe(traces)
                masked_logits = restored.adapter.masked_forward(
                    features,
                    batch.edge_index,
                    batch.edge_type,
                    gates,
                    probe.baselines,
                )
                root_global = int(batch.n_id[0].item())
                root_id = node_ids[root_global]
                target_l1_id = int(batch.y[0].item())
                target_l1 = id_to_label.get(target_l1_id, "__UNLABELED__")
                original_probability = original_logits[0].softmax(dim=-1)
                masked_probability = masked_logits[0].softmax(dim=-1)
                original_prediction_id = int(original_probability.argmax().item())
                masked_prediction_id = int(masked_probability.argmax().item())
                original_correct = (
                    target_l1_id >= 0 and original_prediction_id == target_l1_id
                )
                correct_root_count += int(original_correct)
                all_original_predictions.append(original_prediction_id)
                all_masked_predictions.append(masked_prediction_id)
                if target_l1_id >= 0:
                    labels.append(target_l1_id)
                    original_predictions.append(original_prediction_id)
                    masked_predictions.append(masked_prediction_id)
                divergences.append(float(graphmask_kl(
                    original_logits[:1], masked_logits[:1]
                ).item()))

                node_visibility = source_visibility_codes(
                    batch,
                    torch.arange(batch.num_nodes, device=device),
                    restored.feature_schema,
                    metadata["occupation_unknown_ids"],
                )
                for layer, (trace, gate, probability) in enumerate(
                    zip(traces, gates, probabilities)
                ):
                    layer_hard[layer] += float(gate.sum().item())
                    layer_probability[layer] += float(probability.sum().item())
                    layer_count[layer] += int(gate.numel())
                    if target_l1_id >= 0:
                        target_key = (layer, target_l1_id, target_l1)
                        target_node_counts[COHORT_ALL][target_key] += 1
                        if original_correct:
                            target_node_counts[COHORT_CORRECT][target_key] += 1

                    groups = group_direct_messages(
                        trace.edge_index,
                        trace.edge_type,
                        batch.y,
                        node_visibility,
                        gate,
                        probability,
                        class_count,
                    )
                    root_combined: dict[PairKey, list[float]] = {}
                    for group in groups:
                        relation = id_to_relation[group.relation_id]
                        source_l1 = id_to_label.get(
                            group.source_l1_id, "__UNLABELED__"
                        )
                        visibility = VISIBILITY_NAMES[group.visibility_id]
                        raw_writer.writerow({
                            "split": args.split,
                            "root_index": root_global,
                            "root_id": root_id,
                            "target_l1_id": target_l1_id,
                            "target_l1": target_l1,
                            "original_prediction_id": original_prediction_id,
                            "original_prediction": id_to_label[original_prediction_id],
                            "masked_prediction_id": masked_prediction_id,
                            "masked_prediction": id_to_label[masked_prediction_id],
                            "original_correct": int(original_correct),
                            "layer": layer,
                            "relation_id": group.relation_id,
                            "relation": relation,
                            "source_l1_id": group.source_l1_id,
                            "source_l1": source_l1,
                            "source_visibility": visibility,
                            "candidate_message_n": group.candidate_message_n,
                            "hard_retained_message_n": group.hard_retained_message_n,
                            "expected_retained_mass": group.expected_retained_mass,
                        })
                        if target_l1_id < 0 or group.source_l1_id < 0:
                            if group.source_l1_id < 0:
                                audit = PairAggregate(
                                    candidate_message_n=group.candidate_message_n,
                                    hard_retained_message_n=group.hard_retained_message_n,
                                    expected_retained_mass=group.expected_retained_mass,
                                )
                                unknown_audit[layer].add(audit)
                            continue

                        key = PairKey(
                            layer,
                            group.relation_id,
                            relation,
                            group.source_l1_id,
                            source_l1,
                            target_l1_id,
                            target_l1,
                        )
                        combined = root_combined.setdefault(key, [0.0, 0.0, 0.0])
                        combined[0] += group.candidate_message_n
                        combined[1] += group.hard_retained_message_n
                        combined[2] += group.expected_retained_mass
                        _update_pair(
                            visibility_aggregates[COHORT_ALL][visibility],
                            key,
                            group.candidate_message_n,
                            group.hard_retained_message_n,
                            group.expected_retained_mass,
                        )
                        if original_correct:
                            _update_pair(
                                visibility_aggregates[COHORT_CORRECT][visibility],
                                key,
                                group.candidate_message_n,
                                group.hard_retained_message_n,
                                group.expected_retained_mass,
                            )

                    for key, values in root_combined.items():
                        candidate, hard, expected = values
                        _update_pair(
                            cohort_aggregates[COHORT_ALL],
                            key,
                            int(candidate),
                            int(hard),
                            expected,
                        )
                        if original_correct:
                            _update_pair(
                                cohort_aggregates[COHORT_CORRECT],
                                key,
                                int(candidate),
                                int(hard),
                                expected,
                            )

    all_rows, all_budgets = summarize_pair_aggregates(
        cohort_aggregates[COHORT_ALL],
        target_node_counts[COHORT_ALL],
        COHORT_ALL,
        args.min_node_support,
    )
    correct_rows, correct_budgets = summarize_pair_aggregates(
        cohort_aggregates[COHORT_CORRECT],
        target_node_counts[COHORT_CORRECT],
        COHORT_CORRECT,
        args.min_node_support,
    )
    visibility_rows: list[dict[str, object]] = []
    for cohort in (COHORT_ALL, COHORT_CORRECT):
        for visibility, aggregates in visibility_aggregates[cohort].items():
            rows, _ = summarize_pair_aggregates(
                aggregates,
                target_node_counts[cohort],
                cohort,
                args.min_node_support,
                normalization_scope=f"source_visibility:{visibility}",
            )
            for row in rows:
                row["source_visibility"] = visibility
            visibility_rows.extend(rows)

    _write_csv(
        output_dir / "occupation_pairs_all_test.csv", all_rows, SUMMARY_FIELDS
    )
    _write_csv(
        output_dir / "occupation_pairs_correct_only.csv",
        correct_rows,
        SUMMARY_FIELDS,
    )
    _write_csv(
        output_dir / "occupation_pairs_by_visibility.csv",
        visibility_rows,
        [*SUMMARY_FIELDS[:2], "source_visibility", *SUMMARY_FIELDS[2:]],
    )
    ranked = ranked_rows(all_rows)
    _write_csv(
        output_dir / "occupation_pairs_ranked.csv",
        ranked,
        [
            *SUMMARY_FIELDS,
            "global_dependency_rank_within_Ot",
            "conditional_lift_rank_within_R_Ot",
        ],
    )
    _write_csv(
        output_dir / "target_budgets.csv",
        [*all_budgets, *correct_budgets],
        BUDGET_FIELDS,
    )

    total_count = sum(layer_count)
    metrics = {
        "split": args.split,
        "test_nodes": len(divergences),
        "labeled_test_nodes": len(labels),
        "original_correct_test_nodes": correct_root_count,
        "original": _metric_block(labels, original_predictions),
        "masked": _metric_block(labels, masked_predictions),
        "prediction_agreement": (
            float(np.mean(np.equal(all_original_predictions, all_masked_predictions)))
            if all_original_predictions else None
        ),
        "mean_kl": float(np.mean(divergences)) if divergences else None,
        "hard_retention_rate": (
            sum(layer_hard) / total_count if total_count else None
        ),
        "mean_keep_probability": (
            sum(layer_probability) / total_count if total_count else None
        ),
        "layers": [
            {
                "layer": layer,
                "all_message_observations": layer_count[layer],
                "all_message_hard_retention_rate": (
                    layer_hard[layer] / layer_count[layer]
                    if layer_count[layer] else None
                ),
                "all_message_mean_keep_probability": (
                    layer_probability[layer] / layer_count[layer]
                    if layer_count[layer] else None
                ),
                "known_source_direct_candidate_messages": sum(
                    stats.candidate_message_n
                    for key, stats in cohort_aggregates[COHORT_ALL].items()
                    if key.layer == layer
                ),
                "known_source_direct_hard_retained_messages": sum(
                    stats.hard_retained_message_n
                    for key, stats in cohort_aggregates[COHORT_ALL].items()
                    if key.layer == layer
                ),
                "unknown_source_direct_candidate_messages": (
                    unknown_audit[layer].candidate_message_n
                ),
                "unknown_source_direct_hard_retained_messages": (
                    unknown_audit[layer].hard_retained_message_n
                ),
            }
            for layer in range(len(layer_count))
        ],
    }
    write_json(output_dir / "metrics.json", metrics)

    relation_perturbation = checkpoint.get("relation_perturbation") or {}
    shuffled_relations = bool(relation_perturbation.get("relation_type_shuffle", False))
    write_json(output_dir / "manifest.json", {
        "artifact": "graphmask_l1_direct_occupation_pair_report",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "probe": str(probe_path.resolve()),
        "probe_sha256": sha256_file(probe_path),
        "data": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "model_name": restored.model_name,
        "target_column": metadata["target_column"],
        "split": args.split,
        "fanouts": fanouts,
        "sampling_scope": (
            "full-neighborhood" if all(value == -1 for value in fanouts)
            else "fixed-sampled-neighborhood"
        ),
        "sampling_seed": report_seed,
        "root_batch_size": args.batch_size,
        "message_scope": "typed_messages_with_target_equal_to_prediction_root",
        "layer_policy": "report_separately_never_sum_as_unique_edges",
        "source_label_policy": (
            "main tables use true-labeled sources; unknown sources are retained only in raw audit"
        ),
        "normalization_denominator": (
            "known-source direct typed messages only; zero-message target occupations remain in target_budgets.csv"
        ),
        "source_visibility_policy": (
            "main tables combine known-source visibility groups for old-study comparability; "
            "occupation_pairs_by_visibility.csv normalizes each visibility stratum separately"
        ),
        "target_cohorts": {
            COHORT_ALL: "all true-labeled test nodes",
            COHORT_CORRECT: "test nodes whose frozen original model prediction equals true L1",
        },
        "minimum_candidate_node_support_for_ranking": args.min_node_support,
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_estimands": {
            "retained_share_of_Ot_budget": (
                "K_cell / sum_(source L1, relation) K for the same layer and target L1"
            ),
            "retention_lift_within_R_Ot": (
                "(K_cell / K_relation_target) / (N_cell / N_relation_target)"
            ),
        },
        "robustness_estimands": {
            "expected_retained_share_of_Ot_budget": (
                "expected non-zero mass cell / target-L1 expected non-zero mass"
            ),
            "expected_lift_within_R_Ot": (
                "expected retained share within relation-target divided by opportunity share"
            ),
        },
        "relation_type_semantics": (
            "shuffled_model_assignments" if shuffled_relations else "source_relation_names"
        ),
        "git_revision": git_revision(),
        "note": (
            "GraphMask describes frozen-model dependence on messages; it does not establish "
            "a causal social effect. For hidden validation/test sources, source L1 is a "
            "post-hoc true-label stratum rather than an occupation observed by the model."
        ),
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
