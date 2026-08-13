#!/usr/bin/env python3
"""Sweep conditional counterfactual importance across all observed L1-relation-L1 motifs.

Unlike ``relation-pair-ablation-report``, which re-forwards one requested
motif at a time, this command evaluates every observed direct motif in a
single base forward per batch.  It is exact for the project's frozen one-layer
RGAT configuration: after a typed edge group is removed, the remaining
across-relation attention weights are the original weights renormalised by
``1 - removed_attention_mass``.  The command recomputes the affected root's
RGAT aggregate, residual, GELU, LayerNorm, and classifier from that identity.

For every root-level ``(source L1, exact directed relation, target L1)``
group, the output compares its true-target margin drop with deletion of the
same number of other direct messages entering that root from visible sources
with the same L1.  It reports model reliance, not a causal social claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

from training.attention_common import (
    checkpoint_identity,
    checkpoint_paths,
    fanouts_for_checkpoint,
    git_revision,
    head_alpha,
    model_depth,
    replay_relation_perturbation,
    resolve_device,
    restore_rgat,
    sha256_file,
    source_visibility_codes,
    write_csv,
)
from training.attention_utils import attention_relation_ids
from training.message_contribution import relation_value_vectors
from training.relation_pair_ablation import _batches, _selected_root_ids
from training.train import batch_features, feature_inputs
from training.tie_taxonomy import (
    DEFAULT_TIE_TAXONOMY_PATH,
    TieTaxonomy,
    load_tie_taxonomy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared L1 graph_data.pt used by every checkpoint")
    parser.add_argument("--checkpoint", action="append", default=[], help="Exact one-layer RGAT checkpoint")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob of one-layer RGAT checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--forward-mode",
        choices=["full-graph", "full-neighborhood"],
        default="full-neighborhood",
        help="Full-neighborhood is safest; full-graph requires safely masked held-out roots.",
    )
    parser.add_argument("--num-neighbors", default="full", help="One fan-out for the one-layer checkpoint; full uses -1.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--control-draws", type=int, default=10)
    parser.add_argument("--max-roots", type=int, default=None, help="Optional deterministic split-root cap for a pilot.")
    parser.add_argument("--analysis-seed", type=int, default=20260811)
    parser.add_argument(
        "--min-summary-roots",
        type=int,
        default=10,
        help="Only emit a pair summary when it has this many eligible roots; root-level output is unaffected.",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-5,
        help="Maximum reconstruction error versus the base model logits; set negative to disable the safety check.",
    )
    parser.add_argument(
        "--tie-taxonomy",
        default=str(DEFAULT_TIE_TAXONOMY_PATH),
        help="Versioned inherited/acquired taxonomy JSON used to label all motif outputs",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _forward_with_attention(model, graph, feature_schema, occupation_unknown_ids, forward_mode: str):
    features = (
        feature_inputs(graph, feature_schema)
        if forward_mode == "full-graph"
        else batch_features(graph, feature_schema, occupation_unknown_ids)
    )
    return model(features, graph.edge_index, graph.edge_type, return_attention_weights=True)


def _validate_fast_counterfactual_configuration(model, checkpoint: Mapping[str, object]) -> None:
    if model_depth(checkpoint) != 1 or len(model.convs) != 1:
        raise ValueError(
            "relation-pair-sweep-report currently requires exactly one RGAT layer; "
            "only then is direct edge removal a complete local intervention."
        )
    conv = model.convs[0]
    if conv.attention_mechanism != "across-relation":
        raise ValueError("The fast sweep requires RGATConv(attention_mechanism='across-relation')")
    if conv.attention_mode != "additive-self-attention" or conv.mod is not None or conv.concat:
        raise ValueError(
            "The fast sweep requires additive-self-attention, mod=None, concat=False RGATConv as used by this project"
        )
    if model.training or conv.training:
        raise RuntimeError("The fast sweep requires eval() mode so attention dropout is inactive")


def _base_head_aggregates(model, layer_info: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct per-head message sums and verify their relation to RGAT output.

    Returned tensors correspond to the exact attention edges, not merely the
    input edge list.  The current RGAT setup does not add loops, but rejecting
    synthetic attention edges here prevents a future library change from
    silently invalidating the counterfactual identity.
    """
    attention_edge_index = layer_info["edge_index"]
    relation_ids = attention_relation_ids(layer_info)
    if bool(relation_ids.lt(0).any()):
        raise ValueError(
            "The fast all-pair sweep does not support RGAT synthetic attention edges; use the direct single-pair report"
        )
    alpha = head_alpha(layer_info["alpha"])
    input_state = layer_info["input_node_state"]
    values = relation_value_vectors(
        model.convs[0], input_state, attention_edge_index[0], relation_ids.long()
    )
    weighted_values = alpha.unsqueeze(-1) * values
    aggregate = torch.zeros(
        (input_state.size(0), values.size(1), values.size(2)),
        dtype=values.dtype,
        device=values.device,
    )
    aggregate.index_add_(0, attention_edge_index[1], weighted_values)
    return aggregate, alpha, weighted_values, relation_ids


def _logits_from_head_aggregate(model, input_state: torch.Tensor, head_aggregate: torch.Tensor) -> torch.Tensor:
    """Apply the one-layer RGAT post-aggregation computation exactly."""
    updated = head_aggregate.mean(dim=1)
    bias = model.convs[0].bias
    if bias is not None:
        updated = updated + bias
    hidden = model.norms[0](input_state + model.dropout(F.gelu(updated)))
    return model.classifier(hidden)


def _margin_for_label(logits: torch.Tensor, label_id: int) -> torch.Tensor:
    if logits.numel() == 0 or logits.size(-1) < 2:
        raise ValueError("Relation-pair sweep requires at least two target classes")
    target = logits[label_id]
    competitors = logits.clone()
    competitors[label_id] = float("-inf")
    return target - torch.logsumexp(competitors, dim=0)


def _counterfactual_root_logits(
    model,
    input_state: torch.Tensor,
    base_head_aggregate: torch.Tensor,
    root_index: int,
    removed_mass: torch.Tensor,
    removed_weighted_values: torch.Tensor,
) -> torch.Tensor:
    """Return exact root logits after removing a group of incoming messages."""
    remaining_mass = 1.0 - removed_mass
    numerator = base_head_aggregate[root_index] - removed_weighted_values
    # When every incoming edge in a head is removed, RGAT's additive aggregate
    # becomes zero (before its bias).  Otherwise softmax renormalisation gives
    # the exact quotient below.
    remaining = torch.where(
        remaining_mass.unsqueeze(-1).gt(torch.finfo(numerator.dtype).eps),
        numerator / remaining_mass.unsqueeze(-1),
        torch.zeros_like(numerator),
    )
    updated = remaining.mean(dim=0)
    bias = model.convs[0].bias
    if bias is not None:
        updated = updated + bias
    hidden = model.norms[0](input_state[root_index] + model.dropout(F.gelu(updated)))
    return model.classifier(hidden)


def _edge_groups(
    graph,
    local_roots: torch.Tensor,
    relation_ids: torch.Tensor,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
) -> Tuple[List[Tuple[int, int, int, torch.Tensor]], List[torch.Tensor], torch.Tensor]:
    """Return all observed root/source-L1/relation groups and their controls."""
    root_count = int(local_roots.numel())
    root_slot = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.edge_type.device)
    root_slot[local_roots] = torch.arange(root_count, device=graph.edge_type.device)
    edge_root_slot = root_slot[graph.edge_index[1]]
    enters_root = edge_root_slot.ge(0)
    sources = graph.edge_index[0]
    visibility = source_visibility_codes(graph, sources, feature_schema, occupation_unknown_ids)
    typed = relation_ids.ge(0)
    source_labels = graph.y[sources]
    root_labels = graph.y[local_roots]
    usable = enters_root & typed & source_labels.ge(0) & visibility.eq(0) & root_labels[edge_root_slot.clamp_min(0)].ge(0)
    usable_indices = usable.nonzero(as_tuple=False).view(-1)
    if not int(usable_indices.numel()):
        return [], [torch.empty(0, dtype=torch.long, device=graph.edge_type.device) for _ in range(root_count)], root_labels

    # The tuple key contains the root so aggregation removes every matching
    # edge for that one root while all cross-root interventions remain separate.
    relation_slots = int(relation_ids.max().item()) + 1
    class_slots = int(max(int(graph.y.max().item()) + 1, 1))
    slots = edge_root_slot[usable_indices]
    source_slots = source_labels[usable_indices]
    relation_slots_per_edge = relation_ids[usable_indices]
    encoded = ((slots * class_slots + source_slots) * relation_slots) + relation_slots_per_edge
    unique, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
    groups: List[Tuple[int, int, int, torch.Tensor]] = []
    for group_slot, value in enumerate(unique.tolist()):
        relation_id = value % relation_slots
        decoded = value // relation_slots
        source_label_id = decoded % class_slots
        root_index = decoded // class_slots
        group_edges = usable_indices[inverse.eq(group_slot)]
        groups.append((int(root_index), int(source_label_id), int(relation_id), group_edges))

    # These candidates retain target root, source L1 and source visibility but
    # exclude the selected relation.  Their required count is set per group.
    controls_by_root_source: List[torch.Tensor] = []
    for root_index in range(root_count):
        root_matches = enters_root & edge_root_slot.eq(root_index) & typed & visibility.eq(0) & source_labels.ge(0)
        controls_by_root_source.append(root_matches.nonzero(as_tuple=False).view(-1))
    return groups, controls_by_root_source, root_labels


def _draw_control_edges(
    candidates: torch.Tensor,
    relation_ids: torch.Tensor,
    excluded_relation_id: int,
    source_labels: torch.Tensor,
    source_label_id: int,
    required: int,
    generator: torch.Generator,
) -> torch.Tensor | None:
    available = candidates[
        relation_ids[candidates].ne(excluded_relation_id) & source_labels[candidates].eq(source_label_id)
    ]
    if int(available.numel()) < required:
        return None
    selected_positions = torch.randperm(int(available.numel()), generator=generator)[:required]
    return available.detach().cpu()[selected_positions].to(device=available.device)


def _finite_mean(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _summary_rows(
    records: Sequence[Mapping[str, object]],
    target_root_count: Mapping[int, int],
    min_roots: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            record["experiment"], record["seed"], record["checkpoint"], record["split"],
            record["source_l1_id"], record["source_l1"], record["relation_id"], record["relation"],
            record["tie_group"], record["target_l1_id"], record["target_l1"],
        )
        grouped[key].append(record)
    output = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < min_roots:
            continue
        matched = [row for row in rows if row["has_matched_control"]]
        base_target = [row for row in rows if row["base_predicts_target"]]
        output.append({
            "experiment": key[0],
            "seed": key[1],
            "checkpoint": key[2],
            "split": key[3],
            "source_l1_id": key[4],
            "source_l1": key[5],
            "relation_id": key[6],
            "relation": key[7],
            "tie_group": key[8],
            "target_l1_id": key[9],
            "target_l1": key[10],
            "motif_eligible_root_n": len(rows),
            "target_l1_root_n": int(target_root_count.get(int(key[9]), 0)),
            "coverage_of_target_l1": (
                len(rows) / target_root_count[int(key[9])] if target_root_count.get(int(key[9]), 0) else None
            ),
            "matched_control_root_n": len(matched),
            "mean_pair_edge_count": _finite_mean([float(row["pair_edge_count"]) for row in rows]),
            "mean_pair_margin_drop": _finite_mean([float(row["pair_margin_drop"]) for row in rows]),
            "median_pair_margin_drop": _finite_median([float(row["pair_margin_drop"]) for row in rows]),
            "base_predicts_target_root_n": len(base_target),
            "pair_flip_away_rate": _finite_mean([float(row["pair_flips_away_from_target"]) for row in base_target]),
            "mean_control_margin_drop": _finite_mean([float(row["control_mean_margin_drop"]) for row in matched]),
            "mean_pair_minus_control_margin_drop": _finite_mean([
                float(row["pair_minus_control_margin_drop"]) for row in matched
            ]),
            "median_pair_minus_control_margin_drop": _finite_median([
                float(row["pair_minus_control_margin_drop"]) for row in matched
            ]),
            "mean_control_flip_away_rate": _finite_mean([
                float(row["control_flip_away_rate"])
                for row in matched if row["control_flip_away_rate"] is not None
            ]),
        })
    return output


@torch.no_grad()
def collect_checkpoint_relation_pair_sweep(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    tie_taxonomy: TieTaxonomy,
    split: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str,
    control_draws: int,
    max_roots: int | None,
    analysis_seed: int,
    verify_atol: float,
) -> Tuple[List[Dict[str, object]], Dict[int, int], Dict[str, object]]:
    if control_draws <= 0:
        raise ValueError("--control-draws must be positive")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    perturbation = checkpoint.get("relation_perturbation")
    recorded_taxonomy = perturbation.get("tie_taxonomy") if isinstance(perturbation, Mapping) else None
    if recorded_taxonomy is not None and recorded_taxonomy.get("sha256") != tie_taxonomy.sha256:
        raise ValueError(
            f"Checkpoint tie taxonomy differs from --tie-taxonomy: {path}. "
            "Use the taxonomy recorded by the checkpoint or retrain the condition."
        )
    _validate_fast_counterfactual_configuration(model, checkpoint)
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    roots = _selected_root_ids(data, split, max_roots, analysis_seed)
    fanouts = fanouts_for_checkpoint(path, requested_fanouts, 1)
    id_to_relation = {int(value): key for key, value in metadata["relation_to_id"].items()}
    id_to_label = {int(value): key for key, value in metadata["label_to_id"].items()}
    experiment, seed = checkpoint_identity(path)
    records: List[Dict[str, object]] = []
    target_root_count: Dict[int, int] = defaultdict(int)
    control_generator = torch.Generator().manual_seed(int(analysis_seed))
    max_reconstruction_error = 0.0

    for graph, local_roots, global_node_ids in _batches(
        data, roots, split, feature_schema, metadata["occupation_unknown_ids"], fanouts,
        batch_size, num_workers, device, forward_mode,
    ):
        logits, explanation = _forward_with_attention(
            model, graph, feature_schema, metadata["occupation_unknown_ids"], forward_mode
        )
        layer_info = explanation["attention_layers"][0]
        if not torch.equal(layer_info["edge_index"], graph.edge_index):
            raise ValueError(
                "The fast all-pair sweep requires RGAT attention edges to preserve input edge order; "
                "use relation-pair-ablation-report for a direct re-forward instead."
            )
        base_head_aggregate, alpha, weighted_values, relation_ids = _base_head_aggregates(model, layer_info)
        reconstructed_logits = _logits_from_head_aggregate(model, layer_info["input_node_state"], base_head_aggregate)
        error = float((logits - reconstructed_logits).abs().max().item())
        max_reconstruction_error = max(max_reconstruction_error, error)
        if verify_atol >= 0 and error > verify_atol:
            raise RuntimeError(
                f"Fast counterfactual reconstruction differs from model logits by {error:.3e}, exceeding --verify-atol"
            )

        root_labels = graph.y[local_roots]
        for label_id in root_labels.detach().cpu().tolist():
            if label_id >= 0:
                target_root_count[int(label_id)] += 1
        groups, candidates_by_root, _ = _edge_groups(
            graph, local_roots, relation_ids, feature_schema, metadata["occupation_unknown_ids"]
        )
        if not groups:
            continue
        sources = layer_info["edge_index"][0]
        source_labels = graph.y[sources]
        root_ids_cpu = global_node_ids[local_roots].detach().cpu().tolist()
        base_predictions = logits[local_roots].argmax(dim=-1)

        for root_slot, source_label_id, relation_id, group_edges in groups:
            target_label_id = int(root_labels[root_slot].item())
            if target_label_id < 0:
                continue
            pair_mass = alpha[group_edges].sum(dim=0)
            pair_values = weighted_values[group_edges].sum(dim=0)
            pair_logits = _counterfactual_root_logits(
                model, layer_info["input_node_state"], base_head_aggregate, int(local_roots[root_slot].item()), pair_mass, pair_values
            )
            base_root_logits = logits[local_roots[root_slot]]
            base_margin = _margin_for_label(base_root_logits, target_label_id)
            pair_margin = _margin_for_label(pair_logits, target_label_id)
            control_margins: List[torch.Tensor] = []
            control_predictions: List[int] = []
            candidates = candidates_by_root[root_slot]
            for _ in range(control_draws):
                control_edges = _draw_control_edges(
                    candidates,
                    relation_ids,
                    relation_id,
                    source_labels,
                    source_label_id,
                    int(group_edges.numel()),
                    control_generator,
                )
                if control_edges is None:
                    break
                control_logits = _counterfactual_root_logits(
                    model,
                    layer_info["input_node_state"],
                    base_head_aggregate,
                    int(local_roots[root_slot].item()),
                    alpha[control_edges].sum(dim=0),
                    weighted_values[control_edges].sum(dim=0),
                )
                control_margins.append(_margin_for_label(control_logits, target_label_id))
                control_predictions.append(int(control_logits.argmax().item()))

            matched = len(control_margins) == control_draws
            control_margin_mean = torch.stack(control_margins).mean() if matched else None
            base_prediction = int(base_predictions[root_slot].item())
            pair_prediction = int(pair_logits.argmax().item())
            pair_drop = float((base_margin - pair_margin).item())
            control_drop = float((base_margin - control_margin_mean).item()) if matched else None
            base_is_target = base_prediction == target_label_id
            records.append({
                "experiment": experiment,
                "seed": seed,
                "checkpoint": str(path),
                "split": split,
                "forward_mode": forward_mode,
                "fanouts": ",".join(str(value) for value in fanouts),
                "analysis_seed": int(analysis_seed),
                "control_draws": int(control_draws),
                "root_index": int(root_ids_cpu[root_slot]),
                "source_l1_id": int(source_label_id),
                "source_l1": id_to_label[int(source_label_id)],
                "source_visibility": "visible_train",
                "relation_id": int(relation_id),
                "relation": id_to_relation[int(relation_id)],
                "tie_group": tie_taxonomy.group_for_base_relation(id_to_relation[int(relation_id)]),
                "target_l1_id": target_label_id,
                "target_l1": id_to_label[target_label_id],
                "pair_edge_count": int(group_edges.numel()),
                "matched_control_candidate_edge_count": int(
                    (candidates[
                        relation_ids[candidates].ne(relation_id) & source_labels[candidates].eq(source_label_id)
                    ]).numel()
                ),
                "has_matched_control": matched,
                "base_target_margin": float(base_margin.item()),
                "pair_removed_target_margin": float(pair_margin.item()),
                "pair_margin_drop": pair_drop,
                "base_prediction_l1_id": base_prediction,
                "pair_removed_prediction_l1_id": pair_prediction,
                "base_predicts_target": base_is_target,
                "pair_removed_predicts_target": pair_prediction == target_label_id,
                "pair_flips_away_from_target": base_is_target and pair_prediction != target_label_id,
                "control_mean_target_margin": float(control_margin_mean.item()) if matched else None,
                "control_mean_margin_drop": control_drop,
                "pair_minus_control_margin_drop": pair_drop - control_drop if control_drop is not None else None,
                "control_flip_away_rate": (
                    float(sum(prediction != target_label_id for prediction in control_predictions) / control_draws)
                    if matched and base_is_target else None
                ),
            })

    run_info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": 1,
        "fanouts": fanouts,
        "input_root_n": int(roots.numel()),
        "root_motif_group_n": len(records),
        "max_reconstruction_error": max_reconstruction_error,
        "tie_taxonomy_sha256": tie_taxonomy.sha256,
    }
    return records, dict(target_root_count), run_info


def _tie_group_summary_rows(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Pool root-level counterfactuals by inherited/acquired category per seed."""
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            record["experiment"], record["seed"], record["checkpoint"], record["split"],
            record["tie_group"],
        )
        grouped[key].append(record)
    output: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        matched = [row for row in rows if row["has_matched_control"]]
        base_target = [row for row in rows if row["base_predicts_target"]]
        output.append({
            "experiment": key[0],
            "seed": key[1],
            "checkpoint": key[2],
            "split": key[3],
            "tie_group": key[4],
            "motif_eligible_root_n": len(rows),
            "matched_control_root_n": len(matched),
            "mean_pair_edge_count": _finite_mean([float(row["pair_edge_count"]) for row in rows]),
            "mean_pair_margin_drop": _finite_mean([float(row["pair_margin_drop"]) for row in rows]),
            "median_pair_margin_drop": _finite_median([float(row["pair_margin_drop"]) for row in rows]),
            "base_predicts_target_root_n": len(base_target),
            "pair_flip_away_rate": _finite_mean([float(row["pair_flips_away_from_target"]) for row in base_target]),
            "mean_control_margin_drop": _finite_mean([float(row["control_mean_margin_drop"]) for row in matched]),
            "mean_pair_minus_control_margin_drop": _finite_mean([
                float(row["pair_minus_control_margin_drop"]) for row in matched
            ]),
            "median_pair_minus_control_margin_drop": _finite_median([
                float(row["pair_minus_control_margin_drop"]) for row in matched
            ]),
            "mean_control_flip_away_rate": _finite_mean([
                float(row["control_flip_away_rate"])
                for row in matched if row["control_flip_away_rate"] is not None
            ]),
        })
    return output


def _across_seed_summary(seed_summaries: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_summaries:
        key = (
            row["experiment"], row["source_l1_id"], row["source_l1"], row["relation_id"], row["relation"],
            row["tie_group"], row["target_l1_id"], row["target_l1"],
        )
        groups[key].append(row)
    output = []
    fields = ("mean_pair_margin_drop", "mean_control_margin_drop", "mean_pair_minus_control_margin_drop", "pair_flip_away_rate")
    for key, rows in sorted(groups.items()):
        record = {
            "experiment": key[0],
            "source_l1_id": key[1], "source_l1": key[2], "relation_id": key[3], "relation": key[4],
            "tie_group": key[5], "target_l1_id": key[6], "target_l1": key[7], "seed_n": len(rows),
            "motif_eligible_root_n": int(sum(int(row["motif_eligible_root_n"]) for row in rows)),
            "matched_control_root_n": int(sum(int(row["matched_control_root_n"]) for row in rows)),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            record[f"{field}_seed_mean"] = float(np.mean(values)) if values else None
            record[f"{field}_seed_std"] = float(np.std(values, ddof=0)) if values else None
        output.append(record)
    return output


def _across_seed_tie_group_summary(
    seed_summaries: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object], List[Mapping[str, object]]] = defaultdict(list)
    for row in seed_summaries:
        groups[(row["experiment"], row["tie_group"])].append(row)
    output: List[Dict[str, object]] = []
    fields = (
        "mean_pair_edge_count", "mean_pair_margin_drop", "mean_control_margin_drop",
        "mean_pair_minus_control_margin_drop", "pair_flip_away_rate",
    )
    for (experiment, tie_group), rows in sorted(groups.items()):
        record: Dict[str, object] = {
            "experiment": experiment,
            "tie_group": tie_group,
            "seed_n": len(rows),
            "motif_eligible_root_n": int(sum(int(row["motif_eligible_root_n"]) for row in rows)),
            "matched_control_root_n": int(sum(int(row["matched_control_root_n"]) for row in rows)),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            record[f"{field}_seed_mean"] = float(np.mean(values)) if values else None
            record[f"{field}_seed_std"] = float(np.std(values, ddof=0)) if values else None
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if args.min_summary_roots <= 0:
        raise ValueError("--min-summary-roots must be positive")
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    tie_taxonomy = load_tie_taxonomy(args.tie_taxonomy, base_metadata["relation_to_id"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, object]] = []
    seed_summaries: List[Dict[str, object]] = []
    tie_group_seed_summaries: List[Dict[str, object]] = []
    run_info: List[Dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] all observed relation-pair counterfactuals: {path}", flush=True)
        records, target_counts, info = collect_checkpoint_relation_pair_sweep(
            path, base_data, base_metadata, tie_taxonomy, args.split, args.num_neighbors, args.batch_size, args.num_workers,
            device, args.forward_mode, args.control_draws, args.max_roots, args.analysis_seed, args.verify_atol,
        )
        if not records:
            raise RuntimeError("No visible-source L1 relation motifs were found for the requested roots")
        all_records.extend(records)
        seed_summaries.extend(_summary_rows(records, target_counts, args.min_summary_roots))
        tie_group_seed_summaries.extend(_tie_group_summary_rows(records))
        run_info.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    across_seed = _across_seed_summary(seed_summaries)
    tie_group_across_seed = _across_seed_tie_group_summary(tie_group_seed_summaries)

    roots_path = output_dir / "relation_pair_sweep_roots_by_seed.csv.gz"
    by_seed_path = output_dir / "relation_pair_sweep_summary_by_seed.csv"
    across_seed_path = output_dir / "relation_pair_sweep_summary_across_seeds.csv"
    tie_group_by_seed_path = output_dir / "relation_pair_sweep_tie_group_summary_by_seed.csv"
    tie_group_across_seed_path = output_dir / "relation_pair_sweep_tie_group_summary_across_seeds.csv"
    root_fields = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "fanouts", "analysis_seed", "control_draws",
        "root_index", "source_l1_id", "source_l1", "source_visibility", "relation_id", "relation", "tie_group",
        "target_l1_id", "target_l1", "pair_edge_count", "matched_control_candidate_edge_count",
        "has_matched_control", "base_target_margin", "pair_removed_target_margin", "pair_margin_drop",
        "base_prediction_l1_id", "pair_removed_prediction_l1_id", "base_predicts_target",
        "pair_removed_predicts_target", "pair_flips_away_from_target", "control_mean_target_margin",
        "control_mean_margin_drop", "pair_minus_control_margin_drop", "control_flip_away_rate",
    ]
    write_csv(roots_path, all_records, root_fields)
    if seed_summaries:
        write_csv(by_seed_path, seed_summaries, list(seed_summaries[0].keys()))
    else:
        write_csv(by_seed_path, [], [])
    if across_seed:
        write_csv(across_seed_path, across_seed, list(across_seed[0].keys()))
    else:
        write_csv(across_seed_path, [], [])
    if tie_group_seed_summaries:
        write_csv(tie_group_by_seed_path, tie_group_seed_summaries, list(tie_group_seed_summaries[0].keys()))
    else:
        write_csv(tie_group_by_seed_path, [], [])
    if tie_group_across_seed:
        write_csv(tie_group_across_seed_path, tie_group_across_seed, list(tie_group_across_seed[0].keys()))
    else:
        write_csv(tie_group_across_seed_path, [], [])
    manifest_path = output_dir / "relation_pair_sweep_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(Path(args.data)),
        "split": args.split,
        "forward_mode": args.forward_mode,
        "control_draws": args.control_draws,
        "max_roots": args.max_roots,
        "analysis_seed": args.analysis_seed,
        "min_summary_roots": args.min_summary_roots,
        "verify_atol": args.verify_atol,
        "tie_taxonomy": tie_taxonomy.manifest(),
        "code_git_revision": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "definition": (
            "Every observed root-level (visible source L1, exact directed relation, true held-out target L1) "
            "message group is removed as a unit. pair_margin_drop is the exact fall in the fixed target-class logit "
            "margin after attention renormalisation and the layer's downstream residual/GELU/LayerNorm/classifier."
        ),
        "matched_control": (
            "For every root-level motif, each control draw deletes the same count of other direct typed messages "
            "entering that root from visible sources with the same L1. Positive pair_minus_control_margin_drop means "
            "the selected relation hurts the target margin more than the matched generic-neighbour deletion."
        ),
        "scope": (
            "The fast identity is checked against base logits and only applies to eval-mode, one-layer, additive "
            "across-relation RGAT with mod=None and concat=False. It removes directed messages, not reverse edges or "
            "underlying social facts; results are model reliance rather than social causality."
        ),
        "roots_path": str(roots_path),
        "summary_by_seed_path": str(by_seed_path),
        "summary_across_seeds_path": str(across_seed_path),
        "tie_group_summary_by_seed_path": str(tie_group_by_seed_path),
        "tie_group_summary_across_seeds_path": str(tie_group_across_seed_path),
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (
        roots_path, by_seed_path, across_seed_path, tie_group_by_seed_path,
        tie_group_across_seed_path, manifest_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
