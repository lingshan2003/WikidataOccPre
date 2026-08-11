#!/usr/bin/env python3
"""Measure a typed occupation-pair relation's local counterfactual importance.

This command holds a one-layer RGAT checkpoint fixed.  For every eligible
prediction root it removes all direct messages matching a user-specified
``(source L1, directed relation, target L1)`` motif, then recomputes the true
target-class logit margin.  Its primary contrast is a within-root control that
deletes the same number of *other* direct edges from visible sources with the
same L1.  Consequently the reported ``pair_minus_control_margin_drop`` asks
whether the selected directed relation matters more than a generic observed
neighbour of the same source occupation for the same target person.

Target L1 is used only after training to select and score held-out roots; the
target feature remains unknown during every forward pass.  This is a local
model-reliance counterfactual, not evidence that the social relation causes an
occupation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader

from training.attention_common import (
    checkpoint_identity,
    checkpoint_paths,
    fanouts_for_checkpoint,
    git_revision,
    model_depth,
    replay_relation_perturbation,
    resolve_device,
    restore_rgat,
    root_indices,
    sha256_file,
    source_visibility_codes,
    validate_full_graph_root_mask,
    write_csv,
)
from training.train import batch_features, feature_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared graph_data.pt used by every checkpoint")
    parser.add_argument("--checkpoint", action="append", default=[], help="Exact one-layer RGAT checkpoint")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob of one-layer RGAT checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-l1", required=True, help="Visible source L1 label, e.g. Leadership")
    parser.add_argument("--relation", required=True, help="Exact directed relation, e.g. child or child__rev")
    parser.add_argument("--target-l1", required=True, help="Held-out target L1 label, e.g. Leadership")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--forward-mode",
        choices=["full-graph", "full-neighborhood"],
        default="full-neighborhood",
        help="Full-neighborhood is safest; full-graph is allowed only when held-out root occupations are masked.",
    )
    parser.add_argument(
        "--num-neighbors",
        default="full",
        help="One fan-out for the checkpoint layer; full uses -1.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--control-draws",
        type=int,
        default=10,
        help="Number of matched non-selected-relation deletions per root (must be positive).",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=None,
        help="Optional deterministic sample cap before motif eligibility filtering; useful for a quick pilot.",
    )
    parser.add_argument("--analysis-seed", type=int, default=20260811)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def target_margin(logits: torch.Tensor, root_positions: torch.Tensor, target_label_id: int) -> torch.Tensor:
    """Return the fixed true-target logit margin for the requested roots."""
    root_logits = logits[root_positions]
    if root_logits.size(1) < 2:
        raise ValueError("Relation-pair ablation requires at least two target classes")
    target_scores = root_logits[:, target_label_id]
    competitors = root_logits.clone()
    competitors[:, target_label_id] = float("-inf")
    return target_scores - torch.logsumexp(competitors, dim=1)


def _edge_groups_for_roots(
    graph,
    local_roots: torch.Tensor,
    source_label_id: int,
    relation_id: int,
    target_label_id: int,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return motif drops and per-root matched-control edge candidates.

    A motif edge must enter a root with the requested *true* target label and
    originate at a source whose requested L1 is visible in model input.  The
    latter restriction makes the source-L1 condition about available evidence,
    not an after-the-fact label grouping.  Control candidates retain that same
    visible source L1 and root, but use a different real directed relation.
    """
    edge_count = int(graph.edge_type.numel())
    root_count = int(local_roots.numel())
    root_slot = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.edge_type.device)
    root_slot[local_roots] = torch.arange(root_count, device=graph.edge_type.device)
    edge_root_slot = root_slot[graph.edge_index[1]]
    enters_root = edge_root_slot >= 0
    source_indices = graph.edge_index[0]
    visibility = source_visibility_codes(
        graph, source_indices, feature_schema, occupation_unknown_ids
    )
    root_target_labels = graph.y[local_roots]
    edge_target_labels = torch.full(
        (edge_count,), -1, dtype=graph.y.dtype, device=graph.edge_type.device
    )
    edge_target_labels[enters_root] = root_target_labels[edge_root_slot[enters_root]]
    source_is_requested_l1 = graph.y[source_indices].eq(int(source_label_id))
    source_is_visible = visibility.eq(0)
    typed_edge = graph.edge_type.ge(0)
    pair_drop = (
        enters_root
        & typed_edge
        & edge_target_labels.eq(int(target_label_id))
        & source_is_requested_l1
        & source_is_visible
        & graph.edge_type.eq(int(relation_id))
    )
    pair_counts = torch.bincount(edge_root_slot[pair_drop], minlength=root_count)
    root_is_target = root_target_labels.eq(int(target_label_id))
    eligible = root_is_target & pair_counts.gt(0)
    control_candidate_mask = (
        enters_root
        & typed_edge
        & edge_target_labels.eq(int(target_label_id))
        & source_is_requested_l1
        & source_is_visible
        & graph.edge_type.ne(int(relation_id))
    )
    candidates: List[torch.Tensor] = []
    for root_index in range(root_count):
        candidates.append((control_candidate_mask & edge_root_slot.eq(root_index)).nonzero(as_tuple=False).view(-1))
    return pair_drop, candidates, eligible, pair_counts


def _draw_matched_control_mask(
    candidates: Sequence[torch.Tensor],
    pair_counts: torch.Tensor,
    eligible: torch.Tensor,
    edge_count: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the required count of control edges independently within each root."""
    drop_mask = torch.zeros(edge_count, dtype=torch.bool, device=pair_counts.device)
    paired = torch.zeros_like(eligible, dtype=torch.bool)
    for root_index, candidate_indices in enumerate(candidates):
        required = int(pair_counts[root_index].item())
        if not bool(eligible[root_index]) or int(candidate_indices.numel()) < required:
            continue
        order = torch.randperm(int(candidate_indices.numel()), generator=generator)[:required]
        selected = candidate_indices.detach().cpu()[order].to(device=drop_mask.device)
        drop_mask[selected] = True
        paired[root_index] = True
    return drop_mask, paired


def _drop_edges(graph, drop_mask: torch.Tensor):
    """Return a graph clone with exactly the selected directed input edges removed."""
    if drop_mask.dtype != torch.bool or drop_mask.numel() != graph.edge_type.numel():
        raise ValueError("drop_mask must be boolean with one entry per input edge")
    if not bool(drop_mask.any()):
        return graph
    result = graph.clone()
    keep = ~drop_mask
    result.edge_index = graph.edge_index[:, keep]
    result.edge_type = graph.edge_type[keep]
    return result


def _selected_root_ids(data, split: str, max_roots: int | None, analysis_seed: int) -> torch.Tensor:
    roots = root_indices(data, split)
    if max_roots is None or max_roots >= int(roots.numel()):
        return roots
    if max_roots <= 0:
        raise ValueError("--max-roots must be positive when supplied")
    generator = torch.Generator().manual_seed(int(analysis_seed))
    positions = torch.randperm(int(roots.numel()), generator=generator)[:max_roots]
    return roots[positions].sort().values


def _batches(
    data,
    roots: torch.Tensor,
    split: str,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
    fanouts: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str,
) -> Iterable[Tuple[object, torch.Tensor, torch.Tensor]]:
    if forward_mode == "full-graph":
        validate_full_graph_root_mask(data, roots, split, feature_schema, occupation_unknown_ids)
        graph = data.to(device)
        yield graph, roots.to(device), torch.arange(graph.num_nodes, dtype=torch.long, device=device)
        return
    if forward_mode != "full-neighborhood":
        raise ValueError(f"Unknown forward mode: {forward_mode}")
    loader = NeighborLoader(
        data,
        input_nodes=roots,
        num_neighbors=list(fanouts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    for batch in loader:
        batch = batch.to(device)
        local_roots = torch.arange(batch.batch_size, dtype=torch.long, device=device)
        yield batch, local_roots, batch.n_id


def _forward_logits(model, graph, feature_schema, occupation_unknown_ids, forward_mode: str) -> torch.Tensor:
    features = (
        feature_inputs(graph, feature_schema)
        if forward_mode == "full-graph"
        else batch_features(graph, feature_schema, occupation_unknown_ids)
    )
    return model(features, graph.edge_index, graph.edge_type)


def _finite_mean(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _finite_median(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


@torch.no_grad()
def collect_checkpoint_relation_pair_ablation(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    split: str,
    source_l1: str,
    relation: str,
    target_l1: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str,
    control_draws: int,
    max_roots: int | None,
    analysis_seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    """Collect root-level conditional motif and matched-control ablations."""
    if control_draws <= 0:
        raise ValueError("--control-draws must be positive")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    if model_depth(checkpoint) != 1 or len(model.convs) != 1:
        raise ValueError(
            "relation-pair-ablation-report currently requires an exactly one-layer RGAT checkpoint; "
            "this makes the intervention a complete direct-message counterfactual."
        )
    if source_l1 not in metadata["label_to_id"]:
        raise ValueError(f"Unknown --source-l1 {source_l1!r}; use a retained L1 label")
    if target_l1 not in metadata["label_to_id"]:
        raise ValueError(f"Unknown --target-l1 {target_l1!r}; use a retained L1 label")
    if relation not in metadata["relation_to_id"]:
        raise ValueError(
            f"Unknown --relation {relation!r}; choose an exact directed label, including '__rev' when appropriate"
        )

    source_label_id = int(metadata["label_to_id"][source_l1])
    target_label_id = int(metadata["label_to_id"][target_l1])
    relation_id = int(metadata["relation_to_id"][relation])
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    roots = _selected_root_ids(data, split, max_roots, analysis_seed)
    fanouts = fanouts_for_checkpoint(path, requested_fanouts, 1)
    experiment, seed = checkpoint_identity(path)
    records: List[Dict[str, object]] = []
    input_roots_seen = 0
    target_roots_seen = 0
    motif_roots_seen = 0
    control_generator = torch.Generator().manual_seed(int(analysis_seed))

    for graph, local_roots, global_node_ids in _batches(
        data, roots, split, feature_schema, metadata["occupation_unknown_ids"], fanouts,
        batch_size, num_workers, device, forward_mode,
    ):
        input_roots_seen += int(local_roots.numel())
        pair_drop, control_candidates, eligible, pair_counts = _edge_groups_for_roots(
            graph,
            local_roots,
            source_label_id,
            relation_id,
            target_label_id,
            feature_schema,
            metadata["occupation_unknown_ids"],
        )
        target_roots_seen += int(graph.y[local_roots].eq(target_label_id).sum().item())
        if not bool(eligible.any()):
            continue
        motif_roots_seen += int(eligible.sum().item())

        base_logits = _forward_logits(model, graph, feature_schema, metadata["occupation_unknown_ids"], forward_mode)
        pair_logits = _forward_logits(
            model,
            _drop_edges(graph, pair_drop),
            feature_schema,
            metadata["occupation_unknown_ids"],
            forward_mode,
        )
        base_margin = target_margin(base_logits, local_roots, target_label_id)
        pair_margin = target_margin(pair_logits, local_roots, target_label_id)
        base_predictions = base_logits[local_roots].argmax(dim=-1)
        pair_predictions = pair_logits[local_roots].argmax(dim=-1)

        control_margins: List[torch.Tensor] = []
        control_predictions: List[torch.Tensor] = []
        paired = torch.zeros_like(eligible, dtype=torch.bool)
        control_candidate_counts = torch.tensor(
            [int(candidates.numel()) for candidates in control_candidates],
            dtype=torch.long,
            device=device,
        )
        for _ in range(control_draws):
            control_drop, draw_paired = _draw_matched_control_mask(
                control_candidates,
                pair_counts,
                eligible,
                int(graph.edge_type.numel()),
                control_generator,
            )
            paired |= draw_paired
            if not bool(draw_paired.any()):
                continue
            control_logits = _forward_logits(
                model,
                _drop_edges(graph, control_drop),
                feature_schema,
                metadata["occupation_unknown_ids"],
                forward_mode,
            )
            control_margins.append(target_margin(control_logits, local_roots, target_label_id))
            control_predictions.append(control_logits[local_roots].argmax(dim=-1))
        control_margin_mean = torch.stack(control_margins).mean(dim=0) if control_margins else None
        control_prediction_stack = torch.stack(control_predictions) if control_predictions else None

        root_ids_cpu = global_node_ids[local_roots].detach().cpu().tolist()
        root_labels_cpu = graph.y[local_roots].detach().cpu().tolist()
        for slot, root_index in enumerate(root_ids_cpu):
            if not bool(eligible[slot]):
                continue
            matched = bool(paired[slot]) and control_margin_mean is not None
            pair_drop_margin = float((base_margin[slot] - pair_margin[slot]).item())
            control_drop_margin = (
                float((base_margin[slot] - control_margin_mean[slot]).item()) if matched else None
            )
            base_is_target = bool(base_predictions[slot].eq(target_label_id).item())
            pair_is_target = bool(pair_predictions[slot].eq(target_label_id).item())
            record = {
                "experiment": experiment,
                "seed": seed,
                "checkpoint": str(path),
                "split": split,
                "forward_mode": forward_mode,
                "fanouts": ",".join(str(value) for value in fanouts),
                "analysis_seed": int(analysis_seed),
                "control_draws_requested": int(control_draws),
                "root_index": int(root_index),
                "target_l1_id": int(root_labels_cpu[slot]),
                "target_l1": target_l1,
                "source_l1_id": source_label_id,
                "source_l1": source_l1,
                "source_visibility": "visible_train",
                "relation_id": relation_id,
                "relation": relation,
                "pair_edge_count": int(pair_counts[slot].item()),
                "matched_control_candidate_edge_count": int(control_candidate_counts[slot].item()),
                "has_matched_control": matched,
                "base_target_margin": float(base_margin[slot].item()),
                "pair_removed_target_margin": float(pair_margin[slot].item()),
                "pair_margin_drop": pair_drop_margin,
                "base_prediction_l1_id": int(base_predictions[slot].item()),
                "pair_removed_prediction_l1_id": int(pair_predictions[slot].item()),
                "base_predicts_target": base_is_target,
                "pair_removed_predicts_target": pair_is_target,
                "pair_flips_away_from_target": base_is_target and not pair_is_target,
                "control_mean_target_margin": (
                    float(control_margin_mean[slot].item()) if matched else None
                ),
                "control_mean_margin_drop": control_drop_margin,
                "pair_minus_control_margin_drop": (
                    pair_drop_margin - control_drop_margin if control_drop_margin is not None else None
                ),
                "control_flip_away_rate": (
                    float(
                        (base_predictions[slot].eq(target_label_id) & control_prediction_stack[:, slot].ne(target_label_id))
                        .float().mean().item()
                    ) if matched and control_prediction_stack is not None else None
                ),
            }
            records.append(record)

    matched_records = [record for record in records if record["has_matched_control"]]
    base_target_records = [record for record in records if record["base_predicts_target"]]
    summary = {
        "experiment": experiment,
        "seed": seed,
        "checkpoint": str(path),
        "split": split,
        "forward_mode": forward_mode,
        "source_l1": source_l1,
        "relation": relation,
        "target_l1": target_l1,
        "input_root_n": input_roots_seen,
        "target_l1_root_n": target_roots_seen,
        "motif_eligible_root_n": motif_roots_seen,
        "matched_control_root_n": len(matched_records),
        "base_predicts_target_root_n": len(base_target_records),
        "mean_pair_margin_drop": _finite_mean([float(record["pair_margin_drop"]) for record in records]),
        "median_pair_margin_drop": _finite_median([float(record["pair_margin_drop"]) for record in records]),
        "pair_flip_away_rate": _finite_mean([
            float(record["pair_flips_away_from_target"]) for record in base_target_records
        ]),
        "mean_control_margin_drop": _finite_mean([
            float(record["control_mean_margin_drop"]) for record in matched_records
        ]),
        "mean_pair_minus_control_margin_drop": _finite_mean([
            float(record["pair_minus_control_margin_drop"]) for record in matched_records
        ]),
        "median_pair_minus_control_margin_drop": _finite_median([
            float(record["pair_minus_control_margin_drop"]) for record in matched_records
        ]),
        "mean_control_flip_away_rate": _finite_mean([
            float(record["control_flip_away_rate"]) for record in matched_records
        ]),
        "control_draws_requested": int(control_draws),
        "max_roots": max_roots,
        "analysis_seed": int(analysis_seed),
    }
    run_info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": 1,
        "fanouts": fanouts,
        "input_root_n": input_roots_seen,
        "target_l1_root_n": target_roots_seen,
        "motif_eligible_root_n": motif_roots_seen,
        "matched_control_root_n": len(matched_records),
    }
    return records, summary, run_info


def _combined_summary(seed_summaries: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    numeric_fields = (
        "mean_pair_margin_drop",
        "mean_control_margin_drop",
        "mean_pair_minus_control_margin_drop",
        "pair_flip_away_rate",
        "mean_control_flip_away_rate",
    )
    combined: Dict[str, object] = {
        "checkpoint_seed_n": len(seed_summaries),
        "motif_eligible_root_n": int(sum(int(row["motif_eligible_root_n"]) for row in seed_summaries)),
        "matched_control_root_n": int(sum(int(row["matched_control_root_n"]) for row in seed_summaries)),
    }
    for field in numeric_fields:
        values = [float(row[field]) for row in seed_summaries if row.get(field) is not None]
        combined[f"{field}_seed_mean"] = float(np.mean(values)) if values else None
        combined[f"{field}_seed_std"] = float(np.std(values, ddof=0)) if values else None
    return combined


def main() -> None:
    args = parse_args()
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, object]] = []
    seed_summaries: List[Dict[str, object]] = []
    run_info: List[Dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] conditional relation-pair ablation: {path}", flush=True)
        records, summary, info = collect_checkpoint_relation_pair_ablation(
            path,
            base_data,
            base_metadata,
            args.split,
            args.source_l1,
            args.relation,
            args.target_l1,
            args.num_neighbors,
            args.batch_size,
            args.num_workers,
            device,
            args.forward_mode,
            args.control_draws,
            args.max_roots,
            args.analysis_seed,
        )
        if not records:
            raise RuntimeError(
                f"No eligible roots for {args.source_l1} --{args.relation}--> {args.target_l1}. "
                "Check directed relation semantics and source visibility."
            )
        all_records.extend(records)
        seed_summaries.append(summary)
        run_info.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    roots_path = output_dir / "relation_pair_ablation_roots_by_seed.csv.gz"
    summary_path = output_dir / "relation_pair_ablation_summary_by_seed.csv"
    combined_path = output_dir / "relation_pair_ablation_summary_across_seeds.json"
    preferred_root_fields = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "fanouts", "analysis_seed",
        "control_draws_requested", "root_index", "source_l1_id", "source_l1", "source_visibility",
        "relation_id", "relation", "target_l1_id", "target_l1", "pair_edge_count",
        "matched_control_candidate_edge_count", "has_matched_control", "base_target_margin",
        "pair_removed_target_margin", "pair_margin_drop", "base_prediction_l1_id",
        "pair_removed_prediction_l1_id", "base_predicts_target", "pair_removed_predicts_target",
        "pair_flips_away_from_target", "control_mean_target_margin", "control_mean_margin_drop",
        "pair_minus_control_margin_drop", "control_flip_away_rate",
    ]
    write_csv(roots_path, all_records, preferred_root_fields)
    write_csv(summary_path, seed_summaries, list(seed_summaries[0].keys()))
    combined = _combined_summary(seed_summaries)
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "relation_pair_ablation_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(Path(args.data)),
        "source_l1": args.source_l1,
        "relation": args.relation,
        "target_l1": args.target_l1,
        "split": args.split,
        "forward_mode": args.forward_mode,
        "control_draws": args.control_draws,
        "max_roots": args.max_roots,
        "analysis_seed": args.analysis_seed,
        "code_git_revision": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "definition": (
            "For every held-out root whose true target L1 equals target_l1, remove all direct incoming edges "
            "whose exact directed relation equals relation and whose source has the requested L1 visibly available "
            "to the model. pair_margin_drop is the fall in that root's fixed true-target logit margin."
        ),
        "matched_control": (
            "For each eligible root and each draw, remove the same number of other direct typed incoming edges "
            "from visible sources with the same source L1. pair_minus_control_margin_drop is positive when the "
            "requested relation's deletion hurts the true-target margin more than this within-root matched control."
        ),
        "scope": (
            "Exactly one RGAT layer is required, so deleting target-directed edges is a complete direct-message "
            "intervention. The generated reverse edge is not removed: this command estimates directed message "
            "importance, not deletion of an underlying undirected social fact. This is model reliance, not a social "
            "causal claim."
        ),
        "roots_path": str(roots_path),
        "summary_path": str(summary_path),
        "combined_summary_path": str(combined_path),
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (roots_path, summary_path, combined_path, manifest_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
