#!/usr/bin/env python3
"""Export raw two-hop RGAT attention-path scores for prediction roots.

For a two-layer RGAT, this command reports the transparent attention-only
approximation ``R = mean(A2_heads) @ mean(A1_heads)`` for typed paths
``k --r1--> j --r2--> i``.  It is not an exact numerical attribution: RGAT
value transforms, residual paths, GELU and LayerNorm remain outside the
reported product.
"""

import argparse
import copy
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader

from training.attention_report import (
    _head_alpha,
    checkpoint_identity,
    checkpoint_paths,
    fanouts_for_checkpoint,
    git_revision,
    model_depth,
    prediction_nodes,
    replay_relation_perturbation,
    resolve_device,
    restore_rgat,
    root_indices,
    root_output_fields,
    sha256_file,
    source_visibility_codes,
    validate_full_graph_root_mask,
    write_csv,
)
from training.attention_utils import attention_relation_ids
from training.train import batch_features, feature_inputs


VISIBILITY_NAMES = ("visible_train", "hidden_validation_or_test", "missing_or_unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared L1 graph_data.pt used by every checkpoint")
    parser.add_argument("--checkpoint", action="append", default=[], help="Exact two-layer RGAT checkpoint")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob of two-layer RGAT checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--forward-mode", choices=["full-graph", "full-neighborhood"], default="full-graph",
        help="Use one full-graph eval forward, or complete (-1,-1) receptive-field batches as a memory fallback.",
    )
    parser.add_argument(
        "--num-neighbors", default="full",
        help="Fallback fan-outs for full-neighborhood mode; full uses -1 for both layers.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _source_label(source_label_id: int, id_to_label: Mapping[int, str]) -> str:
    return id_to_label.get(source_label_id, "__UNLABELED__")


def _append_path_block(
    aggregate: Dict[Tuple[int, int, int, int, int], List[float]],
    root_totals: Dict[int, List[float]],
    root_slots: torch.Tensor,
    relation1: torch.Tensor,
    relation2: torch.Tensor,
    source_l1: torch.Tensor,
    source_visibility: torch.Tensor,
    score1: torch.Tensor,
    score2: torch.Tensor,
    class_count: int,
    relation_slots: int,
    path_multiplicity: torch.Tensor | None = None,
) -> None:
    """Aggregate a path block on device, then update compact Python totals.

    Keys are ``(root slot, r1, r2, source L1 slot, visibility slot)``.  The
    Python dictionary intentionally stores only grouped root statistics, never
    one record per raw ``k-j-i`` path.
    """
    if not int(root_slots.numel()):
        return
    source_slots = source_l1.clone().long()
    source_slots[source_slots < 0] = class_count
    encoded = (
        ((((root_slots.long() * relation_slots + relation1.long()) * relation_slots + relation2.long())
          * (class_count + 1) + source_slots) * len(VISIBILITY_NAMES))
        + source_visibility.long()
    )
    unique, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
    if path_multiplicity is None:
        path_multiplicity = torch.ones_like(inverse, dtype=torch.long)
    path_count = torch.zeros(unique.numel(), dtype=torch.long, device=score1.device)
    path_count.scatter_add_(0, inverse, path_multiplicity.long())
    mass = torch.zeros(unique.numel(), dtype=torch.float64, device=score1.device)
    mass.scatter_add_(0, inverse, (score1 * score2).to(dtype=torch.float64))
    for value, count, score in zip(unique.tolist(), path_count.tolist(), mass.tolist()):
        visibility = value % len(VISIBILITY_NAMES)
        decoded = value // len(VISIBILITY_NAMES)
        source_slot = decoded % (class_count + 1)
        decoded //= class_count + 1
        r2 = decoded % relation_slots
        decoded //= relation_slots
        r1 = decoded % relation_slots
        root_slot = decoded // relation_slots
        aggregate[(root_slot, r1, r2, source_slot, visibility)][0] += int(count)
        aggregate[(root_slot, r1, r2, source_slot, visibility)][1] += float(score)
        root_totals[root_slot][0] += int(count)
        root_totals[root_slot][1] += float(score)


def _rollout_for_batch(
    batch,
    local_roots: torch.Tensor,
    explanation: Mapping[str, object],
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
    class_count: int,
    relation_slots: int,
) -> Tuple[Dict[Tuple[int, int, int, int, int], List[float]], Dict[int, List[float]], Dict[str, int]]:
    """Return grouped typed two-hop path statistics for one forward pass."""
    layers = explanation["attention_layers"]
    if len(layers) != 2:
        raise ValueError("Two-hop rollout requires exactly two RGAT message-passing layers")
    layer1, layer2 = layers
    edge1, edge2 = layer1["edge_index"], layer2["edge_index"]
    relation1, relation2 = attention_relation_ids(layer1), attention_relation_ids(layer2)
    score1 = _head_alpha(layer1["alpha"]).mean(dim=-1)
    score2 = _head_alpha(layer2["alpha"]).mean(dim=-1)

    root_slot = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=edge1.device)
    root_slot[local_roots] = torch.arange(local_roots.numel(), device=edge1.device)
    layer2_root_slot = root_slot[edge2[1]]
    usable2 = (layer2_root_slot >= 0) & (relation2 >= 0)
    usable1 = relation1 >= 0
    aggregate: Dict[Tuple[int, int, int, int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    root_totals: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0])
    diagnostics = {
        "typed_layer1_edges": int(usable1.sum().item()),
        "typed_layer2_root_edges": int(usable2.sum().item()),
        "synthetic_layer1_edges": int((relation1 < 0).sum().item()),
        "synthetic_layer2_root_edges": int(((layer2_root_slot >= 0) & (relation2 < 0)).sum().item()),
    }
    if not bool(usable1.any()) or not bool(usable2.any()):
        return aggregate, root_totals, diagnostics

    # An exact raw edge-path expansion can be quadratic around high-degree
    # intermediates.  Instead, within each intermediate we first aggregate
    # Layer-1 edges by (r1, source L1, visibility) and Layer-2 edges by
    # (root, r2).  Products of those sums equal the sum over every raw path,
    # while products of their counts retain the candidate-path opportunity.
    sorted_targets, sorted_indices = torch.sort(edge1[1])
    usable2_indices = usable2.nonzero(as_tuple=False).view(-1)
    sorted_sources2, source_order2 = torch.sort(edge2[0, usable2_indices])
    sorted_indices2 = usable2_indices[source_order2]
    intermediates = torch.unique(sorted_sources2, sorted=True)
    source_visibility_all = source_visibility_codes(
        batch, edge1[0], feature_schema, occupation_unknown_ids
    )
    for intermediate in intermediates.tolist():
        value = torch.tensor(intermediate, device=edge1.device)
        left1 = int(torch.searchsorted(sorted_targets, value))
        right1 = int(torch.searchsorted(sorted_targets, value, right=True))
        left2 = int(torch.searchsorted(sorted_sources2, value))
        right2 = int(torch.searchsorted(sorted_sources2, value, right=True))
        if right1 <= left1 or right2 <= left2:
            continue
        incoming1 = sorted_indices[left1:right1]
        incoming1 = incoming1[relation1[incoming1] >= 0]
        outgoing2 = sorted_indices2[left2:right2]
        if not int(incoming1.numel()) or not int(outgoing2.numel()):
            continue

        source_slots = batch.y[edge1[0, incoming1]].clone().long()
        source_slots[source_slots < 0] = class_count
        encoded1 = (
            ((relation1[incoming1].long() * (class_count + 1) + source_slots) * len(VISIBILITY_NAMES))
            + source_visibility_all[incoming1].long()
        )
        unique1, inverse1 = torch.unique(encoded1, sorted=True, return_inverse=True)
        count1 = torch.bincount(inverse1, minlength=unique1.numel())
        mass1 = torch.zeros(unique1.numel(), dtype=score1.dtype, device=edge1.device)
        mass1.scatter_add_(0, inverse1, score1[incoming1])
        visibility1 = unique1 % len(VISIBILITY_NAMES)
        decoded1 = unique1 // len(VISIBILITY_NAMES)
        source1 = decoded1 % (class_count + 1)
        relation1_group = decoded1 // (class_count + 1)

        encoded2 = layer2_root_slot[outgoing2].long() * relation_slots + relation2[outgoing2].long()
        unique2, inverse2 = torch.unique(encoded2, sorted=True, return_inverse=True)
        count2 = torch.bincount(inverse2, minlength=unique2.numel())
        mass2 = torch.zeros(unique2.numel(), dtype=score2.dtype, device=edge1.device)
        mass2.scatter_add_(0, inverse2, score2[outgoing2])
        root2 = unique2 // relation_slots
        relation2_group = unique2 % relation_slots

        l2 = torch.arange(unique2.numel(), device=edge1.device).repeat_interleave(unique1.numel())
        l1 = torch.arange(unique1.numel(), device=edge1.device).repeat(unique2.numel())
        _append_path_block(
            aggregate,
            root_totals,
            root2[l2],
            relation1_group[l1],
            relation2_group[l2],
            source1[l1],
            visibility1[l1],
            mass1[l1],
            mass2[l2],
            class_count,
            relation_slots,
            count1[l1] * count2[l2],
        )
    return aggregate, root_totals, diagnostics


def _write_rollout_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[Tuple[object, ...], Dict[str, float]] = {}
    key_fields = (
        "experiment", "seed", "checkpoint", "split", "forward_mode", "r1_id", "r1", "r2_id", "r2",
        "source_l1_id", "source_l1", "source_visibility", "target_l1_id", "target_l1",
    )
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        summary = grouped.setdefault(key, {"path_count": 0.0, "rollout_mass": 0.0, "roots_with_paths": 0.0})
        summary["path_count"] += float(row["candidate_path_count"])
        summary["rollout_mass"] += float(row["rollout_mass"])
        summary["roots_with_paths"] += 1.0
    output = []
    for key, values in sorted(grouped.items()):
        output.append({
            **dict(zip(key_fields, key)),
            "path_count": int(values["path_count"]),
            "rollout_mass": values["rollout_mass"],
            "roots_with_paths": int(values["roots_with_paths"]),
        })
    write_csv(path, output, [*key_fields, "path_count", "rollout_mass", "roots_with_paths"])


@torch.no_grad()
def collect_checkpoint_rollout(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    split: str,
    forward_mode: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    if model_depth(checkpoint) != 2 or len(model.convs) != 2:
        raise ValueError(f"Two-hop rollout accepts exactly two-layer RGAT checkpoints: {path}")
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    root_ids = root_indices(data, split)
    if forward_mode == "full-graph":
        validate_full_graph_root_mask(data, root_ids, split, feature_schema, metadata["occupation_unknown_ids"])
    elif forward_mode != "full-neighborhood":
        raise ValueError(f"Unknown forward mode: {forward_mode}")
    fanouts = fanouts_for_checkpoint(path, requested_fanouts, 2)
    id_to_relation = {int(index): relation for relation, index in metadata["relation_to_id"].items()}
    id_to_label = {int(index): label for label, index in metadata["label_to_id"].items()}
    experiment, seed = checkpoint_identity(path)
    fanout_description = "full-graph" if forward_mode == "full-graph" else ",".join(str(value) for value in fanouts)
    rows: List[Dict[str, object]] = []
    roster: List[Dict[str, object]] = []
    diagnostics: Dict[str, int] = defaultdict(int)
    roots_seen = 0

    def batches():
        if forward_mode == "full-graph":
            graph = data.to(device)
            roots = root_ids.to(device)
            logits, explanation = model(
                feature_inputs(graph, feature_schema), graph.edge_index, graph.edge_type, return_attention_weights=True
            )
            del logits
            yield graph, roots, torch.arange(graph.num_nodes, device=device), explanation
            return
        loader = NeighborLoader(
            data, input_nodes=prediction_nodes(data, split), num_neighbors=fanouts, batch_size=batch_size,
            shuffle=False, num_workers=max(0, num_workers), persistent_workers=num_workers > 0,
            pin_memory=device.type == "cuda",
        )
        for batch in loader:
            batch = batch.to(device)
            roots = torch.arange(batch.batch_size, dtype=torch.long, device=device)
            logits, explanation = model(
                batch_features(batch, feature_schema, metadata["occupation_unknown_ids"]),
                batch.edge_index, batch.edge_type, return_attention_weights=True,
            )
            del logits
            yield batch, roots, batch.n_id, explanation

    for batch, local_roots, global_ids, explanation in batches():
        aggregate, totals, batch_diagnostics = _rollout_for_batch(
            batch, local_roots, explanation, feature_schema, metadata["occupation_unknown_ids"],
            int(metadata["num_classes"]), max(int(value) for value in metadata["relation_to_id"].values()) + 1,
        )
        for key, value in batch_diagnostics.items():
            diagnostics[key] += int(value)
        global_root_ids = global_ids[local_roots].detach().cpu().tolist()
        root_labels = batch.y[local_roots].detach().cpu().tolist()
        roots_seen += len(global_root_ids)
        for slot, (root_index, target_l1_id) in enumerate(zip(global_root_ids, root_labels)):
            path_count, rollout_mass = totals.get(slot, [0.0, 0.0])
            roster.append({
                "experiment": experiment,
                "seed": seed,
                "checkpoint": str(path),
                "split": split,
                "forward_mode": forward_mode,
                "fanouts": fanout_description,
                "root_index": int(root_index),
                "target_l1_id": int(target_l1_id),
                "target_l1": id_to_label.get(int(target_l1_id), "__UNLABELED__"),
                "total_typed_path_count": int(path_count),
                "typed_rollout_mass": float(rollout_mass),
            })
        for (slot, r1, r2, source_slot, visibility), (path_count, rollout_mass) in aggregate.items():
            total_paths = totals[slot][0]
            source_l1_id = -1 if source_slot == int(metadata["num_classes"]) else int(source_slot)
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "checkpoint": str(path),
                "split": split,
                "forward_mode": forward_mode,
                "fanouts": fanout_description,
                "root_index": int(global_root_ids[slot]),
                "target_l1_id": int(root_labels[slot]),
                "target_l1": id_to_label.get(int(root_labels[slot]), "__UNLABELED__"),
                "r1_id": int(r1),
                "r1": id_to_relation[int(r1)],
                "r2_id": int(r2),
                "r2": id_to_relation[int(r2)],
                "source_l1_id": source_l1_id,
                "source_l1": _source_label(source_l1_id, id_to_label),
                "source_visibility": VISIBILITY_NAMES[visibility],
                "candidate_path_count": int(path_count),
                "total_typed_path_count": int(total_paths),
                "rollout_mass": float(rollout_mass),
                "opportunity": float(path_count / total_paths) if total_paths else 0.0,
                "rollout_mass_minus_opportunity": (
                    float(rollout_mass - path_count / total_paths) if total_paths else float(rollout_mass)
                ),
            })
    info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "split": split,
        "forward_mode": forward_mode,
        "fanouts": fanouts,
        "prediction_roots": roots_seen,
        "diagnostics": dict(diagnostics),
    }
    return rows, roster, info


def main() -> None:
    args = parse_args()
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    if base_metadata.get("target_column") != "occupation_level1":
        raise ValueError("Two-hop rollout is currently defined for an L1 target artifact")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    all_roster: List[Dict[str, object]] = []
    runs = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] collecting two-hop rollout: {path}", flush=True)
        rows, roster, info = collect_checkpoint_rollout(
            path, base_data, base_metadata, args.split, args.forward_mode, args.num_neighbors,
            args.batch_size, args.num_workers, device,
        )
        all_rows.extend(rows)
        all_roster.extend(roster)
        runs.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all_roster:
        raise RuntimeError("No prediction roots were collected")
    sparse_path = output_dir / "root_two_hop_rollout_sparse_by_seed.csv.gz"
    roster_path = output_dir / "root_two_hop_rollout_roster_by_seed.csv.gz"
    summary_path = output_dir / "two_hop_rollout_summary_by_seed.csv"
    row_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "fanouts", "root_index",
        "target_l1_id", "target_l1", "r1_id", "r1", "r2_id", "r2", "source_l1_id", "source_l1",
        "source_visibility", "candidate_path_count", "total_typed_path_count", "rollout_mass", "opportunity",
        "rollout_mass_minus_opportunity",
    ]
    roster_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "fanouts", "root_index",
        "target_l1_id", "target_l1", "total_typed_path_count", "typed_rollout_mass",
    ]
    write_csv(sparse_path, all_rows, root_output_fields(all_rows, row_preferred))
    write_csv(roster_path, all_roster, root_output_fields(all_roster, roster_preferred))
    _write_rollout_summary(summary_path, all_rows)
    manifest_path = output_dir / "attention_rollout_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(Path(args.data)),
        "split": args.split,
        "forward_mode": args.forward_mode,
        "code_git_revision": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "definition": (
            "Raw typed two-hop attention-path score R[i,k] = sum_j mean_h(alpha2[j->i]) * "
            "mean_h(alpha1[k->j]). Relation-specific values retain ordered exact directed pairs (r1, r2). "
            "Synthetic self-loops and all non-attention message components are excluded."
        ),
        "sparse_path": str(sparse_path),
        "roster_path": str(roster_path),
        "summary_path": str(summary_path),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (sparse_path, roster_path, summary_path, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
