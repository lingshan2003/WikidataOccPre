#!/usr/bin/env python3
"""Export value-aware, root-level RGAT message-contribution magnitudes.

For the project's default ``RGATConv`` configuration, a typed edge ``j -> i``
contributes ``mean_h(alpha[j->i,h] * (h_j @ W_r)[h])`` to the pre-activation
message aggregate at node ``i``.  This command exports L2 magnitudes of sums of
those vectors, grouped per prediction root by exact directed relation, source
L1, and source-visibility state.  It is more value-aware than alpha alone, but
it intentionally stops before the residual, GELU, LayerNorm and classifier, so
it is a message-contribution proxy rather than a causal prediction attribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
    output_fields,
    sha256_file,
    source_visibility_codes,
    validate_full_graph_root_mask,
    write_csv,
)
from training.attention_utils import attention_relation_ids
from training.train import batch_features, feature_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared graph_data.pt used by all checkpoints")
    parser.add_argument("--checkpoint", action="append", default=[], help="Exact best_model.pt path; repeatable")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob of best_model.pt files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "labeled", "all"],
        default="test",
        help="Prediction roots; confirmation analysis should use test",
    )
    parser.add_argument(
        "--num-neighbors",
        default="full",
        help="Comma-separated fan-outs, auto, or full. full uses -1 at every RGAT layer.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--forward-mode",
        choices=["full-graph", "full-neighborhood"],
        default="full-neighborhood",
        help="Use full-graph only for safely masked validation/test roots; full-neighborhood replays complete fan-outs.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def head_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Return PyG alpha as [edge, head], including one-head checkpoints."""
    return alpha.unsqueeze(-1) if alpha.dim() == 1 else alpha


def _validate_value_message_configuration(conv) -> None:
    """Reject RGAT variants whose messages are not exactly alpha * W_r * h_j."""
    if conv.attention_mode != "additive-self-attention":
        raise ValueError(
            "message-contribution-report currently supports additive-self-attention RGATConv only"
        )
    if conv.concat:
        raise ValueError(
            "message-contribution-report currently supports concat=False, used by this project's RGAT"
        )
    if conv.mod is not None:
        raise ValueError(
            "message-contribution-report requires RGATConv(mod=None), because cardinality-preservation "
            "modes add terms beyond alpha * W_r * h_j"
        )


def relation_value_vectors(
    conv,
    node_state: torch.Tensor,
    source_index: torch.Tensor,
    relation_ids: torch.Tensor,
) -> torch.Tensor:
    """Return exact per-head value vectors ``(h_j @ W_r)`` as [edge, head, out].

    This follows the PyG 2.6 RGATConv ``message`` implementation for the
    additive, ``mod=None`` configuration.  Supporting bases and block weights
    keeps the exporter faithful if a compatible checkpoint uses either
    regularisation representation.
    """
    _validate_value_message_configuration(conv)
    source_state = node_state[source_index]
    if conv.num_bases is not None:
        weight = torch.matmul(conv.att, conv.basis.view(conv.num_bases, -1))
        weight = weight.view(conv.num_relations, conv.in_channels, conv.heads * conv.out_channels)
        edge_weight = weight.index_select(0, relation_ids)
        values = torch.bmm(source_state.unsqueeze(1), edge_weight).squeeze(1)
    elif conv.num_blocks is not None:
        edge_weight = conv.weight.index_select(0, relation_ids)
        source_blocks = source_state.view(-1, conv.num_blocks, edge_weight.size(2))
        values = torch.einsum("ebi,ebio->ebo", source_blocks, edge_weight).reshape(
            -1, conv.heads * conv.out_channels
        )
    else:
        edge_weight = conv.weight.index_select(0, relation_ids)
        values = torch.bmm(source_state.unsqueeze(1), edge_weight).squeeze(1)
    return values.view(-1, conv.heads, conv.out_channels)


def _batches(
    model,
    data,
    root_ids: torch.Tensor,
    split: str,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
    fanouts: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str,
):
    if forward_mode == "full-graph":
        graph = data.to(device)
        roots = root_ids.to(device)
        global_ids = torch.arange(graph.num_nodes, dtype=torch.long, device=device)
        logits, explanation = model(
            feature_inputs(graph, feature_schema), graph.edge_index, graph.edge_type, return_attention_weights=True
        )
        del logits
        yield graph, roots, global_ids, explanation
        return

    loader = NeighborLoader(
        data,
        input_nodes=root_ids,
        num_neighbors=list(fanouts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    for batch in loader:
        batch = batch.to(device)
        roots = torch.arange(batch.batch_size, dtype=torch.long, device=device)
        logits, explanation = model(
            batch_features(batch, feature_schema, occupation_unknown_ids),
            batch.edge_index,
            batch.edge_type,
            return_attention_weights=True,
        )
        del logits
        yield batch, roots, batch.n_id, explanation


@torch.no_grad()
def collect_checkpoint_contributions(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    split: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    num_layers = model_depth(checkpoint)
    if len(model.convs) != num_layers:
        raise ValueError(f"Checkpoint depth does not match its RGAT layers: {path}")
    for conv in model.convs:
        _validate_value_message_configuration(conv)

    fanouts = fanouts_for_checkpoint(path, requested_fanouts, num_layers)
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    root_ids = root_indices(data, split)
    if forward_mode == "full-graph":
        validate_full_graph_root_mask(data, root_ids, split, feature_schema, metadata["occupation_unknown_ids"])
    elif forward_mode != "full-neighborhood":
        raise ValueError(f"Unknown forward mode: {forward_mode}")

    id_to_relation = {int(index): relation for relation, index in metadata["relation_to_id"].items()}
    id_to_label = {int(index): label for label, index in metadata["label_to_id"].items()}
    relation_slots = max(id_to_relation) + 1
    class_count = int(metadata["num_classes"])
    fanout_description = "full-graph" if forward_mode == "full-graph" else ",".join(str(value) for value in fanouts)
    experiment, seed = checkpoint_identity(path)
    sparse_records: List[Dict[str, object]] = []
    roster_records: List[Dict[str, object]] = []
    roots_seen = 0
    target_l1_counts = torch.zeros(class_count, dtype=torch.long)
    synthetic_target_edges: Dict[int, int] = defaultdict(int)
    head_count = 0

    for batch, local_roots, global_ids, explanation in _batches(
        model,
        data,
        root_ids,
        split,
        feature_schema,
        metadata["occupation_unknown_ids"],
        fanouts,
        batch_size,
        num_workers,
        device,
        forward_mode,
    ):
        root_count = int(local_roots.numel())
        roots_seen += root_count
        root_labels = batch.y[local_roots]
        target_l1_counts += torch.bincount(root_labels[root_labels >= 0].detach().cpu(), minlength=class_count)
        root_slot = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
        root_slot[local_roots] = torch.arange(root_count, dtype=torch.long, device=device)
        global_root_ids = global_ids[local_roots].detach().cpu().tolist()
        root_label_ids = root_labels.detach().cpu().tolist()

        for layer_info in explanation["attention_layers"]:
            layer = int(layer_info["layer"]) + 1
            conv = model.convs[layer - 1]
            edge_index = layer_info["edge_index"]
            relation_ids = attention_relation_ids(layer_info)
            alpha = head_alpha(layer_info["alpha"])
            head_count = max(head_count, int(alpha.size(1)))
            target_slots = root_slot[edge_index[1]]
            prediction_target = target_slots >= 0
            typed = relation_ids >= 0
            usable = prediction_target & typed
            synthetic_target_edges[layer] += int((prediction_target & ~typed).sum().item())

            root_edge_slots = target_slots[usable]
            typed_counts = torch.bincount(root_edge_slots, minlength=root_count)
            total_counts = torch.bincount(target_slots[prediction_target], minlength=root_count)
            synthetic_counts = torch.bincount(target_slots[prediction_target & ~typed], minlength=root_count)
            output_dim = int(conv.out_channels)
            root_message_vector = torch.zeros((root_count, output_dim), dtype=alpha.dtype, device=device)
            root_absolute_l2_sum = torch.zeros(root_count, dtype=alpha.dtype, device=device)

            if int(usable.sum().item()):
                usable_sources = edge_index[0, usable]
                usable_relations = relation_ids[usable].long()
                values = relation_value_vectors(
                    conv,
                    layer_info["input_node_state"],
                    usable_sources,
                    usable_relations,
                )
                if values.size(1) != alpha.size(1):
                    raise RuntimeError("RGAT value head count does not match returned attention head count")
                per_head_contribution = alpha[usable].unsqueeze(-1) * values
                per_edge_contribution = per_head_contribution.mean(dim=1)
                per_edge_l2 = torch.linalg.vector_norm(per_edge_contribution, dim=-1)
                root_message_vector.index_add_(0, root_edge_slots, per_edge_contribution)
                root_absolute_l2_sum.index_add_(0, root_edge_slots, per_edge_l2)

                source_l1 = batch.y[usable_sources]
                source_slots = source_l1.clone().long()
                source_slots[source_slots < 0] = class_count
                visibility = source_visibility_codes(
                    batch, usable_sources, feature_schema, metadata["occupation_unknown_ids"]
                )
                encoded = (
                    (((root_edge_slots * relation_slots + usable_relations) * (class_count + 1) + source_slots) * 3)
                    + visibility
                )
                unique, inverse, group_counts = torch.unique(
                    encoded, sorted=True, return_inverse=True, return_counts=True
                )
                group_count = int(unique.numel())
                group_head_vector = torch.zeros(
                    (group_count, alpha.size(1), output_dim), dtype=alpha.dtype, device=device
                )
                group_head_vector.index_add_(0, inverse, per_head_contribution)
                group_vector = group_head_vector.mean(dim=1)
                group_l2 = torch.linalg.vector_norm(group_vector, dim=-1)
                group_absolute_l2_sum = torch.zeros(group_count, dtype=alpha.dtype, device=device)
                group_absolute_l2_sum.index_add_(0, inverse, per_edge_l2)
                group_head_l2 = torch.linalg.vector_norm(group_head_vector, dim=-1)

                decoded = unique // 3
                source_label_ids = decoded % (class_count + 1)
                decoded = decoded // (class_count + 1)
                group_relation_ids = decoded % relation_slots
                group_root_slots = decoded // relation_slots
                root_group_l2_sum = torch.zeros(root_count, dtype=alpha.dtype, device=device)
                root_group_l2_sum.index_add_(0, group_root_slots, group_l2)

                for row, edge_count, vector_l2, absolute_l2, per_head_l2, relation_id, source_label_id, visibility_id, slot in zip(
                    range(group_count),
                    group_counts.detach().cpu().tolist(),
                    group_l2.detach().cpu().tolist(),
                    group_absolute_l2_sum.detach().cpu().tolist(),
                    group_head_l2.detach().cpu().tolist(),
                    group_relation_ids.detach().cpu().tolist(),
                    source_label_ids.detach().cpu().tolist(),
                    (unique % 3).detach().cpu().tolist(),
                    group_root_slots.detach().cpu().tolist(),
                ):
                    del row
                    source_l1_id = -1 if source_label_id == class_count else int(source_label_id)
                    denominator = float(root_group_l2_sum[slot].item())
                    record = {
                        "experiment": experiment,
                        "seed": seed,
                        "checkpoint": str(path),
                        "split": split,
                        "forward_mode": forward_mode,
                        "num_layers": num_layers,
                        "fanouts": fanout_description,
                        "message_passing_layer": layer,
                        "root_index": int(global_root_ids[slot]),
                        "target_l1_id": int(root_label_ids[slot]),
                        "target_l1": id_to_label.get(int(root_label_ids[slot]), "__UNLABELED__"),
                        "relation_id": int(relation_id),
                        "relation": id_to_relation[int(relation_id)],
                        "source_l1_id": source_l1_id,
                        "source_l1": id_to_label.get(source_l1_id, "__UNLABELED__"),
                        "source_visibility": (
                            "visible_train" if visibility_id == 0 else
                            "hidden_validation_or_test" if visibility_id == 1 else "missing_or_unknown"
                        ),
                        "candidate_edge_count": int(edge_count),
                        "total_incoming_attention_edges": int(total_counts[slot].item()),
                        "message_contribution_l2": float(vector_l2),
                        "absolute_message_l2_sum": float(absolute_l2),
                        "message_contribution_l2_share": float(vector_l2 / denominator) if denominator else 0.0,
                    }
                    for head, value in enumerate(per_head_l2):
                        record[f"message_contribution_l2_head_{head}"] = float(value)
                    sparse_records.append(record)

            typed_message_l2 = torch.linalg.vector_norm(root_message_vector, dim=-1)
            for slot, (root_index, target_l1_id) in enumerate(zip(global_root_ids, root_label_ids)):
                roster_records.append({
                    "experiment": experiment,
                    "seed": seed,
                    "checkpoint": str(path),
                    "split": split,
                    "forward_mode": forward_mode,
                    "num_layers": num_layers,
                    "fanouts": fanout_description,
                    "message_passing_layer": layer,
                    "root_index": int(root_index),
                    "target_l1_id": int(target_l1_id),
                    "target_l1": id_to_label.get(int(target_l1_id), "__UNLABELED__"),
                    "total_incoming_attention_edges": int(total_counts[slot].item()),
                    "typed_incoming_attention_edges": int(typed_counts[slot].item()),
                    "synthetic_self_loop_edges": int(synthetic_counts[slot].item()),
                    "typed_message_vector_l2": float(typed_message_l2[slot].item()),
                    "typed_absolute_message_l2_sum": float(root_absolute_l2_sum[slot].item()),
                })

    run_info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": num_layers,
        "fanouts": fanouts,
        "forward_mode": forward_mode,
        "prediction_roots": roots_seen,
        "prediction_roots_by_l1": {
            id_to_label[label_id]: int(count) for label_id, count in enumerate(target_l1_counts.tolist())
        },
        "synthetic_self_loop_target_edges": {str(layer): count for layer, count in synthetic_target_edges.items()},
        "attention_head_count": head_count,
    }
    return sparse_records, roster_records, run_info


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_sparse: List[Dict[str, object]] = []
    all_roster: List[Dict[str, object]] = []
    run_info = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] collecting value-aware messages: {path}", flush=True)
        sparse_rows, roster_rows, info = collect_checkpoint_contributions(
            path,
            base_data,
            base_metadata,
            args.split,
            args.num_neighbors,
            args.batch_size,
            args.num_workers,
            device,
            args.forward_mode,
        )
        if not roster_rows:
            raise RuntimeError(f"No prediction roots were collected from {path}")
        all_sparse.extend(sparse_rows)
        all_roster.extend(roster_rows)
        run_info.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    roster_path = output_dir / "root_message_contribution_roster_by_seed.csv.gz"
    sparse_path = output_dir / "root_message_contribution_sparse_by_seed.csv.gz"
    roster_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1",
        "total_incoming_attention_edges", "typed_incoming_attention_edges", "synthetic_self_loop_edges",
        "typed_message_vector_l2", "typed_absolute_message_l2_sum",
    ]
    sparse_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1", "relation_id", "relation",
        "source_l1_id", "source_l1", "source_visibility", "candidate_edge_count",
        "total_incoming_attention_edges", "message_contribution_l2", "absolute_message_l2_sum",
        "message_contribution_l2_share",
    ]
    write_csv(roster_path, all_roster, output_fields(all_roster, roster_preferred))
    write_csv(sparse_path, all_sparse, output_fields(all_sparse, sparse_preferred))
    manifest_path = output_dir / "message_contribution_manifest.json"
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
            "For each typed edge j->i, exact default-RGATConv pre-aggregation contribution vector is "
            "c[j->i] = mean_h(alpha[j->i,h] * ((h_j @ W_relation)[h])). "
            "message_contribution_l2 is ||sum_{edge in root/group} c||_2, so vector directions may cancel. "
            "absolute_message_l2_sum is sum_edge ||c||_2 and does not cancel. "
            "message_contribution_l2_share divides the group L2 norm by the sum of all group L2 norms at the root."
        ),
        "scope": (
            "Exact only through RGATConv message aggregation for additive-self-attention, concat=False, mod=None. "
            "The relation-independent RGAT bias, residual path, GELU, LayerNorm and classifier are deliberately "
            "excluded; these scores are value-aware message magnitudes, not final-logit causal attributions."
        ),
        "source_visibility": {
            "visible_train": "Retained-L1 training source whose occupation_level1 feature is visible to the checkpoint.",
            "hidden_validation_or_test": "Retained-L1 source whose occupation_level1 is unknown to the checkpoint.",
            "missing_or_unknown": "Source without a retained true L1 label.",
        },
        "sparse_path": str(sparse_path),
        "roster_path": str(roster_path),
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (roster_path, sparse_path, manifest_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
