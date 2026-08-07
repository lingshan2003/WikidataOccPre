#!/usr/bin/env python3
"""Export raw prediction-node RGAT attention masses without summarising them.

This command preserves the former ``attention-report --root-level-output``
artifacts.  It deliberately stops at one row per prediction target and sparse
``(source L1, relation, target L1)`` group; population-level node estimators
belong in a later, explicitly defined analysis step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader

from training.attention_common import (
    checkpoint_identity,
    checkpoint_paths,
    fanouts_for_checkpoint,
    git_revision,
    head_alpha,
    model_depth,
    output_fields,
    prediction_nodes,
    replay_relation_perturbation,
    resolve_device,
    restore_rgat,
    root_indices,
    sha256_file,
    source_visibility_codes,
    validate_full_graph_root_mask,
    write_csv,
)
from training.attention_utils import attention_relation_ids
from training.train import batch_features, feature_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared L1 graph_data.pt used by all checkpoints")
    parser.add_argument("--checkpoint", action="append", default=[], help="Exact best_model.pt path; repeatable")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob of best_model.pt files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "labeled", "all"],
        default="test",
        help="Prediction target nodes",
    )
    parser.add_argument(
        "--num-neighbors",
        default="full",
        help="Comma-separated fan-outs, auto, or full; full uses -1 at every RGAT layer",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--forward-mode",
        choices=["full-graph", "full-neighborhood"],
        default="full-neighborhood",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def collect_checkpoint_node_attention(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    split: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str = "full-neighborhood",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    num_layers = model_depth(checkpoint)
    if len(model.convs) != num_layers:
        raise ValueError(
            f"Checkpoint depth metadata ({num_layers}) does not match its state dict "
            f"({len(model.convs)} layers): {path}"
        )
    fanouts = fanouts_for_checkpoint(path, requested_fanouts, num_layers)
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    root_ids = root_indices(data, split)
    if forward_mode == "full-graph":
        validate_full_graph_root_mask(
            data, root_ids, split, feature_schema, metadata["occupation_unknown_ids"]
        )
    elif forward_mode != "full-neighborhood":
        raise ValueError(f"Unknown forward mode: {forward_mode}")

    id_to_relation = {
        int(index): relation for relation, index in metadata["relation_to_id"].items()
    }
    id_to_label = {int(index): label for label, index in metadata["label_to_id"].items()}
    class_count = int(metadata["num_classes"])
    relation_slots = max(int(value) for value in metadata["relation_to_id"].values()) + 1
    fanout_description = (
        "full-graph" if forward_mode == "full-graph" else ",".join(str(value) for value in fanouts)
    )
    experiment, seed = checkpoint_identity(path)
    root_records: List[Dict[str, object]] = []
    roster_records: List[Dict[str, object]] = []
    synthetic_target_edges: Dict[int, int] = defaultdict(int)
    roots_seen = 0
    head_count = 0

    def batches():
        if forward_mode == "full-graph":
            graph = data.to(device)
            local_roots = root_ids.to(device)
            global_ids = torch.arange(graph.num_nodes, dtype=torch.long, device=device)
            logits, explanation = model(
                feature_inputs(graph, feature_schema),
                graph.edge_index,
                graph.edge_type,
                return_attention_weights=True,
            )
            del logits
            yield graph, local_roots, global_ids, explanation
            return

        loader = NeighborLoader(
            data,
            input_nodes=prediction_nodes(data, split),
            num_neighbors=fanouts,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, num_workers),
            persistent_workers=num_workers > 0,
            pin_memory=device.type == "cuda",
        )
        for batch in loader:
            batch = batch.to(device)
            local_roots = torch.arange(batch.batch_size, dtype=torch.long, device=device)
            logits, explanation = model(
                batch_features(batch, feature_schema, metadata["occupation_unknown_ids"]),
                batch.edge_index,
                batch.edge_type,
                return_attention_weights=True,
            )
            del logits
            yield batch, local_roots, batch.n_id, explanation

    for batch, local_roots, global_ids, explanation in batches():
        root_count = int(local_roots.numel())
        roots_seen += root_count
        root_slot = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
        root_slot[local_roots] = torch.arange(root_count, dtype=torch.long, device=device)
        global_root_ids = global_ids[local_roots].detach().cpu().tolist()
        root_label_ids = batch.y[local_roots].detach().cpu().tolist()

        for layer_info in explanation["attention_layers"]:
            layer = int(layer_info["layer"]) + 1
            edge_index = layer_info["edge_index"]
            relation_ids = attention_relation_ids(layer_info)
            per_head_alpha = head_alpha(layer_info["alpha"])
            head_count = max(head_count, int(per_head_alpha.size(1)))
            scores = per_head_alpha.mean(dim=-1)
            target_slots = root_slot[edge_index[1]]
            is_prediction_target = target_slots >= 0
            is_typed_relation = relation_ids >= 0
            usable = is_prediction_target & is_typed_relation
            synthetic_target_edges[layer] += int(
                (is_prediction_target & ~is_typed_relation).sum().item()
            )

            root_slots = target_slots[is_prediction_target]
            total_counts = torch.bincount(root_slots, minlength=root_count)
            typed_slots = target_slots[usable]
            typed_counts = torch.bincount(typed_slots, minlength=root_count)
            synthetic = is_prediction_target & ~is_typed_relation
            synthetic_slots = target_slots[synthetic]
            synthetic_counts = torch.bincount(synthetic_slots, minlength=root_count)

            total_head_mass = torch.zeros(
                (root_count, per_head_alpha.size(1)),
                dtype=per_head_alpha.dtype,
                device=device,
            )
            total_head_mass.index_add_(0, root_slots, per_head_alpha[is_prediction_target])
            typed_head_mass = torch.zeros_like(total_head_mass)
            typed_head_mass.index_add_(0, typed_slots, per_head_alpha[usable])
            synthetic_head_mass = torch.zeros_like(total_head_mass)
            if int(synthetic_slots.numel()):
                synthetic_head_mass.index_add_(0, synthetic_slots, per_head_alpha[synthetic])

            for slot, (root_index, target_label_id) in enumerate(
                zip(global_root_ids, root_label_ids)
            ):
                record = {
                    "experiment": experiment,
                    "seed": seed,
                    "checkpoint": str(path),
                    "split": split,
                    "forward_mode": forward_mode,
                    "num_layers": num_layers,
                    "fanouts": fanout_description,
                    "message_passing_layer": layer,
                    "root_index": int(root_index),
                    "target_l1_id": int(target_label_id),
                    "target_l1": id_to_label.get(int(target_label_id), "__UNLABELED__"),
                    "total_incoming_attention_edges": int(total_counts[slot].item()),
                    "typed_incoming_attention_edges": int(typed_counts[slot].item()),
                    "synthetic_self_loop_edges": int(synthetic_counts[slot].item()),
                    "total_attention_mass": float(total_head_mass[slot].mean().item()),
                    "typed_attention_mass": float(typed_head_mass[slot].mean().item()),
                    "synthetic_self_loop_mass": float(synthetic_head_mass[slot].mean().item()),
                }
                for head in range(per_head_alpha.size(1)):
                    record[f"total_attention_mass_head_{head}"] = float(
                        total_head_mass[slot, head].item()
                    )
                    record[f"typed_attention_mass_head_{head}"] = float(
                        typed_head_mass[slot, head].item()
                    )
                    record[f"synthetic_self_loop_mass_head_{head}"] = float(
                        synthetic_head_mass[slot, head].item()
                    )
                roster_records.append(record)

            if not bool(usable.any()):
                continue
            usable_relation_ids = relation_ids[usable].long()
            source_l1 = batch.y[edge_index[0]]
            source_visibility = source_visibility_codes(
                batch, edge_index[0], feature_schema, metadata["occupation_unknown_ids"]
            )
            source_slots = source_l1.clone().long()
            source_slots[source_slots < 0] = class_count
            encoded_root = (
                (((target_slots[usable].long() * relation_slots + usable_relation_ids)
                  * (class_count + 1) + source_slots[usable]) * 3)
                + source_visibility[usable]
            )
            unique, inverse = torch.unique(encoded_root, sorted=True, return_inverse=True)
            group_count = torch.bincount(inverse, minlength=unique.numel())
            group_mass = torch.zeros(unique.numel(), dtype=torch.float64, device=device)
            group_mass.scatter_add_(0, inverse, scores[usable].to(dtype=torch.float64))
            group_head_mass = torch.zeros(
                (unique.numel(), per_head_alpha.size(1)), dtype=torch.float64, device=device
            )
            group_head_mass.index_add_(
                0, inverse, per_head_alpha[usable].to(dtype=torch.float64)
            )

            for encoded_value, edge_count, mass, per_head in zip(
                unique.tolist(),
                group_count.tolist(),
                group_mass.tolist(),
                group_head_mass.tolist(),
            ):
                visibility_id = encoded_value % 3
                decoded = encoded_value // 3
                source_label_id = decoded % (class_count + 1)
                decoded //= class_count + 1
                relation_id = decoded % relation_slots
                slot = decoded // relation_slots
                total_candidates = int(total_counts[slot].item())
                source_l1_id = -1 if source_label_id == class_count else int(source_label_id)
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
                        "hidden_validation_or_test" if visibility_id == 1 else
                        "missing_or_unknown"
                    ),
                    "candidate_edge_count": int(edge_count),
                    "total_incoming_attention_edges": total_candidates,
                    "attention_mass": float(mass),
                    "opportunity": (
                        float(edge_count / total_candidates) if total_candidates else 0.0
                    ),
                    "attention_mass_minus_opportunity": (
                        float(mass - edge_count / total_candidates)
                        if total_candidates else float(mass)
                    ),
                }
                for head, value in enumerate(per_head):
                    record[f"attention_mass_head_{head}"] = float(value)
                root_records.append(record)

    run_info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": num_layers,
        "fanouts": fanouts,
        "forward_mode": forward_mode,
        "prediction_roots": roots_seen,
        "synthetic_self_loop_target_edges": {
            str(layer): count for layer, count in synthetic_target_edges.items()
        },
        "attention_head_count": head_count,
    }
    return root_records, roster_records, run_info


def main() -> None:
    args = parse_args()
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    if base_metadata.get("target_column") != "occupation_level1":
        raise ValueError("attention-node-report currently requires an L1-prepared graph artifact")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_root_records: List[Dict[str, object]] = []
    all_roster_records: List[Dict[str, object]] = []
    run_info = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] collecting node attention: {path}", flush=True)
        root_records, roster_records, info = collect_checkpoint_node_attention(
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
        if not roster_records:
            raise RuntimeError(f"No prediction target nodes were collected from {path}")
        all_root_records.extend(root_records)
        all_roster_records.extend(roster_records)
        run_info.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    roster_path = output_dir / "root_attention_roster_by_seed.csv.gz"
    sparse_path = output_dir / "root_direct_attention_sparse_by_seed.csv.gz"
    roster_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1",
        "total_incoming_attention_edges", "typed_incoming_attention_edges", "synthetic_self_loop_edges",
        "total_attention_mass", "typed_attention_mass", "synthetic_self_loop_mass",
    ]
    sparse_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1", "relation_id", "relation",
        "source_l1_id", "source_l1", "source_visibility", "candidate_edge_count",
        "total_incoming_attention_edges", "attention_mass", "opportunity", "attention_mass_minus_opportunity",
    ]
    write_csv(roster_path, all_roster_records, output_fields(all_roster_records, roster_preferred))
    write_csv(sparse_path, all_root_records, output_fields(all_root_records, sparse_preferred))

    manifest_path = output_dir / "node_attention_report_manifest.json"
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
            "Raw node-level direct-attention export. Each sparse row sums head-averaged alpha "
            "within one prediction target, layer, exact relation, source L1, and visibility group. "
            "No across-node importance estimator is computed by this command."
        ),
        "sparse_path": str(sparse_path),
        "roster_path": str(roster_path),
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote:")
    for path in (roster_path, sparse_path, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
