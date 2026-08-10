#!/usr/bin/env python3
"""Export root-level RGAT ``attention × gradient`` prediction attributions.

For an RGAT attention coefficient ``alpha[e, h]`` and a scalar prediction
score ``F_v`` at a prediction root ``v``, this command exports the signed
gradient-times-input quantity::

    alpha[e, h] * d F_v / d alpha[e, h]

The reported sparse rows first sum every matching incoming edge inside one
root, exact directed relation, source L1, and source-visibility group.  They
therefore retain the project's node-first statistical unit.  ``attention_mass``
is written alongside the attribution, so allocation and prediction sensitivity
can be analysed together.

This is a local, conditional attribution: alpha is a softmax-normalised
quantity, so treating one alpha as independently set to zero does not preserve
the attention simplex.  It is not a causal edge-deletion effect.  For the
current one-layer RGAT experiment, every root-directed typed edge is included;
for deeper models, the export remains a direct-root, per-layer diagnostic and
does not represent complete multi-hop path attribution.
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


SCORE_CHOICES = (
    "predicted-margin",
    "true-margin",
    "predicted-logit",
    "true-logit",
)


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
        "--score",
        choices=SCORE_CHOICES,
        default="predicted-margin",
        help=(
            "Scalar prediction score to differentiate. predicted-margin explains the model's own decision; "
            "true-* scores require every selected root to have a retained L1 label."
        ),
    )
    parser.add_argument(
        "--num-neighbors",
        default="full",
        help="Comma-separated fan-outs, auto, or full. full uses -1 at every RGAT layer.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--forward-mode",
        choices=["full-graph", "full-neighborhood"],
        default="full-neighborhood",
        help=(
            "full-graph computes one forward/backward pass and can require substantial memory; "
            "full-neighborhood replays complete fan-outs in batches."
        ),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def root_prediction_score(
    logits: torch.Tensor,
    root_indices_local: torch.Tensor,
    labels: torch.Tensor,
    score_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one differentiable score and its selected class for each root.

    The predicted class is selected from the unperturbed forward pass and then
    treated as an integer index, so its non-differentiable argmax does not
    enter the derivative.  Margin scores use log-sum-exp of every competing
    class and are less affected by output-probability saturation than a
    softmax probability score.
    """
    if score_name not in SCORE_CHOICES:
        raise ValueError(f"Unknown score {score_name!r}; expected one of {SCORE_CHOICES}")
    root_logits = logits[root_indices_local]
    if root_logits.size(1) < 2 and score_name.endswith("-margin"):
        raise ValueError("Margin attribution requires at least two output classes")
    predictions = root_logits.detach().argmax(dim=-1)
    selected = predictions if score_name.startswith("predicted-") else labels
    if score_name.startswith("true-") and bool((selected < 0).any()):
        raise ValueError(f"--score {score_name} requires retained L1 labels for every selected root")
    selected = selected.long()
    selected_logits = root_logits.gather(1, selected.unsqueeze(1)).squeeze(1)
    if score_name.endswith("-logit"):
        return selected_logits, selected
    competitor_logits = root_logits.masked_fill(
        torch.nn.functional.one_hot(selected, num_classes=root_logits.size(1)).bool(),
        float("-inf"),
    )
    return selected_logits - torch.logsumexp(competitor_logits, dim=1), selected


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
    """Yield differentiable forwards; deliberately never enters no_grad."""
    if forward_mode == "full-graph":
        graph = data.to(device)
        local_roots = root_ids.to(device)
        global_ids = torch.arange(graph.num_nodes, dtype=torch.long, device=device)
        logits, explanation = model(
            feature_inputs(graph, feature_schema), graph.edge_index, graph.edge_type, return_attention_weights=True
        )
        yield graph, local_roots, global_ids, logits, explanation
        return

    loader = NeighborLoader(
        data,
        input_nodes=prediction_nodes(data, split),
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
        logits, explanation = model(
            batch_features(batch, feature_schema, occupation_unknown_ids),
            batch.edge_index,
            batch.edge_type,
            return_attention_weights=True,
        )
        yield batch, local_roots, batch.n_id, logits, explanation


def collect_checkpoint_gradient_attribution(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    split: str,
    score_name: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    forward_mode: str = "full-neighborhood",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Collect sparse root/group attribution rows for one frozen checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    num_layers = model_depth(checkpoint)
    if len(model.convs) != num_layers:
        raise ValueError(f"Checkpoint depth does not match its RGAT layers: {path}")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
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
    class_count = int(metadata["num_classes"])
    relation_slots = max(id_to_relation) + 1
    fanout_description = "full-graph" if forward_mode == "full-graph" else ",".join(str(value) for value in fanouts)
    experiment, seed = checkpoint_identity(path)
    sparse_records: List[Dict[str, object]] = []
    roster_records: List[Dict[str, object]] = []
    synthetic_target_edges: Dict[int, int] = defaultdict(int)
    roots_seen = 0
    target_l1_counts = torch.zeros(class_count, dtype=torch.long)
    head_count = 0

    model.eval()
    for batch, local_roots, global_ids, logits, explanation in _batches(
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
        score, score_target_ids = root_prediction_score(logits, local_roots, root_labels, score_name)
        raw_alphas = [layer_info["alpha"] for layer_info in explanation["attention_layers"]]
        gradients = torch.autograd.grad(score.sum(), raw_alphas, allow_unused=True)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("An RGAT alpha tensor was disconnected from the selected prediction score")

        root_slot = torch.full((batch.num_nodes,), -1, dtype=torch.long, device=device)
        root_slot[local_roots] = torch.arange(root_count, dtype=torch.long, device=device)
        global_root_ids = global_ids[local_roots].detach().cpu().tolist()
        root_label_ids = root_labels.detach().cpu().tolist()
        score_target_ids_cpu = score_target_ids.detach().cpu().tolist()
        score_cpu = score.detach().cpu().tolist()
        correct_cpu = score_target_ids.eq(root_labels).detach().cpu().tolist()

        for layer_info, raw_gradient in zip(explanation["attention_layers"], gradients):
            layer = int(layer_info["layer"]) + 1
            edge_index = layer_info["edge_index"]
            relation_ids = attention_relation_ids(layer_info)
            alpha = head_alpha(layer_info["alpha"]).detach()
            gradient = head_alpha(raw_gradient).detach()
            if alpha.shape != gradient.shape:
                raise RuntimeError("RGAT alpha and dscore/dalpha have incompatible shapes")
            head_count = max(head_count, int(alpha.size(1)))
            product = alpha * gradient
            target_slots = root_slot[edge_index[1]]
            is_prediction_target = target_slots >= 0
            is_typed_relation = relation_ids >= 0
            usable = is_prediction_target & is_typed_relation
            synthetic_target_edges[layer] += int((is_prediction_target & ~is_typed_relation).sum().item())

            root_slots_all = target_slots[is_prediction_target]
            root_slots_typed = target_slots[usable]
            total_counts = torch.bincount(root_slots_all, minlength=root_count)
            typed_counts = torch.bincount(root_slots_typed, minlength=root_count)
            synthetic_counts = torch.bincount(target_slots[is_prediction_target & ~is_typed_relation], minlength=root_count)
            total_head_attention = torch.zeros((root_count, alpha.size(1)), dtype=alpha.dtype, device=device)
            total_head_attention.index_add_(0, root_slots_all, alpha[is_prediction_target])
            total_head_product = torch.zeros_like(total_head_attention)
            total_head_product.index_add_(0, root_slots_all, product[is_prediction_target])
            typed_head_attention = torch.zeros_like(total_head_attention)
            typed_head_product = torch.zeros_like(total_head_attention)
            typed_head_absolute_product = torch.zeros_like(total_head_attention)
            typed_head_attention.index_add_(0, root_slots_typed, alpha[usable])
            typed_head_product.index_add_(0, root_slots_typed, product[usable])
            typed_head_absolute_product.index_add_(0, root_slots_typed, product[usable].abs())

            for slot, (root_index, target_l1_id, score_target_id, score_value, correct) in enumerate(
                zip(global_root_ids, root_label_ids, score_target_ids_cpu, score_cpu, correct_cpu)
            ):
                record = {
                    "experiment": experiment,
                    "seed": seed,
                    "checkpoint": str(path),
                    "split": split,
                    "forward_mode": forward_mode,
                    "num_layers": num_layers,
                    "fanouts": fanout_description,
                    "score": score_name,
                    "message_passing_layer": layer,
                    "root_index": int(root_index),
                    "target_l1_id": int(target_l1_id),
                    "target_l1": id_to_label.get(int(target_l1_id), "__UNLABELED__"),
                    "score_target_l1_id": int(score_target_id),
                    "score_target_l1": id_to_label.get(int(score_target_id), "__UNLABELED__"),
                    "prediction_score": float(score_value),
                    "prediction_is_correct": bool(correct) if int(target_l1_id) >= 0 else None,
                    "total_incoming_attention_edges": int(total_counts[slot].item()),
                    "typed_incoming_attention_edges": int(typed_counts[slot].item()),
                    "synthetic_self_loop_edges": int(synthetic_counts[slot].item()),
                    "total_attention_mass": float(total_head_attention[slot].mean().item()),
                    "total_gradient_x_attention": float(total_head_product[slot].mean().item()),
                    "typed_attention_mass": float(typed_head_attention[slot].mean().item()),
                    "typed_gradient_x_attention": float(typed_head_product[slot].mean().item()),
                    "typed_absolute_gradient_x_attention_sum": float(typed_head_absolute_product[slot].mean().item()),
                }
                for head in range(alpha.size(1)):
                    record[f"total_attention_mass_head_{head}"] = float(total_head_attention[slot, head].item())
                    record[f"total_gradient_x_attention_head_{head}"] = float(total_head_product[slot, head].item())
                    record[f"typed_attention_mass_head_{head}"] = float(typed_head_attention[slot, head].item())
                    record[f"typed_gradient_x_attention_head_{head}"] = float(typed_head_product[slot, head].item())
                roster_records.append(record)

            if not bool(usable.any()):
                continue
            usable_sources = edge_index[0, usable]
            usable_relations = relation_ids[usable].long()
            source_l1 = batch.y[usable_sources]
            source_slots = source_l1.clone().long()
            source_slots[source_slots < 0] = class_count
            visibility = source_visibility_codes(
                batch, usable_sources, feature_schema, metadata["occupation_unknown_ids"]
            )
            encoded = (
                (((root_slots_typed * relation_slots + usable_relations) * (class_count + 1) + source_slots) * 3)
                + visibility
            )
            unique, inverse, group_counts = torch.unique(encoded, sorted=True, return_inverse=True, return_counts=True)
            group_count = int(unique.numel())
            group_head_attention = torch.zeros((group_count, alpha.size(1)), dtype=alpha.dtype, device=device)
            group_head_product = torch.zeros_like(group_head_attention)
            group_head_absolute_product = torch.zeros_like(group_head_attention)
            group_head_gradient_sum = torch.zeros_like(group_head_attention)
            group_head_attention.index_add_(0, inverse, alpha[usable])
            group_head_product.index_add_(0, inverse, product[usable])
            group_head_absolute_product.index_add_(0, inverse, product[usable].abs())
            group_head_gradient_sum.index_add_(0, inverse, gradient[usable])

            decoded = unique // 3
            source_label_ids = decoded % (class_count + 1)
            decoded = decoded // (class_count + 1)
            group_relation_ids = decoded % relation_slots
            group_root_slots = decoded // relation_slots
            for edge_count, attention_by_head, product_by_head, absolute_product_by_head, gradient_sum_by_head, relation_id, source_label_id, visibility_id, slot in zip(
                group_counts.detach().cpu().tolist(),
                group_head_attention.detach().cpu().tolist(),
                group_head_product.detach().cpu().tolist(),
                group_head_absolute_product.detach().cpu().tolist(),
                group_head_gradient_sum.detach().cpu().tolist(),
                group_relation_ids.detach().cpu().tolist(),
                source_label_ids.detach().cpu().tolist(),
                (unique % 3).detach().cpu().tolist(),
                group_root_slots.detach().cpu().tolist(),
            ):
                source_l1_id = -1 if source_label_id == class_count else int(source_label_id)
                record = {
                    "experiment": experiment,
                    "seed": seed,
                    "checkpoint": str(path),
                    "split": split,
                    "forward_mode": forward_mode,
                    "num_layers": num_layers,
                    "fanouts": fanout_description,
                    "score": score_name,
                    "message_passing_layer": layer,
                    "root_index": int(global_root_ids[slot]),
                    "target_l1_id": int(root_label_ids[slot]),
                    "target_l1": id_to_label.get(int(root_label_ids[slot]), "__UNLABELED__"),
                    "score_target_l1_id": int(score_target_ids_cpu[slot]),
                    "score_target_l1": id_to_label.get(int(score_target_ids_cpu[slot]), "__UNLABELED__"),
                    "prediction_score": float(score_cpu[slot]),
                    "prediction_is_correct": bool(correct_cpu[slot]) if int(root_label_ids[slot]) >= 0 else None,
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
                    "attention_mass": float(sum(attention_by_head) / len(attention_by_head)),
                    "gradient_x_attention": float(sum(product_by_head) / len(product_by_head)),
                    "absolute_gradient_x_attention_sum": float(
                        sum(absolute_product_by_head) / len(absolute_product_by_head)
                    ),
                    "mean_gradient_wrt_attention": float(
                        sum(gradient_sum_by_head) / (len(gradient_sum_by_head) * edge_count)
                    ),
                }
                for head, value in enumerate(attention_by_head):
                    record[f"attention_mass_head_{head}"] = float(value)
                for head, value in enumerate(product_by_head):
                    record[f"gradient_x_attention_head_{head}"] = float(value)
                sparse_records.append(record)

    run_info = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": num_layers,
        "fanouts": fanouts,
        "forward_mode": forward_mode,
        "score": score_name,
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
        print(f"[{index}/{len(paths)}] collecting RGAT gradient × attention: {path}", flush=True)
        sparse_rows, roster_rows, info = collect_checkpoint_gradient_attribution(
            path,
            base_data,
            base_metadata,
            args.split,
            args.score,
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

    roster_path = output_dir / "root_gradient_x_attention_roster_by_seed.csv.gz"
    sparse_path = output_dir / "root_gradient_x_attention_sparse_by_seed.csv.gz"
    roster_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts", "score",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1", "score_target_l1_id",
        "score_target_l1", "prediction_score", "prediction_is_correct", "total_incoming_attention_edges",
        "typed_incoming_attention_edges", "synthetic_self_loop_edges", "total_attention_mass",
        "total_gradient_x_attention", "typed_attention_mass", "typed_gradient_x_attention",
        "typed_absolute_gradient_x_attention_sum",
    ]
    sparse_preferred = [
        "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts", "score",
        "message_passing_layer", "root_index", "target_l1_id", "target_l1", "score_target_l1_id",
        "score_target_l1", "prediction_score", "prediction_is_correct", "relation_id", "relation",
        "source_l1_id", "source_l1", "source_visibility", "candidate_edge_count",
        "total_incoming_attention_edges", "attention_mass", "gradient_x_attention",
        "absolute_gradient_x_attention_sum", "mean_gradient_wrt_attention",
    ]
    write_csv(roster_path, all_roster, output_fields(all_roster, roster_preferred))
    write_csv(sparse_path, all_sparse, output_fields(all_sparse, sparse_preferred))
    manifest_path = output_dir / "gradient_x_attention_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(Path(args.data)),
        "split": args.split,
        "forward_mode": args.forward_mode,
        "score": args.score,
        "code_git_revision": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": torch.version.cuda,
        "definition": (
            "For every typed edge e=j->i that enters a prediction root and head h, the signed local score is "
            "alpha[e,h] * dF_i/dalpha[e,h], where F_i is the selected logit or logit margin. "
            "gradient_x_attention first averages this product across heads and then sums matching edges inside "
            "one root/relation/source-L1/visibility group. attention_mass is the analogous head-averaged alpha sum."
        ),
        "scope": (
            "This is gradient-times-input for post-softmax alpha, conditional on the frozen graph and model. "
            "Because alpha is softmax-normalised across incoming edges, it is not a renormalised edge-deletion "
            "effect or a causal attribution. One-layer direct-root output covers the complete direct receptive "
            "field; for deeper RGATs, rows are direct-root per-layer diagnostics, not complete multi-hop paths."
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
