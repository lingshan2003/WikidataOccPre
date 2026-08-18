#!/usr/bin/env python3
"""Train an amortized GraphMask probe for a frozen relational GNN checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from training.attention_common import root_indices
from training.graphmask.common import (
    evaluate_probe,
    graphmask_kl,
    load_run_context,
    make_loader,
    probe_payload,
    relative_macro_f1_difference,
    resolve_device,
    resolve_fanouts,
    set_seed,
    write_json,
)
from training.graphmask.core import GraphMaskProbe, LagrangianOptimization
from training.train import batch_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-neighbors", default="auto", help="auto, full, or one fan-out per layer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-per-layer", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.03, help="Allowed mean KL divergence")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--multiplier-learning-rate", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=1.0 / 3.0)
    parser.add_argument("--location-bias", type=float, default=3.0)
    parser.add_argument("--max-relative-macro-f1-diff", type=float, default=0.05)
    parser.add_argument("--train-split", choices=["train"], default="train")
    parser.add_argument("--validation-split", choices=["val"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _validation_loader(data, args, fanouts):
    # Evaluation always uses the same sampled computation graphs so that probe
    # checkpoint selection is not driven by sampler noise.
    torch.manual_seed(args.seed + 100_000)
    return make_loader(
        data,
        root_indices(data, args.validation_split),
        fanouts,
        args.batch_size,
        shuffle=False,
        num_workers=0,
    )


def main() -> None:
    args = parse_args()
    if args.epochs_per_layer < 1:
        raise ValueError("--epochs-per-layer must be positive")
    if args.beta < 0:
        raise ValueError("--beta must be non-negative")
    if args.max_relative_macro_f1_diff < 0:
        raise ValueError("--max-relative-macro-f1-diff must be non-negative")
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, checkpoint, restored = load_run_context(
        data_path, checkpoint_path, device
    )
    fanouts = resolve_fanouts(checkpoint_path, checkpoint, args.num_neighbors)
    train_roots = root_indices(data, args.train_split)
    if not int(train_roots.numel()):
        raise ValueError("Training split contains no roots")

    initialization_loader = make_loader(
        data, train_roots, fanouts, args.batch_size, shuffle=False, num_workers=0
    )
    first_batch = next(iter(initialization_loader)).to(device)
    first_features = batch_features(
        first_batch,
        restored.feature_schema,
        metadata["occupation_unknown_ids"],
    )
    _, first_traces = restored.adapter.trace(
        first_features, first_batch.edge_index, first_batch.edge_type
    )
    probe = GraphMaskProbe.from_traces(
        first_traces,
        temperature=args.temperature,
        location_bias=args.location_bias,
    ).to(device)
    optimization = LagrangianOptimization(
        probe.parameters(),
        learning_rate=args.learning_rate,
        multiplier_learning_rate=args.multiplier_learning_rate,
        device=device,
    )

    history: list[dict[str, object]] = []
    best_state = None
    best_validation = None
    best_retention = float("inf")
    global_epoch = 0

    for layer in reversed(range(len(probe.gates))):
        probe.enable_layer(layer)
        for layer_epoch in range(1, args.epochs_per_layer + 1):
            global_epoch += 1
            probe.train()
            torch.manual_seed(args.seed + global_epoch)
            train_loader = make_loader(
                data,
                train_roots,
                fanouts,
                args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
            divergence_total = 0.0
            penalty_total = 0.0
            constraint_total = 0.0
            batches = 0
            roots_seen = 0
            for batch in train_loader:
                batch = batch.to(device)
                root_count = int(batch.batch_size)
                features = batch_features(
                    batch,
                    restored.feature_schema,
                    metadata["occupation_unknown_ids"],
                )
                original_logits, traces = restored.adapter.trace(
                    features, batch.edge_index, batch.edge_type
                )
                gates, _, expected_count, eligible_count = probe(traces)
                if not eligible_count:
                    # Isolated roots provide no gate-training signal. They are
                    # still covered by validation/report metrics, but there is
                    # no differentiable sparsity update to perform here.
                    continue
                masked_logits = restored.adapter.masked_forward(
                    features,
                    batch.edge_index,
                    batch.edge_type,
                    gates,
                    probe.baselines,
                )
                divergence = graphmask_kl(
                    original_logits[:root_count], masked_logits[:root_count]
                ).mean()
                penalty = expected_count / eligible_count
                constraint = torch.relu(divergence - args.beta)
                optimization.update(penalty, constraint)
                divergence_total += float(divergence.detach().item()) * root_count
                penalty_total += float(penalty.detach().item())
                constraint_total += float(constraint.detach().item())
                batches += 1
                roots_seen += root_count

            validation = evaluate_probe(
                restored.adapter,
                probe,
                _validation_loader(data, args, fanouts),
                restored.feature_schema,
                metadata["occupation_unknown_ids"],
                device,
            )
            relative_difference = relative_macro_f1_difference(validation)
            validation["relative_macro_f1_difference"] = relative_difference
            eligible = relative_difference <= args.max_relative_macro_f1_diff
            record = {
                "global_epoch": global_epoch,
                "enabled_through_layer": layer,
                "layer_epoch": layer_epoch,
                "train_roots": roots_seen,
                "train_mean_kl": divergence_total / max(roots_seen, 1),
                "train_mean_expected_l0": penalty_total / max(batches, 1),
                "train_mean_constraint_violation": constraint_total / max(batches, 1),
                "lagrange_multiplier": float(optimization.multiplier.detach().item()),
                "validation": validation,
                "eligible": eligible,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            retention = validation["hard_retention_rate"]
            if eligible and retention is not None and float(retention) < best_retention:
                best_retention = float(retention)
                best_state = copy.deepcopy(probe.state_dict())
                best_validation = copy.deepcopy(validation)

    write_json(output_dir / "training_history.json", {"history": history})
    if best_state is None or best_validation is None:
        raise RuntimeError(
            "No GraphMask probe satisfied the validation fidelity threshold; "
            "inspect training_history.json and adjust training or the explicit threshold"
        )

    probe.load_state_dict(best_state)
    probe.enable_recorded_layers()
    # Re-evaluate the exact selected state for the stable validation artifact.
    best_validation = evaluate_probe(
        restored.adapter,
        probe,
        _validation_loader(data, args, fanouts),
        restored.feature_schema,
        metadata["occupation_unknown_ids"],
        device,
    )
    best_validation["relative_macro_f1_difference"] = relative_macro_f1_difference(
        best_validation
    )
    training_config = vars(args).copy()
    payload = probe_payload(
        probe,
        data_path,
        checkpoint_path,
        checkpoint,
        fanouts,
        args.seed,
        best_validation,
        training_config,
    )
    torch.save(payload, output_dir / "graphmask_probe.pt")
    write_json(output_dir / "validation.json", best_validation)
    write_json(output_dir / "manifest.json", {
        "artifact": "graphmask_probe",
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "data": payload["data"],
        "data_sha256": payload["data_sha256"],
        "model_name": payload["model_name"],
        "fanouts": payload["fanouts"],
        "seed": payload["seed"],
        "git_revision": payload["git_revision"],
        "training_config": training_config,
    })
    print(json.dumps(best_validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
