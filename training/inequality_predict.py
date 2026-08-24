#!/usr/bin/env python3
"""Export aligned probabilities from a frozen checkpoint for inequality audits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import NeighborLoader

from models import build_feature_specs, build_model
from training.attention_common import (
    fanouts_for_checkpoint,
    git_revision,
    model_depth,
    replay_relation_perturbation,
    root_indices,
    sha256_file,
)
from training.relation_controls import apply_relation_controls
from training.tie_taxonomy import (
    DEFAULT_TIE_TAXONOMY_PATH,
    load_tie_taxonomy,
    parse_tie_group_selection,
    resolve_tie_ablation,
)
from training.train import batch_features, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", required=True, help="Stable experiment-condition label")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--num-neighbors", default="auto", help="auto, full, or one fan-out per layer")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tie-taxonomy", default=str(DEFAULT_TIE_TAXONOMY_PATH))
    parser.add_argument(
        "--inference-drop-tie-groups",
        default="none",
        help="Optional frozen-checkpoint inherited/acquired edge intervention",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def restore_model(checkpoint: Mapping[str, object], device: torch.device):
    metadata = checkpoint["metadata"]
    feature_schema = checkpoint.get("model_feature_schema", metadata["feature_schema"])
    model_name = str(checkpoint.get("model_name", "rgat"))
    model = build_model(
        model_name,
        num_relations=int(metadata["num_relations"]),
        num_classes=int(metadata["num_classes"]),
        feature_specs=build_feature_specs(feature_schema, metadata),
        **checkpoint.get("model_config", {}),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), feature_schema, metadata, model_name


def validate_metadata(data_metadata: Mapping[str, object], checkpoint_metadata: Mapping[str, object]) -> None:
    for key in ("relation_to_id", "label_to_id"):
        if data_metadata[key] != checkpoint_metadata[key]:
            raise ValueError(f"Checkpoint and graph artifact have different {key}")
    for key in ("num_relations", "num_classes"):
        if int(data_metadata[key]) != int(checkpoint_metadata[key]):
            raise ValueError(f"Checkpoint and graph artifact have different {key}")


@torch.no_grad()
def predict(
    model,
    data,
    roots: torch.Tensor,
    fanouts: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
) -> dict[str, np.ndarray]:
    workers = max(0, int(num_workers))
    loader = NeighborLoader(
        data,
        input_nodes=roots,
        num_neighbors=fanouts,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    node_indices, labels, probabilities = [], [], []
    for batch in loader:
        batch = batch.to(device)
        root_count = int(batch.batch_size)
        logits = model(
            batch_features(batch, feature_schema, occupation_unknown_ids),
            batch.edge_index,
            batch.edge_type,
        )[:root_count]
        node_indices.append(batch.n_id[:root_count].detach().cpu())
        labels.append(batch.y[:root_count].detach().cpu())
        probabilities.append(logits.softmax(dim=-1).detach().cpu())
    return {
        "node_index": torch.cat(node_indices).numpy().astype(np.int64, copy=False),
        "label": torch.cat(labels).numpy().astype(np.int64, copy=False),
        "probabilities": torch.cat(probabilities).numpy().astype(np.float32, copy=False),
    }


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    probability_tensor = torch.from_numpy(probabilities)
    label_tensor = torch.from_numpy(labels)
    return {
        "loss": float(F.nll_loss(probability_tensor.clamp_min(1e-12).log(), label_tensor).item()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def write_prediction_csv(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    nodes_path: Path,
) -> None:
    node_ids = None
    if nodes_path.is_file():
        import pandas as pd

        node_ids = pd.read_csv(nodes_path, usecols=["node_id"])["node_id"].astype(str).to_numpy()
    id_to_label = {int(identifier): str(label) for label, identifier in metadata["label_to_id"].items()}
    predictions = arrays["probabilities"].argmax(axis=1)
    confidences = arrays["probabilities"].max(axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_index", "node_id", "true_label", "prediction", "confidence"],
        )
        writer.writeheader()
        for node, label, prediction, confidence in zip(
            arrays["node_index"], arrays["label"], predictions, confidences
        ):
            writer.writerow({
                "node_index": int(node),
                "node_id": node_ids[int(node)] if node_ids is not None else "",
                "true_label": id_to_label[int(label)],
                "prediction": id_to_label[int(prediction)],
                "confidence": float(confidence),
            })


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path, checkpoint_path = Path(args.data), Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data, data_metadata = bundle["data"], bundle["metadata"]
    checkpoint_metadata = checkpoint["metadata"]
    validate_metadata(data_metadata, checkpoint_metadata)
    source_hash = checkpoint.get("relation_perturbation", {}).get("data_sha256")
    current_hash = sha256_file(data_path)
    if source_hash and source_hash != current_hash:
        raise ValueError("Checkpoint was trained from a different graph_data.pt payload")

    replay_relation_perturbation(data, checkpoint_metadata, checkpoint)
    inference_groups = parse_tie_group_selection(args.inference_drop_tie_groups)
    inference_manifest = None
    if inference_groups:
        taxonomy = load_tie_taxonomy(args.tie_taxonomy, data_metadata["relation_to_id"])
        relation_ids, base_relations = resolve_tie_ablation(
            taxonomy, inference_groups, data_metadata["relation_to_id"]
        )
        inference_manifest = apply_relation_controls(
            data,
            relation_ids_to_drop=relation_ids,
            relation_to_id=data_metadata["relation_to_id"],
        )
        inference_manifest.update({
            "tie_groups": list(inference_groups),
            "base_relations": list(base_relations),
            "taxonomy": taxonomy.manifest(),
        })

    model, feature_schema, metadata, model_name = restore_model(checkpoint, device)
    depth = model_depth(checkpoint)
    fanouts = fanouts_for_checkpoint(checkpoint_path, args.num_neighbors, depth)
    roots = root_indices(data, args.split)
    arrays = predict(
        model,
        data,
        roots,
        fanouts,
        args.batch_size,
        args.num_workers,
        device,
        feature_schema,
        metadata.get("occupation_unknown_ids", {}),
    )
    if np.any(arrays["label"] < 0):
        raise ValueError(f"The selected {args.split} split contains unlabeled roots")
    run_metrics = metrics(arrays["label"], arrays["probabilities"])
    np.savez_compressed(
        output_dir / "probabilities.npz",
        node_index=arrays["node_index"],
        label=arrays["label"],
        probabilities=arrays["probabilities"],
    )
    write_prediction_csv(
        output_dir / "predictions.csv", arrays, metadata, data_path.parent / "nodes.csv"
    )
    manifest = {
        "format_version": 1,
        "condition": args.condition,
        "seed": int(args.seed),
        "split": args.split,
        "model_name": model_name,
        "data": str(data_path.resolve()),
        "data_sha256": current_hash,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fanouts": fanouts,
        "roots": int(len(arrays["node_index"])),
        "num_classes": int(arrays["probabilities"].shape[1]),
        "checkpoint_relation_perturbation": checkpoint.get("relation_perturbation"),
        "inference_relation_perturbation": inference_manifest,
        "metrics": run_metrics,
        "git_revision": git_revision(),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
