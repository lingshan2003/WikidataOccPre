"""Shared loading, sampling, metrics, and probe serialization helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import NeighborLoader

from training.attention_common import (
    fanouts_for_checkpoint,
    git_revision,
    model_depth,
    replay_relation_perturbation,
    root_indices,
    sha256_file,
)
from training.train import batch_features

from .adapter import GraphMaskModelAdapter, restore_graphmask_model
from .core import GraphMaskProbe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is False")
    return device


def resolve_fanouts(
    checkpoint_path: Path,
    checkpoint: Mapping[str, object],
    requested: str,
) -> list[int]:
    return fanouts_for_checkpoint(checkpoint_path, requested, model_depth(checkpoint))


def make_loader(
    data,
    roots: torch.Tensor,
    fanouts: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> NeighborLoader:
    workers = max(0, int(num_workers))
    return NeighborLoader(
        data,
        input_nodes=roots,
        num_neighbors=list(fanouts),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def load_run_context(data_path: Path, checkpoint_path: Path, device: torch.device):
    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = bundle["data"]
    data_metadata = bundle["metadata"]
    checkpoint_metadata = checkpoint["metadata"]
    if data_metadata["relation_to_id"] != checkpoint_metadata["relation_to_id"]:
        raise ValueError("Checkpoint and --data use different relation mappings")
    if int(data_metadata["num_classes"]) != int(checkpoint_metadata["num_classes"]):
        raise ValueError("Checkpoint and --data use different label spaces")
    if data_metadata["label_to_id"] != checkpoint_metadata["label_to_id"]:
        raise ValueError("Checkpoint and --data use different label mappings")
    replay_relation_perturbation(data, checkpoint_metadata, checkpoint)
    restored = restore_graphmask_model(checkpoint, device)
    return data, data_metadata, checkpoint, restored


def graphmask_kl(original_logits: torch.Tensor, masked_logits: torch.Tensor) -> torch.Tensor:
    """Return KL(original || masked) independently for every row."""
    original_probability = original_logits.softmax(dim=-1)
    return F.kl_div(
        masked_logits.log_softmax(dim=-1), original_probability, reduction="none"
    ).sum(dim=-1).clamp_min(0.0)


def _classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | None]:
    if not labels:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def evaluate_probe(
    adapter: GraphMaskModelAdapter,
    probe: GraphMaskProbe,
    loader: NeighborLoader,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
    device: torch.device,
) -> dict[str, object]:
    adapter.model.eval()
    adapter.reference_model.eval()
    probe.eval()
    labels: list[int] = []
    original_predictions: list[int] = []
    masked_predictions: list[int] = []
    all_original_predictions: list[int] = []
    all_masked_predictions: list[int] = []
    divergences: list[float] = []
    layer_hard = [0.0 for _ in probe.gates]
    layer_probability = [0.0 for _ in probe.gates]
    layer_count = [0 for _ in probe.gates]

    for batch in loader:
        batch = batch.to(device)
        root_count = int(batch.batch_size)
        features = batch_features(
            batch, feature_schema, occupation_unknown_ids
        )
        original_logits, traces = adapter.trace(features, batch.edge_index, batch.edge_type)
        gates, probabilities, _, _ = probe(traces)
        masked_logits = adapter.masked_forward(
            features, batch.edge_index, batch.edge_type, gates, probe.baselines
        )
        root_original = original_logits[:root_count]
        root_masked = masked_logits[:root_count]
        root_labels = batch.y[:root_count]
        valid = root_labels >= 0
        root_original_predictions = root_original.argmax(dim=-1)
        root_masked_predictions = root_masked.argmax(dim=-1)
        all_original_predictions.extend(root_original_predictions.detach().cpu().tolist())
        all_masked_predictions.extend(root_masked_predictions.detach().cpu().tolist())
        labels.extend(root_labels[valid].detach().cpu().tolist())
        original_predictions.extend(
            root_original_predictions[valid].detach().cpu().tolist()
        )
        masked_predictions.extend(
            root_masked_predictions[valid].detach().cpu().tolist()
        )
        divergences.extend(
            graphmask_kl(root_original, root_masked).detach().cpu().tolist()
        )
        for layer, (gate, probability) in enumerate(zip(gates, probabilities)):
            layer_hard[layer] += float(gate.sum().item())
            layer_probability[layer] += float(probability.sum().item())
            layer_count[layer] += int(gate.numel())

    total_count = sum(layer_count)
    original_metrics = _classification_metrics(labels, original_predictions)
    masked_metrics = _classification_metrics(labels, masked_predictions)
    agreement = (
        float(np.mean(np.equal(all_original_predictions, all_masked_predictions)))
        if all_original_predictions else None
    )
    return {
        "roots": len(divergences),
        "labeled_roots": len(labels),
        "original": original_metrics,
        "masked": masked_metrics,
        "prediction_agreement": agreement,
        "mean_kl": float(np.mean(divergences)) if divergences else None,
        "hard_retention_rate": sum(layer_hard) / total_count if total_count else None,
        "mean_keep_probability": sum(layer_probability) / total_count if total_count else None,
        "layers": [
            {
                "layer": layer,
                "message_observations": count,
                "hard_retention_rate": layer_hard[layer] / count if count else None,
                "mean_keep_probability": layer_probability[layer] / count if count else None,
            }
            for layer, count in enumerate(layer_count)
        ],
    }


def relative_macro_f1_difference(metrics: Mapping[str, object]) -> float:
    original = metrics["original"]["macro_f1"]
    masked = metrics["masked"]["macro_f1"]
    if original is None or masked is None:
        raise ValueError("Validation split must contain labeled roots")
    return abs(float(original) - float(masked)) / max(abs(float(original)), 1e-12)


def probe_payload(
    probe: GraphMaskProbe,
    data_path: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, object],
    fanouts: Sequence[int],
    seed: int,
    validation: Mapping[str, object],
    training_config: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "state_dict": probe.state_dict(),
        "probe_config": probe.config(),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "data": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "model_name": checkpoint.get("model_name", "rgat"),
        "model_config": checkpoint.get("model_config", {}),
        "relation_to_id": checkpoint["metadata"]["relation_to_id"],
        "fanouts": list(fanouts),
        "seed": int(seed),
        "validation": dict(validation),
        "training_config": dict(training_config),
        "git_revision": git_revision(),
    }


def load_probe(path: Path, device: torch.device) -> tuple[GraphMaskProbe, Mapping[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError(f"Unsupported GraphMask probe format in {path}")
    config = payload["probe_config"]
    probe = GraphMaskProbe(
        config["layer_specs"],
        temperature=float(config["temperature"]),
        location_bias=float(config["location_bias"]),
    )
    probe.load_state_dict(payload["state_dict"])
    probe.enable_recorded_layers()
    return probe.to(device).eval(), payload


def validate_probe_sources(
    payload: Mapping[str, object], data_path: Path, checkpoint_path: Path
) -> None:
    if payload["source_checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("GraphMask probe was trained for a different source checkpoint")
    if payload["data_sha256"] != sha256_file(data_path):
        raise ValueError("GraphMask probe was trained with a different graph artifact")


def write_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
