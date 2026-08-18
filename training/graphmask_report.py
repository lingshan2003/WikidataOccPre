#!/usr/bin/env python3
"""Export GraphMask edge and relation importance for a frozen checkpoint."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from training.attention_common import git_revision, root_indices, sha256_file
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split", choices=["train", "val", "test", "labeled", "all"], default="test"
    )
    parser.add_argument(
        "--num-neighbors",
        default="auto",
        help="auto reuses the probe fan-outs; full or explicit fan-outs override them",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Edges retained per root and layer")
    parser.add_argument("--seed", type=int, default=None, help="Neighbour-sampling seed")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _base_relation(relation: str) -> str:
    return relation.removesuffix("__rev")


def _node_ids(data_path: Path, count: int) -> list[str]:
    path = data_path.parent / "nodes.csv"
    if not path.is_file():
        return [str(index) for index in range(count)]
    values = pd.read_csv(path, usecols=["node_id"])["node_id"].astype(str).tolist()
    if len(values) != count:
        raise ValueError(f"{path} has {len(values)} rows but graph_data.pt has {count} nodes")
    return values


def _new_stats() -> dict[str, object]:
    return {
        "observations": 0,
        "roots": set(),
        "probability_sum": 0.0,
        "hard_sum": 0.0,
    }


def _update_stats(
    stats: dict[str, object], root: int, probability: torch.Tensor, hard: torch.Tensor
) -> None:
    stats["observations"] += int(probability.numel())
    stats["roots"].add(int(root))
    stats["probability_sum"] += float(probability.sum().item())
    stats["hard_sum"] += float(hard.sum().item())


def _relation_rows(
    aggregates: Mapping[tuple, dict[str, object]],
    layer_hard: list[float],
    layer_probability: list[float],
    directed: bool,
) -> list[dict[str, object]]:
    rows = []
    for key, stats in sorted(aggregates.items()):
        if directed:
            layer, relation_id, relation = key
        else:
            layer, relation = key
            relation_id = None
        observations = int(stats["observations"])
        hard_sum = float(stats["hard_sum"])
        probability_sum = float(stats["probability_sum"])
        row = {
            "layer": int(layer),
            "relation": relation,
            "message_observations": observations,
            "root_coverage_count": len(stats["roots"]),
            "mean_keep_probability": probability_sum / observations if observations else 0.0,
            "hard_retention_rate": hard_sum / observations if observations else 0.0,
            "expected_retained_share": (
                probability_sum / layer_probability[layer] if layer_probability[layer] else 0.0
            ),
            "retained_edge_share": hard_sum / layer_hard[layer] if layer_hard[layer] else 0.0,
        }
        if directed:
            row = {"layer": row.pop("layer"), "relation_id": int(relation_id), **row}
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No records were produced for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_block(labels: list[int], predictions: list[int]) -> dict[str, float | None]:
    if not labels:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None}
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    device = resolve_device(args.device)
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    probe_path = Path(args.probe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata, checkpoint, restored = load_run_context(
        data_path, checkpoint_path, device
    )
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
    node_ids = _node_ids(data_path, data.num_nodes)
    id_to_relation = {
        int(index): relation for relation, index in metadata["relation_to_id"].items()
    }
    id_to_label = {
        int(index): label for label, index in metadata["label_to_id"].items()
    }
    directed = defaultdict(_new_stats)
    base = defaultdict(_new_stats)
    layer_hard = [0.0 for _ in probe.gates]
    layer_probability = [0.0 for _ in probe.gates]
    layer_count = [0 for _ in probe.gates]
    labels: list[int] = []
    original_predictions: list[int] = []
    masked_predictions: list[int] = []
    all_original_predictions: list[int] = []
    all_masked_predictions: list[int] = []
    divergences: list[float] = []

    top_fields = [
        "root_index", "root_id", "true_label_id", "true_label", "original_prediction_id",
        "original_prediction", "original_confidence", "masked_prediction_id", "masked_prediction",
        "masked_confidence", "layer", "source_index", "source_id", "relation_id", "relation",
        "base_relation", "target_index", "target_id", "keep_probability", "hard_keep",
        "is_root_incident",
    ]
    top_path = output_dir / "root_top_edges.csv.gz"
    with gzip.open(top_path, "wt", newline="", encoding="utf-8") as top_handle:
        top_writer = csv.DictWriter(top_handle, fieldnames=top_fields)
        top_writer.writeheader()
        with torch.no_grad():
            probe.eval()
            for batch in loader:
                batch = batch.to(device)
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
                label_id = int(batch.y[0].item())
                original_probability = original_logits[0].softmax(dim=-1)
                masked_probability = masked_logits[0].softmax(dim=-1)
                original_prediction = int(original_probability.argmax().item())
                masked_prediction = int(masked_probability.argmax().item())
                all_original_predictions.append(original_prediction)
                all_masked_predictions.append(masked_prediction)
                if label_id >= 0:
                    labels.append(label_id)
                    original_predictions.append(original_prediction)
                    masked_predictions.append(masked_prediction)
                divergences.append(float(graphmask_kl(
                    original_logits[:1], masked_logits[:1]
                ).item()))

                global_ids = batch.n_id
                for layer, (trace, gate, probability) in enumerate(
                    zip(traces, gates, probabilities)
                ):
                    layer_hard[layer] += float(gate.sum().item())
                    layer_probability[layer] += float(probability.sum().item())
                    layer_count[layer] += int(gate.numel())
                    relation_ids = trace.edge_type.long()
                    for relation_id in relation_ids.unique().detach().cpu().tolist():
                        selected = relation_ids == int(relation_id)
                        relation = id_to_relation[int(relation_id)]
                        _update_stats(
                            directed[(layer, int(relation_id), relation)],
                            root_global,
                            probability[selected],
                            gate[selected],
                        )
                        _update_stats(
                            base[(layer, _base_relation(relation))],
                            root_global,
                            probability[selected],
                            gate[selected],
                        )

                    count = min(args.top_k, int(probability.numel()))
                    if not count:
                        continue
                    top_indices = torch.topk(probability, count, sorted=True).indices
                    for edge_slot in top_indices.detach().cpu().tolist():
                        source_local = int(trace.edge_index[0, edge_slot].item())
                        target_local = int(trace.edge_index[1, edge_slot].item())
                        source_global = int(global_ids[source_local].item())
                        target_global = int(global_ids[target_local].item())
                        relation_id = int(trace.edge_type[edge_slot].item())
                        relation = id_to_relation[relation_id]
                        top_writer.writerow({
                            "root_index": root_global,
                            "root_id": node_ids[root_global],
                            "true_label_id": label_id,
                            "true_label": id_to_label.get(label_id, "__UNLABELED__"),
                            "original_prediction_id": original_prediction,
                            "original_prediction": id_to_label[original_prediction],
                            "original_confidence": float(original_probability[original_prediction].item()),
                            "masked_prediction_id": masked_prediction,
                            "masked_prediction": id_to_label[masked_prediction],
                            "masked_confidence": float(masked_probability[masked_prediction].item()),
                            "layer": layer,
                            "source_index": source_global,
                            "source_id": node_ids[source_global],
                            "relation_id": relation_id,
                            "relation": relation,
                            "base_relation": _base_relation(relation),
                            "target_index": target_global,
                            "target_id": node_ids[target_global],
                            "keep_probability": float(probability[edge_slot].item()),
                            "hard_keep": int(gate[edge_slot].item()),
                            "is_root_incident": (
                                source_global == root_global or target_global == root_global
                            ),
                        })

    total_count = sum(layer_count)
    metrics = {
        "split": args.split,
        "roots": len(divergences),
        "labeled_roots": len(labels),
        "original": _metric_block(labels, original_predictions),
        "masked": _metric_block(labels, masked_predictions),
        "prediction_agreement": (
            float(np.mean(np.equal(all_original_predictions, all_masked_predictions)))
            if all_original_predictions else None
        ),
        "mean_kl": float(np.mean(divergences)) if divergences else None,
        "hard_retention_rate": sum(layer_hard) / total_count if total_count else None,
        "mean_keep_probability": sum(layer_probability) / total_count if total_count else None,
        "layers": [
            {
                "layer": layer,
                "message_observations": layer_count[layer],
                "hard_retention_rate": (
                    layer_hard[layer] / layer_count[layer] if layer_count[layer] else None
                ),
                "mean_keep_probability": (
                    layer_probability[layer] / layer_count[layer] if layer_count[layer] else None
                ),
            }
            for layer in range(len(layer_count))
        ],
    }
    _write_csv(
        output_dir / "relations_directed.csv",
        _relation_rows(directed, layer_hard, layer_probability, directed=True),
    )
    _write_csv(
        output_dir / "relations_base.csv",
        _relation_rows(base, layer_hard, layer_probability, directed=False),
    )
    write_json(output_dir / "test_metrics.json", metrics)
    relation_perturbation = checkpoint.get("relation_perturbation") or {}
    shuffled_relations = bool(relation_perturbation.get("relation_type_shuffle", False))
    write_json(output_dir / "manifest.json", {
        "artifact": "graphmask_report",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "probe": str(probe_path.resolve()),
        "probe_sha256": sha256_file(probe_path),
        "data": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "model_name": restored.model_name,
        "split": args.split,
        "fanouts": fanouts,
        "sampling_scope": (
            "full-neighborhood" if all(value == -1 for value in fanouts)
            else "fixed-sampled-neighborhood"
        ),
        "sampling_seed": report_seed,
        "top_k_per_root_per_layer": args.top_k,
        "relation_type_semantics": (
            "shuffled_model_assignments" if shuffled_relations else "source_relation_names"
        ),
        "git_revision": git_revision(),
        "note": (
            "GraphMask describes frozen-model dependence on messages; it does not establish "
            "a causal social effect of a relationship."
            + (
                " This checkpoint shuffled relation IDs, so exported names describe model "
                "assignments rather than source relation semantics."
                if shuffled_relations else ""
            )
        ),
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
