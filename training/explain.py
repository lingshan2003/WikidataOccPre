#!/usr/bin/env python3
"""Export relation-aware attention candidates for one person's prediction.

Attention is a ranking signal, not a causal statement. Validate exported edge
candidates later by removing them from the sampled graph and measuring the
target-class logit drop.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

from models import build_feature_specs, build_model
from training.relation_controls import apply_relation_controls
from training.attention_utils import attention_relation_ids
from training.train import feature_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/graph_data.pt")
    parser.add_argument("--checkpoint", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--node-id", help="Wikidata Q-id from artifacts/nodes.csv")
    group.add_argument("--node-index", type=int, help="Integer node index in graph_data.pt")
    parser.add_argument("--output-dir", default="explanations")
    parser.add_argument(
        "--num-neighbors",
        default=None,
        help="Comma-separated fan-outs; defaults to the checkpoint's depth with fan-out 20",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_fanouts(value: str, num_layers: int) -> List[int]:
    fanouts = [int(item.strip()) for item in value.split(",")]
    if len(fanouts) != num_layers or any(item < -1 for item in fanouts):
        raise ValueError(
            f"The {num_layers}-layer model needs exactly {num_layers} fan-outs, e.g. "
            + ",".join(["20"] * num_layers)
        )
    return fanouts


def load_node_ids(data_path: Path) -> List[str]:
    nodes_path = data_path.parent / "nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Expected node lookup table: {nodes_path}")
    return pd.read_csv(nodes_path, usecols=["node_id"])["node_id"].astype(str).tolist()


def restore_model(checkpoint: Dict, device: torch.device):
    metadata = checkpoint["metadata"]
    feature_schema = checkpoint.get("model_feature_schema", metadata["feature_schema"])
    specs = build_feature_specs(feature_schema, metadata)
    model_name = checkpoint.get("model_name", "rgat")
    if model_name != "rgat":
        raise ValueError("Attention export is available for RGAT only; RGCN has no alpha coefficients")
    model = build_model(
        model_name,
        num_relations=metadata["num_relations"],
        num_classes=metadata["num_classes"],
        feature_specs=specs,
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), feature_schema


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_path = Path(args.data)
    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    data, metadata = bundle["data"], bundle["metadata"]
    node_ids = load_node_ids(data_path)
    if args.node_id is not None:
        try:
            query_index = node_ids.index(args.node_id)
        except ValueError as error:
            raise ValueError(f"Node id {args.node_id!r} is absent from nodes.csv") from error
    else:
        query_index = args.node_index
    if query_index < 0 or query_index >= data.num_nodes:
        raise ValueError(f"node-index must be in [0, {data.num_nodes})")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    relation_perturbation = checkpoint.get("relation_perturbation")
    if relation_perturbation:
        # Replay the exact in-memory graph transformation used for training.
        # This matters for ablated checkpoints; shuffled relation labels are
        # deliberately not interpretable as the source graph's semantics.
        apply_relation_controls(
            data,
            relation_ids_to_drop=relation_perturbation.get("dropped_relation_ids", ()),
            relation_to_id=metadata["relation_to_id"],
            random_edge_drop_pairs=relation_perturbation.get("random_edge_drop_pairs", 0),
            random_edge_instance_pairs=relation_perturbation.get("random_edge_instance_pairs", 0),
            random_edge_drop_seed=relation_perturbation.get("random_edge_drop_seed"),
            shuffle_relation_types=relation_perturbation.get("relation_type_shuffle", False),
            shuffle_seed=relation_perturbation.get("relation_type_shuffle_seed"),
        )
    model, feature_schema = restore_model(checkpoint, device)
    num_layers = int(checkpoint.get("model_config", {}).get("num_layers", 2))
    default_fanouts = "20,10" if num_layers == 2 else ",".join(["20"] * num_layers)
    fanouts = parse_fanouts(args.num_neighbors or default_fanouts, num_layers)
    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([query_index]),
        num_neighbors=fanouts,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader)).to(device)
    if "occupation_unknown_ids" not in metadata:
        raise ValueError("This checkpoint predates transductive occupation masking; regenerate data and retrain")
    features = feature_inputs(batch, feature_schema)
    for name, unknown_id in metadata["occupation_unknown_ids"].items():
        if name in features:
            features[name] = features[name].clone()
            features[name][:batch.batch_size] = int(unknown_id)
    with torch.no_grad():
        logits, explanation = model(
            features,
            batch.edge_index,
            batch.edge_type,
            return_attention_weights=True,
        )
        probability = logits[0].softmax(dim=-1)

    id_to_label = {index: label for label, index in metadata["label_to_id"].items()}
    id_to_relation = {index: relation for relation, index in metadata["relation_to_id"].items()}
    target_label = int(data.y[query_index])
    prediction = int(probability.argmax())
    global_ids = batch.n_id.cpu()
    records = []
    for layer_info in explanation["attention_layers"]:
        edge_index = layer_info["edge_index"].cpu()
        alpha = layer_info["alpha"].mean(dim=-1).cpu()
        edge_types = attention_relation_ids(layer_info).cpu()
        for edge, score, relation_id in zip(edge_index.t(), alpha, edge_types):
            source_index, destination_index = (int(edge[0]), int(edge[1]))
            source_global = int(global_ids[source_index])
            destination_global = int(global_ids[destination_index])
            records.append({
                "layer": layer_info["layer"],
                "source_index": source_global,
                "source_id": node_ids[source_global],
                "relation_id": int(relation_id),
                "relation": id_to_relation.get(int(relation_id), "__self_loop__"),
                "target_index": destination_global,
                "target_id": node_ids[destination_global],
                "attention_mean": float(score),
                "is_query_destination": destination_global == query_index,
            })

    # Alpha is normalised within a target's sampled neighbourhood. Ranking only
    # incoming edges to the queried person avoids comparing incomparable targets.
    target_edges = [record for record in records if record["is_query_destination"]]
    target_edges.sort(key=lambda record: record["attention_mean"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{node_ids[query_index]}_attention"
    fields = list(records[0]) if records else ["layer", "source_index", "source_id", "relation_id", "relation", "target_index", "target_id", "attention_mean", "is_query_destination"]
    with (output_dir / f"{stem}_all_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / f"{stem}_top_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(target_edges[:args.top_k])

    summary = {
        "node_index": query_index,
        "node_id": node_ids[query_index],
        "true_label": id_to_label.get(target_label, "unlabeled"),
        "prediction": id_to_label[prediction],
        "prediction_confidence": float(probability[prediction]),
        "feature_gates": {
            name: float(values[0].cpu()) for name, values in explanation["feature_gates"].items()
        },
        "incoming_attention_edges": len(target_edges),
        "note": "Attention ranks candidates only; run edge deletion tests before causal claims.",
    }
    if relation_perturbation and relation_perturbation.get("relation_type_shuffle"):
        summary["note"] += " Relation IDs were shuffled for this checkpoint, so exported relation labels are model assignments rather than source semantics."
    with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
