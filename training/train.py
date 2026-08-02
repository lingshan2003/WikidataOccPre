#!/usr/bin/env python3
"""Train a registered relational GNN on a prepared ``graph_data.pt`` artifact."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch_geometric.loader import NeighborLoader

from models import FeatureSpec, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/graph_data.pt")
    parser.add_argument("--output-dir", default="runs/rgat_level3")
    parser.add_argument("--model", choices=["rgcn", "rgat"], default="rgat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="20,10", help="One fan-out per GNN layer")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--attention-dropout", type=float, default=0.10)
    parser.add_argument("--num-bases", type=int, default=30, help="R-GCN basis decomposition count")
    parser.add_argument(
        "--rgcn-backend",
        choices=["fast", "standard"],
        default="fast",
        help="FastRGCNConv trades more memory for speed; ignored by R-GAT",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--early-stop-metric",
        choices=["val_loss", "macro_f1"],
        default="val_loss",
        help="Metric used to select the checkpoint and trigger early stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.002,
        help="Minimum monitor improvement required to reset early-stopping patience",
    )
    parser.add_argument("--lr-patience", type=int, default=3, help="Bad val-loss epochs before halving LR")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--class-weight", action="store_true", help="Use inverse-frequency loss weights")
    return parser.parse_args()


def parse_fanouts(value: str) -> List[int]:
    try:
        fanouts = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("num-neighbors must look like '20,10'") from error
    if not fanouts or any(item < -1 for item in fanouts):
        raise ValueError("Each fan-out must be -1 or a non-negative integer")
    return fanouts


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


def make_loader(data, mask: torch.Tensor, fanouts: Sequence[int], args: argparse.Namespace, shuffle: bool):
    workers = max(0, args.num_workers)
    return NeighborLoader(
        data,
        input_nodes=mask,
        num_neighbors=list(fanouts),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def batch_features(batch, feature_schema: Dict) -> Dict[str, torch.Tensor]:
    return {name: getattr(batch, name) for name in feature_schema if hasattr(batch, name)}


def train_epoch(model, loader, optimizer, device, feature_schema, class_weights=None) -> float:
    model.train()
    total_loss = 0.0
    seed_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features(batch, feature_schema), batch.edge_index, batch.edge_type)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        loss = F.cross_entropy(seed_logits, seed_labels, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        seed_count += batch.batch_size
    return total_loss / max(seed_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, feature_schema) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    all_labels, all_predictions, all_confidences, all_node_ids = [], [], [], []
    total_loss = 0.0
    seed_count = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch_features(batch, feature_schema), batch.edge_index, batch.edge_type)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        probabilities = seed_logits.softmax(dim=-1)
        total_loss += F.cross_entropy(seed_logits, seed_labels).item() * batch.batch_size
        seed_count += batch.batch_size
        all_labels.append(seed_labels.cpu())
        all_predictions.append(probabilities.argmax(dim=-1).cpu())
        all_confidences.append(probabilities.max(dim=-1).values.cpu())
        all_node_ids.append(batch.n_id[:batch.batch_size].cpu())

    labels = torch.cat(all_labels).numpy()
    predictions = torch.cat(all_predictions).numpy()
    metrics = {
        "loss": total_loss / max(seed_count, 1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    metrics["macro_precision"] = float(precision)
    metrics["macro_recall"] = float(recall)
    return metrics, {
        "node_id": torch.cat(all_node_ids).numpy(),
        "label": labels,
        "prediction": predictions,
        "confidence": torch.cat(all_confidences).numpy(),
    }


def compute_class_weights(data, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(data.y[data.train_mask], minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1) * num_classes)
    return weights.to(device)


def save_predictions(predictions: Dict[str, np.ndarray], metadata: Dict, data_path: Path, output_path: Path) -> None:
    node_table = None
    nodes_path = data_path.parent / "nodes.csv"
    if nodes_path.exists():
        # Loading only one column avoids a 20MB dataframe during final export.
        import pandas as pd
        node_table = pd.read_csv(nodes_path, usecols=["node_id"])
    id_to_label = {index: label for label, index in metadata["label_to_id"].items()}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_index", "node_id", "true_label", "prediction", "confidence"],
        )
        writer.writeheader()
        for index, label, prediction, confidence in zip(
            predictions["node_id"],
            predictions["label"],
            predictions["prediction"],
            predictions["confidence"],
        ):
            writer.writerow({
                "node_index": int(index),
                "node_id": node_table.iloc[int(index), 0] if node_table is not None else "",
                "true_label": id_to_label[int(label)],
                "prediction": id_to_label[int(prediction)],
                "confidence": float(confidence),
            })


def main() -> None:
    args = parse_args()
    fanouts = parse_fanouts(args.num_neighbors)
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    data, metadata = bundle["data"], bundle["metadata"]
    if len(fanouts) != 2:
        raise ValueError("The current RelationalGATClassifier has two layers; use exactly two fan-outs")
    if not all(hasattr(data, key) for key in ("country", "temporal", "edge_type", "train_mask", "val_mask", "test_mask")):
        raise ValueError("Prepared graph does not have the required features and masks")

    specs = {
        name: FeatureSpec(
            kind=definition["kind"],
            cardinality=definition.get("cardinality"),
            input_dim=definition.get("input_dim", 1),
        )
        for name, definition in metadata["feature_schema"].items()
    }
    model = build_model(
        args.model,
        num_relations=metadata["num_relations"],
        num_classes=metadata["num_classes"],
        feature_specs=specs,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        num_bases=args.num_bases,
        rgcn_backend=args.rgcn_backend,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    class_weights = compute_class_weights(data, metadata["num_classes"], device) if args.class_weight else None

    train_loader = make_loader(data, data.train_mask, fanouts, args, shuffle=True)
    val_loader = make_loader(data, data.val_mask, fanouts, args, shuffle=False)
    test_loader = make_loader(data, data.test_mask, fanouts, args, shuffle=False)
    is_loss_monitor = args.early_stop_metric == "val_loss"
    best_monitor = float("inf") if is_loss_monitor else float("-inf")
    best_val_f1, best_state, stale_epochs = float("-inf"), None, 0
    history = []
    feature_schema = metadata["feature_schema"]
    backend_info = f"; rgcn_backend: {args.rgcn_backend}" if args.model == "rgcn" else ""
    print(f"Model: {args.model}{backend_info}; device: {device}; train/val/test: {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, feature_schema, class_weights)
        val_metrics, _ = evaluate(model, val_loader, device, feature_schema)
        scheduler.step(val_metrics["loss"])
        monitor_value = val_metrics["loss"] if is_loss_monitor else val_metrics["macro_f1"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "early_stop_metric": args.early_stop_metric,
            "monitor_value": monitor_value,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        best_val_f1 = max(best_val_f1, val_metrics["macro_f1"])
        improved = (
            monitor_value < best_monitor - args.min_delta
            if is_loss_monitor
            else monitor_value > best_monitor + args.min_delta
        )
        if improved:
            best_monitor = monitor_value
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping after {epoch} epochs: "
                    f"{args.early_stop_metric} did not improve by {args.min_delta} "
                    f"for {args.patience} epochs."
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    test_metrics, test_predictions = evaluate(model, test_loader, device, feature_schema)
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "model_name": args.model,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "branch_dim": args.branch_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "attention_dropout": args.attention_dropout,
            "num_bases": args.num_bases,
            "rgcn_backend": args.rgcn_backend,
        },
        "selection_metric": args.early_stop_metric,
        "best_selection_metric": best_monitor,
        "best_val_macro_f1_seen": best_val_f1,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "selection_metric": args.early_stop_metric,
            "best_selection_metric": best_monitor,
            "best_val_macro_f1_seen": best_val_f1,
            "test": test_metrics,
            "history": history,
        }, handle, indent=2)
    save_predictions(test_predictions, metadata, data_path, output_dir / "test_predictions.csv")
    print(json.dumps({
        "selection_metric": args.early_stop_metric,
        "best_selection_metric": best_monitor,
        "best_val_macro_f1_seen": best_val_f1,
        "test": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
