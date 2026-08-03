#!/usr/bin/env python3
"""Legacy-style Contemporary occupation baseline with FastRGCN.

This is deliberately a *comparison* script, not the maintained experiment
pipeline.  It keeps the important mechanics of ``original_baseline``:

* one directed full graph (no reverse edges and no deduplication),
* an 80/20 random node holdout,
* occupation as the only node feature,
* target occupations hidden for all holdout nodes and for the current
  training batch,
* two full-graph relational convolutions for every random mini-batch, and
* selection of the best epoch using that same holdout set.

The latter is intentionally optimistic and should only be used when checking
the historical baseline.  The maintained ``run.py prepare/train`` workflow
has a separate validation set and is the appropriate protocol for new claims.

Run from the repository root, for example:

    python legacy/contemporary_fastrgcn_baseline.py \
      --born-after 1951 --target-level 3 --device cuda

Use ``--born-after 1940`` for the broader contemporary definition requested
for the comparison.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.nn import FastRGCNConv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.extended import ExtendedGraphLoader  # noqa: E402  (repository-local import)


@dataclass(frozen=True)
class GraphAudit:
    source_nodes: int
    source_directed_edges: int
    candidate_period_nodes: int
    candidate_period_directed_edges: int
    directed_edges_after_l1_filter: int
    active_nodes_after_l1_filter: int
    relations_after_l1_filter: int
    target_classes_including_unknown: int
    target_classes_retained: int
    target_unknown_nodes: int
    supervised_nodes: int
    ignored_rare_or_unknown_nodes: int


class LegacyFastRGCN(nn.Module):
    """The original simple embedding + R-GCN architecture, using FastRGCN."""

    def __init__(
        self,
        num_occupation_classes: int,
        num_relations: int,
        hidden_dim: int,
        dropout: float,
        num_bases: int,
    ) -> None:
        super().__init__()
        self.occupation_embedding = nn.Embedding(num_occupation_classes, hidden_dim)
        self.conv1 = FastRGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.classifier = nn.Linear(hidden_dim, num_occupation_classes)
        self.dropout = dropout

    def forward(
        self, occupation: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.occupation_embedding(occupation)
        hidden = F.relu(self.conv1(hidden, edge_index, edge_type))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.conv2(hidden, edge_index, edge_type))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.classifier(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(REPOSITORY_ROOT / "Q_R_Q_extended.txt"))
    parser.add_argument(
        "--output-dir",
        default=str(REPOSITORY_ROOT / "legacy" / "runs" / "contemporary_fastrgcn_level3"),
    )
    parser.add_argument(
        "--born-after",
        type=float,
        default=1951,
        help="Strict birth-year cutoff. Default >1951 approximately matches the screenshot scale.",
    )
    parser.add_argument("--target-level", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=20,
        help=(
            "Keep only target occupations with this many graph-node occurrences for loss/evaluation. "
            "Use 1 for the fully unfiltered historical implementation."
        ),
    )
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


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
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return device


def to_local_graph(
    nodes: pd.DataFrame, edges: pd.DataFrame, target_level: int, min_class_count: int
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int, int, int]:
    """Use LabelEncoder-equivalent lexical IDs, as the old script did."""
    active_global_ids = np.unique(
        np.concatenate([edges["source_id"].to_numpy(), edges["target_id"].to_numpy()])
    )
    active_nodes = nodes.iloc[active_global_ids].copy()
    active_nodes["_global_id"] = active_nodes.index
    active_nodes = active_nodes.sort_values("node_id", kind="stable").reset_index(drop=True)
    global_to_local = {
        int(global_id): local_id for local_id, global_id in enumerate(active_nodes["_global_id"])
    }

    relation_names = sorted(edges["relation"].unique().tolist())
    relation_to_id = {name: relation_id for relation_id, name in enumerate(relation_names)}
    source = edges["source_id"].map(global_to_local).to_numpy(dtype=np.int64)
    target = edges["target_id"].map(global_to_local).to_numpy(dtype=np.int64)
    relation = edges["relation"].map(relation_to_id).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.vstack([source, target])).long()
    edge_type = torch.from_numpy(relation).long()

    label_name = f"occupation_level{target_level}"
    raw_labels = active_nodes[label_name].fillna("unknown").astype(str)
    counts = raw_labels.value_counts()
    retained = sorted(counts[counts >= min_class_count].index.tolist())
    # The unknown ID represents both deliberately hidden target occupations and
    # labels excluded by the frequency threshold.  With --min-class-count 1,
    # any literal missing L2/L3 value is again supervised exactly as in legacy.
    label_names = ["unknown"] + [label for label in retained if label != "unknown"]
    label_to_id = {name: class_id for class_id, name in enumerate(label_names)}
    unknown_id = label_to_id["unknown"]
    supervised = raw_labels.isin(retained).to_numpy()
    occupations = raw_labels.map(label_to_id).fillna(unknown_id).to_numpy(dtype=np.int64)
    labels = occupations.copy()
    return (
        active_nodes,
        edge_index,
        edge_type,
        occupations,
        supervised,
        unknown_id,
        len(retained),
        int((raw_labels == "unknown").sum()),
    )


def prepare_graph(input_path: str, born_after: float, target_level: int, min_class_count: int):
    """Replicate the legacy ordering: time-induced graph, then L1 edge filter."""
    tables = ExtendedGraphLoader(input_path).read()
    nodes = tables.nodes
    # ``ExtendedGraphLoader`` creates reverse edges for the maintained pipeline.
    # Historical R-GCN used only the original export direction.
    source_edges = tables.edges.loc[~tables.edges["relation"].str.endswith("__rev")].copy()
    birth_year = pd.to_numeric(nodes["birth_year"], errors="coerce")
    period_mask = birth_year > born_after
    period_ids = set(nodes.index[period_mask])
    period_edges = source_edges.loc[
        source_edges["source_id"].isin(period_ids) & source_edges["target_id"].isin(period_ids)
    ].copy()

    has_level1 = nodes["occupation_level1"].notna().to_numpy()
    complete_l1_edges = period_edges.loc[
        has_level1[period_edges["source_id"].to_numpy()]
        & has_level1[period_edges["target_id"].to_numpy()]
    ].copy()
    if complete_l1_edges.empty:
        raise RuntimeError("No edges remain after the contemporary and Level-1 filters")

    (
        active_nodes,
        edge_index,
        edge_type,
        occupations,
        supervised,
        unknown_id,
        retained_class_count,
        raw_unknown_count,
    ) = to_local_graph(
        nodes, complete_l1_edges, target_level, min_class_count
    )
    audit = GraphAudit(
        source_nodes=len(nodes),
        source_directed_edges=len(source_edges),
        candidate_period_nodes=int(period_mask.sum()),
        candidate_period_directed_edges=len(period_edges),
        directed_edges_after_l1_filter=len(complete_l1_edges),
        active_nodes_after_l1_filter=len(active_nodes),
        relations_after_l1_filter=int(edge_type.max().item() + 1),
        target_classes_including_unknown=int(occupations.max() + 1),
        target_classes_retained=retained_class_count,
        target_unknown_nodes=raw_unknown_count,
        supervised_nodes=int(supervised.sum()),
        ignored_rare_or_unknown_nodes=int((~supervised).sum()),
    )
    return active_nodes, edge_index, edge_type, occupations, supervised, unknown_id, audit


def sample_batch(node_indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if len(node_indices) < batch_size:
        raise ValueError(
            f"Cannot draw batch_size={batch_size} without replacement from only {len(node_indices)} nodes"
        )
    return rng.choice(node_indices, size=batch_size, replace=False)


def masked_occupations(
    visible_occupations: torch.Tensor, batch_nodes: np.ndarray, unknown_id: int
) -> torch.Tensor:
    occupations = visible_occupations.clone()
    occupations[torch.as_tensor(batch_nodes, device=occupations.device)] = unknown_id
    return occupations


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_nodes: np.ndarray,
    visible_occupations: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    unknown_id: int,
    batch_size: int,
    batches_per_epoch: int,
    rng: np.random.Generator,
) -> float:
    model.train()
    losses: List[float] = []
    for _ in range(batches_per_epoch):
        batch_nodes = sample_batch(train_nodes, batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        logits = model(masked_occupations(visible_occupations, batch_nodes, unknown_id), edge_index, edge_type)
        batch_index = torch.as_tensor(batch_nodes, device=labels.device)
        loss = F.cross_entropy(logits[batch_index], labels[batch_index])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    nodes: np.ndarray,
    visible_occupations: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    unknown_id: int,
    batch_size: int,
    batches: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Use the legacy random 100-batch evaluation, not a full held-out pass."""
    model.eval()
    losses, predictions, truths = [], [], []
    for _ in range(batches):
        batch_nodes = sample_batch(nodes, batch_size, rng)
        logits = model(masked_occupations(visible_occupations, batch_nodes, unknown_id), edge_index, edge_type)
        batch_index = torch.as_tensor(batch_nodes, device=labels.device)
        batch_logits = logits[batch_index]
        batch_labels = labels[batch_index]
        losses.append(float(F.cross_entropy(batch_logits, batch_labels).item()))
        predictions.append(batch_logits.argmax(dim=1).cpu().numpy())
        truths.append(batch_labels.cpu().numpy())

    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "evaluated_samples": int(len(y_true)),
    }


def save_split(
    active_nodes: pd.DataFrame,
    train_nodes: Iterable[int],
    test_nodes: Iterable[int],
    output_dir: Path,
) -> None:
    training = set(int(node) for node in train_nodes)
    held_out = set(int(node) for node in test_nodes)
    with (output_dir / "split_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_index", "node_id", "split"])
        writer.writeheader()
        for index, node_id in enumerate(active_nodes["node_id"]):
            writer.writerow({
                "node_index": index,
                "node_id": node_id,
                "split": "test" if index in held_out else "train" if index in training else "ignored",
            })


def main() -> None:
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("test-ratio must be between 0 and 1")
    if args.min_class_count < 1:
        raise ValueError("min-class-count must be at least 1")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_nodes, edge_index, edge_type, occupations_np, supervised, unknown_id, audit = prepare_graph(
        args.input, args.born_after, args.target_level, args.min_class_count
    )
    node_count = len(active_nodes)
    if node_count < args.batch_size:
        raise ValueError(f"Only {node_count} active nodes remain; lower --batch-size")
    rng = np.random.default_rng(args.seed)
    supervised_nodes = np.flatnonzero(supervised)
    test_count = int(len(supervised_nodes) * args.test_ratio)
    test_nodes = np.sort(rng.choice(supervised_nodes, size=test_count, replace=False))
    train_nodes = np.setdiff1d(supervised_nodes, test_nodes, assume_unique=True)
    if len(train_nodes) < args.batch_size or len(test_nodes) < args.batch_size:
        raise ValueError("train/test split is smaller than --batch-size; lower the batch size")

    labels = torch.as_tensor(occupations_np, dtype=torch.long, device=device)
    visible_occupations = labels.clone()
    visible_occupations[torch.as_tensor(test_nodes, device=device)] = unknown_id
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    model = LegacyFastRGCN(
        num_occupation_classes=audit.target_classes_including_unknown,
        num_relations=audit.relations_after_l1_filter,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_bases=args.num_bases,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(json.dumps({
        "mode": "legacy_full_graph_fastrgcn",
        "target_level": args.target_level,
        "born_after_strict": args.born_after,
        "device": str(device),
        **asdict(audit),
        "train_nodes": int(len(train_nodes)),
        "test_nodes": int(len(test_nodes)),
    }, ensure_ascii=False, indent=2))

    best_score = float("-inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, optimizer, train_nodes, visible_occupations, labels, edge_index, edge_type,
            unknown_id, args.batch_size, args.batches_per_epoch, rng,
        )
        # This is deliberately the historical selection error: the holdout is
        # sampled for every epoch and its Macro-F1 chooses the checkpoint.
        holdout = evaluate(
            model, test_nodes, visible_occupations, labels, edge_index, edge_type, unknown_id,
            args.batch_size, args.eval_batches, rng,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "holdout": holdout}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if holdout["macro_f1"] > best_score:
            best_score = holdout["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    final_holdout = evaluate(
        model, test_nodes, visible_occupations, labels, edge_index, edge_type, unknown_id,
        args.batch_size, args.eval_batches, rng,
    )
    manifest = {
        "script": Path(__file__).name,
        "protocol": "legacy-style: same random 80/20 holdout selects the best epoch and reports final metrics",
        "only_features": [f"occupation_level{args.target_level}"],
        "reverse_edges_added": False,
        "deduplicated": False,
        "audit": asdict(audit),
        "run_config": vars(args),
        "split": {
            "train_nodes": int(len(train_nodes)),
            "test_nodes": int(len(test_nodes)),
            "ignored_nodes": int((~supervised).sum()),
        },
        "best_holdout_macro_f1_during_selection": best_score,
        "final_holdout": final_holdout,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (output_dir / "graph_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"audit": asdict(audit), "run_config": vars(args)}, handle, ensure_ascii=False, indent=2)
    torch.save({"state_dict": model.state_dict(), "manifest": manifest}, output_dir / "best_model.pt")
    save_split(active_nodes, train_nodes, test_nodes, output_dir)
    print(json.dumps({
        "best_holdout_macro_f1_during_selection": best_score,
        "final_holdout": final_holdout,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
