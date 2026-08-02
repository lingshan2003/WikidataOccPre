#!/usr/bin/env python3
"""Train the independent heterogeneous Person--Occupation link predictor."""

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from .model import HeteroOccupationLinkPredictor, OCCUPATION, PERSON
from .prepare import HAS_OCCUPATION, REV_HAS_OCCUPATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="link_artifacts/level3/hetero_graph.pt")
    parser.add_argument("--output-dir", default="link_runs/level3_hgt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="15,10")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--branch-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def fanouts(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",")]
    if len(result) != 2 or any(item < -1 for item in result):
        raise ValueError("--num-neighbors needs exactly two values, e.g. 15,10")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_for(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def loader(data, pairs: torch.Tensor, args, shuffle: bool, negatives: bool):
    kwargs = {}
    if negatives:
        kwargs["neg_sampling"] = NegativeSampling("binary", amount=args.negative_ratio)
    workers = max(0, args.num_workers)
    return LinkNeighborLoader(
        data,
        num_neighbors=fanouts(args.num_neighbors),
        edge_label_index=(HAS_OCCUPATION, pairs),
        edge_label=torch.ones(pairs.size(1), dtype=torch.float),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )


def hide_supervision_edges(batch) -> None:
    """Remove current positive label edges from the sampled message graph.

    This is the link-prediction analogue of masking seed occupations in node
    classification. Other observed training Person--Occupation edges remain.
    """
    store = batch[HAS_OCCUPATION]
    positive = store.edge_label_index[:, store.edge_label > 0]
    if positive.numel() == 0:
        return
    occupation_count = batch[OCCUPATION].num_nodes
    positive_codes = positive[0] * occupation_count + positive[1]
    edge_codes = store.edge_index[0] * occupation_count + store.edge_index[1]
    keep = ~torch.isin(edge_codes, positive_codes)
    store.edge_index = store.edge_index[:, keep]
    reverse = batch[REV_HAS_OCCUPATION]
    reverse_codes = reverse.edge_index[1] * occupation_count + reverse.edge_index[0]
    reverse_keep = ~torch.isin(reverse_codes, positive_codes)
    reverse.edge_index = reverse.edge_index[:, reverse_keep]


def pair_scores(model, batch):
    states = model.encode(batch)
    labels = batch[HAS_OCCUPATION]
    person_indices, local_occupation_indices = labels.edge_label_index
    global_occupation_ids = batch[OCCUPATION].n_id[local_occupation_indices]
    return model.score_pairs(states[PERSON][person_indices], global_occupation_ids), labels.edge_label


def train_epoch(model, train_loader, optimizer, device) -> float:
    model.train()
    total_loss, total_edges = 0.0, 0
    for batch in train_loader:
        batch = batch.to(device)
        hide_supervision_edges(batch)
        optimizer.zero_grad(set_to_none=True)
        scores, labels = pair_scores(model, batch)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_edges += labels.numel()
    return total_loss / max(total_edges, 1)


@torch.no_grad()
def evaluate(model, evaluation_loader, device) -> Dict[str, float]:
    model.eval()
    ranks = []
    for batch in evaluation_loader:
        batch = batch.to(device)
        hide_supervision_edges(batch)
        states = model.encode(batch)
        labels = batch[HAS_OCCUPATION]
        person_indices, local_occupation_indices = labels.edge_label_index
        true_occupations = batch[OCCUPATION].n_id[local_occupation_indices]
        scores = model.score_all_occupations(states[PERSON][person_indices])
        true_scores = scores.gather(1, true_occupations.unsqueeze(1))
        ranks.append((scores >= true_scores).sum(dim=1).cpu())
    rank = torch.cat(ranks).float()
    return {
        "mrr": float((1.0 / rank).mean()),
        "hits_at_1": float((rank <= 1).float().mean()),
        "hits_at_3": float((rank <= 3).float().mean()),
        "hits_at_10": float((rank <= 10).float().mean()),
        "mean_rank": float(rank.mean()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_for(args.device)
    bundle = torch.load(args.data, map_location="cpu", weights_only=False)
    data, splits, metadata = bundle["data"], bundle["splits"], bundle["metadata"]
    if tuple(metadata["link_edge_type"]) != HAS_OCCUPATION:
        raise ValueError("This artifact does not use the expected Person--Occupation link type")
    model = HeteroOccupationLinkPredictor(
        data.metadata(),
        num_occupations=metadata["num_occupations"],
        country_cardinality=len(metadata["country_to_id"]),
        temporal_dim=int(data[PERSON].temporal.size(1)),
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = loader(data, splits["train_pos"], args, shuffle=True, negatives=True)
    val_loader = loader(data, splits["val_pos"], args, shuffle=False, negatives=False)
    test_loader = loader(data, splits["test_pos"], args, shuffle=False, negatives=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mrr, best_state, stale, history = float("-inf"), None, 0, []
    print(f"Model: hgt_distmult; device: {device}; train/val/test links: {splits['train_pos'].size(1)}/{splits['val_pos'].size(1)}/{splits['test_pos'].size(1)}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in validation.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if validation["mrr"] > best_mrr:
            best_mrr, best_state, stale = validation["mrr"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs: validation MRR did not improve for {args.patience} epochs.")
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata, "model_config": vars(args)}, output_dir / "best_model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_val_mrr": best_mrr, "test": test_metrics, "history": history}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"best_val_mrr": best_mrr, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
