#!/usr/bin/env python3
"""Train a registered relational GNN on a prepared ``graph_data.pt`` artifact."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch_geometric.loader import NeighborLoader

from models import build_feature_specs, build_model
from training.relation_controls import (
    apply_relation_controls,
    count_relation_pairs,
    parse_selection,
    resolve_ablation,
    select_relation_pairs,
)
from training.birth_cohorts import load_artifact_birth_cohorts, load_birth_cohort_config
from training.tie_taxonomy import (
    DEFAULT_TIE_TAXONOMY_PATH,
    load_tie_taxonomy,
    parse_tie_group_selection,
    resolve_tie_ablation,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/graph_data.pt")
    parser.add_argument("--output-dir", default="runs/rgat_level3")
    parser.add_argument("--model", choices=["rgcn", "rgat", "compgcn"], default="rgat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--train-mode",
        choices=["sampled", "full"],
        default="sampled",
        help="Use NeighborLoader mini-batches or a true full-graph forward/backward pass",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="20,10", help="One fan-out per GNN layer")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of message-passing layers; RGAT supports one or more layers",
    )
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
    parser.add_argument(
        "--compgcn-composition",
        choices=["mult", "sub"],
        default="mult",
        help="How CompGCN combines a source-node state with its relation embedding",
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
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="Legacy alias for --loss inverse_frequency; cannot be combined with another --loss",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "inverse_frequency", "class_balanced", "logit_adjusted"],
        default="cross_entropy",
        help="Long-tail training objective; default preserves the original cross-entropy protocol",
    )
    parser.add_argument(
        "--class-balanced-beta",
        type=float,
        default=0.9999,
        help="Effective-number beta for --loss class_balanced (must be in [0, 1))",
    )
    parser.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=1.0,
        help="Prior-strength tau for --loss logit_adjusted",
    )
    parser.add_argument(
        "--train-root-sampling",
        choices=["uniform", "class_balanced"],
        default="uniform",
        help="Sample training seed nodes uniformly or balance their target classes each epoch",
    )
    parser.add_argument(
        "--occupation-feature-levels",
        default="1,2,3",
        help="Comma-separated neighbour occupation levels to expose (1,2,3 or none)",
    )
    parser.add_argument(
        "--occupation-representation",
        choices=["categorical", "semantic"],
        default="categorical",
        help="Use trainable categorical occupations or fixed semantic occupation vectors",
    )
    parser.add_argument(
        "--auxiliary-features",
        default="country,temporal",
        help="Comma-separated non-occupation features to expose (country,temporal or none)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["selected", "structural"],
        default="selected",
        help="Use selected attributes, or one shared constant vector for a relation-and-structure-only baseline",
    )
    parser.add_argument(
        "--drop-relation-groups",
        default="none",
        help="Comma-separated groups to remove: kinship, education_mentorship, professional_collaboration, influence_succession, religious, other, or none",
    )
    parser.add_argument(
        "--drop-relations",
        default="none",
        help="Comma-separated original relation names to remove in both directions, or none",
    )
    parser.add_argument(
        "--tie-taxonomy",
        default=str(DEFAULT_TIE_TAXONOMY_PATH),
        help="Versioned inherited/acquired taxonomy JSON used for audit provenance and --drop-tie-groups",
    )
    parser.add_argument(
        "--drop-tie-groups",
        default="none",
        help="Comma-separated inherited/acquired categories to remove in both directions, or none",
    )
    parser.add_argument(
        "--shuffle-relation-types",
        action="store_true",
        help="Permute relation IDs across retained edges while preserving topology and relation frequencies",
    )
    random_drop = parser.add_mutually_exclusive_group()
    random_drop.add_argument(
        "--random-edge-drop-pairs",
        type=int,
        default=None,
        help="Uniformly remove this many base-relation/undirected edge pairs (and both directed counterparts)",
    )
    random_drop.add_argument(
        "--match-random-drop-to-relation-groups",
        default=None,
        help="Remove a uniform random set of as many relation pairs as the named groups contain; e.g. kinship",
    )
    random_drop.add_argument(
        "--match-random-drop-to-tie-groups",
        default=None,
        help="Remove a uniform random set of as many relation pairs as inherited or acquired contains",
    )
    parser.add_argument(
        "--edge-cohort-config",
        default=None,
        help=(
            "Optional versioned birth-cohort JSON. With a tie ablation or matched random control, "
            "restrict deletion candidates to relation pairs incident to this cohort."
        ),
    )
    parser.add_argument(
        "--edge-cohort-id",
        default=None,
        help="Birth-cohort ID from --edge-cohort-config whose incident relation pairs are eligible for deletion",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["sampled", "full"],
        default="sampled",
        help="Use sampled neighbourhoods or deterministic full-graph inference for validation/test",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Save the validation-selected checkpoint without evaluating the test split",
    )
    return parser.parse_args()


def parse_fanouts(value: str) -> List[int]:
    try:
        fanouts = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("num-neighbors must look like '20,10'") from error
    if not fanouts or any(item < -1 for item in fanouts):
        raise ValueError("Each fan-out must be -1 or a non-negative integer")
    return fanouts


def parse_occupation_levels(value: str) -> Tuple[int, ...]:
    if value.strip().lower() == "none":
        return ()
    try:
        levels = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("occupation-feature-levels must look like '1,2,3', '3', or 'none'") from error
    if not levels or any(level not in {1, 2, 3} for level in levels):
        raise ValueError("occupation-feature-levels may contain only 1, 2, and 3")
    return levels


def parse_auxiliary_features(value: str) -> Tuple[str, ...]:
    """Parse the optional ordinary attributes independently of occupations."""
    if value.strip().lower() == "none":
        return ()
    allowed = {"country", "temporal"}
    features = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not features or any(feature not in allowed for feature in features):
        raise ValueError("auxiliary-features may contain country, temporal, or none")
    return features


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


def make_loader(
    data,
    mask: torch.Tensor,
    fanouts: Sequence[int],
    args: argparse.Namespace,
    shuffle: bool,
    persistent_workers: bool = True,
):
    workers = max(0, args.num_workers)
    return NeighborLoader(
        data,
        input_nodes=mask,
        num_neighbors=list(fanouts),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0 and persistent_workers,
        pin_memory=torch.cuda.is_available(),
    )


def feature_inputs(graph, feature_schema: Dict) -> Dict[str, torch.Tensor]:
    """Collect model inputs and synthesize a shared structural baseline feature."""
    features = {}
    for name, definition in feature_schema.items():
        if definition["kind"] == "constant":
            features[name] = torch.zeros(
                graph.num_nodes,
                dtype=torch.long,
                device=graph.edge_index.device,
            )
        elif hasattr(graph, name):
            features[name] = getattr(graph, name)
        else:
            raise KeyError(f"Graph is missing required feature '{name}'")
    return features


def batch_features(batch, feature_schema: Dict, occupation_unknown_ids: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Hide all selected occupation levels of seed people currently predicted.

    The prepared graph exposes known occupations for training people only.
    NeighborLoader places this forward pass's seed people in the first
    ``batch.batch_size`` rows, so cloning and masking those rows prevents a
    target from seeing its own hierarchical occupation while retaining its
    neighbours' observed hierarchical occupations.
    """
    features = feature_inputs(batch, feature_schema)
    selected_occupation_features = [name for name in occupation_unknown_ids if name in features]
    for name in selected_occupation_features:
        occupation = features[name].clone()
        occupation[:batch.batch_size] = occupation_unknown_ids[name]
        features[name] = occupation
    return features


def select_feature_schema(
    metadata: Dict,
    occupation_levels: Tuple[int, ...],
    auxiliary_features: Tuple[str, ...],
    occupation_representation: str,
    feature_mode: str = "selected",
) -> Dict:
    """Keep exactly the requested observed occupations and ordinary attributes."""
    if feature_mode == "structural":
        return {"structural_constant": {"kind": "constant"}}
    selected = {}
    for name, definition in metadata["feature_schema"].items():
        if name.startswith("occupation_level"):
            if occupation_representation != "categorical":
                continue
            level = int(name.removeprefix("occupation_level"))
            if level not in occupation_levels:
                continue
        elif name == "occupation_semantic":
            if occupation_representation != "semantic":
                continue
        elif name not in auxiliary_features:
            continue
        selected[name] = definition
    if not selected:
        raise ValueError("At least one feature must be selected")
    return selected


def select_unknown_feature_ids(
    metadata: Dict, occupation_levels: Tuple[int, ...], occupation_representation: str
) -> Dict[str, int]:
    if occupation_representation == "categorical":
        return {
            f"occupation_level{level}": int(metadata["occupation_unknown_ids"][f"occupation_level{level}"])
            for level in occupation_levels
        }
    try:
        return {"occupation_semantic": int(metadata["occupation_unknown_ids"]["occupation_semantic"])}
    except KeyError as error:
        raise ValueError(
            "This artifact has no fixed semantic occupations; run `python run.py occupation-embed` first"
        ) from error


def semantic_provenance(metadata: Dict, occupation_representation: str) -> Dict:
    """Small, explicit record for metrics/checkpoints of a semantic run."""
    if occupation_representation != "semantic":
        return {}
    details = metadata.get("semantic_features", {}).get("occupation_semantic")
    if not details:
        raise ValueError("This semantic artifact is missing occupation semantic metadata")
    return {
        "semantic_model_name": details.get("model_name"),
        "semantic_model_revision": details.get("resolved_revision"),
        "semantic_prompt_fingerprint": details.get("prompt_fingerprint"),
        "semantic_source_artifact": details.get("source_artifact"),
    }


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mode: str,
    class_weights: Optional[torch.Tensor] = None,
    log_priors: Optional[torch.Tensor] = None,
    logit_adjustment_tau: float = 1.0,
) -> torch.Tensor:
    """Compute one explicit long-tail objective without changing evaluation metrics."""
    if loss_mode == "logit_adjusted":
        if log_priors is None:
            raise ValueError("Log priors are required for logit-adjusted loss")
        logits = logits + logit_adjustment_tau * log_priors.unsqueeze(0)
    return F.cross_entropy(logits, labels, weight=class_weights)


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    feature_schema,
    occupation_unknown_ids,
    loss_mode: str,
    class_weights=None,
    log_priors=None,
    logit_adjustment_tau: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    seed_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch_features(batch, feature_schema, occupation_unknown_ids), batch.edge_index, batch.edge_type
        )
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        loss = classification_loss(
            seed_logits,
            seed_labels,
            loss_mode,
            class_weights,
            log_priors,
            logit_adjustment_tau,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        seed_count += batch.batch_size
    return total_loss / max(seed_count, 1)


def train_full_graph_epoch(
    model,
    data,
    optimizer,
    feature_schema,
    loss_mode: str,
    class_weights=None,
    log_priors=None,
    logit_adjustment_tau: float = 1.0,
) -> float:
    """Run one true full-graph update for a leakage-safe non-occupation setup.

    A single forward cannot selectively hide each training seed's occupation
    while exposing that same occupation to every other training seed.  The
    caller therefore rejects occupation inputs for this mode; this keeps the
    full-batch experiment faithful instead of silently leaking targets.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(feature_inputs(data, feature_schema), data.edge_index, data.edge_type)
    labels = data.y[data.train_mask]
    loss = classification_loss(
        logits[data.train_mask], labels, loss_mode, class_weights, log_priors, logit_adjustment_tau
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, loader, device, feature_schema, occupation_unknown_ids) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    all_labels, all_predictions, all_confidences, all_node_ids = [], [], [], []
    total_loss = 0.0
    seed_count = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(
            batch_features(batch, feature_schema, occupation_unknown_ids), batch.edge_index, batch.edge_type
        )
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


@torch.no_grad()
def evaluate_full_graph(model, data, mask: torch.Tensor, feature_schema: Dict) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate a split with one deterministic full-graph forward pass.

    Prepared validation/test nodes already have every occupation level set to
    ``__UNKNOWN__``. Unlike training seeds, they therefore need no additional
    per-batch masking before full-graph inference.
    """
    model.eval()
    features = feature_inputs(data, feature_schema)
    logits = model(features, data.edge_index, data.edge_type)
    labels = data.y[mask]
    seed_logits = logits[mask]
    probabilities = seed_logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    labels_np, predictions_np = labels.cpu().numpy(), predictions.cpu().numpy()
    metrics = {
        "loss": float(F.cross_entropy(seed_logits, labels).item()),
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "macro_f1": float(f1_score(labels_np, predictions_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_np, predictions_np, average="weighted", zero_division=0)),
    }
    precision, recall, _, _ = precision_recall_fscore_support(
        labels_np, predictions_np, average="macro", zero_division=0
    )
    metrics["macro_precision"] = float(precision)
    metrics["macro_recall"] = float(recall)
    return metrics, {
        "node_id": mask.nonzero(as_tuple=False).view(-1).cpu().numpy(),
        "label": labels_np,
        "prediction": predictions_np,
        "confidence": probabilities.max(dim=-1).values.cpu().numpy(),
    }


def training_class_counts(data, num_classes: int) -> torch.Tensor:
    """Count only supervised training people; held-out labels never set priors."""
    return torch.bincount(data.y[data.train_mask], minlength=num_classes).float()


def loss_components(
    counts: torch.Tensor,
    loss_mode: str,
    class_balanced_beta: float,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return optional class weights and log priors for the selected objective."""
    if (counts <= 0).any():
        raise ValueError("Every retained class must have at least one training example")
    weights = None
    if loss_mode == "inverse_frequency":
        weights = counts.sum() / (counts * counts.numel())
    elif loss_mode == "class_balanced":
        if not 0 <= class_balanced_beta < 1:
            raise ValueError("--class-balanced-beta must be in [0, 1)")
        beta = torch.tensor(class_balanced_beta, dtype=counts.dtype)
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / weights.mean()
    priors = counts / counts.sum()
    return (
        weights.to(device) if weights is not None else None,
        priors.clamp_min(torch.finfo(priors.dtype).tiny).log().to(device),
    )


def class_balanced_train_nodes(data, num_classes: int, generator: torch.Generator) -> torch.Tensor:
    """Draw one epoch of roots with each target class equally likely in expectation."""
    train_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    labels = data.y[train_nodes]
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts <= 0).any():
        raise ValueError("Every retained class must have at least one training example")
    node_weights = counts.reciprocal()[labels]
    sampled_positions = torch.multinomial(
        node_weights, num_samples=train_nodes.numel(), replacement=True, generator=generator
    )
    return train_nodes[sampled_positions]


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
    occupation_levels = parse_occupation_levels(args.occupation_feature_levels)
    auxiliary_features = parse_auxiliary_features(args.auxiliary_features)
    drop_relation_groups = parse_selection(args.drop_relation_groups)
    drop_relations = parse_selection(args.drop_relations)
    drop_tie_groups = parse_tie_group_selection(args.drop_tie_groups)
    random_match_groups = (
        parse_selection(args.match_random_drop_to_relation_groups)
        if args.match_random_drop_to_relation_groups is not None else ()
    )
    random_match_tie_groups = (
        parse_tie_group_selection(args.match_random_drop_to_tie_groups)
        if args.match_random_drop_to_tie_groups is not None else ()
    )
    if args.class_weight and args.loss != "cross_entropy":
        raise ValueError("--class-weight is a legacy alias and cannot be combined with a non-default --loss")
    loss_mode = "inverse_frequency" if args.class_weight else args.loss
    uses_occupation_features = args.feature_mode == "selected" and (
        (args.occupation_representation == "categorical" and bool(occupation_levels))
        or args.occupation_representation == "semantic"
    )
    if args.train_mode == "full" and uses_occupation_features:
        raise ValueError(
            "True full-graph training cannot expose occupations without leaking each training seed's own "
            "label. Use --occupation-feature-levels none (or --feature-mode structural)."
        )
    if args.train_mode == "full" and args.eval_mode != "full":
        raise ValueError("True full-graph training requires --eval-mode full for a deterministic paired evaluation")
    if args.train_mode == "full" and args.train_root_sampling != "uniform":
        raise ValueError("--train-root-sampling applies only to sampled training")
    if args.feature_mode == "structural" and args.occupation_representation != "categorical":
        raise ValueError("--feature-mode structural does not use occupation representations; use categorical")
    if args.random_edge_drop_pairs is not None and args.random_edge_drop_pairs < 0:
        raise ValueError("--random-edge-drop-pairs must be non-negative")
    random_drop_requested = bool(
        args.random_edge_drop_pairs is not None or random_match_groups or random_match_tie_groups
    )
    if random_drop_requested and (
        drop_relation_groups or drop_relations or drop_tie_groups
    ):
        raise ValueError(
            "Random edge-drop controls are standalone comparisons; do not combine them with relation or tie ablations"
        )
    if drop_tie_groups and (drop_relation_groups or drop_relations):
        raise ValueError("--drop-tie-groups cannot be combined with --drop-relation-groups or --drop-relations")
    if random_drop_requested and args.shuffle_relation_types:
        raise ValueError("Random edge-drop controls should not be combined with --shuffle-relation-types")
    if bool(args.edge_cohort_config) != bool(args.edge_cohort_id):
        raise ValueError("--edge-cohort-config and --edge-cohort-id must be supplied together")
    if args.edge_cohort_config and not (drop_tie_groups or random_match_tie_groups):
        raise ValueError(
            "Cohort-restricted edge deletion currently supports --drop-tie-groups or "
            "--match-random-drop-to-tie-groups only"
        )
    if args.edge_cohort_config and (len(drop_tie_groups) > 1 or len(random_match_tie_groups) > 1):
        raise ValueError("Cohort-restricted edge deletion accepts exactly one tie group per run")
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(data_path, map_location="cpu", weights_only=False)
    # All relation perturbations happen to an in-memory copy.  A server run
    # must never overwrite the canonical artifact shared by comparisons.
    data, metadata = copy.deepcopy(bundle["data"]), bundle["metadata"]
    tie_taxonomy = load_tie_taxonomy(args.tie_taxonomy, metadata["relation_to_id"])
    cohort_node_mask = None
    cohort_manifest = None
    if args.edge_cohort_config:
        cohort_config = load_birth_cohort_config(args.edge_cohort_config)
        selected_cohort = cohort_config.cohort(args.edge_cohort_id)
        cohort_nodes = load_artifact_birth_cohorts(data_path, cohort_config, expected_nodes=data.num_nodes)
        cohort_node_mask = (cohort_nodes["birth_cohort"] == selected_cohort.identifier).to_numpy(dtype=bool)
        if not cohort_node_mask.any():
            raise ValueError(
                f"Birth cohort {selected_cohort.identifier!r} has no nodes in {data_path.parent / 'nodes.csv'}"
            )
        cohort_manifest = {
            **cohort_config.manifest(),
            "selected_cohort_id": selected_cohort.identifier,
            "selected_cohort_label": selected_cohort.label,
            "selected_cohort_node_count": int(cohort_node_mask.sum()),
            "edge_scope": "incident_to_selected_cohort",
        }
    if args.num_layers < 1:
        raise ValueError("--num-layers must be at least one")
    if len(fanouts) != args.num_layers:
        raise ValueError(
            f"--num-neighbors has {len(fanouts)} fan-outs but --num-layers is {args.num_layers}; "
            "provide exactly one fan-out per message-passing layer"
        )
    if args.model != "rgat" and args.num_layers != 2:
        raise ValueError("Only RGAT currently supports --num-layers other than 2")
    required_attributes = (
        "occupation_level1", "occupation_level2", "occupation_level3", "country", "temporal",
        "edge_type", "train_mask", "val_mask", "test_mask",
    )
    if not all(hasattr(data, key) for key in required_attributes) or "occupation_unknown_ids" not in metadata:
        raise ValueError(
            "Prepared graph predates hierarchical occupation masking; rerun `python run.py prepare` first"
        )
    if args.occupation_representation == "semantic" and not hasattr(data, "occupation_semantic"):
        raise ValueError(
            "This artifact has no semantic occupation IDs; run `python run.py occupation-embed` first"
        )

    feature_schema = select_feature_schema(
        metadata, occupation_levels, auxiliary_features, args.occupation_representation, args.feature_mode
    )
    occupation_unknown_ids = select_unknown_feature_ids(
        metadata, occupation_levels, args.occupation_representation
    )
    representation_provenance = semantic_provenance(metadata, args.occupation_representation)
    relation_pair_keys_to_drop = None
    random_edge_drop_candidate_pair_keys = None
    if drop_tie_groups:
        relation_ids_to_drop, dropped_base_relations = resolve_tie_ablation(
            tie_taxonomy, drop_tie_groups, metadata["relation_to_id"]
        )
        if cohort_node_mask is not None:
            relation_pair_keys_to_drop = select_relation_pairs(
                data, relation_ids_to_drop, metadata["relation_to_id"], cohort_node_mask
            )
            if not len(relation_pair_keys_to_drop):
                raise ValueError(
                    f"No {drop_tie_groups[0]} relation pairs are incident to cohort {args.edge_cohort_id!r}; "
                    "refusing a no-op cohort ablation"
                )
            relation_ids_to_drop = ()
    else:
        relation_ids_to_drop, dropped_base_relations = resolve_ablation(
            metadata["relation_to_id"], drop_relation_groups, drop_relations
        )
    matched_base_relations: Tuple[str, ...] = ()
    if random_match_tie_groups:
        matching_relation_ids, matched_base_relations = resolve_tie_ablation(
            tie_taxonomy, random_match_tie_groups, metadata["relation_to_id"]
        )
        random_edge_drop_pairs = count_relation_pairs(
            data, matching_relation_ids, metadata["relation_to_id"], cohort_node_mask
        )
        if cohort_node_mask is not None and not random_edge_drop_pairs:
            raise ValueError(
                f"No {random_match_tie_groups[0]} relation pairs are incident to cohort {args.edge_cohort_id!r}; "
                "refusing a no-op matched random control"
            )
        if cohort_node_mask is not None:
            all_relation_ids = tuple(int(value) for value in metadata["relation_to_id"].values())
            random_edge_drop_candidate_pair_keys = select_relation_pairs(
                data, all_relation_ids, metadata["relation_to_id"], cohort_node_mask
            )
    elif random_match_groups:
        matching_relation_ids, matched_base_relations = resolve_ablation(
            metadata["relation_to_id"], random_match_groups, ()
        )
        random_edge_drop_pairs = count_relation_pairs(
            data, matching_relation_ids, metadata["relation_to_id"]
        )
    else:
        random_edge_drop_pairs = args.random_edge_drop_pairs or 0
    relation_perturbation = apply_relation_controls(
        data,
        relation_ids_to_drop=relation_ids_to_drop,
        relation_pair_keys_to_drop=relation_pair_keys_to_drop,
        relation_to_id=metadata["relation_to_id"],
        random_edge_drop_pairs=random_edge_drop_pairs,
        random_edge_drop_seed=args.seed if random_edge_drop_pairs else None,
        random_edge_drop_candidate_pair_keys=random_edge_drop_candidate_pair_keys,
        shuffle_relation_types=args.shuffle_relation_types,
        shuffle_seed=args.seed if args.shuffle_relation_types else None,
    )
    relation_perturbation.update({
        "data": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "tie_taxonomy": tie_taxonomy.manifest(),
        "dropped_relation_groups": list(drop_relation_groups),
        "dropped_base_relations": list(dropped_base_relations),
        "random_drop_matched_relation_groups": list(random_match_groups),
        "random_drop_matched_base_relations": list(matched_base_relations),
        "dropped_tie_groups": list(drop_tie_groups),
        "random_drop_matched_tie_groups": list(random_match_tie_groups),
        "edge_cohort": cohort_manifest,
    })
    specs = build_feature_specs(feature_schema, metadata)
    model = build_model(
        args.model,
        num_relations=metadata["num_relations"],
        num_classes=metadata["num_classes"],
        feature_specs=specs,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        num_bases=args.num_bases,
        rgcn_backend=args.rgcn_backend,
        compgcn_composition=args.compgcn_composition,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    class_counts = training_class_counts(data, metadata["num_classes"])
    class_weights, log_priors = loss_components(
        class_counts, loss_mode, args.class_balanced_beta, device
    )

    train_loader = (
        make_loader(data, data.train_mask, fanouts, args, shuffle=True)
        if args.train_mode == "sampled" else None
    )
    val_loader = make_loader(data, data.val_mask, fanouts, args, shuffle=False) if args.eval_mode == "sampled" else None
    test_loader = make_loader(data, data.test_mask, fanouts, args, shuffle=False) if args.eval_mode == "sampled" else None
    if args.train_mode == "full":
        # Reuse one device copy for train/validation/test to keep the full
        # graph experiment within server memory.
        full_training_data = copy.deepcopy(data).to(device)
        full_evaluation_data = full_training_data
    else:
        full_training_data = None
        full_evaluation_data = copy.deepcopy(data).to(device) if args.eval_mode == "full" else None
    root_sampling_generator = torch.Generator().manual_seed(args.seed)
    is_loss_monitor = args.early_stop_metric == "val_loss"
    best_monitor = float("inf") if is_loss_monitor else float("-inf")
    best_patience_monitor = float("inf") if is_loss_monitor else float("-inf")
    best_val_f1, best_state, stale_epochs = float("-inf"), None, 0
    history = []
    if args.model == "rgcn":
        backend_info = f"; rgcn_backend: {args.rgcn_backend}"
    elif args.model == "compgcn":
        backend_info = f"; compgcn_composition: {args.compgcn_composition}"
    else:
        backend_info = ""
    level_info = (
        "+".join(str(level) for level in occupation_levels)
        if args.occupation_representation == "categorical" and occupation_levels
        else args.occupation_representation
    )
    auxiliary_info = "+".join(auxiliary_features) if auxiliary_features else "none"
    print(
        f"Model: {args.model}{backend_info}; occupation representation: {level_info}; "
        f"auxiliary features: {auxiliary_info}; feature mode: {args.feature_mode}; "
        f"loss: {loss_mode}; train mode: {args.train_mode}; root sampling: {args.train_root_sampling}; "
        f"eval mode: {args.eval_mode}; device: {device}; "
        f"train/val/test: {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}"
    )
    print(
        f"Tie taxonomy: {tie_taxonomy.name} v{tie_taxonomy.version} "
        f"({tie_taxonomy.sha256[:12]}); inherited={','.join(tie_taxonomy.inherited)}"
    )
    print(json.dumps({"relation_perturbation": relation_perturbation}, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        if full_training_data is not None:
            train_loss = train_full_graph_epoch(
                model,
                full_training_data,
                optimizer,
                feature_schema,
                loss_mode,
                class_weights,
                log_priors,
                args.logit_adjustment_tau,
            )
        else:
            epoch_train_loader = train_loader
            if args.train_root_sampling == "class_balanced":
                sampled_roots = class_balanced_train_nodes(
                    data, metadata["num_classes"], root_sampling_generator
                )
                epoch_train_loader = make_loader(
                    data, sampled_roots, fanouts, args, shuffle=True, persistent_workers=False
                )
            train_loss = train_epoch(
                model,
                epoch_train_loader,
                optimizer,
                device,
                feature_schema,
                occupation_unknown_ids,
                loss_mode,
                class_weights,
                log_priors,
                args.logit_adjustment_tau,
            )
        if full_evaluation_data is None:
            val_metrics, _ = evaluate(model, val_loader, device, feature_schema, occupation_unknown_ids)
        else:
            val_metrics, _ = evaluate_full_graph(
                model, full_evaluation_data, full_evaluation_data.val_mask, feature_schema
            )
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
        checkpoint_improved = (
            monitor_value < best_monitor
            if is_loss_monitor
            else monitor_value > best_monitor
        )
        if checkpoint_improved:
            best_monitor = monitor_value
            best_state = copy.deepcopy(model.state_dict())

        patience_improved = (
            monitor_value < best_patience_monitor - args.min_delta
            if is_loss_monitor
            else monitor_value > best_patience_monitor + args.min_delta
        )
        if patience_improved:
            best_patience_monitor = monitor_value
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
    if args.skip_test:
        test_metrics, test_predictions = None, None
    elif full_evaluation_data is None:
        test_metrics, test_predictions = evaluate(model, test_loader, device, feature_schema, occupation_unknown_ids)
    else:
        test_metrics, test_predictions = evaluate_full_graph(
            model, full_evaluation_data, full_evaluation_data.test_mask, feature_schema
        )
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "model_feature_schema": feature_schema,
        "model_name": args.model,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "branch_dim": args.branch_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "attention_dropout": args.attention_dropout,
            "num_bases": args.num_bases,
            "rgcn_backend": args.rgcn_backend,
            "compgcn_composition": args.compgcn_composition,
            "occupation_feature_levels": list(occupation_levels),
            "occupation_representation": args.occupation_representation,
            **representation_provenance,
            "auxiliary_features": list(auxiliary_features),
            "feature_mode": args.feature_mode,
            "train_mode": args.train_mode,
            "loss": loss_mode,
            "class_balanced_beta": args.class_balanced_beta,
            "logit_adjustment_tau": args.logit_adjustment_tau,
            "train_root_sampling": args.train_root_sampling,
            "eval_mode": args.eval_mode,
        },
        "selection_metric": args.early_stop_metric,
        "best_selection_metric": best_monitor,
        "best_val_macro_f1_seen": best_val_f1,
        "relation_perturbation": relation_perturbation,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "run_config": vars(args),
            "occupation_representation": args.occupation_representation,
            "representation_provenance": representation_provenance,
            "model_feature_schema": feature_schema,
            "selection_metric": args.early_stop_metric,
            "best_selection_metric": best_monitor,
            "best_val_macro_f1_seen": best_val_f1,
            "loss": {
                "mode": loss_mode,
                "class_balanced_beta": args.class_balanced_beta,
                "logit_adjustment_tau": args.logit_adjustment_tau,
                "train_root_sampling": args.train_root_sampling,
                "train_mode": args.train_mode,
            },
            "relation_perturbation": relation_perturbation,
            "test": test_metrics,
            "history": history,
        }, handle, indent=2)
    if test_predictions is not None:
        save_predictions(test_predictions, metadata, data_path, output_dir / "test_predictions.csv")
    print(json.dumps({
        "selection_metric": args.early_stop_metric,
        "best_selection_metric": best_monitor,
        "best_val_macro_f1_seen": best_val_f1,
        "test": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
