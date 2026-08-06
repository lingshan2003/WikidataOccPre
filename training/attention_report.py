#!/usr/bin/env python3
"""Aggregate test-node RGAT attention by relation across one or more checkpoints.

The reported value is the mean of head-averaged attention coefficients for
typed *incoming* edges whose destination is a requested prediction node.  It
is a model-mechanism summary, not a causal relation-effect estimate.
"""

import argparse
import copy
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch_geometric.loader import NeighborLoader

from models import build_feature_specs, build_model
from training.attention_utils import attention_relation_ids
from training.relation_controls import apply_relation_controls
from training.train import batch_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared graph_data.pt used by all checkpoints")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Exact best_model.pt path; may be supplied more than once",
    )
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=[],
        help="Glob of best_model.pt paths, e.g. runs_report/level1/rgat_baseline/seed_*/best_model.pt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Prediction roots to include; test is the default report population",
    )
    parser.add_argument(
        "--num-neighbors",
        default="auto",
        help=(
            "Comma-separated analysis fan-outs, or auto to reuse each run's metrics.json value. "
            "Auto falls back to 20 per model layer if the run config is unavailable."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--occupation-matrix-relations",
        default="none",
        help=(
            "Comma-separated directed relation labels for L1 source-to-target attention matrices, "
            "or none. Names are used exactly as stored: 'father' does not also select 'father__rev'."
        ),
    )
    parser.add_argument(
        "--matrix-min-edge-count",
        type=int,
        default=10,
        help="Hide a source-L1/target-L1 matrix cell in Markdown when its mean edge support is below this value",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is False")
    return device


def parse_fanouts(value: str, num_layers: int) -> List[int]:
    try:
        fanouts = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--num-neighbors must look like '15,10' or 'auto'") from error
    if len(fanouts) != num_layers or any(item < -1 for item in fanouts):
        raise ValueError(
            f"The checkpoint has {num_layers} message-passing layers, so --num-neighbors must provide "
            f"{num_layers} values (received {value!r})"
        )
    return fanouts


def parse_matrix_relations(value: str, relation_to_id: Mapping[str, int]) -> Dict[int, str]:
    """Resolve exact directed relation names for a Figure-5-style L1 matrix.

    Prepared graphs deliberately add ``__rev`` edges for message passing.  A
    matrix must retain that direction rather than silently merge both types,
    because the source/target L1 axes would otherwise become ambiguous.
    """
    if value.strip().casefold() == "none":
        return {}
    names = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not names:
        raise ValueError("--occupation-matrix-relations must be relation names or 'none'")
    unknown = [name for name in names if name not in relation_to_id]
    if unknown:
        available = ", ".join(sorted(relation_to_id))
        raise ValueError(
            f"Unknown directed relation names: {unknown}. Use an exact label from relation_to_id, "
            f"including '__rev' when needed. Available: {available}"
        )
    return {int(relation_to_id[name]): name for name in names}


def checkpoint_paths(paths: Sequence[str], patterns: Sequence[str]) -> List[Path]:
    resolved = [Path(path) for path in paths]
    for pattern in patterns:
        matched = sorted(Path(path) for path in glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"No checkpoints match --checkpoint-glob {pattern!r}")
        resolved.extend(matched)
    unique = []
    seen = set()
    for path in resolved:
        normalized = path.resolve()
        if normalized not in seen:
            if not normalized.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {path}")
            unique.append(normalized)
            seen.add(normalized)
    if not unique:
        raise ValueError("Supply at least one --checkpoint or --checkpoint-glob")
    return unique


def model_depth(checkpoint: Mapping[str, object]) -> int:
    config = checkpoint.get("model_config", {})
    if isinstance(config, Mapping) and "num_layers" in config:
        return int(config["num_layers"])
    # All legacy report checkpoints predate explicit depth serialization and
    # were trained with the fixed two-layer protocol.
    return 2


def fanouts_for_checkpoint(path: Path, requested: str, num_layers: int) -> List[int]:
    if requested != "auto":
        return parse_fanouts(requested, num_layers)
    metrics_path = path.parent / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            run_config = json.load(handle).get("run_config", {})
        value = run_config.get("num_neighbors")
        if value is not None:
            return parse_fanouts(str(value), num_layers)
    return [20] * num_layers


def checkpoint_identity(path: Path) -> Tuple[str, str]:
    """Return the report-condition label and seed from a conventional path."""
    seed_match = re.fullmatch(r"seed_(.+)", path.parent.name)
    if seed_match:
        return path.parent.parent.name, seed_match.group(1)
    return path.parent.name, ""


def prediction_nodes(data, split: str) -> torch.Tensor:
    if split == "all":
        return torch.arange(data.num_nodes, dtype=torch.long)
    mask_name = f"{split}_mask"
    if not hasattr(data, mask_name):
        raise ValueError(f"Prepared graph lacks {mask_name}")
    mask = getattr(data, mask_name)
    if mask.dtype != torch.bool:
        raise ValueError(f"Prepared graph field {mask_name} must be a boolean mask")
    return mask


def restore_rgat(checkpoint: Mapping[str, object], device: torch.device):
    metadata = checkpoint["metadata"]
    model_name = checkpoint.get("model_name", "rgat")
    if model_name != "rgat":
        raise ValueError("Aggregate attention export is available for RGAT checkpoints only")
    feature_schema = checkpoint.get("model_feature_schema", metadata["feature_schema"])
    model = build_model(
        "rgat",
        num_relations=metadata["num_relations"],
        num_classes=metadata["num_classes"],
        feature_specs=build_feature_specs(feature_schema, metadata),
        **checkpoint.get("model_config", {}),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), feature_schema, metadata


def replay_relation_perturbation(data, metadata: Mapping[str, object], checkpoint: Mapping[str, object]) -> None:
    manifest = checkpoint.get("relation_perturbation")
    if not manifest:
        return
    apply_relation_controls(
        data,
        relation_ids_to_drop=manifest.get("dropped_relation_ids", ()),
        relation_to_id=metadata["relation_to_id"],
        random_edge_drop_pairs=manifest.get("random_edge_drop_pairs", 0),
        random_edge_drop_seed=manifest.get("random_edge_drop_seed"),
        shuffle_relation_types=manifest.get("relation_type_shuffle", False),
        shuffle_seed=manifest.get("relation_type_shuffle_seed"),
    )


@torch.no_grad()
def collect_checkpoint_attention(
    path: Path,
    base_data,
    base_metadata: Mapping[str, object],
    matrix_relation_ids: Mapping[int, str],
    split: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, feature_schema, metadata = restore_rgat(checkpoint, device)
    if metadata["relation_to_id"] != base_metadata["relation_to_id"]:
        raise ValueError(f"Checkpoint relation mapping differs from --data: {path}")
    num_layers = model_depth(checkpoint)
    if len(model.convs) != num_layers:
        raise ValueError(
            f"Checkpoint depth metadata ({num_layers}) does not match its state dict ({len(model.convs)} layers): {path}"
        )
    fanouts = fanouts_for_checkpoint(path, requested_fanouts, num_layers)
    data = copy.deepcopy(base_data)
    replay_relation_perturbation(data, metadata, checkpoint)
    loader = NeighborLoader(
        data,
        input_nodes=prediction_nodes(data, split),
        num_neighbors=fanouts,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    id_to_relation = {int(index): relation for relation, index in metadata["relation_to_id"].items()}
    id_to_label = {int(index): label for label, index in metadata["label_to_id"].items()}
    totals: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    # Key: message layer, directed relation, neighbour/source L1, target L1.
    # Values are [head-averaged attention sum, typed edge count].
    l1_pair_totals: Dict[Tuple[int, int, int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    synthetic_target_edges: Dict[int, int] = defaultdict(int)
    target_l1_counts = torch.zeros(metadata["num_classes"], dtype=torch.long)
    roots_seen = 0
    matrix_relation_tensor = torch.tensor(
        sorted(matrix_relation_ids), dtype=data.edge_type.dtype, device=device
    ) if matrix_relation_ids else None
    for batch in loader:
        batch = batch.to(device)
        logits, explanation = model(
            batch_features(batch, feature_schema, metadata["occupation_unknown_ids"]),
            batch.edge_index,
            batch.edge_type,
            return_attention_weights=True,
        )
        del logits
        roots_seen += int(batch.batch_size)
        root_l1 = batch.y[:batch.batch_size].detach().cpu()
        target_l1_counts += torch.bincount(root_l1[root_l1 >= 0], minlength=metadata["num_classes"])
        for layer_info in explanation["attention_layers"]:
            layer = int(layer_info["layer"]) + 1
            edge_index = layer_info["edge_index"]
            relation_ids = attention_relation_ids(layer_info)
            alpha = layer_info["alpha"]
            scores = alpha if alpha.dim() == 1 else alpha.mean(dim=-1)
            is_prediction_target = edge_index[1] < batch.batch_size
            is_typed_relation = relation_ids >= 0
            synthetic_target_edges[layer] += int((is_prediction_target & ~is_typed_relation).sum().item())
            usable = is_prediction_target & is_typed_relation
            if not bool(usable.any()):
                continue
            for relation_id in relation_ids[usable].unique(sorted=True).tolist():
                relation_mask = usable & (relation_ids == relation_id)
                total = totals[(layer, int(relation_id))]
                total[0] += float(scores[relation_mask].sum().item())
                total[1] += float(relation_mask.sum().item())

            if matrix_relation_tensor is None:
                continue
            source_l1 = batch.y[edge_index[0]]
            target_l1 = batch.y[edge_index[1]]
            l1_pair_mask = (
                usable
                & torch.isin(relation_ids, matrix_relation_tensor)
                & (source_l1 >= 0)
                & (target_l1 >= 0)
            )
            if not bool(l1_pair_mask.any()):
                continue
            # At most (#relations × #L1²) unique keys occur per batch, so
            # collecting on-device first avoids an expensive Python loop over
            # individual attention edges.
            class_count = int(metadata["num_classes"])
            encoded = (
                relation_ids[l1_pair_mask].long() * class_count * class_count
                + source_l1[l1_pair_mask].long() * class_count
                + target_l1[l1_pair_mask].long()
            )
            unique_encoded, inverse = encoded.unique(sorted=True, return_inverse=True)
            score_sums = torch.zeros(unique_encoded.numel(), dtype=scores.dtype, device=device)
            score_sums.scatter_add_(0, inverse, scores[l1_pair_mask])
            edge_counts = torch.bincount(inverse, minlength=unique_encoded.numel())
            for encoded_key, score_sum, edge_count in zip(
                unique_encoded.tolist(), score_sums.tolist(), edge_counts.tolist()
            ):
                relation_id = encoded_key // (class_count * class_count)
                remainder = encoded_key % (class_count * class_count)
                source_label_id = remainder // class_count
                target_label_id = remainder % class_count
                total = l1_pair_totals[(layer, relation_id, source_label_id, target_label_id)]
                total[0] += float(score_sum)
                total[1] += float(edge_count)

    experiment, seed = checkpoint_identity(path)
    records = []
    for (layer, relation_id), (score_sum, edge_count) in sorted(totals.items()):
        records.append({
            "experiment": experiment,
            "seed": seed,
            "checkpoint": str(path),
            "split": split,
            "num_layers": num_layers,
            "fanouts": ",".join(str(value) for value in fanouts),
            "message_passing_layer": layer,
            "relation_id": relation_id,
            "relation": id_to_relation[relation_id],
            "edge_count": int(edge_count),
            "attention_mean": score_sum / edge_count,
        })
    l1_pair_records = []
    for (layer, relation_id, source_label_id, target_label_id), (score_sum, edge_count) in sorted(l1_pair_totals.items()):
        target_count = int(target_l1_counts[target_label_id])
        l1_pair_records.append({
            "experiment": experiment,
            "seed": seed,
            "checkpoint": str(path),
            "split": split,
            "num_layers": num_layers,
            "fanouts": ",".join(str(value) for value in fanouts),
            "message_passing_layer": layer,
            "relation_id": relation_id,
            "relation": id_to_relation[relation_id],
            "source_l1_id": source_label_id,
            "source_l1": id_to_label[source_label_id],
            "target_l1_id": target_label_id,
            "target_l1": id_to_label[target_label_id],
            "edge_count": int(edge_count),
            "target_l1_count": target_count,
            "attention_mean": score_sum / edge_count,
            "attention_mass_per_target": score_sum / max(target_count, 1),
        })
    run_info = {
        "checkpoint": str(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": num_layers,
        "fanouts": fanouts,
        "prediction_roots": roots_seen,
        "prediction_roots_by_l1": {
            id_to_label[label_id]: int(count) for label_id, count in enumerate(target_l1_counts.tolist())
        },
        "synthetic_self_loop_target_edges": {str(layer): count for layer, count in synthetic_target_edges.items()},
    }
    return records, l1_pair_records, run_info


def summarise_seed_records(records: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int, int, str], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["experiment"]),
            int(record["num_layers"]),
            int(record["message_passing_layer"]),
            int(record["relation_id"]),
            str(record["relation"]),
        )
        grouped[key].append(record)
    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][3], item[0][0], item[0][2])):
        values = [float(row["attention_mean"]) for row in rows]
        edge_counts = [int(row["edge_count"]) for row in rows]
        summaries.append({
            "experiment": key[0],
            "num_layers": key[1],
            "message_passing_layer": key[2],
            "relation_id": key[3],
            "relation": key[4],
            "seed_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows if str(row["seed"])),
            "edge_count_mean": mean(edge_counts),
            "attention_mean": mean(values),
            "attention_std": stdev(values) if len(values) > 1 else 0.0,
        })
    return summaries


def summarise_l1_pair_records(records: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Average a Figure-5-style source-L1/target-L1 cell across seed runs."""
    grouped: Dict[Tuple[str, int, int, int, str, int, str, int, str], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["experiment"]),
            int(record["num_layers"]),
            int(record["message_passing_layer"]),
            int(record["relation_id"]),
            str(record["relation"]),
            int(record["source_l1_id"]),
            str(record["source_l1"]),
            int(record["target_l1_id"]),
            str(record["target_l1"]),
        )
        grouped[key].append(record)
    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][3], item[0][5], item[0][7], item[0][0], item[0][2])):
        attention_values = [float(row["attention_mean"]) for row in rows]
        mass_values = [float(row["attention_mass_per_target"]) for row in rows]
        edge_counts = [int(row["edge_count"]) for row in rows]
        target_counts = [int(row["target_l1_count"]) for row in rows]
        summaries.append({
            "experiment": key[0],
            "num_layers": key[1],
            "message_passing_layer": key[2],
            "relation_id": key[3],
            "relation": key[4],
            "source_l1_id": key[5],
            "source_l1": key[6],
            "target_l1_id": key[7],
            "target_l1": key[8],
            "seed_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows if str(row["seed"])),
            "edge_count_mean": mean(edge_counts),
            "target_l1_count_mean": mean(target_counts),
            "attention_mean": mean(attention_values),
            "attention_std": stdev(attention_values) if len(attention_values) > 1 else 0.0,
            "attention_mass_per_target": mean(mass_values),
            "attention_mass_per_target_std": stdev(mass_values) if len(mass_values) > 1 else 0.0,
        })
    return summaries


def write_csv(path: Path, records: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def wide_table(summary_records: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    columns = sorted({
        (str(row["experiment"]), int(row["message_passing_layer"])) for row in summary_records
    })
    relation_rows: Dict[Tuple[int, str], Dict[str, object]] = {}
    for summary in summary_records:
        relation_key = (int(summary["relation_id"]), str(summary["relation"]))
        row = relation_rows.setdefault(relation_key, {
            "relation_id": relation_key[0], "relation": relation_key[1],
        })
        experiment, layer = str(summary["experiment"]), int(summary["message_passing_layer"])
        prefix = f"{experiment}_layer{layer}"
        row[f"{prefix}_attention_mean"] = float(summary["attention_mean"])
        row[f"{prefix}_attention_std"] = float(summary["attention_std"])
        row[f"{prefix}_seed_runs"] = int(summary["seed_runs"])
    fieldnames = ["relation_id", "relation"]
    for experiment, layer in columns:
        prefix = f"{experiment}_layer{layer}"
        fieldnames.extend([
            f"{prefix}_attention_mean",
            f"{prefix}_attention_std",
            f"{prefix}_seed_runs",
        ])
    return [relation_rows[key] for key in sorted(relation_rows)], fieldnames


def write_markdown_table(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    value_columns = [column for column in columns if column.endswith("_attention_mean")]
    headers = ["关系 ID", "关系"] + [column.removesuffix("_attention_mean") for column in value_columns]
    lines = [
        "# L1 RGAT relation attention summary",
        "",
        "每个值为三条 seed 运行中，测试集预测节点入边的 head-average attention 均值 ± seed 标准差。",
        "`layer1`/`layer2` 表示消息传递层，不应解释为因果效应。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row["relation_id"]), str(row["relation"])]
        for column in value_columns:
            mean_value = row.get(column)
            std_value = row.get(column.removesuffix("_attention_mean") + "_attention_std")
            values.append("" if mean_value is None else f"{float(mean_value):.6f} ± {float(std_value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_l1_pair_matrices(
    path: Path,
    summary_records: Sequence[Mapping[str, object]],
    id_to_label: Mapping[int, str],
    min_edge_count: int,
) -> None:
    """Write one source-L1 by target-L1 matrix per relation/model/layer.

    The matrices deliberately use the exact directed relation label in the
    graph.  For example, ``father`` and ``father__rev`` are different panels;
    this preserves the source-to-target semantics needed for inheritance work.
    """
    labels = [(int(label_id), id_to_label[int(label_id)]) for label_id in sorted(id_to_label)]
    grouped: Dict[Tuple[str, int, int, int, str], Dict[Tuple[int, int], Mapping[str, object]]] = {}
    for record in summary_records:
        group_key = (
            str(record["experiment"]),
            int(record["num_layers"]),
            int(record["message_passing_layer"]),
            int(record["relation_id"]),
            str(record["relation"]),
        )
        grouped.setdefault(group_key, {})[(int(record["source_l1_id"]), int(record["target_l1_id"]))] = record

    lines = [
        "# L1 relation attention matrices",
        "",
        "Rows are source/neighbour true L1 labels; columns are target/test-person true L1 labels.",
        "Each displayed value is mean head-averaged incoming attention ± seed standard deviation; `n` is the mean number of typed edges across seed runs.",
        "A dash means no labelled edge was observed. A cell with fewer than the configured minimum support is masked as `low n`.",
        "The labels are exact directed graph relations: `father` and `father__rev` must be interpreted separately.",
        "",
    ]
    for group_key, cells in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][3])):
        experiment, num_layers, layer, _, relation = group_key
        lines.extend([
            f"## {experiment} - {num_layers}-hop model, Layer {layer}, relation `{relation}`",
            "",
            "| Source L1 \\ Target L1 | " + " | ".join(label for _, label in labels) + " |",
            "| --- | " + " | ".join(["---"] * len(labels)) + " |",
        ])
        for source_id, source_label in labels:
            values = [source_label]
            for target_id, _ in labels:
                cell = cells.get((source_id, target_id))
                if cell is None:
                    values.append("—")
                    continue
                edge_count = float(cell["edge_count_mean"])
                if edge_count < min_edge_count:
                    values.append(f"low n ({edge_count:.0f})")
                    continue
                values.append(
                    f"{float(cell['attention_mean']):.6f} ± {float(cell['attention_std']):.6f}<br>n={edge_count:.0f}"
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.matrix_min_edge_count < 1:
        raise ValueError("--matrix-min-edge-count must be at least one")
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    matrix_relation_ids = parse_matrix_relations(args.occupation_matrix_relations, base_metadata["relation_to_id"])
    if matrix_relation_ids and base_metadata.get("target_column") != "occupation_level1":
        raise ValueError(
            "--occupation-matrix-relations is an L1 analysis and requires an artifact prepared with --target-level 1"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, object]] = []
    all_l1_pair_records: List[Dict[str, object]] = []
    run_info = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] collecting attention: {path}", flush=True)
        records, l1_pair_records, info = collect_checkpoint_attention(
            path,
            base_data,
            base_metadata,
            matrix_relation_ids,
            args.split,
            args.num_neighbors,
            args.batch_size,
            args.num_workers,
            device,
        )
        if not records:
            raise RuntimeError(f"No typed incoming edges were collected from {path}")
        all_records.extend(records)
        all_l1_pair_records.extend(l1_pair_records)
        run_info.append(info)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_records = summarise_seed_records(all_records)
    table_rows, table_fields = wide_table(summary_records)
    raw_fields = [
        "experiment", "seed", "checkpoint", "split", "num_layers", "fanouts", "message_passing_layer",
        "relation_id", "relation", "edge_count", "attention_mean",
    ]
    summary_fields = [
        "experiment", "num_layers", "message_passing_layer", "relation_id", "relation", "seed_runs", "seeds",
        "edge_count_mean", "attention_mean", "attention_std",
    ]
    raw_path = output_dir / "relation_attention_by_seed.csv"
    summary_path = output_dir / "relation_attention_summary.csv"
    table_path = output_dir / "relation_attention_table.csv"
    markdown_path = output_dir / "relation_attention_table.md"
    write_csv(raw_path, all_records, raw_fields)
    write_csv(summary_path, summary_records, summary_fields)
    write_csv(table_path, table_rows, table_fields)
    write_markdown_table(markdown_path, table_rows, table_fields)
    matrix_paths = []
    if matrix_relation_ids:
        if not all_l1_pair_records:
            selected = ", ".join(matrix_relation_ids.values())
            raise RuntimeError(f"No labelled L1 attention edges were collected for the selected relations: {selected}")
        l1_pair_summary = summarise_l1_pair_records(all_l1_pair_records)
        l1_pair_raw_fields = [
            "experiment", "seed", "checkpoint", "split", "num_layers", "fanouts", "message_passing_layer",
            "relation_id", "relation", "source_l1_id", "source_l1", "target_l1_id", "target_l1",
            "edge_count", "target_l1_count", "attention_mean", "attention_mass_per_target",
        ]
        l1_pair_summary_fields = [
            "experiment", "num_layers", "message_passing_layer", "relation_id", "relation",
            "source_l1_id", "source_l1", "target_l1_id", "target_l1", "seed_runs", "seeds",
            "edge_count_mean", "target_l1_count_mean", "attention_mean", "attention_std",
            "attention_mass_per_target", "attention_mass_per_target_std",
        ]
        l1_pair_raw_path = output_dir / "l1_relation_attention_by_seed.csv"
        l1_pair_summary_path = output_dir / "l1_relation_attention_summary.csv"
        l1_pair_markdown_path = output_dir / "l1_relation_attention_matrices.md"
        write_csv(l1_pair_raw_path, all_l1_pair_records, l1_pair_raw_fields)
        write_csv(l1_pair_summary_path, l1_pair_summary, l1_pair_summary_fields)
        id_to_label = {int(index): label for label, index in base_metadata["label_to_id"].items()}
        write_l1_pair_matrices(
            l1_pair_markdown_path,
            l1_pair_summary,
            id_to_label,
            args.matrix_min_edge_count,
        )
        matrix_paths = [l1_pair_raw_path, l1_pair_summary_path, l1_pair_markdown_path]
    manifest_path = output_dir / "attention_report_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "definition": (
            "Mean head-averaged RGAT alpha over typed incoming edges whose destination is a prediction root; "
            "synthetic self-loops are excluded."
        ),
        "l1_relation_matrix": {
            "enabled": bool(matrix_relation_ids),
            "relations": matrix_relation_ids,
            "definition": (
                "For each exact directed relation, source L1 and target L1 cell, mean head-averaged RGAT "
                "alpha over typed incoming test-target edges. Source/target true labels are used only for post-hoc grouping."
            ) if matrix_relation_ids else None,
            "minimum_edge_count_for_markdown": args.matrix_min_edge_count if matrix_relation_ids else None,
        },
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (raw_path, summary_path, table_path, markdown_path, *matrix_paths, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
