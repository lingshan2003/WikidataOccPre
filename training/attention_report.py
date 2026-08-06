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
    split: str,
    requested_fanouts: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
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
    totals: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0])
    synthetic_target_edges: Dict[int, int] = defaultdict(int)
    roots_seen = 0
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
    run_info = {
        "checkpoint": str(path),
        "experiment": experiment,
        "seed": seed,
        "num_layers": num_layers,
        "fanouts": fanouts,
        "prediction_roots": roots_seen,
        "synthetic_self_loop_target_edges": {str(layer): count for layer, count in synthetic_target_edges.items()},
    }
    return records, run_info


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


def main() -> None:
    args = parse_args()
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    device = resolve_device(args.device)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    base_data, base_metadata = bundle["data"], bundle["metadata"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, object]] = []
    run_info = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] collecting attention: {path}", flush=True)
        records, info = collect_checkpoint_attention(
            path,
            base_data,
            base_metadata,
            args.split,
            args.num_neighbors,
            args.batch_size,
            args.num_workers,
            device,
        )
        if not records:
            raise RuntimeError(f"No typed incoming edges were collected from {path}")
        all_records.extend(records)
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
    manifest_path = output_dir / "attention_report_manifest.json"
    manifest_path.write_text(json.dumps({
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "definition": (
            "Mean head-averaged RGAT alpha over typed incoming edges whose destination is a prediction root; "
            "synthetic self-loops are excluded."
        ),
        "runs": run_info,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote:")
    for path in (raw_path, summary_path, table_path, markdown_path, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
