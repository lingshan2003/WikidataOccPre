"""Shared RGAT attention-report loading, sampling, and output helpers.

This module deliberately contains no aggregation policy.  Edge-level and
node-level reports import the same checkpoint/replay utilities but define
their statistics in separate entry points.
"""

from __future__ import annotations

import csv
import gzip
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import torch

from models import build_feature_specs, build_model
from training.relation_controls import apply_relation_controls


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is False")
    return device


def parse_fanouts(value: str, num_layers: int) -> list[int]:
    try:
        fanouts = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--num-neighbors must look like '15,10', 'auto', or 'full'") from error
    if len(fanouts) != num_layers or any(item < -1 for item in fanouts):
        raise ValueError(
            f"The checkpoint has {num_layers} message-passing layers, so --num-neighbors must provide "
            f"{num_layers} values (received {value!r})"
        )
    return fanouts


def checkpoint_paths(paths: Sequence[str], patterns: Sequence[str]) -> list[Path]:
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


def fanouts_for_checkpoint(path: Path, requested: str, num_layers: int) -> list[int]:
    if requested == "full":
        return [-1] * num_layers
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
    if split == "labeled":
        return data.y >= 0
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
        raise ValueError("Attention export is available for RGAT checkpoints only")
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


def replay_relation_perturbation(
    data,
    metadata: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> None:
    manifest = checkpoint.get("relation_perturbation")
    if not manifest:
        return
    apply_relation_controls(
        data,
        relation_ids_to_drop=manifest.get("dropped_relation_ids", ()),
        relation_to_id=metadata["relation_to_id"],
        random_edge_drop_pairs=manifest.get("random_edge_drop_pairs", 0),
        random_edge_instance_pairs=manifest.get("random_edge_instance_pairs", 0),
        random_edge_drop_seed=manifest.get("random_edge_drop_seed"),
        shuffle_relation_types=manifest.get("relation_type_shuffle", False),
        shuffle_seed=manifest.get("relation_type_shuffle_seed"),
    )


def root_indices(data, split: str) -> torch.Tensor:
    """Resolve any supported root selector to an explicit CPU index tensor."""
    selected = prediction_nodes(data, split)
    if selected.dtype == torch.bool:
        return selected.nonzero(as_tuple=False).view(-1).cpu()
    return selected.cpu()


def validate_full_graph_root_mask(
    data,
    root_ids: torch.Tensor,
    split: str,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
) -> None:
    """Reject a full-graph export when a requested root can see its occupation."""
    if split not in {"val", "test"}:
        raise ValueError(
            "--forward-mode full-graph is valid only with --split val or --split test. "
            "Training/labeled roots need root-specific masking, so use --forward-mode full-neighborhood instead."
        )
    for feature_name, unknown_id in occupation_unknown_ids.items():
        if feature_name not in feature_schema or not hasattr(data, feature_name):
            continue
        values = getattr(data, feature_name)[root_ids]
        if bool(values.ne(int(unknown_id)).any()):
            visible = int(values.ne(int(unknown_id)).sum().item())
            raise RuntimeError(
                f"{visible} requested {split} roots expose {feature_name}; refusing full-graph attention export "
                "because alpha would be conditioned on a root's own occupation."
            )


def source_visibility_codes(
    graph,
    source_indices: torch.Tensor,
    feature_schema: Mapping[str, object],
    occupation_unknown_ids: Mapping[str, int],
) -> torch.Tensor:
    """Classify whether each source's true L1 is visible to the model."""
    labels = graph.y[source_indices]
    result = torch.ones_like(labels, dtype=torch.long)
    result[labels < 0] = 2
    if "occupation_level1" not in feature_schema or not hasattr(graph, "occupation_level1"):
        return result
    unknown_id = occupation_unknown_ids.get("occupation_level1")
    if unknown_id is None:
        return result
    observed = getattr(graph, "occupation_level1")[source_indices].ne(int(unknown_id))
    visible = (labels >= 0) & graph.train_mask[source_indices] & observed
    result[visible] = 0
    return result


def head_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Return alpha as [edge, head], including the one-head representation."""
    return alpha.unsqueeze(-1) if alpha.dim() == 1 else alpha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the checked-out source revision when this runs inside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_csv(path: Path, records: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def output_fields(records: Sequence[Mapping[str, object]], preferred: Sequence[str]) -> list[str]:
    """Keep stable metadata columns first while retaining dynamic head columns."""
    observed = {key for record in records for key in record}
    fields = [field for field in preferred if field in observed]
    fields.extend(sorted(observed - set(fields)))
    return fields
