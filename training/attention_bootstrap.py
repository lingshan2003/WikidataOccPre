#!/usr/bin/env python3
"""Cluster-bootstrap sparse root-level attention or rollout statistics.

Rows absent from a sparse input are reconstructed as zero for every root in
the matching roster stratum.  This makes the bootstrap person/root-level,
rather than an invalid edge-level resampling procedure.  The same command can
also consume a future per-root counterfactual-delta file via ``--value-column``.
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse


IDENTITY_COLUMNS = (
    "experiment", "seed", "checkpoint", "split", "forward_mode", "num_layers", "fanouts",
    "message_passing_layer", "target_l1_id", "target_l1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, help="*.csv or *.csv.gz root roster from an attention export")
    parser.add_argument("--sparse", required=True, help="*.csv or *.csv.gz sparse root-level direct/rollout rows")
    parser.add_argument("--output", required=True, help="Output bootstrap CSV")
    parser.add_argument(
        "--value-columns",
        default="auto",
        help="Comma-separated numeric columns to bootstrap, or auto for attention_mass/opportunity/difference.",
    )
    parser.add_argument(
        "--group-columns",
        default="auto",
        help=(
            "Comma-separated sparse grouping columns beyond root/stratum. auto uses relation+source fields for "
            "direct attention and ordered r1/r2 for rollout, keeping all raw dimensions available for later reruns."
        ),
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--batch-resamples", type=int, default=32)
    return parser.parse_args()


def open_csv(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open(
        newline="", encoding="utf-8"
    )


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open_csv(path) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or ())


def parse_columns(value: str, available: Sequence[str], kind: str) -> List[str]:
    if value == "auto":
        if kind == "values":
            candidates = ["attention_mass", "opportunity", "attention_mass_minus_opportunity"]
            if "rollout_mass" in available:
                candidates = ["rollout_mass", "opportunity", "rollout_mass_minus_opportunity"]
            return [column for column in candidates if column in available]
        if "relation_id" in available:
            candidates = ["relation_id", "relation", "source_l1_id", "source_l1", "source_visibility"]
        else:
            candidates = ["r1_id", "r1", "r2_id", "r2"]
        return [column for column in candidates if column in available]
    columns = [column.strip() for column in value.split(",") if column.strip()]
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"Unknown {kind} columns: {missing}; available: {', '.join(available)}")
    if not columns:
        raise ValueError(f"--{kind}-columns must not be empty")
    return columns


def identity_columns(roster_fields: Sequence[str], sparse_fields: Sequence[str]) -> List[str]:
    return [column for column in IDENTITY_COLUMNS if column in roster_fields and column in sparse_fields]


def bootstrap_matrix(
    matrix: sparse.spmatrix,
    root_count: int,
    resamples: int,
    seed: int,
    batch_resamples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return root means and percentile CIs for every sparse-matrix column."""
    if root_count < 1:
        raise ValueError("Bootstrap stratum has no roots")
    if resamples < 1 or batch_resamples < 1:
        raise ValueError("--resamples and --batch-resamples must be positive")
    matrix = matrix.tocsc()
    means = np.asarray(matrix.sum(axis=0)).reshape(-1) / root_count
    samples = np.empty((resamples, matrix.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, batch_resamples):
        size = min(batch_resamples, resamples - start)
        weights = np.empty((size, root_count), dtype=np.int32)
        for offset in range(size):
            sampled = rng.integers(root_count, size=root_count)
            weights[offset] = np.bincount(sampled, minlength=root_count)
        samples[start:start + size] = np.asarray(matrix.T @ weights.T).T / root_count
    return means, np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def group_bootstrap(
    roster_rows: Sequence[Mapping[str, str]],
    sparse_rows: Sequence[Mapping[str, str]],
    roster_fields: Sequence[str],
    sparse_fields: Sequence[str],
    value_columns: Sequence[str],
    group_columns: Sequence[str],
    resamples: int,
    seed: int,
    batch_resamples: int,
) -> List[Dict[str, object]]:
    base_columns = identity_columns(roster_fields, sparse_fields)
    if not base_columns:
        raise ValueError("Roster and sparse inputs have no shared identity columns")
    roster_by_base: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for row in roster_rows:
        base = tuple(row[column] for column in base_columns)
        roster_by_base[base].append(row["root_index"])
    for base, roots in roster_by_base.items():
        if len(roots) != len(set(roots)):
            raise ValueError(f"Root roster has duplicate root_index values within stratum {base}")

    rows_by_base_group: Dict[Tuple[str, ...], Dict[Tuple[str, ...], Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in sparse_rows:
        base = tuple(row[column] for column in base_columns)
        if base not in roster_by_base:
            raise ValueError(f"Sparse row has no matching root roster stratum: {base}")
        group = tuple(row[column] for column in group_columns)
        root = row["root_index"]
        for column in value_columns:
            value = float(row[column])
            previous = rows_by_base_group[base][group][column].get(root, 0.0)
            rows_by_base_group[base][group][column][root] = previous + value

    output: List[Dict[str, object]] = []
    for base_index, (base, groups) in enumerate(sorted(rows_by_base_group.items())):
        root_ids = roster_by_base[base]
        root_position = {root_id: position for position, root_id in enumerate(root_ids)}
        group_keys = sorted(groups)
        for metric_index, metric in enumerate(value_columns):
            matrix_rows: List[int] = []
            matrix_columns: List[int] = []
            matrix_values: List[float] = []
            nonzero_roots = np.zeros(len(group_keys), dtype=np.int64)
            for column_index, group in enumerate(group_keys):
                values = groups[group][metric]
                nonzero_roots[column_index] = len(values)
                for root_id, value in values.items():
                    matrix_rows.append(root_position[root_id])
                    matrix_columns.append(column_index)
                    matrix_values.append(value)
            matrix = sparse.coo_matrix(
                (matrix_values, (matrix_rows, matrix_columns)), shape=(len(root_ids), len(group_keys)), dtype=np.float64
            )
            means, lower, upper = bootstrap_matrix(
                matrix, len(root_ids), resamples, seed + base_index * 101 + metric_index, batch_resamples
            )
            for index, group in enumerate(group_keys):
                output.append({
                    **dict(zip(base_columns, base)),
                    **dict(zip(group_columns, group)),
                    "metric": metric,
                    "root_count": len(root_ids),
                    "roots_with_nonzero_value": int(nonzero_roots[index]),
                    "mean": float(means[index]),
                    "bootstrap_ci_lower": float(lower[index]),
                    "bootstrap_ci_upper": float(upper[index]),
                    "resamples": resamples,
                    "bootstrap_seed": seed + base_index * 101 + metric_index,
                })
    return output


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("Bootstrap produced no rows")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    roster_path, sparse_path, output_path = Path(args.roster), Path(args.sparse), Path(args.output)
    roster_rows, roster_fields = read_rows(roster_path)
    sparse_rows, sparse_fields = read_rows(sparse_path)
    if not roster_rows:
        raise ValueError("Root roster is empty")
    if not sparse_rows:
        raise ValueError("Sparse root-level input is empty")
    values = parse_columns(args.value_columns, sparse_fields, "values")
    groups = parse_columns(args.group_columns, sparse_fields, "group")
    rows = group_bootstrap(
        roster_rows, sparse_rows, roster_fields, sparse_fields, values, groups,
        args.resamples, args.seed, args.batch_resamples,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_path, rows)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps({
        "roster": str(roster_path),
        "sparse": str(sparse_path),
        "output": str(output_path),
        "value_columns": values,
        "group_columns": groups,
        "resamples": args.resamples,
        "seed": args.seed,
        "definition": (
            "Root-cluster percentile bootstrap. Sparse rows absent for a root/group are reconstructed as zero "
            "using the matching root roster; seed/checkpoint strata are never pooled."
        ),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} and {manifest_path}")


if __name__ == "__main__":
    main()
