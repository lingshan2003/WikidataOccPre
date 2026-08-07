#!/usr/bin/env python3
"""Render a root-level alpha kinship/nonkinship matrix for RGAT one-hop.

Each cell is a three-seed mean of the per-root sum of head-averaged attention
coefficients.  Sparse relation/source rows absent for a root are treated as
zero, so this is a root-level attention mass rather than an edge-average.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


KINSHIP = frozenset({
    "child", "child__rev", "spouse", "spouse__rev", "sibling", "sibling__rev",
    "father", "father__rev", "mother", "mother__rev",
})
LABELS = ("Culture", "Discovery/Science", "Leadership", "Other", "Sports/Games")
FAMILIES = ("kinship", "nonkinship")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        type=Path,
        required=True,
        help="root_direct_attention_bootstrap.csv (which has already reconstructed missing root/groups as zero)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG; a matching PDF is also written")
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def family(relation: str) -> str:
    return "kinship" if relation in KINSHIP else "nonkinship"


def load_cells(bootstrap_path: Path):
    """Linearly combine root means over exact relations into the two families."""
    per_seed: dict[tuple[str, str, str, str], float] = defaultdict(float)
    with bootstrap_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["experiment"] != "rgat_one_hop"
                or row["message_passing_layer"] != "1"
                or row["source_visibility"] != "visible_train"
                or row["source_l1"] not in LABELS
                or row["metric"] != "attention_mass"
            ):
                continue
            key = (row["seed"], row["source_l1"], family(row["relation"]), row["target_l1"])
            per_seed[key] += float(row["mean"]) * 100.0

    cells = {}
    for source in LABELS:
        for relation_family in FAMILIES:
            for target in LABELS:
                values = [
                    value for (_seed, row_source, row_family, row_target), value in per_seed.items()
                    if row_source == source and row_family == relation_family and row_target == target
                ]
                cells[(source, relation_family, target)] = {
                    "mean_percent": mean(values),
                    "seed_sd_percent": stdev(values) if len(values) > 1 else 0.0,
                }
    return cells


def write_csv(path: Path, cells) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_l1", "relation_family", "target_l1", "mean_root_attention_mass_percent", "seed_sd_percent"),
        )
        writer.writeheader()
        for source in LABELS:
            for relation_family in FAMILIES:
                for target in LABELS:
                    cell = cells[(source, relation_family, target)]
                    writer.writerow({
                        "source_l1": source,
                        "relation_family": relation_family,
                        "target_l1": target,
                        "mean_root_attention_mass_percent": cell["mean_percent"],
                        "seed_sd_percent": cell["seed_sd_percent"],
                    })


def display(label: str) -> str:
    return label.replace("/", "/\n")


def plot(cells, output: Path) -> None:
    row_keys = [(source, relation_family) for source in LABELS for relation_family in FAMILIES]
    matrix = np.array([
        [cells[(source, relation_family, target)]["mean_percent"] for target in LABELS]
        for source, relation_family in row_keys
    ])
    figure, axis = plt.subplots(figsize=(9.4, 8.7))
    figure.subplots_adjust(left=0.28, right=0.91, top=0.86, bottom=0.13)
    image = axis.imshow(matrix, cmap=plt.colormaps["YlGnBu"], vmin=0.0, vmax=float(matrix.max()), aspect="auto")
    for row in range(matrix.shape[0] + 1):
        axis.axhline(row - 0.5, color="white", linewidth=0.9)
    for column in range(matrix.shape[1] + 1):
        axis.axvline(column - 0.5, color="white", linewidth=0.9)
    for row in range(2, len(row_keys), 2):
        axis.axhline(row - 0.5, color="#535353", linewidth=1.2)
    for row, (source, relation_family) in enumerate(row_keys):
        if relation_family == "kinship":
            axis.text(-1.55, row + 0.5, source, ha="right", va="center", fontsize=10, clip_on=False)
        axis.text(-0.72, row, relation_family, ha="right", va="center", fontsize=9.5, clip_on=False)
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= matrix.max() * 0.55 else "#222222"
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=10, color=color)
    axis.set_xticks(range(len(LABELS)), [display(label) for label in LABELS], fontsize=10)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=9)
    axis.set_yticks([])
    axis.set_xlim(-1.75, len(LABELS) - 0.5)
    axis.set_xlabel("Target test-root L1", labelpad=16)
    axis.set_title("RGAT one-hop: root-level direct attention mass", loc="left", fontsize=14, pad=48)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.040, pad=0.02)
    colorbar.set_label("Mean root attention mass (percent)")
    figure.text(
        0.5,
        0.052,
        "Visible-training L1 neighbours only; each cell first sums α within a root, then averages roots and seeds. "
        "Kinship = child, spouse, sibling, father, mother and their stored reverse directions; nonkinship = all other exact relations.",
        ha="center",
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.023,
        "Cells are attention allocation, not final prediction importance. Attention on hidden/missing sources is not shown, so columns need not sum to 100%.",
        ha="center",
        fontsize=8.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output.suffix.casefold() != ".png":
        raise ValueError("--output must use a .png suffix")
    cells = load_cells(args.bootstrap)
    if args.csv_output:
        write_csv(args.csv_output, cells)
    plot(cells, args.output)


if __name__ == "__main__":
    main()
