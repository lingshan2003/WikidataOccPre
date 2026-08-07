#!/usr/bin/env python3
"""Render a Figure-5-style root-level kinship-attention matrix for RGAT one-hop.

The input is the compact root-level table produced from the test-root sparse
attention records.  Cells show mean attention mass (not edge-average alpha),
in percent, across independently trained seeds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RELATIONS = ("child", "spouse", "sibling", "father", "mother")
L1_LABELS = ("Culture", "Discovery/Science", "Leadership", "Other", "Sports/Games")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root-level cell summary CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF path (defaults to the PNG path with a .pdf suffix)",
    )
    return parser.parse_args()


def load_cells(path: Path) -> dict[tuple[str, str, str], float]:
    cells: dict[tuple[str, str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relation = row["relation"]
            source_l1 = row["source_l1"]
            target_l1 = row["target_l1"]
            if relation in RELATIONS and source_l1 in L1_LABELS and target_l1 in L1_LABELS:
                cells[(relation, source_l1, target_l1)] = float(row["mean_attention_mass"]) * 100.0
    if not cells:
        raise ValueError(f"No selected kinship cells found in {path}")
    return cells


def plot(cells: dict[tuple[str, str, str], float], output: Path, pdf_output: Path) -> None:
    row_keys = [(relation, source_l1) for relation in RELATIONS for source_l1 in L1_LABELS]
    matrix = np.full((len(row_keys), len(L1_LABELS)), np.nan, dtype=float)
    for row_index, (relation, source_l1) in enumerate(row_keys):
        for column_index, target_l1 in enumerate(L1_LABELS):
            value = cells.get((relation, source_l1, target_l1))
            if value is not None:
                matrix[row_index, column_index] = value

    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#f3f3f3")
    figure, axis = plt.subplots(figsize=(8.7, 11.0))
    figure.subplots_adjust(left=0.30, right=0.91, top=0.90, bottom=0.09)
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=np.nanmax(matrix), aspect="auto")

    for row_index in range(matrix.shape[0] + 1):
        axis.axhline(row_index - 0.5, color="white", linewidth=0.8)
    for column_index in range(matrix.shape[1] + 1):
        axis.axvline(column_index - 0.5, color="white", linewidth=0.8)
    for relation_index in range(1, len(RELATIONS)):
        axis.axhline(relation_index * len(L1_LABELS) - 0.5, color="#5b5b5b", linewidth=1.25)

    for row_index, (_, source_l1) in enumerate(row_keys):
        for column_index in range(len(L1_LABELS)):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                label = "—"
                color = "#555555"
            else:
                label = f"{value:.1f}"
                color = "white" if value >= np.nanmax(matrix) * 0.52 else "#202020"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8.2, color=color)

    axis.set_xticks(range(len(L1_LABELS)), [label.replace("/", "/\n") for label in L1_LABELS], fontsize=9)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=7)
    axis.set_yticks(range(len(row_keys)), [source_l1 for _, source_l1 in row_keys], fontsize=8.2)
    axis.tick_params(axis="y", length=0, pad=6)
    axis.set_xlim(-1.55, len(L1_LABELS) - 0.5)
    for relation_index, relation in enumerate(RELATIONS):
        center = relation_index * len(L1_LABELS) + (len(L1_LABELS) - 1) / 2
        axis.text(-1.1, center, relation, ha="center", va="center", fontsize=10.5, fontweight="medium", clip_on=False)

    axis.set_title("RGAT one-hop: root-level direct attention mass on test roots", loc="left", fontsize=13, pad=42)
    axis.set_xlabel("Target test-root L1", labelpad=13)
    axis.set_ylabel("Source L1 (visible training neighbour)", labelpad=20)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.48, pad=0.02)
    colorbar.set_label("Mean root attention mass (percent)")
    figure.text(
        0.5,
        0.020,
        "Exact stored directions: child, spouse, sibling, father, mother. "
        "Each cell is the three-seed mean of per-root summed attention mass; — = no candidate edge.",
        ha="center",
        fontsize=8.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    plot(load_cells(args.input), args.output, pdf_output)


if __name__ == "__main__":
    main()
