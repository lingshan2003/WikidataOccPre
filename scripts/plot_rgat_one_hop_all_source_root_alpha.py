#!/usr/bin/env python3
"""Plot all-source root-level RGAT alpha matrices from the bootstrap export.

Unlike the visible-training-neighbour figures, these plots include every typed
incoming edge to every test root. Hidden validation/test sources use their true
L1 only for post-hoc grouping; sources without a retained L1 are displayed as
``Unlabeled/missing``. With the project's zero synthetic self-loop count, the
kinship/nonkinship panel sums to 100 percent in every target-L1 column.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


L1_LABELS = ("Culture", "Discovery/Science", "Leadership", "Other", "Sports/Games", "__UNLABELED__")
DISPLAY_LABELS = {"__UNLABELED__": "Unlabeled/missing"}
DIRECT_KINSHIP = ("child", "spouse", "sibling", "father", "mother")
KINSHIP = frozenset({
    "child", "child__rev", "spouse", "spouse__rev", "sibling", "sibling__rev",
    "father", "father__rev", "mother", "mother__rev",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True, help="root_direct_attention_bootstrap.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def display(label: str) -> str:
    return DISPLAY_LABELS.get(label, label).replace("/", "/\n")


def read_per_seed_cells(path: Path):
    values: dict[tuple[str, str, str, str], float] = defaultdict(float)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["experiment"] != "rgat_one_hop"
                or row["message_passing_layer"] != "1"
                or row["metric"] != "attention_mass"
                or row["source_l1"] not in L1_LABELS
            ):
                continue
            values[(row["seed"], row["relation"], row["source_l1"], row["target_l1"])] += float(row["mean"]) * 100.0
    if not values:
        raise ValueError("No rgat_one_hop root-level attention rows were found")
    return values


def aggregate(per_seed, groups: dict[str, frozenset[str] | tuple[str, ...]]):
    per_group_seed: dict[tuple[str, str, str, str], float] = defaultdict(float)
    for (seed, relation, source, target), value in per_seed.items():
        for group, relations in groups.items():
            if relation in relations:
                per_group_seed[(seed, group, source, target)] += value

    cells = {}
    for group in groups:
        for source in L1_LABELS:
            for target in L1_LABELS[:-1]:
                values = [
                    value for (_seed, row_group, row_source, row_target), value in per_group_seed.items()
                    if row_group == group and row_source == source and row_target == target
                ]
                cells[(group, source, target)] = {
                    "mean_percent": mean(values) if values else 0.0,
                    "seed_sd_percent": stdev(values) if len(values) > 1 else 0.0,
                }
    return cells


def write_csv(path: Path, cells, group_order) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group", "source_l1", "target_l1", "mean_root_attention_mass_percent", "seed_sd_percent"),
        )
        writer.writeheader()
        for group in group_order:
            for source in L1_LABELS:
                for target in L1_LABELS[:-1]:
                    cell = cells[(group, source, target)]
                    writer.writerow({
                        "group": group,
                        "source_l1": source,
                        "target_l1": target,
                        "mean_root_attention_mass_percent": cell["mean_percent"],
                        "seed_sd_percent": cell["seed_sd_percent"],
                    })


def plot_matrix(cells, group_order, output: Path, title: str, footer: str) -> None:
    targets = L1_LABELS[:-1]
    row_keys = [(group, source) for group in group_order for source in L1_LABELS]
    matrix = np.array([
        [cells[(group, source, target)]["mean_percent"] for target in targets]
        for group, source in row_keys
    ])
    figure_height = 6.2 + 0.31 * len(row_keys)
    figure, axis = plt.subplots(figsize=(10.2, figure_height))
    figure.subplots_adjust(left=0.32, right=0.90, top=0.88, bottom=0.10)
    image = axis.imshow(matrix, cmap=plt.colormaps["YlGnBu"], vmin=0.0, vmax=float(matrix.max()), aspect="auto")
    for row in range(matrix.shape[0] + 1):
        axis.axhline(row - 0.5, color="white", linewidth=0.85)
    for column in range(matrix.shape[1] + 1):
        axis.axvline(column - 0.5, color="white", linewidth=0.85)
    for row in range(len(L1_LABELS), len(row_keys), len(L1_LABELS)):
        axis.axhline(row - 0.5, color="#535353", linewidth=1.25)
    for row, (group, source) in enumerate(row_keys):
        if source == L1_LABELS[0]:
            axis.text(-1.12, row + (len(L1_LABELS) - 1) / 2, group, ha="center", va="center", fontsize=11, clip_on=False)
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= matrix.max() * 0.55 else "#202020"
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=9.2, color=color)
    axis.set_xticks(range(len(targets)), [display(label) for label in targets], fontsize=10)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=9)
    axis.set_yticks(range(len(row_keys)), [display(source) for _, source in row_keys], fontsize=8.8)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.set_xlim(-1.70, len(targets) - 0.5)
    axis.set_xlabel("Target test-root L1", labelpad=16)
    axis.set_title(title, loc="left", fontsize=14, pad=48)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.038, pad=0.02)
    colorbar.set_label("Mean root attention mass (percent)")
    figure.text(0.5, 0.043, footer, ha="center", fontsize=8.5)
    figure.text(
        0.5,
        0.018,
        "All test roots and all typed source states are included. Hidden-source L1 is post-hoc only; values are not claims about observed neighbour occupations.",
        ha="center",
        fontsize=8.5,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_kinship_nonkinship(cells, output: Path) -> None:
    """Use source-L1 blocks with kinship/nonkinship rows, like Fig. 4."""
    targets = L1_LABELS[:-1]
    families = ("kinship", "nonkinship")
    row_keys = [(source, family) for source in L1_LABELS for family in families]
    matrix = np.array([
        [cells[(family, source, target)]["mean_percent"] for target in targets]
        for source, family in row_keys
    ])

    figure, axis = plt.subplots(figsize=(10.2, 9.4))
    figure.subplots_adjust(left=0.31, right=0.90, top=0.87, bottom=0.13)
    image = axis.imshow(
        matrix,
        cmap=plt.colormaps["YlGnBu"],
        vmin=0.0,
        vmax=float(matrix.max()),
        aspect="auto",
    )
    for row in range(matrix.shape[0] + 1):
        axis.axhline(row - 0.5, color="white", linewidth=0.9)
    for column in range(matrix.shape[1] + 1):
        axis.axvline(column - 0.5, color="white", linewidth=0.9)
    for row in range(2, len(row_keys), 2):
        axis.axhline(row - 0.5, color="#535353", linewidth=1.2)

    for row, (source, family) in enumerate(row_keys):
        if family == "kinship":
            axis.text(
                -1.55,
                row + 0.5,
                display(source),
                ha="right",
                va="center",
                fontsize=10,
                clip_on=False,
            )
        axis.text(-0.72, row, family, ha="right", va="center", fontsize=9.5, clip_on=False)
        for column, value in enumerate(matrix[row]):
            color = "white" if value >= matrix.max() * 0.55 else "#222222"
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=10, color=color)

    axis.set_xticks(range(len(targets)), [display(label) for label in targets], fontsize=10)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=9)
    axis.set_yticks([])
    axis.set_xlim(-1.75, len(targets) - 0.5)
    axis.set_xlabel("Target test-root L1", labelpad=16)
    axis.set_title("RGAT one-hop: root-level direct attention mass", loc="left", fontsize=14, pad=48)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.040, pad=0.02)
    colorbar.set_label("Mean root attention mass (percent)")
    figure.text(
        0.5,
        0.052,
        "All typed incoming edges to all test roots; each cell first sums α within a root, then averages roots and seeds. "
        "Kinship = child, spouse, sibling, father, mother and their stored reverse directions; nonkinship = all other exact relations.",
        ha="center",
        fontsize=8.5,
    )
    figure.text(
        0.5,
        0.023,
        "Visible, hidden, and unlabeled/missing sources are included. Hidden-source L1 is post-hoc only; each target column sums to 100.0% (rounding aside).",
        ha="center",
        fontsize=8.5,
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = read_per_seed_cells(args.bootstrap)

    internal_groups = {relation: (relation,) for relation in DIRECT_KINSHIP}
    internal_cells = aggregate(per_seed, internal_groups)
    write_csv(args.output_dir / "rgat_one_hop_all_source_kinship_internal_cells.csv", internal_cells, DIRECT_KINSHIP)
    plot_matrix(
        internal_cells,
        DIRECT_KINSHIP,
        args.output_dir / "rgat_one_hop_all_source_kinship_internal.png",
        "RGAT one-hop: all-source root-level attention within kinship",
        "Exact stored directions shown: child, spouse, sibling, father, mother. Other relations are deliberately outside this internal kinship view.",
    )

    family_groups = {"kinship": KINSHIP, "nonkinship": frozenset()}
    all_relations = {relation for (_seed, relation, _source, _target) in per_seed}
    family_groups["nonkinship"] = frozenset(all_relations - KINSHIP)
    family_cells = aggregate(per_seed, family_groups)
    write_csv(args.output_dir / "rgat_one_hop_all_source_kinship_nonkinship_cells.csv", family_cells, ("kinship", "nonkinship"))
    plot_kinship_nonkinship(
        family_cells,
        args.output_dir / "rgat_one_hop_all_source_kinship_nonkinship.png",
    )


if __name__ == "__main__":
    main()
