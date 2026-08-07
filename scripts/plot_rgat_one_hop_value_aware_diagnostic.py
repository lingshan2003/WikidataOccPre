#!/usr/bin/env python3
"""Plot a cautious diagnostic for RGAT alpha versus value-aware messages.

Panel A resembles an OCI matrix, but its values are *not* outcome importance:
they are the within-root allocation of absolute pre-activation message
magnitude, conditional on a root receiving at least one visible-training L1
message. Panel B checks whether alpha rankings materially change after the
relation-specific value transform is included.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
    parser.add_argument("--message-sparse", type=Path, required=True)
    parser.add_argument("--message-roster", type=Path, required=True)
    parser.add_argument("--attention-bootstrap", type=Path, required=True)
    parser.add_argument("--message-bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG path; a PDF is written beside it")
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def open_csv(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open(
        newline="", encoding="utf-8"
    )


def family(relation: str) -> str:
    return "kinship" if relation in KINSHIP else "nonkinship"


def load_message_share_matrix(message_sparse: Path, message_roster: Path):
    roots: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open_csv(message_roster) as handle:
        for row in csv.DictReader(handle):
            roots[(row["seed"], row["target_l1"])].append(row["root_index"])

    root_total: dict[tuple[str, str, str], float] = defaultdict(float)
    group_total: dict[tuple[tuple[str, str, str], str, str], float] = defaultdict(float)
    with open_csv(message_sparse) as handle:
        for row in csv.DictReader(handle):
            if row["source_visibility"] != "visible_train" or row["source_l1"] not in LABELS:
                continue
            root_key = (row["seed"], row["root_index"], row["target_l1"])
            value = float(row["absolute_message_l2_sum"])
            root_total[root_key] += value
            group_total[(root_key, row["source_l1"], family(row["relation"]))] += value

    per_seed: dict[tuple[str, str, str, str], float] = {}
    coverage: dict[tuple[str, str], float] = {}
    for (seed, target), root_ids in roots.items():
        active = [root for root in root_ids if root_total[(seed, root, target)] > 0]
        coverage[(seed, target)] = len(active) / len(root_ids) if root_ids else 0.0
        for source in LABELS:
            for relation_family in FAMILIES:
                values = [
                    group_total.get(((seed, root, target), source, relation_family), 0.0)
                    / root_total[(seed, root, target)]
                    for root in active
                ]
                per_seed[(seed, source, relation_family, target)] = mean(values) * 100.0 if values else 0.0

    cells = {}
    for source in LABELS:
        for relation_family in FAMILIES:
            for target in LABELS:
                values = [
                    value for (seed, row_source, row_family, row_target), value in per_seed.items()
                    if row_source == source and row_family == relation_family and row_target == target
                ]
                cells[(source, relation_family, target)] = {
                    "mean_percent": mean(values),
                    "seed_sd_percent": stdev(values) if len(values) > 1 else 0.0,
                }
    coverage_mean = {
        target: mean(value for (_seed, row_target), value in coverage.items() if row_target == target)
        for target in LABELS
    }
    return cells, coverage_mean


def rank(values: np.ndarray) -> np.ndarray:
    """Ordinal ranks suffice here because these continuous bootstrap means have no ties."""
    return np.argsort(np.argsort(values))


def load_alpha_message_pairs(attention_bootstrap: Path, message_bootstrap: Path):
    alpha = {}
    with open_csv(attention_bootstrap) as handle:
        for row in csv.DictReader(handle):
            if (
                row["experiment"] == "rgat_one_hop"
                and row["source_visibility"] == "visible_train"
                and row["metric"] == "attention_mass"
            ):
                key = (row["seed"], row["target_l1"], row["relation"], row["source_l1"])
                alpha[key] = float(row["mean"])
    pairs = []
    with open_csv(message_bootstrap) as handle:
        for row in csv.DictReader(handle):
            if row["source_visibility"] != "visible_train" or row["metric"] != "absolute_message_l2_sum":
                continue
            key = (row["seed"], row["target_l1"], row["relation"], row["source_l1"])
            if key in alpha and alpha[key] > 0 and float(row["mean"]) > 0:
                pairs.append((alpha[key] * 100.0, float(row["mean"])))
    x, y = np.asarray(pairs, dtype=float).T
    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    return x, y, rho


def write_csv(path: Path, cells, coverage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_l1", "relation_family", "target_l1", "mean_share_percent",
                "seed_sd_percent", "visible_train_message_coverage",
            ),
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
                        "mean_share_percent": cell["mean_percent"],
                        "seed_sd_percent": cell["seed_sd_percent"],
                        "visible_train_message_coverage": coverage[target],
                    })


def display(label: str) -> str:
    return label.replace("/", "/\n")


def plot(cells, coverage, x: np.ndarray, y: np.ndarray, rho: float, output: Path) -> None:
    row_keys = [(source, relation_family) for source in LABELS for relation_family in FAMILIES]
    matrix = np.array([
        [cells[(source, relation_family, target)]["mean_percent"] for target in LABELS]
        for source, relation_family in row_keys
    ])
    figure, (table_axis, scatter_axis) = plt.subplots(
        1, 2, figsize=(16.0, 8.0), gridspec_kw={"width_ratios": [1.28, 1.0]}
    )
    figure.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.16, wspace=0.34)
    cmap = plt.colormaps["YlGnBu"]
    image = table_axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=float(matrix.max()), aspect="auto")
    for row in range(matrix.shape[0] + 1):
        table_axis.axhline(row - 0.5, color="white", linewidth=0.9)
    for column in range(matrix.shape[1] + 1):
        table_axis.axvline(column - 0.5, color="white", linewidth=0.9)
    for row in range(2, len(row_keys), 2):
        table_axis.axhline(row - 0.5, color="#535353", linewidth=1.2)
    for row, (source, relation_family) in enumerate(row_keys):
        if relation_family == "kinship":
            table_axis.text(-1.55, row + 0.5, source, ha="right", va="center", fontsize=10, clip_on=False)
        table_axis.text(-0.72, row, relation_family, ha="right", va="center", fontsize=9.5, clip_on=False)
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value >= matrix.max() * 0.55 else "#222222"
            table_axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=10, color=color)
    table_axis.set_xticks(range(len(LABELS)), [display(label) for label in LABELS], fontsize=10)
    table_axis.xaxis.tick_top()
    table_axis.tick_params(axis="x", length=0, pad=9)
    table_axis.set_yticks([])
    table_axis.set_xlim(-1.75, len(LABELS) - 0.5)
    table_axis.set_title("A. Pre-activation message-magnitude allocation", loc="left", fontsize=13, pad=48)
    table_axis.set_xlabel("Target test-root L1", labelpad=16)
    colorbar = figure.colorbar(image, ax=table_axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Share of visible-train message magnitude (%)")

    scatter_axis.scatter(x, y, s=11, color="#2c7fb8", alpha=0.34, linewidths=0)
    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.grid(True, which="both", color="#d7d7d7", linewidth=0.7)
    scatter_axis.set_xlabel("Root attention mass (%)")
    scatter_axis.set_ylabel("Absolute message magnitude (arbitrary units)")
    scatter_axis.set_title("B. α ranking versus α·Wᵣh ranking", loc="left", fontsize=13)
    scatter_axis.text(
        0.04,
        0.95,
        f"n = {len(x):,} relation × source-L1 × target-L1 × seed cells\nSpearman ρ = {rho:.3f}",
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "pad": 5},
    )

    coverage_text = "; ".join(f"{label}: {coverage[label] * 100:.1f}%" for label in LABELS)
    figure.suptitle("RGAT one-hop, test roots, visible-training L1 neighbours, three seed means", fontsize=15)
    figure.text(
        0.5,
        0.055,
        "A: kinship = child, spouse, sibling, father, mother and their stored reverse directions; "
        "nonkinship = all other exact relations. Each target column sums to 100% conditional on a root having "
        "at least one visible-train L1 message. Coverage: " + coverage_text,
        ha="center",
        fontsize=8.6,
    )
    figure.text(
        0.5,
        0.022,
        "This is a value-aware mechanism diagnostic, not final-logit importance: residual, GELU, LayerNorm and classifier effects are excluded.",
        ha="center",
        fontsize=8.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output.suffix.casefold() != ".png":
        raise ValueError("--output must have a .png suffix")
    cells, coverage = load_message_share_matrix(args.message_sparse, args.message_roster)
    x, y, rho = load_alpha_message_pairs(args.attention_bootstrap, args.message_bootstrap)
    if args.csv_output:
        write_csv(args.csv_output, cells, coverage)
    plot(cells, coverage, x, y, rho, args.output)


if __name__ == "__main__":
    main()
