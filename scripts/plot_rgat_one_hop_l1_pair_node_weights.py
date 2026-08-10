#!/usr/bin/env python3
"""Render node-first RGAT one-hop L1-pair weight heatmaps.

The input table contains ``mean_a``: the mean attention mass for target nodes
that have at least one edge in a given ``(source L1, relation, target L1)``
cell. The first panel preserves the five forward kinship relations as exact CSV
cells. The second groups exact relations into kinship and nonkinship; within a
group, values are weighted by ``n`` so every matching target/relation
observation contributes equally.

The script uses only the Python standard library and writes editable SVGs.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


L1_LABELS = ("Culture", "Discovery/Science", "Leadership", "Other", "Sports/Games")
FORWARD_KINSHIP = ("child", "spouse", "sibling", "father", "mother")
KINSHIP = frozenset((*FORWARD_KINSHIP, *(f"{relation}__rev" for relation in FORWARD_KINSHIP)))
PALETTE = ("#ffffd9", "#d9f0a3", "#78c8bd", "#2c7fb8", "#081d58")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="L1-pair node-weight summary CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for figures and derived cells")
    return parser.parse_args()


def display(label: str) -> str:
    return label.replace("/", "/\n")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_l1", "target_l1", "relation", "mean_a", "n"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_kinship_cells(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    cells: dict[tuple[str, str, str], float] = {}
    for row in rows:
        relation = row["relation"]
        source = row["source_l1"]
        target = row["target_l1"]
        if relation in FORWARD_KINSHIP and source in L1_LABELS and target in L1_LABELS:
            cells[(relation, source, target)] = float(row["mean_a"])
    if not cells:
        raise ValueError("No forward kinship cells matched the requested L1 labels")
    return cells


def family_cells(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, str], float], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source = row["source_l1"]
        target = row["target_l1"]
        if source not in L1_LABELS or target not in L1_LABELS:
            continue
        family = "kinship" if row["relation"] in KINSHIP else "nonkinship"
        grouped[(source, family, target)].append(row)

    cells: dict[tuple[str, str, str], float] = {}
    output_rows: list[dict[str, object]] = []
    for source in L1_LABELS:
        for family in ("kinship", "nonkinship"):
            for target in L1_LABELS:
                group = grouped.get((source, family, target), [])
                total_n = sum(float(row["n"]) for row in group)
                value = (
                    sum(float(row["mean_a"]) * float(row["n"]) for row in group) / total_n
                    if total_n
                    else float("nan")
                )
                cells[(source, family, target)] = value
                output_rows.append({
                    "source_l1": source,
                    "relation_family": family,
                    "target_l1": target,
                    "mean_a_n_weighted": value,
                    "matching_target_relation_n": total_n,
                    "exact_relation_count": len(group),
                })
    return cells, output_rows


def svg_text(x: float, y: float, value: str, size: float, *, anchor: str = "middle", fill: str = "#202020", weight: str = "400", transform: str = "") -> str:
    attributes = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" font-size="{size:.1f}" font-weight="{weight}"'
    if transform:
        attributes += f' transform="{transform}"'
    lines = html.escape(value).split("\n")
    if len(lines) == 1:
        return f"<text {attributes}>{lines[0]}</text>"
    spans = "".join(
        f'<tspan x="{x:.1f}" dy="{0 if index == 0 else size * 1.04:.1f}">{line}</tspan>'
        for index, line in enumerate(lines)
    )
    return f"<text {attributes}>{spans}</text>"


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def interpolate_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    scaled = value * (len(PALETTE) - 1)
    index = min(int(scaled), len(PALETTE) - 2)
    fraction = scaled - index
    start = hex_rgb(PALETTE[index])
    end = hex_rgb(PALETTE[index + 1])
    return "#" + "".join(f"{round(a + (b - a) * fraction):02x}" for a, b in zip(start, end))


def nice_vmax(values: Sequence[float]) -> float:
    maximum = max(value for value in values if math.isfinite(value))
    magnitude = 10 ** math.floor(math.log10(maximum))
    for candidate in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        bound = candidate * magnitude
        if maximum <= bound:
            return bound
    raise AssertionError("unreachable")


def gradient() -> str:
    stops = "".join(
        f'<stop offset="{100 * index / (len(PALETTE) - 1):.1f}%" stop-color="{color}"/>'
        for index, color in enumerate(PALETTE)
    )
    return f'<defs><linearGradient id="heatmap-gradient" x1="0" x2="0" y1="1" y2="0">{stops}</linearGradient></defs>'


def render_svg(
    *,
    matrix: list[list[float]],
    row_labels: list[str],
    title: str,
    footer: str,
    output: Path,
    block_size: int,
    group_labels: list[str] | None = None,
    family_rows: bool = False,
    colorbar_label: str = "mean_a",
) -> None:
    row_count = len(matrix)
    col_count = len(L1_LABELS)
    cell_width = 132
    cell_height = 45 if row_count > 12 else 62
    data_x = 385 if family_rows else 400
    data_y = 142
    data_width = col_count * cell_width
    data_height = row_count * cell_height
    colorbar_x = data_x + data_width + 38
    colorbar_y = data_y + max(0, (data_height - 330) / 2)
    colorbar_height = min(data_height, 330)
    width = colorbar_x + 155
    height = data_y + data_height + 112
    values = [value for row in matrix for value in row]
    vmax = nice_vmax(values)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        gradient(),
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        svg_text(data_x, 40, title, 23, anchor="start", weight="500"),
    ]

    for column, label in enumerate(L1_LABELS):
        lines.append(svg_text(data_x + (column + 0.5) * cell_width, 88, display(label), 17, weight="500"))
    for row, label in enumerate(row_labels):
        y = data_y + (row + 0.5) * cell_height + 5
        if family_rows:
            source, family = label.split("|", maxsplit=1)
            if family == "kinship":
                lines.append(svg_text(data_x - 147, y + cell_height / 2, display(source), 17, anchor="end", weight="500"))
            lines.append(svg_text(data_x - 25, y, family, 15, anchor="end"))
        else:
            lines.append(svg_text(data_x - 14, y, label, 14, anchor="end"))

    for row, values_row in enumerate(matrix):
        for column, value in enumerate(values_row):
            x = data_x + column * cell_width
            y = data_y + row * cell_height
            normalized = value / vmax if math.isfinite(value) else 0.0
            fill = interpolate_color(normalized) if math.isfinite(value) else "#f1f1f1"
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="white" stroke-width="1"/>')
            label = f"{value:.3f}" if math.isfinite(value) else "—"
            text_color = "white" if math.isfinite(value) and normalized >= 0.57 else "#202020"
            lines.append(svg_text(x + cell_width / 2, y + cell_height / 2 + 5, label, 15, fill=text_color))

    for divider in range(block_size, row_count, block_size):
        y = data_y + divider * cell_height
        lines.append(f'<line x1="{data_x}" y1="{y}" x2="{data_x + data_width}" y2="{y}" stroke="#535353" stroke-width="2"/>')
    lines.append(f'<rect x="{data_x}" y="{data_y}" width="{data_width}" height="{data_height}" fill="none" stroke="#202020" stroke-width="1.4"/>')

    if group_labels:
        for index, label in enumerate(group_labels):
            y = data_y + (index * block_size + block_size / 2) * cell_height + 5
            lines.append(svg_text(data_x - 210, y, label, 17, weight="500"))
    else:
        center_y = data_y + data_height / 2
        lines.append(svg_text(68, center_y, "Source L1", 17, weight="500", transform=f"rotate(-90 68 {center_y:.1f})"))

    lines.append(svg_text(data_x + data_width / 2, data_y + data_height + 55, "Target test-root L1", 17, weight="500"))
    lines.append(f'<rect x="{colorbar_x}" y="{colorbar_y}" width="26" height="{colorbar_height}" fill="url(#heatmap-gradient)" stroke="#202020" stroke-width="1"/>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = colorbar_y + colorbar_height * (1 - fraction)
        value = vmax * fraction
        lines.append(f'<line x1="{colorbar_x + 26}" y1="{y}" x2="{colorbar_x + 33}" y2="{y}" stroke="#202020" stroke-width="1"/>')
        lines.append(svg_text(colorbar_x + 41, y + 5, f"{value:.2f}", 13, anchor="start"))
    center_y = colorbar_y + colorbar_height / 2
    lines.append(svg_text(colorbar_x + 96, center_y, colorbar_label, 16, weight="500", transform=f"rotate(-90 {colorbar_x + 96} {center_y:.1f})"))
    lines.append(svg_text(width / 2, height - 27, footer, 12))
    lines.extend(["</g>", "</svg>"])
    output.write_text("\n".join(lines), encoding="utf-8")


def plot_exact_kinship(cells: dict[tuple[str, str, str], float], output: Path) -> None:
    row_keys = [(relation, source) for relation in FORWARD_KINSHIP for source in L1_LABELS]
    matrix = [
        [cells.get((relation, source, target), float("nan")) for target in L1_LABELS]
        for relation, source in row_keys
    ]
    render_svg(
        matrix=matrix,
        row_labels=[source for _relation, source in row_keys],
        title="RGAT one-hop: L1-pair node weights for exact kinship relations",
        footer="Exact stored directions: child, spouse, sibling, father, mother. Each cell is the CSV mean_a; — = no matching cell.",
        output=output,
        block_size=len(L1_LABELS),
        group_labels=list(FORWARD_KINSHIP),
    )


def plot_relation_families(cells: dict[tuple[str, str, str], float], output: Path) -> None:
    row_keys = [(source, family) for source in L1_LABELS for family in ("kinship", "nonkinship")]
    matrix = [[cells[(source, family, target)] for target in L1_LABELS] for source, family in row_keys]
    render_svg(
        matrix=matrix,
        row_labels=[f"{source}|{family}" for source, family in row_keys],
        title="RGAT one-hop: L1-pair node weights by relation family",
        footer="Kinship includes child, spouse, sibling, father, mother and stored reverse directions; nonkinship includes all other exact relations. Each family cell is the n-weighted mean of CSV mean_a values.",
        output=output,
        block_size=2,
        family_rows=True,
        colorbar_label="n-weighted mean_a",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    exact_cells = exact_kinship_cells(rows)
    family_aggregate, family_rows = family_cells(rows)
    plot_exact_kinship(exact_cells, args.output_dir / "rgat_one_hop_l1_pair_node_weight_exact_kinship.svg")
    plot_relation_families(family_aggregate, args.output_dir / "rgat_one_hop_l1_pair_node_weight_kinship_nonkinship.svg")
    write_rows(
        args.output_dir / "rgat_one_hop_l1_pair_node_weight_kinship_nonkinship_cells.csv",
        family_rows,
        (
            "source_l1",
            "relation_family",
            "target_l1",
            "mean_a_n_weighted",
            "matching_target_relation_n",
            "exact_relation_count",
        ),
    )


if __name__ == "__main__":
    main()
