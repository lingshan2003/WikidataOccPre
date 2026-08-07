#!/usr/bin/env python3
"""Render an OCI-style kinship/nonkinship L1 attention matrix from RGAT exports.

The exporter stores exact directed relations, including ``__rev`` edges.  This
script treats an edge as kinship when its *base* relation is one of
child/spouse/sibling/father/mother, so both stored directions remain kinship.
All other exact relations form the nonkinship comparison group.

Values are edge-count-weighted mean direct-edge attention (alpha), aggregated
within each seed before taking the three-seed mean.  They are displayed as
percentages; they are not causal effects or prediction importance scores.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from xml.sax.saxutils import escape


KINSHIP_BASE_RELATIONS = frozenset({"child", "spouse", "sibling", "father", "mother"})
DISPLAY_LABELS = ("Culture", "Discovery/Science", "Leadership", "Sports/Games")


@dataclass(frozen=True)
class Cell:
    mean_percent: float
    std_percent: float
    edge_count_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="l1_relation_attention_by_seed.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output SVG path")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional audited matrix CSV")
    parser.add_argument("--experiment", default="rgat_baseline", help="Experiment name in the input CSV")
    parser.add_argument("--layer", type=int, default=2, help="Message-passing layer to plot")
    parser.add_argument("--title", default=None, help="Optional figure title")
    return parser.parse_args()


def relation_group(relation: str) -> str:
    return "kinship" if relation.removesuffix("__rev") in KINSHIP_BASE_RELATIONS else "nonkinship"


def load_cells(
    path: Path, experiment: str, layer: int
) -> dict[tuple[str, str, str], Cell]:
    """Aggregate exact-relation alpha means by seed, group, source, and target."""
    seed_totals: dict[tuple[str, str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment"] != experiment or int(row["message_passing_layer"]) != layer:
                continue
            source = row["source_l1"]
            target = row["target_l1"]
            if source not in DISPLAY_LABELS or target not in DISPLAY_LABELS:
                continue
            edge_count = int(row["edge_count"])
            if edge_count <= 0:
                continue
            key = (row["seed"], relation_group(row["relation"]), source, target)
            seed_totals[key][0] += edge_count * float(row["attention_mean"])
            seed_totals[key][1] += edge_count

    per_cell: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (_seed, group, source, target), (alpha_sum, edge_count) in seed_totals.items():
        per_cell[(group, source, target)].append((alpha_sum / edge_count, edge_count))

    cells: dict[tuple[str, str, str], Cell] = {}
    for key, values in per_cell.items():
        alpha_values = [alpha for alpha, _ in values]
        edge_counts = [count for _, count in values]
        cells[key] = Cell(
            mean_percent=mean(alpha_values) * 100.0,
            std_percent=(stdev(alpha_values) * 100.0) if len(alpha_values) > 1 else 0.0,
            edge_count_mean=mean(edge_counts),
        )

    if not cells:
        raise ValueError(f"No matching {experiment!r}, layer {layer} rows found in {path}")
    return cells


def write_matrix_csv(path: Path, cells: dict[tuple[str, str, str], Cell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "relation_group",
                "source_l1",
                "target_l1",
                "mean_attention_percent",
                "seed_std_percent",
                "mean_edge_count_per_seed",
            ),
        )
        writer.writeheader()
        for group in ("kinship", "nonkinship"):
            for source in DISPLAY_LABELS:
                for target in DISPLAY_LABELS:
                    cell = cells.get((group, source, target))
                    if cell is not None:
                        writer.writerow(
                            {
                                "relation_group": group,
                                "source_l1": source,
                                "target_l1": target,
                                "mean_attention_percent": f"{cell.mean_percent:.8f}",
                                "seed_std_percent": f"{cell.std_percent:.8f}",
                                "mean_edge_count_per_seed": f"{cell.edge_count_mean:.2f}",
                            }
                        )


def display_label(label: str) -> str:
    return label.replace("/", "/\n")


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#202020",
) -> str:
    """Return one SVG text element, supporting the label line breaks used here."""
    lines = text.split("\n")
    line_height = size * 1.12
    first_y = y - line_height * (len(lines) - 1) / 2
    spans = "".join(
        f'<tspan x="{x:.2f}" y="{first_y + index * line_height:.2f}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size:.2f}" font-weight="{weight}" fill="{fill}">{spans}</text>'
    )


def plot(
    cells: dict[tuple[str, str, str], Cell],
    output: Path,
    title: str,
) -> None:
    row_keys = [(source, group) for source in DISPLAY_LABELS for group in ("kinship", "nonkinship")]
    values = [
        [cells.get((group, source, target)) for target in DISPLAY_LABELS]
        for source, group in row_keys
    ]
    finite_values = [cell.mean_percent for row in values for cell in row if cell is not None]
    if not finite_values:
        raise ValueError("The selected data contains no plottable cells")
    global_max = max(finite_values)
    width, height = 1320, 800
    table_x, table_y, column_width, row_height = 440, 174, 204, 59
    table_width, table_height = column_width * len(DISPLAY_LABELS), row_height * len(row_keys)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title>",
        '<desc id="desc">Matrix of direct-edge RGAT attention, comparing kinship and nonkinship relations across source and target occupation categories.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        svg_text(table_x, 54, title, size=25, anchor="start", weight="500"),
    ]

    # Header and table rules recreate the compact OCI-table presentation.
    for column_index, target in enumerate(DISPLAY_LABELS):
        lines.append(svg_text(table_x + (column_index + 0.5) * column_width, 128, display_label(target), size=18))
    lines.append(f'<line x1="55" y1="{table_y}" x2="{table_x + table_width}" y2="{table_y}" stroke="#505050" stroke-width="1.4"/>')
    for row_index in range(len(row_keys) + 1):
        y = table_y + row_index * row_height
        lines.append(f'<line x1="55" y1="{y}" x2="{table_x + table_width}" y2="{y}" stroke="#b0b0b0" stroke-width="0.8"/>')
    for row_index in range(2, len(row_keys), 2):
        y = table_y + row_index * row_height
        lines.append(f'<line x1="55" y1="{y}" x2="{table_x + table_width}" y2="{y}" stroke="#505050" stroke-width="1.4"/>')
    for column_index in range(len(DISPLAY_LABELS) + 1):
        x = table_x + column_index * column_width
        lines.append(f'<line x1="{x}" y1="{table_y}" x2="{x}" y2="{table_y + table_height}" stroke="#d0d0d0" stroke-width="0.6"/>')

    for row_index, (source, group) in enumerate(row_keys):
        center_y = table_y + (row_index + 0.5) * row_height
        if group == "kinship":
            lines.append(svg_text(72, center_y + row_height / 2, display_label(source), size=17, anchor="start"))
        lines.append(svg_text(365, center_y, group, size=16))

        row_cells = values[row_index]
        row_values = [cell.mean_percent for cell in row_cells if cell is not None]
        row_max = max(row_values) if row_values else None
        row_min = min(row_values) if row_values else None
        bar_color = "#d66546" if group == "kinship" else "#5b9bc7"
        for column_index in range(len(DISPLAY_LABELS)):
            cell = row_cells[column_index]
            cell_x = table_x + column_index * column_width
            if cell is None:
                lines.append(svg_text(cell_x + column_width / 2, center_y, "—", size=18, fill="#666666"))
                continue
            value = cell.mean_percent
            # A compact bottom bar makes the comparison visible without hiding values.
            bar_width = (column_width - 30) * value / global_max if global_max else 0.0
            lines.append(
                f'<rect x="{cell_x + 15:.2f}" y="{center_y + 13:.2f}" width="{bar_width:.2f}" height="10" '
                f'fill="{bar_color}" fill-opacity="0.82"/>'
            )
            is_max = value == row_max
            is_min = value == row_min and row_min != row_max
            label = f"{value:.2f}" + ("*" if is_max else "")
            lines.append(svg_text(cell_x + column_width / 2, center_y - 7, label, size=17))
            if is_min:
                underline_y = center_y + 3
                lines.append(f'<line x1="{cell_x + 72:.2f}" y1="{underline_y:.2f}" x2="{cell_x + column_width - 72:.2f}" y2="{underline_y:.2f}" stroke="#202020" stroke-width="1.2"/>')

    footer_y = table_y + table_height + 42
    lines.extend(
        [
            svg_text(
                table_x,
                footer_y,
                "Values: edge-count-weighted mean direct-edge attention (%, three-seed mean).",
                size=13,
                anchor="start",
            ),
            svg_text(
                table_x,
                footer_y + 24,
                "* = row maximum; underline = row minimum.",
                size=13,
                anchor="start",
            ),
            svg_text(
                table_x,
                footer_y + 48,
                "Kinship: child, spouse, sibling, father, mother, plus their generated reverse edges.",
                size=13,
                anchor="start",
            ),
            svg_text(
                table_x,
                footer_y + 72,
                "Nonkinship: all other exact relations in the export.",
                size=13,
                anchor="start",
            ),
            "</g></svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output.suffix.casefold() != ".svg":
        raise ValueError("--output must use the .svg extension; SVG is generated without external plotting dependencies")
    cells = load_cells(args.input, args.experiment, args.layer)
    if args.csv_output is not None:
        write_matrix_csv(args.csv_output, cells)
    display_experiment = args.experiment.removeprefix("rgat_").replace("_", " ")
    title = args.title or f"Kinship vs nonkinship RGAT attention ({display_experiment}, layer {args.layer})"
    plot(cells, args.output, title)


if __name__ == "__main__":
    main()
