#!/usr/bin/env python3
"""Plot raw annual GraphMask seed comparisons with a fixed seed-42 RGCN.

Requires the complete 1900–1910, GraphMask seed 42/43/44, five-group TSV.
No smoothing, bootstrap, confidence intervals, model loading, or training.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.plot_sliding_relation_groups import GROUP_COLORS, group_labels
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

SEEDS = (42, 43, 44)
YEARS = tuple(range(1900, 1911))
REQUIRED = (
    "graphmask_seed", "year", "group", "retained_share_pct", "group_keep_pct",
    "sampled_share_pct", "layer0_keep_pct", "original_macro_f1", "masked_macro_f1",
    "prediction_agreement", "layer0_enabled", "checkpoint_epoch", "report_seed", "model_seed",
)
METRICS = {
    "retained_share_pct": "Share of retained Layer 0 messages",
    "group_keep_pct": "Within-group Layer 0 retention",
    "sampled_share_pct": "Share of sampled Layer 0 messages",
}
STYLES = {42: ("-", "o"), 43: ("--", "s"), 44: (":", "^")}


def read_rows(path, groups):
    """Reject missing/duplicate observations and inconsistent frozen-model metadata."""
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing TSV columns: {sorted(missing)}")
        for line, raw in enumerate(reader, 2):
            row = dict(raw)
            try:
                for field in ("graphmask_seed", "year", "checkpoint_epoch", "report_seed", "model_seed"):
                    row[field] = int(raw[field])
                for field in ("retained_share_pct", "group_keep_pct", "sampled_share_pct", "layer0_keep_pct",
                              "original_macro_f1", "masked_macro_f1", "prediction_agreement"):
                    row[field] = float(raw[field])
                    upper = 100 if field.endswith("_pct") else 1
                    if not math.isfinite(row[field]) or not 0 <= row[field] <= upper:
                        raise ValueError(f"{field} must be finite in [0, {upper}]")
                enabled = raw["layer0_enabled"].strip().lower()
                if enabled not in ("true", "false", "1", "0"):
                    raise ValueError("layer0_enabled must be true/false or 1/0")
                row["layer0_enabled"] = enabled in ("true", "1")
                key = (row["graphmask_seed"], row["year"], row["group"])
                if key[0] not in SEEDS or key[1] not in YEARS or key[2] not in groups:
                    raise ValueError(f"Unexpected seed/year/group: {key}")
                if key in rows:
                    raise ValueError(f"Duplicate observation: {key}")
                if row["model_seed"] != 42 or row["checkpoint_epoch"] < 0:
                    raise ValueError("Requires model_seed=42 and nonnegative checkpoint_epoch")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"TSV line {line}: {exc}") from exc
            rows[key] = row
    expected = {(seed, year, group) for seed in SEEDS for year in YEARS for group in groups}
    if expected - rows.keys():
        raise ValueError(f"Missing {len(expected - rows.keys())} seed/year/group observations")
    shared = ("layer0_keep_pct", "original_macro_f1", "masked_macro_f1", "prediction_agreement",
              "layer0_enabled", "checkpoint_epoch", "report_seed", "model_seed")
    for year in YEARS:
        for seed in SEEDS:
            batch = [rows[seed, year, group] for group in groups]
            for field in shared:
                if any(r[field] != batch[0][field] for r in batch):
                    raise ValueError(f"Inconsistent {field} across groups: seed {seed}, year {year}")
            for field in ("retained_share_pct", "sampled_share_pct"):
                if not math.isclose(sum(r[field] for r in batch), 100, abs_tol=0.02):
                    raise ValueError(f"{field} must sum to 100: seed {seed}, year {year}")
        baseline = rows[42, year, next(iter(groups))]
        for seed in (43, 44):
            row = rows[seed, year, next(iter(groups))]
            if row["report_seed"] != baseline["report_seed"]:
                raise ValueError(f"Report sampling seed differs across GraphMask seeds in {year}")
            if not math.isclose(row["original_macro_f1"], baseline["original_macro_f1"], abs_tol=1e-8):
                raise ValueError(f"Original model F1 differs across GraphMask seeds in {year}")
    return rows


def curve(ax, rows, seed, group, metric, label, distinguish_seed=True):
    batch = [rows[seed, year, group] for year in YEARS]
    values = [row[metric] for row in batch]
    style, marker = STYLES[seed] if distinguish_seed else ("-", "o")
    color = GROUP_COLORS[group]
    ax.plot(YEARS, values, linestyle=style, color=color, linewidth=1.7, label=label)
    for enabled in (True, False):
        indices = [i for i, row in enumerate(batch) if row["layer0_enabled"] == enabled]
        ax.scatter([YEARS[i] for i in indices], [values[i] for i in indices], marker=marker,
                   s=30, facecolors=color if enabled else "white", edgecolors=color,
                   linewidths=1.2, zorder=3)


def axes_style(ax, metric, upper=100):
    ax.set_xlim(1899.7, 1910.3)
    ax.set_ylim(0, upper)
    ax.set_xticks(YEARS[::2])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Window center year")
    ax.set_ylabel(METRICS[metric] + " (%)")
    ax.grid(axis="y", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, output, stem):
    fig.text(0.5, 0.012, "Raw annual values · RGCN seed 42 · Hollow markers: Layer 0 gate disabled",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, default="retained_share_pct")
    args = parser.parse_args()
    taxonomy = json.loads((ROOT / "config/freebase_tie_taxonomy_acquired_subgroups_v1.json").read_text())
    labels = group_labels(taxonomy)
    rows = read_rows(args.input_tsv, labels)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "svg.fonttype": "none"})
    handles = [Line2D([], [], color="#444444", linestyle=STYLES[s][0], marker=STYLES[s][1],
                      label=f"GraphMask seed {s}") for s in SEEDS]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (group, label) in zip(axes.flat, labels.items()):
        for seed in SEEDS:
            curve(ax, rows, seed, group, args.metric, f"Seed {seed}")
        upper=min(100,max(0.1,max(rows[s,y,group][args.metric] for s in SEEDS for y in YEARS)*1.15))
        axes_style(ax, args.metric, upper)
        ax.set_title(label)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False)
    fig.suptitle("Freebase: GraphMask seeds with fixed RGCN (1900–1910)")
    save(fig, output, f"freebase_frozen_rgcn_{args.metric}_by_relation")
    for group, label in labels.items():
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for seed in SEEDS:
            curve(ax, rows, seed, group, args.metric, f"GraphMask seed {seed}")
        upper=min(100,max(0.1,max(rows[s,y,group][args.metric] for s in SEEDS for y in YEARS)*1.15))
        axes_style(ax, args.metric, upper)
        ax.legend(handles=handles, frameon=False)
        fig.suptitle(f"Freebase: {label}")
        save(fig, output, f"freebase_frozen_rgcn_{args.metric}_{group}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for group, label in labels.items():
            curve(ax, rows, seed, group, args.metric, label, distinguish_seed=False)
        axes_style(ax, args.metric)
        ax.set_title(f"GraphMask seed {seed}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Freebase: relation groups within each GraphMask seed (RGCN seed 42)")
    save(fig, output, f"freebase_frozen_rgcn_{args.metric}_by_seed")
    print(f"Validated {len(rows)} observations; wrote 7 figures in PNG/PDF/SVG to {output.resolve()}")


if __name__ == "__main__":
    main()
