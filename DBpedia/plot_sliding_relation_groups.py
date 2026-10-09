#!/usr/bin/env python3
"""Plot one unsmoothed annual Layer-0 retained-message-share curve per group.

Input is an extracted package_sliding_results.py bundle. No bootstrap or model
loading. Missing sampled support is a gap, not an observed zero retention.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dbpedia_sliding_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

LABELS = {
    "inherited": "Inherited ties",
    "intimate_partnership": "Intimate partnership",
    "education_mentorship": "Education / mentorship",
    "professional_collaboration": "Professional collaboration",
    "influence_succession": "Influence / succession",
    "religious_authority_recognition": "Religious authority / recognition",
    "other_acquired": "Other acquired ties",
}
COLORS = ["#2b6f9c", "#dc7e36", "#469477", "#9973b2", "#be5e68", "#9a8344", "#647c88"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"Inconsistent {label}: {a} versus {b}")


def rate(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Invalid probability/share: {value}")
    return number


def collect(root, start_year, end_year):
    root = Path(root).resolve()
    bundle = read_json(root / "bundle.json")
    if bundle["bundle_format"] != "dbpedia-sliding-reports-v1":
        raise ValueError("Unsupported report bundle")
    config = read_json(root / "config/experiment.json")
    calendar = read_json(root / config["period_config"])
    taxonomy = read_json(root / config["multi_group_taxonomy"])
    if set(taxonomy["groups"]) != set(LABELS):
        raise ValueError("Expected the frozen seven-group taxonomy")
    periods = {p["id"]: p for p in calendar["periods"]}
    if end_year < start_year:
        raise ValueError("End year precedes start year")
    seed = config["seed"]
    mask = root / config["graphmask_root"]
    matrix = read_json(mask / "matrix_summary.json")
    if read_json(mask / "pipeline_failures.json"):
        raise ValueError("Bundle contains pipeline failures")
    tidy, audit = [], []
    for year in range(start_year, end_year + 1):
        context = f"center_{year}"
        if context not in bundle["contexts"] or context not in periods:
            raise ValueError(f"Missing annual window: {context}")
        period = periods[context]
        if period["start"] + period["end"] != 2 * year:
            raise ValueError(f"Unexpected center/bounds: {period}")
        summary = [r for r in matrix if r["context"] == context and r["representation"] == "multi_group" and int(r["seed"]) == seed]
        if len(summary) != 1 or summary[0]["status"] != "complete":
            raise ValueError(f"Incomplete/duplicate matrix row: {context}")
        summary = summary[0]
        run = mask / context / "multi_group" / f"seed_{seed}"
        report = run / "test_report"
        metrics = read_json(report / "test_metrics.json")
        manifest = read_json(report / "manifest.json")
        probe = read_json(run / "manifest.json")
        validation = read_json(run / "validation.json")
        artifact = root / config["relation_root"] / context / "multi_group"
        split = read_json(artifact / "split_summary.json")
        if metrics["split"] != "test" or metrics["roots"] != split["test_nodes"] or metrics["labeled_roots"] != split["test_nodes"]:
            raise ValueError(f"Test roots/split mismatch: {context}")
        if manifest["model_name"] != "rgcn" or manifest["split"] != "test" or manifest["sampling_seed"] != seed:
            raise ValueError(f"Model/sampling mismatch: {context}")
        fanouts = [int(n) for n in config["train"]["num_neighbors"].split(",")]
        if manifest["fanouts"] != probe["fanouts"] or manifest["fanouts"] != fanouts:
            raise ValueError(f"Fanouts mismatch: {context}")
        threshold = probe["training_config"]["max_relative_macro_f1_diff"]
        close(threshold, config["graphmask_train"]["max_relative_macro_f1_diff"], "fidelity threshold")
        if validation["relative_macro_f1_difference"] > threshold:
            raise ValueError(f"Validation fidelity failed: {context}")
        selected = probe["selected_checkpoint"]
        policy = config["graphmask_train"]["checkpoint_selection"]
        if selected["policy"] != policy or manifest["selected_checkpoint"] != selected or manifest["enabled_layers"] != selected["enabled_layers"]:
            raise ValueError(f"Checkpoint metadata mismatch: {context}")
        history = read_json(run / "training_history.json")["history"]
        candidates = [r for r in history if r["validation"]["relative_macro_f1_difference"] <= threshold
                      and r["validation"]["hard_retention_rate"] is not None
                      and (policy != "all-layers-enabled" or r["enabled_through_layer"] == 0)]
        if not candidates:
            raise ValueError(f"No eligible checkpoint: {context}")
        best = min(candidates, key=lambda r: r["validation"]["hard_retention_rate"])
        enabled = [i >= best["enabled_through_layer"] for i in range(config["train"]["num_layers"])]
        if best["global_epoch"] != selected["global_epoch"] or enabled != selected["enabled_layers"]:
            raise ValueError(f"Checkpoint selection disagrees with training history: {context}")
        close(best["validation"]["hard_retention_rate"], validation["hard_retention_rate"], "selected validation retention")
        layers = [r for r in metrics["layers"] if int(r["layer"]) == 0]
        if len(layers) != 1:
            raise ValueError(f"Missing/duplicate Layer 0 metrics: {context}")
        layer = layers[0]
        observations = int(layer["message_observations"])
        retained = observations * rate(layer["hard_retention_rate"])
        if observations <= 0 or retained <= 0:
            raise ValueError(f"Layer 0 share denominator is empty: {context}")
        with (report / "relations_base.csv").open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if int(r["layer"]) == 0]
        observed = {r["relation"]: r for r in rows}
        if len(rows) != len(observed) or set(observed) - set(LABELS):
            raise ValueError(f"Duplicate/unknown groups: {context}")
        if sum(int(r["message_observations"]) for r in rows) != observations:
            raise ValueError(f"Message counts mismatch: {context}")
        reconstructed = sum(int(r["message_observations"]) * rate(r["hard_retention_rate"]) for r in rows)
        if not math.isclose(reconstructed, retained, rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError(f"Retained counts mismatch: {context}")
        close(sum(rate(r["retained_edge_share"]) for r in rows), 1, "sum of shares")
        with (artifact / "relation_stats.csv").open(newline="", encoding="utf-8") as handle:
            counts = {r["relation"]: int(r["count"]) for r in csv.DictReader(handle) if not r["relation"].endswith("__rev")}
        close(summary["layer0_hard_retention_rate"], layer["hard_retention_rate"], "summary Layer 0 retention")
        audit.append({"year": year, "context": context, "window_start": period["start"], "window_end": period["end"],
                      "selected_epoch": selected["global_epoch"], "layer0_gate_enabled": enabled[0],
                      "layer0_hard_retention_rate": layer["hard_retention_rate"], "message_observations": observations,
                      "hard_retained_messages": round(retained), "test_nodes": metrics["roots"],
                      "prediction_agreement": metrics["prediction_agreement"], "masked_accuracy": metrics["masked"]["accuracy"],
                      "masked_macro_f1": metrics["masked"]["macro_f1"], "checkpoint_selection": policy})
        for group in LABELS:
            row = observed.get(group)
            count = int(row["message_observations"]) if row else 0
            group_retained = count * rate(row["hard_retention_rate"]) if row else 0
            share = rate(row["retained_edge_share"]) if row else 0
            close(share, group_retained / retained, f"share denominator: {context}/{group}")
            graph_count = counts.get(group, 0)
            if count and not graph_count:
                raise ValueError(f"Messages without graph support: {context}/{group}")
            status = "observed" if count else "no_sampled_messages" if graph_count else "no_graph_edges"
            tidy.append({"year": year, "context": context, "window_start": period["start"], "window_end": period["end"],
                         "group": group, "original_triples_after_collapse": graph_count, "message_observations": count,
                         "hard_retained_messages": round(group_retained), "retained_message_share": share,
                         "retained_message_share_percent": 100 * share, "support_status": status,
                         "plot_share_percent": 100 * share if count else None,
                         "layer0_gate_enabled": enabled[0], "seed": seed})
    return tidy, audit, {"bundle": bundle, "experiment": config, "taxonomy": taxonomy,
                         "metric": "Layer 0 hard-retained sampled message share",
                         "definition": "group hard-retained sampled messages / all Layer 0 hard-retained sampled messages",
                         "reverse_policy": "Original and generated reverse messages combined in each base group",
                         "missing_policy": "No sampled support: plot gap; accounting share is zero",
                         "smoothing": None, "bootstrap": False, "confidence_interval": None,
                         "center_years": [start_year, end_year], "received_root": str(root)}


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(tidy, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})

    def draw(ax, group, color):
        rows = [r for r in tidy if r["group"] == group]
        years = [r["year"] for r in rows]
        values = [float("nan") if r["plot_share_percent"] is None else r["plot_share_percent"] for r in rows]
        ax.plot(years, values, color=color, linewidth=1.7)
        disabled = [r for r in rows if not r["layer0_gate_enabled"] and r["plot_share_percent"] is not None]
        if disabled:
            ax.scatter([r["year"] for r in disabled], [r["plot_share_percent"] for r in disabled], s=15,
                       facecolors="white", edgecolors=color, linewidths=0.8, zorder=3, label="Layer 0 gate disabled")
            ax.legend(loc="best", frameon=False, fontsize=8)
        ax.set_title(LABELS[group], loc="left", fontweight="bold")
        ax.set_xlabel("Window center year")
        ax.set_ylabel("Retained message share (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_xlim(min(years), max(years)) if len(years) > 1 else ax.set_xlim(years[0] - 1, years[0] + 1)
        finite = [v for v in values if math.isfinite(v)]
        ax.set_ylim(0, min(100, max(0.2, max(finite, default=0) * 1.15)))

    def save(fig, name):
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for (group, _), color in zip(LABELS.items(), COLORS):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        draw(ax, group, color)
        fig.suptitle("DBpedia annual sliding windows — Layer 0", fontsize=12)
        fig.tight_layout()
        save(fig, f"layer0_{group}_annual")
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for ax, group, color in zip(axes.flat, LABELS, COLORS):
        draw(ax, group, color)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.7, "Raw annual results; no bootstrap or smoothing.\nEach panel uses its own y-axis scale.\nGroups sum to 100% each year.\nOpen circles: Layer 0 gate disabled.\nGaps: no sampled support.", transform=axes.flat[-1].transAxes, va="top", linespacing=1.7)
    fig.suptitle("DBpedia relation-group trends — Layer 0", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2)
    save(fig, "layer0_all_groups_annual")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/dbpedia_sliding_layer0_20th_century"))
    parser.add_argument("--start-year", type=int, default=1901)
    parser.add_argument("--end-year", type=int, default=2000)
    args = parser.parse_args()
    tidy, audit, provenance = collect(args.received_root, args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "layer0_group_annual.tsv", tidy)
    write_tsv(args.output_dir / "annual_audit.tsv", audit)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(tidy, args.output_dir)
    disabled = sum(not row["layer0_gate_enabled"] for row in audit)
    print(f"Validated {len(audit)} years, seven groups; Layer 0 disabled in {disabled} years")
    print(f"Saved seven group plots + overview (PNG/SVG/PDF): {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
