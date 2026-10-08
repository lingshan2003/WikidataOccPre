#!/usr/bin/env python3
"""Plot Layer-0 retained-message shares from a downloaded DBpedia report bundle.

No checkpoints are loaded. Reported hard-retained shares are reconciled with
message counts; generated reverse directions follow their base group. Archive
contents should be extracted into an isolated received-root beforehand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dbpedia_layer0_matplotlib" if Path("/private/tmp").is_dir() else "/tmp/dbpedia_layer0_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np


MULTI_LABELS = {
    "inherited": "inherited",
    "intimate_partnership": "intimate\npartnership",
    "education_mentorship": "education /\nmentorship",
    "professional_collaboration": "professional\ncollaboration",
    "influence_succession": "influence /\nsuccession",
    "religious_authority_recognition": "religious\nauthority /\nrecognition",
    "other_acquired": "other\nacquired",
}
BINARY_LABELS = {"inherited_ties": "inherited", "acquired_ties": "acquired"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def period_label(period):
    start, end = period.get("start"), period.get("end")
    if start is None:
        return f"≤{end}"
    if end is None:
        return f"≥{start}"
    return f"{start}–{end}"


def collect(root, seed, supplement_root=None):
    root = Path(root).resolve()
    resolved = read_json(root / "runs/dbpedia_grouped_20y_v1/grouped_pipeline_resolved_config.json")
    period_config = read_json(root / "config/dbpedia_life_periods_20y_v1.json")
    taxonomy = read_json(root / "config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json")
    periods = period_config["periods"]
    if set(taxonomy["groups"]) != set(MULTI_LABELS):
        raise ValueError("Expected the frozen seven-group DBpedia taxonomy")
    mask_root = root / "runs_graphmask/dbpedia_grouped_20y_v1"
    artifact_root = root / "artifacts/dbpedia_grouped_20y_v1"
    matrix_rows = read_json(mask_root / "matrix_summary.json")
    indexed = {(r["context"], r["representation"]): r for r in matrix_rows}
    supplement_index = {}
    supplement_config = None
    if supplement_root is not None:
        supplement_root = Path(supplement_root).resolve()
        supplement_config = read_json(supplement_root / "grouped_pipeline_resolved_config.json")
        expected_periods = {"through_1500", "1501_1900", "1941_1960", "1981_2000", "since_2001"}
        if set(supplement_config["periods"]) != expected_periods or supplement_config["representations"] != ["multi_group"]:
            raise ValueError("Expected exactly five multi-group supplement periods")
        if supplement_config["graphmask_train"]["checkpoint_selection"] != "all-layers-enabled" or supplement_config["seed"] != seed:
            raise ValueError("Unexpected supplement selection policy/seed")
        supplement_rows = read_json(supplement_root / "matrix_summary.json")
        supplement_index = {(r["context"], r["representation"]): r for r in supplement_rows}
        if len(supplement_rows) != 5 or set(supplement_index) != {(p, "multi_group") for p in expected_periods}:
            raise ValueError("Missing/extra supplement reports")
        if read_json(supplement_root / "pipeline_failures.json"):
            raise ValueError("Supplement pipeline recorded failures")
    tidy, audit, inputs = [], [], []
    for representation, labels in (("binary", BINARY_LABELS), ("multi_group", MULTI_LABELS)):
        expected = set(labels)
        for period in periods:
            period_id = period["id"]
            replacement = (period_id, representation) in supplement_index
            summary = (supplement_index if replacement else indexed)[(period_id, representation)]
            if summary["status"] != "complete" or int(summary["seed"]) != seed:
                raise ValueError(f"Incomplete or differently seeded run: {period_id}/{representation}")
            original_run_dir = mask_root / period_id / representation / f"seed_{seed}"
            run_dir = (supplement_root if replacement else mask_root) / period_id / representation / f"seed_{seed}"
            report_dir = run_dir / "test_report"
            metrics = read_json(report_dir / "test_metrics.json")
            manifest = read_json(report_dir / "manifest.json")
            validation = read_json(run_dir / "validation.json")
            probe_manifest = read_json(run_dir / "manifest.json")
            history = read_json(run_dir / "training_history.json")["history"]
            policy = "all-layers-enabled" if replacement else "any-stage"
            threshold = probe_manifest["training_config"]["max_relative_macro_f1_diff"]
            candidates = [r for r in history if r["validation"]["relative_macro_f1_difference"] <= threshold
                          and r["validation"]["hard_retention_rate"] is not None
                          and (not replacement or r["enabled_through_layer"] == 0)]
            if not candidates:
                raise ValueError(f"No eligible checkpoint: {run_dir}")
            selected = min(candidates, key=lambda r: r["validation"]["hard_retention_rate"])
            enabled = [i >= selected["enabled_through_layer"] for i in range(2)]
            if not math.isclose(selected["validation"]["hard_retention_rate"], validation["hard_retention_rate"], abs_tol=1e-12):
                raise ValueError(f"Selected epoch disagrees with validation: {run_dir}")
            if replacement:
                metadata = probe_manifest.get("selected_checkpoint") or {}
                if metadata.get("policy") != policy or metadata.get("enabled_layers") != [True, True] or metadata.get("global_epoch") != selected["global_epoch"]:
                    raise ValueError(f"Invalid supplement selected checkpoint: {run_dir}")
                if manifest.get("enabled_layers") != [True, True] or manifest.get("selected_checkpoint") != metadata:
                    raise ValueError(f"Report/probe enabled layers disagree: {run_dir}")
                original_manifest = read_json(original_run_dir / "manifest.json")
                for key in ("source_checkpoint", "data", "fanouts", "seed"):
                    if probe_manifest[key] != original_manifest[key]:
                        raise ValueError(f"Changed supplement source/sampling {key}: {run_dir}")
                for key in set(probe_manifest["training_config"]) | set(original_manifest["training_config"]):
                    if key not in ("output_dir", "checkpoint_selection", "device") and probe_manifest["training_config"].get(key) != original_manifest["training_config"].get(key):
                        raise ValueError(f"Changed supplement training option {key}: {run_dir}")
            source = "layer0_enabled_supplement" if replacement else "original"
            artifact_dir = artifact_root / period_id / representation
            split = read_json(artifact_dir / "split_summary.json")
            if metrics["split"] != "test" or metrics["roots"] != split["test_nodes"] or metrics["labeled_roots"] != split["test_nodes"]:
                raise ValueError(f"Test-root mismatch at {report_dir}")
            if manifest["model_name"] != "rgcn" or manifest["split"] != "test" or manifest["sampling_seed"] != seed:
                raise ValueError(f"Unexpected model/split/seed: {report_dir}")
            if manifest["fanouts"] != probe_manifest["fanouts"] or manifest["fanouts"] != [int(n) for n in resolved["train"]["num_neighbors"].split(",")]:
                raise ValueError(f"Inconsistent RGCN/GraphMask sampling at {report_dir}")
            if validation["relative_macro_f1_difference"] > threshold or not math.isclose(threshold, resolved["graphmask_train"]["max_relative_macro_f1_diff"]):
                raise ValueError(f"Validation fidelity setting failed/changed: {run_dir}")
            layer = next(r for r in metrics["layers"] if int(r["layer"]) == 0)
            observations = int(layer["message_observations"])
            retained = observations * float(layer["hard_retention_rate"])
            if not observations or not retained:
                raise ValueError(f"Layer 0 has no observed/retained messages: {report_dir}")
            with (report_dir / "relations_base.csv").open(encoding="utf-8", newline="") as handle:
                layer_rows = [r for r in csv.DictReader(handle) if int(r["layer"]) == 0]
            observed = {r["relation"]: r for r in layer_rows}
            if len(observed) != len(layer_rows) or set(observed) - expected:
                raise ValueError(f"Duplicate/unknown base groups: {report_dir}")
            if sum(int(r["message_observations"]) for r in layer_rows) != observations:
                raise ValueError(f"Layer-0 observation counts disagree: {report_dir}")
            reconstructed = sum(int(r["message_observations"]) * float(r["hard_retention_rate"]) for r in layer_rows)
            if not math.isclose(reconstructed, retained, rel_tol=1e-8, abs_tol=1e-5):
                raise ValueError(f"Layer-0 retained counts disagree: {report_dir}")
            if not math.isclose(sum(float(r["retained_edge_share"]) for r in layer_rows), 1.0, abs_tol=1e-8):
                raise ValueError(f"Shares do not sum to 100%: {report_dir}")
            with (artifact_dir / "relation_stats.csv").open(encoding="utf-8", newline="") as handle:
                graph_counts = {r["relation"]: int(r["count"]) for r in csv.DictReader(handle) if not r["relation"].endswith("__rev")}
            all_retained = math.isclose(float(layer["hard_retention_rate"]), 1.0, abs_tol=1e-10)
            audit.append({"period_id": period_id, "period_label": period_label(period), "representation": representation, "layer": 0, "result_source": source, "checkpoint_selection": policy, "selected_epoch": selected["global_epoch"], "enabled_layers": json.dumps(enabled), "layer0_gate_enabled": enabled[0], "test_roots": metrics["roots"], "message_observations": observations, "hard_retained_messages": round(retained), "layer_hard_retention_rate": layer["hard_retention_rate"], "all_messages_retained": all_retained, "validation_relative_macro_f1_difference": validation["relative_macro_f1_difference"], "validation_fidelity_threshold": threshold, "fanouts": ",".join(map(str, manifest["fanouts"])), "report_dir": str(report_dir)})
            for group in labels:
                row = observed.get(group)
                count = int(row["message_observations"]) if row else 0
                group_retained = count * float(row["hard_retention_rate"]) if row else 0.0
                share = float(row["retained_edge_share"]) if row else 0.0
                if row and not math.isclose(share, group_retained / retained, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Share denominator differs for {period_id}/{group}")
                graph_count = graph_counts.get(group, 0)
                if count and not graph_count:
                    raise ValueError(f"Observed messages without graph support: {period_id}/{group}")
                status = "observed" if count else "no_sampled_messages" if graph_count else "no_graph_edges"
                tidy.append({"period_id": period_id, "period_label": period_label(period), "representation": representation, "seed": seed, "layer": 0, "result_source": source, "selected_epoch": selected["global_epoch"], "layer0_gate_enabled": enabled[0], "group": group, "original_triples_after_collapse": graph_count, "message_observations": count, "hard_retained_messages": round(group_retained), "retained_message_share": share, "retained_message_share_percent": 100 * share, "hard_retention_rate": float(row["hard_retention_rate"]) if row else None, "support_status": status, "layer_message_observations": observations, "layer_hard_retained_messages": round(retained), "layer_hard_retention_rate": layer["hard_retention_rate"], "all_layer_messages_retained": all_retained})
            inputs.extend(str(p) for p in (report_dir / "relations_base.csv", report_dir / "test_metrics.json", report_dir / "manifest.json", run_dir / "validation.json", run_dir / "manifest.json", run_dir / "training_history.json", artifact_dir / "relation_stats.csv", artifact_dir / "split_summary.json"))
    return periods, tidy, audit, {"received_root": str(root), "supplement_root": str(supplement_root) if supplement_root else None, "supplement_periods": list(supplement_config["periods"]) if supplement_config else [], "layer": 0, "seed": seed, "metric": "retained_edge_share", "metric_definition": "group hard-retained sampled messages / all hard-retained sampled messages in Layer 0", "reverse_policy": "base groups include both original and generated reverse message directions", "missing_policy": "grey dash means no sampled messages; zero share used only for reconciliation and full-precision tables", "row_marker": "asterisk means all sampled Layer-0 messages were hard-retained; supplement plots use gates-enabled checkpoints for these five multi-group rows", "life_period_config": period_config, "taxonomy": taxonomy, "resolved_experiment": resolved, "supplement_experiment": supplement_config, "input_paths": inputs}


def plot_heatmap(periods, tidy, audit, representation, output_dir, seed):
    labels = BINARY_LABELS if representation == "binary" else MULTI_LABELS
    groups = list(labels)
    indexed = {(r["period_id"], r["group"]): r for r in tidy if r["representation"] == representation}
    audit_index = {r["period_id"]: r for r in audit if r["representation"] == representation}
    values = np.array([[indexed[(p["id"], g)]["retained_message_share_percent"] for g in groups] for p in periods])
    unavailable = np.array([[indexed[(p["id"], g)]["support_status"] != "observed" for g in groups] for p in periods])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": False})
    binary = representation == "binary"
    figure = plt.figure(figsize=(8.5 if binary else 16, 9.6), facecolor="white")
    axis = figure.add_axes([0.28 if binary else 0.145, 0.175, 0.49 if binary else 0.72, 0.61])
    color_axis = figure.add_axes([0.82 if binary else 0.895, 0.175, 0.03 if binary else 0.023, 0.61])
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#eeeeee")
    normal = Normalize(0, 100)
    heatmap = axis.imshow(np.ma.array(values, mask=unavailable), cmap=cmap, norm=normal, aspect="auto", interpolation="none")
    axis.set_xticks(range(len(groups)), [labels[g] for g in groups], fontsize=13)
    axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=12)
    axis.set_yticks(range(len(periods)), [period_label(p) + (" *" if audit_index[p["id"]]["all_messages_retained"] else "") for p in periods], fontsize=13)
    axis.tick_params(axis="y", pad=16)
    axis.set_ylabel("Life-period window", fontsize=15, labelpad=22)
    axis.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(periods), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#232323")
    for i in range(len(periods)):
        for j in range(len(groups)):
            if unavailable[i, j]:
                text, color = "—", "#666666"
            else:
                text = f"{values[i, j]:.1f}%"
                rgb = cmap(normal(values[i, j]))[:3]
                color = "white" if sum(v * w for v, w in zip(rgb, (0.2126, 0.7152, 0.0722))) < 0.50 else "#252525"
            axis.text(j, i, text, ha="center", va="center", color=color, fontsize=14 if binary else 13)
    colorbar = figure.colorbar(heatmap, cax=color_axis, ticks=np.arange(0, 101, 10))
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    colorbar.ax.tick_params(labelsize=11, length=0, pad=6)
    colorbar.set_label("Global hard-retained message share (%)", fontsize=13, labelpad=15)
    figure.text(0.5, 0.95, "DBpedia relation groups across life periods — Layer 0" if not binary else "DBpedia across life periods — Layer 0", ha="center", va="center", fontsize=23 if not binary else 19)
    supplemented = any(r["result_source"] == "layer0_enabled_supplement" for r in audit_index.values())
    subtitle = "Binary relation vocabulary" if binary else "Multi-group relation vocabulary"
    if supplemented:
        subtitle += " · five-period checkpoint supplement"
    figure.text(0.5, 0.90, subtitle, ha="center", va="center", fontsize=18)
    figure.text(0.04 if binary else 0.145, 0.107, "* All sampled Layer 0 messages retained. Grey cells (—): no sampled messages.", fontsize=10.2, color="#444444")
    figure.text(0.04 if binary else 0.145, 0.077, "Rows sum to 100% over observed groups; original and generated reverse directions combined.", fontsize=9.2 if binary else 10.2, color="#444444")
    fanouts = next(iter(audit_index.values()))["fanouts"]
    figure.text(0.04 if binary else 0.145, 0.047, f"Test-root sampling · seed {seed} · fanouts {fanouts} · life windows may overlap", fontsize=10.2, color="#444444")
    if supplemented:
        figure.text(0.145, 0.022, "New checkpoints: ≤1500, 1501–1900, 1941–1960, 1981–2000, ≥2001 (both layer gates enabled).", fontsize=10.2, color="#444444")
    stem = output_dir / f"dbpedia_{representation}_layer0_retained_message_share"
    for extension in ("png", "svg", "pdf"):
        figure.savefig(stem.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(figure)
    return str(stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-root", type=Path, default=Path("artifacts/dbpedia_visualization_received_2026_10_06"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualization/dbpedia_grouped_layer0_2026_10_06"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer0-supplement-root", type=Path, help="Five-period GraphMask root; replace only these multi-group rows")
    args = parser.parse_args()
    periods, tidy, audit, provenance = collect(args.received_root, args.seed, args.layer0_supplement_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "layer0_relation_groups.tsv", tidy)
    write_tsv(args.output_dir / "layer0_period_audit.tsv", audit)
    for representation in ("multi_group", "binary"):
        plot_heatmap(periods, tidy, audit, representation, args.output_dir, args.seed)
        labels = MULTI_LABELS if representation == "multi_group" else BINARY_LABELS
        pivot = []
        indexed = {(r["period_id"], r["group"]): r for r in tidy if r["representation"] == representation}
        for p in periods:
            pivot.append({"period_id": p["id"], "period_label": period_label(p), **{g: indexed[(p["id"], g)]["retained_message_share_percent"] if indexed[(p["id"], g)]["support_status"] == "observed" else None for g in labels}})
        write_tsv(args.output_dir / f"{representation}_layer0_heatmap_values_percent.tsv", pivot)
    (args.output_dir / "figure_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(audit)} reports; plotted Layer 0 only: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
