#!/usr/bin/env python3
"""Check completed pilot protocols and metrics without loading graphs or weights."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Wikidata.grouped_pipeline import Pipeline
from DBpedia.grouped_pipeline import read_json, write_json, MODEL_FILES, PROBE_FILES, REPORT_FILES


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid metric: {value}")
    return value


def validate(config, periods, dataset="wikidata"):
    pipeline_class = Pipeline
    if dataset == "freebase":
        from Freebase.grouped_pipeline import Pipeline as pipeline_class
    elif dataset != "wikidata":
        raise ValueError(f"Unsupported dataset: {dataset}")
    p = pipeline_class(SimpleNamespace(config=config, stage="summarize", device="cuda:0", periods=periods,
                                 representations="multi_group", include_full=False))
    if dataset == "freebase":
        from Freebase.prepare_graph import settings
        source_dir = p.path(p.config["source_data"]).parent
        _, _, _, signature = settings(config, source_dir, False)
        required = ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json", "source_prepare_manifest.json")
        if any(not (source_dir / name).is_file() or (source_dir / name).stat().st_size == 0 for name in required):
            raise ValueError("Incomplete Freebase source")
        if read_json(source_dir / "source_prepare_manifest.json") != signature:
            raise ValueError("Freebase source provenance changed")
    else:
        p.require_source()
    results = []
    for context in p.contexts:
        artifact, model, mask = p.directories(context, "multi_group")
        p.validate_period(context)
        commands = p.commands(context, "multi_group")
        data = artifact / "graph_data.pt"
        p.require_completed("collapse", commands[0], artifact, p.collapse_outputs("multi_group"), p.collapse_inputs(context), p.collapse_settings("multi_group"))
        p.require_completed("train", commands[1], model, MODEL_FILES, [data, artifact / "collapsed_tie_taxonomy.json"])
        p.require_completed("probe", commands[2], mask, PROBE_FILES, [data, model / "best_model.pt"])
        p.require_completed("report", commands[3], mask / "test_report", REPORT_FILES, [data, model / "best_model.pt", mask / "graphmask_probe.pt", artifact / "nodes.csv"])
        split = read_json(artifact / "split_summary.json")
        imputation = split["period_induced_artifact"]["life_date_imputation"]
        if imputation["rule"] != {"born_after": 1920, "alive_through": 2026} or not imputation["raw_death_year_and_temporal_features_preserved"]:
            raise ValueError(f"Wrong alive rule: {context}")
        validation = read_json(mask / "validation.json")
        threshold = p.config["graphmask_train"]["max_relative_macro_f1_diff"]
        difference = float(validation["relative_macro_f1_difference"])
        if not math.isfinite(difference) or difference > threshold:
            raise ValueError(f"Validation fidelity failed: {context}")
        probe = read_json(mask / "manifest.json")
        report = read_json(mask / "test_report/manifest.json")
        if Path(probe["source_checkpoint"]).resolve() != (model / "best_model.pt").resolve() or Path(report["source_checkpoint"]).resolve() != (model / "best_model.pt").resolve() or Path(report["probe"]).resolve() != (mask / "graphmask_probe.pt").resolve():
            raise ValueError(f"Wrong model/probe source: {context}")
        if probe["source_checkpoint_sha256"] != report["source_checkpoint_sha256"] or probe["data_sha256"] != report["data_sha256"]:
            raise ValueError(f"Recorded source fingerprints disagree: {context}")
        selected = probe["selected_checkpoint"]
        if selected["policy"] != "any-stage" or report["selected_checkpoint"] != selected or report["enabled_layers"] != selected["enabled_layers"]:
            raise ValueError(f"Checkpoint policy mismatch: {context}")
        history = read_json(mask / "training_history.json")["history"]
        candidates = [h for h in history if h["validation"]["relative_macro_f1_difference"] <= threshold and h["validation"]["hard_retention_rate"] is not None]
        best = min(candidates, key=lambda h: h["validation"]["hard_retention_rate"])
        if selected["global_epoch"] != best["global_epoch"]:
            raise ValueError(f"Wrong selected checkpoint: {context}")
        fanouts = [int(n) for n in p.config["train"]["num_neighbors"].split(",")]
        if report["model_name"] != "rgcn" or report["split"] != "test" or report["sampling_seed"] != p.config["seed"] or report["fanouts"] != fanouts or probe["fanouts"] != fanouts:
            raise ValueError(f"Test sampling mismatch: {context}")
        metrics = read_json(mask / "test_report/test_metrics.json")
        if metrics["split"] != "test" or metrics["roots"] != split["test_nodes"] or metrics["labeled_roots"] != split["test_nodes"]:
            raise ValueError(f"Test roots mismatch: {context}")
        for values in (metrics["original"], metrics["masked"]):
            for key in ("accuracy", "macro_f1"):
                probability(values[key])
        probability(metrics["prediction_agreement"])
        probability(metrics["hard_retention_rate"])
        layer = next(r for r in metrics["layers"] if r["layer"] == 0)
        observations = int(layer["message_observations"])
        retained = observations * probability(layer["hard_retention_rate"])
        with (mask / "test_report/relations_base.csv").open(newline="") as f:
            groups = [r for r in csv.DictReader(f) if int(r["layer"]) == 0]
        names = [r["relation"] for r in groups]
        if len(names) != len(set(names)) or set(names) - set(p.multi["groups"]):
            raise ValueError(f"Wrong relation groups: {context}")
        actual_retained = sum(int(r["message_observations"]) * probability(r["hard_retention_rate"]) for r in groups)
        if retained <= 0 or sum(int(r["message_observations"]) for r in groups) != observations or not math.isclose(actual_retained, retained, rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError(f"Wrong Layer 0 counts: {context}")
        if not math.isclose(sum(probability(r["retained_edge_share"]) for r in groups), 1, abs_tol=1e-8):
            raise ValueError(f"Wrong group share total: {context}")
        with (p.source(context).parent / "nodes.csv").open(newline="") as f:
            checked = 0
            for row in csv.DictReader(f):
                birth = float(row["birth_year"]) if row["birth_year"] else float("nan")
                death = float(row["death_year"]) if row["death_year"] else float("nan")
                imputed = row["life_period_death_year_imputed"].lower() == "true"
                expected = math.isfinite(birth) and birth > 1920 and not math.isfinite(death)
                if imputed != expected or imputed and float(row["life_period_effective_death_year"]) != 2026:
                    raise ValueError(f"Raw/imputed dates disagree: {context}/{row['node_id']}")
                checked += 1
        if checked != split["nodes"]:
            raise ValueError(f"Node count mismatch: {context}")
        results.append({"context": context, "nodes": split["nodes"], "test_nodes": metrics["roots"],
                        "original": metrics["original"], "masked": metrics["masked"],
                        "prediction_agreement": metrics["prediction_agreement"], "hard_retention_rate": metrics["hard_retention_rate"],
                        "layer0_hard_retention_rate": layer["hard_retention_rate"], "enabled_layers": selected["enabled_layers"],
                        "validation_relative_macro_f1_difference": difference, "date_rows_checked": checked})
    return {"status": "passed", "contexts": p.contexts, "checks": "stage contracts, dates, split, sampling, checkpoint, fidelity, finite metrics, message counts and shares", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("wikidata", "freebase"), default="wikidata")
    parser.add_argument("--periods", default="center_1900,center_1950,center_2000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.config, args.periods, args.dataset)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
