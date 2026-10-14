#!/usr/bin/env python3
"""Validate completed DBpedia BHHT windows before expanding the pilot."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.bhht_l1_pipeline import Pipeline
from DBpedia.grouped_pipeline import read_json, write_json

DEFAULT_CONFIG = "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value, label):
    require(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid {label}: {value!r}")


def validate(config, periods, data_only=False):
    import pandas as pd
    import torch
    from training.life_periods import load_life_period_config, life_period_membership
    pipeline = Pipeline(SimpleNamespace(config=config, device=None, stage="summarize", periods=periods,
                                        representations="multi_group", include_full=False))
    pipeline.require_source()
    source = pipeline.path(pipeline.config["source_data"])
    raw = pd.read_csv(source.parent / "nodes.csv", keep_default_na=False)
    memberships, _ = life_period_membership(raw, load_life_period_config(pipeline.path(pipeline.config["period_config"])))
    original_edges = pd.read_csv(source.parent / "edges.csv", keep_default_na=False)
    rows = []
    expected_groups = set(pipeline.multi["groups"])
    for context in pipeline.contexts:
        pipeline.validate_period(context)
        pipeline.validate_source(context)
        window = pipeline.source(context).parent
        actual = pd.read_csv(window / "nodes.csv", keep_default_na=False)
        expected = raw.loc[memberships[context]]
        require(actual.node_id.tolist() == expected.node_id.tolist(), f"Window membership differs: {context}")
        for column in ("birth_year", "death_year", "occupation_level1", "occupation_level2", "occupation_level3", "country"):
            require(actual[column].astype(str).tolist() == expected[column].astype(str).tolist(),
                    f"Raw attributes changed in {context}: {column}")
        # Independently verify the induced edge set, including the reverse edges.
        ids = set(actual.node_id)
        expected_edges = original_edges.loc[original_edges.source.isin(ids) & original_edges.target.isin(ids)]
        window_edges = pd.read_csv(window / "edges.csv", keep_default_na=False)
        columns = ["source", "relation", "target"]
        require(set(map(tuple, window_edges[columns].values)) == set(map(tuple, expected_edges[columns].values)),
                f"Induced edge set differs: {context}")
        if data_only:
            rows.append({"context": context, "nodes": len(actual), "original_directed_edges": len(window_edges), "status": "passed"})
            continue
        artifact, model, mask = pipeline.directories(context, "multi_group")
        checkpoint = torch.load(model / "best_model.pt", map_location="cpu", weights_only=False)
        settings = checkpoint["model_config"]
        require(settings["occupation_feature_levels"] == [1] and settings["auxiliary_features"] == [],
                f"Actual model input protocol differs: {context}")
        require(set(checkpoint["metadata"]["label_to_id"]) == set(pipeline.config["expected_l1_classes"]),
                f"Actual model classes differ: {context}")
        trained = read_json(model / "metrics.json")
        for key in ("accuracy", "macro_f1"):
            probability(trained["test"][key], f"{context} RGCN {key}")
        manifest = read_json(mask / "manifest.json")
        require(manifest["training_config"]["checkpoint_selection"] == "any-stage",
                f"GraphMask checkpoint policy differs: {context}")
        validation = read_json(mask / "validation.json")
        difference = validation["relative_macro_f1_difference"]
        require(math.isfinite(difference) and difference <= pipeline.config["graphmask_train"]["max_relative_macro_f1_diff"],
                f"Validation fidelity threshold failed: {context}")
        metrics = read_json(mask / "test_report/test_metrics.json")
        summary = read_json(artifact / "split_summary.json")
        require(metrics["split"] == "test" and metrics["roots"] == metrics["labeled_roots"] == summary["test_nodes"] > 0,
                f"Wrong report roots: {context}")
        for block in ("original", "masked"):
            for key in ("accuracy", "macro_f1"):
                probability(metrics[block][key], f"{context} {block} {key}")
        for key in ("prediction_agreement", "hard_retention_rate"):
            probability(metrics[key], f"{context} {key}")
        with (mask / "test_report/relations_base.csv").open() as handle:
            groups = [r for r in csv.DictReader(handle) if int(r["layer"]) == 0]
        require(set(r["relation"] for r in groups) <= expected_groups, f"Unknown group in {context}")
        require(sum(int(r["message_observations"]) for r in groups) > 0, f"No Layer 0 messages in {context}")
        require(abs(sum(float(r["retained_edge_share"]) for r in groups) - 1) < 1e-6,
                f"Layer 0 shares do not sum to 100%: {context}")
        rows.append({"context": context, "nodes": len(actual), "original_directed_edges": len(window_edges),
                     "rgcn_test": trained["test"], "graphmask_test": metrics,
                     "validation_relative_f1_difference": difference, "status": "passed"})
    if not data_only:
        require(pipeline.summarize() and not pipeline.failures, "Stage provenance or report completeness failed")
    report = {"status": "passed", "config": str(pipeline.config_path), "windows": rows,
              "checks": ["exact life membership", "raw dates and labels preserved", "induced edges",
                         "five L1 classes", "L1-only model without auxiliary features", "any-stage checkpoint policy",
                         "validation fidelity", "test roots and metrics", "Layer 0 message shares", "stage provenance"]}
    if data_only:
        report["checks"] = report["checks"][:4]
    write_json(pipeline.path(pipeline.config["graphmask_root"]) / ("pilot_data_validation.json" if data_only else "pilot_validation.json"), report)
    print(json.dumps({"status": "passed", "contexts": pipeline.contexts}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--periods", default="center_1900,center_1950,center_2000")
    parser.add_argument("--data-only", action="store_true", help="Check preparation without requiring trained models")
    args = parser.parse_args()
    try:
        validate(args.config, args.periods, args.data_only)
    except Exception as error:
        print(f"Pilot validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
