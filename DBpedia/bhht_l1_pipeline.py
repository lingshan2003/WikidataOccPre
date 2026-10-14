#!/usr/bin/env python3
"""Reuse the grouped runner with an isolated, provenance-checked DBpedia BHHT L1 source."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline as GroupedPipeline, MODEL_FILES, PROBE_FILES, REPORT_FILES
from DBpedia.grouped_pipeline import flags, complete

SOURCE_FILES = ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json", "relation_stats.csv")


class Pipeline(GroupedPipeline):
    env_prefix = "DBPEDIA_BHHT"

    def __init__(self, args):
        super().__init__(args)
        if not self.config["name"].startswith("dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_"):
            raise ValueError("BHHT runner requires its isolated DBpedia experiment config")
        source = self.path(self.config["source_data"]).parent
        outputs = [self.path(self.config[k]) for k in ("period_root", "relation_root", "model_root", "graphmask_root")]
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate([source, *outputs]) for b in [source, *outputs][i + 1:]):
            raise ValueError("Source, window, relation, model and probe roots must be separate")
        if self.config["source_prepare"]["target_level"] != 1:
            raise ValueError("This experiment predicts occupation L1")
        if self.config["train"]["occupation_feature_levels"] != "1" or self.config["train"]["auxiliary_features"] != "none":
            raise ValueError("Confirmed protocol requires L1-only inputs and no auxiliary features")
        if self.config["graphmask_train"]["checkpoint_selection"] != "any-stage":
            raise ValueError("Confirmed protocol uses any-stage checkpoint selection")

    def validate_dataset(self):
        manifest = json.loads(self.path(self.config["dataset_manifest"]).read_text())
        expected = set(self.config["expected_l1_classes"])
        if manifest.get("status") != "complete" or set(manifest.get("level1_node_counts", {})) != expected:
            raise ValueError("BHHT dataset manifest is incomplete or has the wrong L1 classes")
        if (manifest.get("graph_nodes"), manifest.get("original_relation_facts"), manifest.get("base_relation_types")) != (55690, 67090, 33):
            raise ValueError("BHHT dataset population differs from the reviewed first edition")
        if manifest.get("temporary_other_nodes") != 597 or manifest.get("l2_l3_policy") != "deferred_and_empty_in_dataset":
            raise ValueError("BHHT temporary-Other or fine-level policy differs")

    def validate_source(self, context):
        bundle = super().validate_source(context)
        if set(bundle["metadata"]["label_to_id"]) != set(self.config["expected_l1_classes"]):
            raise ValueError("Source graph must retain the five BHHT L1 classes")
        if context == "full" and int(bundle["data"].num_nodes) != 55690:
            raise ValueError("Source preparation lost nodes from the reviewed graph")
        return bundle

    def adapter_command(self):
        settings = dict(self.config["source_prepare"])
        settings["input"] = str(self.path(settings["input"]))
        return self.command("prepare", output_dir=self.path(self.config["source_data"]).parent, **settings)

    def source_inputs(self):
        return [self.path(self.config["source_prepare"]["input"]), self.path(self.config["dataset_manifest"])]

    def ensure_source(self):
        self.validate_dataset()
        self.run_step("prepare_source", self.adapter_command(), self.path(self.config["source_data"]).parent,
                      SOURCE_FILES, self.source_inputs(), self.config["source_prepare"])
        self.validate_source("full")

    def require_source(self):
        self.validate_dataset()
        self.require_completed("prepare_source", self.adapter_command(), self.path(self.config["source_data"]).parent,
                               SOURCE_FILES, self.source_inputs(), self.config["source_prepare"])

    def prepare(self):
        # The dispatcher initializes this once before workers can touch windows.
        self.require_source()
        super().prepare()

    def plan(self):
        if self.args.stage in ("all", "prepare"):
            import shlex
            print("[prepare source once] " + shlex.join(self.adapter_command()))
        super().plan()
