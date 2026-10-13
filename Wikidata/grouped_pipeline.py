#!/usr/bin/env python3
"""Reuse the grouped runner with an isolated, provenance-checked Wikidata source."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline as GroupedPipeline, MODEL_FILES, PROBE_FILES, REPORT_FILES
from DBpedia.grouped_pipeline import flags, complete

SOURCE_FILES = ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json", "relation_stats.csv")


class Pipeline(GroupedPipeline):
    env_prefix = "WIKIDATA"

    def __init__(self, args):
        super().__init__(args)
        if not self.config["name"].startswith("wikidata_"):
            raise ValueError("Wikidata runner requires a Wikidata experiment config")
        source = self.path(self.config["source_data"]).parent
        outputs = [self.path(self.config[k]) for k in ("period_root", "relation_root", "model_root", "graphmask_root")]
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate([source, *outputs]) for b in [source, *outputs][i + 1:]):
            raise ValueError("Source, window, relation, model and probe roots must be separate")
        if self.config["source_prepare"]["target_level"] != 1:
            raise ValueError("This experiment predicts occupation L1")

    def adapter_command(self):
        settings = dict(self.config["source_prepare"])
        settings["input"] = str(self.path(settings["input"]))
        return self.command("prepare", output_dir=self.path(self.config["source_data"]).parent, **settings)

    def source_inputs(self):
        return [self.path(self.config["source_prepare"]["input"])]

    def ensure_source(self):
        self.run_step("prepare_source", self.adapter_command(), self.path(self.config["source_data"]).parent,
                      SOURCE_FILES, self.source_inputs(), self.config["source_prepare"])
        self.validate_source("full")

    def require_source(self):
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
