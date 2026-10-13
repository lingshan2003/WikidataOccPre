"""Main-data protocol and isolation checks; CUDA jobs are not launched."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from Wikidata.grouped_pipeline import Pipeline
from training.tie_taxonomy import load_relation_taxonomy
from training.train import parse_args as train_args
from training.graphmask_train import parse_args as probe_args
from data.extended import make_numeric_features
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/wikidata_grouped_sliding_1900_2000_pm20_step1_alive2026_v1.json"


class WikidataSlidingProtocolTest(unittest.TestCase):
    def test_all_missing_training_death_features_are_finite_and_do_not_fit_test_dates(self):
        nodes = pd.DataFrame({"node_id": ["A", "B", "C"], "birth_year": [1930, 1940, 1950],
                              "death_year": [np.nan, np.nan, 2000]})
        original = nodes.copy(deep=True)
        values = make_numeric_features(nodes, train_nodes=["A", "B"])
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:, 1:3], 0)
        np.testing.assert_array_equal(values[:2, 4:6], 1)
        np.testing.assert_array_equal(values[2, 4:6], 0)
        pd.testing.assert_frame_equal(nodes, original)

    def pipeline(self):
        return Pipeline(SimpleNamespace(config=str(CONFIG), device="cpu", stage="all", periods=None,
                        representations=None, include_full=False))

    def test_namespace_prevents_old_dbpedia_environment_from_redirecting_main_data(self):
        with mock.patch.dict(os.environ, {"DBPEDIA_GROUP_SOURCE_DATA": "/wrong/data.pt",
                             "DBPEDIA_GROUP_PERIODS": "bad", "DBPEDIA_GROUP_INCLUDE_FULL": "1",
                             "WIKIDATA_GROUP_TRAIN_BATCH_SIZE": "256"}, clear=True):
            p = self.pipeline()
        self.assertEqual(len(p.contexts), 101)
        self.assertNotIn("full", p.contexts)
        self.assertEqual(p.config["source_data"], "artifacts/wikidata_level1_sliding_source_v1/graph_data.pt")
        self.assertEqual(p.config["train"]["batch_size"], 256)

    def test_explicit_taxonomy_preserves_all_main_experiment_groups(self):
        frozen = json.loads((ROOT / "config/wikidata_tie_taxonomy_acquired_subgroups_explicit_v1.json").read_text())
        members = sorted(r for rs in frozen["groups"].values() for r in rs)
        self.assertEqual(len(members), 40)
        vocabulary = {r: i for i, r in enumerate([name for r in members for name in (r, r + "__rev")])}
        original = load_relation_taxonomy(ROOT / "config/tie_taxonomy_acquired_subgroups_v1.json", vocabulary)
        self.assertEqual({k: set(v) for k, v in original.groups.items()}, {k: set(v) for k, v in frozen["groups"].items()})

    def test_commands_preserve_l1_target_hierarchical_features_and_any_stage_selection(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            p = self.pipeline()
        commands = p.commands("center_1950", "multi_group")
        with mock.patch.object(sys, "argv", ["train", *commands[1][3:]]):
            train = train_args()
        with mock.patch.object(sys, "argv", ["probe", *commands[2][3:]]):
            probe = probe_args()
        self.assertEqual(train.model, "rgcn")
        self.assertEqual(train.occupation_feature_levels, "1,2,3")
        self.assertEqual(train.auxiliary_features, "none")
        self.assertEqual(probe.checkpoint_selection, "any-stage")
        self.assertEqual(p.config["source_prepare"]["target_level"], 1)
        self.assertEqual(p.period_config["birth_only_alive_assumption"], {"born_after": 1920, "alive_through": 2026})
        self.assertEqual(p.periods["center_1950"]["start"], 1930)
        self.assertEqual(p.periods["center_1950"]["end"], 1970)
        self.assertEqual(p.config["num_bases"]["multi_group"], 14)

    def test_plan_lists_source_before_window_training_and_writes_no_results(self):
        result = subprocess.run([sys.executable, str(ROOT / "Wikidata/sliding_multi_gpu.py"),
                                 "plan", "all", "--gpus", "2,3,4", "--periods", "center_1900"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(result.stdout.index("[prepare source once]"), result.stdout.index("[train]"))
        self.assertIn("Wikidata", str(ROOT / "Wikidata/sliding_multi_gpu.py"))

    def test_cpu_plan_does_not_require_a_gpu_pool(self):
        result = subprocess.run([sys.executable, str(ROOT / "Wikidata/sliding_multi_gpu.py"),
                                 "plan", "all", "--device", "cpu", "--cpu-workers", "2", "--periods", "center_1900"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("['cpu_0', 'cpu_1']", result.stdout)
        self.assertIn("--device cpu", result.stdout)


if __name__ == "__main__":
    unittest.main()
