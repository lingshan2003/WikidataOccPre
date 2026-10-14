"""Check isolation and the confirmed DBpedia five-class protocol without CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from DBpedia.bhht_l1_pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1.json"


class BHHTSlidingProtocolTests(unittest.TestCase):
    def pipeline(self):
        return Pipeline(SimpleNamespace(config=str(CONFIG), device="cpu", stage="all", periods=None,
                                        representations=None, include_full=False))

    def test_stale_dbpedia_overrides_cannot_redirect_new_run(self):
        with mock.patch.dict(os.environ, {"DBPEDIA_GROUP_SOURCE_DATA": "/old/graph.pt",
                                         "DBPEDIA_GROUP_MODEL_ROOT": "/old/model"}, clear=True):
            p = self.pipeline()
        self.assertEqual(len(p.contexts), 101)
        self.assertEqual(p.config["train"]["auxiliary_features"], "none")
        self.assertIn("bhht_l1", p.config["source_data"])
        self.assertNotEqual(p.config["model_root"], "/old/model")
        self.assertEqual(len(p.multi["groups"]), 8)

    def test_actual_plan_uses_new_input_l1_only_and_any_stage_policy(self):
        result = subprocess.run([sys.executable, str(ROOT / "DBpedia/sliding_bhht_l1_multi_gpu.py"),
                                 "plan", "all", "--gpus", "1,2,4", "--periods", "center_1950"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bhht_l1_provisional_v1/DBpedia_R_R_extended.csv", result.stdout)
        self.assertIn("--occupation-feature-levels 1 --auxiliary-features none", result.stdout)
        self.assertIn("--checkpoint-selection any-stage", result.stdout)
        self.assertLess(result.stdout.index("[prepare source once]"), result.stdout.index("[train]"))

    def test_manifest_must_keep_provisional_policy_and_population(self):
        p = self.pipeline()
        valid = {"status": "complete", "graph_nodes": 55690, "original_relation_facts": 67090,
                 "base_relation_types": 33, "level1_node_counts": {x: 1 for x in p.config["expected_l1_classes"]},
                 "temporary_other_nodes": 597, "l2_l3_policy": "deferred_and_empty_in_dataset"}
        with mock.patch.object(Path, "read_text", return_value=json.dumps(valid)):
            p.validate_dataset()
        valid["temporary_other_nodes"] = 0
        with mock.patch.object(Path, "read_text", return_value=json.dumps(valid)):
            with self.assertRaisesRegex(ValueError, "temporary-Other"):
                p.validate_dataset()


if __name__ == "__main__":
    unittest.main()
