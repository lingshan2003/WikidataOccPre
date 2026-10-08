"""Calendar validation, membership boundaries and induced graph integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from DBpedia.configure_sliding_windows import configurations
from DBpedia.grouped_pipeline import Pipeline
from types import SimpleNamespace
from data.birth_cohort_artifacts import prepare_period_induced_artifacts
from training.life_periods import load_life_period_config, life_period_membership
from tests import test_birth_cohort_induced_artifacts as induced_tests


ROOT = Path(__file__).resolve().parents[1]


class SlidingWindowsTests(unittest.TestCase):
    def test_annual_boundaries_and_partial_dates_in_every_containing_window(self):
        payload = next(iter(configurations(1900, 1902).values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.json"
            path.write_text(json.dumps(payload))
            config = load_life_period_config(path)
            # Ends at the first lower bound; begins at its upper bound;
            # complete lifespan across all; endpoint-only; missing; invalid.
            nodes = pd.DataFrame({"birth_year": [1800, 1920, 1800, 1880, 1921, None, None, 1920],
                                  "death_year": [1880, 1930, 1930, None, None, 1920, None, 1900]})
            memberships, audit = life_period_membership(nodes, config)
            self.assertEqual(memberships["center_1900"].tolist(), [True, True, True, True, False, True, False, False])
            self.assertEqual(memberships["center_1901"].tolist(), [False, True, True, False, True, True, False, False])
            self.assertEqual(memberships["center_1902"].tolist(), [False, True, True, False, True, True, False, False])
            self.assertEqual(audit.life_period_membership_count.tolist(), [1, 3, 3, 1, 2, 3, 0, 0])
            self.assertEqual(config.manifest()["calendar_layout"], "sliding_windows")
            # Overlap cannot silently weaken the old partition contract.
            payload.pop("calendar_layout")
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "first start"):
                load_life_period_config(path)

    def test_reject_irregular_unequal_open_or_reversed_windows(self):
        base = next(iter(configurations(1900, 1902).values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.json"
            for mutation in ("width", "stride", "open", "order", "layout"):
                payload = json.loads(json.dumps(base))
                periods = payload["periods"]
                if mutation == "width":
                    periods[1]["end"] += 1
                elif mutation == "stride":
                    periods[2]["start"] += 1
                    periods[2]["end"] += 1
                elif mutation == "open":
                    periods[0].pop("start")
                elif mutation == "order":
                    periods.reverse()
                else:
                    payload["calendar_layout"] = "unknown"
                path.write_text(json.dumps(payload))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_life_period_config(path)

    def test_induced_windows_keep_only_both_endpoints_and_manifest_reuses(self):
        fixture = induced_tests.PeriodInducedArtifactTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture._write_source_artifact(root)
            payload = next(iter(configurations(300, 500, half_window=200, step=200).values()))
            path = root / "windows.json"
            path.write_text(json.dumps(payload))
            reports = prepare_period_induced_artifacts(source, root / "windows", path)
            self.assertEqual([(r["nodes"], r["directed_edges"]) for r in reports], [(4, 4), (6, 10)])
            for report in reports:
                self.assertEqual(report["life_period_config"]["calendar_layout"], "sliding_windows")
                edges = pd.read_csv(Path(report["artifact_dir"]) / "edges.csv")
                nodes = pd.read_csv(Path(report["artifact_dir"]) / "nodes.csv")
                self.assertTrue(set(edges.source) | set(edges.target) <= set(nodes.node_id))
            reused = prepare_period_induced_artifacts(source, root / "windows", path)
            self.assertTrue(all(r["reused_existing"] for r in reused))

    def test_default_plan_is_101_independent_multi_group_jobs_in_new_roots(self):
        config = ROOT / "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json"
        with mock.patch.dict(os.environ, {"DBPEDIA_PYTHON_BIN": sys.executable}, clear=True):
            plan = subprocess.run(["bash", "DBpedia/run_sliding_rgcn_graphmask.sh", "plan", "all"],
                                  cwd=ROOT, text=True, capture_output=True, check=True).stdout
        self.assertIn("101 contexts x 1 representations = 101", plan)
        self.assertEqual(plan.count("[train]"), 101)
        self.assertEqual(plan.count("[probe]"), 101)
        self.assertNotIn("/binary/", plan)
        self.assertNotIn("/dbpedia_grouped_20y_v1/", plan)
        self.assertEqual(plan.count("--checkpoint-selection any-stage"), 101)
        self.assertNotIn("--checkpoint-selection all-layers-enabled", plan)
        generated = list(configurations(1900, 2022).values())[1]
        self.assertEqual(generated["graphmask_train"]["checkpoint_selection"], "any-stage")
        self.assertEqual(len(load_life_period_config(ROOT / json.loads(config.read_text())["period_config"]).periods), 101)

    def test_pilot_pins_four_windows_and_reuses_formal_training_commands(self):
        pilot_config = "config/dbpedia_grouped_sliding_pilot_4years_v1.json"
        full_config = "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json"
        args = dict(device=None, periods=None, representations=None, include_full=False, stage="all")
        with mock.patch.dict(os.environ, {}, clear=True):
            pilot = Pipeline(SimpleNamespace(config=pilot_config, **args))
            full = Pipeline(SimpleNamespace(config=full_config, **args))
        self.assertEqual(pilot.contexts, ["center_1900", "center_1901", "center_1950", "center_2000"])
        for context in pilot.contexts:
            self.assertEqual(pilot.commands(context, "multi_group"), full.commands(context, "multi_group"))
        environment = {"DBPEDIA_PYTHON_BIN": sys.executable, "DBPEDIA_GROUP_INCLUDE_FULL": "1",
                       "DBPEDIA_GROUP_PERIODS": "all", "DBPEDIA_GROUP_REPRESENTATIONS": "binary"}
        with mock.patch.dict(os.environ, environment, clear=True):
            result = subprocess.run(["bash", "DBpedia/run_sliding_pilot.sh", "plan", "--gpus", "4,5"],
                                    cwd=ROOT, text=True, capture_output=True, check=True)
            rejected = subprocess.run(["bash", "DBpedia/run_sliding_pilot.sh", "plan", "--gpus", "4,5", "--periods", "all"],
                                      cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.stdout.count("[train]"), 4)
        self.assertNotIn("[train] full/", result.stdout)
        self.assertNotIn("/binary/", result.stdout)
        self.assertEqual(rejected.returncode, 2)

    def test_pilot_audit_failure_stops_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "python-stub"
            calls = root / "calls.jsonl"
            stub.write_text(f"#!{sys.executable}\n" +
                "import json, os, sys\n" +
                "with open(os.environ['PILOT_TEST_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n" +
                "sys.exit(int(os.environ['PILOT_TEST_AUDIT_CODE']) if 'audit_sliding_window_artifacts.py' in sys.argv[1] else 0)\n")
            stub.chmod(0o755)
            for audit_code in ("1", "0"):
                calls.write_text("")
                environment = {"DBPEDIA_PYTHON_BIN": str(stub), "PILOT_TEST_CALLS": str(calls), "PILOT_TEST_AUDIT_CODE": audit_code}
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = subprocess.run(["bash", "DBpedia/run_sliding_pilot.sh", "run", "--gpus", "4,5"],
                                            cwd=ROOT, text=True, capture_output=True)
                recorded = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(recorded[0][1:3], ["run", "prepare"])
                self.assertEqual(recorded[1][1:3], ["run", "collapse"])
                self.assertIn("audit_sliding_window_artifacts.py", recorded[2][0])
                self.assertEqual(result.returncode, int(audit_code))
                self.assertEqual(len(recorded), 3 if audit_code == "1" else 4)
                if audit_code == "0":
                    self.assertEqual(recorded[3][1:3], ["run", "all"])


if __name__ == "__main__":
    unittest.main()
