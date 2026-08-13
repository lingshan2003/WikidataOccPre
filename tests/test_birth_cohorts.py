"""Tests for editable birth-cohort configurations and artifact alignment."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.birth_cohorts import (
    MISSING_COHORT_ID,
    assign_birth_cohorts,
    load_artifact_birth_cohorts,
    load_birth_cohort_config,
)


class BirthCohortTests(unittest.TestCase):
    def _config(self, bins):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cohorts.json"
        path.write_text(json.dumps({
            "name": "test cohorts",
            "version": 1,
            "time_field": "birth_year",
            "missing_policy": "exclude_from_stratified_results",
            "bins": bins,
        }), encoding="utf-8")
        return path

    def test_exhaustive_cohorts_assign_dates_and_keep_missing_separate(self):
        path = self._config([
            {"id": "early", "label": "Early", "end": 1899},
            {"id": "late", "label": "Late", "start": 1900},
        ])
        config = load_birth_cohort_config(path)
        assigned = assign_birth_cohorts([1200, 1899, 1900, None, "unknown"], config)
        self.assertEqual(assigned["birth_cohort"].tolist(), [
            "early", "early", "late", MISSING_COHORT_ID, MISSING_COHORT_ID,
        ])
        self.assertEqual(assigned["included_in_cohort_hypothesis"].tolist(), [True, True, True, False, False])
        self.assertEqual(config.cohort("late").label, "Late")
        self.assertEqual(config.manifest()["sha256"], config.sha256)

    def test_gap_or_non_exhaustive_bins_are_rejected(self):
        path = self._config([
            {"id": "early", "label": "Early", "end": 1898},
            {"id": "late", "label": "Late", "start": 1900},
        ])
        with self.assertRaisesRegex(ValueError, "gap-free"):
            load_birth_cohort_config(path)

        path = self._config([
            {"id": "only", "label": "Only", "start": 1800},
        ])
        with self.assertRaisesRegex(ValueError, "first birth cohort"):
            load_birth_cohort_config(path)

    def test_artifact_nodes_are_attached_in_graph_index_order(self):
        config_path = self._config([
            {"id": "early", "label": "Early", "end": 1899},
            {"id": "late", "label": "Late", "start": 1900},
        ])
        config = load_birth_cohort_config(config_path)
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "graph_data.pt").touch()
            pd.DataFrame({"node_id": ["Q2", "Q1", "Q3"], "birth_year": [1900, 1800, None]}).to_csv(
                artifact_dir / "nodes.csv", index=False
            )
            result = load_artifact_birth_cohorts(artifact_dir / "graph_data.pt", config, expected_nodes=3)
            self.assertEqual(result["node_index"].tolist(), [0, 1, 2])
            self.assertEqual(result["birth_cohort"].tolist(), ["late", "early", MISSING_COHORT_ID])
            with self.assertRaisesRegex(ValueError, "differs from graph node count"):
                load_artifact_birth_cohorts(artifact_dir / "graph_data.pt", config, expected_nodes=4)


if __name__ == "__main__":
    unittest.main()
