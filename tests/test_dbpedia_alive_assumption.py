"""Birth-only lifespan assumptions must remain explicit and separate from facts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from DBpedia.audit_sliding_window_artifacts import interval_mask
from data.birth_cohort_artifacts import build_period_induced_artifact
from training.life_periods import load_life_period_config, life_period_membership
from tests import test_birth_cohort_induced_artifacts as induced_tests

ROOT = Path(__file__).resolve().parents[1]
NEW_WINDOWS = ROOT / "config/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2.json"
OLD_WINDOWS = ROOT / "config/dbpedia_life_windows_1900_2000_pm20_step1_v1.json"


class BirthOnlyAliveAssumptionTests(unittest.TestCase):
    def test_strict_1920_boundary_observed_deaths_and_raw_dates_preserved(self):
        nodes = pd.DataFrame({"birth_year": [1919, 1920, 1921, 1921, 1981, 2026, 2027, None, 1950, None],
                              "death_year": [None, None, None, 1940, None, None, None, 1990, 1949, None]})
        original = nodes.copy(deep=True)
        config = load_life_period_config(NEW_WINDOWS)
        memberships, audit = life_period_membership(nodes, config)
        self.assertEqual(memberships["center_1950"].tolist(), [False, False, True, True, False, False, False, False, False, False])
        self.assertEqual(memberships["center_2000"].tolist(), [False, False, True, False, True, False, False, True, False, False])
        self.assertTrue(nodes.equals(original))
        self.assertEqual(audit.life_period_death_year_imputed.tolist(), [False, False, True, False, True, True, True, False, False, False])
        self.assertTrue(pd.isna(audit.death_year_numeric.iloc[2]))
        self.assertEqual(audit.life_period_effective_death_year.iloc[2], 2026)
        self.assertEqual(audit.life_period_effective_death_year.iloc[3], 1940)
        self.assertEqual(audit.life_interval_status.iloc[6], "death_before_birth")
        birth = pd.to_numeric(nodes.birth_year).to_numpy(float)
        death = pd.to_numeric(nodes.death_year).to_numpy(float)
        for period in config.periods:
            np.testing.assert_array_equal(memberships[period.identifier], interval_mask(birth, death, period.start, period.end,
                                          config.birth_only_alive_assumption.manifest()))
        old_memberships, old_audit = life_period_membership(nodes, load_life_period_config(OLD_WINDOWS))
        self.assertFalse(old_memberships["center_2000"][2])
        self.assertNotIn("life_period_death_year_imputed", old_audit)

    def test_inferred_life_stops_at_2026_inclusively_and_config_rejects_typos(self):
        payload = json.loads(NEW_WINDOWS.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.json"
            nodes = pd.DataFrame({"birth_year": [1921, 1920, 2026, 2027], "death_year": [None] * 4})
            for year, expected in ((2026, [True, False, True, False]), (2027, [False] * 4)):
                payload["periods"] = [{"id": "point", "label": "point", "start": year, "end": year}]
                path.write_text(json.dumps(payload))
                memberships, _ = life_period_membership(nodes, load_life_period_config(path))
                self.assertEqual(memberships["point"].tolist(), expected)
            for invalid in ({"born_after": 1920}, {"born_after": True, "alive_through": 2026},
                            {"born_after": 1920, "alive_through": 1900},
                            {"born_after": 1920, "alive_through": 2026, "extra": 0}):
                payload["birth_only_alive_assumption"] = invalid
                path.write_text(json.dumps(payload))
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "birth_only_alive_assumption"):
                    load_life_period_config(path)

    def test_induced_artifact_retains_assumed_alive_node_without_faking_death_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = induced_tests.PeriodInducedArtifactTests()._write_source_artifact(root)
            nodes = pd.read_csv(source.parent / "nodes.csv")
            nodes["birth_year"] = [1921, 1970, 1970, 1925, 1970, 1970]
            nodes["death_year"] = [None, 2020, 2020, None, 2020, 2020]
            nodes.to_csv(source.parent / "nodes.csv", index=False)
            config = load_life_period_config(NEW_WINDOWS)
            report = build_period_induced_artifact(source, root / "windows", config, "center_2000")
            artifact = root / "windows/center_2000"
            saved = pd.read_csv(artifact / "nodes.csv")
            bundle = torch.load(artifact / "graph_data.pt", weights_only=False)
            self.assertEqual(saved.source_node_index.tolist(), list(range(6)))
            self.assertTrue(saved.death_year.iloc[[0, 3]].isna().all())
            self.assertEqual(saved.life_period_effective_death_year.iloc[[0, 3]].tolist(), [2026, 2026])
            self.assertEqual(report["period_induced_artifact"]["life_date_imputation"]["selected_nodes_with_imputed_death"], 2)
            self.assertEqual(bundle["data"].edge_index.size(1), 10)
            # Temporal feature layout is birth/death/age z-scores followed by
            # their missing indicators; inferred dates must remain missing.
            self.assertTrue(torch.all(bundle["data"].temporal[[0, 3], 4] == 1))
            self.assertTrue(torch.all(bundle["data"].temporal[[0, 3], 5] == 1))
            with self.assertRaisesRegex(ValueError, "incompatible source/period/split provenance"):
                build_period_induced_artifact(source, root / "windows", load_life_period_config(OLD_WINDOWS), "center_2000")


if __name__ == "__main__":
    unittest.main()
