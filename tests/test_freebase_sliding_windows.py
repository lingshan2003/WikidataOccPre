"""Configuration, boundary membership and Pipeline compatibility for Freebase windows."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from Freebase.configure_sliding_windows import configurations
from Freebase.grouped_pipeline import Pipeline, parse_args
from training.life_periods import life_period_membership, load_life_period_config


ROOT = Path(__file__).resolve().parents[1]


class FreebaseSlidingWindowsTests(unittest.TestCase):
    def test_default_grid_has_101_inclusive_center_windows(self):
        generated = configurations()
        windows = next(value for path, value in generated.items() if "life_windows" in path.name)
        self.assertEqual(len(windows["periods"]), 101)
        self.assertEqual(windows["periods"][0], {
            "id": "center_1900", "label": "1900 (1880-1920 CE)", "start": 1880, "end": 1920,
        })
        self.assertEqual(windows["periods"][-1], {
            "id": "center_2000", "label": "2000 (1980-2020 CE)", "start": 1980, "end": 2020,
        })
        self.assertEqual([p["id"] for p in windows["periods"]], [f"center_{year}" for year in range(1900, 2001)])
        self.assertEqual(windows["membership_rule"], "life_interval_or_known_endpoint_in_period")
        self.assertIn("not relationship event dates", windows["description"])

    def test_life_interval_and_known_endpoint_boundaries_are_inclusive(self):
        generated = configurations(1900, 1902)
        windows = next(value for path, value in generated.items() if "life_windows" in path.name)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "windows.json"
            config_path.write_text(json.dumps(windows), encoding="utf-8")
            config = load_life_period_config(config_path)
            # Exact lower/upper endpoints, an interval spanning the whole
            # window despite an earlier birth, partial dates, missing dates,
            # and a reversed interval.
            nodes = pd.DataFrame({
                "birth_year": [1880, 1920, 1700, 1880, None, None, 1921],
                "death_year": [1880, 1920, 1990, None, 1920, None, 1900],
            })
            membership, _ = life_period_membership(nodes, config)
            self.assertEqual(membership["center_1900"].tolist(), [True, True, True, True, True, False, False])
            self.assertEqual(membership["center_1901"].tolist(), [False, True, True, False, True, False, False])
            self.assertEqual(membership["center_1902"].tolist(), [False, True, True, False, True, False, False])

    def test_pipeline_config_paths_are_isolated_and_compatible(self):
        generated = configurations()
        pipeline_config = next(value for path, value in generated.items() if "grouped_sliding" in path.name)
        self.assertEqual(pipeline_config["source_data"], "artifacts/freebase_provisional_l1_v2/graph_data.pt")
        self.assertEqual(pipeline_config["period_root"], "artifacts/freebase_life_windows_1900_2000_pm20_step1_v2")
        self.assertEqual(pipeline_config["relation_root"], "artifacts/freebase_grouped_sliding_1900_2000_pm20_step1_v2")
        self.assertEqual(pipeline_config["model_root"], "runs/freebase_grouped_sliding_1900_2000_pm20_step1_v2")
        self.assertEqual(pipeline_config["graphmask_root"], "runs_graphmask/freebase_grouped_sliding_1900_2000_pm20_step1_v2")
        self.assertEqual(pipeline_config["representations"], ["multi_group"])
        self.assertIsNone(pipeline_config["device"])
        self.assertEqual(pipeline_config["source_prepare"], json.loads(
            (ROOT / "config/freebase_grouped_rgcn_graphmask_20y_v2.json").read_text(encoding="utf-8")
        )["source_prepare"])

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.json"
            config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                pipeline = Pipeline(parse_args(["plan", "all", "--config", str(config_path), "--device", "cpu"]))
            self.assertEqual(len(pipeline.contexts), 101)
            self.assertEqual(pipeline.contexts[0], "center_1900")
            self.assertEqual(pipeline.contexts[-1], "center_2000")
            self.assertEqual(pipeline.representations, ["multi_group"])
            self.assertEqual(pipeline.config["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
