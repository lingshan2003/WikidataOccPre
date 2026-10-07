"""Contracts for overlapping life-period membership."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.life_periods import life_period_membership, load_life_period_config


class LifePeriodTests(unittest.TestCase):
    def test_freebase_finite_windows_and_exact_calendar_boundaries(self):
        path = Path(__file__).resolve().parents[1] / "config/freebase_life_periods_20y_v1.json"
        config = load_life_period_config(path)
        years = [1499, 1500, 1900, 1901, 1920, 1921, 2020, 2021]
        nodes = pd.DataFrame({"birth_year": years, "death_year": [None] * len(years)})
        memberships, audit = life_period_membership(nodes, config)
        selected = [[p for p, mask in memberships.items() if mask[i]] for i in range(len(years))]
        self.assertEqual(selected, [["before_1500"], ["1500_1900"], ["1500_1900"],
                                    ["1901_1920"], ["1901_1920"], ["1921_1940"], ["2001_2020"], []])
        self.assertEqual(audit["life_period_membership_count"].tolist(), [1] * 7 + [0])
        payload = json.loads(path.read_text())
        payload.pop("allow_finite_last_period")
        with tempfile.TemporaryDirectory() as directory:
            finite = Path(directory) / "finite.json"
            finite.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "final end"):
                load_life_period_config(finite)

    def test_life_intervals_can_overlap_periods_and_single_known_endpoints_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "life_periods.json"
            path.write_text(json.dumps({
                "name": "test life periods with known endpoints",
                "version": 2,
                "membership_rule": "life_interval_or_known_endpoint_in_period",
                "birth_field": "birth_year",
                "death_field": "death_year",
                "missing_date_policy": "exclude_if_both_dates_missing",
                "partial_date_policy": "include_known_endpoint_periods",
                "invalid_interval_policy": "exclude_if_death_before_birth",
                "periods": [
                    {"id": "early", "label": "Early", "end": 500},
                    {"id": "late", "label": "Late", "start": 501},
                ],
            }), encoding="utf-8")
            config = load_life_period_config(path)
            nodes = pd.DataFrame({
                "birth_year": [100, 500, 501, 100, None, 700],
                "death_year": [600, 500, 501, None, 800, 600],
            })
            memberships, audit = life_period_membership(nodes, config)
            self.assertEqual(memberships["early"].tolist(), [True, True, False, True, False, False])
            self.assertEqual(memberships["late"].tolist(), [True, False, True, False, True, False])
            self.assertEqual(audit["life_period_membership_count"].tolist(), [2, 1, 1, 1, 1, 0])
            self.assertEqual(audit["life_interval_status"].tolist(), [
                "valid_life_interval", "valid_life_interval", "valid_life_interval",
                "birth_endpoint_only", "death_endpoint_only", "death_before_birth",
            ])


if __name__ == "__main__":
    unittest.main()
