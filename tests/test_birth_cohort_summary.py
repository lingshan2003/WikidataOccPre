"""Regression tests for same-seed birth-cohort prediction summaries."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.birth_cohorts import load_artifact_birth_cohorts, load_birth_cohort_config


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tie_audit_birth_cohorts.py"
SPEC = importlib.util.spec_from_file_location("birth_cohort_summary", SUMMARY_PATH)
birth_cohort_summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(birth_cohort_summary)


class BirthCohortSummaryTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _write_predictions(path, predictions):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["node_index", "true_label", "prediction"])
            writer.writeheader()
            writer.writerows(predictions)

    def test_global_ablation_predictions_are_paired_within_each_birth_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            artifact = base / "artifact" / "graph_data.pt"
            artifact.parent.mkdir()
            artifact.touch()
            pd.DataFrame({
                "node_id": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                "birth_year": [1800, 1801, 1900, 1901, None],
            }).to_csv(artifact.parent / "nodes.csv", index=False)
            config_path = base / "cohorts.json"
            self._write_json(config_path, {
                "name": "test cohorts",
                "version": 1,
                "time_field": "birth_year",
                "missing_policy": "exclude_from_stratified_results",
                "bins": [
                    {"id": "early", "label": "Early", "end": 1899},
                    {"id": "late", "label": "Late", "start": 1900},
                ],
            })
            baseline_root, audit_root = base / "baselines", base / "audit"
            full = [
                {"node_index": 0, "true_label": "A", "prediction": "A"},
                {"node_index": 1, "true_label": "B", "prediction": "B"},
                {"node_index": 2, "true_label": "A", "prediction": "A"},
                {"node_index": 3, "true_label": "B", "prediction": "B"},
                {"node_index": 4, "true_label": "A", "prediction": "A"},
            ]
            altered = {
                "without_inherited": ["B", "B", "A", "B", "A"],
                "random_matched_inherited": ["A", "B", "A", "B", "A"],
                "without_acquired": ["A", "B", "B", "B", "A"],
                "random_matched_acquired": ["A", "B", "A", "B", "A"],
            }
            provenance = {
                "data_sha256": "data-hash",
                "tie_taxonomy": {"sha256": "taxonomy-hash"},
            }
            for model in birth_cohort_summary.MODELS:
                for seed in birth_cohort_summary.SEEDS:
                    baseline_dir = baseline_root / f"{model}_baseline" / f"seed_{seed}"
                    self._write_predictions(baseline_dir / "test_predictions.csv", full)
                    self._write_json(baseline_dir / "metrics.json", {})
                    for condition in birth_cohort_summary.CONDITIONS:
                        run_dir = audit_root / f"{model}__occupation_neighbours__{condition}" / f"seed_{seed}"
                        rows = [dict(row, prediction=prediction) for row, prediction in zip(full, altered[condition])]
                        self._write_predictions(run_dir / "test_predictions.csv", rows)
                        self._write_json(run_dir / "metrics.json", {"relation_perturbation": provenance})

            config = load_birth_cohort_config(config_path)
            nodes = load_artifact_birth_cohorts(artifact, config)
            records = birth_cohort_summary.per_cohort_records(
                audit_root, baseline_root, nodes, config.manifest()
            )
            deltas, specificity = birth_cohort_summary.paired_records(records)
            summaries = birth_cohort_summary.summarise(
                specificity,
                ("model", "tie_group", "birth_cohort"),
                ("relationship_specific_macro_f1_loss",),
            )
            early_inherited = next(
                row for row in summaries
                if row["model"] == "rgcn" and row["tie_group"] == "inherited" and row["birth_cohort"] == "early"
            )
            late_acquired = next(
                row for row in summaries
                if row["model"] == "rgat" and row["tie_group"] == "acquired" and row["birth_cohort"] == "late"
            )
            self.assertEqual(len(records), 90)
            self.assertEqual(len(deltas), 72)
            self.assertEqual(len(specificity), 36)
            self.assertGreater(early_inherited["relationship_specific_macro_f1_loss_mean"], 0.0)
            self.assertGreater(late_acquired["relationship_specific_macro_f1_loss_mean"], 0.0)
            self.assertFalse(next(row for row in specificity if row["birth_cohort"] == "missing_birth_year")[
                "included_in_cohort_hypothesis"
            ])

            targeted_root = base / "targeted"
            for model in birth_cohort_summary.MODELS:
                for seed in birth_cohort_summary.SEEDS:
                    for cohort_id in ("early", "late"):
                        targeted_provenance = {
                            **provenance,
                            "edge_cohort": {
                                "selected_cohort_id": cohort_id,
                                "edge_scope": "incident_to_selected_cohort",
                            },
                        }
                        for condition in birth_cohort_summary.CONDITIONS:
                            run_dir = targeted_root / f"{model}__cohort_{cohort_id}__{condition}" / f"seed_{seed}"
                            rows = [dict(row, prediction=prediction) for row, prediction in zip(full, altered[condition])]
                            self._write_predictions(run_dir / "test_predictions.csv", rows)
                            self._write_json(run_dir / "metrics.json", {"relation_perturbation": targeted_provenance})
            targeted_records = birth_cohort_summary.targeted_cohort_records(
                targeted_root, baseline_root, nodes, config.manifest()
            )
            targeted_deltas, targeted_specificity = birth_cohort_summary.paired_records(targeted_records)
            self.assertEqual(len(targeted_records), 60)
            self.assertEqual(len(targeted_deltas), 48)
            self.assertEqual(len(targeted_specificity), 24)


if __name__ == "__main__":
    unittest.main()
