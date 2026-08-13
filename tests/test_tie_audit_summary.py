"""Regression tests for reuse of pre-taxonomy Level-1 baseline reports."""

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tie_audit_runs.py"
SPEC = importlib.util.spec_from_file_location("tie_audit_summary", SUMMARY_PATH)
tie_audit_summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tie_audit_summary)


class TieAuditSummaryTests(unittest.TestCase):
    def _config(self, model, seed):
        return {
            "model": model,
            "seed": int(seed),
            "occupation_feature_levels": "1,2,3",
            "auxiliary_features": "none",
            "feature_mode": "selected",
            "occupation_representation": "categorical",
            "num_neighbors": "15,10",
            "train_mode": "sampled",
            "eval_mode": "sampled",
            "loss": "cross_entropy",
            "train_root_sampling": "uniform",
            "early_stop_metric": "macro_f1",
            "hidden_dim": 128,
            "branch_dim": 64,
            "heads": 4,
            "rgcn_backend": "fast",
        }

    @staticmethod
    def _metrics(value):
        return {
            "accuracy": value,
            "macro_f1": value,
            "weighted_f1": value,
            "macro_precision": value,
            "macro_recall": value,
        }

    def _write_metrics(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_predictions(self, path, predictions):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["node_index", "node_id", "true_label", "prediction", "confidence"],
            )
            writer.writeheader()
            for node_index, true_label, prediction in predictions:
                writer.writerow({
                    "node_index": node_index,
                    "node_id": f"Q{node_index}",
                    "true_label": true_label,
                    "prediction": prediction,
                    "confidence": 0.9,
                })

    def test_legacy_metrics_are_reused_as_same_seed_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tie_audit"
            baseline_root = Path(directory) / "level1"
            for model in tie_audit_summary.MODELS:
                for seed in tie_audit_summary.SEEDS:
                    self._write_metrics(
                        baseline_root / f"{model}_baseline" / f"seed_{seed}" / "metrics.json",
                        {"run_config": self._config(model, seed), "test": self._metrics(0.70)},
                    )
                    for condition in tie_audit_summary.CONDITIONS:
                        self._write_metrics(
                            root / f"{model}__occupation_neighbours__{condition}" / f"seed_{seed}" / "metrics.json",
                            {
                                "run_config": self._config(model, seed),
                                "test": self._metrics(0.68),
                                "relation_perturbation": {
                                    "data_sha256": "same-data",
                                    "tie_taxonomy": {"sha256": "same-taxonomy"},
                                },
                            },
                        )
            baselines = tie_audit_summary.load_legacy_baselines(baseline_root)
            ablations = tie_audit_summary.load_ablation_records(root)
            per_seed, summaries = tie_audit_summary.paired_records(baselines, ablations)
            self.assertEqual(len(baselines), 6)
            self.assertEqual(len(ablations), 24)
            self.assertEqual(len(per_seed), 24)
            self.assertEqual(len(summaries), 8)
            self.assertAlmostEqual(summaries[0]["macro_f1_delta_mean"], -0.02)
            self.assertEqual(summaries[0]["comparison_provenance"], "same_seed_legacy_baseline_config")
            self.assertFalse(summaries[0]["bootstrap_available"])

    def test_saved_predictions_enable_paired_test_node_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tie_audit"
            baseline_root = Path(directory) / "level1"
            baseline_predictions = [(0, "A", "A"), (1, "B", "B"), (2, "A", "A"), (3, "B", "B")]
            ablation_predictions = [(0, "A", "A"), (1, "B", "A"), (2, "A", "A"), (3, "B", "B")]
            for model in tie_audit_summary.MODELS:
                for seed in tie_audit_summary.SEEDS:
                    baseline_dir = baseline_root / f"{model}_baseline" / f"seed_{seed}"
                    self._write_metrics(
                        baseline_dir / "metrics.json",
                        {"run_config": self._config(model, seed), "test": self._metrics(0.70)},
                    )
                    self._write_predictions(baseline_dir / "test_predictions.csv", baseline_predictions)
                    for condition in tie_audit_summary.CONDITIONS:
                        ablation_dir = root / f"{model}__occupation_neighbours__{condition}" / f"seed_{seed}"
                        self._write_metrics(
                            ablation_dir / "metrics.json",
                            {
                                "run_config": self._config(model, seed),
                                "test": self._metrics(0.68),
                                "relation_perturbation": {
                                    "data_sha256": "same-data",
                                    "tie_taxonomy": {"sha256": "same-taxonomy"},
                                },
                            },
                        )
                        self._write_predictions(ablation_dir / "test_predictions.csv", ablation_predictions)
            baselines = tie_audit_summary.load_legacy_baselines(baseline_root)
            ablations = tie_audit_summary.load_ablation_records(root)
            per_seed, summaries = tie_audit_summary.paired_records(
                baselines, ablations, bootstrap_draws=100, bootstrap_seed=7
            )
            self.assertTrue(all(row["bootstrap_available"] for row in per_seed))
            self.assertTrue(all(row["bootstrap_available"] for row in summaries))
            self.assertTrue(all(row["bootstrap_draws"] == 100 for row in summaries))
            self.assertTrue(all(
                row["comparison_provenance"] == "same_seed_legacy_baseline_predictions"
                for row in summaries
            ))


if __name__ == "__main__":
    unittest.main()
