"""Synthetic contract test for acquired-subgroup audit aggregation."""

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_acquired_subgroup_runs.py"
SPEC = importlib.util.spec_from_file_location("acquired_subgroup_summary", SCRIPT_PATH)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class AcquiredSubgroupSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "runs"
        self.baselines = Path(self.directory.name) / "baselines"
        self.taxonomy = Path(self.directory.name) / "taxonomy.json"
        self.taxonomy.write_text(json.dumps({"name": "test", "version": 1, "groups": {"education": ["student_of"]}}))
        self.taxonomy_sha256 = hashlib.sha256(self.taxonomy.read_bytes()).hexdigest()

    @staticmethod
    def run_config(seed):
        return {
            "model": "rgcn", "seed": seed, "occupation_feature_levels": "1,2,3",
            "auxiliary_features": "none", "feature_mode": "selected", "occupation_representation": "categorical",
            "num_neighbors": "15,10", "train_mode": "sampled", "eval_mode": "sampled",
            "loss": "cross_entropy", "train_root_sampling": "uniform", "early_stop_metric": "macro_f1",
            "hidden_dim": 128, "branch_dim": 64, "heads": 4, "rgcn_backend": "fast",
        }

    def write_metrics(self, path, seed, macro_f1, perturbation):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_config": self.run_config(seed),
            "test": {
                "accuracy": macro_f1, "macro_f1": macro_f1, "weighted_f1": macro_f1,
                "macro_precision": macro_f1, "macro_recall": macro_f1,
            },
            "relation_perturbation": perturbation,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_full_scope_aggregates_direct_random_and_specificity_comparisons(self):
        taxonomy_manifest = {
            "sha256": self.taxonomy_sha256,
            "groups": {"education": ["student_of"]},
        }
        for seed in (42, 43, 44):
            self.write_metrics(
                self.baselines / "rgcn_baseline" / f"seed_{seed}" / "metrics.json", seed, 0.70,
                {"data_sha256": "artifact", "edge_count_before": 10, "edge_count_after_random_drop": 10},
            )
            self.write_metrics(
                self.root / "rgcn__occupation_neighbours__without_education" / f"seed_{seed}" / "metrics.json",
                seed, 0.50,
                {
                    "data_sha256": "artifact", "relation_taxonomy": taxonomy_manifest,
                    "dropped_relation_taxonomy_groups": ["education"],
                    "random_drop_matched_relation_taxonomy_groups": [],
                    "edge_count_before": 10, "edge_count_after_random_drop": 8,
                },
            )
            self.write_metrics(
                self.root / "rgcn__occupation_neighbours__random_matched_education" / f"seed_{seed}" / "metrics.json",
                seed, 0.65,
                {
                    "data_sha256": "artifact", "relation_taxonomy": taxonomy_manifest,
                    "dropped_relation_taxonomy_groups": [],
                    "random_drop_matched_relation_taxonomy_groups": ["education"],
                    "random_control_unit": "original_edge_instance_plus_generated_reverse",
                    "random_edge_instance_pairs": 1,
                    "edge_count_before": 10, "edge_count_after_random_drop": 8,
                },
            )
        argv = [
            str(SCRIPT_PATH), "--scope", "full", "--root", str(self.root), "--baseline-root", str(self.baselines),
            "--relation-taxonomy", str(self.taxonomy), "--groups", "education", "--bootstrap-draws", "0",
        ]
        with patch.object(sys, "argv", argv):
            summary.main()
        output = self.root / "acquired_subgroup_summary" / "acquired_subgroup_paired_summary.csv"
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        specific = next(row for row in rows if row["comparison"] == "random_minus_direct")
        self.assertEqual(specific["seeds_completed"], "3")
        self.assertAlmostEqual(float(specific["macro_f1_delta_mean"]), 0.15)
        self.assertEqual(specific["directed_edges_removed_mean"], "2.0")


if __name__ == "__main__":
    unittest.main()
