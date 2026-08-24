"""Synthetic checks for relational privilege amplification metrics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from models import build_model
from models.features import build_feature_specs

from training.inequality_audit import (
    classification_metrics,
    distribution_metrics,
    js_divergence,
)
from training.inequality_audit import main as audit_main
from training.inequality_predict import main as predict_main


class InequalityAuditMetricTests(unittest.TestCase):
    def test_js_divergence_is_symmetric_and_zero_for_equal_distributions(self):
        first = np.array([0.8, 0.2])
        second = np.array([0.3, 0.7])
        self.assertAlmostEqual(js_divergence(first, first), 0.0)
        self.assertAlmostEqual(js_divergence(first, second), js_divergence(second, first))
        self.assertGreater(js_divergence(first, second), 0.0)

    def test_distribution_amplification_compares_predictions_with_true_base_gap(self):
        labels = np.array([0, 1, 0, 1])
        access = np.array(["yes", "yes", "no", "no"])
        calibrated = np.array([
            [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]
        ])
        amplified = np.array([
            [0.99, 0.01], [0.90, 0.10], [0.10, 0.90], [0.01, 0.99]
        ])
        base = distribution_metrics(labels, calibrated, access, 2)
        biased = distribution_metrics(labels, amplified, access, 2)
        self.assertAlmostEqual(base["true_group_js"], 0.0)
        self.assertAlmostEqual(base["soft_distribution_amplification"], 0.0)
        self.assertGreater(biased["soft_distribution_amplification"], 0.0)

    def test_group_metrics_keep_the_global_class_denominator(self):
        labels = np.array([0, 0])
        probabilities = np.array([[0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
        result = classification_metrics(labels, probabilities, num_classes=2, ece_bins=5)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertGreaterEqual(result["ece"], 0.0)


class InequalityAuditWorkflowTests(unittest.TestCase):
    def test_probability_exports_feed_the_paired_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "graph_data.pt"
            checkpoint_path = root / "best_model.pt"
            taxonomy_path = root / "taxonomy.json"
            predictions_root = root / "predictions"
            summary_root = root / "summary"
            graph = Data(
                edge_index=torch.tensor([[0, 4, 1, 5], [4, 0, 5, 1]]),
                edge_type=torch.tensor([0, 1, 2, 3]),
                y=torch.tensor([0, 1, 0, 1, 0, 1]),
                num_nodes=6,
            )
            graph.train_mask = torch.tensor([True, True, True, False, False, False])
            graph.val_mask = torch.tensor([False, False, False, True, False, False])
            graph.test_mask = torch.tensor([False, False, False, False, True, True])
            for level in (1, 2, 3):
                setattr(graph, f"occupation_level{level}", torch.tensor([1, 2, 1, 0, 0, 0]))
            graph.country = torch.tensor([0, 1, 0, 1, 0, 1])
            graph.temporal = torch.zeros(6, 2)
            feature_schema = {
                "occupation_level1": {"kind": "categorical", "cardinality": 3},
                "occupation_level2": {"kind": "categorical", "cardinality": 3},
                "occupation_level3": {"kind": "categorical", "cardinality": 3},
                "country": {"kind": "categorical", "cardinality": 2},
                "temporal": {"kind": "numeric", "input_dim": 2},
            }
            metadata = {
                "target_column": "occupation_level1",
                "relation_to_id": {
                    "father": 0, "father__rev": 1,
                    "student_of": 2, "student_of__rev": 3,
                },
                "num_relations": 4,
                "label_to_id": {"A": 0, "B": 1},
                "num_classes": 2,
                "occupation_unknown_ids": {
                    "occupation_level1": 0,
                    "occupation_level2": 0,
                    "occupation_level3": 0,
                },
                "feature_schema": feature_schema,
            }
            torch.save({"data": graph, "metadata": metadata}, data_path)
            model = build_model(
                "mlp",
                num_relations=4,
                num_classes=2,
                feature_specs=build_feature_specs(feature_schema, metadata),
                hidden_dim=8,
                branch_dim=4,
                num_layers=2,
                dropout=0.0,
            )
            checkpoint = {
                "model_name": "mlp",
                "model_config": {
                    "hidden_dim": 8, "branch_dim": 4, "num_layers": 2, "dropout": 0.0,
                },
                "model_feature_schema": feature_schema,
                "metadata": metadata,
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            with (root / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump({"run_config": {"num_neighbors": "0,0"}}, handle)
            with taxonomy_path.open("w", encoding="utf-8") as handle:
                json.dump({
                    "name": "synthetic", "version": 1,
                    "groups": {"inherited": ["father"], "acquired": ["student_of"]},
                }, handle)

            for condition in ("mlp_baseline", "rgcn_full"):
                output = predictions_root / condition / "seed_42"
                with patch.object(sys, "argv", [
                    "inequality-predict", "--data", str(data_path),
                    "--checkpoint", str(checkpoint_path), "--output-dir", str(output),
                    "--condition", condition, "--seed", "42", "--device", "cpu",
                    "--num-neighbors", "0,0", "--tie-taxonomy", str(taxonomy_path),
                ]):
                    predict_main()
                self.assertTrue((output / "probabilities.npz").is_file())

            with patch.object(sys, "argv", [
                "inequality-audit", "--data", str(data_path),
                "--predictions-root", str(predictions_root), "--output-dir", str(summary_root),
                "--tie-taxonomy", str(taxonomy_path), "--bootstrap-draws", "2",
            ]):
                audit_main()
            self.assertTrue((summary_root / "paired_comparison_summary.csv").is_file())
            self.assertTrue((summary_root / "test_node_privilege_groups.csv").is_file())


if __name__ == "__main__":
    unittest.main()
