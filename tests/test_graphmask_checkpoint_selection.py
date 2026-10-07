"""Checkpoint selection behavior for the staged GraphMask probe trainer."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from models import build_model
from models.features import FeatureSpec
from training.graphmask.common import load_probe
from training.graphmask_train import main as graphmask_train_main, parse_args as parse_graphmask_train_args


def _validation(retention, *, masked_f1=0.5):
    return {
        "roots": 2,
        "labeled_roots": 2,
        "original": {"accuracy": 0.5, "macro_f1": 0.5, "weighted_f1": 0.5},
        "masked": {"accuracy": masked_f1, "macro_f1": masked_f1, "weighted_f1": masked_f1},
        "prediction_agreement": 1.0,
        "mean_kl": 0.0,
        "hard_retention_rate": retention,
        "mean_keep_probability": retention,
        "layers": [],
    }


class GraphMaskCheckpointSelectionTests(unittest.TestCase):
    def test_any_stage_remains_the_default_cli_policy(self):
        with patch.object(sys, "argv", [
            "graphmask-train", "--data", "data.pt", "--checkpoint", "model.pt",
            "--output-dir", "probe",
        ]):
            args = parse_graphmask_train_args()
        self.assertEqual(args.checkpoint_selection, "any-stage")

    def make_inputs(self, root):
        data_path = root / "graph_data.pt"
        checkpoint_path = root / "best_model.pt"

        sources = list(range(8)) + list(range(8))
        targets = [(node + 1) % 8 for node in range(8)] + [(node - 1) % 8 for node in range(8)]
        graph = Data(
            edge_index=torch.tensor([sources, targets], dtype=torch.long),
            edge_type=torch.tensor([0, 1] * 8, dtype=torch.long),
            y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
            num_nodes=8,
        )
        graph.train_mask = torch.tensor([True, True, True, True, False, False, False, False])
        graph.val_mask = torch.tensor([False, False, False, False, True, True, False, False])
        graph.test_mask = torch.tensor([False, False, False, False, False, False, True, True])

        feature_schema = {"structural_constant": {"kind": "constant"}}
        metadata = {
            "relation_to_id": {"signal": 0, "signal__rev": 1},
            "num_relations": 2,
            "label_to_id": {"A": 0, "B": 1},
            "num_classes": 2,
            "occupation_unknown_ids": {},
            "feature_schema": feature_schema,
        }
        torch.save({"data": graph, "metadata": metadata}, data_path)

        model = build_model(
            "rgcn",
            num_relations=2,
            num_classes=2,
            feature_specs={"structural_constant": FeatureSpec(kind="constant")},
            hidden_dim=4,
            branch_dim=2,
            num_layers=2,
            dropout=0.0,
            num_bases=2,
            rgcn_backend="fast",
        )
        torch.save({
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "model_feature_schema": feature_schema,
            "model_name": "rgcn",
            "model_config": {
                "hidden_dim": 4,
                "branch_dim": 2,
                "num_layers": 2,
                "dropout": 0.0,
                "num_bases": 2,
                "rgcn_backend": "fast",
            },
        }, checkpoint_path)
        return data_path, checkpoint_path

    def run_training(self, root, policy, validations):
        data_path, checkpoint_path = self.make_inputs(root)
        output_dir = root / f"probe-{policy}"
        argv = [
            "graphmask-train", "--data", str(data_path),
            "--checkpoint", str(checkpoint_path), "--output-dir", str(output_dir),
            "--num-neighbors", "full", "--batch-size", "4",
            "--num-workers", "0", "--epochs-per-layer", "1",
            "--max-relative-macro-f1-diff", "0.05", "--seed", "7",
            "--device", "cpu", "--checkpoint-selection", policy,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, "argv", argv), patch(
                "training.graphmask_train.evaluate_probe", side_effect=validations
            ) as evaluate:
                graphmask_train_main()
        return output_dir, evaluate.call_count

    def test_candidate_policy_selects_early_or_all_layer_checkpoint(self):
        early = _validation(0.1)
        later = _validation(0.8)
        for policy, expected_epoch, expected_enabled, expected_validation, exact_validation in (
            ("any-stage", 1, [False, True], early, early),
            ("all-layers-enabled", 2, [True, True], later, later),
        ):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temporary:
                output_dir, eval_count = self.run_training(
                    Path(temporary), policy, [early, later, exact_validation]
                )
                self.assertEqual(eval_count, 3)

                history = json.loads((output_dir / "training_history.json").read_text())
                self.assertEqual(history["history"][0]["global_epoch"], 1)
                self.assertEqual(history["history"][0]["enabled_layers"], [False, True])
                self.assertTrue(history["history"][0]["eligible"])
                self.assertEqual(history["history"][1]["enabled_layers"], [True, True])
                self.assertTrue(history["history"][1]["eligible"])
                self.assertEqual(
                    history["history"][0]["selection_eligible"], policy == "any-stage"
                )
                self.assertTrue(history["history"][1]["selection_eligible"])

                selected = history["selected_checkpoint"]
                self.assertEqual(selected["policy"], policy)
                self.assertEqual(selected["global_epoch"], expected_epoch)
                self.assertEqual(selected["enabled_layers"], expected_enabled)
                self.assertAlmostEqual(selected["validation_hard_retention_rate"], expected_validation["hard_retention_rate"])
                self.assertEqual(selected["enabled_through_layer"], 1 if expected_epoch == 1 else 0)
                self.assertEqual(selected["layer_epoch"], 1)

                payload = torch.load(output_dir / "graphmask_probe.pt", map_location="cpu", weights_only=False)
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(payload["selected_checkpoint"], selected)
                self.assertEqual(manifest["selected_checkpoint"], selected)
                saved_validation = json.loads((output_dir / "validation.json").read_text())
                self.assertEqual(saved_validation["hard_retention_rate"], expected_validation["hard_retention_rate"])

                probe, _ = load_probe(output_dir / "graphmask_probe.pt", torch.device("cpu"))
                self.assertEqual(probe.enabled_layers.tolist(), expected_enabled)

    def test_strict_policy_fails_when_only_early_stage_meets_fidelity(self):
        early_pass = _validation(0.1)
        later_fail = _validation(0.8, masked_f1=0.4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path, checkpoint_path = self.make_inputs(root)
            output_dir = root / "probe-strict-failure"
            argv = [
                "graphmask-train", "--data", str(data_path),
                "--checkpoint", str(checkpoint_path), "--output-dir", str(output_dir),
                "--num-neighbors", "full", "--batch-size", "4",
                "--num-workers", "0", "--epochs-per-layer", "1",
                "--max-relative-macro-f1-diff", "0.05", "--seed", "7",
                "--device", "cpu", "--checkpoint-selection", "all-layers-enabled",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", argv), patch(
                    "training.graphmask_train.evaluate_probe", side_effect=[early_pass, later_fail]
                ) as evaluate:
                    with self.assertRaisesRegex(RuntimeError, "No GraphMask probe satisfied"):
                        graphmask_train_main()

            self.assertEqual(evaluate.call_count, 2)
            history_path = output_dir / "training_history.json"
            self.assertTrue(history_path.is_file())
            history = json.loads(history_path.read_text())
            self.assertIsNone(history["selected_checkpoint"])
            self.assertTrue(history["history"][0]["eligible"])
            self.assertFalse(history["history"][0]["selection_eligible"])
            self.assertFalse(history["history"][1]["eligible"])
            self.assertFalse(history["history"][1]["selection_eligible"])
            self.assertFalse((output_dir / "graphmask_probe.pt").exists())
            self.assertFalse((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
