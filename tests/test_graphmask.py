"""Synthetic checks for GraphMask gates, adapters, and command workflows."""

import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from models import build_model
from models.features import FeatureSpec
from training.graphmask.adapter import GraphMaskModelAdapter, restore_graphmask_model
from training.graphmask.core import GraphMaskProbe, HardConcrete, LagrangianOptimization
from training.graphmask_report import main as report_main
from training.graphmask_train import main as train_main


def tiny_model(name: str, backend: str = "fast"):
    extras = {}
    if name == "rgat":
        extras = {"heads": 2, "attention_dropout": 0.0}
    elif name == "rgcn":
        extras = {"num_bases": 2, "rgcn_backend": backend}
    return build_model(
        name,
        num_relations=2,
        num_classes=2,
        feature_specs={"constant": FeatureSpec(kind="constant")},
        hidden_dim=4,
        branch_dim=2,
        num_layers=2,
        dropout=0.0,
        **extras,
    ).eval()


class GraphMaskCoreTests(unittest.TestCase):
    def test_hard_concrete_is_binary_and_has_differentiable_expected_l0(self):
        gate_module = HardConcrete(temperature=1 / 3, location_bias=3.0)
        logits = torch.tensor([-4.0, 0.0, 4.0], requires_grad=True)
        gate_module.eval()
        gate, probability = gate_module(logits)
        self.assertTrue(set(gate.tolist()).issubset({0.0, 1.0}))
        self.assertTrue(bool(((probability > 0) & (probability < 1)).all()))
        probability.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_layer_enabling_and_lagrangian_update(self):
        message = torch.ones(3, 4)
        from training.graphmask.core import LayerTrace
        trace = LayerTrace(
            source_state=message,
            target_state=message,
            message=message,
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_type=torch.zeros(3, dtype=torch.long),
        )
        probe = GraphMaskProbe.from_traces([trace, trace])
        probe.enable_layer(1)
        probe.train()
        gates, _, expected_count, eligible_count = probe([trace, trace])
        self.assertTrue(torch.equal(gates[0], torch.ones(3)))
        self.assertEqual(eligible_count, 3)
        before = probe.gates[1].output.weight.detach().clone()
        optimizer = LagrangianOptimization(
            probe.parameters(), 1e-3, 1e-2, torch.device("cpu")
        )
        optimizer.update(expected_count / eligible_count, gates[1].mean())
        self.assertFalse(torch.equal(before, probe.gates[1].output.weight.detach()))
        self.assertGreater(float(optimizer.multiplier.item()), 0.0)


class GraphMaskAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.features = {"constant": torch.zeros(5, dtype=torch.long)}
        self.edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
        self.edge_type = torch.tensor([0, 1, 0, 1, 0, 1])

    def test_all_models_capture_replace_and_backpropagate_only_to_probe(self):
        for name in ("rgat", "rgcn", "compgcn"):
            with self.subTest(model=name):
                model = tiny_model(name)
                adapter = GraphMaskModelAdapter(model, name)
                logits, traces = adapter.trace(
                    self.features, self.edge_index, self.edge_type
                )
                self.assertEqual(len(traces), 2)
                self.assertEqual(traces[0].message.size(0), self.edge_index.size(1))
                probe = GraphMaskProbe.from_traces(traces)
                for layer in range(2):
                    probe.enable_layer(layer)
                probe.eval()
                gates, _, _, _ = probe(traces)
                all_open = [torch.ones_like(gate) for gate in gates]
                open_logits = adapter.masked_forward(
                    self.features,
                    self.edge_index,
                    self.edge_type,
                    all_open,
                    probe.baselines,
                )
                self.assertTrue(torch.allclose(logits, open_logits, atol=1e-6, rtol=1e-6))

                all_closed = [torch.zeros_like(gate) for gate in gates]
                closed_logits = adapter.masked_forward(
                    self.features,
                    self.edge_index,
                    self.edge_type,
                    all_closed,
                    probe.baselines,
                )
                self.assertFalse(torch.allclose(logits, closed_logits))

                probe.train()
                learned_gates, _, _, _ = probe(traces)
                masked_logits = adapter.masked_forward(
                    self.features,
                    self.edge_index,
                    self.edge_type,
                    learned_gates,
                    probe.baselines,
                )
                masked_logits.sum().backward()
                self.assertTrue(any(
                    parameter.grad is not None for parameter in probe.parameters()
                ))
                self.assertTrue(all(
                    parameter.grad is None for parameter in model.parameters()
                ))

    def test_standard_rgcn_checkpoint_uses_equivalent_fast_copy(self):
        model = tiny_model("rgcn", backend="standard")
        checkpoint = {
            "model_name": "rgcn",
            "model_config": {
                "hidden_dim": 4,
                "branch_dim": 2,
                "num_layers": 2,
                "dropout": 0.0,
                "num_bases": 2,
                "rgcn_backend": "standard",
            },
            "model_feature_schema": {"constant": {"kind": "constant"}},
            "metadata": {
                "num_relations": 2,
                "num_classes": 2,
                "feature_schema": {"constant": {"kind": "constant"}},
            },
            "state_dict": model.state_dict(),
        }
        restored = restore_graphmask_model(checkpoint, torch.device("cpu"))
        self.assertIsNot(restored.adapter.model, restored.adapter.reference_model)
        logits, traces = restored.adapter.trace(
            self.features, self.edge_index, self.edge_type
        )
        self.assertEqual(tuple(logits.shape), (5, 2))
        self.assertEqual(len(traces), 2)


class GraphMaskWorkflowTests(unittest.TestCase):
    def test_train_and_report_commands_create_reloadable_artifacts(self):
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "graph_data.pt"
            checkpoint_path = root / "best_model.pt"
            train_output = root / "probe"
            report_output = root / "report"
            graph = Data(
                edge_index=torch.tensor([[2, 3, 0, 1, 0, 1], [0, 1, 2, 2, 3, 3]]),
                edge_type=torch.tensor([0, 1, 0, 1, 0, 1]),
                y=torch.tensor([0, 1, 0, 1]),
                num_nodes=4,
            )
            graph.train_mask = torch.tensor([True, True, False, False])
            graph.val_mask = torch.tensor([False, False, True, False])
            graph.test_mask = torch.tensor([False, False, False, True])
            metadata = {
                "relation_to_id": {"signal": 0, "signal__rev": 1},
                "num_relations": 2,
                "label_to_id": {"A": 0, "B": 1},
                "num_classes": 2,
                "occupation_unknown_ids": {},
                "feature_schema": {"constant": {"kind": "constant"}},
            }
            torch.save({"data": graph, "metadata": metadata}, data_path)
            model = build_model(
                "rgat",
                num_relations=2,
                num_classes=2,
                feature_specs={"constant": FeatureSpec(kind="constant")},
                hidden_dim=4,
                branch_dim=2,
                num_layers=1,
                heads=1,
                dropout=0.0,
                attention_dropout=0.0,
            )
            checkpoint = {
                "model_name": "rgat",
                "model_config": {
                    "hidden_dim": 4,
                    "branch_dim": 2,
                    "num_layers": 1,
                    "heads": 1,
                    "dropout": 0.0,
                    "attention_dropout": 0.0,
                },
                "model_feature_schema": metadata["feature_schema"],
                "metadata": metadata,
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            with patch.object(sys, "argv", [
                "graphmask-train", "--data", str(data_path), "--checkpoint",
                str(checkpoint_path), "--output-dir", str(train_output),
                "--num-neighbors", "full", "--batch-size", "1",
                "--epochs-per-layer", "1", "--device", "cpu",
            ]):
                train_main()
            probe_path = train_output / "graphmask_probe.pt"
            self.assertTrue(probe_path.is_file())
            self.assertTrue((train_output / "validation.json").is_file())

            with patch.object(sys, "argv", [
                "graphmask-report", "--data", str(data_path), "--checkpoint",
                str(checkpoint_path), "--probe", str(probe_path), "--output-dir",
                str(report_output), "--split", "test", "--device", "cpu", "--top-k", "2",
            ]):
                report_main()
            for filename in (
                "test_metrics.json", "relations_directed.csv", "relations_base.csv",
                "root_top_edges.csv.gz", "manifest.json",
            ):
                self.assertTrue((report_output / filename).is_file(), filename)
            with (report_output / "relations_directed.csv").open(encoding="utf-8") as handle:
                directed = list(csv.DictReader(handle))
            with (report_output / "relations_base.csv").open(encoding="utf-8") as handle:
                merged = list(csv.DictReader(handle))
            self.assertEqual({row["relation"] for row in directed}, {"signal", "signal__rev"})
            self.assertEqual({row["relation"] for row in merged}, {"signal"})
            with gzip.open(
                report_output / "root_top_edges.csv.gz", "rt", encoding="utf-8"
            ) as handle:
                top_edges = list(csv.DictReader(handle))
            self.assertTrue(top_edges)
            self.assertIn("keep_probability", top_edges[0])


if __name__ == "__main__":
    unittest.main()
