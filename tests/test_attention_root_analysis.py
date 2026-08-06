"""Synthetic correctness tests for root-aware RGAT attention analysis."""

import unittest

import torch
from torch_geometric.data import Data

from training.attention_report import source_visibility_codes, validate_full_graph_root_mask
from training.attention_rollout import _rollout_for_batch


class RootAttentionAnalysisTests(unittest.TestCase):
    def _graph(self):
        graph = Data(
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_type=torch.tensor([4, 5]),
            y=torch.tensor([0, 1, 2]),
            num_nodes=3,
        )
        graph.train_mask = torch.tensor([True, False, False])
        graph.val_mask = torch.tensor([False, True, False])
        graph.test_mask = torch.tensor([False, False, True])
        # Node 0 is an observed train source; validation/test nodes are hidden.
        graph.occupation_level1 = torch.tensor([1, 0, 0])
        return graph

    def test_full_graph_rejects_training_roots_and_visible_held_out_roots(self):
        graph = self._graph()
        schema = {"occupation_level1": {"kind": "categorical"}}
        unknown = {"occupation_level1": 0}
        validate_full_graph_root_mask(graph, torch.tensor([2]), "test", schema, unknown)
        with self.assertRaises(ValueError):
            validate_full_graph_root_mask(graph, torch.tensor([0]), "train", schema, unknown)
        graph.occupation_level1[2] = 2
        with self.assertRaises(RuntimeError):
            validate_full_graph_root_mask(graph, torch.tensor([2]), "test", schema, unknown)

    def test_source_visibility_tracks_model_input_not_posthoc_label_only(self):
        graph = self._graph()
        graph.y = torch.tensor([0, 1, -1])
        codes = source_visibility_codes(
            graph,
            torch.tensor([0, 1, 2]),
            {"occupation_level1": {"kind": "categorical"}},
            {"occupation_level1": 0},
        )
        self.assertEqual(codes.tolist(), [0, 1, 2])

    def test_rollout_multiplies_paths_and_keeps_relation_pairs_separate(self):
        graph = self._graph()
        graph.num_nodes = 5
        graph.y = torch.tensor([0, 1, 2, 0, 1])
        graph.train_mask = torch.tensor([True, False, False, True, False])
        graph.occupation_level1 = torch.tensor([1, 0, 0, 1, 0])
        # Source 0 reaches root 2 via both intermediate 1 and intermediate 4.
        layer1_edge = torch.tensor([[0, 3, 0], [1, 1, 4]])
        layer1_type = torch.tensor([4, 6, 4])
        layer2_edge = torch.tensor([[1, 4], [2, 2]])
        layer2_type = torch.tensor([5, 5])
        explanation = {
            "attention_layers": [
                {
                    "layer": 0,
                    "edge_index": layer1_edge,
                    "input_edge_index": layer1_edge,
                    "edge_type": layer1_type,
                    "input_edge_type": layer1_type,
                    "alpha": torch.tensor([[0.5, 0.5], [0.25, 0.25], [0.1, 0.1]]),
                },
                {
                    "layer": 1,
                    "edge_index": layer2_edge,
                    "input_edge_index": layer2_edge,
                    "edge_type": layer2_type,
                    "input_edge_type": layer2_type,
                    "alpha": torch.tensor([[0.6, 0.6], [0.2, 0.2]]),
                },
            ]
        }
        aggregate, totals, diagnostics = _rollout_for_batch(
            graph,
            torch.tensor([2]),
            explanation,
            {"occupation_level1": {"kind": "categorical"}},
            {"occupation_level1": 0},
            class_count=3,
            relation_slots=8,
        )
        self.assertEqual(diagnostics["typed_layer2_root_edges"], 2)
        self.assertEqual(aggregate[(0, 4, 5, 0, 0)][0], 2.0)
        self.assertAlmostEqual(aggregate[(0, 4, 5, 0, 0)][1], 0.32)
        self.assertEqual(aggregate[(0, 6, 5, 0, 0)][0], 1.0)
        self.assertAlmostEqual(aggregate[(0, 6, 5, 0, 0)][1], 0.15)
        self.assertEqual(totals[0][0], 3.0)
        self.assertAlmostEqual(totals[0][1], 0.47)


if __name__ == "__main__":
    unittest.main()
