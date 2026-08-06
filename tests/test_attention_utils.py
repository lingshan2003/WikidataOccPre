"""Regression tests for relation-ID alignment in RGAT attention exports."""

import unittest

import torch

from training.attention_utils import attention_relation_ids


class AttentionUtilsTests(unittest.TestCase):
    def test_direct_attention_edges_keep_input_relation_ids(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_type = torch.tensor([3, 4, 5])
        result = attention_relation_ids({
            "edge_index": edge_index,
            "input_edge_index": edge_index,
            "input_edge_type": edge_type,
            "edge_type": edge_type,
        })
        self.assertEqual(result.tolist(), [3, 4, 5])

    def test_synthetic_self_loops_are_never_assigned_a_real_relation(self):
        input_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        input_type = torch.tensor([3, 4, 5])
        returned_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 1, 2]])
        result = attention_relation_ids({
            "edge_index": returned_index,
            "input_edge_index": input_index,
            "input_edge_type": input_type,
            "edge_type": input_type,
        })
        self.assertEqual(result.tolist(), [3, 4, 5, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
