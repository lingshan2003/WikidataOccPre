"""Synthetic tests for value-aware RGAT message extraction."""

import unittest
from types import SimpleNamespace

import torch

from training.message_contribution import relation_value_vectors


class MessageContributionTests(unittest.TestCase):
    def _conv(self):
        # One relation, two input dimensions, two heads, one output dimension.
        # h @ W therefore gives [2, 6] for h=[1, 2].
        return SimpleNamespace(
            attention_mode="additive-self-attention",
            concat=False,
            mod=None,
            num_bases=None,
            num_blocks=None,
            weight=torch.tensor([[[2.0, 0.0], [0.0, 3.0]]]),
            heads=2,
            out_channels=1,
        )

    def test_relation_value_vectors_match_pyg_value_projection(self):
        values = relation_value_vectors(
            self._conv(),
            torch.tensor([[1.0, 2.0], [5.0, 7.0]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        self.assertEqual(tuple(values.shape), (1, 2, 1))
        self.assertEqual(values.flatten().tolist(), [2.0, 6.0])
        alpha = torch.tensor([[0.5, 0.25]])
        contribution = (alpha.unsqueeze(-1) * values).mean(dim=1)
        self.assertAlmostEqual(float(contribution.item()), 1.25)

    def test_cardinality_modified_rgat_is_rejected(self):
        conv = self._conv()
        conv.mod = "scaled"
        with self.assertRaises(ValueError):
            relation_value_vectors(
                conv,
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([0]),
                torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
