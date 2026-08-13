"""Exactness checks for the fast all-relation-pair counterfactual sweep."""

import unittest

import torch

from models.features import FeatureSpec
from models.rgat import RelationalGATClassifier
from training.relation_pair_sweep import (
    _base_head_aggregates,
    _counterfactual_root_logits,
    _logits_from_head_aggregate,
    _tie_group_summary_rows,
)


class RelationPairSweepTests(unittest.TestCase):
    def test_reconstruction_and_deleted_group_match_a_direct_rgat_reforward(self):
        torch.manual_seed(91)
        model = RelationalGATClassifier(
            num_relations=2,
            num_classes=3,
            feature_specs={"constant": FeatureSpec(kind="constant")},
            hidden_dim=6,
            branch_dim=3,
            num_layers=1,
            heads=2,
            dropout=0.0,
            attention_dropout=0.0,
        ).eval()
        features = {"constant": torch.zeros(5, dtype=torch.long)}
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 4]])
        edge_type = torch.tensor([0, 1, 0])

        logits, explanation = model(features, edge_index, edge_type, return_attention_weights=True)
        layer = explanation["attention_layers"][0]
        head_aggregate, alpha, weighted_values, _ = _base_head_aggregates(model, layer)
        reconstructed = _logits_from_head_aggregate(model, layer["input_node_state"], head_aggregate)
        self.assertTrue(torch.allclose(logits, reconstructed, atol=1e-6, rtol=1e-6))

        # Delete the first of two messages entering root 3.  The fast formula
        # must equal an ordinary direct forward on the filtered graph.
        analytical = _counterfactual_root_logits(
            model,
            layer["input_node_state"],
            head_aggregate,
            root_index=3,
            removed_mass=alpha[0],
            removed_weighted_values=weighted_values[0],
        )
        direct = model(features, edge_index[:, 1:], edge_type[1:])
        self.assertTrue(torch.allclose(analytical, direct[3], atol=1e-6, rtol=1e-6))

    def test_removing_the_only_message_yields_the_zero_aggregate_before_bias(self):
        torch.manual_seed(92)
        model = RelationalGATClassifier(
            num_relations=1,
            num_classes=2,
            feature_specs={"constant": FeatureSpec(kind="constant")},
            hidden_dim=4,
            branch_dim=2,
            num_layers=1,
            heads=2,
            dropout=0.0,
            attention_dropout=0.0,
        ).eval()
        features = {"constant": torch.zeros(2, dtype=torch.long)}
        edge_index = torch.tensor([[0], [1]])
        edge_type = torch.tensor([0])
        _, explanation = model(features, edge_index, edge_type, return_attention_weights=True)
        layer = explanation["attention_layers"][0]
        head_aggregate, alpha, weighted_values, _ = _base_head_aggregates(model, layer)
        analytical = _counterfactual_root_logits(
            model,
            layer["input_node_state"],
            head_aggregate,
            root_index=1,
            removed_mass=alpha[0],
            removed_weighted_values=weighted_values[0],
        )
        direct = model(features, torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.allclose(analytical, direct[1], atol=1e-6, rtol=1e-6))

    def test_tie_group_summary_pools_root_level_records_without_mixing_groups(self):
        shared = {
            "experiment": "rgat_one_hop",
            "seed": "42",
            "checkpoint": "model.pt",
            "split": "test",
            "pair_edge_count": 1,
            "pair_margin_drop": 0.4,
            "base_predicts_target": True,
            "pair_flips_away_from_target": False,
            "has_matched_control": True,
            "control_mean_margin_drop": 0.1,
            "pair_minus_control_margin_drop": 0.3,
            "control_flip_away_rate": 0.0,
        }
        records = [
            {**shared, "tie_group": "inherited"},
            {**shared, "tie_group": "inherited", "pair_margin_drop": 0.6},
            {**shared, "tie_group": "acquired", "pair_margin_drop": 0.2},
        ]
        summary = _tie_group_summary_rows(records)
        self.assertEqual([row["tie_group"] for row in summary], ["acquired", "inherited"])
        inherited = summary[1]
        self.assertEqual(inherited["motif_eligible_root_n"], 2)
        self.assertAlmostEqual(inherited["mean_pair_margin_drop"], 0.5)


if __name__ == "__main__":
    unittest.main()
