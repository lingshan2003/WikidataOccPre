"""Synthetic checks for the RGAT gradient-times-attention exporter."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from models.rgat import RelationalGATClassifier
from models.features import FeatureSpec
from training.gradient_attribution import (
    collect_checkpoint_gradient_attribution,
    root_prediction_score,
)


class GradientAttributionTests(unittest.TestCase):
    def test_margin_score_uses_fixed_selected_class_and_logsumexp_competitor(self):
        logits = torch.tensor([[3.0, 1.0, -2.0]], requires_grad=True)
        score, selected = root_prediction_score(
            logits, torch.tensor([0]), torch.tensor([1]), "predicted-margin"
        )
        self.assertEqual(selected.tolist(), [0])
        expected = 3.0 - torch.logsumexp(torch.tensor([1.0, -2.0]), dim=0)
        self.assertAlmostEqual(float(score.item()), float(expected.item()))
        score.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0]), 1.0)

    def test_returned_pyg_alpha_is_differentiable_with_respect_to_a_prediction(self):
        torch.manual_seed(17)
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
        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_type = torch.tensor([0, 0])
        logits, explanation = model(
            {"constant": torch.zeros(3, dtype=torch.long)}, edge_index, edge_type, return_attention_weights=True
        )
        alpha = explanation["attention_layers"][0]["alpha"]
        gradient = torch.autograd.grad(logits[2, 0], alpha)[0]
        self.assertEqual(tuple(gradient.shape), tuple(alpha.shape))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_real_rgat_checkpoint_runs_through_the_exporter(self):
        torch.manual_seed(19)
        graph = Data(
            edge_index=torch.tensor([[0, 1], [2, 2]]),
            edge_type=torch.tensor([0, 0]),
            y=torch.tensor([0, 0, 1]),
            num_nodes=3,
        )
        graph.train_mask = torch.tensor([True, True, False])
        graph.val_mask = torch.tensor([False, False, False])
        graph.test_mask = torch.tensor([False, False, True])
        metadata = {
            "relation_to_id": {"child": 0},
            "num_relations": 1,
            "label_to_id": {"Culture": 0, "Science": 1},
            "num_classes": 2,
            "occupation_unknown_ids": {},
            "feature_schema": {"constant": {"kind": "constant"}},
        }
        config = {
            "hidden_dim": 4,
            "branch_dim": 2,
            "num_layers": 1,
            "heads": 2,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        }
        model = RelationalGATClassifier(
            num_relations=1,
            num_classes=2,
            feature_specs={"constant": FeatureSpec(kind="constant")},
            **config,
        )
        checkpoint = {
            "metadata": metadata,
            "model_name": "rgat",
            "model_feature_schema": metadata["feature_schema"],
            "model_config": config,
            "state_dict": model.state_dict(),
        }
        with patch("training.gradient_attribution.torch.load", return_value=checkpoint), patch(
            "training.gradient_attribution.sha256_file", return_value="synthetic"
        ):
            sparse, roster, _ = collect_checkpoint_gradient_attribution(
                Path("seed_42/best_model.pt"),
                graph,
                metadata,
                split="test",
                score_name="predicted-margin",
                requested_fanouts="full",
                batch_size=1,
                num_workers=0,
                device=torch.device("cpu"),
                forward_mode="full-graph",
            )
        self.assertEqual(len(roster), 1)
        self.assertEqual(len(sparse), 1)
        self.assertTrue(torch.isfinite(torch.tensor(sparse[0]["gradient_x_attention"])))

    def test_export_sums_edge_products_inside_the_target_person_group(self):
        graph = Data(
            edge_index=torch.tensor([[0, 1], [2, 2]]),
            edge_type=torch.tensor([0, 0]),
            y=torch.tensor([0, 0, 1]),
            num_nodes=3,
        )
        graph.train_mask = torch.tensor([True, True, False])
        graph.val_mask = torch.tensor([False, False, False])
        graph.test_mask = torch.tensor([False, False, True])
        metadata = {
            "relation_to_id": {"child": 0},
            "label_to_id": {"Culture": 0, "Science": 1},
            "num_classes": 2,
            "occupation_unknown_ids": {},
        }
        checkpoint = {"metadata": metadata, "model_config": {"num_layers": 1}}

        class FakeModel:
            convs = [object()]

            def eval(self):
                return self

            def __call__(self, *_args, **_kwargs):
                alpha = torch.tensor([[0.2, 0.4], [0.3, 0.5]], requires_grad=True)
                # At root 2, dF/dalpha is [1, 1] for edge 0 and [2, 2] for edge 1.
                root_logit = alpha[0].sum() + 2.0 * alpha[1].sum()
                zero = torch.zeros_like(root_logit)
                logits = torch.stack((
                    torch.stack((zero, zero)),
                    torch.stack((zero, zero)),
                    torch.stack((root_logit, zero)),
                ))
                explanation = {
                    "attention_layers": [{
                        "layer": 0,
                        "edge_index": graph.edge_index,
                        "input_edge_index": graph.edge_index,
                        "edge_type": graph.edge_type,
                        "input_edge_type": graph.edge_type,
                        "alpha": alpha,
                    }]
                }
                return logits, explanation

        with patch("training.gradient_attribution.torch.load", return_value=checkpoint), patch(
            "training.gradient_attribution.restore_rgat", return_value=(FakeModel(), {}, metadata)
        ), patch("training.gradient_attribution.sha256_file", return_value="synthetic"):
            sparse, roster, _ = collect_checkpoint_gradient_attribution(
                Path("synthetic.pt"),
                graph,
                metadata,
                split="test",
                score_name="predicted-logit",
                requested_fanouts="full",
                batch_size=1,
                num_workers=0,
                device=torch.device("cpu"),
                forward_mode="full-graph",
            )

        self.assertEqual(len(roster), 1)
        self.assertEqual(len(sparse), 1)
        self.assertAlmostEqual(roster[0]["typed_attention_mass"], 0.7)
        # mean_h((.2, .4) * 1 + (.3, .5) * 2) = .3 + .8 = 1.1
        self.assertAlmostEqual(roster[0]["typed_gradient_x_attention"], 1.1)
        self.assertEqual(sparse[0]["candidate_edge_count"], 2)
        self.assertAlmostEqual(sparse[0]["attention_mass"], 0.7)
        self.assertAlmostEqual(sparse[0]["gradient_x_attention"], 1.1)


if __name__ == "__main__":
    unittest.main()
