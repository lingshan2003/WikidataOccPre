"""Synthetic checks for conditional occupation-pair relation ablation."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from training.relation_pair_ablation import (
    _draw_matched_control_mask,
    _edge_groups_for_roots,
    collect_checkpoint_relation_pair_ablation,
    target_margin,
)
from training.tie_taxonomy import TieTaxonomy


class RelationPairAblationTests(unittest.TestCase):
    def _graph(self):
        # 0 and 1 are observed Leadership sources. Root 3 is Leadership and
        # receives one selected child edge and one matched non-child control.
        graph = Data(
            edge_index=torch.tensor([[0, 1, 2, 0], [3, 3, 3, 4]]),
            edge_type=torch.tensor([0, 1, 0, 0]),
            y=torch.tensor([0, 0, 1, 0, 1]),
            num_nodes=5,
        )
        graph.train_mask = torch.tensor([True, True, True, False, False])
        graph.val_mask = torch.tensor([False, False, False, False, False])
        graph.test_mask = torch.tensor([False, False, False, True, True])
        # Value zero is the unknown category; train sources expose L1 while
        # both held-out roots remain masked.
        graph.occupation_level1 = torch.tensor([1, 1, 2, 0, 0])
        return graph

    def _metadata(self):
        return {
            "relation_to_id": {"child": 0, "spouse": 1},
            "num_relations": 2,
            "label_to_id": {"Leadership": 0, "Culture": 1},
            "num_classes": 2,
            "occupation_unknown_ids": {"occupation_level1": 0},
            "feature_schema": {"occupation_level1": {"kind": "categorical"}},
        }

    def test_margin_uses_fixed_target_class(self):
        logits = torch.tensor([[2.0, -1.0], [3.0, 4.0]])
        values = target_margin(logits, torch.tensor([0, 1]), 0)
        self.assertAlmostEqual(float(values[0]), 3.0)
        self.assertAlmostEqual(float(values[1]), -1.0)

    def test_pair_edges_and_controls_are_matched_within_the_same_root(self):
        graph = self._graph()
        pair_drop, candidates, eligible, counts = _edge_groups_for_roots(
            graph,
            torch.tensor([3, 4]),
            source_label_id=0,
            relation_id=0,
            target_label_id=0,
            feature_schema=self._metadata()["feature_schema"],
            occupation_unknown_ids={"occupation_level1": 0},
        )
        self.assertEqual(pair_drop.tolist(), [True, False, False, False])
        self.assertEqual(eligible.tolist(), [True, False])
        self.assertEqual(counts.tolist(), [1, 0])
        self.assertEqual(candidates[0].tolist(), [1])
        self.assertEqual(candidates[1].tolist(), [])

        control_drop, paired = _draw_matched_control_mask(
            candidates,
            counts,
            eligible,
            edge_count=4,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(control_drop.tolist(), [False, True, False, False])
        self.assertEqual(paired.tolist(), [True, False])

    def test_frozen_model_reports_selected_relation_drop_above_control_drop(self):
        graph = self._graph()
        metadata = self._metadata()
        checkpoint = {"metadata": metadata, "model_config": {"num_layers": 1}}
        taxonomy = TieTaxonomy(
            name="synthetic",
            version=1,
            path=Path("synthetic.json"),
            sha256="synthetic",
            inherited=("child",),
            acquired=("spouse",),
        )

        class FakeModel:
            convs = [object()]

            def eval(self):
                return self

            def __call__(self, _features, edge_index, edge_type):
                logits = torch.zeros((5, 2), dtype=torch.float)
                # Only a child message into root 3 supports Leadership.
                supports = ((edge_type.eq(0)) & edge_index[1].eq(3)).sum().float()
                logits[3, 0] = supports * 10.0
                return logits

        with patch("training.relation_pair_ablation.torch.load", return_value=checkpoint), patch(
            "training.relation_pair_ablation.restore_rgat",
            return_value=(FakeModel(), metadata["feature_schema"], metadata),
        ), patch("training.relation_pair_ablation.sha256_file", return_value="synthetic"):
            records, summary, _ = collect_checkpoint_relation_pair_ablation(
                Path("seed_42/best_model.pt"),
                graph,
                metadata,
                taxonomy,
                split="test",
                source_l1="Leadership",
                relation="child",
                target_l1="Leadership",
                requested_fanouts="full",
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
                forward_mode="full-graph",
                control_draws=3,
                max_roots=None,
                analysis_seed=8,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["pair_edge_count"], 1)
        self.assertEqual(records[0]["tie_group"], "inherited")
        self.assertTrue(records[0]["has_matched_control"])
        self.assertAlmostEqual(records[0]["pair_margin_drop"], 10.0)
        self.assertAlmostEqual(records[0]["control_mean_margin_drop"], 0.0)
        self.assertAlmostEqual(records[0]["pair_minus_control_margin_drop"], 10.0)
        self.assertEqual(summary["motif_eligible_root_n"], 1)
        self.assertEqual(summary["matched_control_root_n"], 1)


if __name__ == "__main__":
    unittest.main()
