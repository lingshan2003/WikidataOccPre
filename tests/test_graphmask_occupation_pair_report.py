"""Tests for direct GraphMask L1 occupation-pair aggregation and reporting."""

import csv
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
from training.graphmask.adapter import GraphMaskModelAdapter
from training.graphmask.common import probe_payload
from training.graphmask.core import GraphMaskProbe
from training.graphmask_occupation_pair_report import (
    COHORT_ALL,
    PairAggregate,
    PairKey,
    group_direct_messages,
    main as occupation_pair_main,
    ranked_rows,
    summarize_pair_aggregates,
)


class GraphMaskOccupationPairAggregationTests(unittest.TestCase):
    def test_direct_grouping_excludes_internal_targets_and_preserves_relation_direction(self):
        edge_index = torch.tensor([
            [1, 2, 3, 1, 4],
            [0, 0, 0, 0, 2],
        ])
        edge_type = torch.tensor([0, 1, 1, 1, 0])
        labels = torch.tensor([1, 0, 1, -1, 0])
        visibility = torch.tensor([1, 0, 1, 2, 0])
        gate = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
        probability = torch.tensor([0.9, 0.2, 0.8, 0.7, 0.6])

        groups = group_direct_messages(
            edge_index,
            edge_type,
            labels,
            visibility,
            gate,
            probability,
            class_count=2,
        )

        self.assertEqual(len(groups), 4)
        keyed = {
            (group.relation_id, group.source_l1_id, group.visibility_id): group
            for group in groups
        }
        self.assertEqual(keyed[(0, 0, 0)].hard_retained_message_n, 1)
        self.assertEqual(keyed[(1, 0, 0)].hard_retained_message_n, 1)
        self.assertEqual(keyed[(1, 1, 1)].hard_retained_message_n, 0)
        self.assertEqual(keyed[(1, -1, 2)].hard_retained_message_n, 1)
        # The final known source targets an internal node and is not counted.
        self.assertEqual(sum(group.candidate_message_n for group in groups), 4)

    def test_shares_sum_to_one_and_lift_adjusts_for_opportunity(self):
        target_counts = {(0, 1, "Target"): 200}
        aggregates = {}
        specifications = [
            (0, "A", 0, "signal", 100, 50),
            (1, "B", 0, "signal", 100, 50),
            (0, "A", 1, "signal__rev", 100, 90),
            (1, "B", 1, "signal__rev", 100, 10),
        ]
        for source_id, source, relation_id, relation, candidate, hard in specifications:
            key = PairKey(
                0, relation_id, relation, source_id, source, 1, "Target"
            )
            stats = PairAggregate()
            # One candidate message per root gives exactly 100-node support.
            retained_remaining = hard
            expected_probability = hard / candidate
            for _ in range(candidate):
                stats.add_root_group(
                    1,
                    int(retained_remaining > 0),
                    expected_probability,
                )
                retained_remaining -= int(retained_remaining > 0)
            aggregates[key] = stats

        rows, budgets = summarize_pair_aggregates(
            aggregates,
            target_counts,
            COHORT_ALL,
            min_node_support=100,
        )
        self.assertAlmostEqual(
            sum(float(row["retained_share_of_Ot_budget"]) for row in rows),
            1.0,
        )
        self.assertAlmostEqual(
            sum(float(row["opportunity_share_of_Ot_budget"]) for row in rows),
            1.0,
        )
        self.assertAlmostEqual(
            sum(float(row["expected_retained_share_of_Ot_budget"]) for row in rows),
            1.0,
        )
        for relation in ("signal", "signal__rev"):
            relation_rows = [row for row in rows if row["relation"] == relation]
            self.assertAlmostEqual(
                sum(float(row["retained_share_within_R_Ot"]) for row in relation_rows),
                1.0,
            )
            self.assertAlmostEqual(
                sum(float(row["opportunity_share_within_R_Ot"]) for row in relation_rows),
                1.0,
            )
        by_cell = {(row["relation"], row["source_l1"]): row for row in rows}
        self.assertAlmostEqual(
            float(by_cell[("signal", "A")]["retention_lift_within_R_Ot"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(by_cell[("signal__rev", "A")]["retention_lift_within_R_Ot"]),
            1.8,
        )
        self.assertAlmostEqual(
            float(by_cell[("signal__rev", "B")]["retention_lift_within_R_Ot"]),
            0.2,
        )
        self.assertTrue(all(bool(row["support_eligible"]) for row in rows))
        self.assertEqual(len(ranked_rows(rows)), 4)
        self.assertTrue(any(row["budget_scope"] == "target_occupation" for row in budgets))


class GraphMaskOccupationPairWorkflowTests(unittest.TestCase):
    def test_command_writes_reloadable_pair_tables_and_visibility_audit(self):
        torch.manual_seed(31)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "graph_data.pt"
            checkpoint_path = root / "best_model.pt"
            probe_path = root / "graphmask_probe.pt"
            output_dir = root / "pair_report"
            graph = Data(
                edge_index=torch.tensor([
                    [0, 1, 4, 1, 2, 3, 0, 1],
                    [3, 3, 3, 5, 0, 1, 2, 2],
                ]),
                edge_type=torch.tensor([0, 1, 0, 0, 0, 1, 0, 1]),
                y=torch.tensor([0, 1, 0, 1, -1, 0]),
                num_nodes=6,
            )
            graph.train_mask = torch.tensor([True, True, False, False, False, False])
            graph.val_mask = torch.tensor([False, False, True, False, False, False])
            graph.test_mask = torch.tensor([False, False, False, True, False, True])
            metadata = {
                "target_column": "occupation_level1",
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
            ).eval()
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.copy_(torch.tensor([0.0, 10.0]))
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

            adapter = GraphMaskModelAdapter(model, "rgat")
            _, traces = adapter.trace(
                {"constant": torch.zeros(graph.num_nodes, dtype=torch.long)},
                graph.edge_index,
                graph.edge_type,
            )
            probe = GraphMaskProbe.from_traces(traces)
            probe.enable_layer(0)
            torch.save(
                probe_payload(
                    probe,
                    data_path,
                    checkpoint_path,
                    checkpoint,
                    fanouts=[-1],
                    seed=42,
                    validation={},
                    training_config={},
                ),
                probe_path,
            )

            with patch.object(sys, "argv", [
                "graphmask-occupation-pair-report",
                "--data", str(data_path),
                "--checkpoint", str(checkpoint_path),
                "--probe", str(probe_path),
                "--output-dir", str(output_dir),
                "--split", "test",
                "--num-neighbors", "auto",
                "--batch-size", "1",
                "--min-node-support", "1",
                "--bootstrap-replicates", "0",
                "--device", "cpu",
            ]):
                occupation_pair_main()

            expected = {
                "root_direct_graphmask_pairs.csv.gz",
                "occupation_pairs_all_test.csv",
                "occupation_pairs_correct_only.csv",
                "occupation_pairs_by_visibility.csv",
                "occupation_pairs_ranked.csv",
                "target_budgets.csv",
                "metrics.json",
                "manifest.json",
            }
            self.assertEqual(expected, {path.name for path in output_dir.iterdir()})
            with (output_dir / "occupation_pairs_all_test.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["relation"] for row in rows}, {"signal", "signal__rev"})
            opportunity_by_target = {}
            for row in rows:
                opportunity_by_target.setdefault(row["target_l1"], 0.0)
                opportunity_by_target[row["target_l1"]] += float(
                    row["opportunity_share_of_Ot_budget"]
                )
            self.assertTrue(all(
                abs(value - 1.0) < 1e-9 for value in opportunity_by_target.values()
            ))
            with (output_dir / "occupation_pairs_correct_only.csv").open(
                encoding="utf-8"
            ) as handle:
                correct_rows = list(csv.DictReader(handle))
            self.assertEqual({row["target_l1"] for row in correct_rows}, {"B"})
            self.assertEqual(
                sum(int(row["candidate_message_n"]) for row in rows), 3
            )
            self.assertEqual(
                sum(int(row["candidate_message_n"]) for row in correct_rows), 2
            )
            with (output_dir / "occupation_pairs_by_visibility.csv").open(
                encoding="utf-8"
            ) as handle:
                visibility_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["source_visibility"] for row in visibility_rows},
                {"hidden_validation_or_test"},
            )
            self.assertEqual(
                sum(
                    int(row["candidate_message_n"])
                    for row in visibility_rows
                    if row["cohort"] == COHORT_ALL
                ),
                sum(int(row["candidate_message_n"]) for row in rows),
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["test_nodes"], 2)
            self.assertEqual(metrics["original_correct_test_nodes"], 1)
            self.assertEqual(
                metrics["layers"][0]["known_source_direct_candidate_messages"], 3
            )
            self.assertEqual(
                metrics["layers"][0]["unknown_source_direct_candidate_messages"], 1
            )
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(
                manifest["message_scope"],
                "typed_messages_with_target_equal_to_prediction_root",
            )


if __name__ == "__main__":
    unittest.main()
