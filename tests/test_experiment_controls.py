"""Small regression tests for server-side experiment controls.

These cover only synthetic tensors; they do not prepare the 88MB export or
start a GNN training job.
"""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from models import build_model
from models.features import NodeFeatureEncoder, build_feature_specs
from training.diagnose import degree_arrays, relation_homophily, visible_occupation_coverage
from training.attention_report import fanouts_for_checkpoint, parse_matrix_relations, prediction_nodes
from training.relation_controls import (
    apply_relation_controls,
    available_relation_groups,
    count_relation_pairs,
    relation_pair_keys,
    resolve_ablation,
)
from training.train import (
    class_balanced_train_nodes,
    classification_loss,
    loss_components,
    select_feature_schema,
    train_full_graph_epoch,
)


class TinyFullGraphModel(nn.Module):
    """A minimal model that verifies the full-graph control flow, not GNN quality."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, features, edge_index, edge_type):
        value = features["structural_constant"].float().unsqueeze(-1)
        return self.classifier(value)


class ExperimentControlTests(unittest.TestCase):
    def test_full_attention_fanouts_follow_each_checkpoint_depth(self):
        missing_checkpoint = Path("/tmp/no-such-rgat-checkpoint.pt")
        self.assertEqual(fanouts_for_checkpoint(missing_checkpoint, "full", 1), [-1])
        self.assertEqual(fanouts_for_checkpoint(missing_checkpoint, "full", 2), [-1, -1])

    def test_l1_matrix_relations_preserve_generated_reverse_direction(self):
        relation_to_id = {"father": 0, "father__rev": 1, "child": 2, "child__rev": 3}
        selected = parse_matrix_relations("father,father__rev", relation_to_id)
        self.assertEqual(selected, {0: "father", 1: "father__rev"})
        self.assertEqual(parse_matrix_relations("all", relation_to_id), {
            0: "father", 1: "father__rev", 2: "child", 3: "child__rev",
        })
        graph = Data(y=torch.tensor([0, -1, 1]), num_nodes=3)
        self.assertEqual(prediction_nodes(graph, "labeled").tolist(), [True, False, True])

    def test_structural_mode_is_one_shared_non_person_specific_feature(self):
        schema = select_feature_schema({}, (), (), "categorical", feature_mode="structural")
        specs = build_feature_specs(schema, {})
        encoder = NodeFeatureEncoder(specs, branch_dim=4, output_dim=6, dropout=0.0)
        output, gates = encoder({"structural_constant": torch.zeros(3, dtype=torch.long)})
        self.assertEqual(list(schema), ["structural_constant"])
        self.assertTrue(torch.allclose(output[0], output[1]))
        self.assertTrue(torch.allclose(output[1], output[2]))
        self.assertTrue(torch.allclose(gates["structural_constant"], torch.ones(3)))

    def test_all_node_classifiers_accept_the_structural_baseline(self):
        schema = {"structural_constant": {"kind": "constant"}}
        specs = build_feature_specs(schema, {})
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
        edge_type = torch.tensor([0, 1, 0, 1])
        features = {"structural_constant": torch.zeros(3, dtype=torch.long)}
        for name in ("rgcn", "rgat", "compgcn"):
            model = build_model(
                name,
                num_relations=2,
                num_classes=2,
                feature_specs=specs,
                hidden_dim=8,
                branch_dim=4,
                heads=2,
                num_bases=2,
            )
            logits = model(features, edge_index, edge_type)
            self.assertEqual(tuple(logits.shape), (3, 2))

    def test_relation_group_ablation_removes_both_directions(self):
        relation_to_id = {
            "father": 0,
            "father__rev": 1,
            "student_of": 2,
            "student_of__rev": 3,
            "patient_of": 4,
            "patient_of__rev": 5,
        }
        self.assertEqual(
            available_relation_groups(relation_to_id),
            ("education_mentorship", "kinship", "other"),
        )
        ids, base_relations = resolve_ablation(relation_to_id, ("kinship",), ())
        self.assertEqual(ids, {0, 1})
        self.assertEqual(base_relations, ("father",))

        graph = Data(
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            edge_type=torch.tensor([0, 1, 2, 3]),
            num_nodes=4,
        )
        manifest = apply_relation_controls(graph, ids)
        self.assertEqual(graph.edge_type.tolist(), [2, 3])
        self.assertEqual(manifest["edge_count_before"], 4)
        self.assertEqual(manifest["edge_count_after_ablation"], 2)

    def test_relation_shuffle_is_seeded_and_preserves_type_frequency(self):
        original = Data(
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
            edge_type=torch.tensor([0, 0, 1, 2]),
            num_nodes=4,
        )
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        manifest = apply_relation_controls(first, shuffle_relation_types=True, shuffle_seed=42)
        apply_relation_controls(second, shuffle_relation_types=True, shuffle_seed=42)
        self.assertTrue(torch.equal(first.edge_index, original.edge_index))
        self.assertEqual(sorted(first.edge_type.tolist()), sorted(original.edge_type.tolist()))
        self.assertTrue(torch.equal(first.edge_type, second.edge_type))
        self.assertTrue(manifest["relation_type_shuffle"])

    def test_random_pair_deletion_keeps_forward_reverse_edges_together(self):
        relation_to_id = {
            "father": 0,
            "father__rev": 1,
            "student_of": 2,
            "student_of__rev": 3,
        }
        graph = Data(
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_type=torch.tensor([0, 1, 2, 3]),
            num_nodes=3,
        )
        father_ids, _ = resolve_ablation(relation_to_id, ("kinship",), ())
        self.assertEqual(count_relation_pairs(graph, father_ids, relation_to_id), 1)
        original_keys = relation_pair_keys(graph, relation_to_id)
        manifest = apply_relation_controls(
            graph,
            relation_to_id=relation_to_id,
            random_edge_drop_pairs=1,
            random_edge_drop_seed=42,
        )
        retained_keys = relation_pair_keys(graph, relation_to_id)
        for key in np.unique(original_keys):
            original_count = int((original_keys == key).sum())
            retained_count = int((retained_keys == key).sum())
            self.assertIn(retained_count, {0, original_count})
        self.assertEqual(manifest["random_edge_drop_pairs"], 1)
        self.assertEqual(graph.edge_index.size(1), 2)

    def test_diagnostics_measure_neighbours_coverage_and_relation_homophily(self):
        relation_to_id = {"father": 0, "father__rev": 1, "student_of": 2, "student_of__rev": 3}
        graph = Data(
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_type=torch.tensor([0, 1, 2, 3]),
            y=torch.tensor([0, 0, 1]),
            num_nodes=3,
        )
        _, _, neighbour_degree = degree_arrays(graph.edge_index, graph.num_nodes)
        self.assertEqual(neighbour_degree.tolist(), [1, 2, 1])
        direct, within_two_hops = visible_occupation_coverage(
            graph.edge_index, np.array([True, False, False]), graph.num_nodes
        )
        self.assertEqual(direct.tolist(), [False, 1, False])
        self.assertTrue(within_two_hops[2])
        report = relation_homophily(
            graph, relation_to_id, np.array([True, True, True]), min_support=1
        )
        father = report.loc[report["relation"] == "father"].iloc[0]
        student = report.loc[report["relation"] == "student_of"].iloc[0]
        self.assertEqual(father["same_label_rate"], 1.0)
        self.assertEqual(student["same_label_rate"], 0.0)

    def test_long_tail_losses_and_root_sampling_use_train_labels_only(self):
        counts = torch.tensor([8.0, 2.0])
        inverse_weights, log_priors = loss_components(counts, "inverse_frequency", 0.9999, torch.device("cpu"))
        self.assertTrue(torch.allclose(inverse_weights, torch.tensor([0.625, 2.5])))
        self.assertTrue(torch.allclose(log_priors.exp(), torch.tensor([0.8, 0.2])))
        balanced_weights, _ = loss_components(counts, "class_balanced", 0.9, torch.device("cpu"))
        self.assertGreater(float(balanced_weights[1]), float(balanced_weights[0]))
        loss = classification_loss(
            torch.tensor([[2.0, 1.0], [0.5, 1.5]]),
            torch.tensor([0, 1]),
            "logit_adjusted",
            log_priors=log_priors,
        )
        self.assertTrue(torch.isfinite(loss))

        graph = Data(
            y=torch.tensor([0] * 8 + [1] * 2),
            train_mask=torch.ones(10, dtype=torch.bool),
            num_nodes=10,
        )
        roots = class_balanced_train_nodes(graph, 2, torch.Generator().manual_seed(7))
        sampled_labels = graph.y[roots]
        self.assertGreaterEqual(int((sampled_labels == 0).sum()), 3)
        self.assertGreaterEqual(int((sampled_labels == 1).sum()), 3)

    def test_full_graph_epoch_accepts_only_the_shared_structural_input(self):
        graph = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_type=torch.tensor([0, 0]),
            y=torch.tensor([0, 1, -1]),
            train_mask=torch.tensor([True, True, False]),
            num_nodes=3,
        )
        model = TinyFullGraphModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = train_full_graph_epoch(
            model,
            graph,
            optimizer,
            {"structural_constant": {"kind": "constant"}},
            "cross_entropy",
        )
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
