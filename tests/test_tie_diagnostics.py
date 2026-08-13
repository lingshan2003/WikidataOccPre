"""Synthetic coverage and homophily checks for inherited/acquired diagnostics."""

import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from training.diagnose import relation_homophily, tie_group_coverage, tie_group_homophily
from training.tie_taxonomy import TieTaxonomy


class TieDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.relation_to_id = {
            "father": 0,
            "father__rev": 1,
            "student_of": 2,
            "student_of__rev": 3,
        }
        self.taxonomy = TieTaxonomy(
            name="synthetic",
            version=1,
            path=Path("synthetic.json"),
            sha256="synthetic",
            inherited=("father",),
            acquired=("student_of",),
        )
        self.data = Data(
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_type=torch.tensor([0, 1, 2, 3]),
            y=torch.tensor([0, 0, 1]),
            num_nodes=3,
        )

    def test_coverage_keeps_directed_message_counts_and_visible_exposure(self):
        columns, summary = tie_group_coverage(
            self.data, self.relation_to_id, self.taxonomy, np.array([True, True, False])
        )
        inherited = summary.loc[summary["tie_group"] == "inherited"].iloc[0]
        acquired = summary.loc[summary["tie_group"] == "acquired"].iloc[0]
        self.assertEqual(int(inherited["directed_edges"]), 2)
        self.assertEqual(int(inherited["relation_pairs"]), 1)
        self.assertEqual(int(acquired["directed_edges"]), 2)
        self.assertEqual(int(acquired["relation_pairs"]), 1)
        self.assertEqual(columns["inherited_visible_train_occupation_messages"].tolist(), [1, 1, 0])
        self.assertEqual(columns["acquired_visible_train_occupation_messages"].tolist(), [0, 0, 1])
        self.assertEqual(columns["tie_exposure"].tolist(), ["inherited_only", "inherited_only", "acquired_only"])

    def test_relation_and_group_homophily_are_labeled_by_the_same_taxonomy(self):
        train_labels = np.array([True, True, False])
        by_relation = relation_homophily(
            self.data, self.relation_to_id, train_labels, min_support=1, tie_taxonomy=self.taxonomy
        )
        self.assertEqual(by_relation.loc[0, "tie_group"], "inherited")
        by_group = tie_group_homophily(
            self.data, self.relation_to_id, self.taxonomy, train_labels, min_support=1
        )
        inherited = by_group.loc[by_group["tie_group"] == "inherited"].iloc[0]
        acquired = by_group.loc[by_group["tie_group"] == "acquired"].iloc[0]
        self.assertEqual(int(inherited["labeled_train_pairs"]), 1)
        self.assertEqual(float(inherited["same_label_rate"]), 1.0)
        self.assertEqual(int(acquired["labeled_train_pairs"]), 0)
        self.assertTrue(np.isnan(acquired["same_label_rate"]))


if __name__ == "__main__":
    unittest.main()
