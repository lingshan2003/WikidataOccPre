"""Tests for sparse root-cluster bootstrap reconstruction."""

import unittest

from training.attention_bootstrap import group_bootstrap


class AttentionBootstrapTests(unittest.TestCase):
    def test_sparse_absence_is_reconstructed_as_zero_for_root_mean(self):
        shared = {
            "experiment": "condition",
            "seed": "42",
            "checkpoint": "checkpoint.pt",
            "split": "test",
            "forward_mode": "full-graph",
            "num_layers": "1",
            "fanouts": "full-graph",
            "message_passing_layer": "1",
            "target_l1_id": "0",
            "target_l1": "Culture",
        }
        roster = [{**shared, "root_index": "10"}, {**shared, "root_index": "11"}]
        sparse = [{
            **shared,
            "root_index": "10",
            "relation_id": "3",
            "relation": "father",
            "attention_mass": "0.4",
        }]
        rows = group_bootstrap(
            roster,
            sparse,
            list(roster[0]),
            list(sparse[0]),
            ["attention_mass"],
            ["relation_id", "relation"],
            resamples=200,
            seed=7,
            batch_resamples=20,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["root_count"], 2)
        self.assertEqual(rows[0]["roots_with_nonzero_value"], 1)
        self.assertAlmostEqual(rows[0]["mean"], 0.2)

    def test_sparse_values_for_same_root_group_are_added_before_resampling(self):
        shared = {
            "experiment": "condition",
            "seed": "42",
            "checkpoint": "checkpoint.pt",
            "split": "test",
            "forward_mode": "full-graph",
            "num_layers": "1",
            "fanouts": "full-graph",
            "message_passing_layer": "1",
            "target_l1_id": "0",
            "target_l1": "Culture",
            "root_index": "10",
            "relation_id": "3",
            "relation": "father",
        }
        roster = [{key: value for key, value in shared.items() if key not in {"relation_id", "relation"}}]
        sparse = [{**shared, "attention_mass": "0.1"}, {**shared, "attention_mass": "0.3"}]
        rows = group_bootstrap(
            roster,
            sparse,
            list(roster[0]),
            list(sparse[0]),
            ["attention_mass"],
            ["relation_id", "relation"],
            resamples=20,
            seed=3,
            batch_resamples=5,
        )
        self.assertAlmostEqual(rows[0]["mean"], 0.4)


if __name__ == "__main__":
    unittest.main()
