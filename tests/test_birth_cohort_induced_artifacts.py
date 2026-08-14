"""Synthetic contracts for fresh, period-induced graph preparation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.birth_cohort_artifacts import (
    INDUCED_EDGE_POLICY,
    build_period_induced_artifact,
    prepare_period_induced_artifacts,
)
from data.prepare import build_pyg_data
from training.life_periods import load_life_period_config


class PeriodInducedArtifactTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        path = root / "periods.json"
        path.write_text(json.dumps({
            "name": "synthetic life periods",
            "version": 1,
            "membership_rule": "life_interval_overlaps_period",
            "birth_field": "birth_year",
            "death_field": "death_year",
            "missing_date_policy": "exclude_if_birth_or_death_missing",
            "invalid_interval_policy": "exclude_if_death_before_birth",
            "periods": [
                {"id": "early", "label": "Early", "end": 500},
                {"id": "late", "label": "Late", "start": 501},
            ],
        }), encoding="utf-8")
        return path

    def _write_source_artifact(self, root: Path) -> Path:
        source_dir = root / "source"
        source_dir.mkdir()
        nodes = pd.DataFrame({
            "node_id": [f"Q{index}" for index in range(6)],
            "birth_year": [100, 200, 300, 400, 550, 700],
            "death_year": [600, 300, 400, 600, 650, 800],
            "occupation_level1": ["A", "A", "A", "B", "B", "B"],
            "occupation_level2": ["A2", "A2", "A2", "B2", "B2", "B2"],
            "occupation_level3": ["A3", "A3", "A3", "B3", "B3", "B3"],
            "country": ["X", "X", "X", "Y", "Y", "Y"],
        })
        # Q0 and Q3 live across the 500/501 boundary, so they must appear in
        # both artifacts. Q1 and Q4 never overlap in time, so their edge pair
        # must not survive either induced graph.
        edges = pd.DataFrame([
            ("Q0", "father", "Q1", 0, 1, 0),
            ("Q1", "father__rev", "Q0", 1, 0, 1),
            ("Q1", "spouse", "Q4", 1, 4, 2),
            ("Q4", "spouse__rev", "Q1", 4, 1, 3),
            ("Q2", "spouse", "Q3", 2, 3, 2),
            ("Q3", "spouse__rev", "Q2", 3, 2, 3),
            ("Q3", "father", "Q4", 3, 4, 0),
            ("Q4", "father__rev", "Q3", 4, 3, 1),
            ("Q4", "spouse", "Q5", 4, 5, 2),
            ("Q5", "spouse__rev", "Q4", 5, 4, 3),
        ], columns=["source", "relation", "target", "source_id", "target_id", "relation_id"])
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        source_train = np.asarray([True, False, False, True, False, False])
        source_val = np.asarray([False, True, False, False, True, False])
        source_test = np.asarray([False, False, True, False, False, True])
        data, country_to_id, occupation_schema, unknown_ids, vocabularies = build_pyg_data(
            nodes, edges, labels, source_train, source_val, source_test
        )
        metadata = {
            "target_column": "occupation_level1",
            "num_relations": 4,
            "num_classes": 2,
            "feature_schema": {
                **occupation_schema,
                "country": {"kind": "categorical", "cardinality": len(country_to_id)},
                "temporal": {"kind": "numeric", "input_dim": int(data.temporal.size(1))},
            },
            "country_to_id": country_to_id,
            "label_to_id": {"A": 0, "B": 1},
            "occupation_unknown_ids": unknown_ids,
            "occupation_vocabularies": vocabularies,
            "relation_to_id": {"father": 0, "father__rev": 1, "spouse": 2, "spouse__rev": 3},
            "seed": 42,
        }
        torch.save({"data": data, "metadata": metadata}, source_dir / "graph_data.pt")
        nodes.to_csv(source_dir / "nodes.csv", index=False)
        edges.to_csv(source_dir / "edges.csv", index=False)
        # This is how the real preparation pipeline represents no detected
        # attribute conflicts: an existing but zero-byte CSV.
        (source_dir / "attribute_conflicts.csv").touch()
        return source_dir / "graph_data.pt"

    def test_induced_artifact_remaps_nodes_removes_cross_period_edges_and_resplits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data = self._write_source_artifact(root)
            config = load_life_period_config(self._write_config(root))
            report = build_period_induced_artifact(source_data, root / "period_artifacts", config, "early", split_seed=9)
            artifact_dir = root / "period_artifacts" / "early"
            bundle = torch.load(artifact_dir / "graph_data.pt", map_location="cpu", weights_only=False)
            data, metadata = bundle["data"], bundle["metadata"]
            nodes = pd.read_csv(artifact_dir / "nodes.csv")
            edges = pd.read_csv(artifact_dir / "edges.csv")

            self.assertEqual(report["nodes"], 4)
            self.assertEqual(report["directed_edges"], 4)
            self.assertEqual(nodes["source_node_index"].tolist(), [0, 1, 2, 3])
            self.assertEqual(nodes["life_period_membership_count"].tolist(), [2, 1, 1, 2])
            self.assertEqual(edges[["source_id", "target_id"]].values.tolist(), [[0, 1], [1, 0], [2, 3], [3, 2]])
            self.assertTrue((edges[["source_id", "target_id"]].to_numpy() < 4).all())
            self.assertEqual(data.edge_index.tolist(), [[0, 1, 2, 3], [1, 0, 3, 2]])
            self.assertEqual(data.edge_type.tolist(), [0, 1, 2, 3])
            self.assertEqual((data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()), (1, 1, 1))
            self.assertTrue(torch.equal(data.train_mask | data.val_mask | data.test_mask, data.y >= 0))
            self.assertEqual(metadata["label_to_id"], {"A": 0, "B": 1})
            details = metadata["period_induced_artifact"]
            self.assertEqual(details["edge_policy"], INDUCED_EDGE_POLICY)
            self.assertEqual(details["selected_life_period"]["id"], "early")
            self.assertEqual(details["period_graph"]["incident_cross_period_directed_edges"], 4)
            self.assertNotEqual(data.train_mask.tolist(), [True, False, False])

            build_period_induced_artifact(source_data, root / "period_artifacts", config, "late", split_seed=9)
            late_nodes = pd.read_csv(root / "period_artifacts" / "late" / "nodes.csv")
            self.assertEqual(late_nodes["source_node_index"].tolist(), [0, 3, 4, 5])

    def test_exact_existing_artifact_is_reused_but_unknown_period_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_data = self._write_source_artifact(root)
            config_path = self._write_config(root)
            config = load_life_period_config(config_path)
            first = build_period_induced_artifact(source_data, root / "period_artifacts", config, "late")
            second = build_period_induced_artifact(source_data, root / "period_artifacts", config, "late")
            self.assertFalse(first["reused_existing"])
            self.assertTrue(second["reused_existing"])
            with self.assertRaisesRegex(ValueError, "incompatible source/period/split provenance"):
                build_period_induced_artifact(
                    source_data, root / "period_artifacts", config, "late", split_seed=10
                )
            with self.assertRaisesRegex(ValueError, "Unknown requested life periods"):
                prepare_period_induced_artifacts(
                    source_data, root / "other_artifacts", config_path, ("not_a_period",)
                )
            self.assertFalse((root / "other_artifacts").exists())


if __name__ == "__main__":
    unittest.main()
