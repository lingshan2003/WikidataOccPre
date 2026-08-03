"""Regression tests for fixed occupation-semantic features.

These tests intentionally use a tiny synthetic graph: no Hugging Face model
download is needed to verify the masking and frozen-table protocol.
"""

import unittest

import torch
from torch_geometric.data import Data

from data.occupation_semantics import FEATURE_NAME, build_semantic_ids
from models.features import NodeFeatureEncoder, build_feature_specs
from training.train import (
    batch_features,
    select_feature_schema,
    select_unknown_feature_ids,
    semantic_provenance,
)


def toy_metadata():
    unknowns = {f"occupation_level{level}": 0 for level in (1, 2, 3)}
    return {
        "occupation_unknown_ids": unknowns,
        "occupation_vocabularies": {
            "occupation_level1": {"__UNKNOWN__": 0, "arts": 1},
            "occupation_level2": {"__UNKNOWN__": 0, "music": 1},
            "occupation_level3": {"__UNKNOWN__": 0, "violinist": 1, "pianist": 2, "Missing": 3},
        },
        "feature_schema": {
            **{
                f"occupation_level{level}": {
                    "kind": "categorical",
                    "cardinality": 2 if level < 3 else 4,
                    "input_dim": 1,
                }
                for level in (1, 2, 3)
            },
            "country": {"kind": "categorical", "cardinality": 2, "input_dim": 1},
            "temporal": {"kind": "numeric", "input_dim": 1},
        },
    }


def toy_data():
    return Data(
        occupation_level1=torch.tensor([1, 1, 1, 0, 1]),
        occupation_level2=torch.tensor([1, 1, 1, 0, 1]),
        occupation_level3=torch.tensor([1, 2, 1, 0, 3]),
        country=torch.zeros(5, dtype=torch.long),
        temporal=torch.zeros(5, 1),
        train_mask=torch.tensor([True, True, False, False, True]),
        val_mask=torch.tensor([False, False, True, False, False]),
        test_mask=torch.tensor([False, False, False, True, False]),
    )


class OccupationSemanticTests(unittest.TestCase):
    def test_only_visible_training_tuples_get_nonzero_ids_and_prompts(self):
        semantic_ids, entries = build_semantic_ids(toy_data(), toy_metadata())
        self.assertTrue(torch.equal(semantic_ids, torch.tensor([1, 2, 0, 0, 0])))
        self.assertEqual(len(entries), 2)
        self.assertTrue(all("missing" not in entry["prompt"].casefold() for entry in entries))
        self.assertTrue(all(entry["semantic_id"] == index for index, entry in enumerate(entries, start=1)))
        self.assertLessEqual(int(semantic_ids.max()), len(entries))

    def test_seed_mask_and_semantic_schema(self):
        metadata = toy_metadata()
        metadata["feature_schema"][FEATURE_NAME] = {
            "kind": "semantic_categorical",
            "cardinality": 3,
            "input_dim": 4,
            "unknown_id": 0,
            "semantic_table_key": FEATURE_NAME,
        }
        metadata["occupation_unknown_ids"][FEATURE_NAME] = 0
        metadata["semantic_feature_tables"] = {FEATURE_NAME: torch.randn(3, 4)}
        data = toy_data()
        data.occupation_semantic = torch.tensor([1, 2, 0, 0, 0])
        schema = select_feature_schema(metadata, (1, 2, 3), (), "semantic")
        self.assertEqual(list(schema), [FEATURE_NAME])
        categorical_schema = select_feature_schema(metadata, (1, 2, 3), (), "categorical")
        self.assertEqual(
            list(categorical_schema), ["occupation_level1", "occupation_level2", "occupation_level3"]
        )
        unknowns = select_unknown_feature_ids(metadata, (1, 2, 3), "semantic")
        masked = batch_features(type("Batch", (), {
            "occupation_semantic": data.occupation_semantic,
            "batch_size": 2,
        })(), schema, unknowns)
        self.assertTrue(torch.equal(masked[FEATURE_NAME], torch.tensor([0, 0, 0, 0, 0])))

    def test_table_is_frozen_but_unknown_and_projection_train(self):
        metadata = toy_metadata()
        metadata["feature_schema"] = {
            FEATURE_NAME: {
                "kind": "semantic_categorical",
                "cardinality": 3,
                "input_dim": 4,
                "unknown_id": 0,
                "semantic_table_key": FEATURE_NAME,
            }
        }
        metadata["semantic_feature_tables"] = {FEATURE_NAME: torch.randn(3, 4)}
        encoder = NodeFeatureEncoder(build_feature_specs(metadata["feature_schema"], metadata), 3, 5, 0.0)
        output, _ = encoder({FEATURE_NAME: torch.tensor([0, 1, 2])})
        output.sum().backward()
        self.assertIsNone(getattr(encoder, f"{FEATURE_NAME}_semantic_embeddings").grad)
        self.assertIsNotNone(encoder.missing_vectors[FEATURE_NAME].grad)
        self.assertIsNotNone(encoder.encoders[FEATURE_NAME][0].weight.grad)

    def test_semantic_provenance_is_explicit(self):
        metadata = toy_metadata()
        metadata["semantic_features"] = {
            FEATURE_NAME: {
                "model_name": "intfloat/multilingual-e5-base",
                "resolved_revision": "abc123",
                "prompt_fingerprint": "fingerprint",
                "source_artifact": "source.pt",
            }
        }
        self.assertEqual(semantic_provenance(metadata, "semantic")["semantic_prompt_fingerprint"], "fingerprint")


if __name__ == "__main__":
    unittest.main()
