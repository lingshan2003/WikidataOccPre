"""Contract tests for the inherited/acquired relation-collapse artifact."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from data.collapse_ties import collapse_tie_artifact


class CollapseTiesTest(unittest.TestCase):
    def test_preserves_direction_and_deduplicates_collapsed_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "binary"
            source.mkdir()
            relation_to_id = {
                "father": 0,
                "father__rev": 1,
                "mother": 2,
                "mother__rev": 3,
                "spouse": 4,
                "spouse__rev": 5,
            }
            edge_index = torch.tensor([
                [0, 1, 0, 1, 0, 2],
                [1, 0, 1, 0, 2, 0],
            ])
            edge_type = torch.tensor([0, 1, 2, 3, 4, 5])
            data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=3)
            torch.save({
                "data": data,
                "metadata": {
                    "target_column": "occupation_level2",
                    "relation_to_id": relation_to_id,
                    "num_relations": 6,
                },
            }, source / "graph_data.pt")
            pd.DataFrame({
                "source": ["a", "b", "a", "b", "a", "c"],
                "relation": list(relation_to_id),
                "target": ["b", "a", "b", "a", "c", "a"],
                "source_id": edge_index[0].tolist(),
                "target_id": edge_index[1].tolist(),
                "relation_id": edge_type.tolist(),
            }).to_csv(source / "edges.csv", index=False)
            pd.DataFrame({"node_id": ["a", "b", "c"]}).to_csv(
                source / "nodes.csv", index=False
            )
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(json.dumps({
                "name": "test",
                "version": 1,
                "groups": {
                    "inherited": ["father", "mother"],
                    "acquired": "all_remaining",
                },
            }), encoding="utf-8")

            manifest = collapse_tie_artifact(source / "graph_data.pt", taxonomy, output)
            bundle = torch.load(output / "graph_data.pt", weights_only=False)

            self.assertEqual(manifest["duplicates_removed"], 2)
            self.assertEqual(bundle["data"].edge_type.tolist(), [0, 1, 2, 3])
            self.assertEqual(bundle["metadata"]["num_relations"], 4)
            self.assertEqual(len(pd.read_csv(output / "edges.csv")), 4)
            binary_taxonomy = json.loads(
                (output / "binary_tie_taxonomy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(binary_taxonomy["groups"]["inherited"], ["inherited_ties"])


if __name__ == "__main__":
    unittest.main()
