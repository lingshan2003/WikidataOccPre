"""Contract tests for generic taxonomy-based relation collapse."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from data.collapse_relations import collapse_relation_artifact
from training.tie_taxonomy import load_tie_taxonomy


class CollapseRelationsTest(unittest.TestCase):
    def test_assigns_parent_names_preserves_direction_and_writes_train_taxonomy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "collapsed"
            source.mkdir()
            relation_to_id = {
                "father": 0,
                "father__rev": 1,
                "mother": 2,
                "mother__rev": 3,
                "spouse": 4,
                "spouse__rev": 5,
                "student_of": 6,
                "student_of__rev": 7,
                "employer": 8,
                "employer__rev": 9,
            }
            edge_index = torch.tensor([
                [0, 1, 0, 1, 0, 2, 0, 3, 0, 4],
                [1, 0, 1, 0, 2, 0, 3, 0, 4, 0],
            ])
            edge_type = torch.arange(10)
            data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=5)
            torch.save({
                "data": data,
                "metadata": {
                    "target_column": "occupation_level2",
                    "relation_to_id": relation_to_id,
                    "num_relations": 10,
                },
            }, source / "graph_data.pt")
            pd.DataFrame({
                "source": [f"q{i}" for i in edge_index[0].tolist()],
                "relation": list(relation_to_id),
                "target": [f"q{i}" for i in edge_index[1].tolist()],
                "source_id": edge_index[0].tolist(),
                "target_id": edge_index[1].tolist(),
                "relation_id": edge_type.tolist(),
            }).to_csv(source / "edges.csv", index=False)
            taxonomy = root / "taxonomy.json"
            taxonomy.write_text(json.dumps({
                "name": "test_subgroups",
                "version": 1,
                "groups": {
                    "inherited": ["father", "mother"],
                    "intimate_partnership": ["spouse"],
                    "education_mentorship": ["student_of"],
                    "other_acquired": "all_remaining",
                },
            }), encoding="utf-8")

            manifest = collapse_relation_artifact(
                source / "graph_data.pt", taxonomy, output
            )
            bundle = torch.load(output / "graph_data.pt", weights_only=False)
            relation_to_id = bundle["metadata"]["relation_to_id"]

            self.assertEqual(manifest["duplicates_removed"], 2)
            self.assertEqual(bundle["metadata"]["num_relations"], 8)
            self.assertEqual(relation_to_id["inherited"], 0)
            self.assertEqual(relation_to_id["inherited__rev"], 1)
            self.assertEqual(len(pd.read_csv(output / "edges.csv")), 8)
            self.assertEqual(
                set(pd.read_csv(output / "edges.csv")["relation"]),
                set(relation_to_id),
            )
            collapsed_ties = json.loads(
                (output / "collapsed_tie_taxonomy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(collapsed_ties["groups"]["inherited"], ["inherited"])
            self.assertEqual(
                set(collapsed_ties["groups"]["acquired"]),
                {
                    "education_mentorship",
                    "intimate_partnership",
                    "other_acquired",
                },
            )
            resolved_ties = load_tie_taxonomy(
                output / "collapsed_tie_taxonomy.json", relation_to_id
            )
            self.assertEqual(resolved_ties.inherited, ("inherited",))


if __name__ == "__main__":
    unittest.main()
