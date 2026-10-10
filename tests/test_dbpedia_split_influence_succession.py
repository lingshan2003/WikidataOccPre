"""Separate influence/succession facts and isolate the eight-group rerun."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import torch
from torch_geometric.data import Data

from data.collapse_relations import collapse_relation_artifact

ROOT = Path(__file__).resolve().parents[1]
OLD_TAXONOMY = ROOT / "config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json"
NEW_TAXONOMY = ROOT / "config/dbpedia_tie_taxonomy_acquired_subgroups_v2.json"
NEW_CONFIG = ROOT / "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json"


class SplitInfluenceSuccessionTests(unittest.TestCase):
    def test_same_person_pair_keeps_distinct_influence_and_succession_messages(self):
        old = json.loads(OLD_TAXONOMY.read_text())
        new = json.loads(NEW_TAXONOMY.read_text())
        members = [relation for group in new["groups"].values() for relation in group]
        self.assertEqual(len(members), 33)
        self.assertEqual(len(set(members)), 33)
        self.assertEqual(set(members), {r for group in old["groups"].values() for r in group})
        self.assertEqual(new["groups"]["influence"], ["influenced", "influencedBy"])
        self.assertEqual(new["groups"]["succession"], ["successor", "predecessor"])
        for group in set(old["groups"]) - {"influence_succession"}:
            self.assertEqual(new["groups"][group], old["groups"][group])
        vocabulary = [name for relation in members for name in (relation, relation + "__rev")]
        mapping = {relation: i for i, relation in enumerate(vocabulary)}
        relations = ["influenced", "influenced__rev", "influencedBy", "influencedBy__rev", "successor", "successor__rev"]
        edge_index = torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])
        edge_type = torch.tensor([mapping[r] for r in relations])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            data = Data(num_nodes=2, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([0, 1]))
            torch.save({"data": data, "metadata": {"target_column": "occupation_level1", "relation_to_id": mapping, "num_relations": 66}}, source / "graph_data.pt")
            pd.DataFrame({"source": ["a", "b"] * 3, "target": ["b", "a"] * 3,
                          "source_id": edge_index[0].tolist(), "target_id": edge_index[1].tolist(),
                          "relation": relations, "relation_id": edge_type.tolist()}).to_csv(source / "edges.csv", index=False)
            old_manifest = collapse_relation_artifact(source / "graph_data.pt", OLD_TAXONOMY, root / "old")
            new_manifest = collapse_relation_artifact(source / "graph_data.pt", NEW_TAXONOMY, root / "new")
            self.assertEqual(old_manifest["edges_after"], 2)
            self.assertEqual(new_manifest["edges_after"], 4)
            edges = pd.read_csv(root / "new/edges.csv")
            self.assertEqual(set(edges.loc[edges.source_id == 0, "relation"]), {"influence", "succession"})
            self.assertEqual(set(edges.loc[edges.source_id == 1, "relation"]), {"influence__rev", "succession__rev"})
            bundle = torch.load(root / "new/graph_data.pt", weights_only=False)
            self.assertEqual(bundle["metadata"]["num_relations"], 16)
            self.assertTrue(torch.equal(bundle["data"].y, data.y))

    def test_wrapper_pins_new_roots_and_all_101_jobs_despite_stale_environment(self):
        old_config = json.loads((ROOT / "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json").read_text())
        new_config = json.loads(NEW_CONFIG.read_text())
        for key in ("source_data", "prepare", "train", "graphmask_train", "graphmask_report"):
            self.assertEqual(new_config[key], old_config[key])
        for key in ("period_config", "period_root"):
            self.assertNotEqual(new_config[key], old_config[key])
        window_config = json.loads((ROOT / new_config["period_config"]).read_text())
        self.assertEqual(window_config["birth_only_alive_assumption"], {"born_after": 1920, "alive_through": 2026})
        environment = {"DBPEDIA_PYTHON_BIN": sys.executable, "DBPEDIA_GROUP_INCLUDE_FULL": "1",
                       "DBPEDIA_GROUP_PERIODS": "center_1900", "DBPEDIA_GROUP_REPRESENTATIONS": "binary"}
        for key in ("period_root", "period_config", "relation_root", "model_root", "graphmask_root", "multi_group_taxonomy"):
            environment["DBPEDIA_GROUP_" + key.upper()] = old_config[key]
        with mock.patch.dict(os.environ, environment, clear=True):
            result = subprocess.run(["bash", "DBpedia/run_sliding_split_influence_succession.sh", "plan", "all", "--gpus", "0,1"],
                                    cwd=ROOT, text=True, capture_output=True, check=True)
        plan = result.stdout
        for stage in ("collapse", "train", "probe", "report"):
            self.assertEqual(plan.count("[" + stage + "]"), 101)
        self.assertEqual(plan.count("--num-bases 16"), 101)
        self.assertEqual(plan.count("--checkpoint-selection any-stage"), 101)
        self.assertNotIn("/binary/", plan)
        self.assertNotIn("[train] full/", plan)
        for key in ("period_root", "relation_root", "model_root", "graphmask_root"):
            self.assertNotIn(str(ROOT / old_config[key]) + "/", plan)
            self.assertIn(str(ROOT / new_config[key]) + "/", plan)
        self.assertIn(str(ROOT / new_config["multi_group_taxonomy"]), plan)


if __name__ == "__main__":
    unittest.main()
