import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.l1_only_sliding import parse_args, make_pipeline, explicit_pool, child_command


class L1OnlyProtocolTests(unittest.TestCase):
    def test_three_effective_train_commands_use_only_l1_and_new_roots(self):
        roots = []
        with patch.dict(os.environ, {}, clear=True):
            for dataset in ("wikidata", "freebase", "dbpedia"):
                args = parse_args([dataset, "plan", "all", "--periods", "center_1900,center_1950,center_2000"])
                pipeline = make_pipeline(args)
                train = pipeline.commands("center_1950", "multi_group")[1]
                self.assertEqual(train[train.index("--occupation-feature-levels") + 1], "1")
                self.assertEqual(train[train.index("--auxiliary-features") + 1], "none")
                self.assertEqual(len(pipeline.contexts), 3)
                self.assertEqual(pipeline.config["prepare"]["split_seed"], 20260814)
                roots += [pipeline.config[k] for k in ("period_root", "relation_root", "model_root", "graphmask_root")]
                old = json.loads((ROOT / "config" / (pipeline.config["comparison_to"] + ".json")).read_text())
                self.assertEqual(pipeline.config["source_prepare"], old["source_prepare"])
                self.assertFalse(set(roots) & {old[k] for k in ("period_root", "relation_root", "model_root", "graphmask_root")})
                if dataset == "dbpedia":
                    merged = pipeline.multi["groups"]["influence_succession"]
                    self.assertEqual(set(merged), {"influenced", "influencedBy", "predecessor", "successor"})
        self.assertEqual(len(roots), len(set(roots)))

    def test_stale_budget_path_and_period_overrides_are_rejected(self):
        for variable, value in [("WIKIDATA_GROUP_TRAIN_EPOCHS", "1"), ("WIKIDATA_GROUP_GRAPHMASK_ROOT", "runs_graphmask/old"), ("WIKIDATA_GROUP_PERIODS", "center_2000")]:
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: value}, clear=True):
                with self.assertRaisesRegex(ValueError, "Environment changed"):
                    make_pipeline(parse_args(["wikidata", "plan"]))

    def test_gpu_count_selects_only_from_explicit_pool_and_normalizes_child(self):
        with patch.dict(os.environ, {}, clear=True):
            args = parse_args(["wikidata", "plan", "--gpus", "2,3,5", "--num-gpus", "2"])
            pipeline = make_pipeline(args)
            self.assertEqual(explicit_pool(args), ["2", "3"])
            args.gpus, args.num_gpus = ",".join(explicit_pool(args)), None
            cmd = child_command(args)
            self.assertEqual(cmd[cmd.index("--gpus") + 1], "2,3")
            self.assertNotIn("--num-gpus", cmd)
            with self.assertRaisesRegex(ValueError, "automatic card enumeration is disabled"):
                explicit_pool(parse_args(["wikidata", "plan", "--num-gpus", "2"]))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=True):
            self.assertEqual(explicit_pool(parse_args(["freebase", "plan", "--num-gpus", "1"])), ["4"])

    def test_alive_boundary_uses_strictly_after_1920(self):
        # Exercise the actual production membership function, no graph/model.
        import pandas as pd
        from training.life_periods import load_life_period_config, life_period_membership
        config = load_life_period_config(ROOT / "config/l1_only_life_windows_1900_2000_pm20_step1_alive2026_v1.json")
        nodes = pd.DataFrame({"node_id": ["born1920", "born1921"], "birth_year": [1920, 1921], "death_year": [None, None]})
        membership, _ = life_period_membership(nodes, config)
        self.assertEqual(list(membership["center_2000"]), [False, True])


if __name__ == "__main__":
    unittest.main()
