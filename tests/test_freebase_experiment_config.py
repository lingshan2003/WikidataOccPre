"""Contract checks for the planned Freebase grouped R-GCN/GraphMask matrix."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
BASE_RELATIONS = {
    "child",
    "sibling",
    "partner",
    "influenced_by",
    "academic_advisor_raw",
    "martial_arts_instructor_raw",
    "peer",
    "celebrity_romantic_relationship",
    "celebrity_friend",
}


def read_config(name):
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


class FreebaseExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = read_config("freebase_tie_taxonomy_v1.json")
        cls.multi = read_config("freebase_tie_taxonomy_acquired_subgroups_v1.json")
        cls.period = read_config("freebase_life_periods_20y_v1.json")
        cls.pipeline = read_config("freebase_grouped_rgcn_graphmask_20y_v1.json")

    def test_binary_and_multigroup_taxonomies_cover_nine_bases_once(self):
        binary_groups = self.binary["groups"]
        self.assertEqual(set(binary_groups), {"inherited", "acquired"})
        inherited = set(binary_groups["inherited"])
        self.assertEqual(inherited, {"child", "sibling"})
        self.assertEqual(binary_groups["acquired"], "all_remaining")

        groups = self.multi["groups"]
        members = [relation for values in groups.values() for relation in values]
        self.assertEqual(set(members), BASE_RELATIONS)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(groups["inherited"]), inherited)
        self.assertNotIn("partner", groups["inherited"])
        self.assertNotIn("celebrity_romantic_relationship", groups["inherited"])
        self.assertIn("partner", groups["intimate_partnership"])
        self.assertIn("celebrity_romantic_relationship", groups["intimate_partnership"])
        self.assertEqual(set(groups["education_mentorship"]), {
            "academic_advisor_raw", "martial_arts_instructor_raw"
        })
        self.assertEqual(groups["influence_succession"], ["influenced_by"])
        self.assertEqual(set(groups["other_acquired"]), {"peer", "celebrity_friend"})
        self.assertNotIn("professional_collaboration", groups)
        self.assertNotIn("religious_ordination", groups)

    def test_period_partition_matches_requested_inclusive_years_and_step(self):
        periods = self.period["periods"]
        self.assertEqual(len(periods), 8)
        self.assertEqual(periods[0]["end"], 1499)
        self.assertNotIn("start", periods[0])
        self.assertEqual((periods[1]["start"], periods[1]["end"]), (1500, 1900))
        modern = periods[2:]
        self.assertEqual(modern[0]["start"], 1901)
        for index, period in enumerate(modern):
            expected_start = 1901 + 20 * index
            self.assertEqual(period["start"], expected_start)
            self.assertEqual(period["end"], expected_start + 19)
            self.assertEqual(period["end"] - period["start"] + 1, 20)
            if index:
                self.assertEqual(period["start"], modern[index - 1]["end"] + 1)
        self.assertEqual((modern[-1]["start"], modern[-1]["end"]), (2001, 2020))
        self.assertTrue(self.period["allow_finite_last_period"])
        description = self.period["description"].lower()
        self.assertIn("half-open", description)
        self.assertIn("end 1499", description)
        self.assertIn("observations end in 2015", description)
        self.assertIn("does not extrapolate", description)
        self.assertEqual(self.period["membership_rule"], "life_interval_or_known_endpoint_in_period")
        self.assertEqual(self.period["missing_date_policy"], "exclude_if_both_dates_missing")
        self.assertEqual(self.period["partial_date_policy"], "include_known_endpoint_periods")
        self.assertEqual(self.period["invalid_interval_policy"], "exclude_if_death_before_birth")

    def test_pipeline_paths_taxonomies_and_graphmask_protocol_are_consistent(self):
        config = self.pipeline
        self.assertEqual(config["source_data"], "artifacts/freebase_provisional_l1_v1/graph_data.pt")
        self.assertEqual(config["period_root"], "artifacts/freebase_life_periods_20y_v1")
        self.assertEqual(config["period_config"], "config/freebase_life_periods_20y_v1.json")
        self.assertEqual(config["binary_taxonomy"], "config/freebase_tie_taxonomy_v1.json")
        self.assertEqual(config["multi_group_taxonomy"], "config/freebase_tie_taxonomy_acquired_subgroups_v1.json")
        self.assertEqual(config["relation_root"], "artifacts/freebase_grouped_20y_v1")
        self.assertEqual(config["model_root"], "runs/freebase_grouped_20y_v1")
        self.assertEqual(config["graphmask_root"], "runs_graphmask/freebase_grouped_20y_v1")
        roots = [Path(config[k]) for k in ("relation_root", "model_root", "graphmask_root")]
        for i, left in enumerate(roots):
            for right in roots[i + 1:]:
                self.assertNotEqual(left, right)
                self.assertNotIn(left, right.parents)
                self.assertNotIn(right, left.parents)

        self.assertEqual(config["representations"], ["multi_group"])
        self.assertEqual(config["num_bases"], {"binary": 4, "multi_group": 10})
        self.assertEqual(config["num_bases"]["multi_group"], 2 * len(self.multi["groups"]))
        self.assertIsNone(config["device"])
        self.assertEqual(config["seed"], 42)

        source = config["source_prepare"]
        self.assertEqual(source["input_dir"], "external_data/freebase/descriptive_v2_local/05_final")
        self.assertEqual(source["audit_file"], "external_data/freebase/profession_l1_review_v1/person_l1_audit.tsv")
        self.assertEqual(source["crosswalk_file"], "docs/freebase_profession_review_2026-10-03/profession_l1_crosswalk_draft.tsv")
        self.assertEqual(source["label_policy"], "mapped_first_other_fallback")
        self.assertEqual(source["seed"], 42)
        self.assertEqual(source["min_class_count"], 3)
        self.assertAlmostEqual(source["train_ratio"] + source["val_ratio"] + source["test_ratio"], 1.0)

        train = config["train"]
        self.assertEqual((train["model"], train["num_layers"], train["epochs"]), ("rgcn", 2, 50))
        self.assertEqual(train["num_neighbors"], "15,10")
        self.assertEqual(train["occupation_feature_levels"], "1")
        self.assertEqual(train["auxiliary_features"], "temporal")
        graphmask = config["graphmask_train"]
        self.assertEqual(graphmask["beta"], 0.03)
        self.assertEqual(graphmask["max_relative_macro_f1_diff"], 0.05)
        self.assertEqual(graphmask["epochs_per_layer"], 3)
        self.assertEqual(graphmask["num_neighbors"], "auto")
        self.assertEqual(graphmask["train_split"], "train")
        self.assertEqual(graphmask["validation_split"], "val")
        self.assertEqual(config["graphmask_report"]["split"], "test")

if __name__ == "__main__":
    unittest.main()
