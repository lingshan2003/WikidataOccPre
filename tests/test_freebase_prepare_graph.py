"""Protect the arbitrary-label audit and relation normalization contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Freebase.prepare_graph import normalize_relations, resolve_input_directory, select_label, settings, single_year, write_rows
from Freebase.review_professions import resolve_professions


class InputLayoutTests(unittest.TestCase):
    def test_missing_local_layout_uses_complete_server_export(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("Freebase.prepare_graph.ROOT", Path(directory)), mock.patch.dict("os.environ", {}, clear=True):
            root = Path(directory).resolve()
            server = root / "external_data/freebase/processed/05_final"
            server.mkdir(parents=True)
            for name in ("nodes.csv", "main_relation_facts.csv"):
                (server / name).write_text("fixture")
            self.assertEqual(resolve_input_directory("external_data/freebase/descriptive_v2_local/05_final"), server)
            # A partial configured export must not silently change source batch.
            local = root / "external_data/freebase/descriptive_v2_local/05_final"
            local.mkdir(parents=True)
            (local / "nodes.csv").write_text("partial fixture")
            self.assertEqual(resolve_input_directory(str(local)), local)

    def test_explicit_input_override_is_used_even_if_missing(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {"FREEBASE_INPUT_DIR": directory}):
            self.assertEqual(resolve_input_directory("unused"), Path(directory).resolve())

    def test_missing_inputs_are_reported_together_before_stamping(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {}, clear=True):
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"source_data": str(root / "output/graph_data.pt"), "source_prepare": {
                "input_dir": str(root / "missing"), "audit_file": str(root / "audit.tsv"),
                "crosswalk_file": str(root / "crosswalk.tsv"), "relation_rules": str(root / "rules.json"),
                "label_policy": "mapped_first_other_fallback", "min_class_count": 3,
                "train_ratio": .7, "val_ratio": .1, "test_ratio": .2}}))
            with self.assertRaises(FileNotFoundError) as caught:
                settings(config, None, False)
            message = str(caught.exception)
            for name in ("nodes.csv", "main_relation_facts.csv", "audit.tsv", "crosswalk.tsv", "rules.json", "FREEBASE_INPUT_DIR"):
                self.assertIn(name, message)


class ProvisionalLabelTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "Writer": {"status": "proposed", "proposed_level1": "Culture"},
            "Artist": {"status": "proposed", "proposed_level1": "Culture"},
            "Scientist": {"status": "proposed", "proposed_level1": "Discovery/Science"},
            "Operator": {"status": "needs_review", "proposed_level1": ""},
        }

    def previous(self, raw):
        labels, unresolved, state, unique = resolve_professions(raw, self.mapping)
        return {"resolution_status": state, "proposed_single_l1": unique,
                "proposed_l1_set_json": json.dumps(labels),
                "unresolved_professions_json": json.dumps(unresolved)}

    def choose(self, raw, policy="mapped_first_other_fallback"):
        return select_label(raw, self.mapping, self.previous(raw), policy)

    def test_preserves_unique_and_selects_recorded_order_for_ambiguous_people(self):
        self.assertEqual(self.choose(["Writer", "Artist"]),
                         ("Culture", "Writer", "preserved_unique_candidate"))
        self.assertEqual(self.choose(["Operator", "Scientist", "Writer"]),
                         ("Discovery/Science", "Scientist", "forced_first_mapped_profession"))
        self.assertEqual(self.choose(["Writer", "Scientist"])[0], "Culture")

    def test_unmapped_occupation_is_auditable_other_or_explicit_raw_fallback(self):
        self.assertEqual(self.choose(["Operator"]),
                         ("Other", "Operator", "forced_unmapped_profession"))
        self.assertEqual(self.choose(["Operator"], "mapped_first_raw_fallback")[0], "Raw::Operator")

    def test_stale_person_audit_cannot_silently_change_labels(self):
        previous = self.previous(["Writer"])
        previous["proposed_single_l1"] = "Leadership"
        with self.assertRaisesRegex(ValueError, "disagree"):
            select_label(["Writer"], self.mapping, previous, "mapped_first_other_fallback")

    def test_date_conflicts_and_future_observations_are_missing(self):
        self.assertEqual(single_year("[1900]", 2026), (1900, "observed"))
        self.assertEqual(single_year("[2040]", 2026), ("", "future_year"))
        self.assertEqual(single_year("[1900, 1901]", 2026), ("", "conflicting_years"))
        self.assertEqual(single_year("[]", 2026), ("", "missing"))
        with self.assertRaises(ValueError):
            single_year("[true]", 2026)


class RelationNormalizationTests(unittest.TestCase):
    def test_symmetric_alias_duplicates_direction_and_self_loops(self):
        rules = Path(__file__).resolve().parents[1] / "Freebase/relation_rules.json"
        rows = [
            ("A", "B", "Sibling", "sibling", "kinship"),
            ("B", "A", "Sibling", "sibling", "kinship"),
            ("A", "A", "Children", "child", "kinship"),
            ("A", "B", "Children", "child", "kinship"),
            ("A", "B", "Influenced By", "influenced_by", "influence"),
            ("B", "A", "kp_lw/philosophy_influencer/influencee", "influenced_by", "influence"),
            ("B", "A", "Academic advisor", "academic_advisor_raw", "education"),
        ]
        fields = ["Node1_Name", "Node2_Name", "RawPredicate", "Relation", "RelationGroup", "SourceLine"]
        with tempfile.TemporaryDirectory() as directory:
            facts = Path(directory) / "facts.csv"
            write_rows(facts, fields, [dict(zip(fields, (*r, i + 1))) for i, r in enumerate(rows)])
            normalized, vocabulary, stats = normalize_relations(facts, rules, {"A", "B"})
            pairs = {(r["source"], r["relation"], r["target"]): r["source_fact_count"] for r in normalized}
            self.assertEqual(pairs, {("A", "sibling", "B"): 2, ("A", "child", "B"): 1,
                                     ("A", "influenced_by", "B"): 2, ("B", "academic_advisor_raw", "A"): 1})
            self.assertEqual(len(vocabulary), 9)
            self.assertEqual(stats["self_loop_facts_removed"], 1)
            self.assertEqual(stats["duplicates_removed_after_self_loops"], 2)
            with self.assertRaisesRegex(ValueError, "endpoint"):
                normalize_relations(facts, rules, {"A"})


if __name__ == "__main__":
    unittest.main()
