"""Check L1 triage boundaries, original priority, and source-node reconciliation."""
import unittest
from DBpedia.match_bhht_occupations import BHHT, load_rules, load_observed, match_occupation, validate_inputs


class DBpediaBHHTMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules(BHHT / "author_keyword_l123_rules.csv")
        cls.observed = load_observed(BHHT / "observed_main_occupation_l123.csv")

    def match(self, value):
        return match_occupation(value, self.rules, self.observed)[0]

    def test_same_l1_multiple_l2_does_not_block_l1(self):
        r = self.match("canonist")
        self.assertEqual(r["strict_level1"], "Leadership")
        self.assertEqual(r["strict_level2"], "")
        self.assertEqual(r["strict_level3_keyword"], "")
        self.assertEqual(r["needs_llm_review"], 0)
        self.assertEqual(r["l2_status"], "conflicting")

    def test_substring_l2_conflict_preserves_l1_candidate(self):
        r = self.match("television producer")
        self.assertEqual(r["candidate_level1"], "Culture")
        self.assertEqual(r["candidate_level2"], "")
        self.assertIn("substring_requires_l1_transfer_review", r["l1_review_flags_json"])
        self.assertNotIn("multiple_author_l2_candidates", r["l1_review_flags_json"])

    def test_embedded_keyword_conflict_is_not_strict_candidate(self):
        r = self.match("environmentalist advocate")
        self.assertEqual(r["candidate_level1"], "")
        self.assertIn("mentalist", r["all_substring_l123_json"])
        self.assertEqual(r["strict_level1"], "")
        self.assertEqual(r["needs_llm_review"], 1)

    def test_full_author_priority_is_not_silently_overridden(self):
        r = self.match("freedom fighter")
        self.assertEqual(r["candidate_level1"], "Leadership")
        self.assertEqual(r["author_first_match_level1"], "Sports/Games")
        self.assertEqual(r["strict_level1"], "")
        self.assertEqual(r["exact_overrides_first_match_l1"], 1)

    def test_cross_l1_conflict_and_unmatched_remain_unassigned(self):
        for value in ("ecologist", "sports manager", "voice acting in japan"):
            r = self.match(value)
            self.assertEqual(r["strict_level1"], "")
            self.assertEqual(r["candidate_level1"], "")
            self.assertEqual(r["needs_llm_review"], 1)

    def test_l3_is_not_arbitrarily_chosen_for_multi_keyword_hit(self):
        r = self.match("film producer")
        self.assertEqual(r["candidate_level1"], "Culture")
        self.assertEqual(r["candidate_level2"], "Culture-core")
        self.assertEqual(r["candidate_level3_keyword"], "")
        self.assertGreater(r["candidate_level3_count"], 1)

    def test_current_source_and_assignments_must_reconcile(self):
        inv = [{"occupation_label": "cleric", "graph_nodes": "1"}]
        ass = [{"person_uri": "person", "occupation_label": "cleric"}]
        self.assertIn("person", validate_inputs(inv, ass, [{"node_id": "person", "occupation_level1": "cleric"}]))
        with self.assertRaises(ValueError):
            validate_inputs(inv, ass, [{"node_id": "person", "occupation_level1": "artist"}])
        with self.assertRaises(ValueError):
            validate_inputs(inv, ass, [{"node_id": "different", "occupation_level1": "cleric"}])


if __name__ == "__main__":
    unittest.main()
