"""Guard lookup policy at semantic ambiguity and evidence boundaries."""
import unittest

from Freebase.match_bhht_professions import lookup


class BHHTLookupTests(unittest.TestCase):
    def setUp(self):
        # Source literals and relative category order from the downloaded prog_occ.do.
        entries = [
            ("Sports/Games", "Sports/Games", "fighter"),
            ("Culture", "Culture-core", "_artist"),
            ("Culture", "Culture-core", "producer"),
            ("Culture", "Culture-core", "_music"),
            ("Culture", "Culture-periphery", "record_producer"),
            ("Discovery/Science", "Academia", "conservationist"),
            ("Leadership", "Politics", "conservationist"),
            ("Leadership", "Administration/Law", "_lawyer"),
            ("Other", "Other", "criminal"),
        ]
        self.rules = [{"level1": a, "level2": b, "level3_keyword": c, "rule_priority": i}
                      for i, (a, b, c) in enumerate(entries, 1)]

    def test_exact_phrase_preferred_but_original_first_match_retained(self):
        r, _, _ = lookup("Record producer", self.rules, {})
        self.assertEqual(r["match_status"], "exact_unique")
        self.assertEqual(r["candidate_level2"], "Culture-periphery")
        self.assertEqual(r["author_first_match_level2"], "Culture-core")
        self.assertEqual(r["exact_overrides_first_match_parent"], 1)

    def test_case_and_spaces_normalized_without_hyphen_aliasing(self):
        r, _, _ = lookup("RECORD PRODUCER", self.rules, {})
        self.assertEqual(r["lookup_method"], "author_exact")
        hyphenated, _, _ = lookup("Record-producer", self.rules, {})
        self.assertEqual(hyphenated["lookup_method"], "author_substring")
        self.assertEqual(hyphenated["candidate_level2"], "Culture-core")

    def test_exact_cross_l1_does_not_choose_first_label(self):
        r, _, _ = lookup("Conservationist", self.rules, {})
        self.assertEqual(r["match_status"], "exact_cross_l1_conflict")
        self.assertEqual(r["candidate_level1"], "")
        self.assertEqual(r["candidate_level2"], "")
        self.assertEqual(r["author_first_match_level1"], "Discovery/Science")

    def test_criminal_defense_lawyer_keeps_both_categories(self):
        r, hits, _ = lookup("Criminal defense lawyer", self.rules, {})
        self.assertEqual(r["match_status"], "substring_cross_l1_conflict")
        self.assertEqual(r["candidate_level1"], "")
        self.assertEqual({h["level1"] for h in hits}, {"Leadership", "Other"})

    def test_observed_only_reference_does_not_supply_author_mapping(self):
        observed = {"agent": [{"level1": "Culture", "level2": "Culture-core"}]}
        r, _, _ = lookup("Agent", self.rules, observed)
        self.assertEqual(r["match_status"], "no_author_rule")
        self.assertEqual(r["candidate_level1"], "")
        self.assertNotEqual(r["observed_main_reference_json"], "[]")

    def test_observed_conflict_flagged_without_replacing_author_candidate(self):
        observed = {"artist": [{"level1": "Sports/Games", "level2": "Sports/Games"}]}
        r, _, _ = lookup("Artist", self.rules, observed)
        self.assertEqual(r["candidate_level1"], "Culture")
        self.assertEqual(r["observed_reference_l1_disagrees"], 1)

    def test_source_nbsp_literal_not_silently_repaired(self):
        rules = [{"level1": "Sports/Games", "level2": "Sports/Games",
                  "level3_keyword": "squash\u00a0", "rule_priority": 1}]
        r, _, _ = lookup("Squash", rules, {})
        self.assertEqual(r["match_status"], "no_author_rule")


if __name__ == "__main__":
    unittest.main()
