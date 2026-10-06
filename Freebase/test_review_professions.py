"""Check the label policy's consequential edge cases, independent of real counts."""
import unittest

from Freebase.review_professions import resolve_professions


class ProfessionResolutionTest(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "Actor": {"status": "proposed", "proposed_level1": "Culture"},
            "Writer": {"status": "proposed", "proposed_level1": "Culture"},
            "Physicist": {"status": "proposed", "proposed_level1": "Discovery/Science"},
            "Agent": {"status": "needs_review", "proposed_level1": ""},
            "Farmer": {"status": "out_of_scope", "proposed_level1": ""},
        }

    def test_distinct_roles_can_share_one_target_class(self):
        self.assertEqual(resolve_professions(["Actor", "Writer"], self.mapping),
                         (["Culture"], [], "proposed_single_l1", "Culture"))

    def test_cross_class_roles_have_no_single_label(self):
        labels, blocked, status, chosen = resolve_professions(["Actor", "Physicist"], self.mapping)
        self.assertEqual(labels, ["Culture", "Discovery/Science"])
        self.assertEqual((blocked, status, chosen), ([], "proposed_multiple_l1", ""))

    def test_unknown_and_excluded_values_cannot_be_silently_dropped(self):
        for extra in ("Agent", "Farmer"):
            with self.subTest(extra=extra):
                labels, blocked, status, chosen = resolve_professions(["Actor", extra], self.mapping)
                self.assertEqual((labels, blocked, status, chosen),
                                 (["Culture"], [extra], "incomplete_mapping", ""))

    def test_unknown_value_and_empty_input_fail_instead_of_inventing_label(self):
        with self.assertRaises(KeyError):
            resolve_professions(["Undocumented"], self.mapping)
        with self.assertRaises(ValueError):
            resolve_professions([], self.mapping)


if __name__ == "__main__":
    unittest.main()
