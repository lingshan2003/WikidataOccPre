"""Contract tests for the versioned inherited/acquired taxonomy loader."""

import json
import tempfile
import unittest
from pathlib import Path

from training.tie_taxonomy import load_tie_taxonomy, parse_tie_group_selection, resolve_tie_ablation


RELATION_TO_ID = {
    "father": 0,
    "father__rev": 1,
    "student_of": 2,
    "student_of__rev": 3,
    "spouse": 4,
    "spouse__rev": 5,
}


class TieTaxonomyTests(unittest.TestCase):
    def write_taxonomy(self, payload):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "taxonomy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_all_remaining_resolves_every_non_inherited_relation_and_both_directions(self):
        path = self.write_taxonomy({
            "name": "test",
            "version": 1,
            "groups": {"inherited": ["father"], "acquired": "all_remaining"},
        })
        taxonomy = load_tie_taxonomy(path, RELATION_TO_ID)
        self.assertEqual(taxonomy.inherited, ("father",))
        self.assertEqual(taxonomy.acquired, ("spouse", "student_of"))
        ids, bases = resolve_tie_ablation(taxonomy, ("inherited",), RELATION_TO_ID)
        self.assertEqual(ids, {0, 1})
        self.assertEqual(bases, ("father",))
        self.assertEqual(taxonomy.group_for_base_relation("father__rev"), "inherited")
        self.assertEqual(taxonomy.group_for_base_relation("student_of__rev"), "acquired")
        self.assertEqual(taxonomy.manifest()["sha256"], taxonomy.sha256)

    def test_explicit_groups_must_cover_every_relation_without_overlap(self):
        path = self.write_taxonomy({
            "name": "incomplete",
            "version": 1,
            "groups": {"inherited": ["father"], "acquired": ["student_of"]},
        })
        with self.assertRaisesRegex(ValueError, "cover every base relation"):
            load_tie_taxonomy(path, RELATION_TO_ID)

        path = self.write_taxonomy({
            "name": "overlap",
            "version": 1,
            "groups": {
                "inherited": ["father"],
                "acquired": ["father", "student_of", "spouse"],
            },
        })
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_tie_taxonomy(path, RELATION_TO_ID)

    def test_unknown_and_generated_reverse_names_are_rejected(self):
        path = self.write_taxonomy({
            "name": "unknown",
            "version": 1,
            "groups": {"inherited": ["mother"], "acquired": "all_remaining"},
        })
        with self.assertRaisesRegex(ValueError, "absent from this artifact"):
            load_tie_taxonomy(path, RELATION_TO_ID)

        path = self.write_taxonomy({
            "name": "reverse",
            "version": 1,
            "groups": {"inherited": ["father__rev"], "acquired": "all_remaining"},
        })
        with self.assertRaisesRegex(ValueError, "__rev"):
            load_tie_taxonomy(path, RELATION_TO_ID)

    def test_selection_parser_is_strict(self):
        self.assertEqual(parse_tie_group_selection("none"), ())
        self.assertEqual(parse_tie_group_selection("acquired,inherited"), ("acquired", "inherited"))
        with self.assertRaisesRegex(ValueError, "Unknown tie groups"):
            parse_tie_group_selection("kinship")


if __name__ == "__main__":
    unittest.main()
