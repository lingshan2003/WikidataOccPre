import csv
from pathlib import Path
import tempfile
import unittest

from DBpedia.prepare_bhht_l1_v1 import load_mapping, map_nodes, rewrite_relations, verify_export, write_rows


def source_mapping(label, level, state, count=1):
    return {"occupation_label": label, "level1": level, "review_status": state, "graph_nodes": str(count),
            "level1_origin": "pending_context" if not level else "llm_semantic_proposal",
            "confidence": "low" if not level else "medium", "semantic_rationale": "原始未定理由" if not level else "职业职责依据"}


def node(person, occupation):
    return {"node_id": person, "birth_year": "1921.0", "death_year": "", "occupation_level1": occupation,
            "occupation_level2": "", "occupation_level3": "", "country": ""}


class ProvisionalDBpediaDatasetTests(unittest.TestCase):
    def test_other_fallback_does_not_erase_original_state(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mapping.tsv"
            write_rows(path, [source_mapping("institution", "", "needs_context"), source_mapping("ordinary worker", "Other", "resolved")])
            mapped = load_mapping(path)
            self.assertEqual(mapped["institution"]["training_level1"], "Other")
            self.assertEqual(mapped["institution"]["temporary_other_fallback"], 1)
            self.assertEqual(mapped["institution"]["level1"], "")
            self.assertEqual(mapped["institution"]["review_status"], "needs_context")
            self.assertEqual(mapped["ordinary worker"]["temporary_other_fallback"], 0)

    def test_nodes_dates_and_original_selection_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mapping.tsv"
            write_rows(path, [source_mapping("institution", "", "needs_context"), source_mapping("writer", "Culture", "resolved")])
            original = [node("a", "institution"), node("b", "writer")]
            assignments = [{"person_uri": "a", "occupation_label": "institution"}, {"person_uri": "b", "occupation_label": "writer"}]
            result, audit = map_nodes(original, load_mapping(path), assignments)
            self.assertEqual([r["occupation_level1"] for r in result], ["Other", "Culture"])
            for old, new in zip(original, result):
                for k in ["node_id", "birth_year", "death_year", "country"]:
                    self.assertEqual(old[k], new[k])
            self.assertEqual(audit[0]["occupation_label"], "institution")
            self.assertEqual(audit[0]["original_l1_review_status"], "needs_context")
            self.assertEqual(original[0]["occupation_level1"], "institution")

    def test_edge_facts_and_nonoccupation_columns_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = {"Node1": "a", "Relation": "influencedBy", "Node2": "b", "Node1_Birth": "1921", "Node1_Death": "",
                   "Node2_Birth": "1900", "Node2_Death": "1980", "Node1_occ_level1": "institution", "Node2_occ_level1": "writer",
                   "Node1_occ_level2": "", "Node2_occ_level2": "", "Node1_occ_level3": "", "Node2_occ_level3": "",
                   "Node1_Country": "", "Node2_Country": ""}
            write_rows(root / "old.csv", [old])
            originals = [node("a", "institution"), node("b", "writer")]
            new_nodes = [node("a", "Other"), node("b", "Culture")]
            counts = rewrite_relations(root / "old.csv", root / "new.csv", root / "facts.csv", new_nodes, originals)
            self.assertEqual(counts, {"influencedBy": 1})
            self.assertEqual(verify_export(root / "old.csv", root / "new.csv"), 1)
            with (root / "new.csv").open() as f:
                new = next(csv.DictReader(f))
            for k, v in old.items():
                if "_occ_level" not in k:
                    self.assertEqual(new[k], v)
            self.assertEqual(new["Node1_occ_level1"], "Other")

    def test_review_status_and_population_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mapping.tsv"
            write_rows(path, [source_mapping("pending", "Other", "needs_context")])
            with self.assertRaises(ValueError):
                load_mapping(path)
            write_rows(path, [source_mapping("writer", "Culture", "resolved", count=2)])
            with self.assertRaises(ValueError):
                map_nodes([node("a", "writer")], load_mapping(path), [{"person_uri": "a", "occupation_label": "writer"}])


if __name__ == "__main__":
    unittest.main()
