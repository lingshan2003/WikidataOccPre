import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from DBpedia.merge_bhht_l1_review import merge, read, validate_batch, validate_decision, write


def decision(rank="1", label="sample", nodes="1", level="Culture"):
    return {"rank": rank, "occupation_label": label, "graph_nodes": nodes,
            "semantic_level1": level, "review_status": "resolved", "confidence": "medium",
            "candidate_l1s_json": json.dumps([level]), "semantic_rationale": "该词描述表演创作职业。",
            "evidence": "semantic_reasoning_only", "reviewer": "dbpedia_l1_batch_1"}


class DBpediaL1ReviewTests(unittest.TestCase):
    def test_uncertain_terms_cannot_receive_other_fallback(self):
        row = decision(level="Other")
        row.update(review_status="needs_context", confidence="low")
        with self.assertRaises(ValueError):
            validate_decision(row)
        row.update(semantic_level1="", candidate_l1s_json='["Other","Leadership"]')
        validate_decision(row)

    def test_resolved_decision_requires_single_legal_l1(self):
        row = decision()
        validate_decision(row)
        for changes in [{"semantic_level1": "Unknown"}, {"candidate_l1s_json": '["Culture","Leadership"]'},
                        {"confidence": "low"}, {"semantic_rationale": ""}]:
            with self.assertRaises(ValueError):
                validate_decision({**row, **changes})

    def test_batch_metadata_and_coverage_are_checked(self):
        row = decision()
        validate_batch([row], [row], row["reviewer"])
        for outputs in [[], [row, row], [{**row, "graph_nodes": "3"}]]:
            with self.assertRaises(ValueError):
                validate_batch([row], outputs, row["reviewer"])

    def test_merge_changes_only_requested_two_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = [{"rank": "1", "occupation_label": "already_unique", "graph_nodes": "10",
                         "l1_lookup_status": "substring_l1_unique", "candidate_level1": "Leadership",
                         "candidate_parent_pairs_json": '[["Leadership","Politics"]]'}]
            for batch, status in [(1, "unmatched"), (2, "cross_l1_conflict"), (3, "unmatched")]:
                row = {"rank": str(batch + 1), "occupation_label": f"term_{batch}", "graph_nodes": "2",
                       "l1_lookup_status": status, "candidate_level1": "", "candidate_parent_pairs_json": "[]"}
                original.append(row)
                write(directory / f"batch_{batch}_input.tsv", [row])
                output = decision(rank=row["rank"], label=row["occupation_label"], nodes=row["graph_nodes"])
                output["reviewer"] = f"dbpedia_l1_batch_{batch}"
                write(directory / f"batch_{batch}_review.tsv", [output])
            source = directory / "source.tsv"
            write(source, original)
            with contextlib.redirect_stdout(io.StringIO()):
                merge(source, directory)
            merged = read(directory / "occupation_l1_semantic_crosswalk.tsv")
            self.assertEqual(merged[0]["level1"], "Leadership")
            self.assertEqual(merged[0]["review_status"], "outside_this_review_scope")
            self.assertEqual({r["level1"] for r in merged[1:]}, {"Culture"})
            self.assertEqual(read(source), original)
            self.assertTrue(all(r["human_confirmed"] == "0" for r in merged))


if __name__ == "__main__":
    unittest.main()
