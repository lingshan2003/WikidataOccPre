"""Small end-to-end fixture for the Freebase Easy staged extractor."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("extract.py")


class ExtractTest(unittest.TestCase):
    def test_multivalue_professions_and_raw_pair_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            triples = [
                ("A", "is-a", "Person"), ("B", "is-a", "Person"),
                ("C", "is-a", "Person"), ("D", "is-a", "Person"),
                ("E", "is-a", "Person"), ("F", "is-a", "Animal"),
                ("A", "Profession", "Writer"), ("A", "Profession", "Actor"),
                ("B", "Profession", "Mathematician"), ("C", "Profession", "Artist"),
                ("D", "Profession", "Singer"), ("E", "Profession", "Poet"),
                ("A", "Date of birth", '"1900"^^<http://www.w3.org/2001/XMLSchema#gYear>'),
                ("B", "Date of death", '"1980"^^<http://www.w3.org/2001/XMLSchema#gYear>'),
                ("C", "Date of Birth", '"1910-03"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'),
                ("D", "Date of birth", '"2000-13-01"^^<http://www.w3.org/2001/XMLSchema#date>'),
                ("E", "Date of birth", '"1920-01-01"^^<http://www.w3.org/2001/XMLSchema#date>'),
                ("A", "Sibling", "B"), ("B", "Sibling", "A"),
                ("A", "Academic advisor", "C"),
                ("A", "Unknown relation", "B"),
                ("A", "Sibling", "D"), ("A", "Sibling", "E"),
                ("B", "Matches Lost", "C"), ("E", "is-a", "Person"),
            ]
            facts = root / "facts.txt"
            facts.write_bytes(b"".join(("\t".join(row) + "\t.\n").encode() for row in triples))
            links = root / "links.txt"
            links.write_bytes(b"".join(
                (((":d:001:" + name) if name == "B" else name)
                 + "\tfreebase-entity\t<http://rdf.freebase.com/ns/" + identifier + ">\t.\n").encode()
                for name, identifier in (("A", "m.a"), ("B", "g/11b"), ("C", "m.c"))
            ))
            output = root / "processed"
            command = [sys.executable, str(SCRIPT), "--facts", str(facts),
                       "--links", str(links), "--output-dir", str(output),
                       "--progress-every", "1000000"]
            subprocess.run(command, check=True, capture_output=True, text=True)

            def summary(stage: str) -> dict:
                return json.loads((output / stage / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary("01_people")["person_names"], 5)
            self.assertEqual(summary("02_cohort")["cohort_people"], 4)
            self.assertEqual(summary("02_cohort")["multi_profession_people"], 1)
            self.assertEqual(summary("02_cohort")["death_only_people"], 1)
            self.assertEqual(summary("03_pairs")["cohort_person_pairs"], 6)
            self.assertEqual(summary("04_relations")["review_facts"], 5)
            self.assertEqual(summary("04_relations")["main_facts"], 4)
            self.assertEqual(summary("05_final")["final_facts"], 4)
            self.assertEqual(summary("05_final")["names_without_id"], 1)
            self.assertEqual(summary("06_stats")["multi_profession_people"], 1)
            with (output / "05_final/main_relation_facts.csv").open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertTrue(any(row["Node1_Name"] == "A" and
                                json.loads(row["Node1_Professions"]) == ["Actor", "Writer"]
                                for row in records))
            self.assertTrue(any(row["Node2_Name"] == "B" and row["Node2_ID"] == "g/11b"
                                for row in records))
            self.assertTrue(any(row["Node2_Name"] == "E" and row["Node2_IDStatus"] == "missing"
                                for row in records))
            self.assertTrue(all(row["Node2_Name"] not in ("D", "F") for row in records))
            subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
