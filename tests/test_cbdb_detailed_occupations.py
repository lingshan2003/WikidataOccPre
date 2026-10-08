"""Data identity, eligibility and lossless multi-office export checks."""
import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.cbdb_build_flat_triples import load_people
from scripts.cbdb_export_detailed_occupations import build, extract


def fixture(conn):
    conn.executescript("""
        CREATE TABLE BIOG_MAIN(c_personid INTEGER,c_name_chn TEXT,c_birthyear INTEGER,c_deathyear INTEGER);
        CREATE TABLE POSTED_TO_OFFICE_DATA(c_personid INTEGER,c_office_id INTEGER,c_posting_id INTEGER,c_firstyear INTEGER);
        CREATE TABLE OFFICE_CODES(c_office_id INTEGER,c_dy INTEGER,c_office_chn TEXT);
        CREATE TABLE DYNASTIES(c_dy INTEGER,c_dynasty_chn TEXT);
        CREATE TABLE STATUS_DATA(c_personid INTEGER,c_status_code INTEGER);
        INSERT INTO BIOG_MAIN VALUES
          (1,'甲',700,760),(2,'乙',0,800),(3,'丙',-5,0),(4,'丁',1000,NULL),
          (5,'戊',NULL,1100),(6,'己',0,NULL),(7,'庚',2000,1900),
          (8,'辛',2027,NULL),(9,'壬',900,NULL),(10,'癸',950,NULL);
        INSERT INTO OFFICE_CODES VALUES(10,15,'知縣'),(11,20,'知縣'),(12,15,'郎中'),(99,15,'未詳');
        INSERT INTO DYNASTIES VALUES(15,'宋'),(20,'明');
        INSERT INTO POSTED_TO_OFFICE_DATA VALUES
          (1,11,1,730),(1,10,2,720),(1,10,3,740),(1,12,4,745),(1,99,5,0),
          (2,99,6,0),(6,10,7,0),(7,10,8,0),(8,10,9,0),
          (9,0,10,0),(10,777,11,0);
        INSERT INTO STATUS_DATA VALUES(1,2),(3,34),(4,114),(4,9),(5,114),(5,210),(9,9);
    """)
    conn.commit()


class DetailedOccupationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rules = self.root / "rules.json"
        self.rules.write_text(json.dumps({"做官": [34], "写作": [114, 210], "艺术": [9]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_name_different_codes_survive_and_repeat_postings_collapse(self):
        conn = sqlite3.connect(":memory:")
        fixture(conn)
        people, counts = load_people(conn)
        data = extract(conn, people, self.rules)
        self.assertEqual(set(people), {1, 2, 3, 4, 5, 9, 10})
        self.assertEqual(counts["no_literal_birth_or_death"], 1)
        self.assertEqual(counts["invalid_birth_or_death"], 2)
        entries = data["occupations"][1]
        self.assertEqual([e.code for e in entries], ["OFFICE:10", "OFFICE:11", "OFFICE:12"])
        self.assertEqual([e.label for e in entries], ["知縣", "知縣", "郎中"])
        self.assertEqual([e.dynasty_name for e in entries[:2]], ["宋", "明"])
        self.assertEqual(entries[0].support, 2)
        self.assertEqual(data["offices"][1][99], 1)  # unknown preserved as evidence, not an occupation
        conn.close()

    def test_unknown_titles_and_status_only_do_not_invent_specific_offices(self):
        conn = sqlite3.connect(":memory:")
        fixture(conn)
        people, _ = load_people(conn)
        data = extract(conn, people, self.rules)
        self.assertEqual(data["occupations"][2][0].kind, "office_unknown_broad")
        self.assertEqual(data["occupations"][10][0].kind, "office_unknown_broad")
        self.assertEqual(data["occupations"][3][0].code, "STATUS_CLASS:做官")
        self.assertEqual(data["occupations"][5][0].label, "写作")
        self.assertEqual(data["occupations"][5][0].status_codes, (114, 210))
        self.assertNotIn(4, data["occupations"])  # conflicting nonofficial broad classes unchanged
        self.assertEqual(data["occupations"][9][0].label, "艺术")  # office 0 supplies no title
        conn.close()

    def test_wide_long_alignment_and_source_rows_are_not_truncated(self):
        database, output = self.root / "db.sqlite3", self.root / "result"
        conn = sqlite3.connect(database)
        fixture(conn)
        conn.close()
        summary = build(database, self.rules, output)
        with (output / "people_with_occupations.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = {int(row["PersonID"]): row for row in csv.DictReader(stream)}
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[1]["occ3"], "郎中")
        self.assertEqual(rows[1]["occ1_code"], "OFFICE:10")
        self.assertEqual(rows[1]["occ2_code"], "OFFICE:11")
        self.assertEqual(rows[2]["occ1"], "做官")
        self.assertEqual(rows[2]["SpecificOfficeCount"], "0")
        self.assertEqual(rows[2]["occ2"], "")
        with gzip.open(output / "person_occupations_long.tsv.gz", "rt", encoding="utf-8", newline="") as stream:
            long = list(csv.DictReader(stream, delimiter="\t"))
        for entry in long:
            wide = rows[int(entry["PersonID"])]
            self.assertEqual(wide[f'occ{entry["OccSlot"]}'], entry["OccupationNameZh"])
            self.assertEqual(wide[f'occ{entry["OccSlot"]}_code'], entry["OccupationCode"])
        with gzip.open(output / "office_postings_source.tsv.gz", "rt", encoding="utf-8", newline="") as stream:
            evidence = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(evidence), 7)
        self.assertEqual(Counter(int(row["c_personid"]) for row in evidence), {1: 5, 2: 1, 10: 1})
        self.assertEqual(summary["people_with_specific_offices"], 1)
        self.assertEqual(summary["unknown_office_codes"], [99, 777])
        self.assertEqual(summary["long_occupation_rows"], 8)
        with self.assertRaises(FileExistsError):
            build(database, self.rules, output)


if __name__ == "__main__":
    unittest.main()
