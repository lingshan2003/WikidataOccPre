"""Export the relation code inventory from a local CBDB SQLite release."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "external_data/cbdb/2026.09.14/cbdb_20260914.sqlite3"
DEFAULT_OUT = ROOT / "docs/cbdb_relation_audit_2026-09-29"


def export_query(conn: sqlite3.Connection, path: Path, query: str) -> int:
    cursor = conn.execute(query)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow([column[0] for column in cursor.description])
        rows = cursor.fetchall()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.database.resolve()
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        queries = {
            "kinship_codes.tsv": """
                SELECT c.c_kincode AS code, COUNT(d.c_kin_code) AS raw_rows,
                       c.c_kinrel_chn AS chinese, c.c_kinrel AS kin_path,
                       c.c_kinrel_simplified AS simplified_path,
                       c.c_kin_pair1 AS inverse_code_1,
                       c.c_kin_pair2 AS inverse_code_2,
                       c.c_upstep AS up_steps, c.c_dwnstep AS down_steps,
                       c.c_marstep AS marriage_steps,
                       c.c_colstep AS collateral_steps
                FROM KINSHIP_CODES c
                JOIN KIN_DATA d ON d.c_kin_code = c.c_kincode
                GROUP BY c.c_kincode ORDER BY raw_rows DESC, code
            """,
            "association_codes.tsv": """
                SELECT c.c_assoc_code AS code, COUNT(d.c_assoc_code) AS raw_rows,
                       c.c_assoc_desc_chn AS chinese, c.c_assoc_desc AS english,
                       c.c_assoc_pair AS inverse_code,
                       r.c_assoc_type_code AS official_subtype,
                       t.c_assoc_type_desc_chn AS subtype_chinese,
                       t.c_assoc_type_desc AS subtype_english
                FROM ASSOC_CODES c
                JOIN ASSOC_DATA d ON d.c_assoc_code = c.c_assoc_code
                LEFT JOIN ASSOC_CODE_TYPE_REL r ON r.c_assoc_code = c.c_assoc_code
                LEFT JOIN ASSOC_TYPES t ON t.c_assoc_type_code = r.c_assoc_type_code
                GROUP BY c.c_assoc_code ORDER BY raw_rows DESC, code
            """,
            "association_subtypes.tsv": """
                SELECT t.c_assoc_type_code AS subtype,
                       t.c_assoc_type_desc_chn AS chinese,
                       t.c_assoc_type_desc AS english,
                       t.c_assoc_type_parent_id AS parent,
                       COUNT(d.c_assoc_code) AS raw_rows,
                       COUNT(DISTINCT d.c_assoc_code) AS used_codes
                FROM ASSOC_TYPES t
                LEFT JOIN ASSOC_CODE_TYPE_REL r
                  ON r.c_assoc_type_code = t.c_assoc_type_code
                LEFT JOIN ASSOC_DATA d ON d.c_assoc_code = r.c_assoc_code
                WHERE t.c_assoc_type_level = 1
                GROUP BY t.c_assoc_type_code ORDER BY t.c_assoc_type_code
            """,
        }
        for name, query in queries.items():
            count = export_query(conn, args.output_dir / name, query)
            print(f"{name}: {count} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
