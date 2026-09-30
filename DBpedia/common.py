"""Shared streaming helpers for the local DBpedia extraction pipeline."""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import re
from pathlib import Path
from urllib.parse import unquote


DBR = "http://dbpedia.org/resource/"
DBO = "http://dbpedia.org/ontology/"
RDF_TYPE = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
LITERAL_RE = re.compile(rb'^"(?P<lex>(?:\\.|[^"\\])*)"\^\^<(?P<datatype>[^>]+)>$')
YEAR_RE = re.compile(rb'^(?P<year>-?\d{1,6})(?P<tz>Z|[+-]\d{2}:\d{2})?$')
DATE_RE = re.compile(rb'^(?P<year>-?\d{1,6})-(?P<month>\d{2})-(?P<day>\d{2})(?P<tz>Z|[+-]\d{2}:\d{2})?$')
XSD_DATE = b"http://www.w3.org/2001/XMLSchema#date"
XSD_GYEAR = b"http://www.w3.org/2001/XMLSchema#gYear"


def triples(path: Path):
    """Yield (1-based line, subject, predicate, object), or None fields if malformed.

    The three DBpedia mapping/type dumps used here have one N-Triple per line.
    Splitting only the first two spaces keeps literal spaces and escapes intact.
    """
    with bz2.open(path, "rb") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                subject, predicate, tail = line.rstrip(b"\r\n").split(b" ", 2)
                if not tail.endswith(b" ."):
                    raise ValueError("missing terminator")
                yield line_no, subject, predicate, tail[:-2]
            except ValueError:
                yield line_no, None, None, None


def uri_from_token(token: bytes) -> str:
    if not token.startswith(b"<") or not token.endswith(b">"):
        raise ValueError(f"Expected URI token: {token[:80]!r}")
    return token[1:-1].decode("utf-8", "replace")


def token_for_uri(uri: str) -> bytes:
    return b"<" + uri.encode("utf-8") + b">"


def resource_label(uri: str) -> str:
    return unquote(uri.rsplit("/", 1)[-1]).replace("_", " ")


def uri_warning(uri: str) -> str:
    local = uri.removeprefix(DBR)
    flags = []
    if "__" in local:
        flags.append("double_underscore")
    if local.startswith("List_of_"):
        flags.append("list_page")
    return ";".join(flags)


def leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def parse_date_literal(raw: bytes) -> tuple[int | None, str, str]:
    """Preserve the legacy year rule: parseable years survive quality warnings."""
    match = LITERAL_RE.fullmatch(raw)
    if match is None:
        return None, "other_literal", "unrecognized_literal"
    lexical, datatype = match.group("lex"), match.group("datatype")
    full_date = DATE_RE.fullmatch(lexical)
    bare_year = YEAR_RE.fullmatch(lexical)
    if full_date is not None:
        year = int(full_date.group("year"))
        month, day = int(full_date.group("month")), int(full_date.group("day"))
        if datatype != XSD_DATE:
            return year, "full_date", "unexpected_datatype"
        if not 1 <= month <= 12:
            return year, "full_date", "invalid_month"
        days = (31, 29 if leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if not 1 <= day <= days[month - 1]:
            return year, "full_date", "invalid_day_gregorian"
        return year, "full_date", ""
    if bare_year is not None:
        year = int(bare_year.group("year"))
        return year, "year_only", "" if datatype == XSD_GYEAR else "unexpected_datatype"
    return None, "other_lexical", "unrecognized_lexical"


def gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def tsv_writer(handle, columns: list[str] | tuple[str, ...]):
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return writer


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
