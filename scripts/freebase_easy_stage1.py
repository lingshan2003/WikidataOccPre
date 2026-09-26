#!/usr/bin/env python3
"""Inspect Freebase Easy facts and extract provisional Person entity names.

This is a read-only, CPU-only first pass. It does not build a graph or treat
``is-a Person`` as a final definition of the study population. Run a small
sample first, inspect its output, and only then request a complete scan.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import BinaryIO, Iterator
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--zip", type=Path, help="Freebase Easy ZIP")
    source_group.add_argument("--facts", type=Path, help="Extracted facts.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--member", default="facts.txt",
        help="ZIP member to scan; a bare filename is also found inside a ZIP subdirectory",
    )
    parser.add_argument(
        "--max-lines", type=int, default=100_000,
        help="Stop after this many facts for a format probe (default: 100000)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Scan the complete facts.txt"
    )
    parser.add_argument("--type-predicate", default="is-a")
    parser.add_argument("--person-type", default="Person")
    parser.add_argument("--sample-rows", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.max_lines < 1 or args.sample_rows < 0 or args.progress_every < 1:
        parser.error("--max-lines and --progress-every must be positive; --sample-rows must be nonnegative")
    return args


def token(value: bytes) -> bytes:
    """Remove RDF-style angle brackets and a trailing triple terminator."""
    value = value.strip()
    if value.endswith(b" ."):
        value = value[:-2].rstrip()
    if value.startswith(b"<") and value.endswith(b">"):
        value = value[1:-1]
    return value


def display(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def write_counts(path: Path, name: str, counts: Counter[bytes]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{name}\tcount\n")
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{display(value)}\t{count}\n")


def unique_sort(source: Path, target: Path, scratch_dir: Path) -> int:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(
        ["sort", "-u", "-T", str(scratch_dir), "-o", str(target), str(source)],
        check=True,
        env=env,
    )
    source.unlink()
    with target.open("rb") as handle:
        return sum(1 for _ in handle)


@contextmanager
def facts_stream(source: Path, member: str, is_zip: bool) -> Iterator[tuple[BinaryIO, str | None, int]]:
    if not is_zip:
        with source.open("rb") as stream:
            yield stream, None, source.stat().st_size
        return
    with zipfile.ZipFile(source) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            matches = [
                item for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name == member
            ] if "/" not in member else []
            if len(matches) != 1:
                raise SystemExit(
                    f"ZIP member {member!r} not found uniquely; "
                    f"matches: {[item.filename for item in matches]}; "
                    f"available: {archive.namelist()}"
                ) from exc
            info = matches[0]
        with archive.open(info) as stream:
            yield stream, info.filename, info.file_size


def main() -> None:
    args = parse_args()
    source = (args.zip or args.facts).expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input file does not exist: {source}")
    expected_predicate = token(args.type_predicate.encode("utf-8"))
    expected_person = token(args.person_type.encode("utf-8"))
    predicate_counts: Counter[bytes] = Counter()
    type_counts: Counter[bytes] = Counter()
    sample_rows: list[str] = []
    malformed_samples: list[str] = []
    lines = parsed = malformed = person_type_facts = uncompressed_bytes = 0
    started = time.monotonic()
    candidate_path = output / "person_candidates_unsorted.txt"

    with facts_stream(source, args.member, args.zip is not None) as (facts, member_name, total_bytes):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise SystemExit(f"Output directory is not empty; choose a new one: {output}")
        output.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("wb") as candidates:
            for raw in facts:
                if not args.full and lines >= args.max_lines:
                    break
                lines += 1
                uncompressed_bytes += len(raw)
                if len(sample_rows) < args.sample_rows:
                    sample_text = display(raw.rstrip(b"\r\n"))
                    sample_rows.append(f"{lines}\t{sample_text}")
                fields = raw.rstrip(b"\r\n").split(b"\t", 3)
                if len(fields) < 3:
                    malformed += 1
                    if len(malformed_samples) < 10:
                        malformed_samples.append(display(raw[:500].rstrip(b"\r\n")))
                    if lines == 10_000 and malformed == lines:
                        raise SystemExit("First 10000 rows have no three tab-separated fields; inspect sample format")
                    continue
                subject, predicate, obj = (token(field) for field in fields[:3])
                if not subject or not predicate or not obj:
                    malformed += 1
                    continue
                parsed += 1
                predicate_counts[predicate] += 1
                if predicate == expected_predicate:
                    type_counts[obj] += 1
                    if obj == expected_person:
                        candidates.write(subject + b"\n")
                        person_type_facts += 1
                if lines % args.progress_every == 0:
                    elapsed = max(time.monotonic() - started, 1)
                    print(
                        f"scanned {lines:,} rows ({lines / elapsed:,.0f}/s), "
                        f"candidate type facts {person_type_facts:,}",
                        file=sys.stderr,
                        flush=True,
                    )

    unique_path = output / "person_candidate_entities.txt"
    unique_persons = unique_sort(candidate_path, unique_path, output)
    write_counts(output / "predicate_counts.tsv", "predicate", predicate_counts)
    write_counts(output / "type_counts.tsv", "type_object", type_counts)
    (output / "sample_rows.txt").write_text(
        "\n".join(sample_rows) + ("\n" if sample_rows else ""), encoding="utf-8"
    )
    summary = {
        "status": "complete_scan" if args.full or uncompressed_bytes == total_bytes else "sample_only",
        "source_zip": str(source) if args.zip else None,
        "source_facts": str(source) if args.facts else None,
        "zip_bytes": source.stat().st_size if args.zip else None,
        "member": member_name,
        "member_uncompressed_bytes": total_bytes,
        "requested_max_lines": None if args.full else args.max_lines,
        "lines_scanned": lines,
        "uncompressed_bytes_scanned": uncompressed_bytes,
        "parsed_rows": parsed,
        "malformed_rows": malformed,
        "malformed_examples": malformed_samples,
        "distinct_predicates": len(predicate_counts),
        "distinct_type_objects": len(type_counts),
        "type_predicate": args.type_predicate,
        "person_type": args.person_type,
        "person_type_facts": person_type_facts,
        "unique_person_candidate_entities": unique_persons,
        "person_like_type_objects": [
            {"name": display(value), "count": count}
            for value, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
            if b"person" in value.lower()
        ][:100],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "interpretation": "Provisional is-a Person candidates; not yet verified as unique human beings or usable graph nodes.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "lines_scanned": lines,
        "malformed_rows": malformed,
        "distinct_predicates": len(predicate_counts),
        "person_type_facts": person_type_facts,
        "unique_person_candidate_entities": unique_persons,
        "output_dir": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
