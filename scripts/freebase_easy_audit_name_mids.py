#!/usr/bin/env python3
"""Audit MID coverage for names appearing in raw Freebase Easy Person pairs.

Read the small exported pair table to collect target names, then stream
freebase-links.txt once. The result measures name-to-MID ambiguity; it does
not resolve ambiguous facts or prove that a matched name is a real person.
CPU only. No facts.txt scan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PAIR_HEADER = b"source_line\tsubject_name\tpredicate\tobject_name"
URI_PREFIXES = (b"<http://rdf.freebase.com/ns/", b"<https://rdf.freebase.com/ns/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def display(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def link_name(subject: bytes) -> bytes | None:
    # Observed subject syntax: :d:000000...:human-readable name
    if not subject.startswith(b":d:"):
        return None
    separator = subject.find(b":", 3)
    if separator < 0 or not subject[3:separator].isdigit():
        return None
    return subject[separator + 1:]


def link_mid(obj: bytes) -> bytes | None:
    for prefix in URI_PREFIXES:
        if obj.startswith(prefix) and obj.endswith(b">"):
            mid = obj[len(prefix):-1]
            if mid.startswith((b"m.", b"g.")):
                return mid
    return None


def main() -> None:
    args = parse_args()
    pair_path = args.pairs.expanduser().resolve()
    links_path = args.links.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for source in (pair_path, links_path):
        if not source.is_file():
            raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")

    started = time.monotonic()
    targets: set[bytes] = set()
    pair_rows = malformed_pair_rows = 0
    with pair_path.open("rb") as pairs:
        if pairs.readline().rstrip(b"\r\n") != PAIR_HEADER:
            raise SystemExit("Unexpected person pair table header")
        for raw in pairs:
            pair_rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields):
                malformed_pair_rows += 1
                continue
            targets.add(fields[1])
            targets.add(fields[3])
    print(f"loaded {len(targets):,} distinct names from {pair_rows:,} pair rows",
          file=sys.stderr, flush=True)

    first_mid: dict[bytes, bytes] = {}
    ambiguous: dict[bytes, set[bytes]] = {}
    link_rows = matched_link_rows = malformed_link_rows = unexpected_subjects = 0
    unexpected_mid_uris = repeated_name_mid_rows = 0
    unexpected_subject_examples: list[str] = []
    unexpected_mid_examples: list[str] = []
    with links_path.open("rb") as links:
        for raw in links:
            link_rows += 1
            if link_rows % args.progress_every == 0:
                print(f"scanned {link_rows:,} link rows; matched {matched_link_rows:,} target rows",
                      file=sys.stderr, flush=True)
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or fields[1] != b"freebase-entity":
                malformed_link_rows += 1
                continue
            name = link_name(fields[0])
            if name is None:
                unexpected_subjects += 1
                if len(unexpected_subject_examples) < 5:
                    unexpected_subject_examples.append(display(fields[0][:200]))
                continue
            if name not in targets:
                continue
            mid = link_mid(fields[2])
            if mid is None:
                unexpected_mid_uris += 1
                if len(unexpected_mid_examples) < 5:
                    unexpected_mid_examples.append(display(fields[2][:200]))
                continue
            matched_link_rows += 1
            original = first_mid.get(name)
            if original is None:
                first_mid[name] = mid
            elif original == mid:
                repeated_name_mid_rows += 1
            else:
                mids = ambiguous.get(name)
                if mids is None:
                    ambiguous[name] = {original, mid}
                elif mid in mids:
                    repeated_name_mid_rows += 1
                else:
                    mids.add(mid)
    print(f"finished {link_rows:,} link rows", file=sys.stderr, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    unique_count = len(first_mid) - len(ambiguous)
    missing_count = len(targets) - len(first_mid)
    with (output / "name_mid_status.tsv").open("wb") as handle:
        handle.write(b"name\tstatus\tmid\tdistinct_mid_count\n")
        for name in sorted(targets):
            if name not in first_mid:
                handle.write(name + b"\tmissing\t\t0\n")
            elif name in ambiguous:
                handle.write(name + b"\tambiguous\t\t"
                             + str(len(ambiguous[name])).encode("ascii") + b"\n")
            else:
                handle.write(name + b"\tunique\t" + first_mid[name] + b"\t1\n")
    with (output / "ambiguous_name_mids.tsv").open("wb") as handle:
        handle.write(b"name\tmid\n")
        for name in sorted(ambiguous):
            for mid in sorted(ambiguous[name]):
                handle.write(name + b"\t" + mid + b"\n")
    with (output / "missing_name_examples.txt").open("wb") as handle:
        for name in sorted(targets - first_mid.keys())[:100]:
            handle.write(name + b"\n")

    summary = {
        "status": "complete_link_scan",
        "pairs": str(pair_path),
        "links": str(links_path),
        "pair_rows_read": pair_rows,
        "malformed_pair_rows": malformed_pair_rows,
        "target_distinct_names": len(targets),
        "link_rows_read": link_rows,
        "malformed_link_rows": malformed_link_rows,
        "unexpected_link_subject_rows": unexpected_subjects,
        "unexpected_link_subject_examples": unexpected_subject_examples,
        "target_rows_with_recognized_mid": matched_link_rows,
        "target_rows_with_unexpected_mid_uri": unexpected_mid_uris,
        "unexpected_mid_uri_examples": unexpected_mid_examples,
        "repeated_name_mid_rows": repeated_name_mid_rows,
        "names_with_unique_mid": unique_count,
        "names_with_multiple_mids": len(ambiguous),
        "names_without_mid": missing_count,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "Counts refer to distinct names in the exported raw pair subset, including false social-pair matches.",
            "Unique name-to-MID mapping in this file is provisional identity evidence, not proof that the Easy fact used that MID.",
            "Ambiguous and missing names must not be silently merged or dropped before relation coverage is reported.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
