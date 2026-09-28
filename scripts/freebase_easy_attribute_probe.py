#!/usr/bin/env python3
"""Probe raw Freebase Easy person-attribute values without filtering people.

Reads only the exported person_attributes.tsv. Counts occupation values and
date literal shapes, and takes reproducible reservoir samples for review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys
import time


HEADER = b"source_line\tperson_name\tpredicate\traw_value"
SAMPLED_PREDICATES = (
    b"Profession", b"Occupation", b"Date of birth", b"Date of Birth",
    b"Date of death", b"Date of Death",
)
DATE_PREDICATES = {b"Date of birth", b"Date of Birth", b"Date of death", b"Date of Death"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.sample_size < 0 or args.progress_every < 1:
        parser.error("--sample-size must be nonnegative; --progress-every must be positive")
    return args


def show(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def date_shape(value: bytes) -> tuple[bytes, bytes]:
    literal, separator, datatype = value.partition(b"^^<")
    if separator and datatype.endswith(b">"):
        datatype = datatype[:-1].rsplit(b"#", 1)[-1].rsplit(b"/", 1)[-1]
    elif separator:
        datatype = b"malformed_datatype"
    else:
        datatype = b"untyped"
    shape = re.sub(rb"[0-9]", b"9", literal)
    return datatype, shape[:100]


def main() -> None:
    args = parse_args()
    source = args.attributes.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input file does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output directory is not empty; choose a new one: {output}")
    output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    rng = random.Random(20260928)
    counts: Counter[bytes] = Counter()
    occupation_values: dict[bytes, Counter[bytes]] = defaultdict(Counter)
    shapes: Counter[tuple[bytes, bytes, bytes]] = Counter()
    samples: dict[bytes, list[dict[str, object]]] = {name: [] for name in SAMPLED_PREDICATES}
    rows = malformed = 0
    with source.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != HEADER:
            raise SystemExit("Unexpected person_attributes.tsv header")
        for raw in handle:
            rows += 1
            fields = raw.rstrip(b"\r\n").split(b"\t", 3)
            if len(fields) != 4 or not all(fields) or not fields[0].isdigit():
                malformed += 1
                continue
            line_no, person, predicate, value = fields
            counts[predicate] += 1
            if predicate in (b"Profession", b"Occupation"):
                occupation_values[predicate][value] += 1
            if predicate in DATE_PREDICATES:
                datatype, shape = date_shape(value)
                shapes[(predicate, datatype, shape)] += 1
            if predicate in samples and args.sample_size:
                seen = counts[predicate]
                position = seen - 1 if seen <= args.sample_size else rng.randrange(seen)
                if position < args.sample_size:
                    item: dict[str, object] = {
                        "source_line": int(line_no),
                        "person_name": show(person),
                        "raw_value": show(value),
                    }
                    if seen <= args.sample_size:
                        samples[predicate].append(item)
                    else:
                        samples[predicate][position] = item
            if rows % args.progress_every == 0:
                print(f"read {rows:,} attribute rows", file=sys.stderr, flush=True)

    with (output / "occupation_values.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\traw_value\tfacts\n")
        for predicate in (b"Profession", b"Occupation"):
            for value, count in occupation_values[predicate].most_common():
                handle.write(f"{show(predicate)}\t{show(value)}\t{count}\n")
    with (output / "date_value_shapes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("predicate\tdatatype\tlexical_shape\tfacts\n")
        for (predicate, datatype, shape), count in sorted(
            shapes.items(), key=lambda item: (-item[1], item[0])
        ):
            handle.write(f"{show(predicate)}\t{show(datatype)}\t{show(shape)}\t{count}\n")
    (output / "value_samples.json").write_text(
        json.dumps({show(name): samples[name] for name in SAMPLED_PREDICATES},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete_subset_scan" if malformed == 0 else "subset_scan_with_malformed_rows",
        "attributes": str(source),
        "rows_read": rows,
        "malformed_rows": malformed,
        "predicate_fact_counts": {show(name): counts[name] for name in SAMPLED_PREDICATES},
        "distinct_profession_values": len(occupation_values[b"Profession"]),
        "distinct_occupation_values": len(occupation_values[b"Occupation"]),
        "date_literal_shapes": len(shapes),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "notes": [
            "A date shape is only a lexical pattern, not proof of a valid calendar date or year.",
            "Occupation values count facts, not distinct people or final model classes.",
            "Samples are reproducible and should be reviewed before implementing date parsing.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
