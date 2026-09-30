"""Run the four-stage DBpedia person/social/occupation extraction pipeline.

Example:
    python3 DBpedia/run.py \
      --raw-dir external_data/dbpedia/2022.12.01 \
      --output-dir external_data/dbpedia/processed

Use --from-stage dates to rerun dates and occupations after prior stages exist.
The pipeline only extracts data; it does not build period graphs or model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dates
import occupations
import people
import relations
from common import require_files, write_json


REPO = Path(__file__).resolve().parent.parent
STAGES = ("people", "relations", "dates", "occupations")
DIRS = {
    "people": "01_people",
    "relations": "02_relations",
    "dates": "03_dates",
    "occupations": "04_graph",
}
REQUIRED_OUTPUTS = {
    "people": ("person_candidates.tsv.gz", "summary.json"),
    "relations": ("raw_person_pairs.tsv.gz", "social_edges.tsv.gz", "summary.json"),
    "dates": ("date_evidence.tsv.gz", "graph_nodes.tsv.gz", "graph_edges.tsv.gz", "summary.json"),
    "occupations": ("candidate_evidence.tsv.gz", "occupation_candidate_nodes.tsv.gz", "graph_nodes.tsv.gz", "graph_edges.tsv.gz", "summary.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO / "external_data/dbpedia/2022.12.01")
    parser.add_argument("--output-dir", type=Path, default=REPO / "external_data/dbpedia/processed")
    parser.add_argument("--rules-dir", type=Path, default=Path(__file__).resolve().parent / "rules")
    parser.add_argument("--from-stage", choices=STAGES, default="people")
    parser.add_argument("--through-stage", choices=STAGES, default="occupations")
    parser.add_argument(
        "--allow-unreviewed-predicates", action="store_true",
        help="Keep unreviewed predicates out of the graph and report them instead of stopping.",
    )
    args = parser.parse_args()
    if STAGES.index(args.from_stage) > STAGES.index(args.through_stage):
        parser.error("--from-stage must precede --through-stage")

    raw = args.raw_dir.resolve()
    processed = args.output_dir.resolve()
    rules = args.rules_dir.resolve()
    type_dump = raw / "instance-types_lang=en_transitive.ttl.bz2"
    object_dump = raw / "mappingbased-objects_lang=en.ttl.bz2"
    literal_dump = raw / "mappingbased-literals_lang=en.ttl.bz2"
    relation_policy = rules / "relation_policy.tsv"
    q_review = rules / "q_class_review.tsv"
    require_files(type_dump, object_dump, literal_dump, relation_policy, q_review)
    processed.mkdir(parents=True, exist_ok=True)
    started = STAGES.index(args.from_stage)
    stopped = STAGES.index(args.through_stage)

    for stage in STAGES[:started]:
        require_files(*(processed / DIRS[stage] / name for name in REQUIRED_OUTPUTS[stage]))

    summaries = {}
    for stage in STAGES[started:stopped + 1]:
        print(f"[DBpedia] Starting {stage}", flush=True)
        if stage == "people":
            summary = people.run(type_dump, processed / DIRS[stage])
        elif stage == "relations":
            summary = relations.run(
                object_dump, processed / DIRS["people"] / "person_candidates.tsv.gz",
                relation_policy, processed / DIRS[stage], args.allow_unreviewed_predicates,
            )
        elif stage == "dates":
            summary = dates.run(
                literal_dump, processed / DIRS["relations"] / "social_edges.tsv.gz",
                processed / DIRS[stage],
            )
        else:
            summary = occupations.run(
                type_dump, object_dump,
                processed / DIRS["dates"] / "graph_nodes.tsv.gz",
                processed / DIRS["dates"] / "graph_edges.tsv.gz",
                q_review, processed / DIRS[stage],
            )
        summaries[stage] = summary
        print(f"[DBpedia] Finished {stage}: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    manifest = {
        "raw_dir": str(raw),
        "output_dir": str(processed),
        "rules_dir": str(rules),
        "raw_inputs": {
            path.name: {"compressed_bytes": path.stat().st_size} for path in (type_dump, object_dump, literal_dump)
        },
        "stage_range": [args.from_stage, args.through_stage],
        "stage_summaries": {
            stage: json.loads((processed / DIRS[stage] / "summary.json").read_text(encoding="utf-8"))
            for stage in STAGES[:stopped + 1]
        },
        "period_note": "Birth/death years and source evidence are preserved. This extraction does not assign historical period memberships or construct experiment inputs.",
    }
    write_json(processed / "pipeline_summary.json", manifest)
    print(f"[DBpedia] Outputs: {processed}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"[DBpedia] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
