#!/usr/bin/env python3
"""Preview life-period graphs using occupation-evidence candidates only.

This is a coverage report, not a class mapping or a model artifact. It follows
config/historical_life_periods_v2.json: a complete valid life interval may
overlap several periods; a sole known endpoint belongs only to its own period.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from .common import gzip_rows, require_files, tsv_writer, write_json
else:
    from common import gzip_rows, require_files, tsv_writer, write_json


ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("occupation", "profession", "dbo_role_type", "wikidata_role_type")


def load_periods(path: Path) -> list[dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("membership_rule") != "life_interval_or_known_endpoint_in_period"
        or config.get("partial_date_policy") != "include_known_endpoint_periods"
        or config.get("invalid_interval_policy") != "exclude_if_death_before_birth"
    ):
        raise ValueError("Period config does not use the existing v2 life-period policy")
    periods = config["periods"]
    if not periods or not isinstance(periods, list):
        raise ValueError("Period config has no periods")
    return periods


def memberships(birth: int | None, death: int | None, periods: list[dict]) -> tuple[str, list[str]]:
    if birth is not None and death is not None and death < birth:
        return "death_before_birth", []
    if birth is not None and death is not None:
        status = "both_dates"
    elif birth is not None:
        status = "birth_only"
    elif death is not None:
        status = "death_only"
    else:
        return "both_missing", []

    selected = []
    for period in periods:
        start, end = period.get("start"), period.get("end")
        if status == "both_dates":
            covered = (end is None or birth <= end) and (start is None or death >= start)
        else:
            known = birth if birth is not None else death
            covered = (start is None or known >= start) and (end is None or known <= end)
        if covered:
            selected.append(period["id"])
    return status, selected


def run(processed: Path, config_path: Path, output: Path) -> dict:
    candidate_file = processed / "04_graph/occupation_candidate_nodes.tsv.gz"
    edge_file = processed / "04_graph/graph_edges.tsv.gz"
    require_files(candidate_file, edge_file, config_path)
    periods = load_periods(config_path)
    period_ids = [period["id"] for period in periods]
    if len(set(period_ids)) != len(period_ids):
        raise ValueError("Period IDs are not unique")

    person_periods = {}
    sources = {period_id: Counter() for period_id in period_ids}
    date_statuses = {period_id: Counter() for period_id in period_ids}
    multiple_primary = {period_id: Counter() for period_id in period_ids}
    raw_value_vocab = {period_id: defaultdict(set) for period_id in period_ids}
    candidate_membership_counts = Counter()
    excluded = Counter()
    candidates = 0
    for row in gzip_rows(candidate_file):
        uri = row["person_uri"]
        if uri in person_periods:
            raise ValueError(f"Duplicate occupation candidate: {uri}")
        source = row["primary_source"]
        if source not in SOURCES:
            raise ValueError(f"Unrecognized priority source {source!r}: {uri}")
        birth = int(row["birth_year"]) if row["birth_year"] else None
        death = int(row["death_year"]) if row["death_year"] else None
        status, selected = memberships(birth, death, periods)
        person_periods[uri] = set(selected)
        candidate_membership_counts[len(selected)] += 1
        candidates += 1
        if not selected:
            excluded[status] += 1
        values = row["primary_raw_values"].split("|") if row["primary_raw_values"] else []
        for period_id in selected:
            sources[period_id][source] += 1
            date_statuses[period_id][status] += 1
            multiple_primary[period_id][source] += len(values) > 1
            raw_value_vocab[period_id][source].update(values)

    edges = Counter()
    edge_sources = Counter()
    active = {period_id: set() for period_id in period_ids}
    predicates = {period_id: set() for period_id in period_ids}
    groups = {period_id: Counter() for period_id in period_ids}
    for row in gzip_rows(edge_file):
        subject, object_ = row["subject_uri"], row["object_uri"]
        if subject not in person_periods or object_ not in person_periods:
            raise ValueError("An occupation graph edge has an endpoint outside its candidate table")
        edge_sources["global_edges"] += 1
        if not person_periods[subject] or not person_periods[object_]:
            edge_sources["edge_touching_excluded_candidate"] += 1
        for period_id in person_periods[subject] & person_periods[object_]:
            edges[period_id] += 1
            active[period_id].update((subject, object_))
            predicates[period_id].add(row["predicate_uri"])
            groups[period_id][row["social_group"]] += 1

    output.mkdir(parents=True, exist_ok=True)
    period_rows = []
    source_rows = []
    for period in periods:
        period_id = period["id"]
        selected_nodes = sum(sources[period_id].values())
        active_nodes = len(active[period_id])
        period_rows.append({
            "period_id": period_id,
            "period_label": period["label"],
            "candidate_nodes": selected_nodes,
            "edge_active_nodes": active_nodes,
            "isolated_candidate_nodes": selected_nodes - active_nodes,
            "original_directed_edges": edges[period_id],
            "directed_edges_with_reverse": 2 * edges[period_id],
            "original_predicates": len(predicates[period_id]),
            "both_dates": date_statuses[period_id]["both_dates"],
            "birth_only": date_statuses[period_id]["birth_only"],
            "death_only": date_statuses[period_id]["death_only"],
            **{f"primary_{source}": sources[period_id][source] for source in SOURCES},
            "multiple_primary_values": sum(multiple_primary[period_id].values()),
        })
        for source in SOURCES:
            source_rows.append({
                "period_id": period_id,
                "primary_source": source,
                "candidate_nodes": sources[period_id][source],
                "multiple_primary_values": multiple_primary[period_id][source],
                "distinct_primary_raw_values": len(raw_value_vocab[period_id][source]),
            })

    with (output / "period_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, list(period_rows[0]))
        writer.writerows(period_rows)
    with (output / "period_sources.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, list(source_rows[0]))
        writer.writerows(source_rows)
    with (output / "period_relation_groups.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["period_id", "social_group", "original_directed_edges"])
        for period_id in period_ids:
            for group, count in sorted(groups[period_id].items()):
                writer.writerow({"period_id": period_id, "social_group": group, "original_directed_edges": count})

    result = {
        "candidate_file": str(candidate_file),
        "edge_file": str(edge_file),
        "period_config": str(config_path),
        "source_priority": list(SOURCES),
        "candidate_people": candidates,
        "global_occupation_graph_edges": edge_sources["global_edges"],
        "candidates_excluded_from_all_periods": dict(excluded),
        "candidate_membership_count_distribution": dict(sorted(candidate_membership_counts.items())),
        "global_edges_touching_period_excluded_candidate": edge_sources["edge_touching_excluded_candidate"],
        "periods": period_rows,
        "note": "Candidate nodes have occupation evidence, not necessarily a unique model class. Isolated candidates have no same-period edge to another candidate. Reverse edges are projected counts, not written here.",
    }
    write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "external_data/dbpedia/processed")
    parser.add_argument("--period-config", type=Path, default=ROOT / "config/historical_life_periods_v2.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    processed = args.processed_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else processed / "period_occupation_preview"
    result = run(processed, args.period_config.resolve(), output)
    print(json.dumps(result["periods"], ensure_ascii=False, indent=2))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
