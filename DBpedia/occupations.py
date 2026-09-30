"""Stage 4: collect direct and type-based career clues and build the graph."""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from common import DBO, DBR, RDF_TYPE, gzip_rows, resource_label, triples, tsv_writer, uri_from_token, write_json


OBJECT_FIELDS = {
    f"<{DBO}occupation>".encode(): "occupation",
    f"<{DBO}profession>".encode(): "profession",
}
SOURCES = ("occupation", "profession", "dbo_role_type", "wikidata_role_type")
ROLE_TYPES = {
    "Actor", "Artist", "Athlete", "Boxer", "ChristianBishop", "Cleric",
    "Coach", "GridironFootballPlayer", "Instrumentalist", "MotorcycleRider",
    "MotorsportRacer", "MusicalArtist", "Politician", "Presenter", "RacingDriver",
    "Scientist", "SportsManager", "WinterSportPlayer", "Wrestler", "Writer",
}
HARD_NONPERSON_TYPES = {
    "Organisation", "Company", "EducationalInstitution", "PoliticalParty", "SportsTeam",
    "Band", "RecordLabel", "Work", "Film", "Album", "Book", "Place", "Country",
    "City", "Settlement", "AdministrativeRegion", "PopulatedPlace",
}
SOFT_PERSON_TYPES = {
    "Person", "Politician", "Athlete", "Artist", "Cleric", "Scientist", "Writer",
    "Actor", "OfficeHolder", "PersonFunction",
}
REVIEW_ONLY_TYPES = {"AcademicSubject", "FieldOfStudy", "Discipline", "Activity", "Sport"}
UNIVERSAL_QIDS = {"Q5", "Q19088", "Q215627", "Q729"}
EVIDENCE_COLUMNS = [
    "person_uri", "candidate_source", "predicate_or_class", "raw_value",
    "readable_value", "source_file", "source_line",
]
CANDIDATE_COLUMNS = [
    "person_uri", "birth_year", "death_year", "primary_source", "source_kinds",
    "wikidata_qids", "primary_raw_values", "all_candidate_raw_values", "composite_uri_flag",
]


def load_q_review(path: Path) -> tuple[dict, dict]:
    all_rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qid = row["qid"]
            if qid in all_rows or row["decision"] not in {"accept", "conditional", "exclude", "unresolved"}:
                raise ValueError(f"Invalid or duplicate Q class decision: {qid}")
            all_rows[qid] = row
    accepted = {qid: row for qid, row in all_rows.items() if row["decision"] == "accept"}
    if not accepted:
        raise ValueError("Q class review contains no accepted classes")
    return all_rows, accepted


def direct_rejection_reason(value: str, target_types: set[str]) -> str:
    if not value.startswith(DBR):
        return "not_dbpedia_resource"
    if "Wiktionary:" in value or "Wikt:" in value:
        return "dictionary_namespace"
    if target_types & HARD_NONPERSON_TYPES and not target_types & SOFT_PERSON_TYPES:
        return "target_typed_org_place_or_work"
    return ""


def run(
    type_dump: Path,
    object_dump: Path,
    date_node_file: Path,
    date_edge_file: Path,
    q_review_file: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_nodes = {row["person_uri"]: row for row in gzip_rows(date_node_file)}
    review, accepted_q = load_q_review(q_review_file)
    if not date_nodes:
        raise ValueError("Date graph has no nodes")

    # First collect direct resource-value facts so the same type-file pass can
    # classify both their targets and the date-graph subjects.
    direct_facts = []
    direct_values = Counter()
    direct_value_targets = set()
    skipped_intermediate = Counter()
    object_rows = object_malformed = 0
    for line_no, subject, predicate, value in triples(object_dump):
        object_rows = line_no
        if subject is None:
            object_malformed += 1
            continue
        field = OBJECT_FIELDS.get(predicate)
        if field is None:
            continue
        uri = uri_from_token(subject)
        if uri not in date_nodes:
            continue
        raw = uri_from_token(value) if value.startswith(b"<") and value.endswith(b">") else value.decode("utf-8", "replace")
        if "__" in raw:
            skipped_intermediate[field] += 1
            continue
        direct_facts.append((uri, field, raw, line_no))
        direct_values[(field, raw)] += 1
        if raw.startswith(DBR):
            direct_value_targets.add(raw)

    target_types = defaultdict(set)
    type_evidence = []
    q_seen_pairs = set()
    q_by_person = defaultdict(set)
    unreviewed_q = Counter()
    type_rows = type_malformed = 0
    for line_no, subject, predicate, obj in triples(type_dump):
        type_rows = line_no
        if subject is None:
            type_malformed += 1
            continue
        if predicate != RDF_TYPE or not obj.startswith(b"<") or not obj.endswith(b">"):
            continue
        uri = uri_from_token(subject)
        class_uri = uri_from_token(obj)
        if uri in direct_value_targets and class_uri.startswith(DBO):
            target_types[uri].add(class_uri[len(DBO):])
        if uri not in date_nodes:
            continue
        if class_uri.startswith(DBO):
            classname = class_uri[len(DBO):]
            if classname in ROLE_TYPES:
                type_evidence.append({
                    "person_uri": uri,
                    "candidate_source": "dbo_role_type",
                    "predicate_or_class": "rdf:type",
                    "raw_value": obj.decode("utf-8", "replace"),
                    "readable_value": classname,
                    "source_file": type_dump.name,
                    "source_line": line_no,
                })
        elif class_uri.startswith("http://www.wikidata.org/entity/"):
            qid = class_uri.rsplit("/", 1)[-1]
            if qid in accepted_q:
                pair = (uri, qid)
                if pair not in q_seen_pairs:
                    q_seen_pairs.add(pair)
                    q_by_person[uri].add(qid)
                    type_evidence.append({
                        "person_uri": uri,
                        "candidate_source": "wikidata_role_type",
                        "predicate_or_class": "rdf:type",
                        "raw_value": obj.decode("utf-8", "replace"),
                        "readable_value": accepted_q[qid]["label_en"],
                        "source_file": type_dump.name,
                        "source_line": line_no,
                    })
            elif qid not in review and qid not in UNIVERSAL_QIDS:
                unreviewed_q[qid] += 1

    with (output_dir / "direct_value_policy.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "field", "raw_value", "facts", "target_dbo_types", "hard_exclude",
            "reason", "review_only_typed_activity_or_subject",
        ])
        rejection_by_value = {}
        for (field, value), n in sorted(direct_values.items()):
            types = target_types.get(value, set())
            reason = direct_rejection_reason(value, types)
            rejection_by_value[(field, value)] = reason
            writer.writerow({
                "field": field,
                "raw_value": value,
                "facts": n,
                "target_dbo_types": "|".join(sorted(types)),
                "hard_exclude": int(bool(reason)),
                "reason": reason,
                "review_only_typed_activity_or_subject": int(bool(types & REVIEW_ONLY_TYPES) and not reason),
            })

    selected_evidence = list(type_evidence)
    rejected_direct = Counter()
    with gzip.open(output_dir / "rejected_direct_evidence.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["person_uri", "field", "raw_value", "source_line", "reason"])
        for uri, field, raw, line_no in direct_facts:
            reason = rejection_by_value[(field, raw)]
            if reason:
                rejected_direct[(field, reason)] += 1
                writer.writerow({
                    "person_uri": uri, "field": field, "raw_value": raw,
                    "source_line": line_no, "reason": reason,
                })
                continue
            selected_evidence.append({
                "person_uri": uri,
                "candidate_source": field,
                "predicate_or_class": field,
                "raw_value": raw,
                "readable_value": resource_label(raw),
                "source_file": object_dump.name,
                "source_line": line_no,
            })

    evidence_by_person = defaultdict(lambda: defaultdict(set))
    evidence_facts = Counter()
    with gzip.open(output_dir / "candidate_evidence.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, EVIDENCE_COLUMNS)
        for row in selected_evidence:
            uri, source = row["person_uri"], row["candidate_source"]
            evidence_by_person[uri][source].add(row["raw_value"])
            evidence_facts[source] += 1
            writer.writerow(row)

    people = set(evidence_by_person)
    direct_people = {
        uri for uri, sources in evidence_by_person.items()
        if sources.keys() & {"occupation", "profession", "dbo_role_type"}
    }
    primary = {uri: next(source for source in SOURCES if evidence_by_person[uri].get(source)) for uri in people}
    with gzip.open(output_dir / "occupation_candidate_nodes.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, CANDIDATE_COLUMNS)
        for uri in sorted(people):
            sources = evidence_by_person[uri]
            writer.writerow({
                "person_uri": uri,
                "birth_year": date_nodes[uri]["birth_year"],
                "death_year": date_nodes[uri]["death_year"],
                "primary_source": primary[uri],
                "source_kinds": "|".join(source for source in SOURCES if sources.get(source)),
                "wikidata_qids": "|".join(sorted(q_by_person.get(uri, ()))),
                "primary_raw_values": "|".join(sorted(sources[primary[uri]])),
                "all_candidate_raw_values": "|".join(sorted(set().union(*sources.values()))),
                "composite_uri_flag": int("__" in uri.rsplit("/", 1)[-1]),
            })

    coverage = Counter()
    group_flow, predicate_flow = defaultdict(Counter), defaultdict(Counter)
    scopes = Counter()
    source_pairs = Counter()
    retained = set()
    direct_graph_edges = 0
    added_graph_edges = 0
    with gzip.open(date_edge_file, "rt", encoding="utf-8", newline="") as source, gzip.open(
        output_dir / "graph_edges.tsv.gz", "wt", encoding="utf-8", newline=""
    ) as destination, gzip.open(
        output_dir / "added_q_graph_edges.tsv.gz", "wt", encoding="utf-8", newline=""
    ) as added_destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = tsv_writer(destination, reader.fieldnames)
        added_writer = tsv_writer(added_destination, reader.fieldnames)
        for row in reader:
            s, o = row["subject_uri"], row["object_uri"]
            count = int(s in people) + int(o in people)
            coverage[count] += 1
            group_flow[row["social_group"]][count] += 1
            predicate_flow[row["predicate_uri"]][count] += 1
            if count != 2:
                continue
            writer.writerow(row)
            retained.update((s, o))
            scopes[row["selection_scope"]] += 1
            source_pairs[tuple(sorted((primary[s], primary[o])))] += 1
            if s in direct_people and o in direct_people:
                direct_graph_edges += 1
            else:
                added_writer.writerow(row)
                added_graph_edges += 1

    node_fields = list(next(iter(date_nodes.values()))) + [
        "primary_source", "source_kinds", "wikidata_qids", "composite_uri_flag",
    ]
    with gzip.open(output_dir / "graph_nodes.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, node_fields)
        for uri in sorted(retained):
            sources = evidence_by_person[uri]
            writer.writerow({
                **date_nodes[uri],
                "primary_source": primary[uri],
                "source_kinds": "|".join(source for source in SOURCES if sources.get(source)),
                "wikidata_qids": "|".join(sorted(q_by_person.get(uri, ()))),
                "composite_uri_flag": int("__" in uri.rsplit("/", 1)[-1]),
            })

    with (output_dir / "relation_flow.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "predicate_uri", "date_first_edges", "neither_endpoint_has_candidate",
            "one_endpoint_has_candidate", "both_endpoints_have_candidate",
        ])
        for predicate, counts in sorted(predicate_flow.items(), key=lambda item: (-sum(item[1].values()), item[0])):
            writer.writerow({
                "predicate_uri": predicate,
                "date_first_edges": sum(counts.values()),
                "neither_endpoint_has_candidate": counts[0],
                "one_endpoint_has_candidate": counts[1],
                "both_endpoints_have_candidate": counts[2],
            })
    with (output_dir / "group_flow.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, [
            "social_group", "date_first_edges", "neither_endpoint_has_candidate",
            "one_endpoint_has_candidate", "both_endpoints_have_candidate",
        ])
        for group, counts in sorted(group_flow.items()):
            writer.writerow({
                "social_group": group,
                "date_first_edges": sum(counts.values()),
                "neither_endpoint_has_candidate": counts[0],
                "one_endpoint_has_candidate": counts[1],
                "both_endpoints_have_candidate": counts[2],
            })
    with (output_dir / "edge_primary_source_pairs.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = tsv_writer(handle, ["subject_object_primary_source_pair_unordered", "edges"])
        for pair, n in source_pairs.most_common():
            writer.writerow({"subject_object_primary_source_pair_unordered": " + ".join(pair), "edges": n})

    result = {
        "stage": "04_occupations",
        "type_dump": str(type_dump),
        "object_dump": str(object_dump),
        "q_review": str(q_review_file),
        "type_rows_scanned": type_rows,
        "type_malformed_rows": type_malformed,
        "object_rows_scanned": object_rows,
        "object_malformed_rows": object_malformed,
        "date_graph_nodes": len(date_nodes),
        "date_graph_edges": sum(coverage.values()),
        "reviewed_q_classes": len(review),
        "accepted_q_classes": len(accepted_q),
        "unreviewed_specific_q_class_facts": sum(unreviewed_q.values()),
        "unreviewed_specific_q_class_count": len(unreviewed_q),
        "skipped_intermediate_direct_facts": dict(skipped_intermediate),
        "rejected_direct_facts": {f"{field}:{reason}": n for (field, reason), n in rejected_direct.items()},
        "candidate_evidence_facts_by_source": dict(evidence_facts),
        "candidate_people_by_source_nonexclusive": {
            source: sum(bool(evidence_by_person[uri].get(source)) for uri in people) for source in SOURCES
        },
        "candidate_people_by_primary_source": dict(Counter(primary.values())),
        "direct_only_candidate_people": len(direct_people),
        "candidate_people": len(people),
        "new_q_only_candidate_people": len(people - direct_people),
        "direct_only_graph_edges": direct_graph_edges,
        "edge_endpoint_coverage": {str(n): coverage[n] for n in range(3)},
        "graph_edges": coverage[2],
        "added_q_graph_edges": added_graph_edges,
        "graph_nodes": len(retained),
        "graph_predicates": sum(bool(counts[2]) for counts in predicate_flow.values()),
        "graph_edges_by_group": {group: counts[2] for group, counts in group_flow.items()},
        "graph_edges_by_selection_scope": dict(scopes),
        "graph_nodes_by_primary_source": dict(Counter(primary[uri] for uri in retained)),
        "occupation_rule": "Direct DBpedia occupation/profession resource values after hard value exclusions, or explicit dbo role type, or reviewed accepted Wikidata Q type. title/office/PersonFunction/contextual fields do not qualify.",
    }
    write_json(output_dir / "summary.json", result)
    return result
