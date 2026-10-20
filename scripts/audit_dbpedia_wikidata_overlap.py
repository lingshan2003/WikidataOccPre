#!/usr/bin/env python3
"""Cache official owl:sameAs links and count overlap of the actual source nodes.

Queries are read-only, max two concurrent requests. Unmapped/ambiguous records
are reported separately; no name-based matches are promoted to identity links.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ENDPOINT = "https://dbpedia.org/sparql"


def read_nodes(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = {r["node_id"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate node IDs: {path}")
    return result


def iri(uri):
    return "<" + re.sub(r'[<>"{}|^`\\\x00-\x20]', lambda m: f"\\u{ord(m[0]):04X}", uri) + ">"


def fetch_batch(index, uris, cache):
    path = cache / f"batch_{index:04d}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["requested_uris"] != uris or result["endpoint"] != ENDPOINT:
            raise ValueError(f"Incompatible cached query: {path}")
        return result
    query = ('SELECT DISTINCT ?i (STRAFTER(STR(?o), "/entity/") AS ?qid) WHERE { VALUES (?i ?s) { '
             + ' '.join(f'({i} {iri(uri)})' for i, uri in enumerate(uris))
             + ' } ?s <http://www.w3.org/2002/07/owl#sameAs> ?o . '
             + 'FILTER(REGEX(STR(?o), "^https?://(www\\\\.)?wikidata\\\\.org/entity/Q[0-9]+$")) }')
    request = urllib.request.Request(ENDPOINT,
        data=urllib.parse.urlencode({"query": query, "format": "text/tab-separated-values"}).encode(),
        headers={"User-Agent": "AcademicEntityOverlapAudit/1.0", "Accept": "text/tab-separated-values"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8-sig")
                headers = dict(response.headers)
                if any("incomplete" in str(v).lower() or "S1TAT" in str(v) for v in headers.values()):
                    raise ValueError("Endpoint returned an incomplete result warning")
            pairs = []
            table = csv.DictReader(io.StringIO(payload), delimiter="\t")
            if table.fieldnames != ["?i", "?qid"]:
                raise ValueError(f"Unexpected SPARQL TSV columns: {table.fieldnames}")
            for row in table:
                index, qid = int(row["?i"]), row["?qid"]
                if not 0 <= index < len(uris) or not re.fullmatch(r"Q\d+", qid):
                    raise ValueError("Unexpected sameAs result")
                pairs.append([uris[index], qid])
            result = dict(endpoint=ENDPOINT, queried_at=datetime.now(timezone.utc).isoformat(),
                          requested_uris=uris, pairs=pairs, response_headers=headers, query=query)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return result
        except Exception as error:
            print(f"[query error] batch={index} attempt={attempt + 1}: {error}", flush=True)
            if isinstance(error, ValueError) or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbpedia-nodes", type=Path, default=Path("artifacts/dbpedia_bhht_l1_sliding_source_v1/nodes.csv"))
    parser.add_argument("--wikidata-nodes", type=Path, default=Path("artifacts/person_overlap_20261020/wikidata_source_nodes.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/person_overlap_20261020"))
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    db, wd = read_nodes(args.dbpedia_nodes), read_nodes(args.wikidata_nodes)
    out = args.output_dir
    cache = out / "official_mapping_batches"
    cache.mkdir(parents=True, exist_ok=True)
    uris = sorted(db)
    batches = [uris[i:i + args.batch_size] for i in range(0, len(uris), args.batch_size)]
    mapping = {uri: set() for uri in uris}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(fetch_batch, i, batch, cache) for i, batch in enumerate(batches)]
        try:
            for done, job in enumerate(as_completed(jobs), 1):
                for uri, qid in job.result()["pairs"]:
                    mapping[uri].add(qid)
                print(f"[mapping] {done}/{len(batches)} batches", flush=True)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    matched, unmapped, ambiguous, absent = [], [], [], []
    for uri, ids in mapping.items():
        if not ids:
            unmapped.append(dict(dbpedia_uri=uri))
        elif len(ids) > 1:
            ambiguous.append(dict(dbpedia_uri=uri, wikidata_ids="|".join(sorted(ids))))
        else:
            qid = next(iter(ids))
            if qid not in wd:
                absent.append(dict(dbpedia_uri=uri, wikidata_id=qid))
            else:
                a, b = db[uri], wd[qid]
                matched.append(dict(dbpedia_uri=uri, wikidata_id=qid,
                    dbpedia_birth=a["birth_year"], wikidata_birth=b["birth_year"],
                    dbpedia_death=a["death_year"], wikidata_death=b["death_year"],
                    dbpedia_l1=a["occupation_level1"], wikidata_l1=b["occupation_level1"]))
    unique = {r["wikidata_id"] for r in matched}
    all_mapped = {next(iter(ids)) for ids in mapping.values() if len(ids) == 1}
    summary = dict(dbpedia_nodes=len(db), wikidata_nodes=len(wd), queried_dbpedia_nodes=len(mapping),
        dbpedia_nodes_with_unique_official_qid=sum(len(ids) == 1 for ids in mapping.values()),
        dbpedia_nodes_with_no_official_qid=len(unmapped), dbpedia_nodes_with_multiple_official_qids=len(ambiguous),
        matched_dbpedia_nodes=len(matched), matched_unique_wikidata_persons=len(unique),
        duplicate_dbpedia_records_for_same_shared_qid=len(matched) - len(unique),
        matched_dbpedia_node_fraction=len(matched) / len(db), matched_wikidata_person_fraction=len(unique) / len(wd),
        uniquely_mapped_dbpedia_distinct_qids=len(all_mapped),
        mapped_but_absent_from_main=len(absent),
        mapping_source=ENDPOINT, predicate="http://www.w3.org/2002/07/owl#sameAs",
        identity_policy="Exactly one official linked QID per DBpedia URI; intersection with actual main source QIDs; no name matching",
        caveat="Official endpoint mapping as queried in October 2026, not a frozen 2022 mapping snapshot. Missing/ambiguous mappings are unresolved, not proof of different people.",
        source_files={"dbpedia":str(args.dbpedia_nodes.resolve()), "wikidata":str(args.wikidata_nodes.resolve())})
    write_csv(out / "confirmed_overlap_people.csv", matched, ["dbpedia_uri","wikidata_id","dbpedia_birth","wikidata_birth","dbpedia_death","wikidata_death","dbpedia_l1","wikidata_l1"])
    write_csv(out / "no_official_qid.csv", unmapped, ["dbpedia_uri"])
    write_csv(out / "ambiguous_official_qids.csv", ambiguous, ["dbpedia_uri","wikidata_ids"])
    write_csv(out / "mapped_but_absent_from_main.csv", absent, ["dbpedia_uri","wikidata_id"])
    (out / "official_overlap_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
