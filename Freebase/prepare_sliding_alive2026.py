#!/usr/bin/env python3
"""Construct and audit Freebase life-extension windows; never train models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline as GroupedPipeline, write_json
from Freebase.grouped_pipeline import Pipeline, parse_args as pipeline_args
from Freebase.sliding_multi_gpu import output_locks, prepare_shared_source, estimate_window_nodes

DEFAULT_CONFIG = "config/freebase_grouped_sliding_1900_2000_pm20_step1_born_ge1920_alive2026_v3.json"
RULE = {"born_after": 1919, "alive_through": 2026}


def validate_protocol(pipeline):
    if pipeline.period_config.get("calendar_layout") != "sliding_windows":
        raise ValueError("Expected sliding life windows")
    if pipeline.period_config.get("birth_only_alive_assumption") != RULE:
        raise ValueError("This constructor requires birth >=1920 and effective life end 2026")
    grid = [(p["id"], p.get("start"), p.get("end")) for p in pipeline.period_config["periods"]]
    if grid != [(f"center_{year}", year - 20, year + 20) for year in range(1900, 2001)]:
        raise ValueError("This protocol requires 101 centers 1900-2000 with inclusive +/-20-year windows")
    if "full" in pipeline.contexts or pipeline.representations != ["multi_group"]:
        raise ValueError("This constructor supports period windows and multi_group only")
    baseline = json.loads((ROOT / "config/freebase_grouped_sliding_1900_2000_pm20_step1_v2.json").read_text())
    for key in ("period_root", "relation_root", "model_root", "graphmask_root"):
        if pipeline.path(pipeline.config[key]) == pipeline.path(baseline[key]):
            raise ValueError(f"{key} points to the old v2 experiment; select a separate v3 root")


def validate_missing_deaths(pipeline):
    """A normalized blank caused by conflicting/future dates is not missing data."""
    source = pipeline.path(pipeline.config["source_data"]).parent
    with (source / "date_audit.tsv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    states = {r["person_name"]: r["death_status"] for r in records}
    if len(states) != len(records):
        raise ValueError("Duplicate person in source date audit")
    count, people = 0, set()
    with (source / "nodes.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["node_id"]
            if name in people or name not in states:
                raise ValueError("Source node/date-audit person identities disagree")
            people.add(name)
            if row["birth_year"] and int(row["birth_year"]) >= 1920 and not row["death_year"]:
                if states[name] != "missing":
                    raise ValueError(f"Cannot assume life end for {name!r}: death status is {states[name]!r}")
                count += 1
    if people != set(states):
        raise ValueError("Source node/date-audit person sets disagree")
    return count


def topology_comparison(pipeline):
    """Compare graph topology on the identical full source, without any training."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from DBpedia.audit_sliding_window_artifacts import interval_mask

    source = pipeline.path(pipeline.config["source_data"]).parent
    nodes = pd.read_csv(source / "nodes.csv", usecols=["birth_year", "death_year"])
    edges = pd.read_csv(source / "edges.csv", usecols=["source_id", "target_id"])
    birth, death = (pd.to_numeric(nodes[k], errors="coerce").to_numpy(float) for k in ("birth_year", "death_year"))
    s, t = edges.source_id.to_numpy(), edges.target_id.to_numpy()
    estimates = estimate_window_nodes(pipeline)

    def statistics(mask):
        keep = mask[s] & mask[t]
        selected = np.flatnonzero(mask)
        local = np.full(len(nodes), -1, dtype=np.int64)
        local[selected] = np.arange(len(selected))
        graph = csr_matrix((np.ones(int(keep.sum()), dtype=bool), (local[s[keep]], local[t[keep]])),
                           shape=(len(selected), len(selected)))
        count, labels = connected_components(graph, directed=False)
        incident = np.zeros(len(nodes), dtype=bool)
        incident[s[keep]], incident[t[keep]] = True, True
        isolated = mask & ~incident
        largest = int(np.bincount(labels).max()) if len(selected) else 0
        return {
            "nodes": len(selected), "directed_edges": int(keep.sum()),
            "isolated_nodes": int(isolated.sum()),
            "isolated_fraction": float(isolated.sum() / len(selected)) if len(selected) else None,
            "components": int(count), "largest_component_nodes": largest,
            "largest_component_fraction": largest / len(selected) if len(selected) else None,
        }, isolated

    rows = []
    for context in pipeline.contexts:
        period = pipeline.periods[context]
        old_mask = interval_mask(birth, death, period["start"], period["end"])
        new_mask = interval_mask(birth, death, period["start"], period["end"], RULE)
        old, old_isolated = statistics(old_mask)
        new, new_isolated = statistics(new_mask)
        if estimates[context] != new["nodes"]:
            raise ValueError(f"Scheduling estimate disagrees with new membership: {context}")
        rows.append({
            "context": context, "start": period["start"], "end": period["end"],
            **{"old_" + key: value for key, value in old.items()},
            **{"new_" + key: value for key, value in new.items()},
            "added_nodes": int((new_mask & ~old_mask).sum()),
            "previously_isolated_now_connected": int((old_isolated & ~new_isolated).sum()),
            "selected_imputed_nodes": int((new_mask & np.isfinite(birth) & ~np.isfinite(death) & (birth >= 1920)).sum()),
        })
    return rows


def code_version():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = ["Freebase", "DBpedia/grouped_pipeline.py", "DBpedia/audit_sliding_window_artifacts.py",
             "data/birth_cohort_artifacts.py", "training/life_periods.py", "config"]
    deployed = [ROOT / p for p in (
        "Freebase/prepare_sliding_alive2026.py", "Freebase/configure_sliding_windows.py",
        "Freebase/prepare_sliding_alive2026.sh", "Freebase/sliding_multi_gpu.py",
        "Freebase/grouped_pipeline.py", "Freebase/prepare_graph.py", "Freebase/bhht_labels.py",
        "data/birth_cohort_artifacts.py", "data/prepare.py", "data/extended.py", "training/life_periods.py",
        "DBpedia/audit_sliding_window_artifacts.py", "DBpedia/grouped_pipeline.py")]
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short", "--", *paths),
            "deployed_files": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in deployed]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--periods", default="all", help="all or comma-separated center_YYYY IDs")
    return parser.parse_args(argv)


def main(argv=None):
    os.chdir(ROOT)
    manifest_path, record = None, None
    try:
        args = parse_args(argv)
        pipeline = Pipeline(pipeline_args([args.mode, "prepare", "--config", args.config,
                                            "--periods", args.periods, "--device", "cpu"]))
        validate_protocol(pipeline)
        if args.mode == "plan":
            print(f"[construction plan] {len(pipeline.contexts)} windows; no model training")
            print("[shared source] " + shlex.join(pipeline.adapter_command()))
            print("[prepare] " + shlex.join(pipeline.prepare_command(pipeline.contexts)))
            for context in pipeline.contexts:
                print(f"[collapse] {context}/multi_group\n  {shlex.join(pipeline.commands(context, 'multi_group')[0])}")
            print("[audit] exact graph membership, induced edges, raw dates, features, splits and old/new topology; no training")
            return 0
        with output_locks(pipeline):
            output = pipeline.path(pipeline.config["period_root"])
            output.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}"
            manifest_path = output / "construction_runs" / run_id / "manifest.json"
            free = shutil.disk_usage(output).free
            record = {
                "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "research_question": "Does extending missing-death lives born >=1920 through 2026 reduce window isolation?",
                "control": "Original known-endpoint policy on the same v2 source and identical windows",
                "config": pipeline.config, "config_path": str(pipeline.config_path),
                "period_config": pipeline.period_config, "contexts": pipeline.contexts,
                "seed": pipeline.config["prepare"]["split_seed"], "data_split": pipeline.config["prepare"],
                "training_status": "not_started", "training_budget_executed": 0,
                "configured_training_budget": {"rgcn_epochs_max": pipeline.config["train"]["epochs"],
                    "graphmask_epochs_per_layer": pipeline.config["graphmask_train"]["epochs_per_layer"],
                    "note": "Existing settings recorded for traceability; construction does not run them"},
                "checkpoint_selection": pipeline.config["graphmask_train"].get("checkpoint_selection", "any-stage"),
                "code_version": code_version(), "host": socket.gethostname(), "working_directory": str(ROOT),
                "python": sys.executable, "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) if argv is None else
                           shlex.join([sys.executable, str(Path(__file__).resolve()), *argv]),
                "log_path": os.environ.get("FREEBASE_CONSTRUCTION_LOG"), "disk_free_bytes_at_start": free,
                "evaluation": ["nodes", "edges", "isolated_nodes_and_fraction", "largest_connected_component", "exact_artifact_integrity"],
                "result_directory": str(manifest_path.parent / "audit"),
            }
            write_json(manifest_path, record)
            # Preserve the actual construction code even when deployed from a
            # dirty checkout, without hashing datasets or model checkpoints.
            for deployed in record["code_version"].get("deployed_files", []):
                path = Path(deployed["path"])
                snapshot = manifest_path.parent / "code_snapshot" / path.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, snapshot)
            print(f"[construction] host={record['host']} cwd={ROOT} python={sys.executable}; windows={len(pipeline.contexts)}; training=not_started", flush=True)
            dependencies = {}
            for name in ("numpy", "pandas", "scipy", "torch", "torch_geometric"):
                module = importlib.import_module(name)
                dependencies[name] = getattr(module, "__version__", "available")
            record["dependencies"] = dependencies
            write_json(manifest_path, record)
            prepare_shared_source(pipeline)
            eligible = validate_missing_deaths(pipeline)
            source = pipeline.path(pipeline.config["source_data"])
            input_bytes = sum(p.stat().st_size for p in (source, source.parent / "nodes.csv", source.parent / "edges.csv"))
            budget = input_bytes * len(pipeline.contexts) * 2 + 512 * 1024**2
            free = shutil.disk_usage(output).free
            minimum = input_bytes * 2 + 512 * 1024**2
            record.update(source_nodes_with_imputed_death=eligible, estimated_disk_budget_bytes=budget,
                          minimum_working_disk_bytes=minimum)
            write_json(manifest_path, record)
            print(f"[disk] free={free/1024**3:.2f} GiB; conservative full-batch estimate={budget/1024**3:.2f} GiB", flush=True)
            if free < minimum:
                raise ValueError(f"Insufficient working space: free={free/1024**3:.2f} GiB, minimum={minimum/1024**3:.2f} GiB")
            print(f"[policy] birth >=1920, no observed death: {eligible} source people; membership life end=2026", flush=True)
            GroupedPipeline.prepare(pipeline)
            pipeline.args.stage = "collapse"
            if pipeline.run(summarize=False):
                raise ValueError(f"Relation grouping failed: {pipeline.failures}")
            from DBpedia.audit_sliding_window_artifacts import audit
            checked_pipeline, result = audit(str(pipeline.config_path), args.periods, pipeline_class=Pipeline, device="cpu")
            comparisons = topology_comparison(checked_pipeline)
            audit_root = manifest_path.parent / "audit"
            write_json(audit_root / "graph_audit.json", result)
            write_json(audit_root / "topology_comparison.json", comparisons)
            for name, rows in (("window_audit", result["windows"]), ("checks", result["checks"]),
                               ("group_counts", result["group_counts"]), ("topology_comparison", comparisons)):
                if rows:
                    fields = list(dict.fromkeys(key for row in rows for key in row))
                    path = audit_root / (name + ".tsv")
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                        writer.writeheader()
                        writer.writerows(rows)
            if not result["passed"]:
                raise ValueError(f"Graph audit failed; inspect {audit_root / 'graph_audit.json'}")
            record.update(status="complete", windows_completed=len(comparisons), audit_passed=True)
            for row in comparisons:
                print(f"[topology] {row['context']}: nodes {row['old_nodes']}->{row['new_nodes']}; isolated "
                      f"{row['old_isolated_nodes']}->{row['new_isolated_nodes']}; "
                      f"rate {row['old_isolated_fraction']:.2%}->{row['new_isolated_fraction']:.2%}", flush=True)
            print(f"[complete construction] {len(comparisons)} windows; audit={audit_root}; no models trained", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        if record is not None:
            record.update(status="interrupted" if code == 130 else "failed", error=str(error))
        print(f"ERROR: {error or 'Interrupted'}", file=sys.stderr, flush=True)
        return code
    finally:
        if manifest_path is not None and record is not None:
            record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), exit_code=0 if record["status"] == "complete" else
                          130 if record["status"] == "interrupted" else 1)
            write_json(manifest_path, record)
            if record["status"] == "complete":
                write_json(manifest_path.parents[2] / "latest_complete_construction.json", {
                    "manifest": str(manifest_path), "audit_directory": record["result_directory"],
                    "contexts": record["contexts"], "training_status": "not_started",
                })


if __name__ == "__main__":
    raise SystemExit(main())
