#!/usr/bin/env python3
"""Verify the two cached windows, preserve summaries and choose available GPUs."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.l1_only_sliding import make_pipeline, parse_args
from DBpedia.grouped_pipeline import MODEL_FILES, PROBE_FILES, REPORT_FILES, write_json
from Freebase.sliding_multi_gpu import output_locks
from Freebase.repeat_two_windows import gpu_snapshot


def main():
    os.chdir(ROOT)
    p = make_pipeline(parse_args(["freebase","plan","all","--seed","43","--periods","center_1940,center_1942"]))
    output = ROOT / "artifacts/freebase_seed43_full_20261030"
    output.mkdir(parents=True,exist_ok=True)
    with output_locks(p):
        for context in p.contexts:
            p.validate_period(context)
            artifact,model,mask = p.directories(context,"multi_group")
            commands = p.commands(context,"multi_group")
            data=artifact/"graph_data.pt"
            p.require_completed("collapse",commands[0],artifact,p.collapse_outputs("multi_group"),p.collapse_inputs(context),p.collapse_settings("multi_group"))
            p.require_completed("train",commands[1],model,MODEL_FILES,[data,artifact/"collapsed_tie_taxonomy.json"])
            p.require_completed("probe",commands[2],mask,PROBE_FILES,[data,model/"best_model.pt"])
            p.require_completed("report",commands[3],mask/"test_report",REPORT_FILES,[data,model/"best_model.pt",mask/"graphmask_probe.pt",artifact/"nodes.csv"])
            with (mask/"test_report/relations_base.csv").open() as f:
                row=next(r for r in csv.DictReader(f) if r["layer"]=="0" and r["relation"]=="intimate_partnership")
            print(f"[compatible cache] {context}: seed43; Intimate partnership={100*float(row['retained_edge_share']):.4f}%",flush=True)
        real=p.path(p.config["graphmask_root"])
        backup=real/"before_101_window_expansion_20261030"
        backup.mkdir(exist_ok=True)
        sources=[(real/name,name) for name in ("matrix_summary.json","matrix_summary.tsv","relation_group_summary.json","relation_group_summary.tsv","pipeline_failures.json")]
        sources += [(p.path(p.config["model_root"])/"grouped_pipeline_resolved_config.json","model_resolved_config.json"),
                    (p.path(p.config["expanded_from"]),"original_two_window_experiment.json")]
        for source,name in sources:
            if source.exists() and not (backup/name).exists(): shutil.copy2(source,backup/name)
        entry=ROOT/p.config["result_entry"]
        if entry.exists() or entry.is_symlink():
            if entry.resolve()!=real.resolve(): raise ValueError("Result entry already points elsewhere")
        else: entry.symlink_to(real.resolve(),target_is_directory=True)
    pool=[]
    for attempt in range(10):
        snapshot=gpu_snapshot()
        pool=[g for g,r in snapshot.items() if g in ("0","1","2","3","4","5") and r["utilization"]<60 and r["total_mib"]-r["used_mib"]>=8000]
        pool.sort(key=lambda g:(snapshot[g]["utilization"],snapshot[g]["used_mib"]))
        pool=pool[:4]
        if pool: break
        time.sleep(2)
    if not pool: raise ValueError("No available GPU; no training started")
    (output/"gpu_pool.txt").write_text(",".join(pool)+"\n")
    write_json(output/"preflight.json",dict(hostname=os.uname().nodename,cwd=str(ROOT),python=sys.executable,
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),compatible_contexts=p.contexts,
        gpu_pool=pool,gpu_snapshot=snapshot,result_entry=str(entry),real_report_root=str(real),backup=str(backup)))
    print(f"[available GPUs] {pool}; result entry={entry}",flush=True)


if __name__=="__main__": main()
