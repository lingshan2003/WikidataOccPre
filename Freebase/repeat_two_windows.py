#!/usr/bin/env python3
"""Fresh full-pipeline repeats for centers 1940/1942 on immutable v4 graphs."""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import read_json, write_json
from Freebase.grouped_pipeline import Pipeline
from Freebase.sliding_multi_gpu import output_locks, select_gpus

BASE = "config/freebase_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v4.json"
OUT = "runs_graphmask/freebase_l1_only_repeat_1940_1942_20261030_v1"
PERIODS = "center_1940,center_1942"
VARIANTS = ((42, 2), (42, 3), (43, 1))


class ReadOnlyGraphPipeline(Pipeline):
    def run_step(self, name, command, directory, outputs, inputs, settings=None):
        if name == "collapse":
            return self.require_completed(name, command, directory, outputs, inputs, settings)
        return super().run_step(name, command, directory, outputs, inputs, settings)


def pipeline(config, periods=PERIODS):
    args = SimpleNamespace(config=str(config), stage="all", device="cuda:0", periods=periods,
                           representations="multi_group", include_full=False)
    p = ReadOnlyGraphPipeline(args)
    expected = read_json(config)
    if any(p.config.get(k) != v for k, v in expected.items() if k != "device"):
        raise ValueError("Stale FREEBASE environment override changed the experiment")
    if p.contexts != periods.split(",") or p.representations != ["multi_group"]:
        raise ValueError("Unexpected window or representation override")
    if p.config["train"]["occupation_feature_levels"] != "1" or p.config["train"]["auxiliary_features"] != "none":
        raise ValueError("This repeat is for the confirmed L1-only v4 experiment")
    return p


def gpu_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return {s[0]:dict(used_mib=int(s[1]), total_mib=int(s[2]), utilization=int(s[3]))
            for line in raw.splitlines() for s in [[x.strip() for x in line.split(",")]]}


def collect(out):
    rows = []
    for seed, run in ((42, 1), *VARIANTS):
        config = ROOT / BASE if (seed, run) == (42, 1) else out / "configs" / f"seed{seed}_run{run}.json"
        if not config.exists():
            continue
        p = pipeline(config)
        for year in (1940, 1942):
            context = f"center_{year}"
            _, model, mask = p.directories(context, "multi_group")
            row = dict(seed=seed, run=run, year=year, status="pending", share_pct=None, report=str(mask / "test_report"))
            try:
                commands = p.commands(context, "multi_group")
                from DBpedia.grouped_pipeline import MODEL_FILES, PROBE_FILES, REPORT_FILES
                artifact, _, _ = p.directories(context, "multi_group")
                data = artifact / "graph_data.pt"
                p.require_completed("train", commands[1], model, MODEL_FILES, [data, artifact / "collapsed_tie_taxonomy.json"])
                p.require_completed("probe", commands[2], mask, PROBE_FILES, [data, model / "best_model.pt"])
                p.require_completed("report", commands[3], mask / "test_report", REPORT_FILES,
                                    [data, model / "best_model.pt", mask / "graphmask_probe.pt", artifact / "nodes.csv"])
                m = read_json(mask / "test_report/test_metrics.json")
                with (mask / "test_report/relations_base.csv").open() as f:
                    group = next(r for r in csv.DictReader(f) if r["layer"] == "0" and r["relation"] == "intimate_partnership")
                probe = read_json(mask / "manifest.json")
                trained = read_json(model / "metrics.json")
                if set(trained["model_feature_schema"]) != {"occupation_level1"}:
                    raise ValueError("Actual feature schema is not L1-only")
                val = read_json(mask / "validation.json")
                if val["relative_macro_f1_difference"] > p.config["graphmask_train"]["max_relative_macro_f1_diff"]:
                    raise ValueError("Selected checkpoint violates validation fidelity")
                row.update(status="complete", share_pct=100 * float(group["retained_edge_share"]),
                           group_keep_pct=100 * float(group["hard_retention_rate"]),
                           layer0_enabled=probe["selected_checkpoint"]["enabled_layers"][0],
                           selected_epoch=probe["selected_checkpoint"]["global_epoch"],
                           original_macro_f1=m["original"]["macro_f1"], masked_macro_f1=m["masked"]["macro_f1"],
                           prediction_agreement=m["prediction_agreement"], message_observations=group["message_observations"])
            except Exception as error:
                row["error"] = str(error)
            rows.append(row)
    write_json(out / "results.json", rows)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "results.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t"); w.writeheader(); w.writerows(rows)
    lines = ["Layer 0 Intimate partnership占全部保留消息的百分比。seed=42 Run 1是原实验；Run 2/3为独立完整重跑。seed=43只安排Run 1。", ""]
    for seed, runs in ((42, (1,2,3)), (43, (1,))):
        lines += [f"| seed={seed} | 1940 | 1942 |", "|---|---:|---:|"]
        for run in runs:
            cells = []
            for year in (1940,1942):
                r = next((r for r in rows if r["seed"] == seed and r["run"] == run and r["year"] == year), None)
                cells.append(f"{r['share_pct']:.2f}%" if r and r["status"] == "complete" else "待完成")
            lines.append(f"| Run {run} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    return rows


def run(args):
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    base = pipeline(ROOT / BASE)
    requested_pool = select_gpus(args.gpus)
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context((out / ".repeat.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.enter_context(output_locks(base))
        snapshot = gpu_snapshot()
        if set(requested_pool) - set(snapshot):
            raise ValueError("Requested GPU is absent from nvidia-smi")
        pool = [g for g in requested_pool if snapshot[g]["utilization"] < 60
                and snapshot[g]["total_mib"] - snapshot[g]["used_mib"] >= 8000]
        pool = sorted(pool, key=lambda g: (snapshot[g]["utilization"], snapshot[g]["used_mib"]))[:4]
        if len(pool) < 2:
            raise ValueError(f"Need at least two available GPUs for this parallel launch: {snapshot}")
        print(f"[selected GPUs] {pool}; candidate pool={requested_pool}; snapshot={snapshot}",flush=True)
        configs = out / "configs"; configs.mkdir(exist_ok=True)
        for seed, repeat in VARIANTS:
            c = dict(base.config)
            name = f"freebase_l1_only_repeat_1940_1942_seed{seed}_run{repeat}_v1"
            c.update(name=name, seed=seed, periods=PERIODS,
                     model_root=f"runs/{name}", graphmask_root=f"{OUT}/seed{seed}_run{repeat}")
            target = configs / f"seed{seed}_run{repeat}.json"
            if target.exists() and read_json(target) != c:
                raise ValueError(f"Existing repeat config differs: {target}")
            if not target.exists(): write_json(target, c)
            p = pipeline(target)
            write_json(p.path(c["model_root"]) / "grouped_pipeline_resolved_config.json", c)
        record = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                      git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      hostname=os.uname().nodename, python=sys.executable, cwd=str(ROOT), argv=sys.argv,
                      baseline=BASE, windows=[1940,1942], variants=VARIANTS, gpu_pool=pool, candidate_gpu_pool=requested_pool, gpu_preflight=snapshot,
                      control="Original seed42 run reused only as baseline Run1; all six repeats have separate output paths",
                      fixed="Original period graphs, collapsed graphs, labels and split_seed=20260814; original training/GraphMask budgets and any-stage checkpoint rule",
                      changed_seed="Global seed controls RGCN, GraphMask and test neighbor sampling; split seed remains fixed",
                      resume="Only compatible completed stages of the SAME repeat; no reuse across repeats", jobs=[])
        snap = out / "code_snapshot"
        for source in ["Freebase/repeat_two_windows.py","Freebase/grouped_pipeline.py","DBpedia/grouped_pipeline.py","training/train.py","training/graphmask_train.py","training/graphmask_report.py","models/features.py",BASE]:
            dest = snap / source; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/source,dest)
        write_json(out / "experiment_manifest.json", record)
        pending = [(seed, repeat, year) for seed,repeat in VARIANTS for year in (1940,1942)]
        active = {}
        def interrupted(signum, frame): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        code = 1
        try:
            while pending or active:
                for gpu in pool:
                    if gpu in active or not pending: continue
                    seed, repeat, year = pending.pop(0)
                    log = out / f"seed{seed}_run{repeat}_{year}.log"
                    stream = log.open("a",buffering=1)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                    env.setdefault("OMP_NUM_THREADS","2"); env.setdefault("MKL_NUM_THREADS","2")
                    command = [sys.executable,"-u",str(Path(__file__).resolve()),"worker","--config",str(configs/f"seed{seed}_run{repeat}.json"),"--year",str(year)]
                    process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    job = dict(seed=seed,run=repeat,year=year,gpu=gpu,pid=process.pid,command=command,log=str(log),status="running")
                    record["jobs"].append(job); active[gpu]=(process,stream,job)
                    print("[start] " + json.dumps(job),flush=True)
                for gpu,(process,stream,job) in list(active.items()):
                    result = process.poll()
                    if result is None: continue
                    stream.close(); job.update(status="complete" if result==0 else "failed",exit_code=result)
                    (out/f"seed{job['seed']}_run{job['run']}_{job['year']}.exit_code").write_text(str(result)+"\n")
                    print("[finished] " + json.dumps(job),flush=True); del active[gpu]
                write_json(out/"experiment_manifest.json",record)
                if active: time.sleep(2)
            ready = True
            for seed,repeat in VARIANTS:
                p = pipeline(configs/f"seed{seed}_run{repeat}.json")
                ready = p.summarize() and ready
            rows = collect(out)
            code = 0 if ready and all(r["status"]=="complete" for r in rows) and all(j["exit_code"]==0 for j in record["jobs"]) else 1
        except KeyboardInterrupt:
            code = 130
        finally:
            for process,stream,job in active.values():
                if process.poll() is None: os.killpg(process.pid,signal.SIGTERM); process.wait()
                stream.close(); job.update(status="interrupted",exit_code=process.returncode)
            record.update(status="complete" if code==0 else "interrupted" if code==130 else "failed",exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
            write_json(out/"experiment_manifest.json",record); (out/"exit_code").write_text(str(code)+"\n")
        return code


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("run","worker","summarize"))
    p.add_argument("--gpus")
    p.add_argument("--config",type=Path)
    p.add_argument("--year",type=int,choices=(1940,1942))
    args=p.parse_args()
    if args.mode=="worker":
        job=pipeline(args.config,f"center_{args.year}")
        job.gpu_preflight(); job.validate_period(f"center_{args.year}")
        job.run_job(f"center_{args.year}","multi_group")
        return 0
    if args.mode=="summarize":
        rows=collect(ROOT/OUT); print((ROOT/OUT/"results.md").read_text())
        return 0 if all(r["status"]=="complete" for r in rows) else 1
    return run(args)


if __name__=="__main__":
    raise SystemExit(main())
