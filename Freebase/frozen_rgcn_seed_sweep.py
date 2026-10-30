#!/usr/bin/env python3
"""Train only GraphMask on immutable original seed42 RGCNs, 1900–1910."""
import argparse
from datetime import datetime, timezone
import csv
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from DBpedia.grouped_pipeline import MODEL_FILES,PROBE_FILES,REPORT_FILES,read_json,write_json
from Freebase.grouped_pipeline import Pipeline
from Freebase.repeat_two_windows import gpu_snapshot

CONFIG=ROOT/"config/freebase_frozen_rgcn42_gm_seeds43_44_1900_1910_20261030_v1.json"


class FrozenPipeline(Pipeline):
    def directories(self,context,rep):
        artifact,model,mask=super().directories(context,rep)
        return artifact,model,mask.parent/f"seed_{self.config['graphmask_seed']}"
    def commands(self,context,rep):
        collapse,train,probe,report=super().commands(context,rep)
        probe[probe.index("--seed")+1]=str(self.config["graphmask_seed"])
        probe=[sys.executable,str(ROOT/"Freebase/train_frozen_rgcn_graphmask.py"),*probe[3:]]
        return collapse,train,probe,report
    def run_step(self,name,*args,**kwargs):
        if name in ("collapse","train"):
            raise ValueError("Frozen control cannot run graph preparation or RGCN training")
        return super().run_step(name,*args,**kwargs)


def make(config,years,baseline=False):
    args=SimpleNamespace(config=str(config),stage="graphmask",device="cuda:0",periods=",".join(f"center_{y}" for y in years),representations="multi_group",include_full=False)
    p=(Pipeline if baseline else FrozenPipeline)(args)
    expected=read_json(config)
    if any(p.config.get(k)!=v for k,v in expected.items() if k!="device"):
        raise ValueError("Environment changed frozen experiment settings")
    if p.config["seed"]!=42 or p.config["train"]["occupation_feature_levels"]!="1" or p.config["train"]["auxiliary_features"]!="none":
        raise ValueError("Expected frozen original seed42 L1-only RGCN")
    if p.contexts!=[f"center_{y}" for y in years] or p.representations!=["multi_group"]:
        raise ValueError("Environment changed requested contexts")
    return p


def check_model(p,context):
    p.validate_period(context)
    artifact,model,mask=p.directories(context,"multi_group")
    commands=p.commands(context,"multi_group"); data=artifact/"graph_data.pt"
    p.require_completed("collapse",commands[0],artifact,p.collapse_outputs("multi_group"),p.collapse_inputs(context),p.collapse_settings("multi_group"))
    p.require_completed("train",commands[1],model,MODEL_FILES,[data,artifact/"collapsed_tie_taxonomy.json"])
    return artifact,model,mask,commands


def collect(cfg,out):
    years=list(range(cfg["start_year"],cfg["end_year"]+1)); rows=[]; audits=[]
    base=make(ROOT/cfg["baseline_config"],years,True)
    for year in years:
        context=f"center_{year}"; fingerprints=None; observed_counts=None
        for seed in [42,*cfg["graphmask_seeds"]]:
            p=base if seed==42 else make(out/"configs"/f"graphmask_seed{seed}.json",[year])
            artifact,model,mask,commands=check_model(p,context); data=artifact/"graph_data.pt"
            p.require_completed("probe",commands[2],mask,PROBE_FILES,[data,model/"best_model.pt"])
            p.require_completed("report",commands[3],mask/"test_report",REPORT_FILES,[data,model/"best_model.pt",mask/"graphmask_probe.pt",artifact/"nodes.csv"])
            probe=read_json(mask/"manifest.json"); report=read_json(mask/"test_report/manifest.json")
            metrics=read_json(mask/"test_report/test_metrics.json"); val=read_json(mask/"validation.json")
            fp=(probe["source_checkpoint_sha256"],probe["data_sha256"])
            if fingerprints is None: fingerprints=fp
            if fp!=fingerprints or report["source_checkpoint_sha256"]!=fp[0] or report["data_sha256"]!=fp[1]:
                raise ValueError("Seeds did not use exactly the same recorded model/data fingerprint")
            if report["sampling_seed"]!=42 or val["relative_macro_f1_difference"]>p.config["graphmask_train"]["max_relative_macro_f1_diff"]:
                raise ValueError("Report sampling or validation fidelity changed")
            if seed!=42 and probe["training_config"]["validation_sampling_seed"]!=100042:
                raise ValueError("Validation sampling was not held fixed")
            layer=next(r for r in metrics["layers"] if r["layer"]==0)
            with (mask/"test_report/relations_base.csv").open() as f:
                groups=[r for r in csv.DictReader(f) if r["layer"]=="0"]
            counts={g["relation"]:int(g["message_observations"]) for g in groups}
            if observed_counts is None: observed_counts=counts
            if counts!=observed_counts: raise ValueError("Test sampled-message counts differ despite fixed sampler")
            if abs(sum(float(g["retained_edge_share"]) for g in groups)-1)>1e-8: raise ValueError("Group shares do not sum to one")
            selected=probe["selected_checkpoint"]
            for group in groups:
                rows.append(dict(graphmask_seed=seed,year=year,group=group["relation"],retained_share_pct=100*float(group["retained_edge_share"]),
                    group_keep_pct=100*float(group["hard_retention_rate"]),sampled_share_pct=100*int(group["message_observations"])/layer["message_observations"],
                    layer0_keep_pct=100*layer["hard_retention_rate"],original_macro_f1=metrics["original"]["macro_f1"],masked_macro_f1=metrics["masked"]["macro_f1"],
                    prediction_agreement=metrics["prediction_agreement"],layer0_enabled=selected["enabled_layers"][0],checkpoint_epoch=selected["global_epoch"],report_seed=42,model_seed=42))
            audits.append(dict(year=year,graphmask_seed=seed,model=str(model/"best_model.pt"),report=str(mask/"test_report"),
                              checkpoint_sha256_recorded=fp[0],data_sha256_recorded=fp[1],validation_relative_f1_difference=val["relative_macro_f1_difference"]))
    if len(rows)!=11*3*5: raise ValueError("Expected 165 complete group observations")
    with (out/"all_relations.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");w.writeheader();w.writerows(rows)
    write_json(out/"comparison_audit.json",audits)
    print(f"[summary] {len(audits)} complete reports, {len(rows)} group points; frozen model/data and sampling checked",flush=True)


def run(cfg,out):
    years=list(range(cfg["start_year"],cfg["end_year"]+1)); base=make(ROOT/cfg["baseline_config"],years,True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/".sweep.lock").open("a+") as lock,(base.path(base.config["model_root"])/".grouped_pipeline.lock").open("a+") as model_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(model_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context in base.contexts:
            artifact,model,mask,commands=check_model(base,context); data=artifact/"graph_data.pt"
            base.require_completed("probe",commands[2],mask,PROBE_FILES,[data,model/"best_model.pt"])
            base.require_completed("report",commands[3],mask/"test_report",REPORT_FILES,[data,model/"best_model.pt",mask/"graphmask_probe.pt",artifact/"nodes.csv"])
        for seed in cfg["graphmask_seeds"]:
            c=dict(base.config,name=f"{cfg['name']}_gm{seed}",periods=",".join(base.contexts),graphmask_seed=seed,
                   graphmask_root=str((out/f"graphmask_seed{seed}").relative_to(ROOT)))
            c["graphmask_train"]=dict(c["graphmask_train"],validation_sampling_seed=cfg["validation_sampling_seed"])
            p=out/"configs"/f"graphmask_seed{seed}.json"
            if p.exists() and read_json(p)!=c: raise ValueError("Existing experiment config differs")
            if not p.exists(): write_json(p,c)
        pending=[(s,y) for s in cfg["graphmask_seeds"] for y in years]; active={}; jobs=[]
        record=dict(status="running",started_at=datetime.now(timezone.utc).isoformat(),hostname=os.uname().nodename,python=sys.executable,
            cwd=str(ROOT),argv=sys.argv,git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),config=cfg,jobs=jobs,
            control="Original GraphMask42; per-window RGCN42 immutable; no RGCN retraining; validation sampler100042; report sampler42")
        for source in ["Freebase/frozen_rgcn_seed_sweep.py","Freebase/train_frozen_rgcn_graphmask.py","Freebase/plot_frozen_rgcn_graphmask_seeds.py",
                       "training/graphmask_train.py","training/graphmask_report.py","training/graphmask/common.py","training/graphmask/core.py","training/graphmask/adapter.py",str(CONFIG.relative_to(ROOT))]:
            dest=out/"code_snapshot"/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
        write_json(out/"experiment_manifest.json",record)
        def stop(signum,frame): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop); code=1; last_heartbeat=0
        try:
            while pending or active:
                if pending:
                    snapshot=gpu_snapshot()
                    available=[str(g) for g in cfg["candidate_gpus"] if str(g) not in active and snapshot[str(g)]["utilization"]<cfg["maximum_gpu_utilization_exclusive"]
                               and snapshot[str(g)]["total_mib"]-snapshot[str(g)]["used_mib"]>=cfg["minimum_free_mib"]]
                    available.sort(key=lambda g:(snapshot[g]["utilization"],snapshot[g]["used_mib"]))
                    for gpu in available[:cfg["maximum_parallel_jobs"]-len(active)]:
                        if not pending: break
                        seed,year=pending.pop(0); log=out/f"gm{seed}_{year}.log"; stream=log.open("a",buffering=1)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1");env.setdefault("OMP_NUM_THREADS","2");env.setdefault("MKL_NUM_THREADS","2")
                        command=[sys.executable,"-u",str(Path(__file__).resolve()),"worker","--variant",str(out/"configs"/f"graphmask_seed{seed}.json"),"--year",str(year)]
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        job=dict(graphmask_seed=seed,year=year,gpu=gpu,pid=child.pid,command=command,log=str(log),status="running",gpu_at_launch=snapshot[gpu])
                        jobs.append(job);active[gpu]=(child,stream,job);print("[start] "+json.dumps(job),flush=True)
                for gpu,(child,stream,job) in list(active.items()):
                    result=child.poll()
                    if result is None: continue
                    stream.close();job.update(status="complete" if result==0 else "failed",exit_code=result)
                    (out/f"gm{job['graphmask_seed']}_{job['year']}.exit_code").write_text(str(result)+"\n")
                    del active[gpu];print("[finished] "+json.dumps(job),flush=True)
                write_json(out/"experiment_manifest.json",record)
                if time.monotonic()-last_heartbeat>30:
                    print(f"[progress] {sum(j['status']=='complete' for j in jobs)}/22 complete; {len(active)} active; {len(pending)} queued",flush=True);last_heartbeat=time.monotonic()
                if pending or active: time.sleep(5)
            if all(j["exit_code"]==0 for j in jobs) and len(jobs)==22:
                collect(cfg,out); code=0
        except KeyboardInterrupt: code=130
        finally:
            for child,stream,job in active.values():
                if child.poll() is None: os.killpg(child.pid,signal.SIGTERM);child.wait()
                stream.close();job.update(status="interrupted",exit_code=child.returncode)
            record.update(status="complete" if code==0 else "interrupted" if code==130 else "failed",exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
            write_json(out/"experiment_manifest.json",record);(out/"exit_code").write_text(str(code)+"\n")
        return code


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("run","worker","summarize"))
    parser.add_argument("--variant",type=Path);parser.add_argument("--year",type=int,choices=range(1900,1911))
    args=parser.parse_args();cfg=read_json(CONFIG);out=ROOT/cfg["output_root"]
    if args.mode=="worker":
        p=make(args.variant,[args.year]);p.gpu_preflight();p.run_job(f"center_{args.year}","multi_group");return 0
    if args.mode=="summarize": collect(cfg,out);return 0
    return run(cfg,out)


if __name__=="__main__": raise SystemExit(main())
