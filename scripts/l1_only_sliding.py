#!/usr/bin/env python3
"""Frozen L1-only protocol guard and logged launcher for existing window runners.

Plan is read-only and requires neither Torch nor a GPU. Run retains the existing
source/stage validation, per-window dispatch, locks and compatible-stage reuse.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import STAGES, read_json, write_json
from DBpedia.sliding_multi_gpu import gpu_list

SPECS = {
    "wikidata": ("wikidata_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v2", "Wikidata.grouped_pipeline", "Wikidata/sliding_multi_gpu.py", 7),
    "freebase": ("freebase_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v4", "Freebase.grouped_pipeline", "Freebase/sliding_multi_gpu.py", 5),
    "dbpedia": ("dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_merged7_alive2026_v2", "DBpedia.bhht_l1_pipeline", "DBpedia/sliding_bhht_l1_multi_gpu.py", 7),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=SPECS)
    p.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    p.add_argument("stage", choices=STAGES, nargs="?", default="all")
    p.add_argument("--gpus", help="Explicit physical GPU IDs, e.g. 1,2,4")
    p.add_argument("--num-gpus", type=int, help="First N cards from the explicit pool")
    p.add_argument("--periods", help="Comma-separated center_YYYY IDs, or all")
    p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    return p.parse_args(argv)


def make_pipeline(args):
    name, module, _, groups = SPECS[args.dataset]
    args.config = str(ROOT / "config" / (name + ".json"))
    args.representations, args.include_full = None, False
    expected = read_json(args.config)
    pipeline = importlib.import_module(module).Pipeline(args)
    c = pipeline.config
    # A stale shell environment must not silently reuse old roots or budgets.
    for key, value in expected.items():
        if key != "device" and c.get(key) != value:
            raise ValueError(f"Environment changed frozen setting {key}; unset the corresponding *_GROUP_* override")
    intended = list(pipeline.periods) if args.periods in (None, "all") else args.periods.split(",")
    if pipeline.contexts != intended or pipeline.representations != ["multi_group"]:
        raise ValueError("Environment changed requested periods/representations; unset *_GROUP_PERIODS/REPRESENTATIONS/INCLUDE_FULL")
    if c["train"]["occupation_feature_levels"] != "1" or c["train"]["auxiliary_features"] != "none":
        raise ValueError("Only L1 occupation inputs are allowed; auxiliary features must be none")
    if args.dataset != "freebase" and c["source_prepare"]["target_level"] != 1:
        raise ValueError("The target must be L1")
    if c["seed"] != 42 or c["prepare"]["split_seed"] != 20260814:
        raise ValueError("Frozen training seed=42 and within-window split seed=20260814 required")
    if c["graphmask_train"]["checkpoint_selection"] != "any-stage":
        raise ValueError("Use the confirmed any-stage checkpoint rule")
    if c["train"]["epochs"] != 50 or c["graphmask_train"]["epochs_per_layer"] != 3:
        raise ValueError("Frozen budget: RGCN max 50 epochs, GraphMask 3 epochs per layer")
    calendar = pipeline.period_config
    if calendar.get("birth_only_alive_assumption") != {"born_after": 1920, "alive_through": 2026}:
        raise ValueError("Birth >1920 with no death must extend to 2026, for membership only")
    if calendar.get("calendar_layout") != "sliding_windows" or len(calendar["periods"]) != 101:
        raise ValueError("Expected the 101 annual +/-20-year windows")
    for year, period in zip(range(1900, 2001), calendar["periods"]):
        if (period["id"], period["start"], period["end"]) != (f"center_{year}", year - 20, year + 20):
            raise ValueError("Unexpected sliding calendar")
    if len(pipeline.multi["groups"]) != groups or c["num_bases"]["multi_group"] != groups * 2:
        raise ValueError("Unexpected relation groups or basis count")
    if args.dataset == "dbpedia":
        if set(pipeline.multi["groups"]["influence_succession"]) != {"influenced", "influencedBy", "successor", "predecessor"}:
            raise ValueError("DBpedia must use the original merged Influence / Succession group")
    return pipeline


def child_command(args):
    command = [sys.executable, "-u", str(ROOT / SPECS[args.dataset][2]), "run", args.stage, "--config", args.config, "--device", args.device]
    for key in ("gpus", "num_gpus", "periods"):
        if getattr(args, key) is not None:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return command


def explicit_pool(args):
    value = args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise ValueError("Specify --gpus 1,2,4 or set CUDA_VISIBLE_DEVICES first; automatic card enumeration is disabled")
    selected = gpu_list(value)
    if args.num_gpus is not None:
        if not 1 <= args.num_gpus <= len(selected):
            raise ValueError("--num-gpus must be positive and within the explicitly selected pool")
        selected = selected[:args.num_gpus]
    return selected


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def main(argv=None):
    os.chdir(ROOT)
    args = parse_args(argv)
    pipeline = make_pipeline(args)
    needs_pool = args.stage in ("all", "train", "graphmask") or (args.dataset == "freebase" and args.stage == "collapse")
    pool = None
    if needs_pool and (args.mode == "run" or args.gpus is not None or args.num_gpus is not None or os.environ.get("CUDA_VISIBLE_DEVICES") is not None):
        pool = explicit_pool(args)
        # Normalize before invoking different legacy CLI pool semantics.
        args.gpus, args.num_gpus = ",".join(pool), None
    command = child_command(args)
    if args.mode == "run" and args.device == "cpu":
        raise ValueError("This training launcher requires cuda:0; use CPU only in plan mode")
    print(f"[{args.mode}] {args.dataset}: {len(pipeline.contexts)} windows; L1 -> L1; auxiliary=none; groups={len(pipeline.multi['groups'])}; GPUs={pool}", flush=True)
    print("[command] " + shlex.join(command), flush=True)
    if args.mode == "plan":
        print("[budget] RGCN <=50 epochs (early stop), GraphMask 3 epochs/layer; seed=42, split_seed=20260814; checkpoint=any-stage")
        print("[membership] birth >1920 and missing death -> alive through 2026; windows +/-20 years, annual centers 1900-2000")
        for key in ("source_data", "period_root", "relation_root", "model_root", "graphmask_root"):
            print(f"[{key}] {pipeline.config[key]}")
        if args.stage in ("all", "prepare"):
            print("[source once] " + shlex.join(pipeline.adapter_command()))
            print("[windows] " + shlex.join(pipeline.prepare_command(pipeline.contexts)))
        for stage, cmd in zip(("collapse", "train", "probe", "report"), pipeline.commands(pipeline.contexts[0], "multi_group")):
            if args.stage == "all" or args.stage == stage or (args.stage == "graphmask" and stage in ("probe", "report")):
                print("[first-window stage] " + shlex.join(cmd))
        print("[summary] " + str(pipeline.path(pipeline.config["graphmask_root"]) / "matrix_summary.tsv"))
        return 0
    launch_root = ROOT / "artifacts/l1_only_sliding_launches" / args.dataset
    launch_root.mkdir(parents=True, exist_ok=True)
    with (launch_root / ".launch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another L1-only launch for this dataset is active; no work started") from error
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        output = launch_root / run_id
        output.mkdir()
        sources = ["scripts/l1_only_sliding.py", "scripts/run_l1_only_sliding.sh", SPECS[args.dataset][2],
                   SPECS[args.dataset][1].replace(".", "/") + ".py", "DBpedia/grouped_pipeline.py", "DBpedia/sliding_multi_gpu.py",
                   "Freebase/prepare_graph.py", "Freebase/bhht_labels.py", "Freebase/relation_rules.json",
                   "data/prepare.py", "data/extended.py", "data/birth_cohort_artifacts.py", "data/collapse_relations.py",
                   "training/life_periods.py", "scripts/prepare_life_period_induced_artifacts.py", "models/features.py", "training/train.py", "training/graphmask_train.py", "training/graphmask_report.py",
                   str(Path(args.config).relative_to(ROOT)), pipeline.config["period_config"], pipeline.config["binary_taxonomy"], pipeline.config["multi_group_taxonomy"]]
        for name in set(sources):
            target = output / "code_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        record = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(), hostname=os.uname().nodename,
                      cwd=str(ROOT), python=sys.executable, command=command, launch_argv=sys.argv, git_revision=git("rev-parse", "HEAD"),
                      git_status=git("status", "--porcelain"), config=pipeline.config, contexts=pipeline.contexts,
                      stage=args.stage, gpu_pool=pool, code_snapshot_files=sorted(set(sources)),
                      outer_log=str(output / "full.log"), control="Unmasked RGCN predictions on the same test roots",
                      evaluation=["accuracy", "macro_f1", "prediction_agreement", "layer0_group_retained_share", "layer0_group_retention_rate"],
                      resume="Compatible completed stages only; not optimizer/epoch continuation")
        write_json(output / "experiment_manifest.json", record)
        print(f"[launch record] {output}", flush=True)
        code, child = 1, None
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            with (output / "full.log").open("w", buffering=1) as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = child.wait()
        except KeyboardInterrupt:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait()
            code = 130
        except Exception as error:
            record["error"] = str(error)
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            raise
        finally:
            (output / "exit_code").write_text(str(code) + "\n")
            record.update(status="requested_stage_complete" if code == 0 else "interrupted" if code == 130 else "failed",
                          exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "experiment_manifest.json", record)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
