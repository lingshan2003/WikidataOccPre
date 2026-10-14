#!/usr/bin/env python3
"""Run independent DBpedia BHHT L1 windows on explicitly selected GPUs; never pick idle GPUs."""

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.bhht_l1_pipeline import Pipeline
from DBpedia.grouped_pipeline import STAGES, write_json
from DBpedia.sliding_multi_gpu import select_gpus, run_workers

DEFAULT_CONFIG = "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1.json"


def gpu_worker(gpu, args, jobs, events, log_path):
    os.setsid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else gpu
    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                pipeline = Pipeline(args)
                if args.stage in ("all", "train", "graphmask"):
                    pipeline.gpu_preflight()
                    pipeline.gpu_preflight = lambda: None
                while True:
                    context = jobs.get()
                    if context is None:
                        break
                    events.put({"type": "start", "gpu": gpu, "context": context})
                    pipeline.contexts = [context]
                    pipeline.failures = []
                    try:
                        code = pipeline.run(summarize=False)
                    except Exception as error:
                        code = 1
                        pipeline.failures.append({"context": context, "error": str(error)})
                        print(f"[failed window] {context}: {error}", flush=True)
                    events.put({"type": "done", "gpu": gpu, "context": context,
                                "returncode": code, "failures": pipeline.failures})
            except Exception as error:
                events.put({"type": "worker_error", "gpu": gpu, "error": str(error)})
                raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gpus")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--periods")
    parser.add_argument("--representations")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"))
    parser.add_argument("--cpu-workers", type=int, default=1, help="CPU smoke/diagnostic workers; no CUDA devices are used")
    return parser.parse_args()


def git_value(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def check_active_jobs(pipeline):
    """Refuse same-root orphan trainers; do not stop or signal any process."""
    proc = Path("/proc")
    if not proc.is_dir():
        return
    roots = [str(pipeline.path(pipeline.config[k])) for k in ("period_root", "relation_root", "model_root", "graphmask_root")]
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if any(root in command for root in roots):
                found.append(entry.name)
        except OSError:
            continue
    if found:
        raise ValueError(f"Same-root processes already active: {found}; inspect them before retrying")


def main():
    os.chdir(ROOT)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    ledger = None
    record = None
    code = 1
    try:
        args = parse_args()
        pipeline = Pipeline(args)
        args.device = pipeline.config["device"]
        if pipeline.period_config.get("calendar_layout") != "sliding_windows":
            raise ValueError("Expected sliding windows")
        gpu_stage = args.stage in ("all", "train", "graphmask")
        if args.cpu_workers < 1:
            raise ValueError("--cpu-workers must be positive")
        if args.device == "cpu":
            if args.gpus is not None or args.num_gpus is not None:
                raise ValueError("CPU mode uses --cpu-workers, not --gpus/--num-gpus")
            gpus = [f"cpu_{i}" for i in range(args.cpu_workers)] if gpu_stage else []
        else:
            gpus = select_gpus(args.gpus, args.num_gpus) if gpu_stage else []
        if args.mode == "plan":
            print(f"[plan dispatch] GPU pool={gpus}; {len(pipeline.contexts)} windows")
            pipeline.plan()
            return 0
        with contextlib.ExitStack() as stack:
            # Source lock prevents different experiments from preparing the same base graph.
            source = pipeline.path(pipeline.config["source_data"]).parent
            locks = [source.parent / ("." + source.name + ".dbpedia_bhht_source.lock"),
                     pipeline.path(pipeline.config["model_root"]) / ".grouped_pipeline.lock"]
            for path in locks:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(path.open("a+"))
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f"Another pipeline owns {path}; no work started") from error
            check_active_jobs(pipeline)
            started = datetime.now(timezone.utc)
            run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
            log_dir = pipeline.path(pipeline.config["graphmask_root"]) / "multi_gpu_runs" / run_id
            ledger = log_dir / "experiment_manifest.json"
            source_files = ["DBpedia/bhht_l1_pipeline.py", "DBpedia/sliding_bhht_l1_multi_gpu.py", "DBpedia/run_sliding_bhht_l1_v1.sh",
                            "DBpedia/grouped_pipeline.py", "DBpedia/sliding_multi_gpu.py", "data/prepare.py", "data/extended.py",
                            "data/birth_cohort_artifacts.py", "data/collapse_relations.py", "training/life_periods.py",
                            "training/train.py", "training/graphmask_train.py", "training/graphmask_report.py"]
            source_files += [str(pipeline.config_path.relative_to(ROOT)), pipeline.config["period_config"], pipeline.config["binary_taxonomy"], pipeline.config["multi_group_taxonomy"]]
            for name in source_files:
                target = log_dir / "code_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            record = {"status": "running", "started_at": started.isoformat(), "hostname": socket.gethostname(),
                      "cwd": str(ROOT), "python": sys.executable, "argv": sys.argv,
                      "git_revision": git_value("rev-parse", "HEAD"), "git_status": git_value("status", "--porcelain"),
                      "code_snapshot_files": source_files,
                      "research_question": pipeline.config["research_question"], "config": pipeline.config,
                      "contexts": pipeline.contexts, "gpu_ids": gpus, "seed": pipeline.config["seed"],
                      "budget": {"rgcn_max_epochs": pipeline.config["train"]["epochs"],
                                 "graphmask_epochs_per_layer": pipeline.config["graphmask_train"]["epochs_per_layer"]},
                      "checkpoint_selection": pipeline.config["graphmask_train"]["checkpoint_selection"],
                      "outer_log": os.environ.get("DBPEDIA_BHHT_RUN_LOG"), "control_condition": "unmasked RGCN predictions on the same test roots"}
            write_json(ledger, record)
            config_root = pipeline.path(pipeline.config["model_root"])
            write_json(config_root / "grouped_pipeline_resolved_config.json", pipeline.config)
            if args.stage in ("all", "prepare"):
                pipeline.ensure_source()
            elif args.stage != "summarize":
                pipeline.require_source()
            if gpu_stage:
                write_json(log_dir / "dispatch_plan.json", {"gpu_ids": gpus, "contexts": pipeline.contexts,
                           "representations": pipeline.representations, "stage": args.stage, "config": pipeline.config})
                run_workers(pipeline, gpus, log_dir, worker_target=gpu_worker)
            elif args.stage != "summarize":
                pipeline.run(summarize=False)
            if args.stage != "prepare":
                ready = pipeline.summarize()
                if args.stage in ("all", "graphmask", "summarize") and not ready and not pipeline.failures:
                    pipeline.failures.append({"error": "Incomplete reports"})
                    write_json(pipeline.path(pipeline.config["graphmask_root"]) / "pipeline_failures.json", pipeline.failures)
            code = 1 if pipeline.failures else 0
    except KeyboardInterrupt:
        code = 130
        print("Interrupted; compatible completed stages preserved", file=sys.stderr)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if record is not None:
            record["error"] = str(error)
        code = 1
    finally:
        if ledger is not None:
            record.update(status="complete" if code == 0 else "interrupted" if code == 130 else "failed",
                          exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
            write_json(ledger, record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
