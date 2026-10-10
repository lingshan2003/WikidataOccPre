#!/usr/bin/env python3
"""Run each Freebase window's RGCN then GraphMask on one manually selected GPU."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline as GroupedPipeline, read_json, write_json
from DBpedia.sliding_multi_gpu import gpu_list, run_workers
from Freebase.grouped_pipeline import Pipeline, STAGES

DEFAULT_CONFIG = "config/freebase_grouped_sliding_1900_2000_pm20_step1_v2.json"


def select_gpus(explicit=None, count=None):
    """Never enumerate or claim cards outside the caller's explicit pool."""
    if count is not None and count < 1:
        raise ValueError("--num-gpus must be positive")
    value = explicit if explicit is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise ValueError("Select GPUs first: export CUDA_VISIBLE_DEVICES=4,5; "
                         "or pass --gpus 4,5. Automatic GPU selection is disabled.")
    selected = gpu_list(value)
    if count is not None and count > len(selected):
        raise ValueError(f"Requested {count} GPUs, but only {len(selected)} were selected")
    return selected if count is None else selected[:count]


@contextlib.contextmanager
def output_locks(pipeline):
    """Also honor the serial runner's model lock; hold locks across all workers."""
    roots = [pipeline.path(pipeline.config[k]) for k in
             ("period_root", "relation_root", "model_root", "graphmask_root")]
    locks = [root.parent / ("." + root.name + ".freebase_sliding.lock") for root in roots]
    locks.append(pipeline.path(pipeline.config["model_root"]) / ".grouped_pipeline.lock")
    with contextlib.ExitStack() as stack:
        for path in sorted(set(locks)):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Another serial or multi-GPU pipeline is using this output root: {path}") from error
        yield


def prepare_shared_source(pipeline):
    """One parent initializes the common graph and provenance before workers start."""
    source_root = pipeline.path(pipeline.config["source_data"]).parent
    lock_path = source_root.parent / ("." + source_root.name + ".freebase_source.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another dispatcher is initializing the shared source; retry after it finishes") from error
        command = pipeline.adapter_command()
        print("[shared source] " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if any(context != "full" for context in pipeline.contexts):
            root = pipeline.path(pipeline.config["period_root"])
            manifest = root / "freebase_source_manifest.json"
            contract = pipeline.period_source_contract()
            if manifest.is_file():
                if read_json(manifest) != contract:
                    raise ValueError("Period source files changed; use a new period/output root")
            else:
                if root.exists() and any(root.glob("*/graph_data.pt")):
                    raise ValueError("Existing periods lack Freebase source provenance; use a new period/output root")
                write_json(manifest, contract)


class WindowPipeline(Pipeline):
    def prepare(self):
        # Common files are immutable during dispatch. Each worker writes only
        # its own window, avoiding the adapter and shared-manifest write races.
        manifest = self.path(self.config["period_root"]) / "freebase_source_manifest.json"
        if any(context != "full" for context in self.contexts):
            if not manifest.is_file() or read_json(manifest) != self.period_source_contract():
                raise ValueError("Shared source is not initialized or changed; rerun the dispatcher")
        GroupedPipeline.prepare(self)


def estimate_window_nodes(pipeline):
    """Count interval/window intersections in O(nodes log windows), without Torch."""
    source = pipeline.path(pipeline.config["source_data"]).parent / "nodes.csv"
    if not source.is_file():
        return {}
    periods = sorted(pipeline.periods.values(), key=lambda p: (p["start"], p["end"]))
    starts, ends = [p["start"] for p in periods], [p["end"] for p in periods]
    if any(a > b for a, b in zip(ends, ends[1:])):
        raise ValueError("Largest-first scheduling requires monotonic sliding-window ends")
    delta, total = [0] * (len(periods) + 1), 0
    assumption = getattr(pipeline, "period_config", {}).get("birth_only_alive_assumption")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            birth = int(row["birth_year"]) if row["birth_year"] else None
            death = int(row["death_year"]) if row["death_year"] else None
            if assumption is not None and birth is not None and death is None and birth > assumption["born_after"]:
                death = assumption["alive_through"]
            if birth is None and death is None:
                continue
            lo, hi = (birth, death) if birth is not None and death is not None else (
                (birth, birth) if birth is not None else (death, death))
            if hi < lo:
                continue
            first, stop = bisect_left(ends, lo), bisect_right(starts, hi)
            if first < stop:
                delta[first] += 1
                delta[stop] -= 1
    result, running = {"full": total}, 0
    for index, period in enumerate(periods):
        running += delta[index]
        result[period["id"]] = running
    return {context: result[context] for context in pipeline.contexts}


def gpu_worker(gpu, args, jobs, events, log_path):
    os.setsid()  # The parent can also stop this worker's child trainers.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, str(args.cpu_threads))
    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                pipeline = WindowPipeline(args)
                if args.stage in ("all", "train", "graphmask"):
                    pipeline.gpu_preflight()
                    pipeline.gpu_preflight = lambda: None
                while True:
                    context = jobs.get()
                    if context is None:
                        return
                    events.put({"type": "start", "gpu": gpu, "context": context})
                    pipeline.contexts, pipeline.failures = [context], []
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
                print(f"[worker failed] GPU {gpu}: {error}", flush=True)
                raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default=os.environ.get("FREEBASE_SLIDING_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--gpus", help="Explicit physical GPU IDs; overrides CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-gpus", type=int, help="Use the first N GPUs in the manually selected pool")
    parser.add_argument("--periods", help="all or comma-separated center_YYYY IDs")
    parser.add_argument("--representations", help="multi_group (default), binary, or binary,multi_group")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), help="Logical worker device; cpu is for smoke tests")
    parser.add_argument("--schedule", choices=("largest-first", "chronological"), default="largest-first")
    parser.add_argument("--cpu-threads", type=int, default=2, help="BLAS/OpenMP threads per worker unless already set")
    return parser.parse_args(argv)


def main(argv=None):
    os.chdir(ROOT)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        args = parse_args(argv)
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        needs_pool = args.stage not in ("prepare", "summarize")
        gpus = select_gpus(args.gpus, args.num_gpus) if needs_pool else []
        # Summary contracts must retain the original logical training device,
        # although summarization itself never initializes CUDA.
        args.device = (args.device or os.environ.get("FREEBASE_GROUP_DEVICE") or
                       os.environ.get("FREEBASE_DEVICE") or ("cpu" if args.stage == "prepare" else "cuda:0"))
        if args.device not in ("cuda:0", "cpu"):
            raise ValueError("Workers each see one card; use FREEBASE_GROUP_DEVICE=cuda:0")
        pipeline = Pipeline(args)
        if pipeline.period_config.get("calendar_layout") != "sliding_windows":
            raise ValueError("This dispatcher requires a sliding-window configuration")
        selected = list(pipeline.contexts)
        print(f"[dispatch] {len(selected)} windows x {len(pipeline.representations)} representations; "
              f"GPUs={gpus}; schedule={args.schedule}; each worker finishes RGCN + GraphMask before next window", flush=True)
        if args.mode == "plan":
            pipeline.plan()
            return 0
        with output_locks(pipeline):
            config_root = pipeline.path(pipeline.config["model_root"])
            write_json(config_root / "grouped_pipeline_resolved_config.json", pipeline.config)
            if args.stage in ("all", "prepare"):
                prepare_shared_source(pipeline)
            if args.stage == "prepare":
                GroupedPipeline.prepare(pipeline)
                print(f"[prepared] {len(selected)} window artifacts: {pipeline.path(pipeline.config['period_root'])}", flush=True)
                return 0
            if args.stage != "summarize":
                sizes = estimate_window_nodes(pipeline) if args.schedule == "largest-first" else {}
                if sizes:
                    pipeline.contexts = sorted(selected, key=lambda context: -sizes[context])
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
                log_dir = pipeline.path(pipeline.config["graphmask_root"]) / "multi_gpu_runs" / run_id
                write_json(log_dir / "dispatch_plan.json", {
                    "gpu_ids": gpus, "selected_contexts": selected, "queue_order": pipeline.contexts,
                    "estimated_nodes": sizes, "schedule": args.schedule,
                    "representations": pipeline.representations, "stage": args.stage, "config": pipeline.config,
                })
                run_workers(pipeline, gpus, log_dir, worker_target=gpu_worker)
                pipeline.contexts = selected
            ready = pipeline.summarize()
            if args.stage in ("all", "graphmask", "summarize") and not ready:
                return 1
            return 1 if pipeline.failures else 0
    except KeyboardInterrupt:
        print("[interrupted] Workers and their child trainers stopped; rerun the same command to resume", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
