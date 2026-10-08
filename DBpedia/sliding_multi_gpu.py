#!/usr/bin/env python3
"""Dispatch independent sliding windows to one serial worker per selected GPU."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import Pipeline, STAGES, write_json

DEFAULT_CONFIG = "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json"


def gpu_list(value):
    values = [v.strip() for v in value.split(",")]
    if not values or any(not re.fullmatch(r"(?:[0-9]+|GPU-[a-fA-F0-9-]+|MIG-[a-fA-F0-9-]+|MIG-GPU-[a-fA-F0-9-]+/[0-9]+/[0-9]+)", v) for v in values):
        raise ValueError("GPU IDs must be comma-separated nonnegative indices or GPU/MIG UUIDs")
    values = [str(int(v)) if v.isdigit() else v for v in values]
    if len(values) != len(set(values)):
        raise ValueError("GPU IDs must be unique")
    return values


def select_gpus(explicit=None, count=None):
    if count is not None and count < 1:
        raise ValueError("--num-gpus must be positive")
    if explicit is not None:
        selected = gpu_list(explicit)
        if count is not None and count != len(selected):
            raise ValueError("--num-gpus must equal the number of IDs in --gpus")
        return selected
    if count is None:
        raise ValueError("Specify --gpus 4,5 or --num-gpus N")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        available = gpu_list(visible)
    else:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("Cannot enumerate GPUs; specify --gpus or set CUDA_VISIBLE_DEVICES") from error
        available = gpu_list(",".join(result.stdout.split()))
    if count > len(available):
        raise ValueError(f"Requested {count} GPUs, but only {len(available)} are selected/visible")
    return available[:count]


def gpu_worker(gpu, args, jobs, events, log_path):
    # A separate session lets the parent terminate training grandchildren too.
    os.setsid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                pipeline = Pipeline(args)
                if args.stage in ("all", "train", "graphmask"):
                    pipeline.gpu_preflight()
                    # This worker keeps its CUDA context between window jobs.
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
                print(f"[worker failed] GPU {gpu}: {error}", flush=True)
                raise SystemExit(1)


def stop_workers(processes):
    """Stop our worker sessions and their child trainers, including Ctrl-C cleanup."""
    # Successfully joined workers have already waited for all their trainers.
    # Avoid signalling a finished worker's old PID/PGID after it may be reused.
    processes = [p for p in processes if p.pid is not None and (p.is_alive() or p.exitcode != 0)]
    for process in processes:
        if process.pid is None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()  # Startup may precede setsid().
    deadline = time.monotonic() + 5
    for process in processes:
        if process.pid is not None:
            process.join(timeout=max(0, deadline - time.monotonic()))
    # A child trainer may outlive a worker; kill the whole group even after exit.
    for process in processes:
        if process.pid is None:
            continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join(timeout=2)


def run_workers(pipeline, gpus, log_dir, worker_target=gpu_worker):
    ctx = mp.get_context("spawn")
    jobs, events = ctx.Queue(), ctx.Queue()
    selected = list(pipeline.contexts)
    # With fewer windows than GPUs, avoid loading unused workers onto extra cards.
    gpus = gpus[:len(selected)]
    for context in selected:
        jobs.put(context)
    for _ in gpus:
        jobs.put(None)
    processes, finished, active, worker_errors = [], {}, {}, []
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def receive(event):
        kind, gpu = event["type"], event["gpu"]
        if kind == "start":
            active[gpu] = event["context"]
            print(f"[start] GPU {gpu}: {event['context']}", flush=True)
        elif kind == "done":
            finished[event["context"]] = event
            active.pop(gpu, None)
            status = "complete" if event["returncode"] == 0 else "failed"
            print(f"[{status}] GPU {gpu}: {event['context']} ({len(finished)}/{len(selected)})", flush=True)
        elif kind == "worker_error":
            worker_errors.append(event)
            print(f"[worker failed] GPU {gpu}: {event['error']}", flush=True)

    interrupted = True
    try:
        for gpu in gpus:
            log_path = log_dir / f"gpu_{gpu.replace('/', '_')}.log"
            process = ctx.Process(target=worker_target, args=(gpu, pipeline.args, jobs, events, str(log_path)))
            processes.append(process)
            process.start()
            print(f"[worker] GPU {gpu}, PID {process.pid}, log={log_path}", flush=True)
        heartbeat = time.monotonic()
        while any(p.is_alive() for p in processes):
            try:
                receive(events.get(timeout=0.5))
            except queue.Empty:
                pass
            if time.monotonic() - heartbeat >= 30:
                print(f"[progress] {len(finished)}/{len(selected)} finished; active={active}", flush=True)
                heartbeat = time.monotonic()
        for process in processes:
            process.join()
        while True:
            try:
                receive(events.get(timeout=0.2))
            except queue.Empty:
                break
        interrupted = False
    finally:
        # Also cleans up child processes if a worker itself crashed.
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            stop_workers(processes)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        snapshot = {
            "interrupted": interrupted, "gpu_ids": gpus, "contexts": selected,
            "finished": finished, "unfinished": [c for c in selected if c not in finished],
            "worker_errors": worker_errors,
            "worker_exitcodes": {gpu: p.exitcode for gpu, p in zip(gpus, processes)},
        }
        write_json(log_dir / "dispatch_status.json", snapshot)
        # A stopped run can leave many queued jobs without consumers.
        jobs.cancel_join_thread()
        events.cancel_join_thread()
        jobs.close()
        events.close()
    for context in selected:
        event = finished.get(context)
        if event is None:
            pipeline.failures.append({"context": context, "error": "Window unfinished; see multi-GPU worker logs"})
        else:
            pipeline.failures.extend(event["failures"])
            if event["returncode"] and not event["failures"]:
                pipeline.failures.append({"context": context, "error": "Worker reported a failed window"})
    for error in worker_errors:
        pipeline.failures.append({"gpu": error["gpu"], "error": error["error"]})
    for gpu, process in zip(gpus, processes):
        if process.exitcode and not any(e["gpu"] == gpu for e in worker_errors):
            pipeline.failures.append({"gpu": gpu, "error": f"Worker exited {process.exitcode}; inspect its log"})
    return snapshot


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default=os.environ.get("DBPEDIA_SLIDING_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 4,5; overrides CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num-gpus", type=int, help="Use the first N selected/visible GPUs")
    parser.add_argument("--periods", help="all or comma-separated center_YYYY IDs")
    parser.add_argument("--representations", help="multi_group, binary, or binary,multi_group")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0", help="cpu is for local smoke tests only")
    return parser.parse_args()


def main():
    os.chdir(ROOT)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        args = parse_args()
        pipeline = Pipeline(args)
        if pipeline.period_config.get("calendar_layout") != "sliding_windows":
            raise ValueError("This dispatcher requires a sliding-window configuration")
        if args.mode == "run" and args.stage == "summarize":
            gpus = []  # Rebuilding a summary never needs GPU enumeration.
        else:
            gpus = select_gpus(args.gpus, args.num_gpus)
        print(f"[dispatch] {len(pipeline.contexts)} windows x {len(pipeline.representations)} representations; "
              f"GPUs={gpus}; one window at a time per worker, next window assigned dynamically", flush=True)
        if args.mode == "plan":
            pipeline.plan()
            return 0
        lock = pipeline.path(pipeline.config["model_root"]) / ".grouped_pipeline.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another serial or multi-GPU pipeline is using this model root") from error
            config_root = pipeline.path(pipeline.config["graphmask_root"]) if args.stage in ("graphmask", "summarize") else lock.parent
            write_json(config_root / "grouped_pipeline_resolved_config.json", pipeline.config)
            if args.stage != "summarize":
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
                log_dir = pipeline.path(pipeline.config["graphmask_root"]) / "multi_gpu_runs" / run_id
                write_json(log_dir / "dispatch_plan.json", {"gpu_ids": gpus, "contexts": pipeline.contexts,
                           "representations": pipeline.representations, "stage": args.stage, "config": pipeline.config})
                run_workers(pipeline, gpus, log_dir)
            if args.stage != "prepare":
                ready = pipeline.summarize()
                if args.stage in ("all", "graphmask", "summarize") and not ready and not pipeline.failures:
                    pipeline.failures.append({"error": "The requested matrix has incomplete reports"})
                    write_json(pipeline.path(pipeline.config["graphmask_root"]) / "pipeline_failures.json", pipeline.failures)
            return 1 if pipeline.failures else 0
    except KeyboardInterrupt:
        print("Interrupted; worker/trainer processes stopped. Rerun to reuse compatible completed stages.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
