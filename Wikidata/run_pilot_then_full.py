#!/usr/bin/env python3
"""Durably run the approved pilot, validate it, then expand to all 101 windows."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import read_json, write_json
from DBpedia.sliding_multi_gpu import gpu_list
from Wikidata.sliding_multi_gpu import DEFAULT_CONFIG
from Wikidata.validate_pilot import validate


def gpu_snapshot(selected):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return [{"gpu": f[0], "used_MiB": int(f[1]), "total_MiB": int(f[2]), "free_MiB": int(f[2]) - int(f[1]), "utilization_percent": int(f[3])}
            for line in output.splitlines() for f in [[v.strip() for v in line.split(",")]] if f[0] in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=("wikidata", "freebase"), default="wikidata")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=1024)
    args = parser.parse_args()
    selected = gpu_list(args.gpus)
    if any(not g.isdigit() for g in selected):
        raise ValueError("This preflight telemetry entry point uses physical GPU indices")
    os.chdir(ROOT)
    cfg = read_json(args.config)
    if args.dataset == "freebase":
        from scripts.l1_only_sliding import SPECS
        expected = ROOT / "config" / (SPECS["freebase"][0] + ".json")
        if Path(args.config).resolve() != expected.resolve():
            raise ValueError("The Freebase L1-only entry point requires its frozen v4 config")
    model_root = ROOT / cfg["model_root"]
    model_root.mkdir(parents=True, exist_ok=True)
    with (model_root / ".pilot_then_full.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another pilot/full orchestration is active")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        directory = ROOT / cfg["graphmask_root"] / "orchestration_runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        record = {"status": "running", "phase": "pilot", "hostname": socket.gethostname(),
                  "started_at": datetime.now(timezone.utc).isoformat(), "gpu_ids": selected,
                  "argv": sys.argv, "config": cfg, "dataset": args.dataset,
                  "pilot_periods": ["center_1900", "center_1950", "center_2000"],
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        status_file = directory / "status.json"
        write_json(status_file, record)
        (directory / "command.sh").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
        print(f"[orchestration] status={status_file}", flush=True)
        stop = threading.Event()
        minimum_free = {g: None for g in selected}
        telemetry_errors = []

        def sample():
            with (directory / "gpu_telemetry.jsonl").open("a", buffering=1) as f:
                while not stop.is_set():
                    try:
                        rows = gpu_snapshot(selected)
                        if len(rows) != len(selected):
                            raise ValueError("Missing approved GPU in telemetry")
                        for row in rows:
                            g = row["gpu"]
                            minimum_free[g] = row["free_MiB"] if minimum_free[g] is None else min(minimum_free[g], row["free_MiB"])
                        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "phase": record["phase"], "gpus": rows}) + "\n")
                    except Exception as error:
                        telemetry_errors.append(str(error))
                        f.write(json.dumps({"error": str(error)}) + "\n")
                    stop.wait(5)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        child = None
        exit_code = 1

        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            for phase, periods in (("pilot", "center_1900,center_1950,center_2000"), ("full", None)):
                record["phase"] = phase
                write_json(status_file, record)
                command = ["bash", "Wikidata/run_sliding_multi_gpu.sh", "run", "all", "--config", args.config, "--gpus", args.gpus]
                if args.dataset == "freebase":
                    command = ["bash", "Freebase/run_sliding_l1_only.sh", "run", "all", "--gpus", args.gpus]
                if periods:
                    command += ["--periods", periods]
                print("[orchestration] " + shlex.join(command), flush=True)
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, start_new_session=True)
                with (directory / f"{phase}.log").open("w", buffering=1) as log:
                    for line in child.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                code = child.wait()
                child = None
                (directory / f"{phase}.exit_code").write_text(str(code) + "\n")
                if code:
                    raise RuntimeError(f"{phase} failed with exit {code}; full expansion stopped")
                if phase == "pilot":
                    record["phase"] = "validate_pilot"
                    write_json(status_file, record)
                    checked = validate(args.config, periods, args.dataset)
                    checked["pilot_minimum_free_MiB"] = dict(minimum_free)
                    checked["gpu_telemetry_errors"] = list(telemetry_errors)
                    write_json(directory / "pilot_validation.json", checked)
                    if telemetry_errors:
                        raise RuntimeError("GPU telemetry was incomplete; full expansion stopped")
                    if any(value is None or value < args.minimum_free_mib for value in minimum_free.values()):
                        raise RuntimeError("Pilot completed but GPU free memory fell below the configured reserve; full expansion stopped")
                    current = gpu_snapshot(selected)
                    if len(current) != len(selected) or any(r["free_MiB"] < args.minimum_free_mib for r in current):
                        raise RuntimeError("GPU memory availability changed; full expansion stopped")
                    print("[pilot passed] all three reports and protocols valid; expanding to 101 windows", flush=True)
            record["status"] = "complete"
            exit_code = 0
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            exit_code = 130
        except Exception as error:
            record.update(status="failed", error=str(error))
            print(f"[orchestration failed] {error}", file=sys.stderr, flush=True)
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            stop.set()
            thread.join(timeout=10)
            record.update(exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat(), minimum_free_MiB=minimum_free)
            write_json(status_file, record)
            (directory / "exit_code").write_text(str(exit_code) + "\n")
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
