#!/usr/bin/env python3
"""Wait for the approved Wikidata run, then run and validate DBpedia BHHT L1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE = Path("config/dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1.json")
PERIOD_CONFIG_RELATIVE = Path("config/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2.json")
RUNNER_RELATIVE = Path("DBpedia/run_sliding_bhht_l1_v1.sh")
VALIDATOR_RELATIVE = Path("DBpedia/validate_bhht_l1_pilot.py")
DEFAULT_PREDECESSOR_DIR = Path(
    "runs_graphmask/wikidata_grouped_sliding_1900_2000_pm20_step1_alive2026_v1/"
    "multi_gpu_runs/20261014T010316Z_3806542"
)
DEFAULT_PREDECESSOR_EXIT_FILE = Path("artifacts/wikidata_sliding_gpu_migrations/20261014T010230Z/exit_code")
PILOT_CONTEXTS = ("center_1900", "center_1950", "center_2000")
EXPECTED_CONTEXTS = tuple(f"center_{year}" for year in range(1900, 2001))
BHHT_LEVELS = {"Culture", "Discovery/Science", "Leadership", "Sports/Games", "Other"}
REPORT_FILES = (
    "test_metrics.json", "relations_base.csv", "relations_directed.csv",
    "root_top_edges.csv.gz", "manifest.json",
)
SUMMARY_COLUMNS = {"context", "representation", "seed", "status", "classes", "report_dir"}
CODE_PATHS = (
    Path(__file__).resolve(),
    RUNNER_RELATIVE,
    VALIDATOR_RELATIVE,
    Path("DBpedia/sliding_multi_gpu.py"),
    Path("DBpedia/sliding_bhht_l1_multi_gpu.py"),
    Path("DBpedia/bhht_l1_pipeline.py"),
    Path("DBpedia/grouped_pipeline.py"),
    Path("DBpedia/prepare_bhht_l1_v1.py"),
    Path("scripts/prepare_life_period_induced_artifacts.py"),
    Path("data/prepare.py"),
    Path("data/extended.py"),
    Path("data/birth_cohort_artifacts.py"),
    Path("data/collapse_relations.py"),
    Path("training/life_periods.py"),
    Path("training/train.py"),
    Path("training/graphmask_train.py"),
    Path("training/graphmask_report.py"),
    CONFIG_RELATIVE,
    PERIOD_CONFIG_RELATIVE,
    Path("config/dbpedia_tie_taxonomy_v1.json"),
    Path("config/dbpedia_tie_taxonomy_acquired_subgroups_v2.json"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_gpus(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",")]
    if not values or any(not re.fullmatch(r"[0-9]+", part) for part in values):
        raise ValueError("--gpus must contain physical GPU indices such as 1,2,4")
    normalized = [str(int(part)) for part in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("--gpus must not contain duplicate GPU indices")
    return normalized


def parse_gpu_snapshot(output: str, selected: list[str]) -> list[dict]:
    rows = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            continue
        gpu = fields[0]
        if gpu not in selected:
            continue
        try:
            used, total = int(fields[1]), int(fields[2])
            utilization = None if fields[3].upper() in {"N/A", "[N/A]"} else int(fields[3])
        except ValueError as error:
            raise ValueError(f"Invalid nvidia-smi telemetry row for GPU {gpu}: {line}") from error
        rows[gpu] = {"gpu": gpu, "used_MiB": used, "total_MiB": total,
                     "free_MiB": total - used, "utilization_percent": utilization}
    missing = [gpu for gpu in selected if gpu not in rows]
    if missing:
        raise ValueError(f"nvidia-smi did not report selected GPU(s): {', '.join(missing)}")
    return [rows[gpu] for gpu in selected]


def gpu_snapshot(selected: list[str]) -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return parse_gpu_snapshot(result.stdout, selected)


def require_gpu_memory(rows: list[dict], selected: list[str], minimum_free_mib: int) -> None:
    if [row["gpu"] for row in rows] != selected:
        raise ValueError("GPU preflight snapshot does not match the requested cards")
    low = [f"GPU {row['gpu']} has {row['free_MiB']} MiB free" for row in rows
           if row["free_MiB"] < minimum_free_mib]
    if low:
        raise ValueError(f"GPU memory reserve is not available ({minimum_free_mib} MiB required): " + "; ".join(low))


def expected_dispatch_contexts() -> list[str]:
    return list(EXPECTED_CONTEXTS)


def validate_dispatch_status(payload: dict) -> None:
    contexts = expected_dispatch_contexts()
    if payload.get("contexts") != contexts:
        raise ValueError("Predecessor dispatch context list is not the expected 101 annual windows")
    finished = payload.get("finished")
    if not isinstance(finished, dict) or set(finished) != set(contexts) or len(finished) != 101:
        raise ValueError("Predecessor dispatch did not finish all 101 contexts")
    for context in contexts:
        event = finished[context]
        if event.get("context") != context or int(event.get("returncode", -1)) != 0:
            raise ValueError(f"Predecessor window failed: {context}")
        if event.get("failures") != []:
            raise ValueError(f"Predecessor reported window failures: {context}")
    if payload.get("unfinished") != []:
        raise ValueError("Predecessor has unfinished windows")
    if payload.get("worker_errors") != []:
        raise ValueError("Predecessor reported worker errors")
    if payload.get("interrupted") is not False:
        raise ValueError("Predecessor dispatch was interrupted")
    exitcodes = payload.get("worker_exitcodes")
    if not isinstance(exitcodes, dict) or not exitcodes or any(int(code) != 0 for code in exitcodes.values()):
        raise ValueError("Predecessor worker exit codes are missing or nonzero")
    gpu_ids = payload.get("gpu_ids")
    if not isinstance(gpu_ids, list) or set(exitcodes) != set(gpu_ids):
        raise ValueError("Predecessor worker exit codes do not match its dispatched GPUs")


def validate_matrix_summary(path: Path, contexts: tuple[str, ...] | list[str], *,
                            expected_classes: int | None = None,
                            expected_representation: str = "multi_group",
                            newer_than_ns: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing matrix summary: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if not SUMMARY_COLUMNS <= fields:
            raise ValueError(f"Matrix summary schema is missing columns: {sorted(SUMMARY_COLUMNS-fields)}")
        rows = list(reader)
    if newer_than_ns is not None and path.stat().st_mtime_ns < newer_than_ns:
        raise ValueError(f"Matrix summary predates this dispatch and may be stale: {path}")
    wanted = list(contexts)
    if len(rows) != len(wanted) or {row.get("context") for row in rows} != set(wanted):
        raise ValueError(f"Matrix summary must contain exactly {len(wanted)} requested contexts")
    by_context = {row["context"]: row for row in rows}
    if len(by_context) != len(rows):
        raise ValueError("Matrix summary contains duplicate contexts")
    for context in wanted:
        row = by_context[context]
        if row.get("representation") != expected_representation:
            raise ValueError(f"Unexpected representation in matrix summary: {context}")
        if row.get("status") != "complete":
            raise ValueError(f"Matrix report is not complete: {context} ({row.get('status')})")
        try:
            seed = int(row.get("seed", ""))
        except ValueError as error:
            raise ValueError(f"Matrix row has no valid seed: {context}") from error
        if expected_classes is not None:
            try:
                classes = int(row.get("classes", ""))
            except ValueError as error:
                raise ValueError(f"Matrix row has no valid class count: {context}") from error
            if classes != expected_classes:
                raise ValueError(f"Expected {expected_classes} active target classes for {context}, got {classes}")
        report_dir = Path(row.get("report_dir", "")).expanduser()
        if not report_dir.is_absolute():
            report_dir = (ROOT / report_dir).resolve()
        expected_report_dir = path.resolve().parent / context / expected_representation / f"seed_{seed}" / "test_report"
        if report_dir.resolve() != expected_report_dir.resolve():
            raise ValueError(f"Report path does not match the matrix summary schema for {context}")
        missing = [name for name in REPORT_FILES if not (report_dir / name).is_file() or (report_dir / name).stat().st_size == 0]
        if missing:
            raise ValueError(f"Completed report files are missing for {context}: {', '.join(missing)}")
    return [by_context[context] for context in wanted]


def predecessor_summary_path(predecessor_dir: Path) -> Path:
    predecessor_dir = Path(predecessor_dir).resolve()
    if len(predecessor_dir.parents) < 2:
        raise ValueError(f"Unexpected predecessor dispatch directory: {predecessor_dir}")
    return predecessor_dir.parents[1] / "matrix_summary.tsv"


def predecessor_ready(predecessor_dir: Path, exit_file: Path) -> bool:
    """Return false while the predecessor runs; raise for any final failure."""
    predecessor_dir, exit_file = Path(predecessor_dir), Path(exit_file)
    status_file = predecessor_dir / "dispatch_status.json"
    if exit_file.is_file():
        raw = exit_file.read_text(encoding="utf-8").strip()
        try:
            exit_code = int(raw)
        except ValueError as error:
            raise ValueError(f"Predecessor exit file is not an integer: {exit_file}") from error
        if exit_code != 0:
            raise RuntimeError(f"Predecessor exited {exit_code}; DBpedia queue will not start")
    if status_file.is_file():
        status = read_json(status_file)
        validate_dispatch_status(status)
        if not exit_file.is_file():
            return False
    elif not exit_file.is_file():
        return False
    else:
        raise RuntimeError("Predecessor exit file is zero but dispatch_status.json is missing")
    status_mtime = status_file.stat().st_mtime_ns
    validate_matrix_summary(predecessor_summary_path(predecessor_dir), EXPECTED_CONTEXTS,
                            newer_than_ns=status_mtime)
    return True


def expected_period_ids(config_path: Path) -> list[str]:
    payload = read_json(config_path)
    periods = payload.get("periods", [])
    return [period.get("id") for period in periods]


def validate_experiment_config(config_path: Path) -> tuple[dict, dict, dict]:
    config_path = Path(config_path)
    config = read_json(config_path)
    if config.get("name") != "dbpedia_grouped_sliding_1900_2000_pm20_step1_bhht_l1_alive2026_v1":
        raise ValueError("BHHT queue must use the approved DBpedia BHHT L1 experiment config")
    if config.get("include_full") is not False or config.get("periods") != "all":
        raise ValueError("BHHT experiment must contain exactly the 101 sliding windows, without a full-graph run")
    if config.get("representations") != ["multi_group"]:
        raise ValueError("BHHT experiment must use the approved multi_group representation")
    if config.get("period_config") != PERIOD_CONFIG_RELATIVE.as_posix():
        raise ValueError("BHHT experiment must use the approved alive-through-2026 period config")
    if config.get("dataset_manifest") != "external_data/dbpedia/bhht_l1_provisional_v1/manifest.json":
        raise ValueError("BHHT experiment must use the reviewed provisional L1 dataset manifest")
    train = config.get("train", {})
    if train.get("model") != "rgcn" or int(train.get("num_layers", 0)) != 2:
        raise ValueError("BHHT experiment requires the configured two-layer RGCN")
    if str(train.get("occupation_feature_levels", "")) != "1":
        raise ValueError("BHHT experiment must expose only the L1 occupation feature")
    if train.get("auxiliary_features") != "none":
        raise ValueError("BHHT experiment must disable all auxiliary features")
    source_prepare = config.get("source_prepare", {})
    if source_prepare.get("input") != "external_data/dbpedia/bhht_l1_provisional_v1/DBpedia_R_R_extended.csv":
        raise ValueError("BHHT source input path differs from the approved L1 dataset")
    if source_prepare.get("target_level", 1) != 1:
        raise ValueError("BHHT source must predict occupation L1")
    source_data = Path(config.get("source_data", ""))
    if source_data.as_posix() != "artifacts/dbpedia_bhht_l1_sliding_source_v1/graph_data.pt":
        raise ValueError("BHHT source graph path differs from the approved dataset artifact")
    period_config_path = (ROOT / config["period_config"]).resolve()
    period_config = read_json(period_config_path)
    if period_config.get("calendar_layout") != "sliding_windows" or expected_period_ids(period_config_path) != list(EXPECTED_CONTEXTS):
        raise ValueError("BHHT period config must define the 101 annual centers from 1900 through 2000")
    if period_config.get("birth_only_alive_assumption") != {"born_after": 1920, "alive_through": 2026}:
        raise ValueError("BHHT period config must use the approved birth-after-1920 alive-through-2026 rule")
    return config, source_prepare, period_config


def file_snapshot(paths: list[Path]) -> list[dict]:
    snapshot = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        if not path.is_file() or stat.st_size == 0:
            raise ValueError(f"Required input is missing or empty: {path}")
        snapshot.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return snapshot


def code_snapshot(root: Path = ROOT) -> list[dict]:
    entries = []
    for relative in CODE_PATHS:
        path = (root / relative).resolve()
        stat = path.stat()
        entries.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "mode": stat.st_mode})
    return entries


def git_revision(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def check_pin(expected_revision: str, expected_code: list[dict], root: Path = ROOT) -> None:
    if git_revision(root) != expected_revision:
        raise RuntimeError("Git revision changed after queue initialization; refusing to launch another stage")
    if code_snapshot(root) != expected_code:
        raise RuntimeError("Experiment code/config changed after queue initialization; refusing to launch another stage")


def require_static_inputs(config: dict, source_prepare: dict) -> tuple[list[Path], dict]:
    source_data = (ROOT / config["source_data"]).resolve()
    input_table = (ROOT / source_prepare["input"]).resolve()
    dataset_manifest = (ROOT / config.get("dataset_manifest", str(input_table.parent / "manifest.json"))).resolve()
    split_summary = source_data.parent / "split_summary.json"
    paths = [input_table, dataset_manifest,
             ROOT / config["binary_taxonomy"], ROOT / config["multi_group_taxonomy"],
             ROOT / config["period_config"], ROOT / RUNNER_RELATIVE, ROOT / VALIDATOR_RELATIVE]
    snapshot = file_snapshot(paths)
    manifest = read_json(dataset_manifest)
    expected_classes = set(config.get("expected_l1_classes", []))
    if expected_classes != BHHT_LEVELS:
        raise ValueError("Experiment config does not name the five approved BHHT L1 classes")
    if int(manifest.get("level1_classes", -1)) != 5 or set(manifest.get("level1_node_counts", {})) != expected_classes:
        raise ValueError("BHHT dataset manifest must contain the five reviewed L1 training classes")
    expected_outputs = ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json", "relation_stats.csv")
    present_outputs = [name for name in expected_outputs if (source_data.parent / name).exists()]
    prepare_record = source_data.parent / "pipeline_prepare_source.json"
    if present_outputs and not prepare_record.is_file():
        raise ValueError("BHHT source directory has unowned partial outputs; preserve them and inspect before retrying")
    summary = read_json(split_summary) if split_summary.is_file() else None
    if summary is not None and (summary.get("target_column") != "occupation_level1" or int(summary.get("target_classes_retained", -1)) != 5):
        raise ValueError("Prepared BHHT source metadata must contain five occupation L1 classes")
    return paths, {"input_snapshot": snapshot, "dataset_manifest": manifest,
                   "source_split_summary": summary}


def verify_static_inputs(paths: list[Path], original_snapshot: list[dict]) -> None:
    if file_snapshot(paths) != original_snapshot:
        raise RuntimeError("Dataset or configuration inputs changed after queue initialization")


def python_command(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        # Resolving a venv's Python symlink can silently select the base environment.
        candidate = Path(os.path.abspath(candidate))
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"--python is not an executable file: {candidate}")
        return str(candidate)
    found = shutil.which(value)
    if not found:
        raise ValueError(f"Cannot find Python executable: {value}")
    return found


class Telemetry:
    def __init__(self, path: Path, selected: list[str], interval: float = 30.0):
        self.path = Path(path)
        self.selected = selected
        self.interval = interval
        self.phase = "preflight"
        self.stop_event = threading.Event()
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def record(self, rows: list[dict] | None = None, error: str | None = None) -> None:
        with self.lock:
            phase = self.phase
        payload = {"time": utc_now(), "phase": phase}
        if rows is not None:
            payload["gpus"] = rows
        if error is not None:
            payload["error"] = error
            self.errors.append(error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def sample_once(self) -> list[dict]:
        try:
            rows = gpu_snapshot(self.selected)
            self.record(rows=rows)
            return rows
        except Exception as error:
            self.record(error=str(error))
            raise

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.record(rows=gpu_snapshot(self.selected))
            except Exception as error:
                self.record(error=str(error))
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def run_command(command: list[str], *, cwd: Path, env: dict, log_path: Path,
                command_path: Path, exit_path: Path) -> int:
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    print("[command] " + shlex.join(command), flush=True)
    code = 1
    child = None
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=True)
        try:
            assert child.stdout is not None
            for line in child.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            raise
    exit_path.write_text(f"{code}\n", encoding="utf-8")
    return code


def check_plan_output(output: str, expected_contexts: tuple[str, ...]) -> None:
    train_lines = [line for line in output.splitlines() if line.startswith("[train]")]
    if len(train_lines) != len(expected_contexts):
        raise ValueError(f"Runner plan listed {len(train_lines)} training jobs; expected {len(expected_contexts)}")
    planned_contexts = [line.split("]", 1)[1].strip().split("/", 1)[0] for line in train_lines]
    if len(set(planned_contexts)) != len(planned_contexts) or set(planned_contexts) != set(expected_contexts):
        raise ValueError("Runner plan contains an unexpected pilot context set")
    required = ("--occupation-feature-levels 1", "--auxiliary-features none")
    # Plan formatting puts the command on the following line. Include nearby text per job.
    plan_lines = output.splitlines()
    blocks = []
    for index, line in enumerate(plan_lines):
        if line.startswith("[train]"):
            blocks.append("\n".join(plan_lines[index:index + 2]))
    if any(any(option not in block for option in required) for block in blocks):
        raise ValueError("Runner plan does not use L1-only occupation features with no auxiliary features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-dir", type=Path, default=DEFAULT_PREDECESSOR_DIR)
    parser.add_argument("--predecessor-exit-file", type=Path, default=DEFAULT_PREDECESSOR_EXIT_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,4")
    parser.add_argument("--python", required=True, help="Python executable for the experiment environment")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")
    if args.min_free_mib < 1:
        raise ValueError("--min-free-mib must be positive")
    gpus = selected_gpus(args.gpus)
    python_bin = python_command(args.python)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    predecessor_dir = args.predecessor_dir.expanduser()
    if not predecessor_dir.is_absolute():
        predecessor_dir = ROOT / predecessor_dir
    predecessor_dir = predecessor_dir.resolve()
    predecessor_exit = args.predecessor_exit_file.expanduser()
    if not predecessor_exit.is_absolute():
        predecessor_exit = ROOT / predecessor_exit
    predecessor_exit = predecessor_exit.resolve()
    config_path = (ROOT / CONFIG_RELATIVE).resolve()
    config, source_prepare, period_config = validate_experiment_config(config_path)
    input_paths, source_info = require_static_inputs(config, source_prepare)
    initial_code = code_snapshot(ROOT)
    revision = git_revision(ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    lock_path = output_dir / ".queue.lock"
    artifact_lock_path = ROOT / "artifacts" / ".experiment_locks" / "dbpedia_bhht_l1_after_wikidata.lock"
    artifact_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as output_lock, artifact_lock_path.open("a+") as artifact_lock:
        for handle, label in ((output_lock, "this output directory"), (artifact_lock, "this experiment")):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another queue process already holds the lock for {label}") from error
        prior = read_json(status_path) if status_path.is_file() else None
        if prior and prior.get("status") != "interrupted":
            raise RuntimeError(f"Queue output already has terminal status {prior.get('status')}; use a new output directory")
        if not prior and any(path.name != ".queue.lock" for path in output_dir.iterdir()):
            raise RuntimeError(f"Refusing to reuse nonempty queue output directory: {output_dir}")
        if prior:
            expected_identity = (str(config_path), revision, gpus, str(predecessor_dir), str(predecessor_exit))
            actual_identity = (prior.get("config_path"), prior.get("git_revision"), prior.get("gpu_ids"),
                               prior.get("predecessor_dir"), prior.get("predecessor_exit_file"))
            if expected_identity != actual_identity:
                raise RuntimeError("Interrupted queue identity differs from this invocation")
            if code_snapshot(ROOT) != prior.get("code_snapshot"):
                raise RuntimeError("Experiment code/config changed since the interrupted queue")
            if source_info["input_snapshot"] != prior.get("input_snapshot"):
                raise RuntimeError("Experiment data inputs changed since the interrupted queue")
            initial_code = prior["code_snapshot"]
        else:
            prior = {}

        record = {
            **prior,
            "status": "running", "phase": "wait_predecessor", "hostname": socket.gethostname(),
            "started_at": prior.get("started_at", utc_now()), "updated_at": utc_now(),
            "config_path": str(config_path), "config_name": config["name"],
            "config": config, "period_config": period_config,
            "predecessor_dir": str(predecessor_dir), "predecessor_exit_file": str(predecessor_exit),
            "output_dir": str(output_dir), "gpu_ids": gpus, "python": python_bin,
            "poll_seconds": args.poll_seconds, "minimum_free_MiB": args.min_free_mib,
            "git_revision": revision, "code_snapshot": initial_code,
            "input_snapshot": source_info["input_snapshot"],
            "dataset_manifest": source_info["dataset_manifest"],
            "source_split_summary": source_info["source_split_summary"],
        }
        write_json(status_path, record)
        (output_dir / "command.sh").write_text(shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + "\n", encoding="utf-8")
        env = os.environ.copy()
        # The queue runs the confirmed configuration, not inherited shell overrides.
        for key in list(env):
            if key.startswith("DBPEDIA_BHHT_GROUP_"):
                del env[key]
        env["DBPEDIA_PYTHON_BIN"] = python_bin
        env["DBPEDIA_SLIDING_CONFIG"] = str(CONFIG_RELATIVE)
        outer_log_dir = output_dir.parent / (output_dir.name + ".runner_logs")
        env["DBPEDIA_BHHT_RUN_LOG_DIR"] = str(outer_log_dir)

        def terminate(signum, frame):
            raise KeyboardInterrupt

        previous_handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
        telemetry = None
        exit_code = 1
        try:
            print(f"[queue] waiting for predecessor at {predecessor_dir}", flush=True)
            while not predecessor_ready(predecessor_dir, predecessor_exit):
                record.update(phase="wait_predecessor", updated_at=utc_now())
                write_json(status_path, record)
                print(f"[queue] predecessor still running; next check in {args.poll_seconds}s", flush=True)
                time.sleep(args.poll_seconds)
            record.update(phase="preflight", predecessor_ready=True, updated_at=utc_now())
            write_json(status_path, record)
            check_pin(revision, initial_code, ROOT)
            verify_static_inputs(input_paths, source_info["input_snapshot"])
            disk_paths = {"queue_output": output_dir.parent}
            for key in ("source_data", "period_root", "relation_root", "model_root", "graphmask_root"):
                disk_paths[key] = (ROOT / config[key]).resolve().parent if key == "source_data" else (ROOT / config[key]).resolve()
            disk_info = []
            for label, configured_path in disk_paths.items():
                probe = configured_path
                while not probe.exists():
                    probe = probe.parent
                if probe.is_file():
                    probe = probe.parent
                usage = shutil.disk_usage(probe)
                disk_info.append({"area": label, "path": str(probe), "total_bytes": usage.total,
                                  "used_bytes": usage.used, "free_bytes": usage.free})
                if usage.free < 1024 ** 3:
                    raise RuntimeError(f"Less than 1 GiB disk space is available at {probe}")
            record["disk_preflight"] = disk_info
            print("[preflight] verified at least 1 GiB free on queue, data, model, and report filesystems", flush=True)
            dependency_check = [python_bin, "-c", "import torch; import torch_geometric; from torch_geometric.loader import NeighborLoader; print('torch=' + torch.__version__ + ' torch_geometric=' + torch_geometric.__version__)"]
            dependency_log = output_dir / "preflight_dependencies.log"
            dep_code = run_command(dependency_check, cwd=ROOT, env=env, log_path=dependency_log,
                                   command_path=output_dir / "preflight_dependencies.command.sh",
                                   exit_path=output_dir / "preflight_dependencies.exit_code")
            if dep_code:
                raise RuntimeError(f"Experiment Python dependency preflight failed with exit {dep_code}")
            telemetry = Telemetry(output_dir / "gpu_telemetry.jsonl", gpus)
            telemetry.start()
            # Resource polling logs utilization for the selected cards and only gates on free memory.
            while True:
                telemetry.set_phase("pre_pilot_wait")
                rows = telemetry.sample_once()
                record["last_gpu_snapshot"] = rows
                if [row["gpu"] for row in rows] != gpus:
                    raise ValueError("GPU preflight snapshot does not match the requested cards")
                if all(row["free_MiB"] >= args.min_free_mib for row in rows):
                    require_gpu_memory(rows, gpus, args.min_free_mib)
                    record["pre_pilot_gpu_snapshot"] = rows
                    break
                record.update(phase="wait_pilot_gpus", updated_at=utc_now())
                write_json(status_path, record)
                print(f"[queue] waiting for {args.min_free_mib} MiB free on GPUs {','.join(gpus)}", flush=True)
                time.sleep(args.poll_seconds)
            runner = str((ROOT / RUNNER_RELATIVE).resolve())
            plan_command = ["bash", runner, "plan", "all", "--config", str(CONFIG_RELATIVE),
                            "--gpus", ",".join(gpus), "--periods", ",".join(PILOT_CONTEXTS)]
            plan_log = output_dir / "plan_pilot.log"
            plan_code = run_command(plan_command, cwd=ROOT, env=env, log_path=plan_log,
                                    command_path=output_dir / "plan_pilot.command.sh",
                                    exit_path=output_dir / "plan_pilot.exit_code")
            if plan_code:
                raise RuntimeError(f"Pilot plan failed with exit {plan_code}")
            plan_output = plan_log.read_text(encoding="utf-8")
            check_plan_output(plan_output, PILOT_CONTEXTS)
            record["pilot_plan"] = {"command": plan_command, "summary": plan_output[-10000:]}
            record["experiment_record"] = {
                "research_question": config.get("research_question", "How do the configured tie groups contribute to retained Layer 0 messages across annual DBpedia BHHT L1 life windows?"),
                "config_path": str(config_path), "config_name": config["name"], "configuration": config,
                "seed": config["seed"],
                "data_split": config["prepare"], "data_input": source_prepare["input"],
                "source_graph": config["source_data"], "train_budget": {"rgcn_epochs": config["train"]["epochs"],
                    "graphmask_epochs_per_layer": config["graphmask_train"]["epochs_per_layer"]},
                "checkpoint_selection": config["graphmask_train"].get("checkpoint_selection", "any-stage"),
                "git_revision": revision, "pilot_command": None,
                "full_command": None, "result_path": str(output_dir),
            }
            write_json(output_dir / "experiment_record.json", record["experiment_record"])
            write_json(status_path, record)

            pilot_already_validated = bool(record.get("pilot_validated"))
            if not pilot_already_validated:
                check_pin(revision, initial_code, ROOT)
                verify_static_inputs(input_paths, source_info["input_snapshot"])
                record.update(phase="pilot", updated_at=utc_now())
                telemetry.set_phase("pilot")
                write_json(status_path, record)
                pilot_command = ["bash", runner, "run", "all", "--config", str(CONFIG_RELATIVE),
                                 "--gpus", ",".join(gpus), "--periods", ",".join(PILOT_CONTEXTS)]
                record["experiment_record"]["pilot_command"] = pilot_command
                write_json(output_dir / "experiment_record.json", record["experiment_record"])
                pilot_started_ns = time.time_ns()
                code = run_command(pilot_command, cwd=ROOT, env=env,
                                   log_path=output_dir / "pilot.log",
                                   command_path=output_dir / "pilot.command.sh",
                                   exit_path=output_dir / "pilot.exit_code")
                record["pilot_exit_code"] = code
                write_json(status_path, record)
                if code:
                    raise RuntimeError(f"Pilot command failed with exit {code}; full expansion stopped")
                record.update(phase="validate_pilot", updated_at=utc_now())
                write_json(status_path, record)
                validator_command = [python_bin, str(ROOT / VALIDATOR_RELATIVE), "--config", str(CONFIG_RELATIVE),
                                     "--periods", ",".join(PILOT_CONTEXTS)]
                validator_code = run_command(validator_command, cwd=ROOT, env=env,
                                             log_path=output_dir / "pilot_validation.log",
                                             command_path=output_dir / "pilot_validation.command.sh",
                                             exit_path=output_dir / "pilot_validation.exit_code")
                if validator_code:
                    raise RuntimeError(f"Pilot protocol validation failed with exit {validator_code}; full expansion stopped")
                graphmask_root = (ROOT / config["graphmask_root"]).resolve()
                validate_matrix_summary(graphmask_root / "matrix_summary.tsv", PILOT_CONTEXTS,
                                        expected_classes=5, newer_than_ns=pilot_started_ns)
                record["pilot_validated"] = True
                record["pilot_matrix_summary"] = str(graphmask_root / "matrix_summary.tsv")
                record["pilot_validation"] = "passed; validator and three-row summary with five classes/report files checked"
                write_json(status_path, record)
            else:
                print("[queue] resuming after validated pilot; skipping pilot rerun", flush=True)

            check_pin(revision, initial_code, ROOT)
            verify_static_inputs(input_paths, source_info["input_snapshot"])
            while True:
                telemetry.set_phase("pre_full_wait")
                rows = telemetry.sample_once()
                record["last_gpu_snapshot"] = rows
                if [row["gpu"] for row in rows] != gpus:
                    raise ValueError("GPU preflight snapshot does not match the requested cards")
                if all(row["free_MiB"] >= args.min_free_mib for row in rows):
                    require_gpu_memory(rows, gpus, args.min_free_mib)
                    record["pre_full_gpu_snapshot"] = rows
                    break
                record.update(phase="wait_full_gpus", updated_at=utc_now())
                write_json(status_path, record)
                print(f"[queue] pilot passed; waiting for {args.min_free_mib} MiB free on GPUs {','.join(gpus)} before full matrix", flush=True)
                time.sleep(args.poll_seconds)
            check_pin(revision, initial_code, ROOT)
            verify_static_inputs(input_paths, source_info["input_snapshot"])
            record.update(phase="full", updated_at=utc_now())
            telemetry.set_phase("full")
            write_json(status_path, record)
            full_command = ["bash", runner, "run", "all", "--config", str(CONFIG_RELATIVE),
                            "--gpus", ",".join(gpus), "--periods", "all"]
            record["experiment_record"]["full_command"] = full_command
            write_json(output_dir / "experiment_record.json", record["experiment_record"])
            full_started_ns = time.time_ns()
            full_code = run_command(full_command, cwd=ROOT, env=env,
                                    log_path=output_dir / "full.log",
                                    command_path=output_dir / "full.command.sh",
                                    exit_path=output_dir / "full.exit_code")
            if full_code:
                raise RuntimeError(f"Full matrix command failed with exit {full_code}")
            graphmask_root = (ROOT / config["graphmask_root"]).resolve()
            validate_matrix_summary(graphmask_root / "matrix_summary.tsv", EXPECTED_CONTEXTS,
                                    expected_classes=5, newer_than_ns=full_started_ns)
            record.update(status="complete", phase="complete", full_matrix_summary=str(graphmask_root / "matrix_summary.tsv"),
                          updated_at=utc_now())
            exit_code = 0
        except KeyboardInterrupt:
            record.update(status="interrupted", phase=record.get("phase", "interrupted"),
                          interruption_signal="SIGTERM_or_SIGINT", updated_at=utc_now())
            exit_code = 130
        except Exception as error:
            record.update(status="failed", error=str(error), updated_at=utc_now())
            print(f"[queue failed] {error}", file=sys.stderr, flush=True)
            exit_code = 1
        finally:
            if telemetry is not None:
                telemetry.close()
                record["gpu_telemetry_errors"] = list(telemetry.errors)
            record.update(exit_code=exit_code, finished_at=utc_now())
            write_json(status_path, record)
            (output_dir / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
