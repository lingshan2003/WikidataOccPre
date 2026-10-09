#!/usr/bin/env python3
"""Package small annual DBpedia reports; no PyTorch, weights or graph tables needed."""

import argparse
import io
import json
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def package(config_path, output):
    config = read_json(config_path)
    periods = read_json(source_path(config["period_config"]))
    contexts = [p["id"] for p in periods["periods"]]
    if config["periods"] != "all":
        requested = config["periods"]
        requested = requested.split(",") if isinstance(requested, str) else requested
        if set(requested) - set(contexts):
            raise ValueError("Configured periods are absent from the window calendar")
        contexts = [c for c in contexts if c in requested]
    if not contexts or periods.get("calendar_layout") != "sliding_windows":
        raise ValueError("Expected nonempty sliding-window results")
    seed = config["seed"]
    roots = {k: source_path(config[k]) for k in ("model_root", "graphmask_root", "relation_root", "period_root")}
    matrix = read_json(roots["graphmask_root"] / "matrix_summary.json")
    for context in contexts:
        rows = [r for r in matrix if r["context"] == context and r["representation"] == "multi_group" and int(r["seed"]) == seed]
        if len(rows) != 1 or rows[0]["status"] != "complete":
            raise ValueError(f"Missing/incomplete summary: {context}/multi_group/seed_{seed}")
    if read_json(roots["graphmask_root"] / "pipeline_failures.json"):
        raise ValueError("pipeline_failures.json is nonempty; resolve failures before packaging")
    resolved_path = roots["model_root"] / "grouped_pipeline_resolved_config.json"
    if not resolved_path.is_file():
        resolved_path = roots["graphmask_root"] / "grouped_pipeline_resolved_config.json"
    resolved = read_json(resolved_path)
    for key in ("seed", "train", "graphmask_train", "graphmask_report", "prepare", "num_bases", "period_config", "multi_group_taxonomy"):
        if resolved.get(key) != config.get(key):
            raise ValueError(f"Requested/resolved config mismatch: {key}")
    for key in roots:
        if source_path(resolved[key]).resolve() != roots[key].resolve():
            raise ValueError(f"Requested/resolved output root mismatch: {key}")
    files = {}

    def add(path, name, required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        files[name] = path

    add(config_path, "config/original_experiment.json")
    add(resolved_path, "config/original_resolved_experiment.json")
    add(source_path(config["period_config"]), "config/windows.json")
    add(source_path(config["multi_group_taxonomy"]), "config/taxonomy.json")
    for name in ("matrix_summary.json", "matrix_summary.tsv", "relation_group_summary.json", "relation_group_summary.tsv", "pipeline_failures.json"):
        add(roots["graphmask_root"] / name, "graphmask/" + name)
    for context in contexts:
        suffix = f"{context}/multi_group/seed_{seed}"
        for name in ("manifest.json", "validation.json", "training_history.json"):
            add(roots["graphmask_root"] / suffix / name, f"graphmask/{suffix}/{name}")
        for name in ("test_metrics.json", "manifest.json", "relations_base.csv", "relations_directed.csv"):
            add(roots["graphmask_root"] / suffix / "test_report" / name, f"graphmask/{suffix}/test_report/{name}")
        add(roots["model_root"] / suffix / "metrics.json", f"models/{suffix}/metrics.json")
        for name in ("split_summary.json", "relation_stats.csv", "relation_collapse_manifest.json"):
            add(roots["relation_root"] / context / "multi_group" / name, f"relations/{context}/multi_group/{name}")
        add(roots["period_root"] / context / "split_summary.json", f"periods/{context}/split_summary.json", required=False)
    normalized = dict(resolved, periods=contexts, period_config="config/windows.json", multi_group_taxonomy="config/taxonomy.json",
                      model_root="models", graphmask_root="graphmask", relation_root="relations", period_root="periods")
    metadata = {"bundle_format": "dbpedia-sliding-reports-v1", "contexts": contexts, "seed": seed,
                "representation": "multi_group", "experiment_name": config["name"], "files": sorted(files),
                "excluded": ["model/probe weights", "graph tensors", "node/edge tables", "root-level edge reports", "logs"]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}; choose another --output")
    temporary = output.with_name(output.name + ".partial")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, payload in (("bundle.json", metadata), ("config/experiment.json", normalized)):
                content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name, recursive=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Packaged {len(contexts)} complete multi-group windows, {len(files)} small report files")
    print(f"Archive: {output.resolve()} ({output.stat().st_size / 1024**2:.2f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="artifacts/dbpedia_sliding_1900_2000_reports.tar.gz")
    args = parser.parse_args()
    package(source_path(args.config), source_path(args.output))


if __name__ == "__main__":
    main()
