#!/usr/bin/env python3
"""Portable CBDB cleaning/grouping/period/RGCN/GraphMask runner.

Planning and the first three data stages need only Python's standard library.
Completed stages are reused only when commands, input stamps and outputs agree.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGES = ("all", "clean", "group", "periods", "prepare", "train", "graphmask", "summarize")
GRAPH_FILES = ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json", "class_stats.csv", "relation_stats.csv")
MODEL_FILES = ("best_model.pt", "metrics.json", "test_predictions.csv")
PROBE_FILES = ("graphmask_probe.pt", "validation.json", "manifest.json", "training_history.json")
REPORT_FILES = ("test_metrics.json", "relations_base.csv", "relations_directed.csv", "root_top_edges.csv.gz", "manifest.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def flags(options):
    result = []
    for name, value in options.items():
        if value is None or isinstance(value, (bool, dict, list)):
            raise ValueError(f"Option {name} must be a string or number")
        result.extend(["--" + name.replace("_", "-"), str(value)])
    return result


def select(value, available):
    items = list(available) if value == "all" else value.split(",") if isinstance(value, str) else list(value)
    if not items or len(items) != len(set(items)) or set(items) - set(available):
        raise ValueError(f"Invalid selection {value!r}; available={list(available)}")
    return items


def complete(directory, outputs):
    return all((directory / f).is_file() and (directory / f).stat().st_size > 0 for f in outputs)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.config_path = self.path(args.config)
        self.config = read_json(self.config_path)
        c = self.config
        if c.get("version") != 1 or not c.get("name"):
            raise ValueError("Expected a named version-1 CBDB pipeline configuration")
        self.output = self.path(args.output_root or c["output_root"])
        if args.smoke and not args.output_root:
            self.output = self.output.with_name(self.output.name + "_smoke")
        c["output_root"] = str(self.output)
        c["device"] = args.device or ("cpu" if args.smoke else c["device"])
        if args.flat_input:
            c["flat_input"] = args.flat_input
        if args.database:
            c["database"] = args.database
        if args.smoke:
            c["train"].update(epochs=1, num_workers=0, hidden_dim=16, branch_dim=8,
                              num_neighbors="3,2", batch_size=256, patience=1)
            c["graphmask_train"].update(epochs_per_layer=1, num_workers=0, batch_size=256)
            c["graphmask_report"]["top_k"] = 5
        if args.epochs is not None:
            c["train"]["epochs"] = args.epochs
        if args.graphmask_epochs is not None:
            c["graphmask_train"]["epochs_per_layer"] = args.graphmask_epochs
        if args.workers is not None:
            c["train"]["num_workers"] = args.workers
            c["graphmask_train"]["num_workers"] = args.workers
        c["seeds"] = [int(s) for s in args.seeds.split(",")] if args.seeds else c["seeds"]
        if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])) or any(type(s) is not int or s < 0 for s in c["seeds"]):
            raise ValueError("seeds must be distinct nonnegative integers")
        self.seeds = c["seeds"]
        from scripts.cbdb_split_periods import load_period_config
        self.period_config = load_period_config(self.path(c["period_config"]))
        self.periods = {p["id"]: p for p in self.period_config["periods"]}
        self.contexts = select(args.periods or ("1700_plus" if args.smoke else "all"), self.periods)
        self.representations = select(args.representations or c["representations"], ("multi_group", "binary"))
        self.from_sqlite = args.from_sqlite or args.stage == "clean"
        self.flat = self.output / "flat/person_relation_triples.tsv.gz" if self.from_sqlite else self.path(c["flat_input"])
        self.grouped = self.output / "grouped"
        self.period_root = self.output / "periods"
        for key in ("flat_input", "database", "occupation_rules", "relation_profile", "period_config"):
            source = self.path(c[key])
            if self.output == source or self.output in source.parents:
                raise ValueError(f"Output root would overwrite input {source}")
        if self.output == ROOT or self.output in ROOT.parents:
            raise ValueError("Output root must not contain the repository")
        self.profile = read_json(self.path(c["relation_profile"]))
        self.groups = tuple(self.profile["groups"])
        for section in ("prepare", "train", "graphmask_train", "graphmask_report"):
            options = c[section]
            if set(options) & {"input", "data", "checkpoint", "probe", "output_dir", "device", "tie_taxonomy", "relation_taxonomy", "num_bases"}:
                raise ValueError(f"Reserved path/identity option in {section}")
            if section != "prepare" and "seed" in options:
                raise ValueError(f"Use top-level seeds, not {section}.seed")
            flags(options)
        if c["prepare"]["target_level"] != 1 or c["prepare"]["min_class_count"] < 3:
            raise ValueError("CBDB requires target_level=1 and min_class_count>=3")
        if c["train"]["model"] != "rgcn" or c["train"]["num_layers"] != 2:
            raise ValueError("This pipeline expects two-layer RGCN")
        if c["train"]["occupation_feature_levels"] != "1" or c["train"]["train_mode"] != "sampled" or c["train"]["eval_mode"] != "sampled":
            raise ValueError("Use sampled train/eval with only the real level-1 occupation features")
        if c["graphmask_train"]["num_neighbors"] != "auto" or c["graphmask_report"]["num_neighbors"] != "auto":
            raise ValueError("GraphMask must reuse the RGCN checkpoint fanouts via auto")
        if c["graphmask_train"]["train_split"] != "train" or c["graphmask_train"]["validation_split"] != "val" or c["graphmask_report"]["split"] != "test":
            raise ValueError("GraphMask uses train for fitting, val for selection and test for reporting")
        for key in ("epochs", "batch_size"):
            if c["train"][key] < 1:
                raise ValueError(f"train.{key} must be positive")
        if c["graphmask_train"]["epochs_per_layer"] < 1 or any(type(c["num_bases"][r]) is not int or c["num_bases"][r] < 1 for r in self.representations):
            raise ValueError("GraphMask epochs and RGCN basis counts must be positive")
        self.failures = []

    @staticmethod
    def path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    @staticmethod
    def command(workflow, **options):
        return [sys.executable, str(ROOT / "run.py"), workflow, *flags(options)]

    def script(self, name, **options):
        return [sys.executable, str(ROOT / "scripts" / name), *flags(options)]

    def clean_command(self):
        c = self.config
        return self.script("cbdb_build_flat_triples.py", database=self.path(c["database"]),
                           occupation_rules=self.path(c["occupation_rules"]), output_dir=self.output / "flat")

    def group_command(self):
        return self.script("cbdb_build_experiment_groups.py", input=self.flat,
                           profile=self.path(self.config["relation_profile"]), output_dir=self.grouped)

    def period_command(self):
        # Export all periods once; selection only controls model jobs, making
        # incremental period/representation/seed selections safely reusable.
        return self.script("cbdb_split_periods.py", input_dir=self.grouped,
                           period_config=self.path(self.config["period_config"]),
                           output_dir=self.period_root, periods="all") + ["--overwrite"]

    def directories(self, period, rep, seed):
        key = Path(period) / rep
        return (self.period_root / key, self.output / "graphs" / key,
                self.output / "models" / key / f"seed_{seed}",
                self.output / "graphmask" / key / f"seed_{seed}")

    def job_commands(self, period, rep, seed):
        source, graph, model, mask = self.directories(period, rep, seed)
        vocab = source / "relation_vocabulary.json"
        count = len(read_json(vocab)["relation_to_id"]) if vocab.exists() else 2 * (len(self.groups) if rep == "multi_group" else 2)
        bases = min(self.config["num_bases"][rep], count)
        return (
            self.command("prepare", input=source / "Q_R_Q_extended.csv.gz", output_dir=graph, **self.config["prepare"]),
            self.command("train", data=graph / "graph_data.pt", output_dir=model,
                         tie_taxonomy=source / "tie_taxonomy.json", relation_taxonomy=source / "relation_taxonomy.json",
                         num_bases=bases, seed=seed, device=self.config["device"], **self.config["train"]),
            self.command("graphmask-train", data=graph / "graph_data.pt", checkpoint=model / "best_model.pt",
                         output_dir=mask, seed=seed, device=self.config["device"], **self.config["graphmask_train"]),
            self.command("graphmask-report", data=graph / "graph_data.pt", checkpoint=model / "best_model.pt",
                         probe=mask / "graphmask_probe.pt", output_dir=mask / "test_report",
                         seed=seed, device=self.config["device"], **self.config["graphmask_report"]),
        )

    def dependencies(self, stage):
        folders = ("data",) if stage == "prepare" else ("models", "training")
        return [ROOT / "run.py", ROOT / "cli.py", *sorted(p for folder in folders for p in (ROOT / folder).rglob("*.py"))]

    def contract(self, command, inputs):
        return {"version": 1, "command": command, "inputs": [stamp(p) for p in inputs]}

    def require(self, name, command, directory, outputs, inputs):
        path = directory / f"pipeline_{name}.json"
        if not path.is_file():
            raise ValueError(f"Run the {name} stage first: missing {path}")
        record = read_json(path)
        if record.get("status") != "complete" or record.get("contract") != self.contract(command, inputs):
            raise ValueError(f"Incomplete or changed {name} stage: {directory}; use a new --output-root if settings changed")
        if not complete(directory, outputs) or record.get("outputs") != [stamp(directory / f) for f in outputs]:
            raise ValueError(f"Missing/changed {name} outputs: {directory}; use a new --output-root")

    def step(self, name, command, directory, outputs, inputs):
        contract = self.contract(command, inputs)
        path = directory / f"pipeline_{name}.json"
        if path.exists():
            record = read_json(path)
            if record.get("contract") != contract:
                raise ValueError(f"Changed inputs/settings for {name}: {directory}; use a new --output-root")
            if record.get("status") == "complete":
                self.require(name, command, directory, outputs, inputs)
                print(f"[reuse {name}] {directory}", flush=True)
                return
        elif any((directory / f).exists() for f in outputs):
            raise ValueError(f"Existing outputs have no pipeline completion record: {directory}; use a new --output-root")
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, {"status": "running", "contract": contract})
        (directory / f"{name}_command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        print(f"[run {name}] {shlex.join(command)}", flush=True)
        try:
            with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f"{name} exited {code}; see {directory / (name + '.log')}")
            if not complete(directory, outputs):
                raise ValueError(f"{name} did not produce all required outputs: {directory}")
            write_json(path, {"status": "complete", "contract": contract,
                              "outputs": [stamp(directory / f) for f in outputs]})
        except Exception as error:
            write_json(path, {"status": "failed", "contract": contract, "error": str(error)})
            raise

    def data_steps(self):
        stage = self.args.stage
        if self.from_sqlite:
            inputs = [self.path(self.config["database"]), self.path(self.config["occupation_rules"]),
                      ROOT / "scripts/cbdb_build_flat_triples.py"]
            method = self.step if stage in ("all", "clean") else self.require
            method("clean", self.clean_command(), self.output / "flat", ("person_relation_triples.tsv.gz", "summary.json"), inputs)
        if stage == "clean":
            return
        profile = self.path(self.config["relation_profile"])
        inputs = [self.flat, profile, self.path(self.profile["fine_mapping"]),
                  ROOT / "scripts/cbdb_build_experiment_groups.py", ROOT / "scripts/cbdb_group_relations.py"]
        grouped_files = ("summary.json", "nodes.tsv.gz", "relation_mapping.tsv", "group_support.tsv", "person_relation_triples_annotated.tsv.gz",
                         *[f"{r}/{f}" for r in ("multi_group", "binary") for f in
                           ("person_relation_triples.tsv.gz", "Q_R_Q_extended.csv.gz", "relation_vocabulary.json", "relation_taxonomy.json", "tie_taxonomy.json")])
        method = self.step if stage in ("all", "group") else self.require
        method("group", self.group_command(), self.grouped, grouped_files, inputs)
        if stage == "group":
            return
        period_files = ("summary.json", "period_summary.tsv", *[f"{period}/{rep}/{f}" for period in self.periods
                         for rep in ("multi_group", "binary") for f in
                         ("summary.json", "person_relation_triples.tsv.gz", "Q_R_Q_extended.csv.gz", "eligible_nodes.tsv.gz", "active_nodes.tsv.gz",
                          "relation_vocabulary.json", "relation_taxonomy.json", "tie_taxonomy.json")])
        inputs = [self.grouped / "nodes.tsv.gz", self.grouped / "summary.json",
                  *[self.grouped / r / "person_relation_triples.tsv.gz" for r in ("multi_group", "binary")],
                  self.path(self.config["period_config"]), ROOT / "scripts/cbdb_split_periods.py",
                  ROOT / "scripts/cbdb_build_experiment_groups.py", ROOT / "scripts/cbdb_group_relations.py"]
        method = self.step if stage in ("all", "periods") else self.require
        # The exporter atomically replaces its output directory. Keep runner
        # logs/records outside that directory so they survive replacement and
        # do not make the first export look like an unrelated nonempty target.
        method("periods", self.period_command(), self.output / "stages/periods",
               tuple(str(self.period_root / f) for f in period_files), inputs)

    def graph_inputs(self, period, rep):
        source, _, _, _ = self.directories(period, rep, self.seeds[0])
        return [source / f for f in ("Q_R_Q_extended.csv.gz", "summary.json", "relation_vocabulary.json", "relation_taxonomy.json", "tie_taxonomy.json")] + self.dependencies("prepare")

    def validate_graph(self, period, rep):
        import torch
        source, graph, _, _ = self.directories(period, rep, self.seeds[0])
        bundle = torch.load(graph / "graph_data.pt", map_location="cpu", weights_only=False)
        data, metadata = bundle["data"], bundle["metadata"]
        vocab = read_json(source / "relation_vocabulary.json")["relation_to_id"]
        if metadata["target_column"] != "occupation_level1" or metadata["relation_to_id"] != vocab:
            raise ValueError(f"Wrong occupation target or relation vocabulary: {graph}")
        if any(int(data[k].sum()) == 0 for k in ("train_mask", "val_mask", "test_mask")):
            raise ValueError(f"No eligible supervised split: {graph}")
        if int(data.num_nodes) != self.node_count(source / "active_nodes.tsv.gz"):
            raise ValueError(f"Graph must contain exactly the period's active people: {graph}")
        unknown = metadata["occupation_unknown_ids"]["occupation_level1"]
        if any(bool(torch.any(data.occupation_level1[data[k]] != unknown)) for k in ("val_mask", "test_mask")):
            raise ValueError("Validation/test occupation features must be unknown")
        return bundle

    @staticmethod
    def node_count(path):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return sum(1 for _ in csv.DictReader(stream, delimiter="\t"))

    def prepare_graphs(self):
        for period in self.contexts:
            bundles, ids = {}, {}
            for rep in self.representations:
                _, graph, _, _ = self.directories(period, rep, self.seeds[0])
                command = self.job_commands(period, rep, self.seeds[0])[0]
                method = self.step if self.args.stage in ("all", "prepare") else self.require
                method("prepare", command, graph, GRAPH_FILES, self.graph_inputs(period, rep))
                bundles[rep] = self.validate_graph(period, rep)
                with (graph / "nodes.csv").open(encoding="utf-8", newline="") as stream:
                    ids[rep] = [r["node_id"] for r in csv.DictReader(stream)]
            if len(bundles) == 2:
                import torch
                a, b = bundles["multi_group"], bundles["binary"]
                if ids["multi_group"] != ids["binary"] or a["metadata"]["label_to_id"] != b["metadata"]["label_to_id"]:
                    raise ValueError(f"Paired representations differ in node order/labels: {period}")
                for key in ("y", "train_mask", "val_mask", "test_mask", "occupation_level1", "occupation_level2", "occupation_level3", "temporal", "country"):
                    if not torch.equal(a["data"][key], b["data"][key]):
                        raise ValueError(f"Paired representations changed {key}: {period}")
            if next(iter(bundles.values()))["metadata"]["num_classes"] == 1:
                print(f"[single-class target] {period}: only one occupation meets min_class_count; "
                      "pipeline execution is valid, classification metrics do not measure multiclass prediction", flush=True)

    def preflight(self):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        device = torch.device(self.config["device"])
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Use cpu or cuda:N")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA unavailable in this Python environment; select the server venv via CBDB_PYTHON_BIN")
            torch.empty(1, device=device)
        try:
            next(iter(NeighborLoader(Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2),
                                     num_neighbors=[1, 1], input_nodes=torch.tensor([0]), batch_size=1, num_workers=0)))
        except Exception as error:
            raise ValueError("PyG sampling backend unavailable; install matching pyg-lib or torch-sparse (CBDB/README.md)") from error
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        print(f"[preflight] Python={sys.executable} torch={torch.__version__} "
              f"CUDA_VISIBLE_DEVICES={visible} device={device} ({gpu}); NeighborLoader OK", flush=True)

    def run_job(self, period, rep, seed):
        source, graph, model, mask = self.directories(period, rep, seed)
        _, train, probe, report = self.job_commands(period, rep, seed)
        training_inputs = [graph / "graph_data.pt", source / "tie_taxonomy.json", source / "relation_taxonomy.json"] + self.dependencies("train")
        method = self.step if self.args.stage in ("all", "train") else self.require
        method("train", train, model, MODEL_FILES, training_inputs)
        if self.args.stage not in ("all", "graphmask", "summarize"):
            return
        probe_inputs = [graph / "graph_data.pt", model / "best_model.pt", model / "metrics.json"] + self.dependencies("graphmask")
        method = self.require if self.args.stage == "summarize" else self.step
        method("probe", probe, mask, PROBE_FILES, probe_inputs)
        validation = read_json(mask / "validation.json")
        if validation["relative_macro_f1_difference"] > self.config["graphmask_train"]["max_relative_macro_f1_diff"]:
            raise ValueError(f"GraphMask validation fidelity exceeds configured threshold: {mask}")
        inputs = [graph / "graph_data.pt", graph / "nodes.csv", model / "best_model.pt", model / "metrics.json", mask / "graphmask_probe.pt"] + self.dependencies("graphmask")
        method("report", report, mask / "test_report", REPORT_FILES, inputs)
        metrics = read_json(mask / "test_report/test_metrics.json")
        expected = read_json(graph / "split_summary.json")["test_nodes"]
        if metrics["roots"] != expected or metrics["labeled_roots"] != expected or metrics["split"] != "test":
            raise ValueError("GraphMask report does not cover the entire supervised test split")

    def require_report(self, period, rep, seed):
        """Never label leftover/stale report files as a completed experiment."""
        source, graph, model, mask = self.directories(period, rep, seed)
        _, train, probe, report = self.job_commands(period, rep, seed)
        self.require("train", train, model, MODEL_FILES,
                     [graph / "graph_data.pt", source / "tie_taxonomy.json", source / "relation_taxonomy.json"] + self.dependencies("train"))
        self.require("probe", probe, mask, PROBE_FILES,
                     [graph / "graph_data.pt", model / "best_model.pt", model / "metrics.json"] + self.dependencies("graphmask"))
        self.require("report", report, mask / "test_report", REPORT_FILES,
                     [graph / "graph_data.pt", graph / "nodes.csv", model / "best_model.pt", model / "metrics.json", mask / "graphmask_probe.pt"] + self.dependencies("graphmask"))
        validation = read_json(mask / "validation.json")
        if validation["relative_macro_f1_difference"] > self.config["graphmask_train"]["max_relative_macro_f1_diff"]:
            raise ValueError("Reported GraphMask probe violates validation fidelity threshold")
        metrics = read_json(mask / "test_report/test_metrics.json")
        expected = read_json(graph / "split_summary.json")["test_nodes"]
        if metrics["roots"] != expected or metrics["labeled_roots"] != expected or metrics["split"] != "test":
            raise ValueError("Report test split coverage differs from the prepared graph")

    def summarize(self):
        matrix, relations = [], []
        for period in self.contexts:
            for rep in self.representations:
                for seed in self.seeds:
                    source, graph, model, mask = self.directories(period, rep, seed)
                    errors = [f["error"] for f in self.failures if (f["period"], f["representation"], f["seed"]) == (period, rep, seed)]
                    def optional(path):
                        return read_json(path) if path.is_file() else {}
                    split = optional(graph / "split_summary.json")
                    training = optional(model / "metrics.json")
                    validation = optional(mask / "validation.json")
                    report_dir = mask / "test_report"
                    reported = complete(report_dir, REPORT_FILES) and not errors
                    if reported:
                        try:
                            self.require_report(period, rep, seed)
                        except Exception as error:
                            failure = dict(period=period, representation=rep, seed=seed, error=str(error))
                            self.failures.append(failure)
                            errors.append(str(error))
                            reported = False
                    metrics = optional(report_dir / "test_metrics.json") if reported else {}
                    eligible = self.node_count(source / "eligible_nodes.tsv.gz")
                    state = "failed" if errors else "complete" if reported else "awaiting_graphmask" if complete(model, MODEL_FILES) else "awaiting_train"
                    matrix.append(dict(period=period, representation=rep, seed=seed, status=state,
                        eligible_people=eligible, graph_people=split.get("nodes"), message_edges=split.get("edges_after_reverse_and_deduplication"),
                        relation_types=split.get("relation_types"), classes=split.get("target_classes_retained"),
                        target_status="single_class" if split.get("target_classes_retained") == 1 else "multiclass" if split.get("target_classes_retained", 0) > 1 else "unknown",
                        train_nodes=split.get("train_nodes"), val_nodes=split.get("val_nodes"), test_nodes=split.get("test_nodes"),
                        rgcn_test_macro_f1=training.get("test", {}).get("macro_f1"),
                        graphmask_original_macro_f1=metrics.get("original", {}).get("macro_f1"),
                        graphmask_masked_macro_f1=metrics.get("masked", {}).get("macro_f1"),
                        hard_retention_rate=metrics.get("hard_retention_rate"), prediction_agreement=metrics.get("prediction_agreement"),
                        validation_relative_f1_difference=validation.get("relative_macro_f1_difference"),
                        error=" | ".join(errors), report_dir=str(report_dir)))
                    if not reported:
                        continue
                    with (report_dir / "relations_base.csv").open(encoding="utf-8", newline="") as stream:
                        observed = {(int(r["layer"]), r["relation"]): r for r in csv.DictReader(stream)}
                    with (graph / "relation_stats.csv").open(encoding="utf-8", newline="") as stream:
                        counts = {r["relation"]: int(r["count"]) for r in csv.DictReader(stream) if not r["relation"].endswith("__rev")}
                    for layer in range(self.config["train"]["num_layers"]):
                        for group in (self.groups if rep == "multi_group" else ("inherited", "acquired")):
                            row = observed.get((layer, group), {})
                            messages = int(row.get("message_observations", 0))
                            relations.append(dict(period=period, representation=rep, seed=seed, layer=layer, group=group,
                                original_triples=counts.get(group, 0), message_observations=messages,
                                support_status="observed" if messages else "no_sampled_messages" if counts.get(group, 0) else "no_graph_edges",
                                hard_retention_rate=row.get("hard_retention_rate") if messages else None,
                                retained_edge_share=row.get("retained_edge_share") if messages else None,
                                mean_keep_probability=row.get("mean_keep_probability") if messages else None))
        directory = self.output / "summary"
        write_json(directory / "matrix_summary.json", matrix)
        write_tsv(directory / "matrix_summary.tsv", matrix, list(matrix[0]))
        write_json(directory / "relation_group_summary.json", relations)
        write_tsv(directory / "relation_group_summary.tsv", relations, list(relations[0]) if relations else
                  ["period", "representation", "seed", "layer", "group", "support_status"])
        write_json(directory / "failures.json", self.failures)
        print(f"[summary] {sum(r['status']=='complete' for r in matrix)}/{len(matrix)} complete: {directory / 'matrix_summary.tsv'}", flush=True)

    def plan(self):
        stage = self.args.stage
        print(f"[plan] {len(self.contexts)} periods x {len(self.representations)} representations x {len(self.seeds)} seeds; output={self.output}")
        if self.from_sqlite and stage in ("all", "clean"):
            print("[clean] " + shlex.join(self.clean_command()))
        for name, command in (("group", self.group_command()), ("periods", self.period_command())):
            if stage in ("all", name):
                print(f"[{name}] " + shlex.join(command))
        for period in self.contexts:
            for rep in self.representations:
                for seed in self.seeds:
                    for name, command in zip(("prepare", "train", "probe", "report"), self.job_commands(period, rep, seed)):
                        if stage in ("all", name) or stage == "graphmask" and name in ("probe", "report"):
                            if name != "prepare" or seed == self.seeds[0]:
                                print(f"[{name}] {period}/{rep}/seed_{seed}\n  {shlex.join(command)}")
        if stage in ("all", "train", "graphmask", "summarize"):
            print(f"[summary] {self.output / 'summary/matrix_summary.tsv'}")

    def run(self):
        if self.args.stage in ("all", "train", "graphmask"):
            self.preflight()
        self.data_steps()
        if self.args.stage in ("clean", "group", "periods"):
            return 0
        self.prepare_graphs()
        if self.args.stage == "prepare":
            return 0
        for period in self.contexts:
            for rep in self.representations:
                for seed in self.seeds:
                    try:
                        self.run_job(period, rep, seed)
                    except Exception as error:
                        self.failures.append(dict(period=period, representation=rep, seed=seed, error=str(error)))
                        print(f"[failed] {period}/{rep}/seed_{seed}: {error}", file=sys.stderr, flush=True)
        self.summarize()
        return 1 if self.failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default="config/cbdb_pipeline_v1.json")
    parser.add_argument("--output-root", help="Separate output root for a new experiment/configuration")
    parser.add_argument("--flat-input", help="Override cleaned raw-relation triple input")
    parser.add_argument("--database", help="Override original SQLite path")
    parser.add_argument("--from-sqlite", action="store_true", help="Rebuild cleaned flat triples first (also pass on later stages)")
    parser.add_argument("--periods", help="all or comma-separated period IDs")
    parser.add_argument("--representations", help="multi_group,binary or one of them")
    parser.add_argument("--seeds", help="Comma-separated model seeds; node split remains fixed and paired")
    parser.add_argument("--device", help="cuda:N or cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--graphmask-epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="One real period, both representations, CPU, one epoch, smaller model; separate default output root")
    parser.add_argument("--check-env", action="store_true", help="Check torch/CUDA/NeighborLoader and exit without building data")
    return parser.parse_args(argv)


def main():
    os.chdir(ROOT)
    try:
        pipeline = Pipeline(parse_args())
        if pipeline.args.check_env:
            pipeline.preflight()
            return 0
        if pipeline.args.mode == "plan":
            pipeline.plan()
            return 0
        pipeline.output.mkdir(parents=True, exist_ok=True)
        with (pipeline.output / ".pipeline.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another CBDB pipeline is using this output root") from error
            write_json(pipeline.output / "resolved_config.json", dict(pipeline.config,
                selected_periods=pipeline.contexts, selected_representations=pipeline.representations,
                source_mode="sqlite" if pipeline.from_sqlite else "cleaned_flat", smoke=pipeline.args.smoke))
            return pipeline.run()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
