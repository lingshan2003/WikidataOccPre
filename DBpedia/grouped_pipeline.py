#!/usr/bin/env python3
"""Plan/run the DBpedia binary and multi-group RGCN/GraphMask matrix.

The Bash entry point supplies the Python environment. Planning needs only the
standard library. Existing repository commands do the graph/model work.
Stage reuse checks JSON settings and file size/mtime, without additional SHA
passes over artifacts. The existing trainers retain their own provenance rules.
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
DEFAULT_CONFIG = "config/dbpedia_grouped_rgcn_graphmask_20y_v1.json"
STAGES = ("all", "prepare", "collapse", "train", "graphmask", "summarize")
MODEL_FILES = ("best_model.pt", "metrics.json", "test_predictions.csv")
PROBE_FILES = ("graphmask_probe.pt", "validation.json", "manifest.json", "training_history.json")
REPORT_FILES = ("test_metrics.json", "relations_base.csv", "relations_directed.csv", "root_top_edges.csv.gz", "manifest.json")
RELATION_COLUMNS = ("context", "representation", "seed", "layer", "group", "original_triples_after_collapse", "message_observations", "support_status", "hard_retention_rate", "retained_edge_share", "mean_keep_probability")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def complete(directory, filenames):
    return all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in filenames)


def flags(values):
    result = []
    for name, value in values.items():
        if isinstance(value, (dict, list, bool)) or value is None:
            raise ValueError(f"CLI setting {name} must be a string or number")
        result.extend(["--" + name.replace("_", "-"), str(value)])
    return result


def select_items(requested, available):
    items = list(available) if requested == "all" else (
        requested.split(",") if isinstance(requested, str) else list(requested)
    )
    if not items or len(items) != len(set(items)) or set(items) - set(available):
        raise ValueError(f"Invalid selection {requested!r}; available={list(available)}")
    return items


def taxonomy_members(payload):
    groups = payload["groups"]
    members = []
    for group, relations in groups.items():
        if not isinstance(relations, list) or not relations:
            raise ValueError(f"Use explicit, non-empty taxonomy members for {group}")
        if any(not isinstance(r, str) or not r or r.endswith("__rev") for r in relations):
            raise ValueError(f"Invalid base relation in {group}")
        members.extend(relations)
    if len(members) != len(set(members)):
        raise ValueError("Taxonomy groups overlap")
    return set(members)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.config_path = self.path(args.config)
        self.config = read_json(self.config_path)
        c = self.config
        if c.get("version") != 1 or not c.get("name"):
            raise ValueError("Expected a named version-1 pipeline configuration")
        # Group-specific output variables never pick up the exact-baseline roots.
        for key in ("source_data", "period_root", "period_config", "binary_taxonomy", "multi_group_taxonomy", "relation_root", "model_root", "graphmask_root"):
            c[key] = os.environ.get("DBPEDIA_GROUP_" + key.upper(), c[key])
        c["device"] = args.device or os.environ.get("DBPEDIA_GROUP_DEVICE", os.environ.get("DBPEDIA_DEVICE", c["device"]))
        c["seed"] = int(os.environ.get("DBPEDIA_GROUP_SEED", c["seed"]))
        for section, key, env, cast in (
            ("train", "batch_size", "TRAIN_BATCH_SIZE", int),
            ("train", "num_workers", "TRAIN_WORKERS", int),
            ("train", "epochs", "TRAIN_EPOCHS", int),
            ("train", "num_neighbors", "TRAIN_FANOUTS", str),
            ("graphmask_train", "batch_size", "GRAPHMASK_BATCH_SIZE", int),
            ("graphmask_train", "num_workers", "GRAPHMASK_WORKERS", int),
            ("graphmask_train", "epochs_per_layer", "GRAPHMASK_EPOCHS_PER_LAYER", int),
            ("graphmask_train", "beta", "GRAPHMASK_BETA", float),
            ("graphmask_train", "max_relative_macro_f1_diff", "GRAPHMASK_MAX_RELATIVE_F1_DIFF", float),
        ):
            if "DBPEDIA_GROUP_" + env in os.environ:
                c[section][key] = cast(os.environ["DBPEDIA_GROUP_" + env])
        if c["train"]["model"] != "rgcn" or c["train"]["num_layers"] != 2:
            raise ValueError("This matrix requires two-layer RGCN")
        if c["graphmask_report"]["split"] != "test":
            raise ValueError("The matrix requires test GraphMask reports")
        for section in ("train", "graphmask_train", "graphmask_report", "prepare"):
            if set(c[section]) & {"data", "output_dir", "checkpoint", "probe", "seed", "device", "tie_taxonomy", "relation_taxonomy", "num_bases"}:
                raise ValueError(f"Reserved path/identity option in {section}")
            flags(c[section])
        self.period_config = read_json(self.path(c["period_config"]))
        self.periods = {p["id"]: p for p in self.period_config["periods"]}
        self.contexts = select_items(args.periods or os.environ.get("DBPEDIA_GROUP_PERIODS", c["periods"]), self.periods)
        include_full = args.include_full or c["include_full"]
        if "DBPEDIA_GROUP_INCLUDE_FULL" in os.environ:
            value = os.environ["DBPEDIA_GROUP_INCLUDE_FULL"]
            if value not in ("0", "1"):
                raise ValueError("DBPEDIA_GROUP_INCLUDE_FULL must be 0 or 1")
            include_full = args.include_full or value == "1"
        if include_full:
            self.contexts.insert(0, "full")
        self.representations = select_items(args.representations or os.environ.get("DBPEDIA_GROUP_REPRESENTATIONS", c["representations"]), ("binary", "multi_group"))
        self.binary = read_json(self.path(c["binary_taxonomy"]))
        self.multi = read_json(self.path(c["multi_group_taxonomy"]))
        self.members = taxonomy_members(self.multi)
        if "inherited" not in self.multi["groups"] or len(self.multi["groups"]) < 2:
            raise ValueError("Multi-group taxonomy needs inherited and acquired groups")
        if set(self.binary["groups"]) != {"inherited", "acquired"}:
            raise ValueError("Binary taxonomy requires inherited/acquired")
        if set(self.binary["groups"]["inherited"]) != set(self.multi["groups"]["inherited"]):
            raise ValueError("Binary and multi-group inherited definitions differ")
        remaining = self.members - set(self.multi["groups"]["inherited"])
        if self.binary["groups"]["acquired"] != "all_remaining" and set(self.binary["groups"]["acquired"]) != remaining:
            raise ValueError("Binary acquired differs from the multi-group union")
        for rep in self.representations:
            limit = 4 if rep == "binary" else 2 * len(self.multi["groups"])
            if not 1 <= c["num_bases"][rep] <= limit:
                raise ValueError(f"num_bases.{rep} must be between 1 and {limit}")
        roots = [self.path(c[key]) for key in ("relation_root", "model_root", "graphmask_root")]
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:]):
            raise ValueError("Relation, model and GraphMask roots must be separate")
        self.failures = []
        self.torch = None

    @staticmethod
    def path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    def source(self, context):
        return self.path(self.config["source_data"]) if context == "full" else self.path(self.config["period_root"]) / context / "graph_data.pt"

    def directories(self, context, rep):
        c = self.config
        return (
            self.path(c["relation_root"]) / context / rep,
            self.path(c["model_root"]) / context / rep / f"seed_{c['seed']}",
            self.path(c["graphmask_root"]) / context / rep / f"seed_{c['seed']}",
        )

    def command(self, subcommand, **options):
        return [sys.executable, str(ROOT / "run.py"), subcommand, *flags(options)]

    def commands(self, context, rep):
        artifact, model, mask = self.directories(context, rep)
        c = self.config
        data = artifact / "graph_data.pt"
        tie = artifact / ("binary_tie_taxonomy.json" if rep == "binary" else "collapsed_tie_taxonomy.json")
        collapse = self.command(
            "collapse-ties" if rep == "binary" else "collapse-relations",
            data=self.source(context), output_dir=artifact,
            **{("tie_taxonomy" if rep == "binary" else "relation_taxonomy"): self.path(c["binary_taxonomy" if rep == "binary" else "multi_group_taxonomy"])},
        )
        train = self.command("train", data=data, output_dir=model, tie_taxonomy=tie, num_bases=c["num_bases"][rep], seed=c["seed"], device=c["device"], **c["train"])
        probe = self.command("graphmask-train", data=data, checkpoint=model / "best_model.pt", output_dir=mask, seed=c["seed"], device=c["device"], **c["graphmask_train"])
        report = self.command("graphmask-report", data=data, checkpoint=model / "best_model.pt", probe=mask / "graphmask_probe.pt", output_dir=mask / "test_report", seed=c["seed"], device=c["device"], **c["graphmask_report"])
        return collapse, train, probe, report

    def prepare_command(self, contexts):
        c = self.config
        return [sys.executable, str(ROOT / "scripts/prepare_life_period_induced_artifacts.py"), "--source-data", str(self.path(c["source_data"])), "--output-root", str(self.path(c["period_root"])), "--life-periods", str(self.path(c["period_config"])), "--periods", ",".join(contexts), *flags(c["prepare"])]

    def plan(self):
        stage = self.args.stage
        job_label = "GraphMask jobs using existing RGCNs" if stage == "graphmask" else "RGCN + GraphMask jobs"
        print(f"[plan] {len(self.contexts)} contexts x {len(self.representations)} representations = {len(self.contexts) * len(self.representations)} independent {job_label}", flush=True)
        if stage in ("all", "prepare"):
            print("[prepare] reuse compatible existing period graphs; create missing periods only")
            selected = [p for p in self.contexts if p != "full"]
            if selected:
                print(shlex.join(self.prepare_command(selected)))
        for context in self.contexts:
            for rep in self.representations:
                for name, command in zip(("collapse", "train", "probe", "report"), self.commands(context, rep)):
                    if stage == "all" or stage == name or stage == "graphmask" and name in ("probe", "report"):
                        print(f"[{name}] {context}/{rep}\n  {shlex.join(command)}")
        if stage in ("all", "graphmask", "summarize"):
            print(f"[summary] {self.path(self.config['graphmask_root'])}/matrix_summary.tsv and relation_group_summary.tsv")

    def load_bundle(self, path):
        if self.torch is None:
            import torch
            self.torch = torch
        return self.torch.load(path, map_location="cpu", weights_only=False)

    def validate_source(self, context):
        source = self.source(context)
        if not complete(source.parent, ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json")):
            raise ValueError(f"Source artifact incomplete: {source.parent}; run prepare first")
        bundle = self.load_bundle(source)
        relations = {r.removesuffix("__rev") for r in bundle["metadata"]["relation_to_id"]}
        if relations != self.members:
            raise ValueError(f"Source vocabulary differs from frozen mapping: missing={sorted(self.members-relations)}, extra={sorted(relations-self.members)}")
        if bundle["metadata"]["target_column"] != "occupation_level1":
            raise ValueError("Expected the DBpedia single-label occupation_level1 artifact")
        if not all(int(bundle["data"][key].sum()) > 0 for key in ("train_mask", "val_mask", "test_mask")):
            raise ValueError(f"Empty supervised split in {context}")
        return bundle

    def validate_period(self, context):
        source = self.source(context)
        if not complete(source.parent, ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json")):
            raise ValueError(f"Existing period artifact is incomplete: {source.parent}")
        summary = read_json(source.parent / "split_summary.json")
        details = summary.get("period_induced_artifact", {})
        recorded = summary.get("life_period_config", {})
        for key, value in self.period_config.items():
            if key != "description" and recorded.get(key) != value:
                raise ValueError(f"Existing period configuration differs at {key}: {source.parent}")
        if summary.get("period_id") != context or summary.get("target_column") != "occupation_level1":
            raise ValueError(f"Wrong period/target artifact: {source.parent}")
        expected = self.config["prepare"]
        actual = details.get("split", {})
        for key, value in expected.items():
            if actual.get("seed" if key == "split_seed" else key) != value:
                raise ValueError(f"Existing period split differs at {key}: {source.parent}")

    def prepare(self):
        self.validate_source("full")
        missing = []
        for context in self.contexts:
            if context == "full":
                continue
            if self.source(context).parent.exists():
                self.validate_period(context)
                print(f"[reuse period] {context}", flush=True)
            else:
                missing.append(context)
        if missing:
            command = self.prepare_command(missing)
            print("[prepare] " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            for context in missing:
                self.validate_period(context)

    def gpu_preflight(self):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        self.torch = torch
        device = torch.device(self.config["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA unavailable; check DBPEDIA_PYTHON_BIN and CUDA_VISIBLE_DEVICES")
            torch.empty(1, device=device)
            print(f"[device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} {torch.cuda.get_device_name(device)}", flush=True)
        elif device.type != "cpu":
            raise ValueError("Use cuda:N for the server, or cpu for a smoke test")
        if self.config["train"]["train_mode"] == "sampled" or self.config["train"]["eval_mode"] == "sampled" or self.config["graphmask_train"]["num_neighbors"] != "full":
            graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
            next(iter(NeighborLoader(graph, num_neighbors=[1, 1], input_nodes=torch.tensor([0]), batch_size=1, num_workers=0)))
        print(f"[preflight] torch={torch.__version__} device={device}; environment ready", flush=True)

    @staticmethod
    def step_contract(command, inputs, settings=None):
        return {"version": 1, "command": command, "inputs": [stamp(p) for p in inputs], "settings": settings}

    def require_completed(self, name, command, directory, outputs, inputs, settings=None):
        record_path = directory / f"pipeline_{name}.json"
        if not record_path.is_file():
            raise ValueError(f"Run {name} first; missing stage record: {record_path}")
        record = read_json(record_path)
        if record.get("status") != "complete" or record.get("contract") != self.step_contract(command, inputs, settings):
            raise ValueError(f"Incomplete or incompatible {name} stage: {directory}; rerun its stage or use new output roots")
        if not complete(directory, outputs) or record.get("outputs") != [stamp(directory / f) for f in outputs]:
            raise ValueError(f"Changed/missing completed outputs: {directory}; use new output roots")

    def run_step(self, name, command, directory, outputs, inputs, settings=None):
        """Reuse only this runner's compatible completed stages; retry its failures."""
        directory = Path(directory)
        contract = self.step_contract(command, inputs, settings)
        record_path = directory / f"pipeline_{name}.json"
        if record_path.exists():
            record = read_json(record_path)
            if record.get("contract") != contract:
                raise ValueError(f"Changed inputs/settings at {directory}; use new output roots")
            if record.get("status") == "complete":
                if not complete(directory, outputs) or record.get("outputs") != [stamp(directory / f) for f in outputs]:
                    raise ValueError(f"Completed outputs were changed/removed: {directory}; use new output roots")
                print(f"[skip {name}] {directory}", flush=True)
                return
        elif any((directory / f).exists() for f in outputs):
            raise ValueError(f"Unowned existing outputs at {directory}; use new output roots")
        directory.mkdir(parents=True, exist_ok=True)
        write_json(record_path, {"status": "running", "contract": contract})
        (directory / f"{name}_command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        print(f"[run {name}] {directory}", flush=True)
        try:
            with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise RuntimeError(f"{name} exited {code}; see {directory / (name + '.log')}")
            if not complete(directory, outputs):
                raise ValueError(f"{name} returned without required outputs: {directory}")
            write_json(record_path, {"status": "complete", "contract": contract, "outputs": [stamp(directory / f) for f in outputs]})
        except Exception as error:
            write_json(record_path, {"status": "failed", "contract": contract, "error": str(error)})
            raise

    def validate_collapsed(self, context, rep):
        original = self.validate_source(context)
        directory, _, _ = self.directories(context, rep)
        self.require_completed("collapse", self.commands(context, rep)[0], directory, self.collapse_outputs(rep), self.collapse_inputs(context), self.collapse_settings(rep))
        collapsed = self.load_bundle(directory / "graph_data.pt")
        # Grouping changes only the edge tensors and relation metadata.
        for key in original["data"].keys():
            if key not in ("edge_index", "edge_type"):
                a, b = original["data"][key], collapsed["data"][key]
                same = self.torch.equal(a, b) if isinstance(a, self.torch.Tensor) else a == b
                if not same:
                    raise ValueError(f"Collapse changed node/label/split/feature field {key}")
        for key in ("label_to_id", "num_classes", "occupation_unknown_ids", "feature_schema"):
            if original["metadata"][key] != collapsed["metadata"][key]:
                raise ValueError(f"Collapse changed metadata {key}")
        bases = {r.removesuffix("__rev") for r in collapsed["metadata"]["relation_to_id"]}
        expected = {"inherited_ties", "acquired_ties"} if rep == "binary" else set(self.multi["groups"])
        if bases != expected or collapsed["metadata"]["num_relations"] != 2 * len(expected):
            raise ValueError("Collapsed relation vocabulary differs from requested representation")
        return collapsed

    @staticmethod
    def collapse_outputs(rep):
        return ("graph_data.pt", "edges.csv", "nodes.csv", "split_summary.json", "relation_collapse_manifest.json", "binary_tie_taxonomy.json" if rep == "binary" else "collapsed_tie_taxonomy.json")

    def collapse_inputs(self, context):
        source = self.source(context)
        return [source, source.parent / "edges.csv", source.parent / "nodes.csv", source.parent / "split_summary.json"]

    def collapse_settings(self, rep):
        return self.binary if rep == "binary" else self.multi

    def run_job(self, context, rep):
        stage = self.args.stage
        artifact, model, mask = self.directories(context, rep)
        collapse, train, probe, report = self.commands(context, rep)
        tie_name = "binary_tie_taxonomy.json" if rep == "binary" else "collapsed_tie_taxonomy.json"
        if stage in ("all", "collapse"):
            self.validate_source(context)
            self.run_step("collapse", collapse, artifact, self.collapse_outputs(rep), self.collapse_inputs(context), self.collapse_settings(rep))
        self.validate_collapsed(context, rep)
        data = artifact / "graph_data.pt"
        if stage in ("all", "train"):
            self.run_step("train", train, model, MODEL_FILES, [data, artifact / tie_name])
        if stage in ("all", "graphmask"):
            self.require_completed("train", train, model, MODEL_FILES, [data, artifact / tie_name])
            self.run_step("probe", probe, mask, PROBE_FILES, [data, model / "best_model.pt"])
            validation = read_json(mask / "validation.json")
            if validation["relative_macro_f1_difference"] > self.config["graphmask_train"]["max_relative_macro_f1_diff"]:
                raise ValueError(f"Probe violates configured fidelity threshold: {mask}")
            if self.config["graphmask_train"].get("checkpoint_selection") == "all-layers-enabled":
                selected = read_json(mask / "manifest.json").get("selected_checkpoint") or {}
                if selected.get("policy") != "all-layers-enabled" or selected.get("enabled_layers") != [True] * self.config["train"]["num_layers"]:
                    raise ValueError(f"Probe does not have all layer gates enabled: {mask}")
            self.run_step("report", report, mask / "test_report", REPORT_FILES, [data, model / "best_model.pt", mask / "graphmask_probe.pt", artifact / "nodes.csv"])

    def summarize(self):
        matrix, relations = [], []
        for context in self.contexts:
            for rep in self.representations:
                artifact, model, mask = self.directories(context, rep)
                summary = read_json(artifact / "split_summary.json") if (artifact / "split_summary.json").is_file() else {}
                collapse = read_json(artifact / "relation_collapse_manifest.json") if (artifact / "relation_collapse_manifest.json").is_file() else {}
                trained = read_json(model / "metrics.json") if (model / "metrics.json").is_file() else {}
                validation = read_json(mask / "validation.json") if (mask / "validation.json").is_file() else {}
                report_dir = mask / "test_report"
                reported = complete(report_dir, REPORT_FILES)
                if reported:
                    try:
                        commands = self.commands(context, rep)
                        data = artifact / "graph_data.pt"
                        tie_name = "binary_tie_taxonomy.json" if rep == "binary" else "collapsed_tie_taxonomy.json"
                        self.require_completed("collapse", commands[0], artifact, self.collapse_outputs(rep), self.collapse_inputs(context), self.collapse_settings(rep))
                        self.require_completed("train", commands[1], model, MODEL_FILES, [data, artifact / tie_name])
                        self.require_completed("probe", commands[2], mask, PROBE_FILES, [data, model / "best_model.pt"])
                        self.require_completed("report", commands[3], report_dir, REPORT_FILES, [data, model / "best_model.pt", mask / "graphmask_probe.pt", artifact / "nodes.csv"])
                    except Exception as error:
                        self.failures.append({"context": context, "representation": rep, "error": str(error)})
                        reported = False
                metrics = read_json(report_dir / "test_metrics.json") if reported else {}
                manifest = read_json(mask / "manifest.json") if (mask / "manifest.json").is_file() else {}
                selected = manifest.get("selected_checkpoint") or {}
                layer_metrics = {r["layer"]: r for r in metrics.get("layers", [])}
                if reported and (metrics.get("split") != "test" or metrics["roots"] != summary["test_nodes"] or metrics["labeled_roots"] != summary["test_nodes"]):
                    raise ValueError(f"Report test roots disagree with source split: {report_dir}")
                errors = [f["error"] for f in self.failures if f.get("context") == context and f.get("representation") == rep]
                state = "failed" if errors else "complete" if reported else "awaiting_graphmask" if complete(model, MODEL_FILES) else "awaiting_train" if summary else "awaiting_collapse"
                matrix.append({
                    "context": context, "period_label": "Full graph" if context == "full" else self.periods[context]["label"], "representation": rep, "seed": self.config["seed"], "status": state,
                    "nodes": summary.get("nodes"), "classes": summary.get("active_target_classes", summary.get("target_classes_retained")), "test_nodes": summary.get("test_nodes"),
                    "directed_relation_types": summary.get("relation_types"), "edges_before": collapse.get("edges_before"), "edges_after": collapse.get("edges_after"), "duplicates_removed": collapse.get("duplicates_removed"),
                    "rgcn_test_macro_f1": (trained.get("test") or {}).get("macro_f1"), "graphmask_original_macro_f1": metrics.get("original", {}).get("macro_f1"), "graphmask_masked_macro_f1": metrics.get("masked", {}).get("macro_f1"),
                    "prediction_agreement": metrics.get("prediction_agreement"), "hard_retention_rate": metrics.get("hard_retention_rate"), "validation_relative_f1_difference": validation.get("relative_macro_f1_difference"), "error": " | ".join(errors), "report_dir": str(report_dir),
                    "checkpoint_selection": manifest.get("training_config", {}).get("checkpoint_selection", "any-stage"),
                    "selected_epoch": selected.get("global_epoch"),
                    "enabled_layers": json.dumps(selected.get("enabled_layers")) if selected else None,
                    "layer0_hard_retention_rate": layer_metrics.get(0, {}).get("hard_retention_rate"),
                    "layer1_hard_retention_rate": layer_metrics.get(1, {}).get("hard_retention_rate"),
                })
                if not reported:
                    continue
                edge_counts = {}
                with (artifact / "edges.csv").open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if not row["relation"].endswith("__rev"):
                            group = row["relation"]
                            edge_counts[group] = edge_counts.get(group, 0) + 1
                with (report_dir / "relations_base.csv").open(encoding="utf-8", newline="") as handle:
                    observations = {(int(r["layer"]), r["relation"]): r for r in csv.DictReader(handle)}
                expected_groups = ("inherited_ties", "acquired_ties") if rep == "binary" else tuple(self.multi["groups"])
                if {group for _, group in observations} - set(expected_groups):
                    raise ValueError(f"Unknown relation group in {report_dir}")
                for layer in range(self.config["train"]["num_layers"]):
                    for group in expected_groups:
                        obs = observations.get((layer, group), {})
                        count = int(obs.get("message_observations", 0))
                        relations.append({"context": context, "representation": rep, "seed": self.config["seed"], "layer": layer, "group": group, "original_triples_after_collapse": edge_counts.get(group, 0), "message_observations": count, "support_status": "observed" if count else "no_sampled_messages" if edge_counts.get(group, 0) else "no_graph_edges", "hard_retention_rate": obs.get("hard_retention_rate") if count else None, "retained_edge_share": obs.get("retained_edge_share") if count else None, "mean_keep_probability": obs.get("mean_keep_probability") if count else None})
        root = self.path(self.config["graphmask_root"])
        root.mkdir(parents=True, exist_ok=True)
        for filename, rows in (("matrix_summary", matrix), ("relation_group_summary", relations)):
            write_json(root / (filename + ".json"), rows)
            with (root / (filename + ".tsv")).open("w", encoding="utf-8", newline="") as handle:
                fields = list(rows[0]) if rows else RELATION_COLUMNS
                if fields:
                    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        write_json(root / "pipeline_failures.json", self.failures)
        print(f"[summary] {sum(r['status']=='complete' for r in matrix)}/{len(matrix)} complete reports: {root / 'matrix_summary.tsv'}", flush=True)
        return all(r["status"] == "complete" for r in matrix)

    def run(self):
        if self.args.stage in ("all", "prepare"):
            self.prepare()
        else:
            for context in self.contexts:
                if context != "full":
                    self.validate_period(context)
        if self.args.stage in ("all", "train", "graphmask"):
            self.gpu_preflight()
        if self.args.stage in ("all", "collapse", "train", "graphmask"):
            for context in self.contexts:
                for rep in self.representations:
                    try:
                        self.run_job(context, rep)
                    except Exception as error:
                        failure = {"context": context, "representation": rep, "error": str(error)}
                        self.failures.append(failure)
                        print(f"[failed] {context}/{rep}: {error}", file=sys.stderr, flush=True)
        if self.args.stage != "prepare":
            ready = self.summarize()
            if self.args.stage == "summarize" and not ready:
                self.failures.append({"error": "The requested matrix has incomplete reports"})
        return 1 if self.failures else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default=os.environ.get("DBPEDIA_GROUP_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--periods", help="all or comma-separated period IDs")
    parser.add_argument("--representations", help="binary,multi_group or one of them")
    parser.add_argument("--include-full", action="store_true", help="Add full-graph comparisons (18 jobs by default)")
    parser.add_argument("--device", help="cuda:N; cpu for local smoke tests")
    return parser.parse_args()


def main():
    os.chdir(ROOT)
    try:
        pipeline = Pipeline(parse_args())
        if pipeline.args.mode == "plan":
            pipeline.plan()
            return 0
        # Advisory lock releases automatically if killed; no stale lock cleanup.
        lock = pipeline.path(pipeline.config["model_root"]) / ".grouped_pipeline.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another pipeline is using this model root") from error
            # Supplementary GraphMask runs reuse the RGCN root; retain its
            # original experiment record and save the new config with the probes.
            config_root = pipeline.path(pipeline.config["graphmask_root"]) if pipeline.args.stage in ("graphmask", "summarize") else lock.parent
            write_json(config_root / "grouped_pipeline_resolved_config.json", pipeline.config)
            return pipeline.run()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
