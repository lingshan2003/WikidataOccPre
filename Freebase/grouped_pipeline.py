#!/usr/bin/env python3
"""Plan/run Freebase grouped RGCN and GraphMask from the local source tables.

Planning uses only the standard library. Freebase supplies its source adapter,
configuration and environment variables; compatible stage execution, resume
checks and neutral matrix summaries reuse the DBpedia runner without changing
that runner's defaults. File stamps avoid extra SHA scans.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DBpedia.grouped_pipeline import (
    Pipeline as GroupedPipeline, flags, read_json, select_items,
    taxonomy_members, write_json,
    stamp,
)

DEFAULT_CONFIG = "config/freebase_grouped_rgcn_graphmask_20y_v1.json"
STAGES = ("all", "prepare", "collapse", "train", "graphmask", "summarize")


class Pipeline(GroupedPipeline):
    def __init__(self, args):
        self.args = args
        self.config_path = self.path(args.config)
        self.config = read_json(self.config_path)
        c = self.config
        if c.get("version") != 1 or not c.get("name"):
            raise ValueError("Expected a named version-1 pipeline configuration")
        # Group-specific output variables never pick up the exact-baseline roots.
        for key in ("source_data", "period_root", "period_config", "binary_taxonomy", "multi_group_taxonomy", "relation_root", "model_root", "graphmask_root"):
            c[key] = os.environ.get("FREEBASE_GROUP_" + key.upper(), c[key])
        # Never select a GPU from config/defaults: the caller chooses it for each run.
        c["device"] = args.device or os.environ.get("FREEBASE_GROUP_DEVICE") or os.environ.get("FREEBASE_DEVICE")
        if not c["device"] or not re.fullmatch(r"cpu|cuda:[0-9]+", c["device"]):
            raise ValueError("Select a device explicitly before running: export CUDA_VISIBLE_DEVICES=4; "
                             "export FREEBASE_GROUP_DEVICE=cuda:0 (or export FREEBASE_GROUP_DEVICE=cpu). "
                             "Automatic device selection is disabled.")
        c["seed"] = int(os.environ.get("FREEBASE_GROUP_SEED", c["seed"]))
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
            if "FREEBASE_GROUP_" + env in os.environ:
                c[section][key] = cast(os.environ["FREEBASE_GROUP_" + env])
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
        self.contexts = select_items(args.periods or os.environ.get("FREEBASE_GROUP_PERIODS", c["periods"]), self.periods)
        include_full = args.include_full or c["include_full"]
        if "FREEBASE_GROUP_INCLUDE_FULL" in os.environ:
            value = os.environ["FREEBASE_GROUP_INCLUDE_FULL"]
            if value not in ("0", "1"):
                raise ValueError("FREEBASE_GROUP_INCLUDE_FULL must be 0 or 1")
            include_full = args.include_full or value == "1"
        if include_full:
            self.contexts.insert(0, "full")
        self.representations = select_items(args.representations or os.environ.get("FREEBASE_GROUP_REPRESENTATIONS", c["representations"]), ("binary", "multi_group"))
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
        if self.path(c["source_data"]).name != "graph_data.pt":
            raise ValueError("source_data must name graph_data.pt")
        roots = [self.path(c["source_data"]).parent, *[
            self.path(c[key]) for key in ("period_root", "relation_root", "model_root", "graphmask_root")
        ]]
        if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:]):
            raise ValueError("Source, period, relation, model and GraphMask roots must be separate")
        self.failures = []
        self.torch = None


    def validate_source(self, context):
        # Shared checks enforce the frozen base-relation vocabulary, target
        # column and nonempty supervised splits. Keep diagnostics source-neutral.
        try:
            return super().validate_source(context)
        except ValueError as error:
            message = str(error).replace("DBpedia single-label", "Freebase provisional single-label")
            raise ValueError(message) from error

    def adapter_command(self):
        return [sys.executable, str(ROOT / "Freebase/prepare_graph.py"),
                "--config", str(self.config_path),
                "--output-dir", str(self.path(self.config["source_data"]).parent)]

    def period_source_contract(self):
        source = self.path(self.config["source_data"])
        return {"version": 1, "inputs": [stamp(p) for p in
                (source, source.parent / "nodes.csv", source.parent / "edges.csv", source.parent / "split_summary.json")]}

    def validate_period(self, context):
        manifest = self.path(self.config["period_root"]) / "freebase_source_manifest.json"
        if not manifest.is_file() or read_json(manifest) != self.period_source_contract():
            raise ValueError("Period source files changed or provenance is missing; use a new period/output root")
        super().validate_period(context)
        recorded = read_json(self.source(context).parent / "split_summary.json")["period_induced_artifact"]["source_data"]
        if self.path(recorded) != self.path(self.config["source_data"]):
            raise ValueError("Period graph belongs to another source artifact")

    def prepare(self):
        command = self.adapter_command()
        print("[adapter] " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if any(context != "full" for context in self.contexts):
            root = self.path(self.config["period_root"])
            manifest = root / "freebase_source_manifest.json"
            contract = self.period_source_contract()
            if manifest.is_file():
                if read_json(manifest) != contract:
                    raise ValueError("Period source files changed; use a new period/output root")
            else:
                if root.exists() and any(root.glob("*/graph_data.pt")):
                    raise ValueError("Existing periods lack Freebase source provenance; use a new period/output root")
                root.mkdir(parents=True, exist_ok=True)
                write_json(manifest, contract)
        super().prepare()

    def plan(self):
        if self.args.stage in ("all", "prepare"):
            print("[adapter] create/reuse compatible Freebase source graph from local tables", flush=True)
            print(shlex.join(self.adapter_command()), flush=True)
        super().plan()

    def gpu_preflight(self):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        self.torch = torch
        device_name = self.config["device"]
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA unavailable; check the active wywikidata environment and CUDA_VISIBLE_DEVICES")
            torch.empty(1, device=device)
            print(f"[device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} {torch.cuda.get_device_name(device)}", flush=True)
        elif device.type != "cpu":
            raise ValueError("Use cuda:N for the server, or cpu for a smoke test")
        if self.config["train"]["train_mode"] == "sampled" or self.config["train"]["eval_mode"] == "sampled" or self.config["graphmask_train"]["num_neighbors"] != "full":
            graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
            next(iter(NeighborLoader(graph, num_neighbors=[1, 1], input_nodes=torch.tensor([0]), batch_size=1, num_workers=0)))
        # A real trace/replacement/backward catches incompatible PyG message
        # hooks before spending hours training any of the period models.
        from models.rgcn import RelationalGCNClassifier
        from models.features import FeatureSpec
        from training.graphmask.adapter import GraphMaskModelAdapter
        from training.graphmask.core import GraphMaskProbe
        model = RelationalGCNClassifier(
            num_relations=2, num_classes=2,
            feature_specs={"constant": FeatureSpec(kind="constant")},
            hidden_dim=4, branch_dim=2, num_layers=2,
            dropout=0.0, num_bases=2, rgcn_backend="fast",
        ).to(device).eval()
        features = {"constant": torch.zeros(3, dtype=torch.long, device=device)}
        edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)
        edge_type = torch.tensor([0, 1, 0, 1], device=device)
        adapter = GraphMaskModelAdapter(model, "rgcn")
        logits, traces = adapter.trace(features, edges, edge_type)
        probe = GraphMaskProbe.from_traces(traces).to(device)
        for layer in range(2):
            probe.enable_layer(layer)
        probe.eval()
        gates, _, _, _ = probe(traces)
        opened = adapter.masked_forward(features, edges, edge_type,
                                        [torch.ones_like(g) for g in gates], probe.baselines)
        if not torch.allclose(logits, opened, atol=1e-5, rtol=1e-5):
            raise ValueError("GraphMask all-open messages do not reproduce FastRGCN logits")
        probe.train()
        gates, _, _, _ = probe(traces)
        adapter.masked_forward(features, edges, edge_type, gates, probe.baselines).sum().backward()
        if not any(p.grad is not None for p in probe.parameters()):
            raise ValueError("GraphMask probe gradients unavailable in this environment")
        print(f"[preflight] torch={torch.__version__} device={device}; sampler and FastRGCN GraphMask ready", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--config", default=os.environ.get("FREEBASE_GROUP_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--periods", help="all or comma-separated period IDs")
    parser.add_argument("--representations", help="binary,multi_group or one of them")
    parser.add_argument("--include-full", action="store_true", help="Add full-graph comparisons")
    parser.add_argument("--device", help="Explicit override for FREEBASE_GROUP_DEVICE: cuda:N or cpu; no automatic default")
    return parser.parse_args(argv)


def main(argv=None):
    os.chdir(ROOT)
    try:
        pipeline = Pipeline(parse_args(argv))
        if pipeline.args.mode == "plan":
            pipeline.plan()
            return 0
        lock = pipeline.path(pipeline.config["model_root"]) / ".grouped_pipeline.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another pipeline is using this model root") from error
            write_json(lock.parent / "grouped_pipeline_resolved_config.json", pipeline.config)
            return pipeline.run()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
