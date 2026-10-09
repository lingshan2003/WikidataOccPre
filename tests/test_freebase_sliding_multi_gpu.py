"""Freebase dispatcher contracts, worker isolation and window-size estimates."""

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from DBpedia.grouped_pipeline import write_json
from Freebase import sliding_multi_gpu as dispatcher
from Freebase.grouped_pipeline import Pipeline, parse_args


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "config/freebase_grouped_sliding_1900_2000_pm20_step1_v2.json"


class FreebaseSlidingMultiGpuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        config.update({
            "source_data": str(self.root / "source/graph_data.pt"),
            "period_root": str(self.root / "periods"),
            "relation_root": str(self.root / "relations"),
            "model_root": str(self.root / "models"),
            "graphmask_root": str(self.root / "graphmask"),
            "device": None,
        })
        self.config_path = self.root / "pipeline.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        source = self.root / "source"
        source.mkdir()
        for name in ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json"):
            (source / name).write_text("fixture\n", encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def pipeline(self, *options):
        return Pipeline(parse_args(["plan", "all", "--config", str(self.config_path), "--device", "cpu", *options]))

    def dispatcher_pipeline(self, *options):
        return dispatcher.WindowPipeline(parse_args(["run", "all", "--config", str(self.config_path), "--device", "cpu", *options]))

    def write_valid_manifest(self, pipeline):
        manifest = pipeline.path(pipeline.config["period_root"]) / "freebase_source_manifest.json"
        write_json(manifest, pipeline.period_source_contract())
        return manifest

    def test_manual_gpu_pool_uses_only_explicit_ids_and_never_calls_nvidia_smi(self):
        with mock.patch("Freebase.sliding_multi_gpu.subprocess.run", side_effect=AssertionError("must not enumerate GPUs")):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,2"}, clear=True):
                self.assertEqual(dispatcher.select_gpus(), ["4", "5", "2"])
                self.assertEqual(dispatcher.select_gpus(count=2), ["4", "5"])
                self.assertEqual(dispatcher.select_gpus(explicit="0,3"), ["0", "3"])
                self.assertEqual(dispatcher.select_gpus(explicit="7,8", count=2), ["7", "8"])
                self.assertEqual(dispatcher.select_gpus(explicit="1,2", count=1), ["1"])
                with self.assertRaisesRegex(ValueError, "only 2"):
                    dispatcher.select_gpus(explicit="1,2", count=3)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "Select GPUs first"):
                    dispatcher.select_gpus()

    def test_worker_pins_gpu_preflights_once_resets_failures_and_never_summarizes(self):
        instances = []

        class StubWindowPipeline:
            def __init__(self, args):
                self.args = args
                self.contexts = []
                self.failures = []
                self.preflights = 0
                self.runs = []
                instances.append(self)

            def gpu_preflight(self):
                self.preflights += 1

            def run(self, *, summarize):
                self.assert_no_shared_summary = summarize
                self.runs.append((list(self.contexts), summarize))
                self.gpu_preflight()
                if self.contexts == ["center_1900"]:
                    self.failures.append({"context": "center_1900", "error": "simulated"})
                    return 1
                return 0

        jobs, events = queue.Queue(), queue.Queue()
        for item in ("center_1900", "center_1901", None):
            jobs.put(item)
        args = SimpleNamespace(stage="all", cpu_threads=2)
        log_path = self.root / "worker.log"
        with mock.patch("Freebase.sliding_multi_gpu.WindowPipeline", StubWindowPipeline), \
                mock.patch("Freebase.sliding_multi_gpu.os.setsid"), \
                contextlib.redirect_stdout(io.StringIO()):
            dispatcher.gpu_worker("5", args, jobs, events, str(log_path))

        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(instances[0].preflights, 1)
        self.assertEqual(instances[0].runs, [(["center_1900"], False), (["center_1901"], False)])
        received = []
        while not events.empty():
            received.append(events.get())
        done = [event for event in received if event["type"] == "done"]
        self.assertEqual([event["context"] for event in done], ["center_1900", "center_1901"])
        self.assertEqual([event["returncode"] for event in done], [1, 0])
        self.assertEqual(done[0]["failures"], [{"context": "center_1900", "error": "simulated"}])
        self.assertEqual(done[1]["failures"], [])
        self.assertTrue(all(event["gpu"] == "5" for event in done))

    def test_window_prepare_requires_shared_contract_and_skips_adapter_and_shared_prepare(self):
        pipeline = self.dispatcher_pipeline("--periods", "center_1900")
        manifest = self.write_valid_manifest(pipeline)
        before = manifest.read_text(encoding="utf-8")
        with mock.patch.object(dispatcher.GroupedPipeline, "prepare") as shared_prepare, \
                mock.patch("Freebase.sliding_multi_gpu.subprocess.run", side_effect=AssertionError("worker must not run adapter")):
            pipeline.prepare()
        shared_prepare.assert_called_once_with(pipeline)
        self.assertEqual(manifest.read_text(encoding="utf-8"), before)

        manifest.write_text(json.dumps({"version": 999}), encoding="utf-8")
        with mock.patch.object(dispatcher.GroupedPipeline, "prepare") as shared_prepare, \
                mock.patch("Freebase.sliding_multi_gpu.subprocess.run", side_effect=AssertionError("worker must not run adapter")):
            with self.assertRaisesRegex(ValueError, "not initialized or changed"):
                pipeline.prepare()
        shared_prepare.assert_not_called()

    def test_shared_initialization_reuses_matching_provenance_and_rejects_mismatch(self):
        pipeline = self.pipeline()
        manifest = self.write_valid_manifest(pipeline)
        before = json.loads(manifest.read_text(encoding="utf-8"))
        with mock.patch("Freebase.sliding_multi_gpu.subprocess.run") as adapter:
            dispatcher.prepare_shared_source(pipeline)
        adapter.assert_called_once()
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), before)

        manifest.write_text(json.dumps({"version": 999}), encoding="utf-8")
        with mock.patch("Freebase.sliding_multi_gpu.subprocess.run") as adapter:
            with self.assertRaisesRegex(ValueError, "source files changed"):
                dispatcher.prepare_shared_source(pipeline)
        adapter.assert_called_once()

    def test_output_lock_blocks_another_dispatcher_before_work_starts(self):
        pipeline = self.pipeline()
        model_root = pipeline.path(pipeline.config["model_root"])
        conflicts = (
            model_root.parent / ("." + model_root.name + ".freebase_sliding.lock"),
            model_root / ".grouped_pipeline.lock",
        )
        for conflicting_lock in conflicts:
            with self.subTest(lock=conflicting_lock.name):
                conflicting_lock.parent.mkdir(parents=True, exist_ok=True)
                with conflicting_lock.open("a+") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaisesRegex(ValueError, "Another serial or multi-GPU pipeline"):
                        with dispatcher.output_locks(pipeline):
                            self.fail("dispatch must not begin while an output lock is held")

    def test_estimated_window_counts_include_inclusive_boundaries_and_policy_cases(self):
        source_root = self.root / "estimate-source"
        source_root.mkdir()
        source = source_root / "nodes.csv"
        source.write_text(
            "birth_year,death_year\n"
            "1880,1880\n"
            "1700,1880\n"
            "1920,1921\n"
            "1880,\n"
            ",1920\n"
            ",\n"
            "1921,1900\n"
            "1850,2020\n",
            encoding="utf-8",
        )
        fake_pipeline = SimpleNamespace(
            path=lambda value: Path(value),
            config={"source_data": str(source_root / "graph_data.pt")},
            periods={
                "center_1900": {"id": "center_1900", "start": 1880, "end": 1920},
                "center_1901": {"id": "center_1901", "start": 1881, "end": 1921},
                "center_1902": {"id": "center_1902", "start": 1882, "end": 1922},
            },
            contexts=["center_1900", "center_1901", "center_1902"],
        )
        self.assertEqual(dispatcher.estimate_window_nodes(fake_pipeline), {
            "center_1900": 6, "center_1901": 3, "center_1902": 3,
        })
        fake_pipeline.contexts = ["full", "center_1900"]
        self.assertEqual(dispatcher.estimate_window_nodes(fake_pipeline), {"full": 8, "center_1900": 6})

    def test_plan_lists_101_train_probe_report_jobs_without_torch_and_keeps_checkpoints_per_window(self):
        code = (
            "import runpy,sys\n"
            "try:\n"
            "    runpy.run_path(sys.argv.pop(1), run_name='__main__')\n"
            "except SystemExit as error:\n"
            "    assert error.code == 0\n"
            "assert 'torch' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "Freebase/sliding_multi_gpu.py"),
             "plan", "all", "--gpus", "4,5", "--device", "cpu"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertIn("GPUs=['4', '5']", output)
        self.assertEqual(output.count("[train] "), 101)
        self.assertEqual(output.count("[probe] "), 101)
        self.assertEqual(output.count("[report] "), 101)

        records = {}
        lines = output.splitlines()
        for index, line in enumerate(lines[:-1]):
            match = re.fullmatch(r"\[(train|probe|report)\] (center_\d+)/multi_group", line)
            if match:
                kind, context = match.groups()
                records.setdefault(context, {})[kind] = lines[index + 1].strip()
        self.assertEqual(len(records), 101)
        model_root = ROOT / "runs/freebase_grouped_sliding_1900_2000_pm20_step1_v2"
        graphmask_root = ROOT / "runs_graphmask/freebase_grouped_sliding_1900_2000_pm20_step1_v2"
        for year in range(1900, 2001):
            context = f"center_{year}"
            with self.subTest(context=context):
                commands = records[context]
                train_output = re.search(r"--output-dir (\S+)", commands["train"]).group(1)
                probe_checkpoint = re.search(r"--checkpoint (\S+)", commands["probe"]).group(1)
                report_checkpoint = re.search(r"--checkpoint (\S+)", commands["report"]).group(1)
                expected = model_root / context / "multi_group/seed_42"
                self.assertEqual(Path(train_output), expected)
                self.assertEqual(Path(probe_checkpoint), expected / "best_model.pt")
                self.assertEqual(Path(report_checkpoint), expected / "best_model.pt")
                self.assertIn(str(graphmask_root / context / "multi_group/seed_42"), commands["probe"])
                self.assertIn(str(graphmask_root / context / "multi_group/seed_42/test_report"), commands["report"])
