"""Stdlib-only checks for Freebase source-to-GraphMask orchestration."""
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from DBpedia.grouped_pipeline import Pipeline as SharedPipeline, stamp
from Freebase.grouped_pipeline import Pipeline, parse_args

ROOT = Path(__file__).resolve().parents[1]


class FreebasePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        # Independent fixtures make tests runnable while the real adapter and
        # Freebase taxonomy configuration are being prepared in parallel.
        config = json.loads((ROOT / "config/dbpedia_grouped_rgcn_graphmask_20y_v1.json").read_text())
        config.update(name="freebase_test", source_data=str(self.root / "source/graph_data.pt"),
                      period_root=str(self.root / "periods"), period_config=str(self.root / "periods.json"),
                      binary_taxonomy=str(self.root / "binary.json"), multi_group_taxonomy=str(self.root / "multi.json"),
                      relation_root=str(self.root / "relations"), model_root=str(self.root / "models"),
                      graphmask_root=str(self.root / "graphmask"), representations=["multi_group"],
                      num_bases={"binary": 4, "multi_group": 4})
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        (self.root / "source").mkdir()
        for name in ("graph_data.pt", "nodes.csv", "edges.csv", "split_summary.json"):
            (self.root / "source" / name).write_text("fixture")
        (self.root / "periods.json").write_text(json.dumps({"periods": [{"id": "pre1500", "label": "<1500"}, {"id": "modern", "label": "1901–1920"}]}))
        (self.root / "binary.json").write_text(json.dumps({"groups": {"inherited": ["parent"], "acquired": "all_remaining"}}))
        (self.root / "multi.json").write_text(json.dumps({"groups": {"inherited": ["parent"], "acquired_work": ["colleague"]}}))
        self.env = mock.patch.dict(os.environ, {"FREEBASE_GROUP_DEVICE": "cpu"}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def pipeline(self, *options):
        return Pipeline(parse_args(["plan", "all", "--config", str(self.config), *options]))

    def plan(self, pipeline):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.plan()
        return output.getvalue()

    def test_plan_uses_no_torch_and_starts_with_source_adapter(self):
        command = [sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__'); assert 'torch' not in sys.modules",
                   str(ROOT / "Freebase/grouped_pipeline.py"), "plan", "all", "--config", str(self.config)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        output = result.stdout
        self.assertIn("2 contexts x 1 representations = 2 independent", output)
        self.assertLess(output.index("Freebase/prepare_graph.py"), output.index("prepare_life_period_induced_artifacts.py"))
        self.assertLess(output.index("prepare_life_period_induced_artifacts.py"), output.index("run.py collapse-relations"))
        self.assertIn("--output-dir " + str(self.root / "source"), output)
        self.assertIn("run.py train", output)
        self.assertIn("run.py graphmask-train", output)
        self.assertIn("run.py graphmask-report", output)
        self.assertNotIn("/binary/", output)

    def test_command_paths_and_test_report_split(self):
        pipeline = self.pipeline("--device", "cpu", "--periods", "pre1500", "--representations", "binary,multi_group")
        for rep in pipeline.representations:
            collapse, train, probe, report = pipeline.commands("pre1500", rep)
            self.assertIn("--device", train)
            self.assertEqual(train[train.index("--device") + 1], "cpu")
            self.assertEqual(report[report.index("--split") + 1], "test")
            self.assertIn(str(self.root / "relations/pre1500" / rep / "graph_data.pt"), train)
            self.assertIn(str(self.root / "models/pre1500" / rep / "seed_42/best_model.pt"), probe)
            self.assertIn(str(self.root / "graphmask/pre1500" / rep / "seed_42/test_report"), report)
            self.assertIn("collapse-ties" if rep == "binary" else "collapse-relations", collapse)

    def test_dbpedia_environment_cannot_change_freebase_configuration(self):
        with mock.patch.dict(os.environ, {"DBPEDIA_GROUP_DEVICE": "cuda:99", "DBPEDIA_GROUP_SEED": "7", "DBPEDIA_GROUP_PERIODS": "wrong", "DBPEDIA_GROUP_SOURCE_DATA": "/wrong/graph_data.pt"}):
            pipeline = self.pipeline()
        self.assertEqual(pipeline.config["seed"], 42)
        self.assertEqual(pipeline.contexts, ["pre1500", "modern"])
        self.assertEqual(pipeline.config["device"], "cpu")
        self.assertEqual(pipeline.source("full"), self.root / "source/graph_data.pt")

    def test_freebase_environment_overrides_and_cli_device_precedence(self):
        with mock.patch.dict(os.environ, {"FREEBASE_GROUP_DEVICE": "cuda:2", "FREEBASE_GROUP_SEED": "9", "FREEBASE_GROUP_PERIODS": "modern", "FREEBASE_GROUP_TRAIN_EPOCHS": "1", "FREEBASE_GROUP_SOURCE_DATA": str(self.root / "override/graph_data.pt")}):
            pipeline = self.pipeline("--device", "cpu")
        self.assertEqual(pipeline.config["seed"], 9)
        self.assertEqual(pipeline.contexts, ["modern"])
        self.assertEqual(pipeline.config["device"], "cpu")
        self.assertEqual(pipeline.config["train"]["epochs"], 1)
        command = pipeline.adapter_command()
        self.assertEqual(command[command.index("--output-dir") + 1], str(self.root / "override"))

    def test_missing_device_is_rejected_even_when_old_config_has_a_gpu(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "export FREEBASE_GROUP_DEVICE"):
                self.pipeline()

    def test_auto_device_is_rejected(self):
        with mock.patch.dict(os.environ, {"FREEBASE_GROUP_DEVICE": "auto"}):
            with self.assertRaisesRegex(ValueError, "Automatic device selection is disabled"):
                self.pipeline()

    def test_export_selects_visible_gpu_without_rewriting_physical_ids(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4", "FREEBASE_GROUP_DEVICE": "cuda:0"}):
            pipeline = self.pipeline()
            for command in pipeline.commands("pre1500", "multi_group")[1:]:
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "4")

    def test_prepare_invokes_adapter_before_period_engine(self):
        pipeline = self.pipeline()
        calls = []
        with mock.patch("Freebase.grouped_pipeline.subprocess.run", side_effect=lambda *a, **k: calls.append("adapter")), mock.patch.object(SharedPipeline, "prepare", side_effect=lambda: calls.append("periods")), contextlib.redirect_stdout(io.StringIO()):
            pipeline.prepare()
        self.assertEqual(calls, ["adapter", "periods"])

    def test_adapter_failure_stops_period_preparation(self):
        pipeline = self.pipeline()
        with mock.patch("Freebase.grouped_pipeline.subprocess.run", side_effect=subprocess.CalledProcessError(7, "adapter")), mock.patch.object(SharedPipeline, "prepare") as prepare, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.prepare()
            prepare.assert_not_called()

    def test_changed_source_cannot_reuse_old_periods_without_hash_scans(self):
        pipeline = self.pipeline()
        with mock.patch("Freebase.grouped_pipeline.subprocess.run"), mock.patch.object(SharedPipeline, "prepare"), contextlib.redirect_stdout(io.StringIO()):
            pipeline.prepare()
            (self.root / "source/graph_data.pt").write_text("changed source")
            with self.assertRaisesRegex(ValueError, "source files changed"):
                pipeline.prepare()
        with self.assertRaisesRegex(ValueError, "provenance"):
            pipeline.validate_period("pre1500")

    def test_stages_and_optional_full_binary_plan(self):
        for stage in ("collapse", "train", "graphmask", "summarize"):
            pipeline = self.pipeline()
            pipeline.args.stage = stage
            output = self.plan(pipeline)
            self.assertNotIn("[adapter]", output)
            self.assertNotIn("prepare_life_period_induced_artifacts.py", output)
        output = self.plan(self.pipeline("--include-full", "--representations", "binary,multi_group"))
        self.assertIn("3 contexts x 2 representations = 6 independent", output)
        self.assertIn("[collapse] full/binary", output)

    def test_protects_overlapping_source_and_output_roots(self):
        config = json.loads(self.config.read_text())
        config["relation_root"] = str(self.root / "source/collapsed")
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "roots must be separate"):
            self.pipeline()

    def test_summary_is_source_neutral_and_uses_freebase_root(self):
        pipeline = self.pipeline("--periods", "pre1500")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(pipeline.summarize())
        summary = json.loads((self.root / "graphmask/matrix_summary.json").read_text())
        self.assertEqual(summary[0]["context"], "pre1500")
        self.assertEqual(summary[0]["status"], "awaiting_collapse")
        self.assertTrue(summary[0]["report_dir"].startswith(str(self.root / "graphmask")))
        self.assertNotIn("dbpedia", json.dumps(summary).lower())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "torch/PyG unavailable; plan remains stdlib-only")
    def test_explicit_cpu_preflight_runs_real_graphmask_trace(self):
        for device in ("cpu",):
            pipeline = self.pipeline("--device", device)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                pipeline.gpu_preflight()
            self.assertIn("FastRGCN GraphMask ready", output.getvalue())
            # Keep the user-facing command identity stable for resume checks.
            self.assertEqual(pipeline.config["device"], device)

    def test_resume_rejects_mutated_outputs_and_changed_settings(self):
        pipeline = self.pipeline()
        source = self.root / "input.txt"
        source.write_text("input")
        output = self.root / "stage/result.txt"
        command = [sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('complete')", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_step("demo", command, output.parent, (output.name,), (source,), {"mode": "original"})
            before = stamp(output)
            pipeline.run_step("demo", command, output.parent, (output.name,), (source,), {"mode": "original"})
            self.assertEqual(before, stamp(output))
            with self.assertRaisesRegex(ValueError, "Changed inputs/settings"):
                pipeline.run_step("demo", command, output.parent, (output.name,), (source,), {"mode": "different"})
            output.write_text("mutated")
            with self.assertRaisesRegex(ValueError, "outputs were changed"):
                pipeline.run_step("demo", command, output.parent, (output.name,), (source,), {"mode": "original"})


if __name__ == "__main__":
    unittest.main()
