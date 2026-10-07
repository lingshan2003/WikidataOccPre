"""Server runner tests: planning, protocol, safe retry and stale-output rejection."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from CBDB.pipeline import Pipeline, parse_args, flags, read_json

ROOT = Path(__file__).resolve().parents[1]


class CBDBPipelineTest(unittest.TestCase):
    def pipeline(self, *options):
        return Pipeline(parse_args(["plan", "all", *options]))

    def test_plan_needs_no_ml_environment_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "outputs"
            pipeline = self.pipeline("--output-root", str(target))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                pipeline.plan()
            self.assertIn("7 periods x 2 representations x 1 seeds", output.getvalue())
            self.assertEqual(output.getvalue().count("[train]"), 14)
            self.assertEqual(output.getvalue().count("[probe]"), 14)
            self.assertEqual(output.getvalue().count("[report]"), 14)
            self.assertFalse(target.exists())

    def test_single_occupation_and_fixed_paired_prepare_seed(self):
        pipeline = self.pipeline("--seeds", "42,43")
        commands = pipeline.job_commands("1700_plus", "multi_group", 42)
        commands2 = pipeline.job_commands("1700_plus", "binary", 43)
        for command in (commands[0], commands2[0]):
            self.assertEqual(command[command.index("--target-level")+1], "1")
            self.assertEqual(command[command.index("--seed")+1], "20261006")
        train = commands[1]
        self.assertEqual(train[train.index("--occupation-feature-levels")+1], "1")
        self.assertEqual(train[train.index("--train-mode")+1], "sampled")
        self.assertEqual(commands[2][commands[2].index("--validation-split")+1], "val")
        self.assertEqual(commands[3][commands[3].index("--split")+1], "test")

    def test_smoke_has_separate_output_and_preserves_fidelity_threshold(self):
        formal = self.pipeline()
        smoke = self.pipeline("--smoke")
        self.assertNotEqual(formal.output, smoke.output)
        self.assertEqual(smoke.contexts, ["1700_plus"])
        self.assertEqual(smoke.config["device"], "cpu")
        self.assertEqual(smoke.config["train"]["epochs"], 1)
        self.assertEqual(smoke.config["graphmask_train"]["max_relative_macro_f1_diff"], 0.05)

    def test_sqlite_rebuild_is_optional_and_readonly_builder_is_used(self):
        self.assertNotIn("/flat/", str(self.pipeline().flat))
        pipeline = self.pipeline("--from-sqlite")
        self.assertEqual(pipeline.flat.parent, pipeline.output / "flat")
        self.assertIn("cbdb_build_flat_triples.py", pipeline.clean_command()[1])

    def test_selection_and_output_input_protection(self):
        for options in (("--periods", "unknown"), ("--seeds", "42,42"),
                        ("--representations", "detailed"), ("--output-root", str(ROOT)),
                        ("--epochs", "0")):
            with self.assertRaises(ValueError):
                self.pipeline(*options)
        with self.assertRaises(ValueError):
            flags({"shell_flag": True})

    def test_reuse_and_changed_output_or_input_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = self.pipeline("--output-root", str(root / "pipeline"))
            source, destination = root / "source.txt", root / "output"
            source.write_text("input")
            command = [sys.executable, "-c", "from pathlib import Path; Path('result.txt').write_text('complete')"]
            # Subprocess cwd is the repo: make the output path explicit.
            command[-1] = "from pathlib import Path; Path(" + repr(str(destination / "result.txt")) + ").write_text('complete')"
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.step("fixture", command, destination, ("result.txt",), [source])
            with patch("CBDB.pipeline.subprocess.Popen", side_effect=AssertionError("must reuse")):
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.step("fixture", command, destination, ("result.txt",), [source])
            source.write_text("changed input")
            with self.assertRaisesRegex(ValueError, "Changed inputs/settings"):
                pipeline.step("fixture", command, destination, ("result.txt",), [source])
            (destination / "result.txt").write_text("changed output")
            with self.assertRaises(ValueError):
                pipeline.require("fixture", command, destination, ("result.txt",), [source])

    def test_failed_stage_retries_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = self.pipeline("--output-root", str(root / "pipeline"))
            source, destination, marker = root / "source", root / "output", root / "retry"
            source.write_text("input")
            script = ("from pathlib import Path; import sys; "
                      f"p=Path({str(marker)!r}); existed=p.exists(); p.touch(); "
                      f"Path({str(destination / 'result.txt')!r}).write_text('done') if existed else None; "
                      "sys.exit(0 if existed else 1)")
            command = [sys.executable, "-c", script]
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    pipeline.step("fixture", command, destination, ("result.txt",), [source])
                self.assertEqual(read_json(destination / "pipeline_fixture.json")["status"], "failed")
                pipeline.step("fixture", command, destination, ("result.txt",), [source])
            self.assertEqual(read_json(destination / "pipeline_fixture.json")["status"], "complete")
            self.assertTrue((destination / "fixture.log").is_file())

    def test_atomic_period_export_has_separate_runner_state_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.pipeline("--output-root", str(Path(temporary) / "pipeline"))
            with patch.object(pipeline, "require") as mocked:
                # Stage prepare requires group + periods but must never create
                # the exporter's target just to hold runner logs.
                pipeline.args.stage = "prepare"
                pipeline.data_steps()
            calls = mocked.call_args_list
            period_call = calls[-1].args
            self.assertEqual(period_call[0], "periods")
            self.assertEqual(period_call[2], pipeline.output / "stages/periods")
            self.assertTrue(all(Path(f).is_absolute() for f in period_call[3]))

    def test_stale_report_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.pipeline("--smoke", "--output-root", str(Path(temporary) / "pipeline"),
                                     "--representations", "binary")
            from CBDB.pipeline import REPORT_FILES
            _, _, _, mask = pipeline.directories("1700_plus", "binary", 42)
            report = mask / "test_report"
            report.mkdir(parents=True)
            for name in REPORT_FILES:
                (report / name).write_text("{}")
            with patch.object(pipeline, "node_count", return_value=1), patch.object(
                pipeline, "require_report", side_effect=ValueError("stale report")):
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.summarize()
            summary = read_json(pipeline.output / "summary/matrix_summary.json")
            self.assertEqual(summary[0]["status"], "failed")
            self.assertIsNone(summary[0]["graphmask_masked_macro_f1"])
            self.assertIn("stale report", summary[0]["error"])

    def test_sampler_wheel_page_matches_build_not_system_cuda(self):
        from CBDB.install_sampling import wheel_url
        self.assertEqual(wheel_url("2.5.1+cu121", "12.1"), "https://data.pyg.org/whl/torch-2.5.0+cu121.html")
        self.assertEqual(wheel_url("2.8.0", None), "https://data.pyg.org/whl/torch-2.8.0+cpu.html")
        with self.assertRaises(ValueError):
            wheel_url("unparseable", None)

    def test_default_cleaned_input_can_be_synchronized_with_git(self):
        pipeline = self.pipeline()
        self.assertEqual(pipeline.flat, ROOT / "CBDB/data/person_relation_triples.tsv.gz")
        self.assertTrue(pipeline.flat.is_file())
        import subprocess
        result = subprocess.run(["git", "check-ignore", str(pipeline.flat)], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
