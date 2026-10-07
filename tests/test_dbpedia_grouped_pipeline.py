"""Behavioral checks for the grouped DBpedia experiment runner."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from DBpedia.grouped_pipeline import Pipeline
from data.collapse_relations import parse_args as parse_collapse_relations_args
from data.collapse_ties import parse_args as parse_collapse_ties_args
from training.graphmask_report import parse_args as parse_graphmask_report_args
from training.graphmask_train import parse_args as parse_graphmask_train_args
from training.train import parse_args as parse_train_args


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "DBpedia" / "grouped_pipeline.py"
CONFIG = ROOT / "config" / "dbpedia_grouped_rgcn_graphmask_20y_v1.json"
PERIOD_CONFIG = ROOT / "config" / "dbpedia_life_periods_20y_v1.json"
MULTI_TAXONOMY = ROOT / "config" / "dbpedia_tie_taxonomy_acquired_subgroups_v1.json"


class GroupedPipelinePlanTest(unittest.TestCase):
    def plan(self, stage="all", *options):
        return subprocess.run(
            [sys.executable, str(PIPELINE), "plan", stage, *options],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout

    def test_real_default_plan_has_sixteen_jobs_and_representation_taxonomies(self):
        output = self.plan()
        period_config = json.loads(PERIOD_CONFIG.read_text(encoding="utf-8"))
        multi_taxonomy = json.loads(MULTI_TAXONOMY.read_text(encoding="utf-8"))
        config = json.loads(CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(len(period_config["periods"]), 8)
        self.assertEqual(len(multi_taxonomy["groups"]), 7)
        self.assertIn("[plan] 8 contexts x 2 representations = 16 independent", output)

        binary_train = next(
            line for line in output.splitlines()
            if line.startswith("  ") and "run.py train " in line and "/binary/seed_42" in line
        )
        multi_train = next(
            line for line in output.splitlines()
            if line.startswith("  ") and "run.py train " in line and "/multi_group/seed_42" in line
        )
        self.assertIn("--tie-taxonomy", binary_train)
        self.assertIn("/binary/binary_tie_taxonomy.json", binary_train)
        self.assertIn("--num-bases 4", binary_train)
        self.assertIn("--tie-taxonomy", multi_train)
        self.assertIn("/multi_group/collapsed_tie_taxonomy.json", multi_train)
        self.assertIn("--num-bases 14", multi_train)
        self.assertEqual(config["num_bases"]["binary"], 4)
        self.assertEqual(config["num_bases"]["multi_group"], 2 * len(multi_taxonomy["groups"]))

    def test_plan_stage_selection_limits_commands_to_requested_work(self):
        options = ("--periods", "through_1500", "--representations", "binary")

        prepare = self.plan("prepare", *options)
        self.assertIn("[prepare]", prepare)
        self.assertNotIn("[collapse]", prepare)
        self.assertNotIn("[train]", prepare)

        collapse = self.plan("collapse", *options)
        self.assertIn("[collapse] through_1500/binary", collapse)
        self.assertIn("run.py collapse-ties", collapse)
        self.assertNotIn("[train]", collapse)
        self.assertNotIn("[probe]", collapse)

        train = self.plan("train", *options)
        self.assertIn("[train] through_1500/binary", train)
        self.assertNotIn("[collapse]", train)
        self.assertNotIn("[probe]", train)

        graphmask = self.plan("graphmask", *options)
        self.assertIn("[probe] through_1500/binary", graphmask)
        self.assertIn("[report] through_1500/binary", graphmask)
        self.assertNotIn("[train]", graphmask)

        summarize = self.plan("summarize", *options)
        self.assertIn("[summary]", summarize)
        self.assertNotIn("[collapse]", summarize)
        self.assertNotIn("[train]", summarize)

    def test_generated_commands_parse_with_their_real_cli_parsers(self):
        args = SimpleNamespace(
            config=str(CONFIG), device="cpu", stage="all", periods="through_1500",
            representations="binary,multi_group", include_full=False,
        )
        pipeline = Pipeline(args)
        parser_by_command = {
            "collapse-ties": parse_collapse_ties_args,
            "collapse-relations": parse_collapse_relations_args,
            "train": parse_train_args,
            "graphmask-train": parse_graphmask_train_args,
            "graphmask-report": parse_graphmask_report_args,
        }

        for representation in ("binary", "multi_group"):
            for command in pipeline.commands("through_1500", representation):
                parser = parser_by_command[command[2]]
                with self.subTest(representation=representation, command=command[2]):
                    with mock.patch.object(sys, "argv", [command[2], *command[3:]]):
                        parsed = parser()
                    self.assertIsNotNone(parsed)

    def test_layer0_supplement_reuses_rgcn_contracts_and_only_plans_five_probes(self):
        supplement = ROOT / "config" / "dbpedia_multi_group_layer0_enabled_5periods_v1.json"
        args = SimpleNamespace(
            config=str(supplement), device=None, stage="graphmask", periods=None,
            representations=None, include_full=False,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            new = Pipeline(args)
            old = Pipeline(SimpleNamespace(**{**vars(args), "config": str(CONFIG)}))
        self.assertEqual(new.contexts, ["through_1500", "1501_1900", "1941_1960", "1981_2000", "since_2001"])
        self.assertEqual(new.representations, ["multi_group"])
        for period in new.contexts:
            new_commands = new.commands(period, "multi_group")
            old_commands = old.commands(period, "multi_group")
            self.assertEqual(new_commands[:2], old_commands[:2])
            self.assertNotEqual(new.directories(period, "multi_group")[2], old.directories(period, "multi_group")[2])
            with mock.patch.object(sys, "argv", new_commands[2][2:]):
                self.assertEqual(parse_graphmask_train_args().checkpoint_selection, "all-layers-enabled")
        with mock.patch.dict(os.environ, {}, clear=True):
            output = self.plan("graphmask", "--config", str(supplement))
        self.assertEqual(output.count("[probe]"), 5)
        self.assertEqual(output.count("[report]"), 5)
        self.assertNotIn("[train]", output)
        self.assertNotIn("[collapse]", output)

    def test_supplement_wrapper_pins_scope_root_and_original_numeric_settings(self):
        environment = dict(os.environ)
        environment.update({
            "DBPEDIA_PYTHON_BIN": sys.executable,
            "DBPEDIA_GROUP_PERIODS": "all", "DBPEDIA_GROUP_REPRESENTATIONS": "binary",
            "DBPEDIA_GROUP_INCLUDE_FULL": "1", "DBPEDIA_GROUP_GRAPHMASK_ROOT": "wrong_old_root",
            "DBPEDIA_GROUP_GRAPHMASK_BETA": "0.9", "DBPEDIA_GROUP_SEED": "999",
        })
        output = subprocess.run(
            ["bash", str(ROOT / "DBpedia/run_layer0_enabled_5periods.sh"), "plan"],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=True,
        ).stdout
        self.assertEqual(output.count("[probe]"), 5)
        self.assertNotIn("wrong_old_root", output)
        self.assertNotIn("/binary/", output)
        self.assertNotIn("[probe] full/", output)
        self.assertIn("--seed 42", output)
        self.assertIn("--beta 0.03", output)


class RunStepResumeTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def call_step(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.pipeline.run_step(*args, **kwargs)

    def command_writing(self, output):
        code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('complete')"
        return [sys.executable, "-c", code, str(output)]

    def test_compatible_completed_step_is_skipped(self):
        directory = self.root / "stage"
        source = self.root / "source.dat"
        source.write_text("input", encoding="utf-8")
        output = directory / "result.bin"
        command = self.command_writing(output)

        self.call_step("demo", command, directory, ("result.bin",), (source,), {"mode": "fixed"})
        before = self.pipeline_stamp(output)
        self.call_step("demo", command, directory, ("result.bin",), (source,), {"mode": "fixed"})

        self.assertEqual(output.read_text(encoding="utf-8"), "complete")
        self.assertEqual(self.pipeline_stamp(output), before)
        self.assertEqual(json.loads((directory / "pipeline_demo.json").read_text())["status"], "complete")

    def pipeline_stamp(self, path):
        from DBpedia.grouped_pipeline import stamp

        return stamp(path)

    def test_changed_settings_or_input_are_rejected(self):
        source = self.root / "source.dat"
        source.write_text("old", encoding="utf-8")

        for changed_contract in ("settings", "input"):
            with self.subTest(changed_contract=changed_contract):
                directory = self.root / changed_contract
                output = directory / "result.bin"
                command = self.command_writing(output)
                self.call_step("demo", command, directory, ("result.bin",), (source,), {"mode": "old"})

                if changed_contract == "settings":
                    settings = {"mode": "new"}
                else:
                    settings = {"mode": "old"}
                    source.write_text("changed input", encoding="utf-8")
                    stat = source.stat()
                    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

                with self.assertRaisesRegex(ValueError, "Changed inputs/settings"):
                    self.call_step("demo", command, directory, ("result.bin",), (source,), settings)

    def test_failed_incomplete_step_can_be_retried(self):
        directory = self.root / "retry"
        source = self.root / "source.dat"
        source.write_text("input", encoding="utf-8")
        output = directory / "result.bin"
        first_run = self.root / "first-run.marker"
        # Use an explicit marker transition so the command string stays identical on retry.
        retry_script = self.root / "retry_command.py"
        retry_script.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "marker, output = map(Path, sys.argv[1:])\n"
            "if marker.exists():\n"
            "    marker.unlink()\n"
            "    raise SystemExit(7)\n"
            "output.write_text('complete', encoding='utf-8')\n",
            encoding="utf-8",
        )
        first_run.touch()
        command = [sys.executable, str(retry_script), str(first_run), str(output)]

        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            self.call_step("demo", command, directory, ("result.bin",), (source,), {"mode": "retryable"})
        self.assertEqual(json.loads((directory / "pipeline_demo.json").read_text())["status"], "failed")

        self.call_step("demo", command, directory, ("result.bin",), (source,), {"mode": "retryable"})
        self.assertEqual(output.read_text(encoding="utf-8"), "complete")
        self.assertEqual(json.loads((directory / "pipeline_demo.json").read_text())["status"], "complete")

    def test_require_completed_rejects_changed_settings_or_source(self):
        directory = self.root / "verified"
        source = self.root / "source.dat"
        source.write_text("original", encoding="utf-8")
        output = directory / "result.bin"
        command = self.command_writing(output)
        outputs = ("result.bin",)
        settings = {"configuration": "v1"}
        self.call_step("demo", command, directory, outputs, (source,), settings)

        self.pipeline.require_completed("demo", command, directory, outputs, (source,), settings)
        with self.assertRaisesRegex(ValueError, "incompatible demo stage"):
            self.pipeline.require_completed(
                "demo", command, directory, outputs, (source,), {"configuration": "v2"}
            )

        other_source = self.root / "other-source.dat"
        other_source.write_text("original", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incompatible demo stage"):
            self.pipeline.require_completed("demo", command, directory, outputs, (other_source,), settings)

        source.write_text("changed", encoding="utf-8")
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        with self.assertRaisesRegex(ValueError, "incompatible demo stage"):
            self.pipeline.require_completed("demo", command, directory, outputs, (source,), settings)

    def test_require_completed_rejects_deleted_output(self):
        directory = self.root / "deleted-output"
        source = self.root / "source.dat"
        source.write_text("input", encoding="utf-8")
        output = directory / "result.bin"
        command = self.command_writing(output)
        self.call_step("demo", command, directory, ("result.bin",), (source,), None)
        output.unlink()

        with self.assertRaisesRegex(ValueError, "Changed/missing completed outputs"):
            self.pipeline.require_completed("demo", command, directory, ("result.bin",), (source,), None)


if __name__ == "__main__":
    unittest.main()
