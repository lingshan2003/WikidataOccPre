"""A pilot must pass before any full-matrix command is issued."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from Wikidata import run_pilot_then_full as runner


class PilotOrchestrationTest(unittest.TestCase):
    def exercise(self, pilot_code=0, validation_error=None, free_mib=5000):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"model_root": "models", "graphmask_root": "mask"}))
            commands = []
            def launch(command, **kwargs):
                commands.append(command)
                code = pilot_code if len(commands) == 1 else 0
                return mock.Mock(stdout=io.StringIO("fixture job output\n"), wait=mock.Mock(return_value=code), poll=mock.Mock(return_value=code))
            snapshots = [{"gpu": g, "free_MiB": free_mib, "used_MiB": 24564-free_mib, "total_MiB": 24564, "utilization_percent": 0} for g in ["0", "1", "3"]]
            with mock.patch.object(runner, "ROOT", root), \
                 mock.patch.object(runner.os, "chdir"), \
                 mock.patch.object(runner, "gpu_snapshot", return_value=snapshots), \
                 mock.patch.object(runner.subprocess, "check_output", return_value="fixture_revision\n"), \
                 mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(runner, "validate", return_value={"status": "passed"}, side_effect=validation_error), \
                 mock.patch.object(runner.signal, "signal"), \
                 mock.patch.object(runner.sys, "argv", ["run_pilot_then_full.py", "--config", str(root/'config.json'), "--gpus", "0,1,3"]), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.main()
            status = json.loads(next((root/'mask/orchestration_runs').glob('*/status.json')).read_text())
            return code, commands, status

    def test_passing_pilot_expands_using_the_same_approved_gpu_pool(self):
        code, commands, status = self.exercise()
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("center_1900,center_1950,center_2000", commands[0])
        self.assertNotIn("--periods", commands[1])
        self.assertTrue(all(c[c.index("--gpus")+1]=="0,1,3" for c in commands))
        self.assertEqual(status["status"], "complete")

    def test_pilot_failure_never_starts_full_matrix(self):
        code, commands, status = self.exercise(pilot_code=1)
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 1)
        self.assertEqual(status["status"], "failed")

    def test_validation_or_memory_failure_never_starts_full_matrix(self):
        for kwargs in ({"validation_error": ValueError("bad fidelity")}, {"free_mib": 500}):
            with self.subTest(kwargs=kwargs):
                code, commands, status = self.exercise(**kwargs)
                self.assertEqual(code, 1)
                self.assertEqual(len(commands), 1)
                self.assertEqual(status["status"], "failed")


if __name__ == "__main__":
    unittest.main()
