"""Focused checks for the predecessor gate and selected-GPU preflight."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from DBpedia import queue_bhht_l1_after_wikidata as queue


def dispatch_fixture():
    contexts = queue.expected_dispatch_contexts()
    return {
        "contexts": contexts,
        "finished": {context: {"context": context, "returncode": 0, "failures": []}
                     for context in contexts},
        "unfinished": [],
        "worker_errors": [],
        "interrupted": False,
        "worker_exitcodes": {gpu: 0 for gpu in ("1", "2", "4")},
        "gpu_ids": ["1", "2", "4"],
    }


def matrix_fixture(path, contexts):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for context in contexts:
        report_dir = path.parent / context / "multi_group" / "seed_42" / "test_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in queue.REPORT_FILES:
            (report_dir / name).write_text("fixture\n", encoding="utf-8")
        rows.append({"context": context, "representation": "multi_group", "seed": "42", "status": "complete",
                     "classes": "5", "report_dir": str(report_dir)})
    fields = ["context", "representation", "seed", "status", "classes", "report_dir"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class DBpediaBHHTQueueTests(unittest.TestCase):
    def test_requested_python_symlink_keeps_its_environment_path(self):
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch("os.access", return_value=True):
            result = queue.python_command("/test/venv/bin/python")
        self.assertEqual(result, "/test/venv/bin/python")

    def test_predecessor_waits_for_exit_file_then_checks_dispatch_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predecessor_dir = root / "graphmask" / "multi_gpu_runs" / "dispatch"
            predecessor_dir.mkdir(parents=True)
            (predecessor_dir / "dispatch_status.json").write_text(
                json.dumps(dispatch_fixture()), encoding="utf-8"
            )
            matrix_fixture(root / "graphmask" / "matrix_summary.tsv", queue.EXPECTED_CONTEXTS)
            exit_file = root / "migration" / "exit_code"

            self.assertFalse(queue.predecessor_ready(predecessor_dir, exit_file))
            exit_file.parent.mkdir()
            exit_file.write_text("0\n", encoding="utf-8")
            self.assertTrue(queue.predecessor_ready(predecessor_dir, exit_file))

    def test_nonzero_exit_and_dispatch_failure_block_successor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predecessor_dir = root / "graphmask" / "multi_gpu_runs" / "dispatch"
            predecessor_dir.mkdir(parents=True)
            exit_file = root / "migration" / "exit_code"
            exit_file.parent.mkdir()
            exit_file.write_text("7\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                queue.predecessor_ready(predecessor_dir, exit_file)

            exit_file.unlink()
            failed = dispatch_fixture()
            failed["finished"]["center_1950"]["returncode"] = 1
            (predecessor_dir / "dispatch_status.json").write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "window failed"):
                queue.predecessor_ready(predecessor_dir, exit_file)

    def test_gpu_preflight_checks_only_selected_cards_and_accepts_shared_utilization(self):
        rows = queue.parse_gpu_snapshot(
            "0, 24500, 24576, 100\n1, 12000, 24576, 93\n2, 10000, 24576, 74\n4, 16000, 24576, 88\n",
            ["1", "2", "4"],
        )
        self.assertEqual([row["gpu"] for row in rows], ["1", "2", "4"])
        self.assertEqual([row["utilization_percent"] for row in rows], [93, 74, 88])
        queue.require_gpu_memory(rows, ["1", "2", "4"], 4096)

        low = [dict(row) for row in rows]
        low[1]["free_MiB"] = 4095
        with self.assertRaisesRegex(ValueError, "4096 MiB required"):
            queue.require_gpu_memory(low, ["1", "2", "4"], 4096)
        with self.assertRaisesRegex(ValueError, "did not report selected GPU"):
            queue.parse_gpu_snapshot("1, 100, 24576, 10\n2, 100, 24576, 10\n", ["1", "2", "4"])

    def test_matrix_summary_rejects_noncomplete_or_missing_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix_summary.tsv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["context", "representation", "seed", "status", "classes", "report_dir"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"context": "center_1900", "representation": "multi_group", "seed": "42", "status": "awaiting_train", "classes": "5", "report_dir": ""})
            with self.assertRaisesRegex(ValueError, "exactly 2 requested contexts"):
                queue.validate_matrix_summary(path, ["center_1900", "center_1901"])

    def test_actual_pilot_plan_passes_and_wrong_auxiliary_features_fail(self):
        env = os.environ.copy()
        env["DBPEDIA_PYTHON_BIN"] = sys.executable
        command = ["bash", str(queue.ROOT / queue.RUNNER_RELATIVE), "plan", "all",
                   "--config", str(queue.CONFIG_RELATIVE), "--gpus", "1,2,4",
                   "--periods", ",".join(queue.PILOT_CONTEXTS)]
        result = subprocess.run(command, cwd=queue.ROOT, env=env, capture_output=True,
                                text=True, check=True)
        queue.check_plan_output(result.stdout, queue.PILOT_CONTEXTS)

        lines = []
        for context in queue.PILOT_CONTEXTS:
            lines.extend([f"[train] {context}/multi_group",
                          "  python run.py train --occupation-feature-levels 1 --auxiliary-features temporal"])
        with self.assertRaisesRegex(ValueError, "no auxiliary features"):
            queue.check_plan_output("\n".join(lines), queue.PILOT_CONTEXTS)


if __name__ == "__main__":
    unittest.main()
