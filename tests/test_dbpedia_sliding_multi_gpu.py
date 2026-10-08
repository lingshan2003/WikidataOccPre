"""GPU selection, real spawn scheduling, failure isolation and shared locking."""

import contextlib
import fcntl
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from DBpedia.sliding_multi_gpu import gpu_list, gpu_worker, run_workers, select_gpus, stop_workers


ROOT = Path(__file__).resolve().parents[1]


def simulated_worker(gpu, args, jobs, events, log_path):
    """Spawned stand-in for slow trainers; exercises the real parent scheduler."""
    os.setsid()
    Path(log_path).write_text(f"worker GPU={gpu}\n")
    if gpu == getattr(args, "failed_gpu", None):
        events.put({"type": "worker_error", "gpu": gpu, "error": "simulated GPU startup failure"})
        raise SystemExit(1)
    while True:
        context = jobs.get()
        if context is None:
            return
        events.put({"type": "start", "gpu": gpu, "context": context})
        if gpu == getattr(args, "crashed_gpu", None):
            raise SystemExit(7)
        time.sleep(0.02)
        failures = [{"context": context, "error": "simulated training failure"}] if context == "fail" else []
        events.put({"type": "done", "gpu": gpu, "context": context,
                    "returncode": 1 if failures else 0, "failures": failures})


def worker_with_child(marker):
    os.setsid()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    def terminated(signum, frame):
        child.wait(timeout=3)
        Path(marker + ".child_exit").write_text(str(child.returncode))
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminated)
    Path(marker).write_text(str(child.pid))
    time.sleep(60)


class SlidingMultiGpuTests(unittest.TestCase):
    def test_count_uses_selected_devices_and_explicit_ids_override_old_mask(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,2"}, clear=True):
            self.assertEqual(select_gpus(count=2), ["4", "5"])
            self.assertEqual(select_gpus(explicit="0,3"), ["0", "3"])
            with self.assertRaisesRegex(ValueError, "only 3"):
                select_gpus(count=4)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("DBpedia.sliding_multi_gpu.subprocess.run") as run:
            run.return_value.stdout = "0\n1\n2\n3\n"
            self.assertEqual(select_gpus(count=3), ["0", "1", "2"])
            self.assertEqual(run.call_args.args[0][0], "nvidia-smi")

    def test_reject_ambiguous_duplicate_empty_or_invalid_gpu_selection(self):
        for value in ("", "4,4", "04,4", "4,", "-1,2", "foo", "GPU-../../tmp"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gpu_list(value)
        for count in (0, -1):
            with self.assertRaises(ValueError):
                select_gpus(count=count)
        with self.assertRaises(ValueError):
            select_gpus()
        with self.assertRaises(ValueError):
            select_gpus(explicit="4,5", count=3)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True):
            with self.assertRaises(ValueError):
                select_gpus(count=1)

    def test_two_spawn_workers_run_every_window_once_and_keep_failures_separate(self):
        contexts = [f"center_{1900+i}" for i in range(12)] + ["fail", "center_2000"]
        pipeline = SimpleNamespace(contexts=contexts, args=SimpleNamespace(), failures=[])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = run_workers(pipeline, ["4", "5"], directory, worker_target=simulated_worker)
            saved = json.loads((Path(directory) / "dispatch_status.json").read_text())
            self.assertEqual(set(result["finished"]), set(contexts))
            self.assertEqual(len(result["finished"]), len(contexts))
            self.assertEqual(result["unfinished"], [])
            self.assertEqual(set(r["gpu"] for r in result["finished"].values()), {"4", "5"})
            self.assertEqual(pipeline.failures, [{"context": "fail", "error": "simulated training failure"}])
            self.assertEqual(result["finished"]["center_2000"]["returncode"], 0)
            self.assertEqual(saved["worker_exitcodes"], {"4": 0, "5": 0})
            self.assertTrue((Path(directory) / "gpu_4.log").is_file())
            self.assertTrue((Path(directory) / "gpu_5.log").is_file())

    def test_failed_gpu_start_does_not_strand_the_remaining_windows(self):
        contexts = [f"center_{1900+i}" for i in range(5)]
        pipeline = SimpleNamespace(contexts=contexts, args=SimpleNamespace(failed_gpu="4"), failures=[])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = run_workers(pipeline, ["4", "5"], directory, worker_target=simulated_worker)
        self.assertEqual(set(result["finished"]), set(contexts))
        self.assertEqual({r["gpu"] for r in result["finished"].values()}, {"5"})
        self.assertEqual(len(pipeline.failures), 1)
        self.assertEqual(pipeline.failures[0]["gpu"], "4")

    def test_worker_crash_records_lost_window_and_returns_without_queue_deadlock(self):
        contexts = [f"center_{1900+i}" for i in range(5)]
        pipeline = SimpleNamespace(contexts=contexts, args=SimpleNamespace(crashed_gpu="4"), failures=[])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = run_workers(pipeline, ["4", "5"], directory, worker_target=simulated_worker)
        self.assertEqual(len(result["unfinished"]), 1)
        self.assertEqual(len(result["finished"]), 4)
        self.assertEqual(len(pipeline.failures), 2)  # Lost window + crashed worker.
        self.assertIn("Worker exited 7", pipeline.failures[-1]["error"])

    def test_real_worker_pins_cuda_mapping_and_clears_previous_window_failures(self):
        instances = []
        class StubPipeline:
            def __init__(self, args):
                self.failures = []
                self.preflights = 0
                instances.append(self)
            def gpu_preflight(self):
                self.preflights += 1
            def run(self, *, summarize):
                self.gpu_preflight()
                if summarize:
                    raise AssertionError("Workers must not write a shared summary")
                if self.contexts == ["fail"]:
                    self.failures.append({"context": "fail", "error": "first failure"})
                    return 1
                return 0
        jobs, events = queue.Queue(), queue.Queue()
        for job in ("fail", "next", None):
            jobs.put(job)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("DBpedia.sliding_multi_gpu.os.setsid"), mock.patch("DBpedia.sliding_multi_gpu.Pipeline", StubPipeline):
                gpu_worker("5", SimpleNamespace(stage="all"), jobs, events, str(Path(directory) / "worker.log"))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        done = [events.get() for _ in range(events.qsize())]
        done = [e for e in done if e["type"] == "done"]
        self.assertEqual([e["returncode"] for e in done], [1, 0])
        self.assertEqual(done[1]["failures"], [])
        self.assertEqual(instances[0].preflights, 1)

    def test_cleanup_stops_the_workers_training_child_as_well(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.pid"
            process = mp.get_context("spawn").Process(target=worker_with_child, args=(str(marker),))
            process.start()
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "Worker did not start its child")
                stop_workers([process])
                self.assertFalse(process.is_alive())
                child_exit = Path(str(marker) + ".child_exit")
                self.assertTrue(child_exit.is_file(), "Worker did not observe its child stopping")
                self.assertEqual(int(child_exit.read_text()), -signal.SIGTERM)
            finally:
                stop_workers([process])

    def test_plan_supports_gpu_count_without_loading_torch_and_preserves_any_stage(self):
        env = {"DBPEDIA_PYTHON_BIN": sys.executable, "CUDA_VISIBLE_DEVICES": "4,5"}
        with mock.patch.dict(os.environ, env, clear=True):
            output = subprocess.run(["bash", "DBpedia/run_sliding_multi_gpu.sh", "plan", "all", "--num-gpus", "2"],
                                    cwd=ROOT, capture_output=True, text=True, check=True).stdout
        self.assertIn("GPUs=['4', '5']", output)
        self.assertEqual(output.count("[train]"), 101)
        self.assertEqual(output.count("--checkpoint-selection any-stage"), 101)
        self.assertNotIn("--device cuda:4", output)

    def test_same_model_root_serial_lock_prevents_dispatch_before_any_workers_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / ".grouped_pipeline.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                env = {"DBPEDIA_GROUP_MODEL_ROOT": directory}
                with mock.patch.dict(os.environ, env, clear=True):
                    result = subprocess.run([sys.executable, "DBpedia/sliding_multi_gpu.py", "run", "prepare", "--gpus", "4,5", "--device", "cpu"],
                                            cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Another serial or multi-GPU pipeline", result.stderr)
            self.assertFalse((root / "grouped_pipeline_resolved_config.json").exists())


if __name__ == "__main__":
    unittest.main()
