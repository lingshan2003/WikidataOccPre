import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from DBpedia.package_sliding_results import package


class SlidingPartialPackageTests(unittest.TestCase):
    def make_fixture(self, root):
        contexts = ["center_1900", "center_1901"]
        config = {
            "name": "wikidata_grouped_sliding_test",
            "seed": 7,
            "periods": "all",
            "period_config": str(root / "windows.json"),
            "multi_group_taxonomy": str(root / "taxonomy.json"),
            "model_root": str(root / "models"),
            "graphmask_root": str(root / "graphmask"),
            "relation_root": str(root / "relations"),
            "period_root": str(root / "periods"),
            "train": {"epochs": 2},
            "graphmask_train": {"epochs": 3},
            "graphmask_report": {},
            "prepare": {},
            "num_bases": 4,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "windows.json").write_text(json.dumps({
            "calendar_layout": "sliding_windows",
            "periods": [{"id": context} for context in contexts],
        }))
        (root / "taxonomy.json").write_text(json.dumps({"name": "wikidata_test", "groups": {}}))
        (root / "experiment.json").write_text(json.dumps(config))
        resolved = dict(config)
        resolved_path = root / "models/grouped_pipeline_resolved_config.json"
        resolved_path.parent.mkdir(parents=True)
        resolved_path.write_text(json.dumps(resolved))

        rows = [
            {"context": contexts[0], "representation": "multi_group", "seed": 7, "status": "complete"},
            {"context": contexts[1], "representation": "multi_group", "seed": 7, "status": "failed"},
        ]
        graphmask = root / "graphmask"
        graphmask.mkdir()
        for name, value in (
            ("matrix_summary.json", rows),
            ("relation_group_summary.json", [{"context": context} for context in contexts]),
            ("pipeline_failures.json", [{"context": contexts[1], "representation": "multi_group", "error": "OOM"}]),
        ):
            (graphmask / name).write_text(json.dumps(value))
        (graphmask / "matrix_summary.tsv").write_text("context\trepresentation\tseed\tstatus\ncenter_1900\tmulti_group\t7\tcomplete\ncenter_1901\tmulti_group\t7\tfailed\n")
        (graphmask / "relation_group_summary.tsv").write_text("context\ncenter_1900\ncenter_1901\n")

        suffix = f"{contexts[0]}/multi_group/seed_7"
        for name in ("manifest.json", "validation.json", "training_history.json"):
            path = graphmask / suffix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        report = graphmask / suffix / "test_report"
        report.mkdir(parents=True)
        for name in ("test_metrics.json", "manifest.json", "relations_base.csv", "relations_directed.csv"):
            (report / name).write_text("{}\n")
        model_metrics = root / "models" / suffix / "metrics.json"
        model_metrics.parent.mkdir(parents=True)
        model_metrics.write_text("{}\n")
        relation_dir = root / "relations" / contexts[0] / "multi_group"
        relation_dir.mkdir(parents=True)
        for name in ("split_summary.json", "relation_stats.csv", "relation_collapse_manifest.json"):
            (relation_dir / name).write_text("{}\n")
        return root / "experiment.json", contexts

    def test_default_rejects_incomplete_context(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ = self.make_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "Missing/incomplete summary: center_1901"):
                package(config, Path(temp) / "strict.tar.gz")

    def test_partial_bundle_keeps_full_matrix_and_failures_but_only_complete_details(self):
        with tempfile.TemporaryDirectory() as temp:
            config, contexts = self.make_fixture(Path(temp))
            output = Path(temp) / "partial.tar.gz"
            package(config, output, allow_incomplete=True)
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
                bundle = json.load(archive.extractfile("bundle.json"))
                matrix = json.load(archive.extractfile("graphmask/matrix_summary.json"))
                failures = json.load(archive.extractfile("graphmask/pipeline_failures.json"))

            self.assertEqual(bundle["contexts"], contexts)
            self.assertEqual(bundle["complete_contexts"], [contexts[0]])
            self.assertEqual(bundle["incomplete_contexts"], [contexts[1]])
            self.assertEqual(bundle["context_status"], {contexts[0]: "complete", contexts[1]: "failed"})
            self.assertEqual(bundle["completeness"], "partial")
            self.assertTrue(bundle["partial_export"])
            self.assertEqual([row["context"] for row in matrix], contexts)
            self.assertEqual(failures[0]["context"], contexts[1])
            self.assertIn(f"graphmask/{contexts[0]}/multi_group/seed_7/test_report/test_metrics.json", names)
            self.assertFalse(any(f"/{contexts[1]}/" in name for name in names))

    def test_partial_bundle_rejects_failure_on_complete_context(self):
        with tempfile.TemporaryDirectory() as temp:
            config, contexts = self.make_fixture(Path(temp))
            failures_path = Path(temp) / "graphmask/pipeline_failures.json"
            failures_path.write_text(json.dumps([{"context": contexts[0], "representation": "multi_group", "error": "bad"}]))
            with self.assertRaisesRegex(ValueError, "complete-context failure"):
                package(config, Path(temp) / "invalid.tar.gz", allow_incomplete=True)


if __name__ == "__main__":
    unittest.main()
