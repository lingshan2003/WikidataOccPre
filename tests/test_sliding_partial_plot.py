"""Incomplete training windows must stay missing, never become zero shares."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from DBpedia.plot_sliding_relation_groups import collect, WIKIDATA_LABELS


class PartialPlotTests(unittest.TestCase):
    def fixture(self, root):
        def js(path, value):
            path = root / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        js("bundle.json", {"bundle_format": "wikidata-sliding-reports-v1", "dataset": "Wikidata",
                           "contexts": ["center_1900", "center_1901"], "complete_contexts": ["center_1900"],
                           "incomplete_contexts": ["center_1901"], "completeness": "partial", "partial_export": True,
                           "context_status": {"center_1900": "complete", "center_1901": "failed"}})
        js("config/experiment.json", {"period_config": "config/windows.json", "multi_group_taxonomy": "config/taxonomy.json",
                                    "graphmask_root": "graphmask", "relation_root": "relations", "seed": 42,
                                    "train": {"num_neighbors": "15,10", "num_layers": 2},
                                    "graphmask_train": {"max_relative_macro_f1_diff": 0.05, "checkpoint_selection": "any-stage"}})
        js("config/windows.json", {"periods": [{"id": f"center_{y}", "start": y - 20, "end": y + 20} for y in [1900, 1901]]})
        js("config/taxonomy.json", {"name": "wikidata_test", "groups": {k: [k] for k in WIKIDATA_LABELS}})
        js("graphmask/matrix_summary.json", [{"context": "center_1900", "representation": "multi_group", "seed": 42,
                                             "status": "complete", "layer0_hard_retention_rate": 1},
                                            {"context": "center_1901", "representation": "multi_group", "seed": 42,
                                             "status": "failed", "error": "CUDA out of memory"}])
        js("graphmask/pipeline_failures.json", [{"context": "center_1901", "representation": "multi_group", "error": "OOM"}])
        prefix = "graphmask/center_1900/multi_group/seed_42/"
        selected = {"policy": "any-stage", "global_epoch": 0, "enabled_layers": [False, True]}
        js(prefix + "manifest.json", {"fanouts": [15, 10], "selected_checkpoint": selected,
                                      "training_config": {"max_relative_macro_f1_diff": 0.05}})
        js(prefix + "validation.json", {"relative_macro_f1_difference": 0, "hard_retention_rate": 0.8})
        js(prefix + "training_history.json", {"history": [{"global_epoch": 0, "enabled_through_layer": 1,
                                                            "validation": {"relative_macro_f1_difference": 0, "hard_retention_rate": 0.8}}]})
        js(prefix + "test_report/manifest.json", {"model_name": "rgcn", "split": "test", "sampling_seed": 42,
                                                   "fanouts": [15, 10], "selected_checkpoint": selected, "enabled_layers": [False, True]})
        js(prefix + "test_report/test_metrics.json", {"split": "test", "roots": 1, "labeled_roots": 1,
                                                      "prediction_agreement": 1, "masked": {"accuracy": 1, "macro_f1": 1},
                                                      "layers": [{"layer": 0, "message_observations": 7, "hard_retention_rate": 1}]})
        js("relations/center_1900/multi_group/split_summary.json", {"test_nodes": 1})
        for path, fields, rows in [
            (prefix + "test_report/relations_base.csv", ["layer", "relation", "message_observations", "hard_retention_rate", "retained_edge_share"],
             [{"layer": 0, "relation": g, "message_observations": 1, "hard_retention_rate": 1, "retained_edge_share": 1 / 7} for g in WIKIDATA_LABELS]),
            ("relations/center_1900/multi_group/relation_stats.csv", ["relation", "count"], [{"relation": g, "count": 1} for g in WIKIDATA_LABELS])]:
            with (root / path).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def test_partial_requires_opt_in_and_keeps_failed_shares_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "requires --allow-incomplete"):
                collect(root, 1900, 1901)
            tidy, audit, provenance = collect(root, 1900, 1901, allow_incomplete=True)
            complete = [r for r in tidy if r["year"] == 1900]
            missing = [r for r in tidy if r["year"] == 1901]
            self.assertAlmostEqual(sum(r["retained_message_share"] for r in complete), 1)
            self.assertTrue(all(r["plot_share_percent"] is None and r["retained_message_share"] is None for r in missing))
            self.assertTrue(all(r["support_status"] == "incomplete_window" for r in missing))
            self.assertEqual(provenance["incomplete_years"], [1901])
            self.assertEqual(provenance["complete_windows"], 1)
            self.assertIsNone(audit[1]["layer0_gate_enabled"])

    def test_failure_in_complete_year_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            (root / "graphmask/pipeline_failures.json").write_text(json.dumps([{"context": "center_1900", "error": "bad report"}]))
            with self.assertRaisesRegex(ValueError, "global failure or a failure on a completed context"):
                collect(root, 1900, 1901, allow_incomplete=True)


if __name__ == "__main__":
    unittest.main()
