"""Contracts for fresh-baseline, period-induced audit summaries."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from data.birth_cohort_artifacts import prepare_period_induced_artifacts


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_period_induced_tie_audit.py"
SPEC = importlib.util.spec_from_file_location("period_induced_summary", SUMMARY_PATH)
period_summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(period_summary)


class PeriodInducedSummaryTests(unittest.TestCase):
    def _source_and_config(self, root):
        # Import locally so unittest discovery does not register the fixture
        # class itself as a second copy of its two test methods.
        from tests.test_birth_cohort_induced_artifacts import PeriodInducedArtifactTests
        helper = PeriodInducedArtifactTests()
        source = helper._write_source_artifact(root)
        config = helper._write_config(root)
        return source, config

    @staticmethod
    def _run_config(model, seed):
        return {
            "model": model,
            "seed": int(seed),
            "occupation_feature_levels": "1,2,3",
            "auxiliary_features": "none",
            "feature_mode": "selected",
            "occupation_representation": "categorical",
            "num_neighbors": "15,10",
            "train_mode": "sampled",
            "eval_mode": "sampled",
            "loss": "cross_entropy",
            "train_root_sampling": "uniform",
            "early_stop_metric": "macro_f1",
            "hidden_dim": 128,
            "branch_dim": 64,
            "heads": 4,
            "rgcn_backend": "fast" if model == "rgcn" else "standard",
            "edge_cohort_config": None,
            "edge_cohort_id": None,
        }

    @staticmethod
    def _write_prediction(path, nodes, true, prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["node_index", "true_label", "prediction"])
            writer.writeheader()
            for node in nodes:
                writer.writerow({"node_index": node, "true_label": true, "prediction": prediction})

    def _write_complete_run_matrix(self, root, artifact_root, config_path, taxonomy_path):
        config = period_summary.load_life_period_config(config_path)
        for period in config.periods:
            artifact = period_summary.load_period_artifact(artifact_root, period, config, taxonomy_path)
            expected_test = list(range(int(artifact["test_nodes"])))
            # The synthetic artifacts contain one test node per period.  The
            # exact node index is read from the graph in the production code;
            # here the split helper places one test observation at a valid
            # local index and we retrieve it below from the saved graph.
            import torch
            bundle = torch.load(Path(artifact["artifact_path"]), map_location="cpu", weights_only=False)
            expected_test = bundle["data"].test_mask.nonzero(as_tuple=False).view(-1).tolist()
            for model in period_summary.MODELS:
                for seed in period_summary.SEEDS:
                    for condition in period_summary.CONDITIONS:
                        run_dir = root / f"{model}__period_{period.identifier}__{condition}" / f"seed_{seed}"
                        inherited = condition in {"without_inherited", "random_matched_inherited"}
                        acquired = condition in {"without_acquired", "random_matched_acquired"}
                        direct = ["inherited"] if condition == "without_inherited" else (["acquired"] if condition == "without_acquired" else [])
                        random = ["inherited"] if condition == "random_matched_inherited" else (["acquired"] if condition == "random_matched_acquired" else [])
                        payload = {
                            "run_config": self._run_config(model, seed),
                            "relation_perturbation": {
                                "data_sha256": artifact["data_sha256"],
                                "tie_taxonomy": {"sha256": artifact["taxonomy_sha256"]},
                                "edge_cohort": None,
                                "dropped_tie_groups": direct,
                                "random_drop_matched_tie_groups": random,
                                "edge_count_before": 10,
                                "edge_count_after_random_drop": 10 if condition == "full" else 8,
                                "dropped_relation_pair_count": 1 if direct else 0,
                                "random_edge_drop_pairs": 1 if random else 0,
                            },
                            "test": {metric: 0.80 if condition == "full" else 0.70 for metric in period_summary.METRICS},
                        }
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
                        self._write_prediction(run_dir / "test_predictions.csv", expected_test, "A", "A")

    def test_period_summary_requires_fresh_period_artifacts_and_same_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config_path = self._source_and_config(root)
            artifact_root = root / "artifacts"
            prepare_period_induced_artifacts(source, artifact_root, config_path)
            taxonomy_path = root / "taxonomy.json"
            taxonomy_path.write_text(json.dumps({
                "name": "test ties", "version": 1,
                "groups": {"inherited": ["father"], "acquired": "all_remaining"},
            }), encoding="utf-8")
            run_root = root / "runs"
            self._write_complete_run_matrix(run_root, artifact_root, config_path, taxonomy_path)
            config = period_summary.load_life_period_config(config_path)
            records = period_summary.load_records(run_root, artifact_root, config, taxonomy_path, 10, 7)
            summaries, specificity = period_summary.summarise(records)
            self.assertEqual(len(records), 60)
            self.assertEqual(len(summaries), 20)
            self.assertEqual(len(specificity), 24)
            inherited = next(row for row in specificity if row["tie_group"] == "inherited")
            self.assertAlmostEqual(inherited["relationship_specific_macro_f1_loss"], 0.0)
            direct = next(row for row in summaries if row["condition"] == "without_inherited")
            self.assertAlmostEqual(direct["accuracy_delta_mean"], -0.1)
            self.assertTrue(direct["bootstrap_available"])

            bad = run_root / "rgcn__period_early__full" / "seed_42" / "metrics.json"
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload["relation_perturbation"]["data_sha256"] = "wrong-artifact"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different period artifact"):
                period_summary.load_records(run_root, artifact_root, config, taxonomy_path, 0, 7)


if __name__ == "__main__":
    unittest.main()
